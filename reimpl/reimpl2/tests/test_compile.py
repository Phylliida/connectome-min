"""End-to-end differential test of a compile (emit.select) against the real library's
AutobiographicalStrategy.select plus ContextManager.compile's split (oracle_compile.mjs).

    python3 tests/test_compile.py [sessions] [seed] [shard k/n]

Sessions mix conversation, tool cycles (paired, bundled into one turn, orphaned, dangling), images
and reasoning, with summaries arriving late, merges, usage reports and budgets near the session's
size. Memories arrive as mint stores them, some with the reply's reasoning and text blocks, and are
rendered under the harness's carrierPolicy 'live-strip'. After every event the compiled messages (or the refusal), committed levels, L1 queue,
multiplier and armed calibration must match.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import drive  # noqa: E402
import emit  # noqa: E402
import mint  # noqa: E402
from chunking import post_strip_tokens  # noqa: E402
from config import Config  # noqa: E402
from state import State  # noqa: E402
from library import cases, compare  # noqa: E402

LIBRARY_NAMES = {
    'target_chunk_tokens': 'targetChunkTokens', 'recent_window_tokens': 'recentWindowTokens',
    'head_window_tokens': 'headWindowTokens', 'l1_holdback_chunks': 'l1HoldbackChunks',
    'image_strip_depth_tokens': 'imageStripDepthTokens', 'max_live_images': 'maxLiveImages',
    'max_live_image_bytes': 'maxLiveImageBytes', 'slack': 'compressionSlackRatio',
    'merge_threshold': 'mergeThreshold', 'reach_tokens': 'kvStableReachTokens',
    'quality_gap_ratio': 'kvStableQualityGapRatio', 'over_budget_grace': 'overBudgetGraceRatio',
    'summary_label': 'summaryContextLabel', 'summary_participant': 'summaryParticipant',
}


def simulate(cfg, events):
    """The port, driven as the oracle drives the library. Fills in, for the oracle, what events
    leave to the session's state: a merge's children (the oldest k parentless summaries at the lowest
    level that has k), a usage report given as a ratio of the armed estimate, a budget given as a
    fraction of the priced total."""
    state, states = State(), []
    for event in events:
        compiled = None
        if event['kind'] == 'append':
            state = drive.append(state, event['message'], cfg)
        elif event['kind'] == 'usage':
            if 'ratio' in event:
                event['tokens'] = max(1, round(state.calibration.armed * event.pop('ratio')))
            state = drive.observe(state, event['tokens'])
        elif event['kind'] == 'summarize':
            for memory in event['memories']:
                if not state.queue:
                    break
                index = state.queue[0]
                state = state._replace(queue=state.queue[1:], entries=state.entries + (entry(f'L1-{index}', 1, state.chunks[index], memory),))
        elif event['kind'] == 'merge':
            summaries = state.entries
            if 'children' not in event:
                event['children'] = next(([s.id for s in orphans[:event['k']]] for level in range(1, 9)
                                          if len(orphans := [s for s in summaries if s.level == level and s.parent is None]) >= event['k']), [])
                event['id'] = f'M{len(summaries)}'
            if event['children']:
                level = next(s.level for s in summaries if s.id == event['children'][0]) + 1
                summaries = tuple(s._replace(parent=event['id']) if s.id in event['children'] else s for s in summaries)
                state = state._replace(entries=summaries + (entry(event['id'], level, tuple(event['children']), event),))
        elif event['kind'] == 'compile':
            if 'fraction' in event:
                priced_total = sum(post_strip_tokens(state.priced, state.calibration.multiplier, cfg))
                event['budget'] = max(1, round(priced_total * event.pop('fraction')))
            state, result = emit.select(state, event['budget'], cfg)
            compiled = {'refusal': list(result)} if isinstance(result, emit.Refusal) else {'messages': result}
        states.append({'compiled': compiled, 'levels': {k: v for k, v in state.levels.items() if v != 0},
                       'queue': list(state.queue),
                       'multiplier': state.calibration.multiplier, 'armed': state.calibration.armed})
    return states


def entry(sid, level, sources, memory):
    """A memory as mint stores it. (A compile never reads its range.)"""
    return mint.Entry(sid, level, memory['text'], memory['tokens'], sources, None, None, tuple(memory['blocks']) if 'blocks' in memory else None)


# --- session generation ---

WORDS = 'the a memory tool file read wrote error fixed test passes ok so then why because é 漢字 😀'.split()


def words(r, n):
    return ' '.join(r.choice(WORDS) for _ in range(n))


def memory(r, text):
    """A memory's text, the summarizer's token count and, when the reply reasoned, its reasoning and
    text blocks, which need not join to the text: mint joins them and strips a <thinking> preamble."""
    tokens = r.choice([-(-len(text) // 4), r.randint(1, 6000)])
    if r.random() < 0.5:
        return {'text': text, 'tokens': tokens}
    reasoning = r.choice([{'type': 'thinking', 'thinking': words(r, 5), 'signature': 'c2ln'}, {'type': 'thinking', 'thinking': '', 'signature': 'c2ln'},
                          {'type': 'redacted_thinking', 'data': 'ZW5j' * r.randint(1, 30)}])
    prose = r.choice([[{'type': 'text', 'text': text}], [{'type': 'text', 'text': ''}, {'type': 'text', 'text': text}],
                      [{'type': 'text', 'text': '<thinking>first</thinking>\n' + text}], [{'type': 'text', 'text': words(r, 4)}, {'type': 'text', 'text': text}],
                      [{'type': 'text', 'text': text, 'citations': []}], [{'type': 'text', 'text': ' \n'}], [{'type': 'text', 'text': '\x85'}], []])
    return {'text': text, 'tokens': tokens, 'blocks': [reasoning] + prose if r.random() < 0.8 else prose + [reasoning]}


def session(r):
    cfg = Config(target_chunk_tokens=r.choice([20, 60, 150]), recent_window_tokens=r.choice([0, 60, 200, 600]),
                 head_window_tokens=r.choice([0, 0, 30, 120]), l1_holdback_chunks=r.choice([0, 1, 1, 2, 3]),
                 image_strip_depth_tokens=r.choice([0, 50, 400, 30000]), max_live_images=r.choice([0, 1, 2, 6]),
                 max_live_image_bytes=r.choice([0, 300, 20 * 1024 * 1024]), slack=r.choice([0.1, 0.1, 0.0, 0.3]),
                 merge_threshold=r.choice([2, 3, 6]), reach_tokens=r.choice([None, None, 0, 50, 400]),
                 quality_gap_ratio=r.choice([0.35, 0.35, 0.0, 5.0]), over_budget_grace=r.choice([0.02, 0.02, 0.0, 0.2]),
                 summary_participant=r.choice(['Claude', 'assistant']),
                 summary_label=r.choice(['What do you remember from earlier?', 'Recall:']))
    events, ids, size, calls = [], iter(range(10**6)), 0, iter(range(10**6))
    pictures = r.random() < 0.25                       # image-heavy: stripped images push prices below zero

    def say(participant, content):
        nonlocal size
        size += sum(len(json.dumps(b)) for b in content) // 3
        events.append({'kind': 'append', 'message': {'id': f'm{next(ids)}', 'participant': participant, 'content': content}})

    if r.random() < 0.2:                               # a compile before the log holds anything
        events.append({'kind': 'compile', 'budget': r.choice([10**9, 50])})
    for _ in range(r.randint(4, 45)):
        shape = r.choices(['user', 'reply', 'tools', 'bundled', 'orphan', 'dangling', 'image', 'empty', 'duplicate', 'late', 'relayed',
                           'distant'], [30, 25, 20, 4, 3, 4, 40 if pictures else 8, 2, 2, 4, 4, 2])[0]
        if shape == 'user':
            say(r.choice(['user', 'user', 'User', 'Alice']), [{'type': 'text', 'text': words(r, r.randint(1, 40))}])
        elif shape == 'reply':
            thinking = [{'type': 'thinking', 'thinking': words(r, 8), **({'signature': 'c2ln'} if r.random() < 0.5 else {})}]
            say('assistant', thinking * (r.random() < 0.3) + [{'type': 'text', 'text': words(r, r.randint(1, 60))}])
        elif shape in ('tools', 'dangling'):
            uses = [f'tu{next(calls)}' for _ in range(r.randint(1, 3))]
            say('assistant', [{'type': 'text', 'text': words(r, 5)}] * (r.random() < 0.5)
                + [{'type': 'tool_use', 'id': u, 'name': r.choice(['read', 'bash']), 'input': None if r.random() < 0.05 else {'path': words(r, 2)}}
                   for u in uses])
            if shape == 'tools':
                result = lambda u: {'type': 'tool_result', 'toolUseId': u, 'content': r.choice(
                    [words(r, r.randint(0, 80)), [{'type': 'text', 'text': words(r, 20)}]])}
                answered = uses if r.random() < 0.85 else uses[:-1]           # sometimes one goes unanswered
                say('user', [result(u) for u in answered] + [{'type': 'text', 'text': 'and?'}] * (r.random() < 0.2))
        elif shape == 'bundled':                                              # a whole cycle in one turn
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}},
                              {'type': 'tool_result', 'toolUseId': u, 'content': words(r, 10)},
                              {'type': 'text', 'text': words(r, 6)}])
        elif shape in ('late', 'relayed'):          # the result arrives a turn late, or in a non-user turn
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'bash', 'input': {'cmd': words(r, 2)}}])
            if shape == 'late':
                say('user', [{'type': 'text', 'text': words(r, 3)}])
            say('user' if shape == 'late' else r.choice(['assistant', 'Alice', 'User']),
                [{'type': 'tool_result', 'toolUseId': u, 'content': words(r, 6)}, {'type': 'text', 'text': words(r, 2)}])
        elif shape == 'duplicate':                                            # one call answered twice
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}}])
            say('user', [{'type': 'tool_result', 'toolUseId': u, 'content': words(r, 4)}] * 2)
        elif shape == 'orphan':
            say('user', [{'type': 'tool_result', 'toolUseId': f'tu-gone{next(calls)}', 'content': words(r, 5)}])
        elif shape == 'distant':                    # the result arrives past the six entries searched for it
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}}])
            for _ in range(r.randint(5, 8)):
                say(r.choice(['user', 'assistant']), [{'type': 'text', 'text': words(r, 2)}])
            say('user', [{'type': 'tool_result', 'toolUseId': u, 'content': words(r, 4)}])
        elif shape == 'image':
            say('user', [{'type': 'text', 'text': words(r, 3)},
                         {'type': 'image', 'source': {'type': 'base64', 'data': 'A' * r.randint(10, 400)}}])
        else:
            say(r.choice(['user', 'assistant']), [])
        if r.random() < 0.35:
            events.append({'kind': 'summarize', 'memories': [memory(r, words(r, r.randint(3, 120))) for _ in range(r.randint(1, 3))]})
        if r.random() < 0.15:
            events.append({'kind': 'merge', 'k': cfg.merge_threshold, **memory(r, words(r, r.randint(3, 80)))})
        if r.random() < 0.4:
            events.append(r.choice([{'kind': 'compile', 'budget': 10**9},
                                    {'kind': 'compile', 'budget': max(1, round(size * r.uniform(0.15, 1.3)))},
                                    {'kind': 'compile', 'fraction': r.uniform(0.85, 1.3)}]))
            if r.random() < 0.7:                       # below the estimate lowers the multiplier
                events.append({'kind': 'usage', 'ratio': r.uniform(0.6, 0.95) if pictures else r.uniform(0.5, 1.9)})
    return {'cfg': cfg, 'events': events}


def library_config(cfg):
    return {LIBRARY_NAMES[k]: v for k, v in vars(cfg).items() if k in LIBRARY_NAMES and v is not None}


def main(count=150, seed=0, shard='0/1'):
    checked = compiles = refusals = 0

    def port(s):
        states = json.loads(json.dumps(simulate(s['cfg'], s['events'])))      # first: it fills in the events
        return {'config': library_config(s['cfg']), 'events': s['events']}, states

    def check(index, s, states, line):
        nonlocal checked, compiles, refusals
        for step, (library, mine) in enumerate(zip(json.loads(line), states, strict=True)):
            for key in mine:
                if mine[key] != library[key]:
                    (HERE / 'failing_session.json').write_text(json.dumps(
                        {'config': library_config(s['cfg']), 'events': s['events'][:step + 1]}))
                    return (f'session {index} event {step} ({s["events"][step]["kind"]}): {key}\n'
                            f'  library {json.dumps(library[key])[:1500]}\n  port    {json.dumps(mine[key])[:1500]}')
            if mine['compiled']:
                compiles += 1
                refusals += 'refusal' in mine['compiled']
        checked += 1

    sessions = cases([], lambda i: session(random.Random(f'{seed}/{i}')), int(count), shard)
    if message := compare(HERE / 'oracle_compile.mjs', sessions, port, check):
        print(message)
        return 1
    print(f'{checked} sessions, {compiles} compiles ({refusals} refused): identical')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
