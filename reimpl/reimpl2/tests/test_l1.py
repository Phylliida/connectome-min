"""Differential test of memory formation (mint.tick: L1s and merges) against the real library's
AutobiographicalStrategy.tick driven by a scripted model (oracle_l1.mjs).

    python3 tests/test_l1.py [sessions] [seed] [shard k/n]

Sessions append conversation, tool cycles, images, reasoning and stray 'Context Manager' turns; ticks
answer from scripts of complete, refused, truncated, tool-calling, empty, <thinking>-wrapped,
malformed and failing replies, while the strategy's own merges build the pyramid above L1; the host
declares tools and a system prompt part way through. After every event the requests sent, byte for
byte, and the summaries, chunks, queues, quarantines and id counter must match.

fixtures_l1.json adds sessions, found by searching generated ones and cut down to what still reaches
their rule, that reach rules the generated ones seldom do; each says which. They run first.
"""
import copy
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import drive  # noqa: E402
import mint  # noqa: E402
import prompts  # noqa: E402
from calls import Host  # noqa: E402
from chunking import close_chunks  # noqa: E402
from config import Config  # noqa: E402
from estimate import Calibration, price  # noqa: E402
from js import js_json, js_len, js_trim, sha256  # noqa: E402
from state import State  # noqa: E402
from library import cases, compare  # noqa: E402

LIBRARY_NAMES = {
    'target_chunk_tokens': 'targetChunkTokens', 'recent_window_tokens': 'recentWindowTokens',
    'head_window_tokens': 'headWindowTokens', 'l1_holdback_chunks': 'l1HoldbackChunks',
    'summary_participant': 'summaryParticipant', 'compression_model': 'compressionModel',
    'summary_target_tokens': 'summaryTargetTokens', 'compression_max_tokens': 'compressionMaxTokens',
    'min_chunk_chars': 'minChunkCharsForLLM', 'recall_budget_tokens': 'compressionRecallBudgetTokens',
    'curve_fallbacks': 'compressionRefusalCurveFallbacks', 'context_budget_tokens': 'compressionContextBudgetTokens',
    'merge_threshold': 'mergeThreshold', 'merge_attempt_limit': 'mergeAttemptLimit',
    'merge_span_messages': 'mergeMaxSourceSpanMessages', 'compression_image_bytes': 'maxCompressionImageBytes',
}
FIXTURES = HERE / 'fixtures_l1.json'     # sessions found by search that reach rules generated ones seldom do
DEFAULT_REPLY = {'stopReason': 'end_turn', 'content': [{'type': 'text', 'text': 'Nothing else stood out.'}]}
SERVER_ERROR = {'message': 'overloaded', 'status': 529, 'type': 'server', 'retryable': True}
RATE_LIMIT = {'message': 'rate limited', 'status': 429, 'type': 'rate_limit', 'retryable': True}
RARE_BLOCKS = [                          # blocks a reply seldom carries, well formed or not
    {'type': 'image', 'source': {'type': 'url', 'url': 'https://x.y/z.png'}}, {'type': 'image', 'source': {'type': 'url', 'url': 5}},
    {'type': 'image', 'source': {'type': 'base64', 'data': 'QQ==', 'mediaType': 'image/png'}}, {'type': 'image', 'source': {'type': 'base64', 'data': 'QQ=='}},
    {'type': 'image', 'source': {'type': 'base64', 'data': 7, 'mediaType': 'image/png'}}, {'type': 'image', 'source': 'nope'},
    {'type': 'document', 'source': {'type': 'url', 'url': 'https://x.y/z.pdf'}}, {'type': 'audio', 'source': {'type': 'base64', 'data': 'QQ==', 'mediaType': 'audio/wav'}},
    {'type': 'video', 'source': {'type': 'base64', 'data': 'QQ==', 'mediaType': 5}}, {'type': 'generated_image', 'data': 'QQ==', 'mimeType': 'image/png'},
    {'type': 'generated_image', 'data': 'QQ=='}, {'type': 'tool_result', 'toolUseId': 't', 'content': 'out'},
    {'type': 'tool_result', 'toolUseId': 't', 'content': [{'type': 'text', 'text': 'x'}]}, {'type': 'tool_result', 'toolUseId': 't', 'content': [{'type': 'text'}]},
    {'type': 'tool_result', 'toolUseId': 9, 'content': 'out'}, {'type': 'image', 'source': {'type': 'base64', 'url': 'https://x.y/z'}},
    {'type': 'document', 'source': {'data': 'QQ==', 'mediaType': 'application/pdf'}}, {'type': 'tool_use', 'id': 't', 'name': 'read', 'input': {}}]
EDGE_KEYS = {'4294967295': 'not an index', '4294967294': 'the last index', 'big': 9999999999999999}   # JS: 1e16
TOOL = {'name': 'read', 'description': 'Read a file', 'inputSchema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}


class ProviderError(Exception):
    def __init__(self, message, status=None, type=None, retryable=None):
        super().__init__(message)
        self.status, self.type, self.retryable = status, type, retryable


def simulate(cfg, events):
    """The port, driven as the oracle drives the library."""
    state, host, states = State(), Host(None), []
    for event in events:
        requests, error = [], None
        match event:
            case {'kind': 'append', 'message': message}:
                state = drive.append(state, message, cfg)
            case {'kind': 'calibrate', 'multiplier': multiplier}:
                state = state._replace(calibration=Calibration(multiplier))
            case {'kind': 'host'}:
                host = host._replace(tools=event.get('tools'), system=event.get('system'))
            case {'kind': 'tick', 'replies': replies}:
                state, error = mint.tick(state, host._replace(complete=scripted(replies, requests)), cfg)
        states.append(observed(state, requests, error))
    return states


def scripted(replies, requests):
    """The model: it records each request as JSON.stringify sends it, and answers from `replies`, then
    a default."""
    replies = list(replies)

    def complete(request):
        requests.append(js_json(request))
        reply = replies.pop(0) if replies else DEFAULT_REPLY
        if 'throw' in reply:
            raise ProviderError(**reply['throw'])
        return copy.deepcopy(reply)
    return complete


def observed(state, requests, error):
    """The state the oracle reports after an event, from the port's."""
    return {
        'requests': requests, 'error': error is not None and (str(error) if isinstance(error, ProviderError) else True),
        'summaries': [{'id': s.id, 'level': s.level, 'content': s.content, 'tokens': s.tokens, 'sourceIds': list(s.sources),
                       'sourceRange': {'first': s.span[0], 'last': s.span[1]}, 'mergedInto': s.parent,
                       'responseContent': list(s.blocks) if s.blocks else None,
                       'provenance': dict(zip(('stopReason', 'requestHash', 'model'), s.provenance)) if s.provenance else None}
                      for s in state.entries],
        'chunks': [list(c) for c in state.chunks], 'compressed': sorted(state.summarized), 'queue': list(state.queue),
        'quarantine': sorted([sha256(list(ids)), hashlib.sha256(request.encode()).hexdigest()] for ids, request in state.quarantine),
        'counter': len(state.entries),
        'merges': [{'level': m.level, 'sourceIds': list(m.sources), 'attempts': len(m.failures), 'lastStopReason': m.stop,
                    'lastOutcome': m.outcome, 'hadRefusal': m.refused, 'serverErrorStreak': m.streak} for m in state.merges],
        'mergeQuarantine': sorted(js_json(list(q)) for q in state.merge_quarantine),
    }


# --- session generation ---

WORDS = 'the a memory tool file read wrote error fixed test passes ok so then why because é 漢字 😀'.split()


def words(r, n):
    return ' '.join(r.choices(WORDS, k=n))


def usage(r, cfg):
    """A reply's token counts. Inputs just under the context budget less the output reserve put the
    recall variants' admission on its boundary."""
    reserve = max(16000, round(cfg.summary_target_tokens * 1.5))
    edge = cfg.context_budget_tokens - (min(reserve, cfg.compression_max_tokens) if cfg.compression_max_tokens else reserve)
    return r.choice([{}, {}, {'usage': {'inputTokens': r.randint(0, 3000), 'outputTokens': r.randint(0, 900)}},
                     {'usage': {'inputTokens': r.randint(0, 3000), 'outputTokens': r.choice([2500, 6000])}},   # fills a recall budget
                     {'usage': {'inputTokens': 190000, 'outputTokens': 50}}, {'usage': {'inputTokens': 0, 'outputTokens': 3}},
                     {'usage': {'inputTokens': edge - r.choice([0, r.randint(0, 600), r.randint(0, 3000)]), 'outputTokens': 0}},
                     {'usage': {'outputTokens': 7}, 'details': {
                         'usage': {'inputTokens': r.choice([0, r.randint(1, 5000)]), 'cacheCreationTokens': 10, 'cacheReadTokens': r.choice([0, 25000])},
                         'model': {'provider': r.choice(['anthropic', 'Bedrock', 'openai'])}}},
                     {'usage': {'outputTokens': 3}, 'details': {'usage': {'inputTokens': r.randint(1, 5000)}, 'model': {'provider': 'anthropic'}}},
                     {'usage': {'inputTokens': 100, 'outputTokens': -5}}])


def reply(r, cfg, refusals):
    """One scripted model reply."""
    text = lambda: [{'type': 'text', 'text': words(r, r.randint(1, 40) if r.random() < 0.9 else 400)}]
    tokens = usage(r, cfg)
    kind = r.choices(['ok', 'refusal', 'max_tokens', 'tool_use', 'empty', 'wrapped', 'malformed', 'odd', 'throw', 'rare', 'shapeless'],
                     [40, refusals, 6, 10, 6, 6, 5, 4, 8, 5, 1])[0]
    if kind == 'ok':
        thought = [{'type': 'thinking', 'thinking': words(r, 6), 'signature': 'c2ln'}] if r.random() < 0.3 else []
        blank = [{'type': 'text', 'text': ''}] * (bool(thought) and r.random() < 0.3)   # kept with the reasoning
        extra = [r.choice(RARE_BLOCKS[9:10] + RARE_BLOCKS[-1:])] * (bool(thought) and r.random() < 0.2)   # left out of what's kept
        return {'stopReason': 'end_turn', 'content': thought + blank + text() + text() * (r.random() < 0.2) + extra, **tokens}
    if kind == 'rare':
        return {'stopReason': 'end_turn', 'content': text() + [r.choice(RARE_BLOCKS)], **tokens}
    if kind == 'shapeless':                    # content that isn't a list of blocks
        return {'stopReason': r.choice(['end_turn', 'refusal', 'tool_use']), **({'content': c} if (c := r.choice([None, 'text'])) else {}), **tokens}
    if kind == 'refusal':
        return {'stopReason': 'refusal', 'content': [], **tokens}
    if kind == 'max_tokens':
        return {'stopReason': 'max_tokens', 'content': text(), **tokens}
    if kind == 'tool_use':
        return {'stopReason': 'tool_use', 'content': [{'type': 'tool_use', 'id': 't', 'name': 'read', 'input': {}}], **tokens}
    if kind == 'empty':
        return {'stopReason': 'end_turn', 'content': r.choice([[], [{'type': 'text', 'text': ' \n'}], [{'type': 'text', 'text': '\x85'}]]), **tokens}
    if kind == 'wrapped':
        return {'stopReason': 'end_turn', 'content': [{'type': 'text', 'text': r.choice(
            ['<thinking>all of it</thinking>', '<Thinking>never closed', '  <THINKING>a</thinking>\n the memory',
             'memory <thinking>mid</thinking>', '　<thinking>x</thinking>'])}], **tokens}
    if kind == 'malformed':                    # never with reasoning blocks, which a mint would keep
        return {'stopReason': 'end_turn', 'content': r.choice([[{'type': 'text'}], [{'type': 'widget'}], text() + [{'type': 'image', 'source': {}}],
                                                               text() + [{'type': 'tool_use', 'id': 1, 'name': 'x', 'input': {}}],
                                                               [{'type': 'thinking', 'thinking': 'x', 'signature': None}] + text()]), **tokens}
    if kind == 'odd':
        return {**({'stopReason': stop} if (stop := r.choice([None, 'paused', 'stop_sequence', 'abort'])) else {}), 'content': text(), **tokens}
    return {'throw': r.choice([SERVER_ERROR, RATE_LIMIT,
                               {'message': 'invalid_request_error: thinking blocks cannot be modified', 'status': 400, 'type': 'invalid_request', 'retryable': False},
                               {'message': 'messages.3: reasoning block rejected', 'type': 'invalid'},
                               {'message': 'thinking signature mismatch', 'status': 400, 'type': 'provider'},
                               {'message': 'Invalid_Request: reasoning not allowed here'},
                               {'message': 'thinking took too long', 'status': 500, 'type': 'server', 'retryable': True},
                               {'message': 'prompt is too long', 'status': 400, 'type': 'invalid_request', 'retryable': False},
                               {'message': 'bad gateway', 'status': 502, 'type': 'server', 'retryable': False}, {'message': 'gateway', 'type': 'server'},
                               {'message': 'unparseable', 'retryable': False}, {'message': 'socket hang up'}])}


def session(r):
    cfg = Config(target_chunk_tokens=r.choice([20, 60, 150]), recent_window_tokens=r.choice([0, 60, 200]),
                 head_window_tokens=r.choice([0, 0, 30, 120]), l1_holdback_chunks=r.choice([0, 1, 1, 2]),
                 summary_participant=r.choice(['Claude', 'assistant']),
                 compression_model=r.choice(['deepseek-chat', 'claude-fable-5-1', 'anthropic/claude-mythos-1', 'claude-opus-4-6']),
                 summary_target_tokens=r.choice([2000, 2000, 100, 20000]), compression_max_tokens=r.choice([None, None, 4096]),
                 min_chunk_chars=r.choice([200, 200, 0, 50, 2000]), recall_budget_tokens=r.choice([100_000, 100_000, 300, 1500, 20_000, 400_000]),
                 curve_fallbacks=r.choice([3, 3, 0, 1, 2]), context_budget_tokens=r.choice([200_000, 30_000, 25_000, r.randint(21_000, 30_000)]),
                 merge_threshold=r.choice([2, 2, 3, 6]), merge_attempt_limit=r.choice([5, 5, 1, 2, 7, 0]),
                 merge_span_messages=r.choice([1500, 1500, 6, 20]), compression_image_bytes=r.choice([12 * 1024 * 1024, 100, 300, 0]))
    holes = r.random() < 0.15                 # tool chunks wait for late tools, while their neighbours merge and fail
    if holes:
        cfg = replace(cfg, merge_threshold=3, merge_attempt_limit=1, compression_model='deepseek-chat')
    stubborn = not holes and r.random() < 0.4  # refuses requests more often as it ages
    flaky = r.random() < 0.2                  # the provider goes down half way through
    turns = r.randint(4, 50 + 50 * stubborn)
    events, ids, calls = [], iter(range(10**6)), iter(range(10**6))
    prefix = 'L1-' if r.random() < 0.05 else 'm'   # message ids that read as summary ids

    def say(participant, content):
        events.append({'kind': 'append', 'message': {'id': f'{prefix}{next(ids)}', 'participant': participant, 'content': content}})

    for turn in range(turns):
        shape = r.choices(['user', 'reply', 'tools', 'bundled', 'orphan', 'dangling', 'late', 'image', 'empty', 'cm', 'cached', 'interrupted'],
                          [30, 25, 20, 3, 3, 4, 3, 5, 2, 2, 2, 3])[0]
        if shape == 'user':
            pad = r.choice(['', '', ' ' * 30, '\n' * 12, '\x85'])
            say(r.choice(['user', 'user', 'Alice']), [{'type': 'text', 'text': pad + words(r, r.choice([1, 5, 40, 120])) + pad}])
        elif shape == 'reply':
            thinking = [{'type': 'thinking', 'thinking': words(r, 8), 'signature': 'c2ln'}]
            say(r.choice(['assistant', 'Claude']), thinking * (r.random() < 0.3) + [{'type': 'text', 'text': words(r, r.randint(1, 60))}])
        elif shape in ('tools', 'dangling'):
            uses = [f'tu{next(calls)}' for _ in range(r.randint(1, 3))]
            say('assistant', [{'type': 'text', 'text': words(r, 5)}] * (r.random() < 0.5)
                + [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {'path': words(r, 2), **({'2': 'b', '10': 'a'} if r.random() < 0.3 else {}),
                                                                           **(EDGE_KEYS if r.random() < 0.15 else {})}}
                   for u in uses])
            if shape == 'tools':
                answered = uses if r.random() < 0.85 else uses[:-1]
                say('user', [{'type': 'tool_result', 'toolUseId': u, 'content': r.choice([words(r, 20), [{'type': 'text', 'text': words(r, 9)}], [
                    {'type': 'text', 'text': words(r, 2)}, {'type': 'image', 'source': {'type': 'base64', 'mediaType': 'image/png', 'data': 'B' * r.randint(10, 400)}}]])}
                             for u in answered])
        elif shape == 'bundled':
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}},
                              {'type': 'tool_result', 'toolUseId': u, 'content': words(r, 10)}, {'type': 'text', 'text': words(r, 6)}])
        elif shape == 'late':                       # the result a turn after its call
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}}])
            say('user', [{'type': 'text', 'text': words(r, 3)}])
            say('user', [{'type': 'tool_result', 'toolUseId': u, 'content': words(r, 6)}])
        elif shape == 'interrupted':                # another speaker between a call and its result
            u = f'tu{next(calls)}'
            say('assistant', [{'type': 'tool_use', 'id': u, 'name': 'read', 'input': {}}])
            say('Alice', [{'type': 'text', 'text': words(r, 3)}])
            say('user', [{'type': 'tool_result', 'toolUseId': u, 'content': words(r, 6)}])
        elif shape == 'orphan':
            say('user', [{'type': 'tool_result', 'toolUseId': f'tu-gone{next(calls)}', 'content': words(r, 5)}])
        elif shape == 'image':
            say('user', [{'type': 'text', 'text': words(r, 3)}, {'type': 'image', 'source': {'type': 'base64', 'mediaType': 'image/png', 'data': 'A' * r.choice([r.randint(10, 400), 50, 100, 300])}}])
        elif shape == 'cm':                         # turns that look like the request's own
            say('Context Manager', [{'type': 'text', 'text': r.choice(['[CM] Recall memory L1-0.', '[CM] Recall memory L1-0. Again', prompts.MARKER, words(r, 4)])}])
        elif shape == 'cached':
            say('user', [{'type': 'text', 'text': words(r, 30), 'cache_control': {'type': 'ephemeral'}}])
        else:
            say('user', [])
        if holes and turn == int(turns * 0.6):
            events.append({'kind': 'host', 'tools': [TOOL], 'system': None})
        elif not holes and r.random() < 0.08:
            events.append({'kind': 'host', 'tools': r.choice([None, [], [TOOL], [TOOL]]), 'system': r.choice([None, '', 'You are Claude.'])})
        if r.random() < 0.06:
            events.append({'kind': 'calibrate', 'multiplier': r.choice([0.7, 1.0, 1.3, 1.8])})
        down, age = flaky and turn > turns / 2, turn / turns
        for _ in range(r.randint(2, 6) if down else r.choice([1, 1, 2, 3]) if stubborn else r.choice([0, 0, 1, 1, 2, 3])):
            first = [{'stopReason': 'refusal', 'content': [], **usage(r, cfg)}] * (stubborn and r.random() < age ** 2) \
                or [{'throw': SERVER_ERROR if r.random() < 0.9 else RATE_LIMIT}] * (down and r.random() < 0.95)
            refusals = 2 if stubborn and age < 0.5 else 40 if holes and age < 0.6 else 15   # stubborn: the pyramid first
            events.append({'kind': 'tick', 'replies': first + [reply(r, cfg, refusals) for _ in range(r.randint(0, 6))]})
    if r.random() < 0.25:                     # a stub threshold exactly at some text-only chunk's length
        messages, priced, chunks, multiplier = [], [], [], 1.0
        for e in events:
            if e['kind'] == 'calibrate': multiplier = e['multiplier']
            if e['kind'] == 'append':
                messages.append(e['message'])
                priced.append(price(e['message']))
                chunks += close_chunks(priced, chunks, multiplier, cfg)
        by_id = {m['id']: m for m in messages}
        lengths = [sum(js_len(js_trim(b['text'])) for i in c for b in by_id[i]['content']) for c in chunks
                   if all(b['type'] == 'text' for i in c for b in by_id[i]['content'])]
        cfg = replace(cfg, min_chunk_chars=r.choice(lengths)) if lengths else cfg
    return {'cfg': cfg, 'events': events}


def library_config(s):
    return {LIBRARY_NAMES[k]: v for k, v in vars(s['cfg']).items() if k in LIBRARY_NAMES and v is not None}


def main(count=150, seed=0, shard='0/1'):
    checked = ticks = calls = minted = 0
    fixtures = [{'cfg': Config(**f['cfg']), 'events': f['events']} for f in json.loads(FIXTURES.read_text())]

    def port(s):
        return {'config': library_config(s), 'events': s['events']}, json.loads(json.dumps(simulate(s['cfg'], s['events'])))

    def check(index, s, states, line):
        nonlocal checked, ticks, calls, minted
        for step, (library, mine) in enumerate(zip(json.loads(line), states, strict=True)):
            for key in mine:
                if mine[key] != library[key]:
                    (HERE / 'failing_session.json').write_text(json.dumps({'config': library_config(s), 'events': s['events'][:step + 1]}))
                    return (f'session {index} event {step} ({s["events"][step]["kind"]}): {key}\n'
                            f'  library {json.dumps(library[key])[:2500]}\n  port    {json.dumps(mine[key])[:2500]}')
            ticks += s['events'][step]['kind'] == 'tick'
            calls += len(mine['requests'])
        minted += sum(1 for e in states[-1]['summaries'] if e['provenance']) if states else 0
        checked += 1

    sessions = cases(fixtures, lambda i: session(random.Random(f'{seed}/{i}')), int(count), shard)
    if message := compare(HERE / 'oracle_l1.mjs', sessions, port, check):
        print(message)
        return 1
    print(f'{checked} sessions, {ticks} ticks, {calls} calls, {minted} memories minted by a call: identical')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
