"""Differential test of estimate.py and chunking.py against the real library (oracle.mjs).

    python3 tests/test_faithful.py [scenarios] [seed] [shard k/n]

Needs node and the harness's node_modules (or CONTEXT_MANAGER, see oracle.mjs). Each scenario is a
random message log with random config, calibration samples, demands and summarizer progress;
after every event, every chunk, queue entry, window boundary, multiplier and per-message estimate
must match. A failing scenario is written to failing_scenario.json for replay.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from chunking import close_chunks, demand, head_end, post_strip_tokens, windows  # noqa: E402
from config import Config  # noqa: E402
from cut import Summary  # noqa: E402
from library import cases, compare  # noqa: E402
from planner import ceiling, inputs  # noqa: E402
from segment import backlog, enqueue  # noqa: E402
from estimate import Calibration, price, store_tokens  # noqa: E402

CONFIG_FIELDS = {
    'targetChunkTokens': 'target_chunk_tokens',
    'recentWindowTokens': 'recent_window_tokens',
    'headWindowTokens': 'head_window_tokens',
    'attachmentsIgnoreSize': 'attachments_ignore_size',
    'l1HoldbackChunks': 'l1_holdback_chunks',
    'imageStripDepthTokens': 'image_strip_depth_tokens',
    'maxLiveImages': 'max_live_images',
    'maxLiveImageBytes': 'max_live_image_bytes',
}


# --- the port, driven the way the oracle drives the library ---

def simulate(config, events):
    cfg = Config(**{CONFIG_FIELDS[k]: v for k, v in config.items()})
    messages, chunks, summarized, demanded, queue, cal = [], [], set(), {}, [], Calibration()
    states = []
    for event in events:
        kind, planner = event['kind'], None
        if kind == 'append':
            messages.append(price(event['message']))
            chunks += close_chunks(messages, chunks, cal.multiplier, cfg)
            queue = backlog(len(chunks), summarized, cfg.l1_holdback_chunks, demanded)
        elif kind == 'compile':
            chunks += close_chunks(messages, chunks, cal.multiplier, cfg)
            queue = backlog(len(chunks), summarized, cfg.l1_holdback_chunks, demanded)
            l1s = [Summary(f'L1-{i}', 1, 0, chunks[i]) for i in sorted(summarized)]
            view = inputs(messages, l1s, {}, cal.multiplier, cfg)
            if 'within' in event:                # a budget whose grace the windows land in, or just past
                event['budget'] = max(1, round((view.head_tokens + view.tail_tokens) / event.pop('within')))
            if view.head_tokens + view.tail_tokens <= ceiling(event['budget'], Config()):
                planner = {'leaves': [list(leaf) for leaf in view.leaves], 'headTokens': view.head_tokens,
                           'tailTokens': view.tail_tokens, 'zone': len(view.prices) - len(view.leaves),
                           'now': len(view.prices) - 1}
            cal = cal.arm(event['window'])
        elif kind == 'usage':
            cal = cal.observe(event['tokens'])
        elif kind == 'demand':
            wanted = demand(chunks, event['first'], event['last'])
            demanded |= dict.fromkeys(wanted)
            queue = enqueue(queue, wanted, summarized)
        elif kind == 'summarize':
            for _ in range(event['count']):
                if queue:
                    summarized.add(queue.pop(0))
        m = cal.multiplier
        states.append({
            'chunks': [list(chunk) for chunk in chunks],
            'queue': list(queue),
            'multiplier': m,
            'head': head_end(messages, m, cfg),
            'tail': windows(messages, m, cfg)[1],
            'postStrip': post_strip_tokens(messages, m, cfg),
            'store': [store_tokens(msg, m) for msg in messages],
            'textOnly': [msg.text_only for msg in messages],
            'salience': [msg.salience for msg in messages],
            'planner': planner,
        })
    return states


# --- scenario generation ---

POOLS = ['abcdefghijklmnopqrstuvwxyz     ', 'ABC .,;!?-', '{}[]":_/=`', 'éüñßø漢字かな',
         '😀🎉🧠', '\x00\x01\x1f\t\n\r"\\\x7f', '\ud800􏰀\udfff']
NUMBERS = [0, 1, -1, 7, 2**53 + 1, 10**21, 123456789012345678901, 1.0, -0.0, 0.1, 2.5, 1e-7, 1e-6,
           1.5e-10, 1e20, 1e21, 1.5e300, 5e-324, 1.7976931348623157e308, 123456.789,
           9999999999999999]                  # a double rounds it up a digit: 10000000000000000


def rand_text(r):
    if r.random() < 0.1:   # dense share straddling the 12% threshold
        n = r.randint(1, 80) * 25
        dense = n * 12 // 100 + r.randint(0, 1)
        chars = list('{' * dense + 'a' * (n - dense))
        r.shuffle(chars)
        return ''.join(chars)
    if r.random() < 0.05:  # fenced code and link lines, some padded with whitespace JS doesn't trim
        pad = r.choice(['', ' ', '\xa0', '\ufeff', '\x1c', '\u2028', '\x85'])
        return r.choice(['', 'see:\n']) + '```py\nx = 1\n```' * r.randint(0, 2) + '\n' \
            + pad + r.choice(['https://a.b/c', 'http://x.y/z?q=1', 'https://a b', 'ftp://c.d']) + pad
    if r.random() < 0.04:  # the probe's 2000 code units hold fewer characters than 2000
        return r.choice('😀🎉🧠') * r.randint(110, 140) + 'a' * r.randint(1800, 2200)
    if r.random() < 0.04:  # the cut splits a surrogate pair, with the threshold one unit away
        return '{' * r.randint(239, 241) + 'a' * (1999 - 240) + '😀' + 'a' * r.randint(0, 100)
    n = r.choice([0, r.randint(1, 30), r.randint(30, 300), r.randint(1990, 2100)])
    pools = r.sample(POOLS, r.randint(1, 3))
    return ''.join(r.choice(r.choice(pools)) for _ in range(n))


def rand_json(r, depth=0):
    roll = r.random()
    if depth >= 3 or roll < 0.5:
        return r.choice([None, True, False, r.choice(NUMBERS), r.uniform(-1e6, 1e6), rand_text(r)[:40]])
    if roll < 0.75:
        return [rand_json(r, depth + 1) for _ in range(r.randint(0, 4))]
    return {rand_text(r)[:12]: rand_json(r, depth + 1) for _ in range(r.randint(0, 4))}


def maybe(r, block, key, *values):
    if r.random() < 0.4:
        block[key] = r.choice(values)
    return block


def rand_block(r, depth=0):
    kind = r.choices(['text', 'thinking', 'redacted_thinking', 'tool_use', 'tool_result', 'image',
                      'document', 'audio', 'mystery'], [30, 8, 4, 10, 10, 6, 1, 1, 1])[0]
    if kind == 'text':
        return {'type': 'text', 'text': rand_text(r)}
    if kind == 'thinking':
        block = maybe(r, {'type': 'thinking', 'thinking': rand_text(r)}, 'signature', '', 'c2ln')
        return maybe(r, block, 'tokenEstimate', 0, 37, 600.5, True, 0.6, 0.7, 0.2)
    if kind == 'redacted_thinking':
        return maybe(r, maybe(r, {'type': 'redacted_thinking'}, 'data', '', 'Q' * r.randint(1, 5000)),
                     'tokenEstimate', 0, 99, 12.5, True)
    if kind == 'tool_use':
        return {'type': 'tool_use', 'id': 'tu', 'name': 'tool',
                'input': None if r.random() < 0.05 else {rand_text(r)[:8]: rand_json(r) for _ in range(r.randint(0, 5))}}
    if kind == 'tool_result':
        stamps = [{'type': 'thinking', 'thinking': '', 'tokenEstimate': v} for v in (0.6, 0.7, 0.2)]   # 1.4999999999999998 in JS
        content = r.choice([rand_text(r), '', None, [], stamps,
                            [rand_block(r, depth + 1) for _ in range(r.randint(1, 3))] if depth < 2 else 'x'])
        return {'type': 'tool_result', 'toolUseId': 'tu', 'content': content}
    if kind == 'image':
        return maybe(r, {'type': 'image', 'source': {'type': 'base64', 'data': 'A' * r.randint(0, 4000)}},
                     'tokenEstimate', 0, 100, 1600, 2500, True)
    return {'type': kind}


def scenario(r):
    config = {
        'targetChunkTokens': r.choice([20, 60, 150, 400]),
        'recentWindowTokens': r.choice([0, 50, 300, 1500]),
        'headWindowTokens': r.choice([0, 0, 40, 400]),
        'attachmentsIgnoreSize': r.random() < 0.7,
        'l1HoldbackChunks': r.choice([0, 1, 1, 2, 3]),
        'imageStripDepthTokens': r.choice([0, 1, 100, 2000]),   # 1: depth equals it exactly
        'maxLiveImages': r.choice([0, 1, 3]),
        'maxLiveImageBytes': r.choice([0, 3000, 20 * 1024 * 1024]),
    }
    # Drift: a rising multiplier shrinks the head window behind closed chunks, so the messages it
    # releases form a run that the gap rule must close where it meets them.
    drift = r.random() < 0.25
    if drift:
        config |= {'targetChunkTokens': 30, 'recentWindowTokens': 0, 'headWindowTokens': 300}
    events = []
    for i in range(r.randint(5, 90)):
        if r.random() < 0.15:   # a tool call, then its result
            events.append({'kind': 'append', 'message': {'id': i, 'content': [rand_block(r) for _ in range(r.randint(0, 2))]
                                                         + [{'type': 'tool_use', 'id': 'tu', 'name': 'tool', 'input': {}}]}})
            content = [{'type': 'tool_result', 'toolUseId': 'tu', 'content': rand_text(r)}]
        elif drift:
            content = [{'type': 'text', 'text': 'word ' * r.randint(6, 24)}]
        else:
            content = [rand_block(r) for _ in range(r.randint(0, 4))]
        events.append({'kind': 'append', 'message': {'id': i + 1000, 'content': content}})
        if r.random() < (0.5 if drift else 0.2):
            window = r.choice([0, r.uniform(1, 20000), r.randint(1, 5000)])
            budget = r.choice([{'budget': 10**12}, {'budget': r.randint(1, 3000)}, {'budget': r.randint(1, 30000)},   # small ones refuse
                               {'within': r.uniform(1, 1.025)}])
            events.append({'kind': 'compile', 'window': window, **budget})
            if r.random() < 0.8:
                ratio = 1.8 if drift and r.random() < 0.7 else r.uniform(0.3, 2.2)
                events.append({'kind': 'usage', 'tokens': r.choice([window * ratio, 0, -3])})
        if r.random() < 0.05:
            events.append({'kind': 'usage', 'tokens': r.uniform(1, 20000)})
        if r.random() < 0.12:
            ids = [e['message']['id'] for e in events if e['kind'] == 'append']
            a, b = sorted(r.sample(range(len(ids)), 2)) if len(ids) > 1 else (0, 0)
            a, b = (b, a) if r.random() < 0.2 else (a, b)        # a range given newest end first
            events.append({'kind': 'demand', 'first': ids[a], 'last': ids[b] if r.random() < 0.8 else 99999})
        if r.random() < 0.25:
            events.append({'kind': 'summarize', 'count': r.randint(1, 3)})
    return {'config': config, 'events': events}


def first_difference(expected, actual):
    for key in expected:
        if expected[key] != actual[key]:
            return f'{key}: library {expected[key]!r}\n{" " * len(key)}  port    {actual[key]!r}'
    return None


def main(count=300, seed=0, shard='0/1'):
    checked = events = 0

    def port(s):
        parsed = json.loads(json.dumps(s))   # as the oracle reads it, ASCII-escaped: both sides parse identical values
        states = list(simulate(parsed['config'], parsed['events']))     # first: it fills in budgets
        return parsed, states

    def check(index, s, states, line):
        nonlocal checked, events
        for step, (expected, actual) in enumerate(zip(json.loads(line), states, strict=True)):
            if (diff := first_difference(expected, actual)) is not None:
                (HERE / 'failing_scenario.json').write_text(json.dumps([s]))
                return f'scenario {index} event {step} {s["events"][step]["kind"]}, config {s["config"]}\n{diff}'
            events += 1
        checked += 1

    scenarios = cases([], lambda i: scenario(random.Random(f'{seed}/{i}')), int(count), shard)
    if message := compare(HERE / 'oracle.mjs', scenarios, port, check):
        print(message)
        return 1
    print(f'{checked} scenarios, {events} events: identical')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
