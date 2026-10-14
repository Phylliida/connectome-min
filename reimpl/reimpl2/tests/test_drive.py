"""Differential test of the harness's pass (drive.step) against the real library run through the
harness's own sequence (oracle_l1.mjs's pass event).

    python3 tests/test_drive.py [sessions] [seed] [shard k/n]

Sessions are test_l1's, regrouped into passes: the messages and host changes before each of its ticks
join the pass that tick becomes, with a usage report given as a ratio of the armed estimate (some
outside the sanity band), a response reserve, a budget near the session's size or a share of its
priced total, and a routed window that leaves the compile retry room, or no room. After every pass
the compiled context or its refusal, the requests sent, the memories, queues and quarantines, the
committed levels and the calibration must match.

fixtures_drive.json adds passes, found by searching generated ones and cut down to what still reaches
their rule, that reach rules the generated ones seldom do; each says which. They run first.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import drive  # noqa: E402
import emit  # noqa: E402
import mint  # noqa: E402
import test_l1  # noqa: E402
from chunking import post_strip_tokens  # noqa: E402
from estimate import js_json  # noqa: E402
from config import Config  # noqa: E402
from library import cases, compare  # noqa: E402

FIXTURES = HERE / 'fixtures_drive.json'


def simulate(cfg, events):
    """The port, driven as the oracle drives the library. Fills in, for the oracle, what a pass leaves
    to the port's state: its usage report from the armed estimate, a budget from the priced total,
    and the window above the budget."""
    state, host, states = drive.State(), mint.Host(None), []
    for event in events:
        if 'host' in event:
            host = host._replace(tools=event['host']['tools'], system=event['host']['system'])
        pending = state                      # the pass's view of what it compiles, its usage report taken
        for message in event['messages']:
            pending = drive.append(pending, message, cfg)
        if 'ratio' in event:
            armed = pending.session.calibration.armed
            event['usage'] = max(1, round(armed * event.pop('ratio'))) if armed else None
        if event['usage'] is not None:
            pending = drive.observe(pending, event['usage'])
        if 'fraction' in event:
            total = sum(post_strip_tokens(list(pending.priced), pending.session.calibration.multiplier, cfg))
            event['budget'] = event['reserve'] + max(1, round(total * event.pop('fraction')))
        if 'room' in event:
            event['window'] = event['budget'] + event['reserve'] + event.pop('room')
        requests = []
        state, compiled, error = drive.step(state, event['messages'], event['usage'], event['budget'], event['reserve'],
                                            event['window'], host._replace(complete=test_l1.scripted(event['replies'], requests)), cfg)
        states.append({**test_l1.observed(state.memory, state.chunks, requests, error),
                       'compiled': {'refusal': list(compiled)} if isinstance(compiled, emit.Refusal) else {'messages': compiled},
                       'levels': {k: v for k, v in state.session.levels.items() if v != 0},
                       'multiplier': state.session.calibration.multiplier, 'armed': state.session.calibration.armed})
    return states


def session(r):
    """test_l1's session, regrouped into passes."""
    generated, events, pending, host, size = test_l1.session(r), [], [], None, 0
    if r.random() < 0.2:                       # a pass before the log holds anything
        events.append({'kind': 'pass', 'messages': [], 'reserve': 0, 'budget': 10**9, 'room': 0, 'usage': None, 'replies': []})
    for event in generated['events']:
        if event['kind'] == 'append':
            pending.append(event['message'])
            size += sum(len(json.dumps(b)) for b in event['message']['content']) // 3
        elif event['kind'] == 'host':
            host = {'tools': event.get('tools'), 'system': event.get('system')}
        elif event['kind'] == 'tick':
            reserve = r.choice([0, 0, 50, 400])
            budget = r.choice([{'budget': 10**9}, {'budget': reserve + max(1, round(size * r.uniform(0.15, 1.3)))},
                               {'fraction': r.uniform(0.85, 1.3)}, {'fraction': r.uniform(0.85, 1.3)}])
            usage = {'ratio': r.uniform(0.5, 1.9)} if r.random() < 0.7 else {'usage': None}
            events.append({'kind': 'pass', 'messages': pending, **({'host': host} if host else {}), 'reserve': reserve, **budget,
                           'room': r.choice([0, 0, r.randint(0, size), 10**9]), **usage, 'replies': event['replies']})
            pending, host = [], None
    return {'cfg': generated['cfg'], 'events': events}


def main(count=150, seed=0, shard='0/1'):
    checked = passes = calls = refused = 0
    fixtures = [{'cfg': Config(**f['cfg']), 'events': f['events']} for f in json.loads(FIXTURES.read_text())]

    def port(s):
        # As JSON.stringify writes them: the oracle read the messages' numbers as doubles, which the compiled context copies
        states = json.loads(js_json(simulate(s['cfg'], s['events'])))       # first: it fills in the passes
        return {'config': test_l1.library_config(s), 'events': s['events']}, states

    def check(index, s, states, line):
        nonlocal checked, passes, calls, refused
        for step, (library, mine) in enumerate(zip(json.loads(line), states, strict=True)):
            for key in mine:
                if mine[key] != library[key]:
                    (HERE / 'failing_session.json').write_text(json.dumps({'config': test_l1.library_config(s), 'events': s['events'][:step + 1]}))
                    return (f'session {index} pass {step}: {key}\n'
                            f'  library {json.dumps(library[key])[:2500]}\n  port    {json.dumps(mine[key])[:2500]}')
            passes += 1
            calls += len(mine['requests'])
            refused += 'refusal' in mine['compiled']
        checked += 1

    sessions = cases(fixtures, lambda i: session(random.Random(f'{seed}/{i}')), int(count), shard)
    if message := compare(HERE / 'oracle_l1.mjs', sessions, port, check):
        print(message)
        return 1
    print(f'{checked} sessions, {passes} passes ({refused} compiles refused), {calls} calls: identical')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
