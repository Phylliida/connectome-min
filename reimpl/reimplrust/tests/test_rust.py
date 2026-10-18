"""Differential test of the Rust port against reimplclean2: each generated session (reimplclean2's
tests/generate.py, forks included) runs through the Python port, which records it as a script (each
pass's fork, messages, usage and budgets, and the summarizer's replies in call order); the Rust
replay binary runs the script, and every summarizer request, every compile (read out in full) and
every pass's error must be the same, and so must every record of the sessions' files, hashes
included. Values compare as JSON (1 == 1.0, key order aside). Every third session is handed from
the Python port to the Rust one halfway, on the Python port's files, and every third the other way
round: each must go on from the other's files as it would from its own. Then the planner alone, on
test_counter's generated frontiers with made-up prices: the Rust plan must be the Python one.

    python3 tests/test_rust.py [sessions] [seed]
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parents[1] / 'reimplclean2'
REPLAY = HERE.parent / 'target' / 'release' / 'replay'
SHARDS, PASSES = 16, 150
HANDOFF = PASSES // 2


def drain(turns) -> list[dict]:
    return [{**t, 'content': list(t['content'])} for t in turns]


def python(r, folder: str, snapshot: str | None = None) -> tuple[list, dict]:
    """reimplclean2's events for one session, and the script that replays it; with `snapshot`, a copy
    of the folder as it was after HANDOFF passes."""
    from config import Config
    from drive import step
    from emit import Refusal
    from generate import settings, summarizer, turn
    from memory import Host
    from store import Session

    events, replies, passes, now = [], [], [], [0]
    odd = random.Random(str(r.getstate()[1][:4]))                  # its own stream: r's draws stay as they were
    cfg, W, tools, refusals = settings(r, Config)
    inner = summarizer(r, refusals, lambda request: events.append([f'request {now[0]}', {**request, 'messages': drain(request['messages'])}]))

    def complete(request):
        try:
            response = inner(request)
        except Exception as error:
            replies.append({'raise': repr(error)})
            raise
        replies.append({'response': response})
        return response

    host, s = Host(complete, tools=tools), Session(os.path.join(folder, 'main'))
    for t in range(PASSES):
        now[0], before, fork = t, os.path.basename(s.path), None
        if t == HANDOFF and snapshot:
            copy(folder, snapshot)
        if r.random() < 0.02 and (n := s.count('messages')) > 2:
            fork = [r.randrange(n), f'fork{t}']
            s = Session.fork(s, fork[0], os.path.join(folder, fork[1]))
        if r.random() < 0.03:
            W = r.choice([4000, 12000, 40000])
        last = s.last('compiles') or {}
        usage = round(last['armed'] * r.uniform(0.8, 1.2)) if last.get('armed') and r.random() < 0.5 else None
        messages = turn(r, t) + odd_turn(odd, t)
        passes.append({'t': t, 'before': before, 'replies': len(replies), 'fork': fork, 'messages': messages, 'usage': usage,
                       'budget': W + 1000, 'reserve': 1000, 'window': W + 5000})
        compiled, error = step(s, messages, usage, W + 1000, 1000, W + 5000, host, cfg)
        events.append([f'compile {t}', list(compiled) if isinstance(compiled, Refusal) else drain(compiled)])
        events.append([f'error {t}', repr(error)])
    return json.loads(json.dumps(events, ensure_ascii=False)), {'config': asdict(cfg), 'tools': tools, 'start': 'main', 'passes': passes, 'replies': replies}


def odd_turn(r, t) -> list[dict]:
    """Now and then, tool traffic generate.turn doesn't make: a call answered twice in one turn, a non-user
    turn mixing text and a tool result (the newest message, so it carries the cache mark when it splits),
    and a call that repeats its id."""
    text = {'type': 'text', 'text': 'odd é'}
    call = lambda tid, n=1: {'participant': 'assistant', 'content': [text] + [{'type': 'tool_use', 'id': tid, 'name': 'read', 'input': {}}] * n}
    result = lambda tid: {'type': 'tool_result', 'toolUseId': tid, 'content': 'ok'}
    out = []
    if r.random() < 0.04:
        out += [call(f'd{t}'), {'participant': 'user', 'content': [result(f'd{t}'), text, result(f'd{t}')]}]
    if r.random() < 0.04:
        out += [call(f'u{t}', 2), {'participant': 'user', 'content': [result(f'u{t}')]}]
    if r.random() < 0.04:
        out += [call(f'm{t}'), {'participant': r.choice(['assistant', 'Claude']), 'content': [text, result(f'm{t}'), text, result(f'x{t}')]}]
    return out


class Scripted(Exception):
    def __repr__(self):
        return self.args[0]


def replay_python(script: dict, folder: str) -> list:
    """reimplclean2 running a script on a session folder, as the Rust replay does."""
    from config import Config
    from drive import step
    from emit import Refusal
    from memory import Host
    from store import Session

    events, replies, now = [], iter(script['replies']), [0]

    def complete(request):
        events.append([f'request {now[0]}', {**request, 'messages': drain(request['messages'])}])
        reply = next(replies)
        if 'raise' in reply:
            raise Scripted(reply['raise'])
        return reply['response']

    cfg, host, s = Config(**script['config']), Host(complete, tools=script['tools']), Session(os.path.join(folder, script['start']))
    for p in script['passes']:
        now[0] = p['t']
        if p['fork']:
            s = Session.fork(s, p['fork'][0], os.path.join(folder, p['fork'][1]))
        compiled, error = step(s, p['messages'], p['usage'], p['budget'], p['reserve'], p['window'], host, cfg)
        events.append([f'compile {p["t"]}', list(compiled) if isinstance(compiled, Refusal) else drain(compiled)])
        events.append([f'error {p["t"]}', repr(error)])
    return json.loads(json.dumps(events, ensure_ascii=False))


def rust(script: dict, folder: str, work: str) -> list | str:
    """The Rust replay's events, or why it failed."""
    path = os.path.join(work, 'script.json')
    Path(path).write_text(json.dumps(script, ensure_ascii=False))
    run = subprocess.run([str(REPLAY), path, folder], capture_output=True, text=True)
    return f'replay failed: {run.stderr[-2000:]}' if run.returncode else json.loads(run.stdout)


def head(script: dict, h: int) -> dict:
    return {**script, 'passes': script['passes'][:h]}


def tail(script: dict, h: int) -> dict:
    """The script from pass h on, from the session and reply it reached."""
    first = script['passes'][h]
    return {**script, 'start': first['before'], 'passes': script['passes'][h:], 'replies': script['replies'][first['replies']:]}


def copy(folder: str, to: str) -> None:
    """A session folder's copy, its forks' parents in the copy."""
    shutil.copytree(folder, to)
    for meta in Path(to).rglob('fork.json'):
        fork = json.loads(meta.read_text())
        meta.write_text(json.dumps({**fork, 'parent': fork['parent'].replace(os.path.abspath(folder), os.path.abspath(to))}))


def passes(events: list, lo: int, hi: int) -> list:
    return [e for e in events if lo <= int(e[0].split()[1]) < hi]


def files(folder: str) -> dict:
    """Every record of every file, fork.json without the parent's path."""
    out = {}
    for path in sorted(Path(folder).rglob('*.json*')):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out[str(path.relative_to(folder))] = [{k: v for k, v in r.items() if k != 'parent'} for r in records]
    return out


def first_difference(a, b, path='') -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f'{path}.{k}: only in {"python" if k in a else "rust"}'
            if found := first_difference(a[k], b[k], f'{path}.{k}'):
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        for k, (x, y) in enumerate(zip(a, b)):
            if found := first_difference(x, y, f'{path}[{k}]'):
                return found
        return None if len(a) == len(b) else f'{path}: {len(a)} items in python, {len(b)} in rust'
    return None if a == b and (type(a) is bool) == (type(b) is bool) else f'{path}: python {a!r:.200}, rust {b!r:.200}'


def session(i: int, seed: str) -> tuple[int, str | None]:
    """(events compared, the first difference or None). Every third session the Python port hands the
    session over to the Rust one halfway, and every third the other way round."""
    work = tempfile.mkdtemp()
    try:
        mine, theirs, snapshot = (os.path.join(work, name) for name in ('python', 'rust', 'snapshot'))
        handoff = i % 3
        events, script = python(random.Random(f'{seed}/{i}'), mine, snapshot if handoff == 1 else None)
        if handoff == 0:
            stages = [(events, lambda: rust(script, theirs, work))]
        elif handoff == 1:
            theirs = snapshot
            stages = [(passes(events, HANDOFF, PASSES), lambda: rust(tail(script, HANDOFF), theirs, work))]
        else:
            stages = [(passes(events, 0, HANDOFF), lambda: rust(head(script, HANDOFF), theirs, work)),
                      (passes(events, HANDOFF, PASSES), lambda: replay_python(tail(script, HANDOFF), theirs))]
        compared = 0
        for expected, run in stages:
            if isinstance(replayed := run(), str):
                return compared, replayed
            for e, (a, b) in enumerate(zip(expected, replayed)):
                if a[0] != b[0] or (found := first_difference(a[1], b[1])):
                    return compared, f'handoff {handoff}, event {a[0]} / {b[0]}: {found if a[0] == b[0] else "different events"}'
            if len(expected) != len(replayed):
                return compared, f'handoff {handoff}: {len(expected)} events expected, {len(replayed)} replayed'
            compared += len(expected)
        if found := first_difference(files(mine), files(theirs), 'files'):
            return compared, f'handoff {handoff}: {found}'
        return compared, None
    finally:
        shutil.rmtree(work)


def shard(k: int, n: int, count: int, seed: str) -> None:
    """Child process: this shard's sessions, as JSON lines (index, events, difference)."""
    sys.path[:0] = [str(PYTHON), str(PYTHON / 'tests')]
    for i in range(k, count, n):
        events, difference = session(i, seed)
        print(json.dumps([i, events, difference]), flush=True)
        if difference:
            return


def plans(count: int, seed: str) -> str:
    """The planner on test_counter's frontier sequences, each plan carried into the next."""
    sys.path[:0] = [str(PYTHON)]
    from config import Config
    from planner import chunks, plan
    cases, expected = [], []
    for i in range(count):
        r = random.Random(f'{seed}/plan/{i}')
        k, slack, unfold_below, salt = r.choice([2, 3, 6]), r.choice([0.1, 0.0, 0.3]), r.choice([0.8, 0.5, 0.9]), r.randrange(10 ** 6)
        cfg = Config(merge_threshold=k, slack=slack, unfold_below=unfold_below)
        cost = lambda node: (800 + (node[1] * 7919 + salt) % 2201) if node[0] == 0 else 50 + (node[0] * 104729 + node[1] * 7919 + salt) % 551
        W, have, frontier = r.choice([2000, 10000, 40000]), [0], []
        exists = lambda node: node[0] < len(have) and node[1] < have[node[0]]
        for _ in range(r.randint(5, 200)):
            have[0] += r.choice([0, 1, 1, 2])
            frontier = frontier + [(0, j) for j in range(sum(1 for n in frontier for _ in chunks(n, k)), have[0])]
            for n in range(len(have)):
                allowed = have[0] if n == 0 else have[n] // k
                if n + 1 == len(have) and allowed:
                    have.append(0)
                if n + 1 < len(have):
                    have[n + 1] = max(have[n + 1], allowed - r.choice([0, 0, 1, 2]))
            if r.random() < 0.05:
                W = max(500, round(W * r.uniform(0.5, 2)))
            fixed = r.randint(0, W // 4)
            cases.append({'frontier': frontier, 'have': list(have), 'salt': salt, 'fixed': fixed, 'W': W, 'k': k, 'slack': slack, 'unfold_below': unfold_below})
            frontier = plan(frontier, cost, exists, fixed, W, cfg)
            expected.append([list(n) for n in frontier])
    run = subprocess.run([str(REPLAY), '--plan'], input=json.dumps(cases), capture_output=True, text=True)
    if run.returncode:
        return f'replay --plan failed: {run.stderr[-2000:]}'
    for case, mine, rust in zip(cases, expected, json.loads(run.stdout), strict=True):
        if mine != rust:
            return f'plan of {case}: python {mine}, rust {rust}'
    return f'{len(cases)} plans'


def main(count=48, seed='0'):
    build = subprocess.run(['cargo', 'build', '--release', '--quiet', '--bin', 'replay'], cwd=HERE.parent, capture_output=True, text=True)
    if build.returncode:
        print(build.stderr)
        return 1
    runs = [subprocess.Popen([sys.executable, __file__, '--shard', str(k), str(SHARDS), str(count), seed], stdout=subprocess.PIPE, text=True)
            for k in range(SHARDS)]
    planned = plans(300, seed)
    if not planned.endswith(' plans'):
        print(planned)
        return 1
    results = sorted(json.loads(line) for run in runs for line in run.communicate()[0].splitlines())
    if failed := [run.returncode for run in runs if run.returncode]:
        print(f'a shard failed (exit {failed[0]})')
        return 1
    if differences := [(i, d) for i, _, d in results if d]:
        print(f'session {differences[0][0]}: {differences[0][1]}')
        return 1
    print(f'{len(results)} sessions, {sum(e for _, e, _ in results)} events and every file record, and {planned}: the same as reimplclean2')
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--shard']:
        shard(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        sys.exit(main(*sys.argv[1:]))
