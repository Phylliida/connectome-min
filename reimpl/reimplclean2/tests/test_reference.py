"""Differential test against reimplclean, which builds every request in memory: on the same generated
sessions (generate.py, forks included), every compile and every summarizer request must stream the
same turns as reimplclean returns, and the sessions' files must hold the same records but for the
hashes (compiles' keys, memories' requestHash), which this version takes over where content is, not
over the content.

Each version runs in its own processes (the two share module names), sharded; a trace records a
digest of every event and file, and at the first difference both versions rerun that session in full
to show it.

    python3 tests/test_reference.py [sessions] [seed]
"""
import difflib
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSIONS = {'reimplclean': HERE.parents[1] / 'reimplclean', 'reimplclean2': HERE.parent}
SHARDS, PASSES = 8, 150


def drain(turns) -> list[dict]:
    return [{**t, 'content': list(t['content'])} for t in turns]


def trace(r, folder: str, full: bool) -> list:
    """One session's events and files: their digests, or with `full` themselves."""
    from config import Config
    from drive import step
    from emit import Refusal
    from generate import settings, summarizer, turn
    from memory import Host
    from store import Session

    events = []
    record = lambda kind, value: events.append([kind, value if full else hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()])
    cfg, W, tools, refusals = settings(r, Config)
    host = Host(summarizer(r, refusals, lambda request: record('request', {**request, 'messages': drain(request['messages'])})), tools=tools)
    s = Session(os.path.join(folder, 'main'))
    for t in range(PASSES):
        if r.random() < 0.02 and (n := s.count('messages')) > 2:
            s = Session.fork(s, r.randrange(n), os.path.join(folder, f'fork{t}'))
        if r.random() < 0.03:
            W = r.choice([4000, 12000, 40000])
        last = s.last('compiles') or {}
        usage = round(last['armed'] * r.uniform(0.8, 1.2)) if last.get('armed') and r.random() < 0.5 else None
        compiled, error = step(s, turn(r, t), usage, W + 1000, 1000, W + 5000, host, cfg)
        record(f'compile {t}', list(compiled) if isinstance(compiled, Refusal) else drain(compiled))
        record(f'error {t}', repr(error))
    for path in sorted(Path(folder).rglob('*.json*')):
        record(str(path.relative_to(folder)), [unhashed(line) for line in path.read_text().splitlines()])
    return events


def unhashed(line: str):
    """A file's record without the hashes the versions take differently, or the parent's path."""
    value = json.loads(line)
    if isinstance(value, dict):
        value.pop('keys', None), value.pop('parent', None)
        if isinstance(value.get('provenance'), dict):
            value['provenance'].pop('requestHash', None)
    return value


def run_traces(shard: str, count: int, seed: str, full: bool) -> None:
    """Child process: this version's traces of its shard's sessions, as one JSON object on stdout."""
    k, n = map(int, shard.split('/'))
    out = {}
    for i in range(k, count, n):
        folder = tempfile.mkdtemp()
        try:
            out[i] = trace(random.Random(f'{seed}/{i}'), folder, full)
        finally:
            shutil.rmtree(folder)
    print(json.dumps(out, ensure_ascii=False))


def traces(version: str, shards: list[str], count: int, seed: str, full=False) -> list[subprocess.Popen]:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(VERSIONS[version]), str(HERE)])}
    return [subprocess.Popen([sys.executable, __file__, '--trace', shard, str(count), seed, str(full)], env=env, cwd=VERSIONS[version],
                             stdout=subprocess.PIPE, text=True) for shard in shards]


def collect(runs: list[subprocess.Popen]) -> dict:
    out = {}
    for run in runs:
        text, _ = run.communicate()
        if run.returncode:
            raise SystemExit(f'a trace failed (exit {run.returncode})')
        out.update(json.loads(text))
    return out


def show(i: str, seed: str, event: int) -> str:
    """Session i's first differing event, in full from both versions."""
    a, b = (collect(traces(v, [f'{i}/{int(i) + 1}'], int(i) + 1, seed, True))[i][event] for v in VERSIONS)
    lines = lambda e: json.dumps(e[1], ensure_ascii=False, indent=1).splitlines()
    return '\n'.join(list(difflib.unified_diff(lines(a), lines(b), 'reimplclean', 'reimplclean2', lineterm='', n=2))[:60])


def main(count=40, seed='0'):
    count = int(count)
    shards = [f'{k}/{SHARDS}' for k in range(SHARDS)]
    runs = {v: traces(v, shards, count, seed) for v in VERSIONS}
    reference, mine = (collect(runs[v]) for v in VERSIONS)
    events = 0
    for i in sorted(reference, key=int):
        for e, (a, b) in enumerate(zip(reference[i], mine[i])):
            if a != b:
                print(f'session {i}: {a[0]} differs ({b[0]} here)\n{show(i, seed, e)}')
                return 1
        if len(reference[i]) != len(mine[i]):
            print(f'session {i}: {len(reference[i])} events in reimplclean, {len(mine[i])} here')
            return 1
        events += len(mine[i])
    print(f'{count} sessions, {events} compiles, requests and files: the same as reimplclean')
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--trace']:
        run_traces(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == 'True')
    else:
        sys.exit(main(*sys.argv[1:]))
