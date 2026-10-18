"""Property test of whole sessions on disk (drive.step), with a scripted summarizer that writes, fails,
replies empty and now and then refuses. After every pass:
  - the last compile's frontier covers every closed chunk once, in order, with levels falling toward
    the present, by memories that exist;
  - layer 1's entry p covers chunk p, and layer n+1's entry p the span of layer n's p*k .. p*k+k-1;
  - in the compiled messages, as streamed, every tool_result answers a tool_use in the turn before it,
    and every tool_use is answered in the turn after it; so in every summarizer request, which also
    carries no stale cache_control, blank text or empty turn;
  - after a refusal no memory is written, and no memory fails more than attempt_limit times at one
    log length.
Now and then a run forks at an earlier message and goes on in the fork (forks of forks too): the fork
must share its parent's messages through that one, pass the same checks, and leave its parent's files
untouched. Then one long session checks the point of the design: the records a pass reads don't grow
with the log, streaming the context included.

    python3 tests/test_session.py [sessions] [seed]
"""
import os
import random
from collections import Counter
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from config import Config  # noqa: E402
from drive import step  # noqa: E402
from emit import Refusal  # noqa: E402
from generate import settings, summarizer, turn  # noqa: E402
from memory import Host  # noqa: E402
from planner import chunks  # noqa: E402
from store import Session  # noqa: E402


class Failed(Exception):
    pass


def drain(turns) -> list[dict]:
    """Streamed turns, read out."""
    return [{**t, 'content': list(t['content'])} for t in turns]


def paired(turns: list[dict]) -> None:
    """Every tool_result answers a tool_use in the turn before it, and every tool_use is answered in the
    turn after it."""
    for before, after in zip([{'content': []}] + turns, turns + [{'content': []}]):
        uses = {b['id'] for b in before['content'] if b['type'] == 'tool_use'}
        results = {b['toolUseId'] for b in after['content'] if b['type'] == 'tool_result'}
        if uses != results and (uses or results):
            raise Failed(f'tool calls {sorted(uses)} answered by {sorted(results)}')


def request(r):
    """What a summarizer's request must be: tool calls paired, no stale cache_control, nothing blank."""
    turns = drain(r['messages'])
    paired(turns)
    if any('cache_control' in b or b['type'] == 'text' and not b['text'].strip() for t in turns for b in t['content']) or not all(t['content'] for t in turns):
        raise Failed('a request carries a cache_control marker, blank text or an empty turn')


def check(s: Session, compiled, cfg: Config):
    k, closed = cfg.merge_threshold, s.count('layer0')
    if (last := s.last('compiles')) and last.get('frontier') is not None:
        F = [tuple(n) for n in last['frontier']]
        covered = [c for node in F for c in chunks(node, k)]
        if covered != list(range(len(covered))) or len(covered) > closed or any(b[0] > a[0] for a, b in zip(F, F[1:])) \
                or not all(node[1] < s.count(f'layer{node[0]}') for node in F):
            raise Failed(f'bad frontier {F} with {closed} chunks')
    for n in range(1, 9):
        for m in s.span(f'layer{n}', 0, s.count(f'layer{n}') - 1):
            lo, hi = (s.get('layer0', m['id']),) * 2 if n == 1 else (s.get(f'layer{n - 1}', m['id'] * k), s.get(f'layer{n - 1}', m['id'] * k + k - 1))
            if (m['first'], m['last']) != (lo['first'], hi['last']):
                raise Failed(f'memory L{n}-{m["id"]} covers {m["first"]}..{m["last"]}, not {lo["first"]}..{hi["last"]}')
    attempts = list(s.span('attempts', 0, s.count('attempts') - 1))
    if any(a['outcome'] == 'refusal' for a in attempts[:-1]):
        raise Failed('an attempt followed a refusal')
    if over := [key for key, n in Counter((tuple(a['target']), a['messages']) for a in attempts).items() if n > cfg.attempt_limit]:
        raise Failed(f'{over[0][0]} tried more than {cfg.attempt_limit} times at {over[0][1]} messages')
    if not isinstance(compiled, Refusal):
        paired(drain(compiled))


def run(r, folder, passes, checked=True):
    cfg, W, tools, refusals = settings(r, Config)
    s, host, frozen = Session(os.path.join(folder, 'main')), Host(summarizer(r, refusals, request), tools=tools), {}   # frozen: each forked session's file sizes when forked
    for t in range(passes):
        if checked and r.random() < 0.02 and (n := s.count('messages')) > 2:
            at = r.randrange(n)
            frozen[s.path] = sizes(s.path)
            fork = Session.fork(s, at, os.path.join(folder, f'fork{t}'))
            if list(fork.span('messages', 0, at)) != list(s.span('messages', 0, at)) or fork.count('messages') != at + 1:
                raise Failed(f'fork at {at} does not share its parent\'s messages')
            s = fork
        if r.random() < 0.03:
            W = r.choice([4000, 12000, 40000])
        last = s.last('compiles') or {}
        usage = round(last['armed'] * r.uniform(0.8, 1.2)) if last.get('armed') and r.random() < 0.5 else None
        memories = [s.count(f'layer{n}') for n in range(1, 9)]
        refused = (s.last('attempts') or {}).get('outcome') == 'refusal'
        compiled, _ = step(s, turn(r, t), usage, W + 1000, 1000, W + 5000, host, cfg)
        if checked:
            check(s, compiled, cfg)
            if changed := [path for path, before in frozen.items() if sizes(path) != before]:
                raise Failed(f'a fork wrote to its parent {changed[0]}')
            if refused and [s.count(f'layer{n}') for n in range(1, 9)] != memories:
                raise Failed('a memory was written after a refusal')
    return len(frozen)


def sizes(path):
    return {f: os.path.getsize(os.path.join(path, f)) for f in sorted(os.listdir(path))}


def bounded(seed) -> str:
    """Records read per pass over one long session, early against late."""
    reads, wrap = [0], lambda it: (reads.__setitem__(0, reads[0] + 1) or x for x in it)
    span, backwards, read = Session.span, Session.backwards, Session.read      # reads from disk; Session.get caches
    Session.span = lambda self, *a: wrap(span(self, *a))
    Session.backwards = lambda self, *a: wrap(backwards(self, *a))
    Session.read = lambda self, *a: reads.__setitem__(0, reads[0] + 1) or read(self, *a)
    folder, per_pass = tempfile.mkdtemp(), []
    try:
        r = random.Random(f'{seed}/long')
        cfg = Config(target_chunk_tokens=1500, recent_window_tokens=3000, summary_target_tokens=200, recall_budget_tokens=20000)
        s, host = Session(folder), Host(summarizer(r, 0, lambda request: drain(request['messages'])), tools=[{'name': 'read'}])
        for t in range(1200):
            reads[0] = 0
            drain(step(s, turn(r, t), None, 21000, 1000, 25000, host, cfg)[0])
            per_pass.append(reads[0])
        early, late = max(per_pass[200:400]), max(per_pass[1000:1200])
        if late > 1.5 * early:
            raise Failed(f'reads grow with the log: at most {early} per pass early on, {late} late, '
                         f'over {s.count("messages")} messages')
        return f'{s.count("messages")} messages: at most {early} records read per pass early on, {late} late'
    finally:
        Session.span, Session.backwards, Session.read = span, backwards, read
        shutil.rmtree(folder)


def main(count=30, seed=0):
    forks = 0
    for i in range(int(count)):
        folder = tempfile.mkdtemp()
        try:
            forks += run(random.Random(f'{seed}/{i}'), folder, 150)
        except Failed as error:
            print(f'session {i}: {error}')
            return 1
        finally:
            shutil.rmtree(folder)
    try:
        detail = bounded(seed)
    except Failed as error:
        print(error)
        return 1
    print(f'{count} sessions hold, with {forks} forks; {detail}')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
