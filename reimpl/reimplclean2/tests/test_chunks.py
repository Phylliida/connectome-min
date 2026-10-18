"""Differential test of chunk formation (chunking.py, through drive.append) against the real library
(oracle.mjs), where the clean version means to match it: append-only sessions with no images and no
calibration, the head window off. After every append the chunks (as message ids) and the tail's
first message must match.

    python3 tests/test_chunks.py [sessions] [seed] [shard k/n]
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from chunking import tail_start  # noqa: E402
from config import Config  # noqa: E402
from drive import append  # noqa: E402
from library import cases, compare  # noqa: E402
from store import Session  # noqa: E402

WORDS = 'the a memory tool file read wrote error fixed test passes ok so then why because é 漢字 😀 {"k": 1} ``` x'.split()


def words(r, n):
    return ' '.join(r.choice(WORDS) for _ in range(n))


def session(r):
    cfg = Config(target_chunk_tokens=r.choice([30, 150, 600, 3000]), recent_window_tokens=r.choice([0, 100, 400, 1500]),
                 l1_holdback_chunks=r.choice([0, 1, 2]))
    messages, pending = [], None
    for i in range(r.randint(1, 120)):
        size = r.choice([r.randint(0, 20), r.randint(20, 200), r.randint(200, 1500)])
        if pending:                                           # answer the last call, usually
            blocks = [{'type': 'tool_result', 'toolUseId': pending, 'content': words(r, size)}]
            participant, pending = 'user', None
        else:
            blocks = [{'type': 'text', 'text': words(r, size)}] * (r.random() < 0.9)
            if r.random() < 0.4:
                pending = f't{i}'
                blocks.append({'type': 'tool_use', 'id': pending, 'name': 'read', 'input': {'path': words(r, 3)}})
            participant = r.choice(['user', 'assistant'])
        messages.append({'id': f'm{i}', 'participant': participant, 'content': blocks or [{'type': 'text', 'text': ''}]})
    return cfg, messages


def simulate(cfg, messages):
    folder = tempfile.mkdtemp()
    try:
        s, states = Session(folder), []
        for message in messages:
            append(s, message, cfg)
            chunks = [[f'm{i}' for i in range(c['first'], c['last'] + 1)] for c in s.span('layer0', 0, s.count('layer0') - 1)]
            states.append({'chunks': chunks, 'tail': tail_start(s, 1.0, cfg)})
        return states
    finally:
        shutil.rmtree(folder)


def library_config(cfg):
    return {'targetChunkTokens': cfg.target_chunk_tokens, 'recentWindowTokens': cfg.recent_window_tokens, 'headWindowTokens': 0,
            'attachmentsIgnoreSize': True, 'l1HoldbackChunks': cfg.l1_holdback_chunks}


def main(count=300, seed=0, shard='0/1'):
    checked = appends = 0

    def port(case):
        cfg, messages = case
        payload = {'config': library_config(cfg), 'events': [{'kind': 'append', 'message': m} for m in messages]}
        return payload, simulate(cfg, messages)

    def check(index, case, states, line):
        nonlocal checked, appends
        for step, (library, mine) in enumerate(zip(json.loads(line), states, strict=True)):
            for key in mine:
                if mine[key] != library[key]:
                    return f'session {index} append {step}: {key}\n  library {library[key]}\n  clean   {mine[key]}'
        checked, appends = checked + 1, appends + len(states)

    sessions = cases([], lambda i: session(random.Random(f'{seed}/{i}')), int(count), shard)
    if message := compare(HERE / 'oracle.mjs', sessions, port, check):
        print(message)
        return 1
    print(f'{checked} sessions, {appends} appends: chunks and tail identical')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
