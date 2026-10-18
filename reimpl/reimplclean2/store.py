"""A session on disk: append-only JSONL logs in one directory, read through jsonlparse without loading
them. Every record has an `id` counting up from 0, so a lookup is a binary search, and the session's
state is the logs' tails:

  messages.jsonl   the chat log: {id, participant, content}
  layer0.jsonl     chunks, in log order: {id, first, last, raw} (message ids; raw = uncalibrated price)
  layer<n>.jsonl   memories of level n, in order: {id, first, last, text, blocks, recall, tokens,
                   provenance}. Layer 1's entry p summarizes chunk p; layer n+1's entry p merges layer
                   n's entries p*k .. p*k+k-1 (k = merge_threshold, fixed for a session).
  compiles.jsonl   one record per compile: the frontier shown, calibration, cache keys, escalation
  attempts.jsonl   memory attempts that failed: {id, target, messages, outcome}

A fork (Session.fork) shares its parent's history through one message: fork.json names the parent and,
per file, how many of the parent's records the fork shares (the messages through that one, and the
chunks and memories that end by then). The fork's own records continue those ids.
"""
import json
import os
from bisect import bisect_left
from functools import lru_cache
from itertools import chain, count, takewhile
from typing import Iterator

from jsonlparse import JsonlFile


class Session:
    def __init__(self, path: str, cached: int = 4096):
        os.makedirs(path, exist_ok=True)
        self.path, self.files = path, {}
        self.get = lru_cache(maxsize=cached)(self.read)       # records never change once written
        meta = os.path.join(path, 'fork.json')
        fork = json.load(open(meta)) if os.path.exists(meta) else {}
        self.parent, self.shared = Session(fork['parent'], cached) if fork else None, fork.get('shared', {})

    @classmethod
    def fork(cls, parent: 'Session', at: int, path: str) -> 'Session':
        """A new session at `path` whose history is `parent`'s through message `at`. It starts from the
        parent's last frontier, cut to the nodes it shares, and calibration."""
        shared = {'messages': at + 1}
        for n in takewhile(lambda n: parent.count(f'layer{n}'), count(0)):
            shared[f'layer{n}'] = bisect_left(range(parent.count(f'layer{n}')), True, key=lambda i: parent.get(f'layer{n}', i)['last'] > at)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'fork.json'), 'w') as f:
            json.dump({'parent': os.path.abspath(parent.path), 'at': at, 'shared': shared}, f)
        child, last = cls(path), parent.last('compiles') or {}
        frontier = list(takewhile(lambda node: node[1] < shared.get(f'layer{node[0]}', 0), last.get('frontier', [])))
        child.append('compiles', {'frontier': frontier, 'multiplier': last.get('multiplier', 1.0), 'armed': 0, 'keys': None, 'escalated': False})
        return child

    def file(self, name: str) -> JsonlFile:
        if name not in self.files:
            self.files[name] = JsonlFile(os.path.join(self.path, f'{name}.jsonl'))
        return self.files[name]

    def last(self, name: str) -> dict | None:
        """The newest record, or None (an absent file stays absent)."""
        exists = name in self.files or os.path.exists(os.path.join(self.path, f'{name}.jsonl'))
        if exists and (own := self.file(name).last()):
            return own
        return self.parent.get(name, n - 1) if (n := self.shared.get(name, 0)) else None

    def count(self, name: str) -> int:
        return last['id'] + 1 if (last := self.last(name)) else 0

    def read(self, name: str, i: int) -> dict:
        """Record i, from disk: the parent's while the fork shares it (get: the same, cached, for the
        small records; a message is only ever read)."""
        return self.parent.read(name, i) if i < self.shared.get(name, 0) else self.file(name).get('id', i)

    def span(self, name: str, first: int, last: int) -> Iterator[dict]:
        """Records first..last, in order."""
        n, start = self.shared.get(name, 0), max(first, self.shared.get(name, 0))
        own = takewhile(lambda r: r['id'] <= last, self.file(name).iter_at('id', start)) if start <= last else ()
        return chain(self.parent.span(name, first, min(last, n - 1)) if first < n else (), own)

    def backwards(self, name: str) -> Iterator[dict]:
        own = self.file(name).backwards() if self.count(name) > self.shared.get(name, 0) else ()
        return chain(own, (self.get(name, i) for i in range(self.shared.get(name, 0) - 1, -1, -1)))

    def append(self, name: str, record: dict) -> dict:
        record = {**record, 'id': self.count(name)}
        self.file(name).append(record)
        return record
