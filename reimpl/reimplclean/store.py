"""A session on disk: append-only JSONL logs in one directory, read through jsonlparse without loading
them. Every record has an `id` counting up from 0 in its file, so a lookup is a binary search, and
the session's state is the logs' tails:

  messages.jsonl   the chat log: {id, participant, content}
  layer0.jsonl     chunks, in log order: {id, first, last, raw} (message ids; raw = uncalibrated price)
  layer<n>.jsonl   memories of level n, in order: {id, first, last, text, blocks, recall, tokens,
                   provenance}. Layer 1's entry p summarizes chunk p; layer n+1's entry p merges layer
                   n's entries p*k .. p*k+k-1 (k = merge_threshold, fixed for a session).
  compiles.jsonl   one record per compile: the frontier shown, calibration, cache keys, escalation
  attempts.jsonl   memory attempts that failed: {id, target, messages, outcome}
"""
import os
from collections import OrderedDict
from itertools import takewhile
from typing import Iterator

from jsonlparse import JsonlFile


class Session:
    def __init__(self, path: str, cached: int = 4096):
        os.makedirs(path, exist_ok=True)
        self.path, self.files, self.memos, self.cached = path, {}, OrderedDict(), cached

    def memo(self, key, compute):
        """compute(), kept under `key` while among the `cached` most recently used: for what derives from
        records, which never change once written."""
        if key in self.memos:
            self.memos.move_to_end(key)
            return self.memos[key]
        value = self.memos[key] = compute()
        if len(self.memos) > self.cached:
            self.memos.popitem(last=False)
        return value

    def file(self, name: str) -> JsonlFile:
        if name not in self.files:
            self.files[name] = JsonlFile(os.path.join(self.path, f'{name}.jsonl'))
        return self.files[name]

    def last(self, name: str) -> dict | None:
        """The newest record, or None for an empty or absent file (which this leaves absent)."""
        return self.file(name).last() if name in self.files or os.path.exists(os.path.join(self.path, f'{name}.jsonl')) else None

    def count(self, name: str) -> int:
        return last['id'] + 1 if (last := self.last(name)) else 0

    def get(self, name: str, i: int) -> dict:
        return self.memo((name, i), lambda: self.read(name, i))

    def read(self, name: str, i: int) -> dict:
        """Record i, from disk."""
        return self.file(name).get('id', i)

    def span(self, name: str, first: int, last: int) -> Iterator[dict]:
        """Records first..last, in order."""
        return takewhile(lambda r: r['id'] <= last, self.file(name).iter_at('id', first)) if first <= last else iter(())

    def backwards(self, name: str) -> Iterator[dict]:
        return self.file(name).backwards()

    def append(self, name: str, record: dict) -> dict:
        record = {**record, 'id': self.count(name)}
        self.file(name).append(record)
        return record
