"""Compares the port with the real library, case by case: an oracle script drives the library under
node on each case's payload, and its output for each case is one line.

A test runs its cases lazily, in doubling batches, and stops at the first difference, so a failing
run pays only for the cases before it. With ORACLE_CACHE naming a directory, each case's output is
kept there, compressed, under the hash of the script, the library's files and the payload, and a later
run that sends the same payload reads it back instead of starting node; test_mutants keeps one between
runs.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import zlib
from functools import cache
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator

HERE = Path(__file__).resolve().parent
ENTRY = Path(os.environ.get('CONTEXT_MANAGER') or HERE / '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/'
                                                         'node_modules/@animalabs/context-manager/dist/src/index.js').resolve()


def cases(fixtures: list, make: Callable[[int], dict], count: int, shard: str = '0/1') -> Iterator[tuple[int, dict]]:
    """(index, case): the fixtures, then `count` generated cases, make(i) each, as shard 'k/n' takes
    every n-th from the k-th. A generated case has its own seed, so a shard generates only its own."""
    k, n = map(int, shard.split('/'))
    for i in range(k, len(fixtures) + count, n):
        yield i, fixtures[i] if i < len(fixtures) else make(i - len(fixtures))


def compare(script: Path, indexed: Iterable[tuple[int, dict]], port: Callable, check: Callable) -> str | None:
    """The first difference, or None. port(case) gives (payload, expected): the oracle's input, and what
    the port did, which may have filled in the payload; check(index, case, expected, line) gives a
    message when the oracle's line disagrees."""
    stream, size = iter(indexed), 4
    while batch := list(islice(stream, size)):
        results = [port(case) for _, case in batch]
        for (index, case), (_, expected), line in zip(batch, results, replay(script, [payload for payload, _ in results]), strict=True):
            if message := check(index, case, expected, line):
                return message
        size *= 2
    return None


def replay(script: Path, payloads: list) -> list[str]:
    """The oracle's output line for each payload: from the cache, else from one node run over the misses."""
    texts = [json.dumps(p) for p in payloads]
    keys = [hashlib.sha256(b'\0'.join([script.read_bytes(), library_hash().encode(), t.encode()])).hexdigest() for t in texts]
    folder = Path(os.environ['ORACLE_CACHE']) if os.environ.get('ORACLE_CACHE') else None
    found = {key: zlib.decompress((folder / key).read_bytes()).decode() for key in keys if folder and (folder / key).exists()}
    if missing := [(key, text) for key, text in zip(keys, texts) if key not in found]:
        for (key, _), line in zip(missing, node(script, f'[{",".join(text for _, text in missing)}]'), strict=True):
            found[key] = line
            if folder:
                partial = folder / f'{key}.{os.getpid()}'
                partial.write_bytes(zlib.compress(line.encode(), 1))
                partial.replace(folder / key)
    return [found[key] for key in keys]


@cache
def library_hash() -> str:
    """The library's identity: a hash of its built files."""
    digest = hashlib.sha256()
    for path in sorted(ENTRY.parents[1].rglob('*.js')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def node(script: Path, text: str) -> list[str]:
    """One run of the oracle on a JSON array of payloads: its output lines, split at '\\n' only
    (str.splitlines also splits at U+0085)."""
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        f.write(text)
    try:
        result = subprocess.run(['node', str(script), f.name], capture_output=True, text=True,
                                env={**os.environ, 'CONTEXT_MANAGER': str(ENTRY)})
    finally:
        os.unlink(f.name)
    if result.returncode != 0:
        raise RuntimeError(f'{script.name} failed:\n{result.stderr[-3000:]}')
    return result.stdout.split('\n')[:-1]
