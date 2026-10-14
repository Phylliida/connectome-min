"""Runs an oracle script, which drives the real library under node, on a JSON payload.

With ORACLE_CACHE naming a directory, each output is kept there under the hash of the script, the
library entry (CONTEXT_MANAGER) and the payload, and replayed for the same three: the first process
to need an output runs node while the others wait for it. test_mutants gives each suite run a fresh
directory, so mutants that leave a test's payload unchanged share one node run; a directory kept
across runs would replay stale output once the library changes.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def replay(script: Path, payload) -> list[str]:
    """The oracle's output lines, split at '\\n' only: str.splitlines also splits at U+0085."""
    text = json.dumps(payload)
    if not os.environ.get('ORACLE_CACHE'):
        return node(script, text).split('\n')[:-1]
    key = hashlib.sha256(b'\0'.join([script.read_bytes(), os.environ.get('CONTEXT_MANAGER', '').encode(), text.encode()]))
    cached = Path(os.environ['ORACLE_CACHE'], key.hexdigest())
    with open(cached.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cached.exists():
            cached.write_text(node(script, text))
    return cached.read_text().split('\n')[:-1]


def node(script: Path, text: str) -> str:
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        f.write(text)
    try:
        result = subprocess.run(['node', str(script), f.name], capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    if result.returncode != 0:
        raise RuntimeError(f'{script.name} failed:\n{result.stderr[-3000:]}')
    return result.stdout
