"""Mutation testing: each mutant, an exact edit of a source file, is built and tested in a copy of the
project of its own, many at once. A mutant is CAUGHT if the test fails, MISSED if it passes (the tests
can't tell it from the code: a gap in them, or a simplification), TIMEOUT if the test runs past its
time, and NOBUILD if it doesn't compile. Mutants are JSON, {label: [text, replacement]}, per file:

    python3 tests/mutants.py tests/test_app.py app/src/main.rs=main.json [app/src/bash.rs=bash.json ...]

Mutants build as the release build does but without link-time optimization (FAST), the same code in
a fifth of the time. A copy has the project's sources, a copy of a build of them kept in
target/mutants (so only what a mutant changes builds again), and ../reimplclean2 beside it, which
test_rust.py runs. A test passes when its last line says so: 'chat turns and' (test_app.py) or 'the
same as' (test_rust.py).
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 240
FAST = {**os.environ, 'CARGO_PROFILE_RELEASE_LTO': 'false', 'CARGO_PROFILE_RELEASE_CODEGEN_UNITS': '16', 'CARGO_PROFILE_RELEASE_INCREMENTAL': 'true'}


def copy(at: Path) -> Path:
    tree = at / ROOT.name
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns('target', 'sessions', '__pycache__'))
    (tree / 'target').mkdir()
    subprocess.run(['cp', '-a', str(ROOT / 'target' / 'mutants' / 'release'), str(tree / 'target')], check=True)
    (at / 'reimplclean2').symlink_to(ROOT.parent / 'reimplclean2')
    return tree


def run(tree: Path, test: str, file: str, text: str, replacement: str) -> tuple[str, str]:
    source, started = tree / file, time.monotonic()
    original = source.read_text()
    source.write_text(original.replace(text, replacement))
    try:
        p = subprocess.Popen([sys.executable, test], cwd=tree, env=FAST, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out = p.communicate(timeout=TIMEOUT)[0]
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 'TIMEOUT', f'past {TIMEOUT} s'
        last = f'({time.monotonic() - started:.0f} s) ' + (out.strip().splitlines() or [''])[-1]
        if 'could not compile' in out:
            return 'NOBUILD', next((line for line in out.splitlines() if line.startswith('error')), last)
        return ('MISSED' if 'chat turns and' in last or 'the same as' in last else 'CAUGHT'), last
    finally:
        source.write_text(original)


def main(test: str, *specs: str) -> int:
    mutants = []
    for spec in specs:
        file, path = spec.split('=')
        source = (ROOT / file).read_text()
        for label, (text, replacement) in json.loads(Path(path).read_text()).items():
            if source.count(text) != 1:
                print(f'{label}: its text is in {file} {source.count(text)} times, not once')
                return 1
            mutants.append((file, label, text, replacement))
    workers, started = min(len(mutants), os.cpu_count() or 4), time.monotonic()
    subprocess.run(['cargo', 'build', '--release', '--quiet'], cwd=ROOT, env={**FAST, 'CARGO_TARGET_DIR': str(ROOT / 'target' / 'mutants')}, check=True)
    with tempfile.TemporaryDirectory() as scratch, ThreadPoolExecutor(workers) as pool:
        trees = queue.Queue()
        for tree in pool.map(lambda i: copy(Path(scratch, str(i))), range(workers)):
            trees.put(tree)

        def one(mutant):
            tree = trees.get()
            try:
                return mutant, run(tree, test, mutant[0], *mutant[2:])
            finally:
                trees.put(tree)
        verdicts = []
        for done in as_completed([pool.submit(one, m) for m in mutants]):
            (file, label, *_), (verdict, line) = done.result()
            verdicts.append(verdict)
            print(f'{verdict:8} {file}: {label}: {line[:150]}', flush=True)
    print(f'{len(mutants)} mutants in {time.monotonic() - started:.0f} s: ' + ', '.join(f'{verdicts.count(v)} {v}' for v in dict.fromkeys(verdicts)))
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
