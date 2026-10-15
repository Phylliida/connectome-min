"""The clean version's tests, run in parallel: chunk formation against the library (test_chunks), the
counter's guarantees (test_counter), and whole sessions on disk (test_session).

    python3 tests/suite.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ['test_chunks.py', 'test_counter.py', 'test_session.py']


def main():
    start = time.monotonic()
    runs = {test: subprocess.Popen([sys.executable, str(HERE / test)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for test in TESTS}
    failed = 0
    for test, run in runs.items():
        out = run.communicate()[0].strip().splitlines()
        failed += run.returncode != 0
        print(f'{"ok    " if run.returncode == 0 else "FAILED"}  {test:16} {out[-1] if out else ""}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed in {time.monotonic() - start:.0f} s')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
