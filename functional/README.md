# functional/

A purely functional reimplementation of the whole `connectome-min` artifact, file for file.
Everything the original does, this does -- verified, not asserted: the policy's three documented
runs, the demo's printed story, and the config report are **byte-identical** to the original's
output (`diff` against fresh captures of each), the legacy solver's frontier maps are identical
across 500 fuzzed archives, and the ported test suite runs green: `44 passed` under pytest,
`all checks passed` under its own runner, with the runner output itself byte-identical to the
original's.

## The rule, and how it landed

The assignment: reimplement everything in a functional style; any function that genuinely must
mutate state carries a `_` prefix. What the port found is that the original's mutations all had a
pure form one threading-step away, so **no function in the folder needs the `_` prefix at all** --
the surviving underscores (`_count`, `_no_network`) mark *private helpers*, matching the original's
own `_count` convention, not mutation. The three sites that once looked irreducible all yielded:

- the mock's queues and the HTTP client's config ride the `Model`'s `state` field, so tests assert
  on the value instead of a recording closure;
- the fake transports are pure functions -- what a test would have recorded (the dispatched
  request, the dispatch count) it reads off the Summarizer value (`last_request`, `usage()`);
- the check harness is `recorder()`: each test gets a failures list it returns, so there is no
  module-level mutable state in the test file either.

Local scratch that never escapes a call -- the chunker's pending buffer, the DP tables inside
`tiling.cut`, the recorder's per-test list, building a fresh list and rebinding it -- is written
with plain loops. Rebinding a fresh structure is not the mutation the convention forbids.

## What changed, module by module

Every function keeps the original's name and semantics; only the state plumbing moves:

| original | here |
|---|---|
| the archive dict, mutated by `mint`/`add_chunk` | frozen `Archive` / `Chunk` / `Summary` dataclasses; every write is `{**old, ...}` and the functions return the next `Archive` |
| `emit(log, ...)` appends in place | the log is an immutable tuple; `log = emit(log, ...)` |
| `commit(log, v, ...)` mutates or returns None | `log, committed = commit(log, v, ...)` -- the CAS compares the input log's length, as before |
| `step(log, item, engine, v)` returns an outcome | `log, engine, outcome = step(log, item, engine, v)` |
| `class MockModel` with queues popped in place | `mock_model(plan)`: a `Model(kind, name, complete, state)` Mealy machine; the remaining queue is bound into the returned model and exposed as `state` |
| `class HttpModel` holding config | `http_model(...)` resolves env into a frozen `HttpConfig` carried as `state`; stateless, so its `complete` returns the same machine |
| `class Summarizer` with per-call field writes | frozen `Summarizer` dataclass; `complete(s, ...) -> ((text, reason), s')` threads counters, calibration, `last_request`/`last_reply` via `dataclasses.replace` |
| `s.request(...)`, `s.estimate(...)`, `s.usage()` | module functions `request(s, ...) -> (req, s')`, `estimate(s, req)`, `usage(s) -> dict` |
| `minisystem.Refused` carrying plan and demand | also carries the `(log, model)` the run had reached -- an exception is the one way a value cannot ride a return |
| `legacy/relevance_cut.py`'s passes mutate the frontier dict | each pass returns the next frontier; the original's roll-back write becomes simply not assigning |

The projections (`messages`, `spans`, `archive`, `ledger`, `work`, ...) were already folds over
the log and keep their names and shapes; several are now stated as literal `reduce` calls.

`config.py` and `llm/prompts/` are byte-identical copies -- they were pure data already.

## Layout

```
config.py             every knob -- copied unchanged
connectome_min.py     the POLICY -- planner, renderer, cascade
minisystem.py         the SYSTEM -- log, derived work, guards, failure policy
llm/summarizer.py     the BOUNDARY -- request assembly, admission, mock, HTTP client, gate
llm/prompts/          the six prompt files -- copied unchanged
legacy/relevance_cut.py   the superseded solver, kept for comparison
demo.py               the HOST -- session loop and printed story, byte-identical output
test_summarizer.py    the tests -- same 44 tests, 379 checks: python3 test_summarizer.py
```

## Verifying it yourself

From the parent directory, capture the original; from here, capture the port; diff:

```
cd .. && python3 demo.py > /tmp/orig.txt && cd functional && python3 demo.py | diff /tmp/orig.txt -
python3 test_summarizer.py          # or: python3 -m pytest test_summarizer.py -q
```

The same recipe holds for `connectome_min.py` (plain, `--reach 1250`, `--window 700`) and
`config.py` (plain and `--gaps`).
