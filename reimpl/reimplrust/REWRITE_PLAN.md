# reimplrust rewrite plan

Rewrite the library and the app to `../../CODE_PREFERENCES.md`, keeping every feature. The plan has two parts:
- **Part A** changes no behavior, so the existing tests prove it.
- **Part B** lists behavior changes that would delete more code. Each is a separate decision, made after Part A.

## What stays fixed in Part A

- **Session files**, record for record, nulls included, so the Python port can still carry on from them and the other way round.
- **The page** (`ui.html`), the HTTP routes and the event lines a turn streams. Routes: `/`, `/libs.js`, `/katex.css`, `/fonts/*`, `GET /api/sessions`, `GET`/`POST /api/settings`, `POST /api/sessions`, and under `/api/s/NAME`: `GET` (state), `messages`, `layer/N`, `live`, and `POST` `fork`, `settings`, `say`, `continue`, `approve`. Events: `logged`, `fold`, `spent`, `prompt`, `approve`, `answered`, `result`, `forming`, `end`, and the pieces `thinking`, `text`, `calling`, `input`.
- **Nothing holds a session or a context in memory.** Logs are read by binary search over byte offsets, or scanned forwards from any record or backwards from the end. A request's context is written out from disk, block by block, as it's sent. Part B keeps this too.
- **A session has threads only while it has work.** An idle session is just its files on disk.
- **The tests:** `python3 tests/test_rust.py`, `python3 tests/test_app.py` and `cargo test` pass after every step. Tests change only where an API they call goes away.

## Part A: the same behavior, clean code

### Library (`connectome/src`), one file per feature

| File | Holds | Replaces |
|---|---|---|
| `log.rs` | `Log<T>`: one JSONL file of typed records with ids from 0. It repairs a half-written last line, finds a record by binary search over byte offsets, scans forwards from any id or backwards from the end, and appends. A fork's log reads the parent's first `n` records. Modeled on `jsonlparse.py`. The backwards scan replaces loops like `(0..n).rev().map(read)` in costs and attempts, where each read was a separate binary search. | the file half of `store.rs` |
| `session.rs` | The records as serde structs with integer ids (`Message`, `Chunk`, `Memory`, `Compile`, `Attempt`), the `Session` holding a `Log` of each, and `fork`. | the rest of `store.rs` |
| `pricing.rs` | What a message costs: block prices (text-only and store), the density rule, JavaScript's lengths, rounding, trim and number format (prices and hashes depend on them), and calibration. | `blocks.rs` prices, `estimate.rs`, `js.rs`, `json.rs` |
| `context.rs` | A request's turns: `Turn`, blocks located on disk, the provider rules (split, collapse, tool pairing, normalize, image cap, reasoning strip), and `Reader`, which loads each block from disk as the body is written. | `turns.rs`, `blocks.rs` shapes, `emit.rs` `Reader` |
| `compaction.rs` | Top to bottom: chunks and the tail, the planner, then the compile (render, pairing, cache marks, commit). | `chunking.rs`, `planner.rs`, `emit.rs` `select` |
| `memory.rs` | The prompts, which memory is due, building its request, and recording the reply. | `memory.rs` |
| `bin/replay.rs` | The differential test's driver, now including the old `drive::step`, which nothing else uses. | `drive.rs` |

Details:
- **Message blocks stay `serde_json::Value`.** The differential test feeds odd and malformed blocks (`odd_turn`), and the JS library reads whatever is there. Records the library writes itself become typed structs. Blocks are read through a few named functions in `pricing.rs` (`text_of`, `kind_of`), not through a blanket trait.
- **Memory formation does no IO with the model.** `memory::next(&session, tools, system, cfg)` appends any quiet stubs that are due and returns the next request that needs a call, or `None`. The app sends it and passes the reply to `memory::record`. The `Host` trait goes.
- **Errors are values.** Every log read and write returns a `Result`, and the message names the file and the record. No panics on a bad log.
- **Deleted, each without a measured need:** `segment.rs` (its two helpers become loops where they're used), the `Json` trait and `obj!`, the count cache keyed on file size, the `{"id":` byte-prefix shortcut, and the `priced` cache. If a step turns out measurably slow without one of these, it comes back with that number in a comment.

### App (`app/src`)

Threads, each owning its state and talking through std channels:

A session has threads only while it has work: a turn and the memories after it, or a fork. A request about an idle session (its state, messages, layers or settings) is served from disk by the request's own thread, and no session thread starts. Today any message about a session, even the state query a page makes when it opens, starts that session's thread, and the thread never exits.

| Thread | Owns | Mailbox |
|---|---|---|
| main: the accept loop | nothing; hands each request to a new thread, with the coordinator's `Sender` | tiny_http's incoming requests |
| a thread per HTTP request | its request | – |
| coordinator | the active sessions only, each with its job queue, the running job's followers, the events since its log last grew, and calls waiting for an answer; the only writer of `settings.json` and every `session_settings.json` | `enum Msg { Page(name, PageMsg), Job(name, JobMsg) }` |
| a job thread per running job, at most one per session | that session's logs while the job runs; the turn is plain blocking code, top to bottom | – |
| MCP server (one per configured server) + its stdout reader | the child process, pending calls by id, and their deadlines | `enum McpMsg { Call, Line, Exited }` |

A session's lifecycle:
- **A job arrives for an idle session:** the coordinator adds the session and starts a job thread for it.
- **A job arrives for an active session:** the coordinator queues it.
- **A job thread finishes:** it tells the coordinator and exits. The coordinator starts the next queued job, or, if none is queued, drops the session, which ends its followers' streams.

One coordinator owns the whole set of active sessions, so starting and stopping a session can't race with a job arriving. A coordinator per session would need a shutdown handshake to avoid losing a job sent just as it exits. The coordinator never blocks, so pages get answers at once while a job thread waits on the model.

These are the flows that cross threads:
- **Approval:** the job thread sends `Ask { call, tool, answer: Sender<Decision> }` to the coordinator and waits with `recv_timeout(10 min)`. When a page approves, the coordinator writes `allow` first if the answer is "always", then sends the decision.
- **Following a turn:** the coordinator keeps each page's `Sender`, replays the events since the log last grew, then forwards new ones. A page that asks to follow an idle session gets a stream that ends at once.
- **MCP call:** the job thread sends `Call { method, params, deadline, reply }` and blocks on `reply.recv()`. The server thread waits with `recv_timeout` until the nearest deadline. At a deadline it answers with an error and sends `notifications/cancelled`. After the child exits, the next call restarts it.
- **Bash:** the job thread spawns the command in its own process group, writing output to a file, and polls `try_wait` every 50 ms until it exits or reaches its timeout. At the timeout it kills the group. This replaces wait-timeout.
- **Fork:** a job on the parent's queue, so it runs after the parent's current turn. The fork is a new session with no other writer yet, so the job thread creates all its files, including its copy of the parent's settings.

Files:

| File | Holds | Replaces |
|---|---|---|
| `main.rs` | flags into a `&'static Config`, MCP servers started and their tools listed, the coordinator started, then the accept loop | `main.rs` |
| `server.rs` | routes on tiny_http: the Host and `x-connectome` checks, the 32 MiB body limit, JSON and NDJSON responses, the state and listing views read from disk | `api.rs` |
| `sessions.rs` | the coordinator: active sessions, their queues, followers and waiting calls, and the settings files | `hub.rs` |
| `turn.rs` | a job thread: one turn top to bottom (compile, request, stream, approvals, tools, append, memories, costs), or a fork | `work.rs` |
| `llm.rs` | the request body written from the context through a pipe while ureq sends it, and the server-sent events folded into a reply | `llm.rs` |
| `tools.rs`, `bash.rs`, `edit.rs`, `mcp.rs` | the tools a session declares, as one list where a later tool of the same name replaces an earlier one | `tools/` |
| `settings.rs`, `costs.rs` | as now, typed; costs read through `Log<Cost>` | same |

Also deleted: the `Toolbox` monoid and its tests, and `catch_unwind` (a job that fails ends with its error message). `mallopt` stays: `test_app.py` checks that the app's peak memory doesn't grow with the contexts it streams, and that measurement is what `mallopt` is for.

Dependencies: add tiny_http (measured: +158 KB, 4 crates). Drop wait-timeout. Keep ureq, serde, serde_json, include_dir, open and libc. Keep itertools and smart-default only if the rewritten code still uses them.

### Order of work

Each step ends with all three test suites passing, measured run times, and a commit.

1. `log.rs` and `session.rs`; port every library caller to them.
2. `pricing.rs`, `context.rs`, `compaction.rs`, `memory.rs`, with memory formation no longer doing IO; move `step` into `replay.rs`; delete the old modules.
3. `mcp.rs`: the server thread.
4. The coordinator and job threads, so that only sessions with work have threads; `sessions.rs` replaces `hub.rs`. Add a test that opening an idle session's page starts no thread.
5. `server.rs` on tiny_http.
6. `turn.rs`, `llm.rs`, tools, settings and costs, tidied to the doc.
7. README: what the app is, how to run it, settings, tools, tests. Short.
8. New mutant lists for the new files, run with `tests/mutants.py` (under a minute per batch). A surviving mutant gets a test, or is reported as a simplification if it's equivalent.

At the end, compare code lines per file, binary size and test times before and after.

**My estimate:** Part A removes perhaps 15–25% of the lines, and more of the concepts. It doesn't get near 300 lines. That takes Part B.

## Part B: behavior changes to decide

Each of these deletes code but gives up exact agreement with reimplclean2, so `test_rust.py` would then test behavior directly instead of comparing outputs. They're independent of each other.

1. **Store the log in the Messages API's format** (`role`, `tool_use_id`, `media_type`). Deletes `to_api` and the participant names. Existing sessions are converted once.
2. **Keep the log valid as it's written.** Before appending anything after an interrupted turn, the app logs an "interrupted" result for each unanswered call. Deletes split, collapse (the body writer already joins same-role turns), tool pairing and its lookahead, the trailing-call drop, and the "newest turn retained" refusal.
3. **Describe a context as parts:** memory nodes, then one range of messages (raw content is always a suffix, since levels fall toward the present), then the image cutoff. It still streams from disk. Deletes per-block locations and block shapes. Memory requests use the same parts.
4. **One price:** the tokens of what is actually sent, times the calibration multiplier. Deletes the separate text-only price and the JavaScript string rules.
5. **Cache breakpoints at part edges** (after the memories, and at the end) for both kinds of request. Deletes the cache keys, digests and seams.
6. **One image cutoff:** newest first, stopping at the first image that doesn't fit.
7. **One way to handle "over budget":** form the due memory, then compile again.

With all of Part B, I'd expect the library at roughly half its Part A size.

## Decisions for you

- **Errors:** I'd use `Result<T, String>` with messages that name the file and record, as the app does now. The alternative is anyhow, which adds `.context(...)`.
- **Part B:** which items, after Part A lands.
