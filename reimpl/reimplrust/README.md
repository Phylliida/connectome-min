# connectome (Rust)

A port of `../reimplclean2` (a clean reimplementation of `@animalabs/context-manager` 0.10.1's
autobiographical compaction) and a web app around it.

- `connectome/`: the library, over serde_json (with smart-default and itertools). A session is append-only JSONL files on disk
  (`store.rs`), in the same format the Python port writes, so either can carry on from the other's
  files. A compile plans the context from prices and block shapes only (`emit::select`), and an
  `emit::Reader` reads each block from disk as the caller writes the request, so neither a session nor
  a context is held in memory.
- `connectome/src/bin/replay.rs`: replays a scripted session, for the tests.
- `app/`: `connectome-web`, a chat with a model whose context the library manages, in one web page
  that shows the memory pyramid and what the context shows, and forks a session at any message. It
  speaks the Anthropic Messages API (Anthropic, DeepSeek at `https://api.deepseek.com/anthropic`,
  OpenRouter at `https://openrouter.ai/api/v1`), each request body streaming in chunks straight from
  the session's files, over ureq with TLS compiled in (rustls, ring, and Mozilla's root certificates),
  so the binary needs nothing from the system but sockets. The page shows the assistant's text as
  markdown, through marked and then DOMPurify, which keeps only the HTML markdown makes, so a reply can
  neither run code on the page nor load anything; both are in `app/vendor/` as published, built in.

`.github/workflows/connectome-web.yml` builds standalone binaries on GitHub's runners: Linux (static,
x86_64 and arm64), Windows (x86_64) and macOS (arm64 and x86_64). Locally, `cargo build --release`.

```sh
./target/release/connectome-web        # opens http://127.0.0.1:8080; add a provider's key in the page
```

Settings are JSON files you can edit by hand, read again at each use, and only you can read them,
since one holds the keys:

- `ROOT/settings.json`: `providers`, each a name with a `base` URL and a `key`; `defaults`, what a
  new session starts with; and `sandbox`, what a sandboxed command sees. The first run writes it with
  a provider named `default` for `https://api.anthropic.com` without a key, which the page then asks
  for, and defaults for a window of 60k tokens.
- `ROOT/NAME/session_settings.json`: the session's `provider` (by name) and `model`, its
  `workspace`, whether bash runs in the `sandbox`, the tools it always allows (`allow`), its
  `window`, `reserve` and `system` prompt, and the library's settings (`config`). A new session's
  file is a copy of `defaults` (its workspace given on the page), and a fork's a copy of its parent's;
  a session without one has none to run with, and the page says so.
  The page edits the main fields; `config.merge_threshold` can't change once memories have merged.

Tools: the model can call `bash` (built in: a command in a new shell in the session's workspace, or
`cmd` on Windows; `--no-bash` leaves it out) and the tools
of the MCP servers in `ROOT/mcp.json` (or `--mcp FILE`), listed all at once at startup (one whose tools
can't be listed is stopped), in the usual format:

```json
{"mcpServers": {"NAME": {"command": "...", "args": ["..."], "env": {"KEY": "..."}}}}
```

A session without a workspace still declares bash, but each call fails at once, without asking, with
a note to set one: so a session whose history calls bash still declares it, as the Messages API
requires, and the model's calls come back as calls (without the declaration, some providers return
the model's own tool-call markup as text).

In a session's sandbox (Linux's bubblewrap, `bwrap`), a command sees its workspace, read-write, and
of everything else only the paths `settings.json`'s `sandbox` lists that exist: `ro` read-only, `rw`
read-write (a leading `~/` is the home directory); it has the network if `network`, and a PID
namespace of its own, so a timeout still kills everything it started.

Each call asks first: the page shows it with Allow, Always allow and Deny, and a denied call (or one
left unanswered for ten minutes) gets an error result the model sees. Always allowed tools are kept in
the session's settings; the page's tool list toggles them. A call has `--tool-timeout` seconds (600 by
default): bash kills a command still running then, with what it started, and an MCP call that runs out
is cancelled without holding up the server's next call; a server that stops starts again at its next
call. A turn takes steps, a model request each, until a reply calls no tools (at most 64), and the page
shows each call as a card on its reply, which its result then joins. A turn's log is the session's: a
page following a turn shows the log as it grows, read from disk, and besides it only what the log
doesn't hold yet (a reply streaming in, the questions whether its calls may run, their results,
memories forming), which is all the app keeps of a turn; its stream ends once the turn's memories
have formed. A page opened while
a turn runs (after a reload, or on another device) follows it the same way, and answers its questions;
the turn runs on whether any page follows it or not, on a thread of its session's, and a message sent
while it runs waits for it. A turn on a log edited into something the library can't read fails, and
the session goes on once the log is mended. The page's Continue button runs a turn without a message: after an
interruption (a failed request, the 64 steps, a restart) the model goes on from the log as it is, and
after a reply cut short it goes on from that reply (the request ends with it, as a prefill). Tool sources compose as a monoid: a later source's tool shadows an earlier
one of the same name. Every request carries the system prompt and the tool definitions, so their
tokens come out of the window the context compiles to, and out of the usage that calibrates prices;
while calibration catches up with a request bigger than its estimate, the next context compiles that
much smaller. Memory requests declare the same tools with tool choice "none", as the library does, so
a memory can cover tool calls.

Flags: `--root DIR` (sessions and settings; `sessions` by default), `--addr` (`127.0.0.1:8080`),
`--ca PEM` (root certificates to trust instead of Mozilla's: a company proxy's, a test server's),
`--mcp FILE`, `--tool-timeout SECONDS`, `--no-bash`, and `--no-open` (else the page opens in the
browser). The server answers only its own page: requests must name it as their Host, and POSTs must
carry its `x-connectome` header. A request's body takes 32 MiB at most.

Tests:

```sh
cargo test                  # unit tests of the JavaScript and Python arithmetic the ports share
python3 tests/test_rust.py  # every compile, request and file record the same as reimplclean2's, handoffs both ways, the planner
python3 tests/test_app.py   # the app against mock APIs over HTTPS, its parts at once: chat, memories, forks, settings, certificates, bash, its sandbox, approvals, timeouts and tools through fake MCP servers, flat memory
python3 tests/mutants.py tests/test_app.py app/src/main.rs=mutants.json   # mutation testing, each mutant in a copy of its own, many at once
```

The three take under 30 seconds together. A batch of mutants takes about as long as its slowest one
(68 of the app's, under a minute): they build without link-time optimization, from a build kept in
`target/mutants`.
