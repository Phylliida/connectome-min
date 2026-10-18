# connectome (Rust)

A port of `../reimplclean2` (a clean reimplementation of `@animalabs/context-manager` 0.10.1's
autobiographical compaction) and a web app around it.

- `connectome/`: the library, whose one dependency is serde_json. A session is append-only JSONL files on disk
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
  so the binary needs nothing from the system but sockets.

`.github/workflows/connectome-web.yml` builds standalone binaries on GitHub's runners: Linux (static,
x86_64 and arm64), Windows (x86_64) and macOS (arm64 and x86_64). Locally, `cargo build --release`.

```sh
./target/release/connectome-web        # opens http://127.0.0.1:8080; set the provider, model and key in the page
```

Tools: the model can call `bash` (built in: a command in a new shell in `--workdir`, or `cmd` on
Windows; `--no-bash` leaves it out) and the tools of the MCP servers in `ROOT/mcp.json` (or `--mcp FILE`),
in the usual format:

```json
{"mcpServers": {"NAME": {"command": "...", "args": ["..."], "env": {"KEY": "..."}}}}
```

Each call asks first: the page shows it with Allow, Always allow and Deny, and a denied call (or one
left unanswered for ten minutes) gets an error result the model sees. Always allowed tools are kept in
`settings.json`; the page's tool list toggles them. A call has `--tool-timeout` seconds (600 by default):
bash kills a command still running then, with what it started, and an MCP call that runs out is cancelled
without holding up the server's next call; a server that stops starts again at its next call. A turn takes
steps, a model request each, until a reply calls no tools (at most 64), and the page shows each call as it
runs. Tool sources compose as a monoid: a later source's tool shadows an earlier one of the same name.
Every request carries the system prompt and the tool definitions, so their tokens come out of the window
the context compiles to, and out of the usage that calibrates prices. Memory requests declare the same
tools with tool choice "none", as the library does, so a memory can cover tool calls.

Flags: `--root DIR` (sessions, and `settings.json`, which holds the key), `--addr`, `--window TOKENS`
(the context budget), `--reserve TOKENS` (the reply's), `--system TEXT`, `--ca PEM` (root certificates
to trust instead of Mozilla's), `--mcp FILE`, `--workdir DIR`, `--tool-timeout SECONDS`, `--no-bash`, `--no-open`; `--base`, `--model` and `API_KEY` give the settings until the
page saves its own. The server answers only its own page: requests must name it as their Host, and
POSTs must carry its `x-connectome` header.

Tests:

```sh
cargo test                  # unit tests of the JavaScript and Python arithmetic the ports share
python3 tests/test_rust.py  # every compile, request and file record the same as reimplclean2's, handoffs both ways, the planner
python3 tests/test_app.py   # the app against a mock API over HTTPS: chat, memories, forks, settings, certificates, bash, approvals, timeouts and tools through fake MCP servers, flat memory
```
