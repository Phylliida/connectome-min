# connectome (Rust)

A port of `../reimplclean2` (a clean reimplementation of `@animalabs/context-manager` 0.10.1's
autobiographical compaction) and a web app around it.

- `connectome/`: the library, over serde_json (with smart-default and itertools). A session is
  append-only JSONL files on disk (`store.rs`), in the same format the Python port writes, so either
  can carry on from the other's files while memories merge by count. One thing differs from the
  port: by default a merge takes the fewest memories of the layer below, at least `min_merge_memories`
  (3 by default; 2 when a compile can't fold far enough without), that hold `target_chunk_tokens`, as chunks close, so a merge's request stays about the same size however
  long memories are, and its record says what it merged (`merges`). And where the library only
  merges within a layer, a compile that can't fold enough lifts: each layer keeps memories no merge
  has taken yet, short of a group, and with enough layers those and the tail outgrow the window with
  nothing left to fold, so the oldest of them past the top layer's, across layers, the fewest (again at
  least `min_merge_memories`) that hold `target_chunk_tokens`, merge into a memory one layer above the highest it takes (`memory.rs`,
  `lifted`), even past a memory that keeps failing, which would hold back any other. `merge_by_count` merges `merge_threshold` at a time and never lifts, as the port does
  (the differential test runs that way); a memory without `merges` is read as such a group, so a
  session that merged by count goes on by tokens from where it was. A compile plans the context from prices and block shapes only (`emit::select`), and
  an `emit::Reader` reads each block from disk as the caller writes the request, so neither a
  session nor a context is held in memory.
- `connectome/src/bin/replay.rs`: replays a scripted session, for the tests.
- `app/`: `connectome-web`, a chat with a model whose context the library manages, in one web page
  that shows the memory pyramid and what the context shows, and forks a session at any message. It
  speaks the Anthropic Messages API (Anthropic, DeepSeek at `https://api.deepseek.com/anthropic`,
  OpenRouter at `https://openrouter.ai/api/v1`), each request body streaming in chunks straight from
  the session's files, over ureq with TLS compiled in (rustls, ring, and Mozilla's root
  certificates), so the binary needs nothing from the system but sockets. Its threads share nothing
  that changes: each request has one (`api.rs`), the hub owns the settings files and what the pages
  following a session hear (`hub.rs`), and each session has one, which runs its turns and alone
  writes its log (`work.rs`).

The page draws itself with Preact, its templates written with htm, as a function of what it shows:
the session as the server has it, its log's messages, and the turn it follows, which only the page's
actions change. A turn it follows goes through phases the line under the chat shows (waiting for the
model, streaming, asking whether a call may run, running a call, which its card says too, its
request tried again, and why, forming memories); its reply streams in below the log and is drawn at
most once a frame, its finished markdown blocks once. Opening a session shows its log's newest 200
messages; scrolling up, two views from the top, loads the 200 before them, until the first, so
they're there before the view gets to them, and the messages in view stay where they are as they go
in above, even mid-scroll (the browser's scroll anchoring). The log follows a reply only while the
view is at its end. A tab on each edge of the chat hides the side next to it, or shows it again, as
this browser remembers. The page shows the assistant's text as markdown, through marked and then
DOMPurify, which keeps only the HTML markdown makes, so a reply can neither run code on the page nor
load anything; its math (`$…$` or `\(…\)` inline, `$$…$$` or `\[…\]` displayed) KaTeX sets in its
own fonts, and highlight.js colors its code blocks. These, jsdiff for edits' diffs, Preact and htm
are in `app/vendor/` as published, built in.

The page also shows what a session's requests have used, from the session's `costs.jsonl`
(`costs.rs`): a record for each step's reply and each round of memories, `{id, for, used, total}`.
`used` is `{cost, input, cached, output}`: dollars as the provider reports them (OpenRouter does, in
each reply's usage; null while none has), the prompts' tokens, of them those read from the cache,
and the replies'. `total` is the running totals by what they were for, `{reply, memories}`, each the
same. A reply's record also says which `message` it is and which `compile` its request was built
from: its prompt, as the provider counted it, is the context the page shows (the compile's estimate
before there is one), and calibrates the compile after it. The page shows the cost of replies, of
memories and in all, from the last record. A fork's costs count only its own requests. Where the
context folded before a reply (its compile shows other chunks and memories than the reply's before
it did), the page draws a line before it: what it shows instead of what, and the tokens that saves,
priced as the planner prices them.

`.github/workflows/connectome-web.yml` builds standalone binaries on GitHub's runners: Linux
(static, x86_64 and arm64), Windows (x86_64) and macOS (arm64 and x86_64). Locally, `cargo build
--release`.

```sh
./target/release/connectome-web        # opens http://127.0.0.1:8080; add a provider's key in its Settings
```

Settings are JSON files you can edit by hand, read again at each use, and only you can read them,
since one holds the keys:

- `ROOT/settings.json`: `providers`, each a name with a `base` URL, a `key` and the `models` the page
  offers a session on it (the page's Settings edits them, adds providers and deletes them, all but
  the one new sessions start on); `defaults`, what a
  new session starts with; `sandbox`, what a sandboxed command sees; and `search`, a SearXNG
  server's address (say `http://10.0.0.139:8288`), which gives sessions the `web_search` tool. The
  first run writes it with a provider named `default` for `https://api.anthropic.com` without a key,
  which the page then asks for, listing `claude-sonnet-5-5`, and defaults for a window of 180k tokens,
  30k of them reserved for the reply (and its reasoning: a reply that reaches its reserve stops
  there), a prompt that folds past 90k tokens down to 65k (`fold_above`, `fold_to`), a raw tail of
  20k, chunks of 3k, merges of 3 memories at least, memories of up to 20k tokens
  (`config.compression_max_tokens`), and,
  on Linux, bash in the sandbox.
- `ROOT/NAME/session_settings.json`: the session's `provider` (by name) and `model`, its
  `workspace`, whether bash runs in the `sandbox`, the tools it always allows (`allow`), its
  `window`, `reserve`, the band its prompt folds in (`fold_above` and `fold_to`, always numbers: past
  `fold_above` tokens the context folds, down to `fold_to`; the band the window gives is what it leaves
  the reply, down to `config.slack`, 10%, under that), its `system` prompt, and the library's settings (`config`). A new session's
  file is a copy of `defaults` (its workspace given on the page), and a fork's a copy of its
  parent's; a session without one has none to run with, and the page says so. The page edits the
  main fields, the chunk size (`config.target_chunk_tokens`: the tokens of text a chunk closes at,
  and that a merge gathers memories to), the least memories a merge takes (`config.min_merge_memories`, 2 or more, though a merge takes 2 when the context can't fold down to `fold_to` otherwise; it can change at any time, since each merge records what it took), the raw tail (`config.recent_window_tokens`: the newest tokens, never folded) and a
  memory's max tokens (`config.compression_max_tokens`, its reasoning included: unlike the library,
  where it only lowers the 16000 a memory gets by default, it's the max_tokens a memory asks for), each saved alone as it changes (a choice at once, typing once it pauses for 2.5 s or
  leaves the field, a number only once it is one), its model one its provider lists (or any, typed,
  for a provider that lists none); `config.merge_threshold` can't change once memories have merged,
  nor `config.merge_by_count` turn on; `fold_above` and `fold_to` can't be blank, `fold_to` stays at
  most `fold_above`, and that at most what the window leaves the reply (so a smaller window needs a
  lower band first); a chunk holds more than 0 tokens; and a merge takes 2 memories or more.

Tools (`app/src/tools/`): the model can call `bash` (a command in a new shell in the session's
workspace, or `cmd` on Windows; `--no-bash` leaves it out), `edit` (a file in the workspace changed
by replacing text in it, which must occur once unless the call says `all`, or made; its paths can't
lead out of the workspace, through `..`, an absolute path or a link; in the sandbox it reads and
writes through commands run in bash's sandbox, so the kernel keeps it to what a sandboxed command
may touch), `web_search` (when settings.json names a SearXNG server: a query asked through its JSON
API, which its settings.yml must allow by listing `json` in `search.formats`, the pages found given
back as text, each one's title, address, date when known and excerpt) and the tools of the MCP
servers in `ROOT/mcp.json` (or `--mcp FILE`), listed all at once at startup (one whose tools can't
be listed is stopped), in the usual format:

```json
{"mcpServers": {"NAME": {"command": "...", "args": ["..."], "env": {"KEY": "..."}}}}
```

A session without a workspace still declares bash and edit, but each call fails at once, without
asking, with a note to set one: so a session whose history calls them still declares them, as the
Messages API requires, and the model's calls come back as calls (without the declaration, some
providers return the model's own tool-call markup as text).

In a session's sandbox (Linux's bubblewrap, `bwrap`), a command sees its workspace, read-write, and
of everything else only the paths `settings.json`'s `sandbox` lists that exist: `ro` read-only, `rw`
read-write (a leading `~/` is the home directory); a write anywhere else fails (`Read-only file
system`), so a tool that keeps a cache in the home directory needs that directory in `rw`. It has
the network if `network`, and a PID namespace of its own, so a timeout still kills everything it
started.

Each call asks first: the page shows it with Allow, Always allow and Deny, and a denied call (or one
left unanswered for ten minutes) gets an error result the model sees. Always allowed tools are kept
in the session's settings; the page's tool list toggles them. A call has `--tool-timeout` seconds
(600 by default): bash kills a command still running then, with what it started, and an MCP call
that runs out is cancelled without holding up the server's next call; a server that stops starts
again at its next call. Tool sources compose as a monoid: a later source's tool shadows an earlier
one of the same name.

A session can have a companion (`app/src/companion.rs`): another session, made from the page's Auto
switch with the user's goal, that talks with it as a collaborator. It starts on the session's
provider, model, window and library settings, without a workspace, its system prompt (which the
user can edit) telling it the goal and its part. Their settings name each other (`companion`, and
`accompanies` on the companion), and `auto` says whether Auto is on, which the switch sets for both.
Each has the `message` tool, which tells the other something at once, or with `wait` waits for the
answer (up to `--tool-timeout`); the companion also has `finish`, which turns Auto off when the goal
is done or the work needs the user. A turn's end goes to the other, which takes a turn on it: what
the turn last said, and what it ran (bash's commands, edit's paths, web_search's queries, other
tools' names, failed calls marked). It goes while Auto is on; when the other messaged the session
during the turn, so every message gets its answer; and after `finish`, so the session hears the
companion's last word. A turn that fails or is stopped turns Auto off for both. What a session hears
from the other joins its log as the other's message, its participant the other's name, headed by
how it came: the other ended its turn, sends a message while it works, or asks and waits for the
answer. A turn hears it between its steps, and goes on while there's more; an idle session takes a
turn on it. Turning Auto on sends the session's last turn to the companion, unless a turn runs in
either. A fork has no companion of its own.

Channels (`app/src/channels.rs`) are logs of posts that sessions take part in, made from the page's
Channels list on the left, each shown with its posts and a box that posts to it as the user. A
session's settings panel adds it to a channel (its choice of channels, and +) and sets whether it
listens there and the most posts a day it may make there, kept in its settings' `channels`
(`{"NAME": {"inputs": true, "per_day": 5}}`; `per_day` 0 or less, or left out, is no limit). Past
its limit (counted over the last 24 hours), `message_channel` is refused, "Rate limit exceeded",
saying when the next post can go, and under one each post's result says how many it has left; the
tool's description lists each channel's limit, which changes only when you change it. A session in a
channel has `message_channel`, which posts there; a post, the user's or a session's, goes at once to
every session in the channel with `inputs` on but the one that posted it, as a message (its
participant `#NAME`) headed by who posted it ("onion posted in #NAME:", or "The user posted…") and
ending with a reminder of how to reply there, then the channel's description ("About #NAME: …"), if
you gave it one in the box above its members (say, to post sparingly when an expensive model
listens), the way a companion's messages come: between the steps
of a turn running, or in a turn of its own. The call's result names the sessions that heard it.
What a session says otherwise stays in its session, so it stops posting when it chooses to; only the
user sets who's in a channel and who listens. A channel is `ROOT/.channels/NAME/`: `posts.jsonl`,
each post `{id, from, text, at}` (`from` null for the user's, `at` when, in seconds since 1970), and `channel.json`, its `description`;
the hub alone writes both. A fork, and a new companion, are in no channel. Sessions can't yet read a channel's history
beyond what they hear.

Boards (`app/src/boards.rs`) are kanban boards kept as files, which sessions work on with the tools
they have. A board is `ROOT/.boards/NAME/`: `board.json` (its columns' order and its description),
and a folder per column, `done/` always among them and last, each holding a `.md` file per task:

```markdown
---
title: Document the merge minimum
owner: onion
---
What it's for…

## Comments
- **onion**: started on this
```

The page's Boards list on the left makes boards (with todo, doing and done) and shows one as columns
of cards, read again every 3 s, where you add and take out columns (an empty one, not done), reorder
them by dragging a column's header (done stays last), collapse one to its header (kept in your
browser), make tasks, move one by dragging its card to another column, and open one to change its
title, column, owner and description, comment, or delete it. A
session's settings panel puts it on a board (`boards` in its settings), which takes a workspace and
not being a companion: the board's folder is then bound into its sandbox, read-write, besides what
`settings.json`'s `sandbox` gives it, its `edit` tool reaches it, and its system prompt says where the
board is and how it's laid out, so it makes, moves (`mv`) and comments on tasks as files. A task's
owner, if it's a session on the board, hears as a message (participant `board:NAME`) when the task is
handed to it and when someone else comments on it, the message ending with a sentence on how boards
work and then the board's description. The hub looks at a board again after each step of a session
on it that ran tools, and after each change from the page, and puts what changed down to that
session, or to you, so a session that takes a task itself isn't told; a change made by hand outside
the page is put down to whoever looks next. A fork, and a new companion, are on no board.

A turn takes steps, a model request each, as many as it takes until a reply calls no tools (or the
user stops it). After a step that called tools, and after the turn, the next memory due forms. Below
`fold_above` the context doesn't fold; past it, it folds down to `fold_to`, and waits for the memories
that takes: they form, as many as it takes, until it gets there, or none can form, or memories couldn't
make the room (the tail and the messages no chunk holds yet are over `fold_to` alone). Then it goes as
it is if the window takes it, and a context the window can't take says what it holds, and why a
memory it waits for doesn't form (its call failed, the summarizer's reply ran out of tokens or had no
memory in it, or it failed `config.attempt_limit` times since the log last grew). A request that fails in a way that may pass (no answer, a stream cut off, an HTTP
408, 409, 425, 429 or 5xx, an error event of overload, rate limit, timeout or the API's own) is
tried again, up to five times: 500 ms after, then 500 ms, then 2 s each; what streamed of the reply
is dropped, and the page says why it's trying again. A failure that would come again (a bad key, a
request the API rejects) isn't retried, and only a reply that comes whole counts toward the
session's costs. A reply that calls tools goes back in later requests with its reasoning as it came,
signatures and all, as DeepSeek's thinking mode requires (without it the model reasons from scratch
at each step); other replies' reasoning, and the reasoning that wrote a memory, stay in the log and
are never sent. Every request carries the system prompt and the tool definitions, so their tokens
come out of the window the context compiles to, and out of the usage that calibrates prices; while
calibration catches up with a request bigger than its estimate, the next context compiles that much
smaller. Memory requests declare the same tools with tool choice "none", as the library does, so a
memory can cover tool calls. The page's Continue button runs a turn without a message: after an
interruption (a failed request, a stop, a restart) the model goes on from the log as it is (and if
the server stopped in a step, its calls logged but not their results, each call first gets a failed
result saying so, as it does before any turn), and after a reply cut short it goes on from that
reply (the request ends with it, as a prefill). A turn on a log edited into something the library
can't read fails, and the session goes on once the log is mended. The page's Stop button (`POST
/api/s/NAME/stop`) ends the turn running where it is: a reply streaming in stops there and joins the
log so, without a call it was still writing; a call not yet run doesn't run, its result saying so;
one running ends (bash kills its command and what it started, and an MCP call is left to the
server's timeout); a memory forming is dropped, not counted as a failed attempt; and no step comes
after. A message (or Continue) sent while the session is busy waits, queued, above the message box,
where you can edit it (✎, in place) or take it back (×) until it goes to the model: the hub holds it
until the session's thread has done everything it was sent (`POST /api/s/NAME/queued` with `{id,
text}` or `{id, cancel: true}`; the session lists them as `queued`, and the stream says
`{"queue": […]}` as they change). What a companion, a channel or a board tells a session still joins
the turn running.

A page follows its session on one stream (`GET /api/s/NAME/stream`, JSON lines; `hub::Event` lists
them): the log's newest messages, read from disk, then the session, then whatever happens in it,
turn after turn: each message as it joins the log, and of each turn (numbered: sending a message, or
Continue, answers with its job's number, and the turn's events start with `{"turn": n}` and end with
`{"done": n}`) only what the log doesn't hold yet (a reply streaming in, the questions whether its
calls may run, their results, memories forming), which is all the app keeps of a turn. The page
shows each call as a card on its reply, its tool and input as the reply writes it, which its result
then joins; an edit's card shows its change as a diff, colored in the file's language. A turn runs
on its session's thread whether any page follows it or not, and a message sent while it runs waits
for it. Any number of pages can follow a session, each hearing everything, and any of them can
answer a call's question; a page that comes in during a turn (after a reload, or on another device)
hears the turn so far. A page out of view lets go of its stream (a browser keeps only a few
connections to a server open) and catches up once back in view, and a page whose server restarted
follows its session again on its own. If a session's files are
edited into something the library can't read, its stream says so, and reads them again at the next
event.

The gateway (`app/src/gateway.rs`), for now, only records. Claude Code (or the Agent SDK, which runs
it, or anything else that speaks the Messages API) run with
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080/gateway` and `ANTHROPIC_CUSTOM_HEADERS='x-connectome: 1'`
(the header every POST here carries) sends its requests here, and each goes on to `--upstream` as it
came: its body byte for byte, and its headers but this connection's (host, connection, content-length,
transfer-encoding and the like; accept-encoding, so the answer comes uncompressed; and x-connectome),
so its own credentials go with it, an API key or a Claude login's token. The answer streams back as it
arrives, with its status and headers. Each exchange joins
`ROOT/.gateway/SESSION/exchanges.jsonl` (SESSION is the request's `x-claude-code-session-id`, or
`unnamed`): `{id, at, method, path, headers, body, status, answer_headers, answer, failed}`, its
`authorization` and `x-api-key` left out, so a real session's requests can be read before the
gateway changes any of them. A record holds the whole request, which for Claude Code is the whole
conversation so far, so a long session's recording grows fast.

Claude Code sessions (`app/src/claude.rs`) take their turns in Claude Code: a provider in
settings.json with a `command`, how to run Claude Code (`["claude"]`, or whatever runs it here, say
a loader and the binary on NixOS), makes every session on it one, its `models` what Claude Code's
`--model` takes (`opus`, `sonnet`, `haiku` or full names); `base` and `key` go unused, so there's
nothing more to install than Claude Code itself. The page's Settings shows such a provider's
command, which only settings.json sets, and wants no key for it. Each turn runs the command headless in the
session's workspace, speaking JSON lines on its stdin and stdout as the Agent SDK does
(`--print --input-format stream-json --output-format stream-json`, its control requests asking
before a tool runs), with the session's model and its system prompt appended to Claude Code's
(`--append-system-prompt`, boards' note included), the MCP servers of `mcp.json` and no others
(`--strict-mcp-config`), none of the user's Claude Code settings files (`--setting-sources ""`),
and Claude Code's own sandbox off. Its requests go through the gateway (ANTHROPIC_BASE_URL), headed
`x-connectome-session: NAME`. In the session's sandbox, bubblewrap wraps all of Claude Code, as it
does bash, with the same paths, besides Claude Code's own: its command's absolute paths, read-only,
and `~/.claude` and `~/.claude.json`, read-write; and with the network even when the sandbox has
none, since the gateway is reached on it, so the commands Claude Code runs have it too. Its reply
streams to the page and joins the log block by block as Claude Code finishes each, its calls'
results as they come; a call Claude Code asks about shows on its card in the page with Allow,
Always allow and Deny, as any session's (Claude Code runs read-only ones without asking), a call of
one of its subagents, which stays in Claude Code and out of the log, on a card of its own below the
log, headed "a subagent asks"; and the
tools the session always allows are its chips. Claude Code keeps the conversation, in a session
of its own that the first turn starts (`claude_session` in the session's settings, once Claude
Code has started it) and later turns resume. A turn's cost, as Claude Code's result counts it,
joins the session's costs. Stop kills Claude Code and what it started, the reply joining the log as
far as it came. What a channel or a board tells a Claude Code session takes a turn of its own
rather than joining one running. Not yet: forks, a companion, Continue without a message, and
memories (the gateway forwards the requests unchanged).

Flags: `--root DIR` (sessions and settings; `sessions` by default), `--addr` (`127.0.0.1:8080`),
`--ca PEM` (root certificates to trust instead of Mozilla's: a company proxy's, a test server's),
`--mcp FILE`, `--tool-timeout SECONDS`, `--no-bash`, `--upstream URL` (where the gateway sends
requests on; `https://api.anthropic.com` by default), and `--no-open` (else the page opens in the
browser). The server answers only its own page: requests must name it as their Host, and POSTs must
carry its `x-connectome` header. A request's body takes 32 MiB at most.

Tests:

```sh
cargo test                  # unit tests of the JavaScript and Python arithmetic the ports share
python3 tests/test_rust.py  # every compile, request and file record the same as reimplclean2's, handoffs both ways, the planner
python3 tests/test_app.py   # the app against mock APIs over HTTPS, its parts at once: chat, memories, forks, settings, certificates, bash, its sandbox, approvals, timeouts and tools through fake MCP servers, flat memory, the gateway, Claude Code sessions through a fake of it (fake_claude.py)
python3 tests/mutants.py tests/test_app.py app/src/work.rs=mutants.json   # mutation testing, each mutant in a copy of its own, many at once
```

The three take under 30 seconds together. A batch of mutants takes about as long as its slowest one
(85 of the app's, under a minute): they build without link-time optimization, from a build kept in
`target/mutants`.
