# Anima's other memory systems

What Anima Labs runs beside the autobiographical memory this port implements, and how each would fit
here. Surveyed 2026-10-21 from the clones in `tmp/anima-latest/`: context-manager `9daa4d7`,
connectome-host `210ca3c`, agent-framework `2099c40`. Kept for later; none is built here.

## First: the window and the raw tail

A session that runs long on our defaults (a 90k window, a raw tail of at most 30k) shows mostly
memories, each a 15k chunk in about 2k to 8k tokens, so the context gets dense. Anima's memory
guide (connectome-host `docs/AGENT-MEMORY-GUIDE.md`) calls the tail the setting that does the work:
"real deployments use tails of hundreds of thousands of tokens, not the small library fallback".
Since 2026-10-15 their host defaults to a 300k budget with 40% of it (120k) raw, and its larger
agents run 300k to 450k tails. With adaptive resolution the middle stays raw while it fits, so a
larger window alone turns most of a dense context back into raw turns, at the price of reading more
tokens each step (mostly cache hits).

The same commit moved their host's solver from kv-stable, which this port implements, to
kv-unified (`standardKvUnifiedConfig` in connectome-host `src/framework-strategy.ts`).

## History tools

agent-framework's `HistoryModule` (`src/modules/history/index.ts`) gives the agent read-only tools
over its whole raw record: `search` (substring or regex, by author, oldest or newest first),
`extract` (a time range, or the conversation around one message), `overview` (a table of contents
built from the memories already written, with no model call, and raw message counts where nothing
is summarized yet) and `stats`. An optional `semantic_search` finds messages and memories by
meaning, from an embedding index.

The best fit here: it recovers the exact detail folding blurred, and our logs are already the whole
record, so the tools only read them, streaming, with no index beside the log.

## Workspace notes

The memory guide tells the agent that anything which must stay exact belongs in its workspace,
whose files never fold; a journal entry in the conversation folds like the rest. Here the agent
has the sandbox's files already, so this is a line in the system prompt.

## Shared instructions

connectome-host's `modules.instructions` puts a living AGENTS.md-like file from the workspace into
every turn, for the agent and its subagents. It is never stored in the history, so it never folds,
and an edit costs a cache miss on every agent's next turn. Here it would be model-visible content
outside the log, which the log as the single record rules out unless each version the requests used
is logged.

## Lessons and retrieval

connectome-host's `lessons` module is a library of lessons (content, confidence 0 to 1, tags,
evidence links) the agent keeps with `create`, `update`, `deprecate` and `query` tools.
`retrieval` adds to each turn up to a few lessons chosen in three steps: a model call names the
concepts under discussion, a keyword match finds lessons for them, and a second model call keeps
the relevant ones. Both are off by default even there: two extra calls a turn, useful only to an
agent that curates lessons.

## Pins

An operator can hold a range of history raw, or at a memory level, from their web console
(context-manager `pinRange`, `markDocument`). The solver treats a pin as a constraint on the plan.

## Agent memory tools (a spec, not built)

context-manager `docs/memory-tools-design.md` (2026-08-19) specifies tools that give the agent
authority over its own past: a cheap "dry index" of terse bullets per memory, written by an
inexpensive model and used only for navigation; `memory_timeline` and `memory_search` over it;
`memory_recall`, which shows a span raw or as its memory as a document that rolls out with the
tail; and `memory_pin` (hold a span at, at most or at least a level), admitted only after a dry-run
compile shows it fits a pin budget, and expiring after 30 days unless renewed. No implementation is
in any of the three repositories.

## Not relevant here

context-manager's `KnowledgeStrategy` chunks at phase changes and tracks open leads, for knowledge
extraction runs; connectome-host's `frontdesk` strategy is for agents that staff chat channels.
