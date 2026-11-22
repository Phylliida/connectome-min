# KV splice: building a request's cache from pieces

For the Kimi-K3 server (the llama.cpp fork run by `./run-kimi.sh` at http://127.0.0.1:8091) and connectome-web. Proposed
2026-11-09; the server has it since 2026-11-10 as `POST /kv/splice2` (also `/kv/splice` with a `request` body), and the
form connectome-web sends (`sources`, pieces by index) since 2026-11-20, with the changes listed under "The server's
implementation". It replaces the span-for-span form of `POST /kv/splice`.

## Why

connectome-web folds the main session's context: old messages are replaced by memories the model wrote in separate
memory requests, and memories merge into higher ones. A fold changes the context right after the head and the older
memories, so everything after that point is prefilled again, tens of thousands of tokens, while the turn waits. Memory
requests prefill a lot too: an L1 request reads its chunk again, and a merge of L1s reads their chunks again (since
2026-11-20 only under `merge_by_count`; merging by tokens, it shows each L1 as its recall pair).

Almost all of that text already sits in some cached conversation, computed once:

- a memory's text is the reply of the memory request that wrote it;
- raw messages are in the main session;
- a chunk is also in the memory request that wrote its L1, which showed it raw.

What differs between the two sides is the framing around that text. The main session shows a memory after the Context
Manager's label; a memory request shows it after `[CM] Recall memory L1-3.`; a memory request opens its chunk with a
marker and closes it with an instruction; consecutive turns of one role join into one message differently on each side,
so role tokens fall in other places. One span can't simply replace another: the shared text has to be copied and the
framing computed fresh.

## The call

```json
POST /kv/splice
{
  "format": "anthropic",
  "request": { ...the next request, exactly as it will be sent... },
  "sources": [
    {"reply": "<a memory's text>"},
    {"request": { ...a cached conversation's request... }}
  ],
  "pieces": [
    {"src": 0, "anchor": "<the memory's text>"},
    {"src": 1, "anchor": "<a message's text>"},
    {"src": 1, "anchor": "<the next message's text>"}
  ]
}
```

- `format`: `"anthropic"` (bodies as sent to `/v1/messages`) or `"openai"` (as sent to `/v1/chat/completions`, the
  default). Every body in the call is in it.
- `request`: the next request, exactly as it will be sent; `max_tokens`, `stream` and the like are ignored. Sending it
  afterwards, or a request that extends it, reuses what the call built.
- `sources`: the cached conversations pieces come from, each named once, however many pieces it gives (a memory
  request's chunk gives one per text block, and its source request can be 90k tokens long):
  - `{"request": body}`: the cached conversation of that request, with its reply if one was generated, found as the
    current API finds `src` and `dst`.
  - `{"reply": text}`: the cached conversation whose reply was `text`. A reply's key is its text content as the API
    returned it (its text blocks joined with `"\n"`, reasoning left out), compared with surrounding whitespace trimmed;
    if several conversations match, the most recent. The server records the key as it caches a conversation.
- `pieces`: in the order their anchors appear in `request`, each its source's index in `sources` (`src`) and the
  text it starts with (`anchor`), normally a message's text. A piece may still give its source inline, as
  `"src": {"request": body}` or `"src": {"reply": text}`.
- `min_tokens` (optional, default 16): a piece shorter than this isn't worth a copy and is prefilled.

The current form (`src`, `src_span`, `dst`, `dst_span`) can stay; a body with `request` is this form.

## What the server does

1. Render and tokenize `request` as its endpoint would.
2. The prefix: the longest common prefix with a cached conversation, the same lookup a request makes. Kept as it is.
3. Place each piece. A piece that breaks a rule is skipped, and its tokens are prefilled like any other:
   1. Its anchor occurs exactly once in `request` after the prefix, and exactly once in the source.
   2. The anchor starts on a token boundary in both.
   3. From the anchor, the piece runs while the tokens of `request` and of the source are identical. It stops at the
      first token that differs, at the next piece's anchor, at the end of the source's cached tokens (the last generated
      token has no cache entry), or at the end of `request`.
   4. Its anchor comes after the end of the previous piece placed.
   5. It is at least `min_tokens` long.

   A placed piece's cache entries are copied from the source and moved to their new positions.
4. Gaps: each token of `request` between the prefix's end and the last placed piece's end that no piece covers is
   evaluated by the model at its position, attending to everything before it. These are the labels, recall labels,
   markers and role tokens around the pieces. Tokens after the last piece are left for the request itself.
5. Recurrent state, below.
6. The slot then holds `request`'s tokens up to the last piece's end. The draft model's cache restarts, as after a splice
   now.

Nothing is matched except at an anchor the client gave, and no piece runs across a place where the two sides differ, so
a piece can't land anywhere unexpected. A token the source sampled differently from how `request` tokenizes the same
text ends the piece there (rule 3), which replaces the current API's `canonical` check. A source that is missing,
evicted or wrong means more prefill, never an error.

### Recurrent state

A piece copies attention (MLA) entries only. The recurrent (KDA) layers have one state per conversation, and it must
come from somewhere:

- after a placed piece: its source's state at the piece's end;
- after a gap: the state the gap's evaluation leaves, starting from the state after whatever precedes the gap (the
  prefix, or the previous piece);
- the slot keeps the state after the last thing built.

A conversation's state at position `p` is a checkpoint at `p`, taken from that conversation or from any cached one whose
tokens up to `p` are the same. Without one, it is the latest such checkpoint before `p`, advanced over the
conversation's own tokens up to `p`, or used as it is if advancing would take too long. A state from after `p` is never
used. For example, a memory request takes its chunk from the main session, and the main session's final state has seen
everything after the chunk; the memory could then tell of later events.

For a fold, the last piece runs to the end of the main session's cached tokens, so its state is the main session's own
final state.

### The server's implementation

Where the server built on K3 departs from the above:

- Anchor alignment: a piece starts at the first token boundary within the anchor's first 64 characters that both sides
  share, since a tokenizer can glue the character before an anchor onto its first word on one side only. Exact token
  equality still decides how far a piece runs.
- Recurrent state (since 2026-11-20): a piece takes its source's state only where that state is exact, when the piece
  runs to the source's cached end or a checkpoint sits at the piece's end. Only the last piece's state stays in the
  slot, so only it falls back to the latest checkpoint before its end, advanced or used as it is. After any other piece
  the gap starts from the state built so far. Advancing a state for every piece cost up to `max_advance` tokens of
  evaluation per piece, more than a fold of one piece per message saves, for the few tokens of the gap that follows.
- `max_advance` (default 2048 tokens): the furthest a checkpoint is advanced, for the last piece and for the prefix;
  past that, the nearer checkpoint's state is used as it is.
- A checkpoint is stored at the end of each piece that took a state of its own, so later rollbacks stay cheap (not at
  every piece: the slot keeps at most 32 checkpoints, and a fold can have hundreds of pieces).
- Each source is loaded once, however many pieces it gives. A piece's `src` is an index in `sources` or a source
  inline (`sources` and indices since 2026-11-20).
- Reply keys are recorded at each completion and saved to `reply-keys.jsonl` beside the disk cache, so they survive
  restarts; only replies generated since the feature exists are known. A reply cut off at `max_tokens` can't be a
  source: its last token has no cache entry, so its full text is not in the conversation.
- The response also has `prefix_recurrent` and a `timing_ms` breakdown.

## Response (200)

```json
{
  "built": 65120, "prefix": 12011, "copied": 51873, "prefilled": 1236,
  "pieces": [
    {"placed": true, "range": [12034, 13890], "src_range": [40112, 41968]},
    {"placed": false, "why": "the anchor occurs 2 times in the request after the prefix"}
  ],
  "recurrent": {"from": "piece 3", "at": 65120, "advanced": 0},
  "t_ms": 1400
}
```

- `built`: the tokens the slot now holds for `request`; `prefix + copied + prefilled == built`.
- `pieces`: one entry per piece, in order: where it went in `request` and where it came from in its source, or why it
  wasn't placed.
- `recurrent`: what the slot's recurrent state came from: a piece or a gap, the position of the state used, and how many
  tokens it was advanced over (or how many it fell short, as a negative number, if it was used as it was).

## Errors (HTTP 400, nothing changed)

- `request` is missing, or a body doesn't render.
- A piece's `src` is neither an index in `sources` nor a source.
- `request` or a source contains images.
- A request is running.

## How connectome-web uses it

The app (connectome-web's `work.rs`, since 2026-11-10) calls it before a request that a fold changed and before each
memory request that recalls a memory, then sends the request, whatever the call came to; the costs log keeps each call's counts or error. It
gives a piece for each text block of `request` after the cached prefix whose text occurs there once: a memory's text, a user
message, an assistant's text, a tool result's text. A piece runs only to the next anchor, so where both sides render the
same, consecutive pieces cover it end to end, and where they differ, copying starts again at the next text block.
Sources follow one rule: a memory's text comes from the conversation that wrote it (`{"reply": its text}`); raw messages
come from the main session's last request, or from the conversation that wrote their chunk's L1, which showed the chunk
raw. Memory requests copy only the memories they recall, since 2026-11-20: what they summarize, a chunk's messages or a
merge's members, is prefilled (see "Checking quality").

Every text block gets an anchor, not only a stretch's first, because memory requests replay messages without the agent's
reasoning (the library's `raw`, as Anima does), while main requests carry it on the turns that call tools. On K3 the two
render the same but for that turn's think block:

```
main:   <|open|>message role="assistant"<|sep|><|open|>think<|sep|>REASONING<|close|>think<|sep|><|open|>response<|sep|>Let me look.<|close|>response...
memory: <|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>Let me look.<|close|>response...
```

So a piece of a chunk stops at each such turn, and the next starts at the turn's text ("Let me look."), or at the tool
result's text when the turn has none, the tool call then prefilled.

A fold, which replaces two chunks with memories M3 and M4:

```
[head][label][M1][label][M2] [label] [M3] [label] [M4] [the rest of the main session ...] [new message]
 prefix                       gap     piece gap    piece gap  pieces                        left for the request
```

The pieces are `{"reply": M3}` and `{"reply": M4}` anchored at their texts, then one from `{"request": the last main
request}` per text block of the rest, which together run to the end of the main session's cached tokens.

An L1 request for chunk k:

```
[head][recall L1-3][M3][recall L1-4][M4][marker][chunk k ...][instruction]
 prefix gap         piece gap        piece gap   pieces       left for the request
```

The pieces are `{"reply": M3}` and `{"reply": M4}`; the chunk is left for the request, which prefills it. (Until
2026-11-20 the chunk also had a piece per text block from `{"request": the last main request}`, and a merge of L1s took
each member's chunk from `{"reply": that L1's text}`; see "Checking quality" for why that stopped.) A merge likewise
copies only the memories it recalls, and its members, shown as recall pairs (an L1 as its own, a higher memory as its
members), are prefilled. A piece that copies a whole reply takes that conversation's state, so with its members copied a
merge would be left the state of the conversation that wrote the last of them, which read that member's chunk raw: the
failure below, inferred and not run on K3.

The app keeps the last main request it sent, a list of references to blocks on disk. After the app restarts it has
none until its next request, so a fold before then is prefilled.

### Prerequisites in connectome-web

- Memory requests send the session's system prompt. K3's template puts it after the tool declarations and before the
  first message, so without it (`app/src/work.rs:124` builds a memory request's head without one) a memory request
  shares only the tool declarations with the main session. `tool_choice: none`, which memory requests also send,
  renders the same as no `tool_choice` (checked with `/apply-template`, 2026-11-10).
- Requests use `format: "anthropic"`, so the app sends the same bodies it sends to `/v1/messages`.
- A session with images doesn't use it.

## Checking quality

A built cache isn't what prefilling `request` would give. Pieces were computed in their source's context: a memory's
text attended to its chunk and its reasoning, and the main session's rest attended to messages that have since been
folded away. Recurrent states come from elsewhere too. Before relying on it, compare built and prefilled caches on a
synthetic session:

- folds: the log-probabilities of the same continuation, or answers at temperature 0, with and without the call, and
  questions about facts that only the folded messages held;
- memory requests: memories written with and without the call, and whether a built one tells of events after its chunk.

Results on K3 (2026-11-20, `k3-research/kv-splice` in llm-tests, greedy, thinking off and low):

- A fold matches a fresh prefill (same answer, first-token KL 0.004).
- The mechanics are exact: a memory request built whole from a copy of itself (only its system prompt differed) wrote the
  same memory as a fresh prefill, word for word, in two runs (KL 0.025-0.039).
- A memory request whose chunk is copied from the main session writes the wrong memory: of days 1-8 instead of its
  chunk's days 5-8. K3 writes a memory only of what its recurrent layers (69 of 93) read, and the copied chunk's
  attention entries don't suffice. Given no state for the chunk, K3 summarized the recalled memory or looped; given the
  main session's state at the chunk's end, it summarized every earlier chunk too, since that state read them raw. No
  cached state has read only the chunk, so memory requests don't copy their messages.
- Copying the recalled memories alone is harmless: with the chunk prefilled, the memory was right (KL 0.02-0.04), even
  with the state of the conversation that wrote the recalled memory, which read chunk 1 raw.

connectome-web logs each call's response next to the request it preceded, so a turn that ran on a built cache can be
told from one that didn't.
