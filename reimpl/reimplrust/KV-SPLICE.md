# KV splice: building a request's cache from pieces

Proposal, 2026-11-09, for the Kimi-K3 server (the llama.cpp fork run by `./run-kimi.sh` at http://127.0.0.1:8091) and
connectome-web. It replaces the span-for-span form of `POST /kv/splice`.

## Why

connectome-web folds the main session's context: old messages are replaced by memories the model wrote in separate
memory requests, and memories merge into higher ones. A fold changes the context right after the head and the older
memories, so everything after that point is prefilled again, tens of thousands of tokens, while the turn waits. Memory
requests prefill a lot too: an L1 request reads its chunk again, and a merge of L1s reads their chunks again.

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
  "pieces": [
    {"src": {"reply": "<a memory's text>"}, "anchor": "<the same text>"},
    {"src": {"request": { ...a cached conversation's request... }}, "anchor": "<the text of a stretch's first message>"}
  ]
}
```

- `format`: `"anthropic"` (bodies as sent to `/v1/messages`) or `"openai"` (as sent to `/v1/chat/completions`, the
  default). Every body in the call is in it.
- `request`: the next request, exactly as it will be sent; `max_tokens`, `stream` and the like are ignored. Sending it
  afterwards, or a request that extends it, reuses what the call built.
- `pieces`: in the order their anchors appear in `request`. A piece's `src` names a cached conversation:
  - `{"request": body}`: the cached conversation of that request, with its reply if one was generated, found as the
    current API finds `src` and `dst`.
  - `{"reply": text}`: the cached conversation whose reply was `text`. A reply's key is its text content as the API
    returned it (its text blocks joined with `"\n"`, reasoning left out), compared with surrounding whitespace trimmed;
    if several conversations match, the most recent. The server records the key as it caches a conversation.
- `anchor`: the text a piece starts with, normally the start of a message's text.
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
- `request` or a source contains images.
- A request is running.

## How connectome-web uses it

The app calls it before a request that a fold changed and before each memory request, then sends the request. Sources
follow one rule: a memory's text comes from the conversation that wrote it (`{"reply": its text}`); raw messages come
from the main session's last request, or, in a merge, from the conversation that wrote the member's L1, which showed the
chunk raw. An anchor is the first text block of a message: a memory's text, a user message, an assistant's text, a tool
result's text. When a message's text isn't unique, the app anchors at a later message of the stretch, and the messages
before it are prefilled.

A fold, which replaces two chunks with memories M3 and M4:

```
[head][label][M1][label][M2] [label] [M3] [label] [M4] [the rest of the main session ...] [new message]
 prefix                       gap     piece gap    piece gap  piece                         left for the request
```

The pieces are `{"reply": M3}` and `{"reply": M4}` anchored at their texts, then `{"request": the last main request}`
anchored at the text of the first message after the fold, running to the end of the main session's cached tokens.

An L1 request for chunk k:

```
[head][recall L1-3][M3][recall L1-4][M4][marker][chunk k ...][instruction]
 prefix gap         piece gap        piece gap   piece        left for the request
```

The pieces are `{"reply": M3}` and `{"reply": M4}`, then `{"request": the last main request}` anchored at the text of
the chunk's first message. A merge of L1s takes each member's chunk from `{"reply": that L1's text}`, anchored at the
text of the chunk's first message; a merge of higher memories takes their members' texts from `{"reply": ...}` too.

The app keeps the last main request it sent, a list of references to blocks on disk. After the app restarts it has
none until its next request, so a fold before then is prefilled.

### Prerequisites in connectome-web

- The prefix covers the head only if memory requests render the same tokens as main requests from the first token.
  Today they don't send the session's system prompt (`app/src/work.rs:124` builds their head without one), and they send
  `tool_choice: none`. Memory requests should send the system prompt. If `/apply-template` shows that `tool_choice: none`
  changes the rendered prompt, drop it and reject a reply that calls a tool, as Anima does.
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

connectome-web logs each call's response next to the request it preceded, so a turn that ran on a built cache can be
told from one that didn't.
