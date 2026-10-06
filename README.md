# connectome-min

A four-file model of the Anima Connectome's compaction (`@animalabs/context-manager` 0.10.1,
vendored at `ref/context-manager`), plus its tests:

| file | lines | what it is |
|---|---|---|
| `connectome_min.py` | 151 | the POLICY: which level each chunk renders at (the kv-stable cascade), plus the demand-side `produced` runs an escalated solve emits |
| `minisystem.py` | 513 | the SYSTEM: log, derived work, guards, request projection, failure policy |
| `demo.py` | 179 | the HOST: session loop, synthetic messages, scripted model, the printed story |
| `llm/summarizer.py` | 309 | the MODEL BOUNDARY: request assembly, pre-send admission, offline mock, HTTP client, terminal-disposition gate, usage |
| `config.py` | 770 | every KNOB: ours vs production's default, why it exists, audit status -- and a status check that refuses a self-contradicting table |
| `test_summarizer.py` | 1617 | the boundary's and the system's tests: `python3 test_summarizer.py` (stdlib only, no network, 371 checks) |

Same branch cascade, same formulas; not the same system. `python3 config.py --gaps` is the
authoritative itemized list of where we differ, and the last section here explains the shape of
those differences.

Both rewrites were verified, not asserted: the policy's output is byte-identical across its four
documented runs, and the system demo is diffed against a fresh pre-change capture with every
changed line accounted for (`AUDIT.md` §6, §7, §8 and §9).

## Layout: three layers, one seam

```
connectome_min.py     the ALGORITHM -- which level each chunk renders at. Imports config only:
                      no model, no HTTP, no prompts.
minisystem.py         the SYSTEM -- log, derived work, guards, request projection, failure
                      policy. Imports the algorithm and config; takes a boundary as an argument.
llm/                  the INFERENCE boundary -- request assembly, gate, mock, HTTP client,
                      admission, calibration, and llm/prompts/ as data. Nothing below imports it.
demo.py               a HOST -- the session loop and the synthetic session that tells the story.
config.py             the single source of constants, and the audit of our fidelity to production.
```

That direction is enforced by the imports, not by convention: `grep -E '^(import|from)' connectome_min.py`
shows `config` and nothing model-shaped, and `minisystem.py` names no class from `llm/`. Swapping the
boundary is meant to leave the memory untouched.

## The two layers

* **Production** (`add_chunk`, `mint`, `summarize`) — append-only, model-calling, not
  pressure-gated. A chunk closes at `targetChunkTokens` and gets an L1; every `mergeThreshold`
  contiguous unmerged siblings become one L(k+1). Nothing is deleted: the archive is monotonic
  and the live context is only a projection of it.
* **Presentation** (`plan_controlled_frontier` and friends) — per turn, pure CPU, no model
  calls. It picks one level per chunk. That projection is the compaction decision.

## Where each piece comes from

| here | real source | what it is |
|---|---|---|
| `render`, `rendered` | `adaptive/render-offsets.ts:52`, `picker.ts:234` | the request as a unit stream: per chunk a raw shard or a recall pair |
| `kv_cost` | `render-offsets.ts:111` | perturbation = tokens re-read from the first divergence |
| `fold_depth_cap` | `kv-control.ts:120` | soft shape prior: log-age banding; `-1` = hard protection |
| `fold_pass` | `kv-control.ts:347` | group-atomic fold, phase A (under prior) and B (past it) |
| `pack` | `kv-control.ts:399` | phase C: un-fold youngest-first to spend headroom |
| `relevance_cut` | `kv-control.ts:268` | the ideal cut from scratch, A/B/C, P never consulted |
| `suffix_adopt` | `kv-control.ts:523` | partial adoption: suffix cuts, binary search under P |
| `plan_controlled_frontier` | `kv-control.ts:809` | the cascade: hold / ideal / suffix / override |
| `loss` (inline) | `kv-control.ts:166` | salience-weighted misallocation, the quality-gap comparator |
| `add_chunk`, `mint` | `autobiographical.ts:10110`, `:6824` | chunk closure, L1 minting, base-k merges |
| `summarize` | — | the policy's own word-truncating stand-in for a memory body (the real compression call is `llm/summarizer.py`, whose L1 shape is production's) |

## Reading the output

`dP` is the plan's perturbation: tokens the provider must re-read. `hold` always shows `0` —
that is the point of the dead band `[target, W]`. Branches, in the order the solver tries them:

1. `bootstrap` — nothing carried, so there is no cache to preserve.
2. `hold` — carried fits the dead band and is not badly misallocated.
3. `adopt-ideal` — switching to the whole ideal cut costs less than P.
4. `suffix-adopt` — it costs more than P, so adopt only its *newest* changes; the rest waits.
   Three consecutive turns spending 1240/940/940 in the `--reach 1250` run are the
   receding-horizon repair: one P per turn instead of one 1840-token stall.
5. `override` — nothing within P works, so feasibility (`infeasible`) or quality
   (`quality-gap`) wins and the ideal is adopted anyway, recorded loudly.

`escalated: over W` means even full folding cannot fit the window. That is the state in which
production's kv-stable solve produces demand-side L1 requests, and ours does too: the plan carries
`produced` -- one coalesced run per stretch of uncovered, foldable chunks -- and `minisystem.work`
lets those chunks past the `l1HoldbackChunks` window. It is computed as `kv-stable.ts:180-198`
computes it (an `if (plan.escalated)` block over chunks with no L1 that are not raw-zone), and the
four documented runs are byte-identical because their replay mints an L1 with every chunk, so no
chunk is ever uncovered there.

`raw zone` is the same boundary from the other side. The solver may never fold a head-window,
tail-window or pinned chunk (`fold_depth_cap` returns `-1`), and production never even chunks one:
its chunker is handed `getCompressibleMessages`, so the zone is the complement of what it
compresses. `minisystem.raw_zone` is that zone for the derivation, and the demo prints how many
chunks it covers. One consequence is worth stating plainly, because it looks like a bug and is
not: a session shorter than head + tail compresses *nothing*, so the demo's first L1 arrives at
message 19 of 37.

## `minisystem.py` — the system layer

`connectome_min.py` decides *which level to render each chunk at*. Everything else a compactor
needs — where state lives, what a failed compression does, who may not be folded — is
`minisystem.py`: 513 lines, driven by the host in `demo.py`. It imports the policy unchanged; the
policy's
synthetic production (`add_chunk`/`mint`) is bypassed for a log-derived archive, and the model
itself lives behind the boundary in `llm/summarizer.py`.

Two primitives collapsed on the second pass, which is the interesting part:

* **the work queue is not stored** — work is *derived* (chunks without an L1, base-runs without a
  parent), so there is no enqueue, no work id, no dedupe, and no queue to rebuild;
* **the ledger is not a state machine** — the log remembers only *failures*, and `attempts`,
  `streak`, `stalled` and `terminal` are projections over them. A terminal unit is simply one
  whose failures crossed the bound: it stops being derived, which *is* quarantine. It is also
  clearable, because every one of those projections reads the unit's failures **since its last
  clear event**: append a `clear` (or let the span mint later, which clears it by itself) and the
  unit is derivable work again. Production needs a map, a lock and an event append for the same
  thing (`clearCompressionRefusalQuarantine`, `:1263`); here it is one more record.

```
python3 demo.py
```

It prints the failure-policy table, then a scripted session (refusals, a tool-calling model, a
truncating model, an oversized request, a mid-flight fork, a window cut) and a restart:

```
          discarded-stale L1:c3   a fork lands mid-call: the reply is dropped, no attempt burned
          retry           L1:c3   refused twice, then the source-only shape passes
  terminal        L1:c4   attempts=3     bounded attempts exhausted -> terminal debt
  transient       L1:c5   attempts=0     a provider blip burns no attempt
  shrink          L1:c6   attempts=1     the request did not fit: halve it
  turn  7  plan adopt-ideal tokens= 2345 dP= 1247 OVER W (inside grace)
  phase ends at 2345 tokens: W=2300 and the enforced wall is 2346 -- the grace band lets a
                             plan render over W without stopping the session
  REFUSED: turn 8: 2296 tokens > wall 1224 (W=1200 + 2%)   <- production raises OverBudgetError
  demand path: the escalated plan produced [('c4', 'c4'), ('c7', 'c7')] -- ['L1:c4', 'L1:c7']
               would skip the holdback, blocked by terminal debt: ['L1:c4']
  stalled work (terminal debt): ['L1:c4']   111 raw tokens that no plan can fold away yet
  compressible zone: 5 of 10 chunks (5 protected by head/tail/pin: ['c0','c1','c2','c8','c9'])
  message salience (composition): [1.0, 0.85, 0.2, 0.94, 1.0, 1.0, 1.0, 0.94]
  -> chunk salience is its cheapest message: {'c0': 0.2, 'c1': 0.2, 'c2': 0.94, 'c3': 0.2}
  shards minted from the one long message: ['c5', 'c6', 'c7']
  compression calls spent: 12 (one per message, not per turn)
  model usage: 12616 in / 139 out tokens (MockModel estimates at 4 chars/token; HttpModel bills
               what the provider reports)
  ...
  explicit half: L1:c4 attempts=3 stalled=True; derivable work for it: []
                 clear_debt(L1:c4) -> attempts=0 stalled=False, derivable again: ['L1:c4']
```

The scripted outcomes are keyed by the chunk they land on, so they move with the compressible
zone: the first chunk this session can compress is `c3`, because until the head and the tail stop
covering everything the derivation is empty. That is the behaviour rather than a missing line, and
it has a visible consequence: 37 messages over the 9 turns that run only ever bring four chunks to
the boundary, so the pyramid never reaches six unmerged L1s and **no merge runs in this demo**. The
merge request is pinned by `test_summarizer.py` instead; `AUDIT.md` §9 has the numbers.

The accounting/rendering rules the model implements (all default-ON in production, and all absent
before the first pass): computed salience, recall-pair pricing (production's `recallPairCost`:
question label + content), `maxMessageTokens` truncation, image stripping (`maxLiveImages` +
`imageStripDepthTokens`), bodyGroup sharding, the tool_use pairing guard on chunk closure,
`l1HoldbackChunks`, `autoTickOnNewMessage` (one compression attempt per message) and
`overBudgetGraceRatio` (the enforced wall, which turns "degraded" into "refused"). Each is
documented in `config.py`, including the ones whose label had to change: the tick's *drain* is
still missing, the demand path now exists, and the truncation below is real.

The rules that are not accounting, added by the audits that produced `AUDIT.md`:

* **the compressible zone** — production does not chunk what it will never fold: `rebuildChunks`
  is handed `getCompressibleMessages` (`:10054-10056`), which is everything before the recent
  window minus the head window and every pin (`:9967-9980`). Our chunker runs over every message
  (the chunk projection also prices the window and renders raw units), so `minisystem.raw_zone`
  applies the policy's own boundary to the *derivation* instead: the head chunk, the tail chunks
  and every pinned chunk get no L1. Before this, the head chunk always had one — which made the
  head section of every L1 request empty, and the section order it exists to protect unexercised.
* **the overlap guard** — a chunk whose **messages** a live L1 already stands for is not
  compressed again. Production's three arms (`findExactL1` `:5142-5155`, fully-covered
  `:5164-5181`, partial overlap `:5183-5203`) are one span test here, and `overlap_blocked` is the
  loud half. An id test cannot see a re-derivation that moves chunk boundaries while the messages
  stay put; the drift is real (image stripping re-prices old messages), and the fixture in
  `test_overlap_guard_is_a_span_test` uses it.
* **the merge guard** — a merge candidate whose source span no longer resolves to live messages is
  frontier debt, and one that is wide *for its level* is quarantined
  (`contiguousMergeCandidates`, `:6654-6688`). The level scaling is the load-bearing half: a flat
  limit is what made an L5 structurally impossible in production's own store.
* **the demand path** — described above; its honest limit is that at this configuration the
  holdback window sits inside the picker's raw zone, so the bypass only bites when the two windows
  diverge (the test raises the holdback to two to show it).
* **the coverage invariant** — `minisystem.assert_coverage` runs on every plan before anything is
  emitted and raises `UncoveredDropError`, production's fatal error (`src/index.ts:38`, thrown at
  `:4655`): every chunk is raw, or inside a recall whose leaves cover it. Checked on the emitted
  render, so a stale mint whose leaves drifted onto another chunk is caught rather than absorbed.
* **clearing terminal debt** — `clear_debt(log, target)` appends a `clear` event (the log is the
  only store) and the debt vocabulary reads failures *since* the last clear, so the unit is
  derivable work again. The automatic half is production's success-clear: a unit that later mints
  is cleared with no event at all.

### What each primitive replaces

| primitive | production mechanisms it replaces | measured in `ref/` |
|---|---|---|
| `log` + projections | chronicle archive, state slots, branch tokens, reconcile/repair tooling | DSH's log-derived store: `store.ts` + `seed.ts` = 853 lines |
| derived `work()` + `fails` | `compressionQueue`, `mergeQueue`, both quarantine maps, the alarm subsystem, status APIs | 4 mechanisms, 137 references, 15 quarantine-named functions, 118 quarantine lines in `autobiographical.ts` |
| `step()` boundary | the refusal ladder: canonical → 3 curve variants → source-only → split-stitch → placeholder | 3+ knobs, `fallbackLimit`, the `admission_rejected` path |
| `policy()` + `ACTIONS` | the retry/streak/quarantine policies the L1 and merge sites each grew | "the merge site had already grown this retry" (`autobiographical.ts:5910`) |
| `protected_ids()` | pins, documents, locks, `pinLevel`, `pinMaxLevel`, witnessed ranges | 6 APIs → 1 predicate |
| version compare in `step()` | branch tokens, discard logging, `settleInFlight`, `phaseChannel` | 1 compare vs. checks at every await |

Caveats: the merge picker still takes the first `mergeThreshold` *eligible* unmerged siblings,
without production's strictly-contiguous run split and interior-run escape
(`contiguousMergeCandidates`) — orthogonal to the primitives, but a real difference. Its two
candidate filters (unresolved source span, wide for its level) are implemented and cannot bind at
demo scale, which `config.py` and `AUDIT.md` §9 both say out loud. `MockModel` is deterministic,
and the numbers above measure anatomy, not performance.

## `llm/summarizer.py` — the compression boundary

`minisystem.py` used to call `Scripted`, a stub that returned truncated source text. This is what
replaced it, and it is the only file that knows a provider exists. Four pieces and one contract:

* **`build_request(kind, target, source, prefix, reading_tokens, source_only)`** — the one place a
  request is assembled, for L1 and merge alike. **Both requests are production's six sections now**
  (`autobiographical.ts:5266-5294` for the L1, `:6867-6891` for the merge), which is what closed the
  last wholesale deviation this artifact had: the L1's head window, prior recall pairs and raw
  middle come from the same projection (`minisystem.source`) the merge's prefix always did, so the
  two requests now differ only in what their leaves are.

  The L1 turn, in production's order:

  1. **HEAD** — the raw chronicle opening, the identity anchor, **first**. The order is
     load-bearing: when the head *followed* the recall pairs it read as the most recent live
     conversation, thin chunks narrated it as fresh events, and the error compounded across merges
     into runaway false memories (`:5268-5278`, the "68 initiations" incident). Chronological order
     is also the KV-stable order — the head never changes.
  2. **PRIOR SUMMARIES** — the **unmerged frontier** as `[CM] Recall memory <id>.` CM-ask /
     agent-recall pairs, in source order, capped by `capRecallPairs` (`:2489-2514`, budget read at
     `:5430-5439`) — walked newest-first and re-sorted chronologically. Children *plus* their parent
     would double the prompt size unboundedly: the original "all L1s regardless of merge state" rule
     converged to ~500 L1s that never aged out and blew the window around chunk 118 of a
     4000-message import (`:5305-5311`).
  3. **RAW MIDDLE** — messages between the head window and the chunk that no summary covers. Empty
     in a normal run here — see the caveat below.
  4. **MARKER** — the in-band `COMPRESSION_MARKER` (`:99-102`, pushed at `:5470-5473`).
  5. **CHUNK** — the raw messages being compressed.
  6. **INSTRUCTION** — `formatInstruction(targetTokens)` (`:192-206`, reached at `:9683`, pushed at
     `:5503-5506`), **doc-aware**: when the chunk is one shard of a substantially larger message,
     `formatReadingChunkInstruction` (`:250-266`, reached from `:9696-9701`) replaces it, because the
     consolidation framing makes a model reading someone else's document adopt its author's voice
     (`:5497-5502`). The detection is `detectDocContext` (`:9716-9757`), with both of production's
     L1-site guards: the chunk must *be* one shard, and the whole group must be at least twice it.

  **No tail after the chunk** (`:5292-5294`): that would leak future information into the model's KV
  state and corrupt the as-of framing of memory formation. Section order and that absence are pinned
  by `test_summarizer.py` against a live session's request.

  **One shape difference, and it is packaging rather than content:** production pushes each section
  as its own participant turn — the head and raw messages as the original participants, the pairs as
  `Context Manager` / the agent, the marker and directive as `Context Manager` (`:5392-5504`) — while
  this boundary sends one system turn and one user turn, joining the sections with a blank line. The
  same trade the merge request already made; the section order, text and boundaries are
  production's.

  `source_only=True` selects production's `compressionSourceOnly` arm instead
  (`types/strategy.ts:966-980`, implemented at `:5411-5466`): the marker, the chunk and the directive
  — sections 4-6 of that builder by construction. It is a **rung**, not a shape: when the canonical
  request is refused, the next attempt for that unit is built on it, derived from the refusal the log
  already recorded, exactly as an `overflow` derives the reshape. Production's incident note is the
  justification — L1 compression was refused when the request compiled the raw recent-window room
  traffic alongside the target chunk, the failing contribution was localized to that block as a
  class, and *"handing the summarizer only the thing it is summarizing is the correct scope; the
  copied drain cleared all quarantined chunks first-try and the summaries passed a full fidelity
  audit"* (`types/strategy.ts:966-969`).

  The merge turn is production's **three parts** (`:6867-6891`), selected by `minisystem.source` and
  laid out here:

  1. **PREFIX** — the head window raw, then the prior recall pairs (the unmerged frontier whose span
     starts before the merge range, walked newest-first under `compressionRecallBudgetTokens`,
     `capRecallPairs` `:2489-2514`), then the raw middle; each element skipped when a live summary
     already covers that stretch, so the head cannot leak covered text back in (`:6948-6990`).
  2. **TARGET** — the sources expanded **one level deeper than they themselves are** (`:7076-7148`):
     our L1's only leaf is its chunk, so an L2 merge is shown the raw L0 chunk text the six L1s
     consolidate, and an L3 merge the L1s under each L2, as `[CM] Recall memory <id>.` pairs.
  3. **INSTRUCTION** — `formatMergeInstruction` (`:277-295`) last, with `{seen_description}` derived
     from the layer **actually shown** (`sourceLevelShown`, `:7151-7154`): an L2 merge says "the
     slices of recent experience above (raw conversation)", an L3 merge "the L1 memories above".
     `formatReadingMergeInstruction` (`:340-365`) replaces it when every leaf under the merge is a
     shard of one body group (`:7156-7175`). No tail after the merge range, and no marker:
     production's merge request has none.

  **Section 3 is the one section a session here does not reach.** The raw middle is non-empty only
  when an L1 is taken *out of order*: `work` derives an L1 for every chunk in order, so the chunk
  preceding the current one is either already minted (and so arrives as its recall pair) or is the
  current one. The two paths that take one out of order are the demand path's holdback bypass — an
  escalated plan asks for the newest uncovered run while an older chunk is still unminted — and a
  terminal-debt clear, which makes an older chunk derivable again while a newer one is being
  compressed. The code is the merge's (verified non-empty there by
  `test_merge_prefix_is_the_prior_content_only`); the L1 case is not pinned by a test, and that is
  the honest status.

  The placeholders are filled from the work item and the shown layer: the target level from the work
  id (`L{n}:{payload}`), the description of that layer, and the token target. `merge.txt`'s,
  `reading_merge.txt`'s and `reading_l1.txt`'s trailing comment blocks document those placeholders
  and are not sent. The prompt files are read per call rather than cached: a prompt is configuration,
  and the log-only design has no room for a module-level copy that drifts from the file on disk.
  The request carries `source` and `prefix` as their own fields so they stay inspectable without
  re-parsing the prompt, carries `scope` so which shape was sent is inspectable too, and carries
  `max_tokens` because admission reserves it. None of the three is a field of the dispatched payload,
  so the bound counts each exactly once. Work ids are `L{n}:{payload}`, so `n` is the level the
  memory is minted at, and a merge's sources always sit one level beneath it.

  **What the L1 turn used to be, and why that was wrong.** The system prompt was
  `summarySystemPrompt` and the L1 user turn `summaryUserPrompt` (`types/strategy.ts:1485-1486`) —
  two knobs production *declares and never sends*: a grep for either name across `src/`, `bench/`,
  `test/` and `scripts/` returns the declarations and the default object and nothing else. The
  prompts here are now the strings production actually sends for a mint. The system turn is a
  different matter: production serves the host's **live identity prompt** there, conditionally
  spread (`:5604`) and supplied by `connectome-host/framework-strategy`, and this artifact has no
  live agent identity — `llm/prompts/system.txt` is a stand-in of our own and says so in its own header
  rather than citing a dead knob.
* **`MockModel` / `HttpModel`** — the same protocol, `complete(request) -> Reply`: offline and
  deterministic, or a real `POST` with stdlib `urllib` only.
* **`gate(reply)`** — production's terminal-disposition gate
  (`assessFallbackCompressionResponse`, `autobiographical.ts:2758-2794`), which exists because a
  163-char truncated refusal preamble was once persisted as an L4 parent over six real L3 children.
* **`Summarizer`** — assemble → admit → complete → gate → account. `minisystem.step()` sees exactly
  `(text | None, reason | None)`, and every reason is a key of `config.ACTIONS`, so the model
  cannot invent a failure class the policy cannot route.

**The invariant, restated for the shape that replaced it: a canonical L1 carries the chunk plus
exactly the material that precedes it and is unrepresented — the head window raw, the unmerged
frontier as recall pairs, the raw middle — then the marker, the chunk and the directive, and nothing
after the chunk.** The as-of half is the one that would regress silently, because adding later
material makes a request look richer rather than broken; it is asserted at the system level, from a
request built during a real session whose archive holds five other chunks, as the absence of every
chunk *after* the target in the assembled body. The source-only scope survives as the refusal rung:
production's own note credits it with clearing an entire quarantined backlog first-try
(`types/strategy.ts:946-980`), and that is now its reason to exist rather than its claim to be the
shape. A **merge** is not source-only and production's is not either: its merge request is a prefix,
a target one level deeper and an instruction, because the quality of a consolidation rests on the
material under its sources rather than on summaries of them. What is enforced in one function and
asserted in `test_summarizer.py` for both requests is the same three things — what is shown, what
precedes it, and what the instruction says about it — plus, for the L1, the section order, the
frontier rule and the absent tail.

Outcomes map into the system's vocabulary in one place:

| what came back | reason | what the policy does |
|---|---|---|
| complete stop + usable text + no tool call | — | minted |
| 429 / 408 / 5xx / timeout / connection error | `provider_error`, retryable | transient: no attempt burned, a streak instead |
| any other 4xx (context_length, invalid_request, auth) | `provider_error`, non-retryable | counted against the bounded attempts |
| a tool / function call in the response | `tool_call` | retried once with the no-tools line |
| a truncating stop (`length`, `max_tokens`), or an abort | `truncated` | production's `incomplete`, retried bounded |
| a body that is empty or only whitespace | `empty` | retried bounded |
| a refusal field or stop reason | `refusal` | retried bounded |
| a request that did not fit | `overflow` | the source is halved on the retry |

Retryable and non-retryable come back on the same `provider_error` reason, because they feed
different counters in production: a blip feeds the server-error streak, a deterministic 400 feeds
the bounded attempts (`autobiographical.ts:4268-4320`). Our table has one `provider_error` rung, so
nothing in the system reads the flag yet; it is kept, named on the reply and totalled in
`usage()`, for the day the two diverge.

**`overflow` is decided before the request leaves, and it is decided on a bound, not an
estimate.** `Summarizer.admit` takes `request_input_bound` — production's fail-closed bound
(`compressionRequestInputBoundTokens`, `autobiographical.ts:2636-2649`): the UTF-8 bytes of the
COMPLETE dispatched request (`summarizer.wire_payload`: one system turn, one user turn — `source`
rides the request for inspectability and is already inside the user turn, so serializing the
request dict would count it twice), plus 512 and 128 per message for the role envelopes and special
tokens JSON does not represent. Bytes rather than tokens because every token is at least one byte:
the bound needs no tokenizer and cannot under-count. It is added to the request's own output reserve
(`request["max_tokens"]`, production's `request.config.maxTokens`) and compared with
`compressionContextBudgetTokens`. The **arithmetic** is term for term production's
(`admittedTokens > budgetTokens → admission_rejected`, `:3060-3068`); the **scope is not**, and
`config.py` now says so: production gates the refusal-curve *fallback variants* only and always
attempts the canonical request unchanged ("The canonical request remains unchanged and is always
attempted first; only fallback variants are gated", `types/strategy.ts:1013-1023`;
`compressionRefusalPlan` iterates `variants`, `:3027-3042`). We have one rung and we gate it, so a
request production would have sent is refused here — a deliberate deviation, kept because a
teaching artifact has no operator to widen a budget. Over the budget the boundary returns `overflow`
without dispatching: nothing is sent, nothing is billed, and the mock is not consulted either, and
`Reply.detail` carries both numbers
(`admission: bound 38782 + 1600 reserved = 40382 > budget 20000 (estimate 9466)`), so a request the
estimate would have admitted is visibly refused. That is the point of keeping two numbers: a live
400 `context_length` is still classified as a **non-retryable `provider_error`**, which is what a
provider-side rejection looks like when the budget sits above the provider's real window (the byte
bound itself cannot under-count tokens) — and which the estimate alone would have admitted.

The chars/4 **estimate** (`request_input_tokens`, production's `estimateCompressionRequestTokens`,
`:2630-2634`) is what production calls it: metadata, not accounting. It rides every reply as
`Reply.estimated_tokens` and is totalled in `usage()`, nothing gates on it, and it is the only thing
that gets **calibrated**: `Summarizer.observe` keeps an EMA of provider-reported input over the
estimate (alpha 0.2, band [0.6, 1.8], production's `reportRealInputTokens`, `:8345-8400`) and
applies it to future estimates — never to the bound, because a bound needs no calibrating. Pairing
here is structural (the estimate belongs to the very request whose reply carries the usage), so
production's arm-once-per-compile trap, which exists because a turn's later calls answer a request
that has since grown (`:8351-8357`), cannot arise. Production's multiplier targets the *window*
estimator, fed by the session's main-model usage; this artifact has no such lane, so ours stays on
the compression request alone — `config.py` records that as deliberate.

The three numbers behind that line, all demo-scaled and all inspectable from `config.py`:
`summaryTargetTokens` 200 (production's 2000 ÷10), the request ceiling `max(1600, target × 1.5)`
— production's expression with its 16000 floor scaled ÷10, so as at production scale the **floor**
is the term that binds — and the budget 20000 (production's 200k ÷10). So the bound's room is
`20000 - 1600 = 18400` bytes, a source of 16 418 ASCII characters (measured at the line: admitted
total exactly 20 000; the rest of the bound is production's marker and directive, which are not
scaled). The reserve is 1600 = 8% of the budget, which is production's share exactly
(16000 of 200000), and 8× the target in both.

**The merge request is the one that now tests that bound**, because the target expansion is what
made it large: the demo's L2 merge goes from a 706-character source (six L1 digests) to 11 694
characters of raw L0, so its bound goes 2263 → 13 399 bytes and its estimate 337 → 3092. Admitted,
with 5001 bytes of the budget left — and that is the honest state of it rather than a number to
tune: an L2 merge now carries the conversation its sources stand for, so a merge over six shards of
one long document is the shape that would not fit. Production's answers are the recall budget that
caps the *prefix* (`capRecallPairs`) and the per-attempt halving of that budget (`:7017-7020`),
neither of which bounds the target; its shards are slices of one message, where each of ours
carries the whole text (the `bodyGroup sharding` row in `config.py`), so a document-heavy merge is
bigger here than it would be there. The check still fires on a request that could not fit rather
than on one that is merely large: the demo's own merge is admitted, and nothing in its session is
refused by admission.

### Running it against a real endpoint

`python3 demo.py` stays offline on `MockModel`, because its demo story is scripted. The
client is exercised by the tests through an injected transport, and in a session the boundary is
three lines:

```python
from summarizer import HttpModel, Summarizer
compress = Summarizer(HttpModel(model="qwen2.5:7b", base_url="http://localhost:11434/v1"))
text, reason = compress("L1", "L1:c7", chunk_text)
```

`HttpModel` speaks one dialect: OpenAI-compatible `POST {base_url}/chat/completions`, which covers
OpenAI, DeepSeek, vLLM, Ollama (`http://localhost:11434/v1`) and llama.cpp server
(`http://localhost:8080/v1`); Anthropic's `/v1/messages` is a change to `build_wire`/`parse` alone,
and is not wired. `CONNECTOME_MODEL` / `CONNECTOME_BASE_URL` / `CONNECTOME_API_KEY` /
`CONNECTOME_TIMEOUT` (seconds, default 120) configure it, and the API key is optional so a local
server that wants none works as-is.

Streaming is not modelled — one call is one request and one response — and no cancellation hook
ships with it: without streaming a cancel cannot interrupt a request already on the wire, so rather
than offer one that can only check before dispatch and after the reply, the boundary offers none. A
caller that wants a stale result thrown away compares its own version stamp, as `minisystem.step()`
does.

### What the mock does, and the tests

`MockModel(plan)` maps a work id's payload to a queue of outcomes — any class in the table above,
or a hand-built `Reply` for anything the vocabulary does not name — and otherwise succeeds with a
deterministic 40-word digest of the source. That digest is deliberately fixed-size rather than
target-sized: the demo's recall pairs are priced by word count, so making it target-sized would
rewrite every token number `minisystem.py` prints. Its usage numbers are estimated at 4 chars per
token; the real client reports what the provider billed. Admission runs ahead of the model, so a
scripted `overflow` and a refused oversized request take the same route into the policy table.

```
python3 test_summarizer.py     # 371 checks, stdlib only, no network, exit 1 on failure
```

The suite disables `urllib.request.urlopen` at import, so a test that reached for a provider would
fail loudly rather than pass slowly: `HttpModel` is only ever driven by an injected fake transport
returning canned bodies.

## `config.py` — knobs and the audit

Every knob the other two files use lives here, documented against production's default with a
source citation and tagged with its audit status:

```
python3 config.py            # the whole table, grouped by area
python3 config.py --gaps     # only what differs: MISSING / deviation / simplified / unmodelled

implemented:25  simplified:19  MISSING:0  deviation:3  unmodelled:4  ignored:0
```

(there is no `MISSING` row left: the wide-span merge guard was the last one, and the
compressible zone — the head/tail/pin boundary applied to the derivation — has a row of its own.
`PRIORITY` lists what is still open and explains the rest.)

The vocabulary is a claim about the production *default*, so `config.py` refuses to contradict
itself: `ignored` says "default-OFF, there is nothing to implement", `unmodelled` says "default-ON
in production, deliberately not modelled here", `MISSING` says "default-ON and absent". `ignored:0`
above is the audit's D1 correction — five rows used to carry it while production has them **on**
(`attachmentsIgnoreSize` is a live pricing rule at `autobiographical.ts:10013` and `:10112`, the
level budgets are set whenever `hierarchical` is on, `compressionRecallBudgetTokens` defaults to
100k, the write-lane cache markers are set unconditionally, and the host names the participant).
`audit_status_conflicts()` is the check, `report()` calls it, and a contradictory table makes
`python3 config.py` exit non-zero instead of printing a falsehood.

`PRIORITY` at the bottom is the close-gap order: per-section request participants, durable
failure receipts, and the drain-to-empty tick. Two entries used to be on that list and are closed. A merge request's view
depth: the row reads `implemented`, and `compressionRecallBudgetTokens` moved with it, from
`unmodelled` to `simplified`, because a request that carries a prefix is a request with a ladder to
cap. And the L1's request shape: `request shape` used to read `deviation` — "source-only L1,
experience replay" — and now reads `implemented`, because the L1 carries production's six sections
and the source-only shape is the refusal rung beneath them. What replaces it on the list is the
narrower carve-out the row itself names: production's per-section participant turns, which this
boundary flattens into one user turn.

Keeping the audit as data rather than prose here is deliberate — the table cannot drift from the
code it describes, and all three of the other files import their constants from it.

### The tick, and why the demo cannot show a merge

`tick()` carries production's two priorities in one call -- compress one chunk (`:4214`), then
execute one merge (`:4238`) -- re-deriving the work set between them, because production enqueues
the merge during the compress block and executes it in the same tick. Taking one item in total is
what starved consolidation here: L1 work always sorts first, new chunks keep producing it, and merge
candidacy needs six unmerged L1s.

The demo still does not exhibit a merge, and that is arithmetic rather than a wiring problem:
measured across phase-1 lengths of 8-16 turns and windows of 2300-5000, the session mints at most
**5** unmerged L1s -- six compressible chunks against a scripted failure load that spends calls on
`c3`-`c6`. The merge request shape and the wide-span guard are therefore exercised by fixtures, not
by the shipped run; giving the demo a merge needs more compressible content or a lighter failure
script, which is a demo-content change rather than a systems one.

### What stays irreducible

The demo shows it. `provider_error` stays transient for the first 11 hits (no attempt burned, no
work lost), because *something* must bound retries without punishing a blip. `overflow` still
needs a reshape. And in phase 2 cutting the window to 1200 is refused outright —
`turn 8: 2296 tokens > wall 1224` — because once terminal debt sits in the window the fold floor is
above the wall. That part is irreducible; what is new is that the debt itself is not. An escalated
plan demands L1s for the uncovered runs, and a quarantined span answers that demand only after a
clear — explicit (`clear_debt`) or automatic (a later mint of the same span).

## Where this still differs from the real solver

(`python3 config.py --gaps` is the itemized, cited version of this section.)

**Inputs differ, so no run is reproducible against production.** `salience` here is assigned
synthetically; the real one is computed per message from composition (`:8939`) and clamped to
`[0.2, 1]`. Recall token counts come from `connectome_min.py`'s own `summarize` (a word-truncating
stand-in), not a model call priced through `recallPairCost` (`live-strip` carrier policy in DSH).
Chunk boundaries are synthetic: no running-sum closure, no tool_use pairing guard, no
`l1HoldbackChunks`, and an L1 appears the instant its chunk closes rather than after a tick's LLM
call — so level *availability over time* differs even when the solve is identical.

**Deliberate omissions, each with a condition under which behavior does diverge:**

* `project_to_valid_cut` (`:600`) and the two unanimity exemptions besides boundary-cut
  tolerance: equivalent for the nested trees this toy builds, not for a damaged (non-nested)
  store, which is the only place they fire.
* Prepared-window goals (`goalTotalTokens` / `goalTargetTokens`) and `strictReach` (`13.7`):
  inactive by default. With them the solver aims at a future window, `foldAt` stops equalling
  `W`, and `made_progress` / dead-band tests differ from this toy's W-based ones; `blocked`
  results (`reach-floor`, `target-floor`) have no analogue here.
* Pins (classic, `pinLevel`, `pinMaxLevel`) and locked chunks: absent. Not defaults anywhere,
  but a host can lock a chunk at runtime and then plans diverge.
* Merge cascade timing: one merge per mint instead of the recursive fixpoint over a persisted
  queue. Same steady state, lagging availability on long runs.
* `fold_depth_cap` clamps to `MAX_FOLD_LEVEL` where the real one clamps to
  `maxAvailableLevel(tree)`. No observable effect (a level with no summary can't be folded to),
  but it is a difference in the input value.
* The merge request's two retry-only shapes: the refusal-fallback payload (the sources shown as
  their own recall pairs, with the instruction describing *them* rather than the layer beneath
  them, `autobiographical.ts:7095-7125`, `:7191-7192`) and the per-attempt halving of the recall
  budget (`:7017-7020`). A refused or oversized merge here retries the canonical shape, so a
  deterministically-refusing merge is retried as the same bytes and then quarantined, where
  production's retry is a genuinely smaller request. The L1 is no longer in that position: it has
  the source-only rung, so a refused L1 is retried as a smaller request too.
* The L1 prefix's per-section participant turns. Production pushes the head and the raw middle as
  the original participants, the recall pairs as `Context Manager` / the agent, and the marker and
  directive as `Context Manager` (`:5392-5504`) — nine-odd API messages carrying the same sections
  this boundary joins into one user turn with blank lines between them. The section order, text and
  boundaries are production's; the envelope is not.
* The rest of the L1 refusal ladder: the three recall-curve variants
  (`buildRecallCurveVariants`, `:2893-3008`) and the split-stitch fallback (`:992`, `:6186-6191`).
  Both need a provider that actually refuses to be exercised, and an unreachable rung cannot be
  checked. What is here is the canonical shape and the source-only rung production's own incident
  data credits (`types/strategy.ts:966-969`).
* The L1 raw middle (section 3) is built and correct but not *reached* by a normal session here: it
  is non-empty only when an L1 is taken out of order (the demand path's holdback bypass, or a clear
  that makes an older chunk derivable while a newer one is compressed). The merge's non-empty raw
  middle is pinned; the L1's is not.
* The reading-mode **L1** instruction was on this list and is closed: an L1 over a document shard
  now asks what reading was like (`llm/prompts/reading_l1.txt`, `formatReadingChunkInstruction`,
  `:250-266`) rather than for an impersonal memory of events that never happened. The merge-side
  variant was already implemented; the chunk-side one, named by the previous round as the last
  unbuilt prompt, is now both implemented and pinned.

Two cuts that *did* change behavior were fixed rather than documented: phase C (`pack`) was
missing, which silently made the cut one-directional and left headroom unspent, and the
misallocation sum skipped cap-`0` chunks where the real `relevanceLoss` counts them. Both are
now ported.

**What is faithful:** the cascade order and conditions under default configuration, the
priority order and group atomicity of the ideal cut, phases A/B/C including accept-if-closer,
the suffix binary search, `kvCost`, the `foldDepthCap` formula, boundary-cut tolerance, and the
misallocation formula.

The only way to *prove* the remaining deltas don't matter is to run both solvers on identical
state and diff the plans — feed `planControlledFrontier` the same `PickerInputs`, `SummaryTree`,
`F_prev`, `W` and `P`, and compare `resolutions`, `tokens`, `perturbation` and `override`. That
harness does not exist yet.

## Appendix: rationale moved out of the code (second minimization pass)

`minisystem.py`, `llm/summarizer.py` and `connectome_min.py` now carry a 1-3 line docstring at each call
site: what it does, and the one non-obvious reason it exists. Everything that used to be spelled out
there is below, per function and with its production citations -- nothing was dropped, only relocated.
The narrative sections above (`## minisystem.py`, `## llm/summarizer.py`) state the same rules in reading
order; this appendix is the audit trail for the move, so a reader can check that the code lost no claim.
An inline comment that was rewritten rather than moved is not listed here, because its text survives in
the code.

### Prose moved out of `minisystem.py`

**module docstring**

  Minimal Connectome *system*: the primitives, plus the default-on rules we were missing.
      python3 demo.py
  `connectome_min.py` is the POLICY (which level each chunk renders at) and is untouched by this
  file. This is the system around it: one log, work that is DERIVED (never queued), one model
  boundary scoped to what each request is SHOWN (`llm/summarizer.py`: request assembly, client,
  terminal-disposition gate), one failure policy (a table), one protection type, one version compare
  -- plus the default-on accounting/rendering rules from `config.py --gaps`:
      salience              computed per message from composition (code fences, tool payloads,
                            images, bare links) -> the coefficient on information loss
      recall pricing        a recall pair costs its question label and its content
      maxMessageTokens      one message cannot dominate: the content is TRUNCATED and the emitted
                            bytes are what get priced, so the plan cannot understate the window
      overBudgetGraceRatio  the enforced wall is W * (1 + grace); crossing it THROWS
      l1HoldbackChunks      the newest closed chunk waits for a newer one before compressing --
                            unless the picker's demand path produced it
      autoTickOnNewMessage  one compression attempt per message, not per turn
      image strip           images past N live / a depth become markers
      bodyGroup sharding     a message > 2x targetChunkTokens splits into shards
  and the six rules that are not accounting, in the order the audits asked for them:
      compressible zone     the chunks the policy can never fold -- head window, tail window, pinned
                            -- are not compressed at all: production feeds its chunker
                            `getCompressibleMessages` (:9967-9980) rather than the whole store, and
                            the boundary is the same one `plan_controlled_frontier` protects
      overlap guard         a chunk whose MESSAGES a live L1 already stands for is not compressed
                            again, by span rather than by chunk id (`_overlapBlocked`, :5185-5203)
      merge guard           a merge candidate is refused when its source span no longer resolves to
                            live messages, or is wide FOR ITS LEVEL (:6654-6688)
      demand path           an escalated plan (even full folding exceeds W) produces L1 requests for
                            the uncovered foldable runs; those chunks bypass the holdback
      coverage              fatal: every chunk renders raw or inside a recall whose leaves cover it
      terminal debt         clearable, explicitly by an event and automatically by a later mint
  and the projection that decides what each request is SHOWN: production's six sections for both -- an
  L1's head, prior recall pairs (the unmerged frontier), raw middle, marker, chunk and instruction, and
  a merge's prefix, target one level deeper and instruction last, with the L1's source-only arm kept
  as the refusal rung (`source`, assembled in `summarizer.build_request`).

**`salience`**

  Port of `computeStaticSalience` (autobiographical.ts:8939-8995): how much of this
      message is payload that lives somewhere else -- fenced code, tool traffic, images, bare
      links -- so folding it loses nothing. Never free: floors at 0.2.

**`strip_images`**

  Images past `maxLiveImages` newest, or deeper than `imageStripDepthTokens`, become a
      marker. Production computes every budget site on the post-strip estimate, so we do too.

**`truncate_blocks`**

  Port of `truncateContent` (autobiographical.ts:11096-11135): `maxMessageTokens` truncates the
      content that is EMITTED, and the emitted bytes are what get priced -- production does both (the
      seven calls behind `msgCap > 0` at :5041, :7613, :8154, :8172, :9148, :9505, :9574, the price
      at :9148). Doing only the price is how a plan comes to stand for a window it does not render,
      so here the cut happens BEFORE anything prices the message and `spans` prices the cut blocks.
      Non-text blocks are kept: production's tool_result branch says they must always be included,
      and a dropped tool block would break the pairing the chunk closed on.

**`spans`**

  The chunker: a chunk closes at targetChunkTokens; a sharded message closes whatever is
      pending first, as a partial chunk, because the alternative is dropping those messages from
      every chunk, silently. A message larger
      than 2x that target shards into pieces sharing a bodyGroupId (production renders the shards
      as ONE message; we approximate with adjacent per-shard units -- see config.py).

**`archive`**

  Projection: chunks and summaries from events alone. Recalls are PRICED as production
      prices them -- the question label plus the content (`recallPairCost`, :9006-9061). The header
      a positioned pair EMITS is not part of that price (:9468 vs :9007), and the +50 per summary
      belongs to the prompt-side cap (`capRecallPairs`, :2489). `children` is the mint event's own
      child list, which is the layer a merge expands one level deeper (`source`).

**`fails_since_clear`**

  One unit's failures since its last clear event. Every bit of the debt vocabulary reads
      this, so a clear cannot leave a stale count or a stale reshape decision behind.

**`clear_debt`**

  Production's escape hatch (`clearCompressionRefusalQuarantine`, :1263-1306), as one event:
      the quarantine map production mutates is here a projection over events, so clearing a unit
      means SAYING SO in the log rather than editing a structure the projection derives. Trigger:
      a caller judges the span worth another attempt -- an operator, or a caller who has changed the
      request shape. Nothing else about the unit changes: the next attempt is canonical again.

**`cleared`**

  Both of production's clears as one predicate: the explicit event above, and the automatic
      one -- a span that mints under a later request shape pays off the debt its earlier shapes
      recorded (`clearQuarantineForCompressedChunk`, :3441-3459, called on every accepted L1 at
      :6459 and :6530).

**`chunk_spans`**

  Every chunk's span, from the same projection that built the chunks: the message range it
      covers plus an ordinal that separates the SHARDS of one message. A body-group shard is a slice
      of a message rather than a whole one, so its siblings share its range and only the ordinal tells
      them apart -- production's shards are separate ingress chunks carrying a `shardIndex`
      (:1578-1583). Without the ordinal the shards of one document would cover each other and only
      the first of them could ever be compressed.

**`same_span`**

  Do two chunk spans name the same lived messages? Equal message ranges mean the same message,
      so the ordinal decides (two shards of one message are different slices, not duplicates);
      different ranges overlap when the intervals do.

**`raw_zone`**

  The chunks the POLICY can never fold: the head window, the tail window and every pin. This is
      `connectome_min.plan_controlled_frontier`'s own raw zone -- the same two constants
      (`HEAD_CHUNKS`, `TAIL_TOKENS`), the same chunk projection, the same accumulation -- recomputed
      in the system layer because the frozen policy does not export it. The policy protects these
      chunks from folding; production never even chunks them (:9967-9980). Two computations of one
      boundary is exactly the drift this duplicates to avoid, so `test_summarizer.py` asserts the two
      agree on a live session rather than trusting the copy.

**`l1_spans`**

  The chunk spans each live L1 OWNS, resolved at the moment it was minted. Production stamps a
      summary with the `sourceRange` of its source messages (`:5238`) and keeps it, so coverage stays
      a fact about messages rather than about whatever a later rebuild calls the same position. Our
      mint event records the chunk ids it was handed, and the chunker is a pure projection of the log,
      so the span it covered is `chunk_spans(log[:i])` at the mint's own index: the boundaries as they
      stood when the memory was written. That is what makes the guard below a SPAN test rather than an
      id test -- after a re-derivation that shifts chunk ids (image stripping is enough: `spans` prices
      the post-strip blocks, so appending a message can move an old boundary), the ids no longer say
      which messages already have a memory, and these still do.

**`covered_by_l1`**

  Every chunk a live L1 already stands for, as one projection over the same coverage:
      `{chunkId: (summaryId, exact)}`. Production has three arms and this is all three, in its order:
        * EXACT (`findExactL1`, :5142-5155): a live L1 whose span IS this chunk's span -- the chunk is
          already compressed, adopt it, silently.
        * FULLY COVERED (:5164-5181): every message is inside some L1 under different boundaries --
          dropped rather than re-compressed, with a warning, because "a warning in the log is strictly
          better than a duplicate memory in an agent's head".
        * PARTIAL OVERLAP (:5183-5203): some messages are -- refused outright and added to
          `_overlapBlocked`, loudly, since "with close-then-compress there is NO legitimate way for a
          chunk to partially overlap an existing L1's span".
      Ours is one test for all three: the two arms differ only in whether the covering span is the
      chunk's own, so `exact` is the arm and `overlap_blocked` below is the loud half.

**`overlap_blocked`**

  The loud half: chunks whose messages a live L1 covers under DIFFERENT boundaries, and the
      summary that covers them. Production warns and errors on exactly these (`_overlapBlocked`,
      :5192-5201) and dedupes the line per chunk key per process; here the fact is a projection over
      the log, so it is reportable instead of remembered -- `work` refuses the span either way, and
      the demo prints this list beside the terminal debt rather than leaving a silent drop.

**`span_limit`**

  `spanBase * mergeK ** max(0, level - 3)` (:6678).

**`merge_span`**

  The candidate's source span in MESSAGES, or None when it no longer resolves: production maps
      `sourceRange.first/last` through the live message order and a candidate whose endpoints are gone
      "can NEVER merge" -- it is frontier debt, and it is warned rather than dropped silently (:6660-6667,
      itself a review finding). Ours resolves the same way, through the chunks that still exist.

**`merge_exclusions`**

  `[(summaryId, reason)]` for every candidate the guard removes, with production's own two
      reasons: a span that is wide FOR ITS LEVEL (`wide-span quarantine`, :6680-6686) and one that no
      longer resolves to live messages (`permanently unmergeable, frontier debt`, :6660-6667).
      Production logs one warn per id per process (`warnMergeExclusion`, :6630-6642) because a
      silently-dropped candidate "is a silently-stalled pyramid"; this is that line as a projection,
      over the same UNMERGED candidates `work` selects from -- a node already consolidated into a
      parent is not a candidate and is never warned about.

**`demanded_chunks`**

  kv-stable hands back chunk-id RANGES (`produce` ops, :180-198); the consumer maps a range
      back onto closed chunks (`handleProducedOps` -> `enqueueL1ForRange` -> `_demandedL1Chunks`).
      A range is one coalesced run of uncovered, foldable chunks, so the mapping is the run itself.

**`work`**

  Chunks with no L1 (minus the holdback window), plus a merge per base-run of unmerged
      siblings. The log's only ledger state is failures, so nothing is enqueued or quarantined.
      Two things a chunk must be for an L1 to be derived for it, and neither is an id test:
      `raw_zone` -- the head window, the tail window and the pins, computed by the same expression
      the policy protects them with. Production's chunker is fed `getCompressibleMessages`, so a
      chunk it never folds does not exist there at all (:9967-9980, :10054-10056); here the chunk
      exists (the chunk projection prices the window too) and the derivation is what has to refuse
      it. Compressing one bought an L1 no fold can ever use, and -- because a head chunk with an L1
      is a head section that renders as its recall pair -- an L1 request whose identity anchor was
      empty.
      `covered_by_l1` -- a live L1 already stands for these MESSAGES. The old test was
      `c.id not in a["l1_of"]`, an id test: re-deriving the chunks can shift ids while the messages
      stay put, and then a message gets compressed twice (AUDIT.md's §1.3 row). The span test sees
      that, and the loud half of it is `overlap_blocked`.
      `demanded` is the picker's demand path: the newest closed chunks normally wait, but a chunk a
      `produce` op asked for goes straight into the queue (production does exactly this by marking
      it in `_demandedL1Chunks`, :998, which is what `rebuildChunks` checks at :3992-3999). Demand
      cannot name a raw-zone chunk -- `connectome_min.demand_runs` refuses to produce one for a span
      that can never render a recall -- so this bypass opens the holdback and nothing else.
      A merge takes the first `MERGE_THRESHOLD` candidates that survive `merge_exclusions`: a span
      that no longer resolves, and one that is wide for its level, are both left on the frontier.

**`UncoveredDropError`**

  Production's fatal coverage error, exported as a cross-package surface (`src/index.ts:38`)
      and thrown from `assertFullCoverage` (:4655). Fatal here too, and for the same reason: a plan
      that would lose a stretch of the agent's own history must not be emitted at all.

**`assert_coverage`**

  The coverage invariant, checked on the EMITTED render rather than on the frontier that was
      planned: every chunk is either raw in the window, or covered by the leaves of the recall that
      stands for it. A recall whose leaves do not include the chunk that claims it -- a stale or
      duplicated mint over drifted chunk ids -- is exactly the failure this exists to make loud.

**`Shown`**

  What a request is SHOWN, which is not what its work item names: `target` is the sources
      expanded one level deeper than they are, `prefix` the content that precedes them, and
      `reading_tokens` the size of the whole document when the request is a doc-reading request --
      every leaf under the merge, or the chunk itself for an L1, being a shard of one body group
      (`reading_stretch`). `retry` is the refusal rung: the source-only shape, sections 4-6 only.

**`source`**

  The projection: what the request for this work item is SHOWN, in production's parts.
      BOTH requests are production's six-section shape now (the spec comment at :5266-5294). What
      differs is only what each one's LEAVES are, and production expands each to the same structure:
      PREFIX (sections 1-3) -- the head window raw, then the prior recall pairs, then the raw middle,
      every element of it skipped when a live summary already covers that stretch, so the head cannot
      leak covered text back in (`priorSummaryMessageIds`, :6948-6965, the dedup that fixed the
      525-message merge request; the same rule at :5350-5385 for the head and :5445-5457 for the raw
      middle). For an L1 the prior set is the unmerged frontier that starts BEFORE this chunk
      (:5318-5341), for a merge the one that starts before the merge range (:6921-6946).
      TARGET -- for a merge, the sources expanded ONE LEVEL DEEPER than they themselves are
      (:7076-7148): our L1's only leaf is its chunk, so an L2 merge is shown the raw L0 chunk text the
      six L1s consolidate, and an L3 merge the L1s under each L2 as recall pairs. For an L1, the chunk
      being compressed, which is what :5470-5478 pushes.
      Nothing after the chunk or the merge range is visible -- the same as-of rule for both
      (:5292-5294, :6889-6891), and the reason `prefix` never reaches past `start`.
      `retry` selects production's own `sourceOnly` arm (:5411-5466): sections 1-3 skipped
      structurally and the recall budget zeroed, which is what makes it a rung rather than a shape.
      It is built here rather than in the branch above so that no prefix is computed for a request
      that will not carry one.

**`source.uncovered`**

  Raw, in order: a chunk this request will not show deeper (it is not a leaf) and that no
          summary covers (`headCoveredSkipped`, :6981-6990: ownership wins over the head boundary).

**`recall_memory`**

  One recall pair as production emits it INSIDE a request: the `[CM] Recall memory <id>.`
      question turn, then the memory body (:7052-7061 for the prefix pairs, :7136-7147 for the
      target's). This header is not the live-view template (`recallHeaderTemplate`, types:1504) and
      is not what the fold planner prices (`recallPairCost`) -- it is request bytes, which is what
      the `+50` in the cap below is for (:2489).

**`cap_recalls`**

  `capRecallPairs` (:2489-2514): walk NEWEST-first, keeping each recall pair that still fits
      and `continue`-ing rather than breaking, so one oversized pair cannot hide the smaller siblings
      behind it; the kept set goes back into chronological order. The cap is what stops a ladder that
      grows with the session from overflowing the window -- production hit that at ~chunk 118 of a
      4000-message import. `budget=None` reads the configured one, so a test can patch it the way the
      holdback is patched.
      The cap is the request's own recall budget, so one mechanism serves both requests: an L1 prefix
      carries the same frontier a merge prefix does (:5430-5439).

**`groups_of`**

  The bodyGroupId of each chunk, as a projection over the same `spans` that built them: a
      shard carries `g<seq>`, a whole message carries none. Production keeps this on the message
      (`bodyGroupId`), which is what `detectDocContext` reads and what the reading merge reads off
      each leaf (`:7156-7175`).

**`reading_stretch`**

  Reading mode (detected at :7156-7175, instruction at :340-365): when EVERY leaf under the
      merge is a shard of the same body group, the stretch was the agent reading one long document,
      and the merge asks what the reading was like instead of forcing an impersonal consolidation --
      the drift into the document author's voice that instruction exists to stop. Production's
      `totalTokens` is the WHOLE group, every message carrying that bodyGroupId, not just the leaves
      in this merge; ours is `spans`' per-shard group tag summed over the chunks that carry it.

**`reading_chunk_tokens`**

  `detectDocContext` (:9716-9757): the SAME body-group test the reading merge makes, plus the
      two guards production's L1 site adds that the merge site does not. The chunk must be one shard
      of a larger message -- production walks the chunk's messages and returns null unless every one
      shares the group (:9728-9735), which in our chunk model means a chunk `spans` tagged as a shard
      at all, since an aggregate of separate messages is never one -- and the whole group must be at
      least twice this chunk (:9746-9752), or the chunk effectively IS the whole message and the
      standard instruction is the right one.
      Recorded rather than silently harmonized: the merge path (:7166) applies neither guard. The
      2x floor is what the reading L1 instruction's own words assert when it calls the piece
      "substantially larger", so it is the L1 site that needs it.

**`step`**

  One attempt: the boundary assembles the request (scoped to what this work item is shown) and
      gates the reply, so all this side sees is `(text | None, reason | None)`; one stale-version
      rule, nothing about providers.
      The request SHAPE is derived from the unit's own failure history, exactly as the reshape below
      is: a refusal moves the L1 to the source-only rung (:5466-5489), production's last rung and the
      one every shape after it in its ladder is a variant of. That is the ladder production documents
      as spendable-or-not per call, and it is derived here for the same reason nothing else in this
      file is queued.

**`Refused`**

  The enforced wall: W * (1 + overBudgetGraceRatio). Production raises OverBudgetError.
      Carries the plan that crossed it, and the demand that plan produced, so a refused turn can
      still be explained: the solve runs to completion before the emission refuses.

**`run`**

  Session loop. One compression call per MESSAGE (autoTickOnNewMessage), the display
      re-planned after each turn, and the wall enforced after every plan. Returns the last plan.
      The plan is not only a frontier: when it ESCALATES -- the ideal cut itself is over W, so no
      folding can fit -- it also produces L1 requests for the uncovered foldable runs, and the NEXT
      turn's ticks derive those chunks past the holdback (kv-stable :180-198 -> `_demandedL1Chunks`).
      Demand is recorded before the wall check, on purpose: the solve produces whether or not the
      emission that follows is refused.

**`tick`**

  One tick, carrying production's TWO priorities in one call -- compress one chunk (:4214),
      then execute one merge (:4238). Both blocks run per tick, and the work set is re-derived
      between them because production ENQUEUES the merge during the compress block
      (`checkMergeThreshold` after a successful compression) and then executes it in the same tick.
      Taking one item in total instead is what starved consolidation here: L1 work always sorts
      first, new chunks keep producing it, and merge candidacy needs six unmerged L1s -- so a
      session could run to its wall without ever consolidating. Under production's defaults there
      is no speculative cap on unmerged L1s (no `maxSpeculativeL1s`), so nothing else gates the
      compress block; the two-priority tick alone is what keeps a backlog from starving merges.
      `demanded` is the picker's produce list from the last plan; it opens the holdback window and
      changes nothing else.

**`content`**

  Synthetic messages with real shape, so salience, image stripping, the message cap and
      sharding all bite: conversation text, tool traffic, fenced code, a bare link and images.

### Prose moved out of `llm/summarizer.py`

**module docstring**

  `minisystem.step()` sees only `(text | None, reason | None)`, every reason being a key of
  `config.ACTIONS`; citations and capabilities not shipped are in README.md and config.py.

**`prompt_text`**

  Read a prompt file and fill its placeholders. Comment lines document them for us, not
      for the model, so they are dropped.

**`l1_request`**

  production's L1 request, section for section (:5266-5294): `prefix` (1-3), the marker (4),
      the chunk raw (5), then the instruction (6) -- with no tail after the chunk (:5292-5294), and
      with the instruction LAST so it directs the turn it follows.
      The prefix carries sections 1-3 in order -- head, prior recall pairs (the unmerged frontier,
      source order, capped newest-first), raw middle -- assembled by `minisystem.source`, which is the
      same code the merge request's prefix uses (:5430-5439 vs :6921-7074; the section comment calls
      one "prior L1 recall pairs" and the other "prior summaries", but both walk the merging-excluded
      frontier). It is empty for the source-only retry rung, which skips those three sections
      structurally (:5411-5466).
      One shape difference, and it is presentation rather than content: production pushes the marker
      as its own participant turn and we have a single user turn (the merge request makes the same
      trade, and every nine-section assembly here joins with a blank line). The marker therefore leads
      the user turn when there is no prefix, and follows the prefix joined the same way when there is
      one -- the same relative order either way, and still the last thing before the chunk.
      Section 6 is doc-aware: a chunk that is one shard of a substantially larger message gets
      `formatReadingChunkInstruction` (:250-266) rather than `formatInstruction` (:192-206), so the
      model is asked what reading was like instead of being pushed to narrate a document it did not
      experience (the mechanism that puts the document author's voice in the agent's mouth).

**`build_request`**

  THE one place a request is assembled, for L1 and merge alike. `source` and `prefix` ride it
      so the request stays inspectable, `scope` says which shape was sent, and `max_tokens` rides it
      because admission reserves the ceiling -- none of them is serialized twice (see `wire_payload`).
      The L1 turn is production's six-section request (see `l1_request`): head, prior recall pairs,
      raw middle, marker, chunk, instruction -- the default shape, and the one production compresses
      an L1 with today. `source_only` selects production's `compressionSourceOnly` arm instead
      (:5411-5466, doc at types/strategy.ts:966-980): the marker, the chunk and the directive, which
      is sections 4-6 of that builder by construction. Production keeps it as a rung -- the last one
      before the split fallback, sent when the canonical shape was refused -- and so do we.
      The merge turn is production's three-part request (:6867-6891, assembled by
      `minisystem.source`): the PREFIX, the TARGET (the sources expanded one level deeper than they
      are), then the INSTRUCTION last, with no tail after the merge range. `{seen_description}` is
      derived from the layer actually SHOWN, not from the sources (`sourceLevelShown`, :277-295,
      :7151-7154): an L2 merge over L1s is shown raw L0, an L3 merge over L2s the L1s beneath them.
      When every leaf under the merge is a shard of one body group, production's reading-mode
      instruction replaces the consolidation one (:340-365, detected at :7156-7175). There is no
      marker: production's merge request has none.

**`wire_payload`**

  The JSON object a client actually dispatches for this request: one system turn, one user
      turn. None of `source`, `prefix` or `scope` is a field of it -- they ride the request for
      inspectability and the first two are already inside the user turn, so counting the request dict
      would count them twice.

**`wire_body`**

  The exact bytes of that payload, UTF-8 and unescaped -- one serializer, used by the client
      and by the bound, so neither can silently fall behind the other (production serializes the same
      way: `JSON.stringify` plus `Buffer.byteLength`).

**`request_input_bound`**

  Admission's fail-closed bound, production's term for term (:2636-2649): the COMPLETE
      dispatched request as UTF-8 bytes, plus a fixed reserve and one per message for the role
      envelopes and special tokens JSON does not represent. Bytes, not tokens: every token is at
      least one byte, so this needs no tokenizer and cannot under-count.

**`request_input_tokens`**

  The request's input as an ESTIMATE, chars/4 -- metadata, not accounting, which is how
      production labels its own estimator (:2630-2634). `request_input_bound` is what gates.

**`Reply`**

  One normalized reply, before the gate judges it. `error` is a failure that happened before
      any disposition existed -- transport, or admission -- and is already a reason the policy routes.

**`gate`**

  The terminal-disposition gate: a complete stop with usable text and no tool call, or a
      failure whose reason the policy routes. The ordering is load-bearing (config.py).

**`mock_reply`**

      Usage is reported only where a generation happened, which is what the demo's totals count.

**`MockModel`**

  Deterministic offline stand-in. `plan` maps a work id's payload to a queue of outcomes -- a
      class above, or a hand-built `Reply` -- consumed one per call; an exhausted queue succeeds.

**`urlopen_transport`**

  The client's dependency surface: an HTTPError is a normal outcome, timeouts and connection
      errors propagate.

**`HttpModel`**

  One POST to an OpenAI-compatible endpoint, stdlib only. Every outcome maps into the failure
      vocabulary the rest of the system already routes: README.md's table, config.py's "model boundary".

**`HttpModel.build_wire`**

  The exact bytes, as a seam a test can assert without a provider.

**`HttpModel.parse`**

  A malformed 2xx body is a provider_error, as production's `malformed_response` is (:2766).

**`Summarizer`**

      `__call__(kind, target, source, prefix, reading_tokens)` is the interface -- the four fields of
      a `minisystem.Shown` -- and returns `(text, None)` on a memory, else `(None, reason)`.

**`Summarizer.estimate`**

  The chars/4 estimate in calibrated units: metadata, never a gate. Calibration rides here
          and nowhere near `request_input_bound` -- a bound needs no calibrating, which is its point.

**`Summarizer.observe`**

  Closed-loop calibration (reportRealInputTokens, :8345-8400): one sample per dispatched
          call, paired with ITS OWN request's estimate. The pairing is structural here -- we build the
          request and then read that request's reply -- so production's arm-once-per-compile trap (a
          turn's later calls answer a request that has since grown, :8351-8357) cannot arise. An
          out-of-band ratio is a structural mismatch rather than evidence: drop it, do not learn.

**`Summarizer.admit`**

  Pre-send admission: production's `bound + output reserve > budget -> admission_rejected`
          (:3060-3068), the reserve being the request's own ceiling. The input side is the fail-closed
          BYTE bound, never the estimate: an estimate that is too low must not be able to admit a
          request the provider would refuse. The refusal's detail carries both numbers.
          A merge request is much larger than an L1's -- it carries the layer BENEATH its sources --
          so this is the check that can now refuse a merge. Production's own answer is the recall
          budget that caps the prefix (`capRecallPairs`, :2489) and its per-attempt halving
          (:7017-7020), not a raised budget.

### Prose moved out of `connectome_min.py`

**module docstring**

  Mirrors adaptive/kv-control.ts (design doc 13.4), plus kv-stable's demand-side production: an
  ESCALATED plan -- even the ideal cut is over W -- also carries `produced`, one coalesced run per
  contiguous stretch of uncovered foldable chunks, which the system consumes as a holdback bypass
  (`plan_controlled_frontier` -> `demand_runs`). Everything else about the plan is unchanged, and the
  four documented runs print byte-identically: this file's own replay mints an L1 with every chunk,
  so no chunk is ever uncovered here. README.md maps code to source and lists what this still differs
  on. Deliberately omitted: project_to_valid_cut and the pins/locked/prepared-window machinery.

**`demand_runs`**

  When even the IDEAL cut is over W (`plan.escalated`, kv-control.ts:925) folding has nothing
      left to give, and the only lever is to produce L1s for the chunks that have none, so a later
      solve can fold them. One request per CONTIGUOUS run of uncovered foldable chunks, in sequence
      order -- coalesced exactly as kv-stable does, so re-emission on later escalated plans is
      convergent. A chunk already carrying an L1 is covered rather than uncovered; a raw-zone chunk
      (head, tail, pinned) can never render a recall, so producing one for it would be wasted work.
      Locked chunks (`frozen` in production, `:193`) have no analogue here. Returns [] unless the
      plan escalated, and the consumer is minisystem.work(), which lets these bypass the holdback.

### Comment blocks moved out of `minisystem.py`

Two standalone comment blocks were deleted at their call sites because README.md's `## minisystem.py`
already carried them; they are reproduced here so the move is auditable rather than asserted.

**above `chunk_spans`**

  Production does not chunk the whole store. `rebuildChunks` is handed `getCompressibleMessages`
  (:10054-10056), which is `[0, headStart) U [headEnd, recentStart)` minus every pinned position
  (:9967-9980) -- so a head-window, recent-window or pinned message is never inside a chunk, never
  compressed, and never has an L1 to fold. Our chunker runs over every message (the chunk projection
  is also what prices the window), so the same boundary has to be applied to the DERIVATION instead.

**above `span_limit`**

  The two candidate filters of `contiguousMergeCandidates` (:6654-6688), on our chunk spans. The level
  scaling is the load-bearing half: "Legitimate spans grow ~k× per level, so a FLAT limit silently
  forbids all consolidation above the level where it matches a healthy node's span" -- the mythos
  store's L4s spanned 3.0k-6.9k messages, so an L5 was structurally impossible and the fold floor sat
  ~23k above where one L5 puts it. Limits for L1-L3 are the base; L4 gets base×k, L5 base×k², ... so a
  healthy pyramid is always reachable one level up.
