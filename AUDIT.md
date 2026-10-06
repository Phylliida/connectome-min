# Independent audit: `connectome-min` vs `@animalabs/context-manager` v0.10.1

Auditor: fresh session, no prior exposure to this artifact's self-assessment.
Method: production derived first (public API → strategy → adaptive layer → docs → host
overrides), then the artifact read *and run*, then a diff against `config.py`.

Read-only. No file other than this one was created or modified.

**Path conventions.** `src/…`, `docs/…`, `scripts/…` are relative to
`ref/context-manager`. `HOST:n` = `ref/connectome-host/src/framework-strategy.ts:n`.
`port:n` = `packages/compaction/compaction-autobiographical/src/index.ts:n`. Ours =
`tmp/connectome-min/…`.

**The default baseline used throughout** (library defaults ∪ constructor conditional
defaults ∪ host overrides):

| layer | source | what it sets |
|---|---|---|
| library | `src/types/strategy.ts:1478-1509` | `targetChunkTokens 3000`, `recentWindowTokens 30000`, `headWindowTokens 0`, `maxMessageTokens 0`, `maxLiveImages 6`, `imageStripDepthTokens 30000`, `positionedRecallPairs true`, `recallHeaderTemplate '[Recall {id}]'`, `carrierPolicy 'full'`, `recallEnvelope 'none'`, `compressionRefusalCurveFallbacks 3`, `compressionContextBudgetTokens 200000`, `overBudgetGraceRatio 0.02`, `autoTickOnNewMessage false` |
| library | `src/strategies/autobiographical.ts:910-932` (`conditionalLibraryDefaults`) | `hierarchical true`, `mergeThreshold 6`, `summaryTargetTokens 2000`, `l1/l2/l3BudgetTokens 30000`, `compressionCacheMarkers true`, `compressionCacheTtl '1h'` |
| library | `??`-defaults read in code | `mergeMaxSourceSpanMessages 1500` (`:6654`), `mergeAttemptLimit 5` (`:3755`), `l1HoldbackChunks 1` (`:3981`), `minChunkCharsForLLM 200` (`:5216`), `quarantineAlarmIntervalMs 900000` (`:3346`), `maxLiveImageBytes 20 MiB` (`:10399`), `maxCompressionImageBytes 12 MiB` (`:5647`, `:7258`), `compressionRecallBudgetTokens 100000` (`:5416`), `enforceBudget true` (`:9816`) |
| host | `HOST:82-132` | `headWindowTokens 4000`, `recentWindowTokens 30000`, `maxMessageTokens 10000`, `autoTickOnNewMessage true`, `adaptiveResolution true` (→ fills `speculativeProduction true`, `compressionSlackRatio 0.1`, `foldingStrategy 'flat-profile'` then overridden), `foldingStrategy 'kv-stable'`, `summaryParticipant <agent name>` |

**Verification runs.** `python3 minisystem.py` (full demo, exit 0); `python3
test_summarizer.py` → `all checks passed`; `python3 config.py --gaps` → `implemented:20
simplified:14 MISSING:1 deviation:5 ignored:5`; `python3 connectome_min.py --window 700`;
plus four inline `python3 -` probes whose results are quoted where they are used. I could
**not** run production: `ref/context-manager/node_modules` does not exist, so every
production claim below is a read of source, never an execution.

---

## 1. The inventory

Status: **present** (the rule is there and behaves like production's) · **partial** (the
rule is there for one case where production has several, or the mechanism differs
materially) · **different** (deliberate divergence, verified) · **absent**.

### 1.1 Compaction — how a long context becomes a short one

| capability | production evidence | default-on? | what it does | ours | our evidence | consequence of the gap |
|---|---|---|---|---|---|---|
| Chunk as the fold unit, closed at `targetChunkTokens`, never partial | `src/strategies/autobiographical.ts:9990` + `:10123` `"currentChunk.length >= 4 &&"` | yes (library 3000; 4-message floor) | a chunk exists only once it closes; the pyramid's leaf is the chunk | present (scale 300/4) | `minisystem.py:97-126`; `python3 minisystem.py` → `chunks=9` for 36 messages | none |
| `tool_use` pairing guard: never close a chunk on a message holding `tool_use` | `:10137-10144`, `:10205` | yes | the matching `tool_result` rides in the next message; a split pair is a provider 400 | present | `minisystem.py:122-123` (`has_tool_use`) | none |
| Unbounded pyramid levels; fold depth capped by what is *built* | `src/types/strategy.ts:1255` (`SummaryLevel = number`), `src/adaptive/kv-control.ts:824` `"const maxFoldLevel = maxAvailableLevel(tree);"` | yes | L4/L5… are reachable as merges produce them | present | `connectome_min.py:21-38`; `ancestor_at` returns `None` when no summary exists | none |
| `mergeThreshold` = 6, applied to **strictly contiguous** unmerged siblings | `:6644`, `:6698` `"if (run.length > 0 && x.first !== runEnd + 1) {"` | yes | a merge group must be adjacent in source order | partial | `minisystem.py:205-208` takes the first 6 unmerged at a level, no adjacency test | a merge can bridge a hole production refuses to bridge, producing a pyramid production would not have |
| Stranded interior runs merge at **2** | `:6718-6721` `"if (!isNewest && r.length >= 2) return r.slice(0, threshold)…"` | yes | anti-starvation: an interior run can never grow, so it consolidates early | absent | same code path as above, no run concept | a stranded run waits for 6 forever; the fold floor climbs exactly as in the mythos starvation incident |
| `mergeMaxSourceSpanMessages` = `1500 × k^(level−3)` | `:6654`, `:6678` | yes | refuses "wide for its level" replay-era nodes, level-scaled so L5 stays reachable | absent | `config.py:159-164` (MISSING); no span check in `work()` | merges spans production quarantines; the level stops meaning what it means |
| `l1HoldbackChunks` = 1 | `:3981` `"const holdback = this.config.l1HoldbackChunks ?? 1;"` | yes | newest closed chunk waits for a newer one to close | present | `minisystem.py:201`; `config.py:394` | none |
| **Demand-side production under escalation** (kv-stable emits `produce` ops) | `src/adaptive/strategies/kv-stable.ts:180-198` `"if (plan.escalated) { … produced.push({ level: 1, range: … }); }"`; holdback bypass `:3981-3985` (`_demandedL1Chunks`) | yes (host selects kv-stable, `HOST:128`) | when even full folding exceeds W and foldable chunks have no L1, demand them — the only remaining lever | absent | a grep for the strings `produce`, `demand` and `escalat` across `connectome_min.py`, `minisystem.py` and `llm/summarizer.py` returns only the demo's label at `connectome_min.py:127`; `:107-118` returns no `produced` list | at the escalation the demo itself reaches (`REFUSED: turn 8: 2190 tokens > wall 1224`), production keeps producing L1s to fold under W; ours stops |
| Speculative production drains to empty (recursive, macrotask-paced) | `:4120-4166` `"setTimeout(() => this.driveSpeculativeDrain(ctx), 0);"` | yes | one tick per message, but the tick loop runs until no progress | partial | `minisystem.py:291-300` — `tick()` `return`s inside `for item in work(log)`, i.e. **one** work item per message | merge debt accumulates behind L1 debt; on a long run availability lags where production's converges |
| The relevance-ideal cut: phases A (shape prior) / B (prior yields to W) / C (pack youngest-first), accept-if-closer | `src/adaptive/kv-control.ts:347-473` | yes | fold cheapest-information-first by log-age band, then pack budget headroom back | present | `connectome_min.py:56-84`; `README.md:324-327` records the two cuts that were fixed | none observable |
| `foldDepthCap` = `floor(log(age/flatZoneChunks)/log(k)) + 1`, clamped | `src/adaptive/kv-control.ts:120-134` | yes | the soft shape prior's per-chunk depth | present | `connectome_min.py:52-55` | none |
| Salience = coefficient on information loss, floor 0.2 | `:8939-8995`, `:8993` `"return Math.max(0.2, 1 - 0.8 * externalized);"` | yes | fenced code / tool traffic / images / bare links fold cheap | present | `minisystem.py:53-73` — I diffed it against production line by line, including the `6400`-char image cost and the `^https?://\S+$` link test | none — **but see D7: the demo cannot exercise it** |
| Trust region P; suffix adoption by binary search on exact `kvCost`; overrides `bootstrap`/`infeasible`/`quality-gap`; dead band with self-heal | `src/adaptive/kv-control.ts:809-994`; `:523-573` | yes | per-turn prefix re-read is bounded; bigger repairs amortize | present | `connectome_min.py:85-118`; the demo prints `bootstrap`/`hold`/`adopt-ideal`/`suffix-adopt`/`override` | none observable |
| `qualityGapRatio` 0.35 | `src/adaptive/kv-control.ts:821` `"const gapRatio = p.qualityGapRatio ?? 0.35;"` | yes | a certifiably misallocated hold self-heals instead of fossilizing | present | `connectome_min.py:108`; `config.py:403` | none |
| `compressionSlackRatio` 0.1 → target = W·0.9 | `src/adaptive/kv-control.ts:820` + `:923-927` | yes | the dead band that makes a cache hit possible | present | `connectome_min.py:96` | none |
| `renderLayout` unit model: a folded chunk costs its `L_k` ancestor's recall **once per distinct ancestor** | `src/adaptive/picker.ts:244-282` (`accountFrontier`) | yes | the picker prices what the renderer emits | present | `connectome_min.py:41-47` (`seen` set) | none |
| `projectToValidCut`: fold → project fixpoint (round cap 3), deepest-first, boundary-cut tolerance (`−1` caps exempt), overlap tolerance | `src/adaptive/kv-control.ts:600-655`, `:392-512`, `:479-484` | yes | a frontier that cuts a group non-unanimously is lowered toward raw | absent | `connectome_min.py:6-7` declares the omission; README:309-311 claims equivalence | **I could not produce a divergence** (see §5) — the omission is unexercised in every probe I ran |
| Picker validation alarms: `[picker-unrealizable]`, `[picker-dead-ids]`, plan/emission drift refusal | `src/adaptive/picker.ts:295-312`; `:7962-7975` | yes | a solver bug is reported loudly, never silently absorbed | absent | no analogue in `connectome_min.py` | a bad plan renders silently; the toy cannot tell a bug from a plan |
| head window raw, never chunked or compressed | `:10489-10507`; `:9967` (`getCompressibleMessages` excludes it) | yes (host 4000 tok) | the identity/preamble survives at full granularity | partial | `connectome_min.py:97` `HEAD_CHUNKS = 1` chunk; `config.py:85-88` | we count chunks, production counts tokens — **and ours still mints an L1 for the head chunk** (`work()` excludes only the holdback), so the archive holds a memory production would never write |
| tail window raw and stable (`recentWindowTokens`) | `:10463-10488` | yes (host 30000) | the thing a fold must not disturb | partial | `TAIL_TOKENS = 400` (`config.py:80-84`) | scale only |
| `maxMessageTokens`: **truncate the emitted content** (`truncateContent`) **and** price it at `cap+50` | `:11096` (`truncateContent`); applied at `:5041`, `:7613`, `:8154`, `:8172`, `:9148`, `:9505`, `:9574` | yes (host 10000) | no single message can dominate the window | partial | `minisystem.py:116-117` caps the *price* only; probe: a 424-word message is priced `333` tokens and its text is never truncated | the plan prices a window smaller than the text it stands for — the fold floor is understated, which is the class of lie production's emission-overflow refusal exists to catch |
| Image stripping: `maxLiveImages 6`, `imageStripDepthTokens 30000`, `maxLiveImageBytes 20 MiB` | `:10289-10420`, `:10399` | yes | images are the first thing worth shedding | partial | `minisystem.py:76-94` (2 / 800 tokens); **no byte wall** | a burst of dense images passes a count+depth test production's byte wall would stop |
| `maxCompressionImageBytes` 12 MiB on the compression prompt | `:5647`, `:7258` | yes | the summarizer prompt gets a tighter image budget than the live window | absent | — (moot while requests are source-only) | none today; returns the moment the request shape reverts |
| Budget every site on the **post-strip** estimate | `:10362-10420` (`postStripEstimates`) | yes | the plan prices the stripped render | present | `minisystem.py:76-94` strips, then `spans()` prices the stripped blocks | none |
| `overBudgetGraceRatio` 0.02, hard `OverBudgetError` beyond W·1.02 | `src/types/strategy.ts:1507`; `:7552` `"const rejectionBudget = Math.floor(maxTokens * (1 + overBudgetGraceRatio));"` | yes | a sub-percent drift does not refuse an ordinary turn | present | `python3 minisystem.py` → `REFUSED: turn 8: 2190 tokens > wall 1224 (W=1200 + 2%)` | none |
| Fatal coverage invariant (`assertFullCoverage` + `assertMiddleCoverage` → `UncoveredDropError`) | `:4655-4791`, `:4576-4595`; exported at `src/index.ts:38` | yes | nothing is *ever* silently dropped from the window | absent | no analogue; `render` falls back to raw when no summary exists | the invariant holds by construction in the toy but is never asserted, so a stale/partial mint that loses a span has no detector |
| `checkReadiness()` / pending work gate on compile | `:4066-4093` | yes | a compile refuses to proceed and says what it is waiting for | absent | `minisystem.run()` plans on every turn unconditionally | a compile can plan against a half-built forest |
| `previewContext()` dry run: no fold persistence, no enqueue, no spend | `:1319-1444`; `src/types/strategy.ts:196-201` | yes under host (`:1413-1415` requires `adaptiveResolution`) | an operator can see a hypothetical budget without mutating state | absent | no analogue | no way to ask "what would W=1200 do" without doing it |
| Prompt-cache seams on the **live** compile lane (`placeCacheMarkers`) | `:8548`, called `:8296` | yes | head / measured stable prefix breakpoints | absent | — | billed cost only; no layout effect |
| Cache markers on the **write** lane (`compressionCacheMarkers true`, TTL `'1h'`, seams at end-of-head / last L≥2 pair / last pair, suppressed on a capped ladder, stale `cache_control` stripped) | `:2548-2596`, `:929-930`, `:2553`, `:2573` | yes | ~60–93% synthetic stable prefix stops being re-sent uncached | absent | `config.py:259-262` calls it `ignored` — see **D1** | billed cost only; the artifact's own pretence (below) is the real issue |
| `getCompressionDebt()` — healthy/degraded/critical from pending age | `:4425-4470` (`DEGRADED_AFTER_MS = 60*60*1000`) | yes | "is the memory organ keeping up" as a first-class number | absent | `minisystem.py:346-354` prints a derived *stalled-work* projection instead | no staleness signal; the artifact cannot say "am I behind" |

### 1.2 Summarization — how memories get written

| capability | production evidence | default-on? | what it does | ours | our evidence | consequence of the gap |
|---|---|---|---|---|---|---|
| The L1 request is **six in-band sections**: head → prior recall ladder → raw middle → compression marker → target chunk raw → directive (and deliberately no tail-after-chunk) | `:5266-5292`; spec `docs/compression-recall-curve-fallback-spec.md:11-18` | yes | the summarizer replays the agent's continuous experience | different (source-only) | `llm/summarizer.py:40-56`; `test_summarizer.py:110,121`; `config.py:185-194` | disclosed as a deviation, honestly; summaries read differently from production's |
| In-band `COMPRESSION_MARKER` primer | `:99-102` `"System: You will soon form a new memory, get ready. The messages that follow are the slice of recent experience you are about to compress…"` | yes | the structural pre-memory cue, never text-searchable | absent | not in `llm/prompts/`, not in `config.py` — see **D8a** | the write is unprimed; the model is asked for a recollection with no marker that this *is* the memory write |
| The **actual** default L1 directive: `formatInstruction(targetTokens)` | `:192-206` `"'Write the memory of events since the most recent memory system notification. Speak in the first person from your own perspective. Preserve concrete details — file paths, exact values, decisions, unresolved questions, the user's active asks…'"` | yes | voice + what to preserve + anti-padding rule | absent | `llm/prompts/l1_chunk.txt` is `summaryUserPrompt`'s text; `README.md:131-132` cites `types/strategy.ts:1486` as its source | **`summarySystemPrompt`/`summaryUserPrompt` have zero read sites in `src/`** (grep below) — the artifact copies two dead knobs and calls them "the library's own" |
| System prompt = the host's **live identity prompt**, conditionally spread | `:5604` `"…(ctx.systemPrompt ? { system: ctx.systemPrompt } : {}),"`; `HOST` never sets it | yes-when-declared | identity comes from the live prompt, never a synthetic summarizer header | different | `llm/summarizer.py:55` always sends `llm/prompts/system.txt` (= the dead `summarySystemPrompt`) | a summarizer header production deliberately does not synthesise |
| Declare the agent's live tools on the request | `:5640` `"tools: ctx.tools,"`; rationale `:5631-5638` | yes (and the call is **deferred** without tools when the chunk has tool blocks or the model is Fable/Mythos, `:5546`) | a tools-less replay of tool history is a deterministic `reasoning_extraction` refusal | different | `llm/summarizer.py:53-55` `tools: None`; `config.py:278-288` discloses it | disclosed; consequence is bounded because we replay no tool history |
| Reading-mode voice for sharded documents: `formatReadingChunkInstruction`, `formatReadingMergeInstruction`, `detectDocContext` | `:250-266`, `:340-365`, `:9721-9731` | yes (fires when a chunk's messages share one `bodyGroupId` and the piece is ≥2× the chunk) | asks "what was reading this like", forcing the agent's own vantage instead of the document author's voice | absent | not mentioned in `config.py`; `minisystem.spans` tags shards with `g{seq}` (`:112-113`) but nothing consumes the tag | a summarizer handed a document shard adopts the document's voice — the exact drift this instruction was written to stop |
| Merge request shows the sources **one level deeper** | `:7112-7148`, `:6867-6891` | yes | an L2 merge sees raw messages under its L1s; an L3 merge sees L1s | different | `llm/summarizer.py:50-52` shows the children themselves; `config.py:195-204` discloses it | disclosed as quality-relevant/​layout-neutral; correct classification |
| `formatMergeInstruction` + `sourceLevelShown` derived from the shown content | `:277-295`, `:9763` `"const sourceLevelShown = sources.length > 0 ? Math.max(0, sources[0].level - 1) : 0;"` | yes | the "seen description" matches what was actually shown | present | `llm/prompts/merge.txt` is production's text verbatim (I diffed it against `:286-294`), `llm/summarizer.py:50-52` | none |
| Degraded merge rung: sources as their own recall pairs after a refusal | `:7095-7122` | yes | a refused merge retries with the material shown differently | absent (we are always on this rung) | same row as above | — |
| Merge recall budget **halves per failed attempt** (`max(8000, configured × 0.5^min(attempts,4))`) | `:7017-7020` | yes | each retry is a genuinely smaller request | absent | not mentioned in `config.py` | a retried merge re-sends the identical bytes; production's retry changes the request |
| Recall pair = Q `[Recall {id}]` + A (answer content), positioned chronologically | `:9445-9502`, `:9846-9852` | yes | each memory sits in its temporal place | present | `minisystem.archive` `:140-141`; `render` `:41-47` | none |
| Recall-pair **price** = `summaryContextLabel` + answer content | `:9006-9061` (`recallPairCost`), `:9029` | yes | what the fold planner weighs | different | `connectome_min.py:140-141` prices header **+** label **+** content; `config.py:117-123` says production prices "header + question label (summaryContextLabel) + content" | see **D6**: ours over-prices every recall relative to the picker's own number, so it folds more than production would |
| `carrierPolicy 'full'`: capture signed reasoning blocks (`responseContent`) and replay them verbatim on every surface; `tokens` is a floor, never replaced by a smaller text estimate | `:891-902`, `:10950-10988`, `:9029`, `src/types/strategy.ts:713-760` | yes | measured anti-refusal duty: a deterministically-refusing mint passed once its recall pairs carried their signed reasoning | absent | mentioned once, inside another knob's cost text (`config.py:123`), with no status of its own | recalls lose the carriers production measured as load-bearing against refusals, and the price of a recall is computed from prose alone |
| `capRecallPairs`: `compressionRecallBudgetTokens` 100000, newest-first keep, `continue`-not-`break` on oversize, `+50` for the `[CM] Recall memory <id>.` wrapper | `:2497-2545`, `:5416`, `:2489` | **yes** (100000) | the recall ladder cannot itself overflow the window | absent | `config.py:237-242` calls it `ignored` — see **D1** | moot while requests are source-only; returns with the request shape |
| Terminal-disposition gate: only a complete `end_turn` + nonempty text may be canonized | `:2758-2794` (`"if (stopReason !== 'end_turn') return { outcome: 'incomplete', stopReason };"`), rationale `:473-486` (the 163-char refusal that became an L4 parent) | yes | a truncated or interrupted generation never becomes a permanent memory | present | `llm/summarizer.py:105-118`; ordering difference (tool call before empty) disclosed at `config.py:317-323` | none; the disclosed ordering difference is real and honestly stated |
| `minChunkCharsForLLM` = 200: **no LLM call**, store a mechanical stub | `:5216-5241` `"if (!hasNonText && substantiveChars < minChunkChars) {"` → `"(A quiet stretch: N messages of routine system traffic …)"`, `tokens: 40` | yes | a silent/bare-system chunk makes the summarizer confabulate ("68 initiations") | absent | not mentioned in `config.py`; no stub path in `minisystem.step` | a quiet stretch costs a call and produces a hallucinated memory |
| Retry rungs: `NO_TOOLS_RETRY_LINE`, `PLAIN_PROSE_RETRY_LINE`, tools-less escalation, carrier-transport degraded retry | `:123-136`, `:5913-5948`, `:7335-7338`, `:819-850` | yes | each rung reshapes rather than repeats | partial | `config.py:248-252` (one invented line, mechanism matches); `minisystem.py:225-228` | three of four production retry rungs have no analogue |
| Refusal-curve fallbacks: 3 single-node recall expansions, coverage-proven, canonical always first | `:3009-3011`, `:2893-3007`, `:2801-2862`; spec `docs/compression-recall-curve-fallback-spec.md` | yes (3) | a refused mint retries at a different already-authored recall resolution | absent | `config.py:362-369` (one rung) — honestly classified as a deviation | a deterministically-refusing request is retried as the same bytes and then quarantined |
| Admission of refusal-curve variants: `admittedTokens ≤ compressionContextBudgetTokens` | `:3065-3068` `"const disposition = admittedTokens > budgetTokens ? 'admission_rejected' : 'provider_attempt';"` | yes (200000) | a variant that cannot fit is receipted, not dispatched | different | `llm/summarizer.py:278-289` gates **every** request this way — see **D3** | ours gates more than production does, and `config.py` describes the production rule wrongly |
| Fail-closed input bound: complete normalized request, UTF-8 bytes, `+512 + 128·messages`; estimate is metadata only | `:2649` `"return Buffer.byteLength(serialized, 'utf8') + 512 + request.messages.length * 128;"`, `:2630-2633` | yes | a bound needs no tokenizer and cannot under-count | present | `llm/summarizer.py:75-87`; `test_summarizer.py:258` (`bound refuses what the estimate admits`) | none — this one is term-for-term |
| Output ceiling `capCompressionTokens(max(16000, target × 1.5))` | `:5628` | yes | a memory is never truncated mid-thought | present (scaled /10) | `llm/summarizer.py:20-22` | none at demo scale |
| `pushSummary` refuses to persist an empty summary at the write boundary | `:2446-2450` | yes | no empty memory can enter the archive | present | `minisystem.py:233` `if text and text.strip():` before the `mint` event is emitted | none |
| Request sanitizers before dispatch: `splitMixedToolMessages`, `stripUnpairedToolBlocks`, empty-block strip | `src/normalize-tool-messages.ts:22-27`; applied `:5517-5526`, `:5590` | yes | a mixed or unpaired tool pair is a provider 400 | absent | nothing in `llm/summarizer.py`; we never emit tool blocks, and the pairing rule lives at chunk-close instead (`minisystem.py:122-123`) | none today; returns with any request that replays tool history |
| `mergeAttemptLimit` 5, persisted counter, durable merge quarantine, `sanitizePersistedMergeQueue`, `setMergedInto` by id against the persisted array | `:3755`, `:3741-3760`, `:3770-3802`, `:3701-3721`, `:3631-3653` | yes | bounded retries, never infinite, never silent, never by stale index | partial | `minisystem.py:166-187` (attempts derived from the log, terminal at 3); `config.py:205-208` | the count survives a restart only because our log is the store; production's queue repair (`sanitizePersistedMergeQueue`) has no analogue |
| `quarantineAlarmIntervalMs` 15 min, repeating until cleared; all-clear fires once; paid-off sweeps | `:3345-3405`, `:3346`, `:3872` | yes | deferred debt keeps announcing itself | partial | `config.py:253-258` (`deviation`, honest) | no repeated alarm; the demo prints debt once at the end |
| Empty-generation quarantine exhaustion; `CHUNK_QUARANTINE_SHAPE_CAP` 3 | `:6421-6441`, `:1032` | yes | a chunk that refuses in every shape goes sticky by hash instead of re-minting a new family every pass | partial | `minisystem.policy` has one terminal rung, no shape cap | a shape-drifting retry loop is unbounded in principle (bounded in practice by the terminal rung) |
| `provenance { stopReason, requestHash, model }` stamped on every accepted summary | `:6493-6501` (L1), `:7468-7472` (merge); `src/types/strategy.ts:1330-1368` | yes | which request authored this memory is auditable | absent | `config.py:183-184` mentions "durable receipts" inside another knob's cost text | no way to attribute a memory to the request that wrote it |
| `previewContext` / `resetHeadWindow` + `generateTransitionSummary` | `:1319`, `:9863`, `:9873-9939` | yes (on demand) | preview a budget; re-anchor the head on a topic change | absent | no analogue in any of the four files | no dry run, no topic transition |
| Calibration EMA (alpha 0.2, band [0.6,1.8]) fed once per compile | `:8345-8400`, `:8366`, `:8377-8379` | yes | the estimate learns from the wire; the bound never takes the multiplier | present but on a different lane | `llm/summarizer.py:262-276`; `config.py:324-347` discloses the lane difference | disclosed honestly; production's multiplier is fed by a host usage event that is not vendored here (see §5) |

### 1.3 Memory — what the archive is for

| capability | production evidence | default-on? | what it does | ours | our evidence | consequence of the gap |
|---|---|---|---|---|---|---|
| Durable archive: summaries, chunk records, merge queue, quarantine, pins, recalibration, resolutions, locks as named Chronicle slots; register cadences; reload + **repair on load** (drop empty summaries, dedupe by id, clear dangling parents and persist the canonical array, un-compress records whose L1 vanished) | `src/strategies/autobiographical.ts:1821-1908` (`registerStates`), `:1910-2088` (`loadPersistedState`), `:1938` `"Drop empty-content summaries (bugged/empty generations from before the production guards)."` | yes | restart is a first-class, self-healing path | partial | `python3 minisystem.py` → `restart: rebuild everything from the log alone / messages=36 chunks=9 mints=8 fails=6 protections=1 events=59`; `minisystem.py:129-151` re-derives from events | ours rebuilds but never *repairs*: a corrupt mint is re-adopted verbatim, where production canonicalizes |
| `migrateChunkRecords`: a store with L1s but no chunk slot predates chunk persistence | `:1749-1778` | yes | old stores keep working | absent | — | schema evolution is unmodelled (acceptable for a toy, invisible here) |
| **Fail-closed orphan latch**: >50% of chunk records resolving to zero live messages halts **all** compression | `:10029-10039` `"FAIL-CLOSED: … Compression halted to prevent duplicate memory formation"` | yes | a chain break must not re-compress lived history | absent | — | the artifact has no equivalent of the failure it is guarding |
| **L1 overlap guard**: a chunk whose messages are already covered by a live L1 is blocked loudly and added to `_overlapBlocked` | `:5185-5198` | yes | duplicate-memory formation blocked | partial | `work()` excludes chunks by id (`c.id not in a["l1_of"]`) — an id test, not a **span** test | after any re-derivation that shifts chunk ids, a message can be compressed twice |
| `findExactL1`: adopt an existing L1 for the same chunk (crash between append and record edit; a sibling instance's mint) | `:2456-2496`, `:5142-5147`, `:5768-5772` | yes | no duplicate memories from cross-instance races | absent | — | no cross-instance story at all |
| **Fatal coverage** (see compaction lens) — a memory-system invariant, not a layout one | `:4655`, `:4576`, `src/adaptive/picker.ts:78-111` | yes | an agent never quietly loses a stretch of its own history | absent | — | no detector for the single worst memory failure |
| Retrieval: `queryMessages`, `queryMessagesByTime` (native index), `queryMessagesByChannel`, `getMessageWindow` (O(window)), `getChannelTokenStats`, `findMessageByExternalId` | `src/context-manager.ts:462-565` | yes | the archive is queryable without recompiling | absent | no search of any kind in `minisystem.py` | the archive is write-only for the agent |
| Memory search: `searchSummaries` ("Suitable for building memory-search agent tools at the framework layer", `:965`), `getSummary`, `getSummariesInRange`, `listSummariesInRange`, `getMaxSummaryLevel` | `src/context-manager.ts:964-1007`; `:2216-2368` | yes | read over what compression already produced | absent | — | no retrieval surface whatsoever |
| Pins: `pinRange`, `markDocument`, `unpin`, `listPins`, `pinnedPositions`, `pinLevelBounds` (finest requirement wins), persisted as a snapshot guarded by `requireBranchMutation` | `src/context-manager.ts:910-961`; `:2144-2213`, `:2379-2443`, `:2090-2098` | API-reachable at defaults (see excluded list) | a span the agent wants kept renders raw in place | partial | `minisystem.py:255-256` + `:247-252`; `connectome_min.py:97` takes `protected` as a range set; the demo prints `protections honored … {'c1': 0, 'c2': 0}` | honored in the plan, but there is no pin API, no persistence beyond a log event, and no level bounds |
| Leveled pins (`pinLevel` / `pinMaxLevel`), clamped to produced depth, group-consistent | `src/adaptive/strategies/kv-stable.ts:108-142` | yes under host (kv-stable) | fix the cut through an `L_k` node, or cap fold depth | absent | `frozen`/`fixedLevels`/`pinCaps` have no analogue in `plan_controlled_frontier` | a host cannot pin a span at a resolution |
| Chunk locking (`lockChunk`/`unlockChunk`, `lockedByAgent`, `autobio:locks` slot) | `:1509-1543`, `:1889` | yes under host (`adaptiveResolution: true`) | a chunk keeps its carried resolution and is never touched | absent | — | no programmatic freeze |
| Branches: `branchAt`, `switchBranch`, `fork`, `currentBranch`, `listBranches`; branch **generation** identity (5-tuple); `clearBranchMirrors`; stale reads/writes fail closed; a compression result crossing a branch boundary is **discarded** | `src/context-manager.ts:567-657`; `src/branch-generation.ts:11-20`; `:3216-3251`, `:5852-5856` | yes | time travel, forks and away-and-back are safe | absent | `minisystem.py:295-296` injects `msg 900_000` mid-call and discards by version stamp (`:230-232`) — a stale-write rule with no branch model behind it; `frontier(log)` reads the last `frontier` event | one log is one history; a fork overwrites the single frontier rather than scoping one |
| `resetHeadWindow` + transition summary; the anchor is recovered on restart by backward scan | `src/context-manager.ts:1022-1046`; `:1642-1651` | yes | the head can be re-anchored to a new topic | absent | — | no topic-transition story |
| Quarantine **and debt** observability: `getCompressionQuarantineStatus`, `getMergeQuarantineStatus`, `clearCompressionRefusalQuarantine`, `clearMergeQuarantine` (generation-tombstoned), `setQuarantineAlarmHandler`, durable-before-external alerts, paid-off sweeps | `:3306-3333`, `:1263-1318`, `:3345-3405`, `:3474-3524` | yes (strategy-level; `getStrategy()`) | quarantine is visible, clearable and never a resting state | partial | `minisystem.py:346-354` prints `stalled work (terminal debt): ['L1:c4']` / `permanent fold floor from it: 234 tokens` | debt is reported but **cannot be cleared**: production's `clearCompressionRefusalQuarantine` has no analogue, so a terminal chunk is terminal forever |
| `getRenderStats`, `getStats`, `getProgressSnapshot`, `stats()` | `:4909-4970`, `:4387-4400`, `:4341-4360`; `src/context-manager.ts:1008`, `:1265` | yes | per-render and per-store health | absent | `minisystem.py:372-375` prints a profile and a pin map | no health surface |
| Cross-session / multi-resident: `namespace`/`isolate`, `viewFilter` (single choke point, explicitly *not* a confidentiality boundary), auxiliary message views + `mergeMessageStoreViews` | `src/context-manager.ts:107-130`, `:207-216`, `:333-360`; `src/message-view.ts` | yes as capability | several residents share a store without sharing a view | absent | — | no multi-resident story |
| Image/blob retention: `BlobManager` extracts base64 into `blob_ref`, LRU resolve at a 3 GiB default, missing blob **throws** | `src/blob-manager.ts:45`, `:144-161`, `:70`, `:193` | yes | images are stored once and referenced; a lost blob is loud | absent | `minisystem.py:44-50` treats an image as a flat 400-char block | no blob store, so no byte accounting and no way to say a blob went missing |
| Repair / audit tooling: `repair-pyramid.ts` (keeper = union of record-backed and coverage-sweep L1s), `scan-corrupt-memories.ts`, `apply-l4-repair.ts`, `repair-fable/mythos-ownership.ts` (hash-pinned, dry-run), `scan/repair-think-echo.ts`, `split-merge-quarantine.ts`, `migrate-llr.ts`, `reopen-test.ts`, `inspect-store.ts`, `drain-autobiographical.ts`, `src/surgery/zero-recall-compression.ts` | `scripts/*.ts:1-50`, `src/surgery/zero-recall-compression.ts:21-53` | shipped, operator-invoked | a live archive can be diagnosed and repaired without losing coverage | absent | — | the failure modes these exist for are unmodelled (fair for a teaching artifact, but it means "one failure policy" is only half the production story) |
| Log-native port: the event payload **is** the archive; restart/crash/fork cost zero inference; chunks deliberately re-derived by the library's own migration | `packages/compaction/compaction-autobiographical/README.md:11-17`, `src/seed.ts:12-24` | shipped | the archive is the log | present | `minisystem.py:31-37` (`emit`), `:129-151` (`archive` from events), `:367-370` (restart from the log alone) | none — this is the design the artifact gets right |
| `getAntiRedundantSummaries` (drop a parent whose children are all visible), `selectL1Summaries`, `sortSummariesChronologically` | `:5065-5106`, `:9792-9845` | legacy hierarchical path only — `getAntiRedundantSummaries` is called **only** from `selectHierarchical` (`:9193`) | no duplicate representation in the window | n/a | probe: across 10 windows × 10 turns I found **zero** frontiers where a parent recall and its own child recall render together | no gap under adaptive defaults |

---

### Excluded: reachable only by non-default configuration

Each row names the setting that enables it. Nothing here is charged against the artifact.

| capability | enabling setting | production evidence |
|---|---|---|
| kv-unified solve (exact-cut enumeration, Pareto propagation, receipt chain, 4-slot marker contract) | `foldingStrategy: 'kv-unified'` + a complete `kvUnified` config — `HOST:128` sets `'kv-stable'`, and the constructor throws without the config (`:8847-8851` `"live defaults are forbidden"`) | `src/adaptive/kv-unified.ts`, `kv-unified-policy.ts`, `kv-unified-pareto.ts`, `kv-unified-receipts.ts`; `src/types/strategy.ts:1141-1143` |
| Prepared-window transitions / `strictReach` / `blocked: 'reach-floor' \| 'target-floor'` | `productionBudgetTokens` + the hot-settings API, which is what sets `strictReach: true` (`:8841` `"strictReach: preparedBudget !== undefined,"`) | `src/adaptive/strategies/kv-stable.ts:148-153`; `src/adaptive/kv-control.ts:980-989` |
| Classic pins, at-level pins, max-level pins, locked chunks | **no setting**: `pinRange`/`markDocument`/`lockChunk` are callable at defaults (leveled semantics bind only under `foldingStrategy: 'kv-stable'`). Excluded here per this audit's default-*path* scope, not because a config turns them on | `src/context-manager.ts:910-961`, `:923` |
| `toolResultMaxLastN` per-tool pruning | `toolResultMaxLastN` (default `undefined`) | `src/types/strategy.ts:1066-1080` |
| `toolUseInputMaxTokens` truncation | `toolUseInputMaxTokens` (default `0`) | `src/types/strategy.ts:1082-1088` |
| Split-stitch rung + operator placeholder | `compressionSplitFallback` / `compressionSplitPlaceholder` (both default false) | `src/types/strategy.ts:985-1003`; `:6173` |
| Source-only L1 (the shape this artifact always uses) | `compressionSourceOnly` (default undefined/false) | `src/types/strategy.ts:947-981` |
| Final source-only fallback rung | `compressionSourceOnlyFallback` (default false) | `src/types/strategy.ts:983-984` |
| Source-only merge + its attribution-discipline line | `compressionMergeSourceOnly` / `compressionMergeSourceOnlyFallback` (both default false) | `src/types/strategy.ts:881-884`; `:7202-7203` |
| Speculation cap | `maxSpeculativeL1s` (default `undefined`) | `src/types/strategy.ts:1090-1100` |
| Compression output-ceiling override | `compressionMaxTokens` (default unset) | `src/types/strategy.ts:765-774` |
| Live-window carrier stripping | `carrierPolicy: 'live-strip'` (default `'full'`; the port sets it, `port:458`) | `src/types/strategy.ts:713-760` |
| XML recall envelope | `recallEnvelope: 'xml'` (default `'none'`) | `src/types/strategy.ts:673-711` |
| Legacy combined (unpositioned) recall pair | `positionedRecallPairs: false` (default true) | `:9445`; `:9527-9555` |
| Witnessed voice for inherited history + propagation of the `witnessed` flag up merges | `witnessedBeforeSequence` (default undefined; a host may pass it — `HOST:50`) | `src/types/strategy.ts:628-647`; `:9652`, `:6487`, `:7464`, `:216-229`, `:304-326` |
| Identity reminder appended to every instruction | `identityReminder` (default undefined; `HOST:52`) | `src/types/strategy.ts:648-669`; `:9672-9674` |
| `enforceBudget: false` (emit the full ideal context and let the API refuse) | `enforceBudget` (default true) | `src/types/strategy.ts:1102-1116`; `:9816` |
| Mint preimage persistence + retrieval by `requestHash` | `persistMintPreimages: true` (default false) | `src/types/strategy.ts:1216-1236`; `src/mint-preimage.ts:549-660` |
| One-shot effective-config report | `logEffectiveConfig: true` (default false) | `src/types/strategy.ts:1197-1214` |
| Agent-facing memory tools: `memory_timeline`, `memory_search`, `memory_recall`, pins at a chosen resolution, a dry chronological index | **not shipped at all** — `docs/memory-tools-design.md:3` `"Status: **SPEC — ready for review**"`; zero hits for `memory_timeline`/`memory_recall`/`dryIndex` in `src/` | `docs/memory-tools-design.md:1-210` |
| Unified-solve rev 6 | `docs/unified-solve-design.md:60` `"Nothing here is deployed."` | — |
| Best-fit DP solver | removed (`docs/best-fit-frontier-resolution.md:3` `"built and then **removed**"`) | — |
| `mergeContiguityGapLimit` | declared with `"Default 300"` at `src/types/strategy.ts:814` and **read nowhere** in `src/`, `test/`, `bench/`, `scripts/` — a dead knob, so there is nothing to enable | `:6698` implements strict adjacency with no gap tolerance at all |
| `chunkOnMessageBoundary` | declared `true` at `src/types/strategy.ts:1482` and read nowhere in `src/` — dead knob | — |

---

## 2. Disagreements with `config.py`

Ordered by how much they mislead. **D1–D4 are places where `config.py` asserts something
production does not do; D5–D6 are wrong statuses; D7 is an evidence claim that is not
supportable; D8 is capabilities it omits entirely.**

### D1 — `ignored` is defined as "default-OFF in production" and is used for five default-ON knobs

`config.py:18`:

```
    ignored       default-OFF in production, so there is nothing to implement
```

Every entry carrying that status:

```
attachmentsIgnoreSize                  ours=None         production=True
l1/l2/l3BudgetTokens                   ours=None         production=30000
compressionRecallBudgetTokens          ours=None         production=100000
compressionCacheMarkers / TTL          ours=None         production=true / '1h'
summaryParticipant                     ours=None         production=the agent's own name
```

(`python3 -c "from config import KNOBS; ..."`, output above.) All five are on by default:
`src/types/strategy.ts:1483` (`attachmentsIgnoreSize: true`), `:1478-1509` region and
`src/strategies/autobiographical.ts:916-920` (`l1BudgetTokens = 30000` inside
`if (supplied.hierarchical ?? true)`), `:5416` `"const recallBudget = sourceOnly ? 0 :
(this.config.compressionRecallBudgetTokens ?? 100_000);"`, `:929-930`
`"defaults.compressionCacheMarkers = true; defaults.compressionCacheTtl = '1h';"`,
`HOST:130-132` (`summaryParticipant = recipe.agent.name`).

The arithmetic consequence: `README.md:271` and `python3 config.py` both print
`ignored:5`, and the vocabulary tells the reader those five are default-off in production.
None is. This is the artifact's own stated distinction — "that distinction is the whole
point" — inverted for a quarter of its non-implemented rows. It also means the
"excluded, with the setting that enables it" list this audit was asked to produce cannot be
reconstructed from `config.py`: those five items have no enabling setting because they need
none.

`attachmentsIgnoreSize` is worse than mislabelled: `:10112` reads it
(`"if (this.config.attachmentsIgnoreSize) {"`) and `:10013` uses it to price chunks
(`msgs.reduce((sum, m) => sum + (this.config.attachmentsIgnoreSize …`)). It is a live
default-ON pricing rule, not an irrelevance.

*(Two further statuses are stale in the same direction: `README.md:271` quotes
`simplified:13`, the actual count is `simplified:14`.)*

### D2 — `maxMessageTokens` is called `implemented`; production *truncates the content*, ours only discounts the price

`config.py:89-95`:

```
    Knob("layout", "maxMessageTokens", 250, "0 library / 10000 connectome-host", "implemented",
         "types/strategy.ts:1500, framework-strategy.ts:90",
         "keeps one enormous message from dominating the accounting: its contribution to "
         "head/tail is capped at cap+50.",
         "IMPLEMENTED (scaled to 250). Production caps every message's contribution at "
         "compile (min(pse, cap+50)); we cap every message's accounting the same way. It binds "
         "on the document message in the demo."),
```

Production does both halves. The cap on the *price* is `Math.min(store.estimateTokens(msg),
msgCap + 50)` (`:9148`, `:9507`), and the content itself is truncated at seven emission
sites: `:5041`, `:7613`, `:8154`, `:8172`, `:9148`, `:9505`, `:9574`, all reading
`msgCap > 0 ? this.truncateContent(msg.content, msgCap) : msg.content`; `truncateContent` is
defined at `:11096`. The comment at `:8051-8057` is explicit that the deliberate exception
is *body-group composites only*: `"Deliberately do NOT apply maxMessageTokens here: …
(`maxMessageTokens` is for per-message caps on chat / tool results…)"`.

Ours has no truncation. Probe:

```
$ python3 - <<'EOF'   # 1 message of "word "*400 plus 5 short messages
chunk c0: priced=333 tok | text has 424 words (~424 tok) | cap=250
```

`minisystem.py:116-117` (`n = min(n, MAX_MESSAGE_TOKENS + 50)`) is the whole rule. So the
plan prices a window **smaller than the text it stands for**: at cap 250 a 424-token message
is charged 333 while the chunks' text is unchanged. That is the failure class production's
emission-overflow refusal exists to catch (`:7988-7975`, `"the picker planned this layout to
fit, so an overrun here is estimator drift"`, `:7988-7997`). Status should be `simplified` at best, and
the cost text should say the content is never truncated.

### D3 — `compressionContextBudgetTokens`' description inverts which requests production admits

`config.py:213-236` is titled "refuse to send a request that cannot fit: admission is
checked BEFORE the call, so an oversized request costs nothing — no provider round trip, no
burned attempt", and cites `autobiographical.ts:3027-3088`.

In production the gate covers **only the refusal-curve fallback variants**. The variants are
built from a canonical request that is issued first and unconditionally
(`docs/compression-recall-curve-fallback-spec.md` invariant 1: `"Canonical first. Always
issue the current canonical compression request unchanged. Fallback activates only after
canonical stopReason === "refusal"."`), and the config's own doc says so in as many words
(`src/types/strategy.ts:1013-1023`):

```
 * The canonical request remains unchanged and is always attempted first;
 * only fallback variants are gated.
```

`compressionRefusalPlan` returns `[]` when the fallback limit is 0 and otherwise iterates
`variants` (`:3027-3042`), computing `admittedTokens` per variant (`:3065-3068`). There is no
admission site on the canonical path.

So ours gates **more** than production does — every request, including the first. That may
well be the right choice for a teaching artifact, but it is a *deviation*, not a
simplification of production's behaviour, and the sentence "admission is checked BEFORE the
call, so an oversized request costs nothing" describes a rule production only applies to
rungs 2..n. `README.md:187-188` repeats it (`"term for term production's `admittedTokens >
budgetTokens → admission_rejected`"`); the *arithmetic* is term-for-term, the *scope* is not.

### D4 — "we have no demand path because the solver emits none by default" is false

`config.py:142-147`:

```
    Knob("production", "l1HoldbackChunks", 1, 1, "implemented", "autobiographical.ts:3979",
         …
         "IMPLEMENTED: the newest closed chunk is excluded from derived work until a newer "
         "one closes (production also lets a demand op override the holdback; we have no demand "
         "path because the solver emits none by default)."),
```

The solver the host selects — `kv-stable` — emits produce ops **whenever the solve
escalates**, which is the default behaviour, not an opt-in:

`src/adaptive/strategies/kv-stable.ts:165-198`:

```
    // DEMAND-SIDE PRODUCTION (issue #56). Speculative pre-production stays the
    // pre-producer's job — but when even full folding exceeds the hard wall W
    // (plan.escalated) while foldable chunks have no L1 at all, producing
    // those L1s is the only lever left…
    if (plan.escalated) {
```

and the consumer bypasses the holdback explicitly (`:3981-3985`):

```
      } else {
        // Demand path: mark the chunk so the l1HoldbackChunks window in
        // rebuildChunks never filters it back out of the queue.
        if (lastId !== undefined) this._demandedL1Chunks.add(lastId);
```

The artifact's own demo reaches exactly that state — phase 2 ends
`REFUSED: turn 8: 2190 tokens > wall 1224 (W=1200 + 2%)`, i.e. `tokens > W` ⇒
`plan.escalated` — and `connectome_min.py` has no `produced` output at all
(`grep -n "produce\|demand\|escalat" connectome_min.py minisystem.py llm/summarizer.py` returns
only the demo's label string at `:127`). This is also the one place where the omission has a
visible consequence: production keeps minting L1s to try to fold under a wall it is over;
ours has already stopped.

### D5 — `mergeThreshold` is `implemented` while its own cost text describes a behavioural difference

`config.py:38-42`:

```
    Knob("solve", "mergeThreshold", 6, 6, "implemented", "types/strategy.ts:870",
         "the pyramid's branching factor sets the fold quantum: …",
         "our run selection is coarser: production needs strictly CONTIGUOUS unmerged runs and "
         "merges a stranded interior run at 2 (autobiographical.ts:6694-6721). ~6 lines."),
```

A row whose `cost` field says "our run selection is coarser" cannot be `implemented` under
`config.py:13` (`"implemented   matches production's default path"`). Production's rule is
three-part — strict adjacency (`:6698`), level-scaled wide-span refusal (`:6678`), and the
interior-run-at-2 anti-starvation escape (`:6718-6721`); ours is
`free[:MERGE_THRESHOLD]` over all unmerged summaries at a level (`minisystem.py:205-208`),
with no adjacency test and no interior escape. `simplified` is the honest status, and the
interior-run rule is a separate four-line capability the row currently hides inside a cost
sentence.

The same pattern appears twice more, in the opposite direction: `autoTickOnNewMessage` is
`implemented` (`config.py:153-158`) while `speculativeProduction` is `simplified` with the
note `"production cascades every level in one drain (checkMergeThresholdRecursive)"`
(`config.py:148-152`) — that is the *same* drainto-empty gap seen from two labels, and it is
counted once as a gap and once as a match. `tick()` in ours consumes exactly one work item
per call (`minisystem.py:293-300`, `return` inside the loop); production recurses until no
progress (`:4149-4163`, `setTimeout(() => this.driveSpeculativeDrain(ctx), 0)`).

### D6 — the recall-pair price is attributed to the wrong label

`config.py:117-123`:

```
    Knob("layout", "recallHeaderTemplate", "'[Recall {id}]'", "'[Recall {id}]'", "implemented",
         "types/strategy.ts:1504",
         "keeps a recall's identity in its rendered bytes, …",
         "IMPLEMENTED: minisystem prices a recall pair as header + question label "
         "(summaryContextLabel) + content, which is what the picker weighs. …"),
```

Production prices **two different strings on two different surfaces**, and the picker weighs
only one of them:

- what is *emitted* as the Q-side of a positioned recall pair is `buildRecallHeader`
  (`:9468-9474`, template `'[Recall {id}]'` at `:9846-9852`);
- what is *priced* is `recallPairCost` (`:9006-9061`), whose only label is
  `const label = this.config.summaryContextLabel ?? 'What do you remember from earlier?';`
  (`:9007`), and whose answer side under `carrierPolicy: 'full'` is
  `Math.max(s.tokens, estimatedAnswer)` (`:9029-9032`);
- the `+50` for `"[CM] Recall memory <id>."` belongs to a third thing entirely — the
  **prompt**-side cap (`capRecallPairs`, `:2489` `"Per-summary +50 token overhead accounts
  for the "[CM] Recall memory <id>." question turn that wraps each recall body."`).

Ours (`connectome_min.py:140-141` pricing `header + label + content`) is header **plus**
label plus content. That is a superset: every recall pair is priced above the number
production's picker would use for it, so our solver folds more aggressively than
production's at the same W. The row is `implemented` and its cost text asserts a production
behaviour that does not exist. A two-string difference with a systematic sign is a
`different`, not a match.

### D7 — chunk salience is faithfully ported but cannot be exercised by the demo, and the row's evidence for it is nil

`config.py:66-74` is `implemented`, cost text `"IMPLEMENTED (minisystem.salience, ported
line-for-line)"`. The port is indeed faithful — I diffed `minisystem.py:53-73` against
`:8939-8995` including the `6400`-char image cost, the `^https?://\S+$` link test, the
`externalized = min(1, external/total)` clamp and the `max(0.2, 1 - 0.8·externalized)`
floor. But the demo's content generator gives **every** chunk a tool block, so every chunk's
salience is the 0.2 floor:

```
$ python3 minisystem.py
  message salience (composition): [1.0, 0.85, 0.2, 0.95, 1.0, 0.85, 0.2, 0.95]
  -> chunk salience is its cheapest message: {'c0': 0.2, 'c1': 0.2, 'c2': 0.2, 'c3': 0.2}
$ python3 -   # distinct chunk saliences across the demo's own content()
    distinct chunk saliences: [0.2]
```

The priority in `relevance_cut` is `sorted(chunks, key=(salience, seq))`
(`connectome_min.py:78`) — with a constant salience this degenerates to pure age order, so
the salience term is dead weight in the only run the artifact ships, and the claim that it is
exercised rests on the code rather than on any output. (`README.md:300`, by contrast,
understates the port: `"salience here is assigned synthetically"` — it is computed from
composition, not assigned.)

Not a wrong status; a wrong *evidence* claim. Five lines in `content()` (drop the tool block
from the `i == 3` branch) would make the term observable.

### D8 — capabilities `config.py` does not mention at all

Each of these is default-ON in production and absent (or materially different) in ours. In
several cases `README.md` mentions the area but `config.py` — the file that claims to be
"the fidelity audit trail" and whose table "cannot drift from the code it describes"
(`README.md:283`) — has no row.

**(a) The prompt pair is dead in production and live in ours.** This is the most misleading
omission. `config.py` has no row for prompt text; `README.md:131-132` says

> The system prompt is the library's own (`types/strategy.ts:1485`), the L1 user turn is
> `summaryUserPrompt` with `{content}` filled (`types/strategy.ts:1486`)

`summarySystemPrompt` and `summaryUserPrompt` have **zero read sites** in production:

```
$ grep -rn "summarySystemPrompt\|summaryUserPrompt\|diarySystemPrompt\|diaryUserPrompt" src/ bench/ test/ scripts/
src/types/strategy.ts:625:  summarySystemPrompt?: string;
src/types/strategy.ts:627:  summaryUserPrompt?: string;
src/types/strategy.ts:860:  /** @deprecated Use summarySystemPrompt */
src/types/strategy.ts:861:  diarySystemPrompt?: string;
src/types/strategy.ts:862:  /** @deprecated Use summaryUserPrompt */
src/types/strategy.ts:863:  diaryUserPrompt?: string;
src/types/strategy.ts:1485:  summarySystemPrompt: 'You are forming a memory of an earlier part of this conversation. …'
src/types/strategy.ts:1486:  summaryUserPrompt: `What do you recall from this part of the conversation?
```

(the grep also covers `bench/`, `test/` and `scripts/`; only declarations and the default
object appear). Production's L1 request is `COMPRESSION_MARKER` + `formatInstruction`
(`:99-102`, `:192-206`), and its system turn is the host's live identity prompt, conditionally
spread (`:5604`). So `llm/prompts/system.txt` and `llm/prompts/l1_chunk.txt` are faithful copies of
two strings production *declares* and never *sends*, and the artifact's L1 write is the one
part of the request shape whose production counterpart it does not implement. Only
`llm/prompts/merge.txt` is the real thing (I diffed it against `:286-294` — verbatim).

**(b) `minChunkCharsForLLM` = 200 mechanical stub** — `:5216-5241`, no LLM call for a chunk
of silent/bare-system traffic, `tokens: 40`. Absent from `config.py`; no stub path in
`minisystem.step`.

**(c) Reading-mode voice** — `formatReadingChunkInstruction` (`:250-266`),
`formatReadingMergeInstruction` (`:340-365`), `detectDocContext` (`:9721-9731`). Default-on
whenever a document shards. Absent from `config.py`; `minisystem.spans` writes the `g{seq}`
group tag (`:112-113`) and nothing reads it.

**(d) Both image byte walls** — `maxLiveImageBytes` 20 MiB (`:10399`) and
`maxCompressionImageBytes` 12 MiB (`:5647`, `:7258`). The image row (`config.py:99-106`)
registers only the count and the depth.

**(e) `responseContent` carrier capture and replay** — `:891-902`, `:10950-10988`, and the
pricing consequence `:9029`. Mentioned once inside another row's cost text
(`config.py:123`, `"Reasoning carriers (carrierPolicy 'full') are not modelled."`) with no
status of its own; production's own doc calls the anti-refusal duty on that path *measured*
(`src/types/strategy.ts:728-734`).

**(f) The coverage invariant** — `assertFullCoverage`, `assertMiddleCoverage`,
`UncoveredDropError`, exported as a cross-package behavioural surface (`src/index.ts:38`).
No row.

**(g) Picker validation alarms** — `[picker-unrealizable]`, `[picker-dead-ids]`,
plan-vs-emission drift refusals (`src/adaptive/picker.ts:295-312`; `:7962-7975`). No row.

**(h) `projectToValidCut` and the fold→project fixpoint** — `README.md:309-311` records the
omission; `config.py` has no row, even though `planControlledFrontier` calls it
unconditionally twice (`:484`, `:875`) and `suffixAdopt` calls it on **every** binary-search
candidate (`:545`).

**(i) Merge recall budget halving per failed attempt** — `:7017-7020`. No row; a retried
merge in ours re-sends identical bytes.

**(j) Retrieval, branches, pins-as-API, quarantine clearing, health surfaces** — the whole
memory lens is unregistered: `searchSummaries`/`getSummary`/`getSummariesInRange`
(`src/context-manager.ts:964-1007`), `branchAt`/`fork`/`switchBranch` and branch-generation
identity (`:567-657`, `src/branch-generation.ts`), `clearCompressionRefusalQuarantine`
(`:1263`), `getCompressionDebt` (`:4425`), `getRenderStats` (`:4909`). `config.py`'s
`ignore`/`deviation` rows cover the *solver's* protections and the quarantine *alarm*, but
nothing tells a reader that the artifact has no archive-access surface, no branch model and
no way to clear terminal debt.

**(k) `compressionCacheMarkers` / TTL** is registered — but as `ignored` (see D1), i.e. as
default-off, when `conditionalLibraryDefaults` sets it unconditionally true
(`:929-930`, `"compressionCacheMarkers = true; compressionCacheTtl = '1h';"`, outside the
`adaptiveResolution` gate). "Cost only, no layout effect" is a fair scoping judgement; the
status word is not.

---

## 3. Architectural vs additive

**Would change the design (the log/derivation model or the policy/system split):**

1. **Coverage as a fatal invariant** (D8f). The artifact's log-derivation model makes
   coverage hold *by construction* — `render` falls back to raw whenever no summary exists —
   so there is nothing to assert, and that is exactly the property that makes the assertion
   look unnecessary. Making it real means the derivation must be able to *fail*, i.e. the
   plan needs an emitted-vs-planned distinction the current `render` collapses. Either the
   policy grows a "what was actually emitted" return value, or the system owns a post-render
   check over the frontier. That is a change to the split, not a new file.
2. **The demand path** (D4). `work()` is the whole derivation — `chunks with no L1, plus a
   merge per base-run`. Demand-side production is the case where the derivation must react to
   *the plan* (escalation), not only to the log. Closing it means `plan_controlled_frontier`
   returns something the `work()` derivation consumes — a second input to the system layer,
   which is what "work is derived, never queued" was built to avoid. This is the most
   structurally interesting gap: the one place where production's queue is load-bearing and
   the artifact's purity is a limitation.
3. **Branches** (D8j, memory lens). One log is one history. A real branch model changes
   `frontier(log)` from "the last frontier event" to a branch-scoped read, and every derived
   projection (`archive`, `work`, `mints`, `fails`) with it. `minisystem.py:230-232` already
   has a stale-version rule; it is the branch model's *absence* that makes it a version stamp
   rather than an identity.
4. **Per-branch identity for in-flight work.** Related, and the smaller half of (3):
   production discards a compression result that crossed a branch boundary (`:5852-5856`,
   seven discard sites). One generation counter is enough only if branches are out of scope.

**Component-sized, addable without touching the split:**

5. Chunk-salience observability (D7) — five lines in `content()`.
6. The `ignored` status audit (D1) — a data change in `KNOBS`.
7. `maxMessageTokens` truncation (D2) — a `truncate` helper plus a re-price in `spans()`;
   ~15 lines, and it makes priced == rendered.
8. `minChunkCharsForLLM` stub (D8b) — a branch in `work()`/`step()` that emits a `mint`
   without calling the model; ~10 lines, and it is testable offline.
9. The reading-mode instruction (D8c) — a new prompt file selected by the `g{seq}` group
   tag `spans` already produces; ~15 lines.
10. Contiguity + interior-run-at-2 merge selection (D5) — replace `free[:MERGE_THRESHOLD]`
    in `work()` with a run split; ~10 lines.
11. The wide-span guard (`mergeMaxSourceSpanMessages`) — a filter in the same function;
    ~8 lines, as `config.py` estimates.
12. Merge-attempt budget halving (D8i) — one expression in `step()`, keyed on
    `attempts(log, target)`.
13. Recall-pair price correction (D6) — drop the header term from the price, or keep the
    header and drop the label; two lines, but it changes the fold floor, so it needs the
    probe re-run.
14. `clearCompressionRefusalQuarantine` analogue — a `clear` event the `stalled` predicate
    consults; ~10 lines, and it removes the artifact's worst asymmetry (terminal debt that
    can never be cleared).
15. Image byte walls (D8d) — a `bytes()` helper alongside `chars()`; ~10 lines.
16. `getCompressionDebt`-style staleness — `spans` already carries seqs; a
    `pending-age` projection; ~15 lines.
17. `previewContext` / `checkReadiness` — both are cheap once the policy is pure, which it
    already is (`connectome_min.py:95-118` has no side effects).

---

## 4. The five gaps I would close first

| # | gap | one-line reason | rough size |
|---|---|---|---|
| 1 | **D3 + D4: the two claims about which requests are admitted and whether demand exists** | both are wrong in the same direction — they make the artifact look like it implements a production rule it actually exceeds (D3) and like it has no demand path when the default solver has one it is not modelling (D4); a reader who trusts `config.py` here will mis-plan the work | D3 is a doc fix (~6 lines); D4 is a `produced` return value + a `work()` input (~25 lines) |
| 2 | **D1: five default-ON knobs labelled `ignored`** | the artifact's own vocabulary defines `ignored` as default-off, so `ignored:5` currently asserts five falsehoods and destroys the enable-setting distinction the audit is built around | data-only, ~30 lines of `KNOBS` edits |
| 3 | **D2: `maxMessageTokens` truncation** | the only gap that makes the plan *lie* — priced 333 for 424 tokens of text — and it is the same failure class production's emission-overflow refusal was written for | ~15 lines + a test asserting priced == rendered |
| 4 | **D8a: the L1 directive is the wrong string** | the artifact believes it copies production's prompt and copies two knobs with zero read sites; the real directive (`COMPRESSION_MARKER` + `formatInstruction`) is one prompt file away and is what the memories actually read like | ~20 lines (one prompt file + a marker turn + a truthful README/`config.py` row) |
| 5 | **D8f: coverage assertion** | it is the invariant production treats as fatal and cross-package (`UncoveredDropError` is exported for `instanceof` across boundaries), and closing it is what forces the plan/emit split the artifact currently does not have | ~30 lines, but it is the architectural one — see §3.1 |

Honourable mention: `clearCompressionRefusalQuarantine` (item 14 in §3) — five lines of
policy, and it removes a state the artifact can enter and never leave.

---

## 5. What I could not determine

1. **Whether production's `projectToValidCut` ever changes a plan here.** README:309-311
   claims equivalence "for the nested trees this toy builds". I probed hard for a
   counterexample and did **not** find one: across the 8 plans of `python3 minisystem.py`, a
   6×~10-turn sweep over `W = 900..1300` with a deliberately divergent carried frontier
   (`ideal − 1` per chunk), and a 60-candidate `suffix_adopt` sweep over `P`, the count of
   frontiers on which the projection would fire (a leaf at level `L>0` whose own
   `ancestor_at(L)` node has a sibling not at `L`) was **0**. I also found something that
   *looks* like a violation and is not: the ideal cut can leave a group with leaves at
   different levels (probe: `L2-L1-c24` leaves `[0,1]`) — but no leaf there sits at level 2,
   so the projection, which only examines leaves at their own level, is a no-op. **This is
   unexercised, not refuted.** Blocker: production cannot be executed
   (`ref/context-manager/node_modules` is absent), so a differential run of
   `planControlledFrontier` on identical `PickerInputs`/`SummaryTree`/`F_prev` is impossible
   here; `README.md:334-337` names the same missing harness.
2. **Whether the tool manifest is present on production mint requests, and whether the
   calibration multiplier is ever fed.** `ContextManager.setToolDefinitions`
   (`src/context-manager.ts:1089`) is what populates `ctx.tools`, and
   `reportRealInputTokens` (`:8345`) is what feeds the EMA; neither has a caller in this
   checkout outside `test/`. The host that wires them is not vendored. Blocker: absent host
   code. Consequence: the "tools declared" and "calibration learns from the wire" rows are
   library-side verified only.
3. **Whether `phaseChannel` does anything.** `src/phase-channel.ts:19` is
   `report: () => {}` by default, and I found no consumer in `ref/connectome-host/src`. Any
   wedge-attribution claim resting on that channel is unverified here.
4. **Whether anything consumes `getCompressionDebt`.** Its doc says `/healthz`
   (`:4405-4406`); no `/healthz` consumer exists in the library or in the vendored host.
5. **What the artifact would do against a real provider.** `python3 minisystem.py` runs on
   `MockModel`; `HttpModel` is never dispatched in my runs. So every claim about the *mock's*
   behaviour is verified by execution, and every claim about a live model's refusals,
   truncations or usage numbers is a read of the classification tables
   (`llm/summarizer.py:27-29`, `:197-235`) and of the tests' injected transports, not an
   observation. Blocker: no network, no credentials (both out of scope by instruction).
6. **The size of two claimed forthcoming changes.** `config.py:42` and `:164` estimate
   "~6 lines" and "~8 lines" for the contiguity rule and the wide-span guard. I did not
   implement either, so my §3 sizes for them are my own reading of `work()`, not measurements.

---

## 6. Closing: what was fixed, with evidence

A later session worked this audit's priority list. Every claim below was re-checked against
`ref/context-manager` before the fix, and every "fixed" line names a command or a `file:line` that
can be re-run. Files touched: `config.py` (456 -> 602), `minisystem.py` (380 -> 531),
`connectome_min.py` (136 -> 159), `llm/summarizer.py` (313 -> 321), `test_summarizer.py` (591 -> 851),
`README.md`, `llm/prompts/` (three files, one new).

**Verification runs for this section.** `python3 test_summarizer.py` -> exit 0, 212 checks;
`python3 minisystem.py` -> exit 0 (diff against the pre-fix capture is explained line by line
below); `python3 config.py` and `--gaps` -> exit 0, `implemented:20 simplified:16 MISSING:1
deviation:6 unmodelled:5 ignored:0`; the four `connectome_min.py` runs are **byte-identical** to
`/tmp/v136_def.txt`, `/tmp/v136_r1250.txt`, `/tmp/v136_w700.txt` and `/tmp/v136_r300.txt` (diff,
empty).

| finding | verdict | evidence |
|---|---|---|
| **D1** five default-ON knobs marked `ignored` | **fixed** | `unmodelled` added to the vocabulary (`config.py:11-29`); all five re-labelled with their read sites (`config.py:117-124` `attachmentsIgnoreSize` at `autobiographical.ts:10013`/`:10112`, `:239-250` `l1/l2/l3BudgetTokens` at `:916-920`/`:9196-9198`, `:320-327` `compressionRecallBudgetTokens` at `:5416`/`:7017`, `:328-336` `compressionCacheMarkers / TTL` at `:929-930`/`:2553`/`:5621`, `:459-465` `summaryParticipant` at `HOST:130-132`). Count line now reads `unmodelled:5 ignored:0`. |
| **D1, enforcement** a status may not contradict its own definition | **fixed** | `config.audit_status_conflicts()` (`config.py:561-573`) refuses `ignored` over an ON production value and `unmodelled`/`MISSING` over an OFF one; `report()` calls it, so a contradictory table exits non-zero. Test: `test_config_status_integrity` (`test_summarizer.py:652-679`). Mutation (c) below. |
| **D2** `maxMessageTokens` capped the price only | **fixed** | `minisystem.truncate_blocks` ports `truncateContent` (`:11096-11135`, the seven `msgCap > 0` sites) and `spans` prices the truncated blocks (`minisystem.py:109-134`, priced in `spans` at `:136-167`); `config.py:105-116` now says production does both halves and that ours does too. Test: `test_truncation_prices_what_it_renders` asserts `priced == rendered` for an over-cap message (259 tokens, not the 300 ceiling). |
| **D3** `compressionContextBudgetTokens` scope inverted | **fixed** (record) | status `simplified` -> `deviation` (`config.py:287-319`) with the `types/strategy.ts:1013-1023` citation and the statement that we gate the only rung we send; `README.md:232-238` repeats it. The behaviour is unchanged and deliberate. |
| **D4** "no demand path because the solver emits none" | **fixed** | `connectome_min.demand_runs` (`connectome_min.py:99-116`) computes `produced` exactly as `kv-stable.ts:180-198` does, from `plan.escalated` = `ideal.tokens > windowTokens` (`kv-control.ts:925`; comparable to the pre-fix text at `config.py:146`, which was false); `minisystem.demanded_chunks` + `work(log, demanded)` consume it as the holdback bypass (`minisystem.py:271-302`; the production analogue is `:3992-3999`). Demo evidence: `demand path: the escalated plan produced [('c4', 'c4')]`. Test: `test_demand_path_bypasses_the_holdback`. Mutation (b) below. **Limit found while fixing:** at this configuration the holdback window (1 chunk) always sits inside the picker's raw zone (the tail walk always takes the newest chunk), and demand never names a raw-zone chunk — so the bypass has nothing to bypass in the shipped demo, and the test raises the holdback to 2 to exercise it. That is a property of our scaled tail, not of the port. |
| **D4, plans unchanged** | **verified** | `diff /tmp/v136_*.txt` vs a fresh run: empty for all four. `connectome_min.py`'s own replay mints an L1 per chunk, so `produced` is `[]` there by construction. |
| **D5** `mergeThreshold` labelled `implemented` | **fixed** (record) | `simplified`, naming all three missing rules with citations (`config.py:48-58`): strict adjacency `:6698`, level-scaled wide-span refusal `:6678`, interior-run-at-2 `:6718-6721`. No code change (the guard remains `PRIORITY[0]`). |
| **D5** `autoTickOnNewMessage` vs `speculativeProduction` | **fixed** (record) | `autoTickOnNewMessage` -> `simplified` (`config.py:199-208`, and `:190-198` for `speculativeProduction`): production's auto-tick *calls* `driveSpeculativeDrain` (`:4105`), which recurses until a tick makes no progress (`:4120-4166`), so the knob's observable promise is the drain and not the call count that our `implemented` label cited. `speculativeProduction` stays `simplified` and both rows now name the same single gap from its own side. |
| **D6** recall-pair price charged the header | **fixed** (price, not record) | `minisystem.archive` now prices `summaryContextLabel + content` (`minisystem.py:170-190`), production's `recallPairCost` term for term (`:9006-9061`, `:9007`); the emitted-but-unpriced header and the prompt-side `+50` (`capRecallPairs`, `:2489`) are named in `config.py:143-164`. Consequence, measured: the demo's phase-1 plan went 2276 -> 2274 tokens and phase 2's refusal 2190 -> 2188. |
| **D7** salience unexercised by the demo | **not fixed** | Outside the assigned scope; `content()` still gives every chunk a tool block, so chunk salience is the 0.2 floor in the shipped run. `README.md:359-360`'s "assigned synthetically" sentence was left as it is: it describes `connectome_min.py`'s replay, which does pass synthetic salience (`connectome_min.py:149`), so I read it as true of that file. |
| **D8a** the L1 prompt was two dead knobs | **fixed** | `llm/prompts/marker.txt` is `COMPRESSION_MARKER` verbatim (`:99-102`) and `llm/prompts/l1_chunk.txt` is `formatInstruction` verbatim (`:192-206`); `build_request` sends marker + source + directive (`llm/summarizer.py:40-62`), which is production's order around the chunk (`:5480` -> chunk -> `:5503-5506`). `llm/prompts/system.txt` is relabelled in its own header as our stand-in, `config.py:209-226` records that production serves the host's live identity prompt there (`:5604`, `HOST`), and `README.md:176-183` corrects the provenance. Tests: `test_l1_request`, `test_prompt_provenance`. |
| **D8f** no coverage invariant | **fixed** | `minisystem.assert_coverage` over the emitted render (`minisystem.py:304-331`) raises `UncoveredDropError` on the emit path in `run()` (`minisystem.py:403`), before the frontier is recorded. Test: `test_coverage_invariant_is_fatal`, whose fixture is a stale re-mint onto a drifted chunk id — and which mutation (a) below breaks. |
| **D8j** terminal debt that can never be cleared | **fixed** | `clear_debt` appends a `clear` event and every debt projection reads failures *since* the last clear (`minisystem.py:205-255`: `fails_since_clear`, `clear_debt`, `cleared`, `stalled`); the automatic half is `cleared()` returning true for a span that later mints (`clearQuarantineForCompressedChunk`, `:3441-3459`). `config.py:353-366` records it as `implemented`. Test: `test_clear_restores_derivable_work`; demo: `clear_debt(L1:c4) -> attempts=0 stalled=False, derivable again: ['L1:c4']`. |
| **D8k** cache markers registered as `ignored` | **fixed** | Same as D1: `unmodelled`, with read sites. |
| §4 item 15 (image byte walls), §4 item 16 (`getCompressionDebt`), items 5-12 not in the assigned list | **not fixed** | Out of scope for this session; `config.py --gaps` still carries the wide-span guard as the only `MISSING`. |

**`python3 minisystem.py` diff against the pre-fix capture, line by line** (`diff`, 5 hunks):

1. `turn 7 plan adopt-ideal tokens= 2276 dP= 1132` -> `2274 dP= 1130`, and the same in the phase-1
   summary line: the D6 price correction. Each priced recall lost the 2-word header
   `[Recall L2-L1-c0]`.
2. `REFUSED: turn 8: 2190 tokens` -> `2188 tokens`: the same two words, one hunk later.
3. **+** `demand path: the escalated plan produced [('c4', 'c4')] ...`: new, D4.
4. `permanent fold floor from it: 234 tokens no plan can fold away` -> `fold floor from it: ... until
   the debt is cleared`: new, D8j.
5. `model usage: 6971 in` -> `8746 in`: the L1 request now carries production's marker and its real
   directive, which is longer than the `summaryUserPrompt` text it replaced, so the mock's chars/4
   estimate rises ~127 tokens per call over 14 calls. Nothing else moved; the `out` total, the
   frontier, the profiles and the protections are identical.

**Deliberate omissions that remain** (so a reader does not mistake this for a full port):
`projectToValidCut`; pins/locks/prepared windows; branches; retrieval and memory-search surfaces;
the `checkReadiness` gate; the READING-MODE L1 instruction (`formatReadingChunkInstruction`; the
merge-side variant is implemented as of §7); the `minChunkCharsForLLM` stub (D8b); the wide-span
merge guard and the contiguity/interior-run merge selection; merge-attempt budget halving (D8i);
image byte walls (D8d); reasoning carriers (D8e); picker alarms (D8g).

---

## 7. The merge request's view depth (closing the last quality deviation)

The audit's §4 list carried "show the merge the layer beneath its sources" as a quality gap, and
`config.py` recorded it as a `deviation`: *ours = "children summaries", production = "one level
deeper"*. `PRIORITY` listed it second. It is closed, and it was wrong in more ways than that row
said. Files touched: `config.py` (602 -> 670), `minisystem.py` (531 -> 643), `llm/summarizer.py`
(321 -> 348), `test_summarizer.py` (851 -> 1082), `README.md`, `llm/prompts/merge.txt` (comments only),
`llm/prompts/reading_merge.txt` (new). `connectome_min.py` was not touched.

**What production actually does** (re-read at the cited lines before any edit).
`executeMerge` assembles the request in three parts, in this order (`:6867-6891`):

1. **PREFIX** — the head window raw, then the prior recall pairs, then the raw middle, each element
   skipped when a live summary already covers that stretch (`:6967-6990`, and the leaf-coverage dedup
   at `:6948-6965` that fixed the 525-message merge request).
2. **TARGET** — the sources expanded ONE LEVEL DEEPER than they themselves are (`:7076-7148`):
   sources at L1 (whose `sourceIds` are message ids) are emitted raw, so an L2 merge shows the actual
   conversation the six L1s consolidate; sources at L2+ are emitted as `[CM] Recall memory <id>.`
   pairs one level down, so an L3 merge shows its 36 L1s.
3. **INSTRUCTION** — `formatMergeInstruction` last, and no tail after the merge range (`:6889-6891`).
   `{seen_description}` comes from the layer ACTUALLY shown, not from the sources:
   `sources[0].sourceLevel === 0 ? 0 : sources[0].level - 1` (`:7151-7154`, `:277-295`) — L_n → L_{n-2}.

Three things that belong to the fix and were each re-checked: `formatReadingMergeInstruction`
(`:340-365`) replaces the consolidation instruction when every leaf under the merge is a shard of
one body group (`:7156-7175`), with `totalTokens` the size of the WHOLE group; the refusal-fallback
path calls the formatter with the sources' OWN level because it shows the sources and expands
nothing (`:7095-7125`, `:7191-7192`); and the prefix ladder is capped by
`compressionRecallBudgetTokens` (`:7017`, `:2489-2514`), walked newest-first with `continue` rather
than `break`.

**What changed here.** `minisystem.source` is now a projection of what each request is SHOWN, not of
the work item's identity: it returns a `Shown(prefix, target, reading_tokens)`, and
`summarizer.build_request` lays the three parts out (prefix, target, instruction last) and picks the
instruction by the shown layer. An L1 request is unchanged and still source-only, which is
production's own zeroed-ladder branch (`:5416`); a merge request is production's canonical shape.
The `{seen_description}` wording is generated from `max(0, level - 2)` with production's four
literals verbatim, and the two prompt bodies were diffed against the TS formatters mechanically
(joins of the backtick templates, `${...}` mapped to our placeholders): `merge.txt` and
`reading_merge.txt` are byte-identical to `:277-295` and `:340-365`.

**Verification runs.** `python3 test_summarizer.py` -> exit 0, **257 checks** (was 212), with new
coverage for each part: the L2 target is the raw L0 (`request["source"] == the six chunk texts`, and
no `MEMORY-OF-c*` anywhere in the turn), the L3 target is twelve `[CM] Recall memory L1-c*` pairs and
not the two L2s, the prefix is exactly head + prior pairs + raw middle with nothing after the merge
range and no covered stretch leaking back in raw, `{seen_description}` in BOTH branches (the raw
wording restored — see below), the recall cap's newest-first eviction and chronological re-sort, and
the reading-mode instruction with `totalTokens` over the whole group. `python3 config.py` and
`--gaps` -> exit 0, `implemented:21 simplified:17 MISSING:1 deviation:5 unmodelled:4 ignored:0`; the
`merge request scope` row now reads `[ok  ]` and is absent from `--gaps`.
`python3 minisystem.py` -> exit 0, **one changed line** (below). The four `connectome_min.py` runs
are byte-identical to `/tmp/v136_def.txt`, `/tmp/v136_r1250.txt`, `/tmp/v136_w700.txt` and
`/tmp/v136_r300.txt` (`diff`, empty).

**The demo diff is one line, and here is what it is.** `model usage: 8746 in / 247 out` ->
`11501 in / 251 out`. The merge is the only request whose shown text changed: its source went from
706 characters (six L1 digests) to 11 694 characters of raw L0, so its pre-send bound went
2263 -> 13 399 bytes and its estimate 337 -> 3092 — the `in` total rises by exactly that 2755. The
`out` total rises by 4 because the re-rendered layer is the chunk AS THE ARCHIVE PROJECTS IT NOW:
the digest recorded at L1-mint time read "look at this m0 …" (87 chars), and by turn 6 that turn's
image has been stripped, so the re-rendered text begins "[image stripped]" (100 chars). Nothing else
moved: no plan, no frontier, no profile, no protection, and **no merge was refused by admission**.

**The one thing to watch, reported rather than tuned away.** The bound grew by 11 136 bytes and the
demo's merge now sits at 14 999 of its 20 000-unit admission budget (the input side is BYTES and the
reserve tokens, which is production's own arithmetic — a bound, not a bill), 5001 units left, 25%. That is the
honest cost of showing a merge its material: production's own answers are the recall budget, which
caps the PREFIX and not the target, and the per-attempt halving of that budget
(`max(8000, configured × 0.5^min(attempts,4))`, `:7017-7020`), which we do not model. The merge that
would not fit here is one over many shards of one long document, for a reason that is ours: each of
our per-shard units carries the WHOLE message text (production's are slices of ~targetChunkTokens,
`:1567-1580`, rendered as one combined message, `:7982-7990`), so a merge over N shards repeats that
message N times. Measured on the reading-mode fixture: a 2000-token document shards into 7 chunks of
~8000 characters each, six of them in one target is ~48 000 characters, and that request is refused
(`shrink`) before dispatch. `test_merge_reading_mode_instruction` inspects the instruction with
room given to the engine and then asserts that same request's refusal at the shipped budget, so the
finding is a check and not a footnote; the `bodyGroup sharding` and `compressionRecallBudgetTokens`
rows of `config.py` say so too. The demo's own merge, whose sources include two of the three shards of turn 6, is well inside.

## §10 — the merge starvation (fixed) and the merge-coverage finding (open)

`tick()` now runs production's two priorities in one call: one chunk compressed (`:4214`), then one
merge executed (`:4238`), with the work set re-derived between them. Our previous tick took exactly
one item, and since L1 work always sorts first while new chunks keep producing it, a session could
reach its wall without ever consolidating. Verified behavior-neutral at the demo's size (its output
is byte-identical) and green at 361 checks.

**Still open, with measurements.** The demo does not reach a merge at any size we swept: phase-1
lengths 8/10/12/14/16 turns and windows 2300/3000/4000/5000 all top out at five unmerged L1s, because
only ~six chunks are compressible (head, tail and the operator pin are excluded) and the scripted
failures on `c3`-`c6` spend calls that would otherwise mint. Six are needed for candidacy. Restoring
merge coverage in the shipped run is therefore a demo-content change -- more compressible material or
a lighter failure script -- not a systems change; the merge request shape and the wide-span guard
remain fixture-exercised.

**FIXED in review**: the duplication was a defect, not a scale artifact -- `spans()` built each
shard from `text_of(blocks)`, the whole message, so N shards carried N copies of the document while
being priced as slices (probe: three shards, one distinct text, 2812 chars each against a 234-token
price). Slicing the text the way production's `chunkMessage` does (:1567-1580) fixes it: shards are
now 937/937/938 chars for a 234-token price, and the reading merge measures bound 9104 of 20 000
(46%) and is admitted. The two checks above it that asserted the refusal were asserting the bug and
now assert admission plus the size.

**Two claims corrected in the process.**

1. **"Prior L1 recall pairs" is the comment, not the code.** The section comment (`:6870-6876`) and
   the assignment both said L1 pairs, but `priorSummariesAll` is the UNMERGED FRONTIER — every
   summary with no `mergedInto` whose span starts before the merge range (`:6921-6946`) — and the
   code carries its own reason: filtering to L1 produced hundreds of pairs at 4000+ messages and
   overflowed the window. We ported the code's rule, and `config.py` records the difference between
   the comment and the code rather than repeating the comment.
2. **"Unreachable" was a property of the deviation, not of the code.** A previous round deleted the
   test for `formatMergeInstruction`'s raw branch (`sourceLevelShown === 0`) as dead code. It was
   dead only because our merges were shown their children, so the description was always
   `the L{sources' level} memories above`. With the expansion, an L2 merge is shown raw L0 and that
   branch is the COMMON one; `test_merge_instruction_describes_what_is_shown` pins both branches.
   (I could not verify the deletion itself: `tmp/` is untracked, so the file has no history — I take
   it on the task's word, and the branch is covered either way.)

**§1.2 rows this fix overwrites** (the inventory above is the state as audited; these are the
entries in it that are no longer true):

| §1.2 row | then | now |
|---|---|---|
| Merge request shows the sources **one level deeper** | ours: different — "`llm/summarizer.py:50-52` shows the children themselves" | ours: yes, three parts (`minisystem.source` + `summarizer.build_request`) |
| `formatMergeInstruction` + `sourceLevelShown` derived from the shown content | ours: present; the row's verdict was "none" | ours: present AND correct — the old derivation named a layer one deeper than the one actually sent, so "none" was wrong |
| Degraded merge rung: sources as their own recall pairs after a refusal | ours: absent — "we are always on this rung" | ours: absent, and we are no longer always on it: the canonical, deeper rung is what we send, and a refusal retries it unchanged |
| `capRecallPairs` / `compressionRecallBudgetTokens` | ours: absent — "moot while requests are source-only; returns with the request shape" | ours: implemented for the merge prefix, scaled ÷10, newest-first with `continue`-not-`break`; moot nowhere now |

**Left open by this fix** (each recorded in `config.py`, not merely here): the refusal-fallback
payload and its formatter argument (`:7095-7125`, `:7191-7192`); the per-attempt recall-budget
halving (`:7017-7020`); the witnessed-voice variant `formatWitnessedMergeInstruction` (`:304-338`),
which needs a `witnessed` flag our archive does not carry; and the reading-mode **L1** instruction
(`formatReadingChunkInstruction`, `:250-266`) — the merge-side variant is now implemented, the
chunk-side one still is not. Also unchanged, and independent of this fix: the merge picker still
takes the first base-run of unmerged siblings without production's contiguity, span and
interior-run rules.

## 8. The L1 request (closing the last wholesale deviation)

`config.py`'s `request shape` row read *ours = "source-only, by invariant"*, *production =
"experience replay"*, status `deviation`, and `PRIORITY` listed it second with the note "reverting
to replay would undo the fix". The task brief cited the builder's
section comment as `:5266-5292` and the source-only doc as `types/strategy.ts:966-980`; the block
ends at `:5294` (the "There is intentionally NO tail_after_chunk" lines) and the doc at `:981` (the
`)` closing the JSDoc before `compressionSourceOnly?: boolean;`), and this section cites the
inclusive range. It is closed. Files touched: `config.py` (670 -> 723),
`minisystem.py` (646 -> 707), `llm/summarizer.py` (348 -> 382), `test_summarizer.py` (1082 -> 1395),
`README.md` (442 -> 524), `AUDIT.md` (785 -> 909), `llm/prompts/reading_l1.txt` (new).
`connectome_min.py` was not touched (159 lines before and after), and its four documented runs are
byte-identical to the captures.

**What production actually does** (re-read at the cited lines before any edit). The L1 builder
documents six sections (`:5266-5294`):

1. **HEAD** — the raw chronicle opening, FIRST, "exactly where the original instance saw it. It MUST
   precede the recall pairs: when it followed them (pre-2026-07 order), it read as the most recent
   live conversation, and for thin chunks the summarizer narrated the head as fresh events …
   compounding across merges into runaway false memories (the '68 initiations' incident)."
2. **PRIOR SUMMARIES** — "narrativized as CM-asks / agent-recalls pairs, in source order. The
   **unmerged frontier** of the summary forest … using the children plus their parent doubles the
   prompt size unboundedly."
3. **RAW MIDDLE** — messages between head and chunk not covered by any summary (usually empty).
4. **MARKER** — `COMPRESSION_MARKER` (`:99-102`, pushed at `:5470-5473`).
5. **CHUNK** — the raw messages being compressed.
6. **INSTRUCTION** — doc-aware if the chunk is part of a bodyGroup.

And explicitly no tail after the chunk (`:5292-5294`). What we sent was sections 4-6, which is
production's own `compressionSourceOnly` mode by its own documentation — *"the marker/target/
directive (sections 4–6 of the builder)"*, `types/strategy.ts:972-973`. So the deviation was real
but narrower than the row implied: we were on production's optional arm, not on a shape production
never sends. What was missing was sections 1-3, and with them the frontier's cap.

**What changed.**

* **One projection, both requests.** `minisystem.source` calculates the head window, the unmerged
  frontier whose span starts before this range, and the raw middle once, for an L1 and a merge
  alike — the same code the merge fix already built (`cap_recalls`, `recall_memory`, the
  `l1_of`-aware `uncovered`). An L1 is now literally the same projection with `start` = its own
  chunk and `leaves` = `(chunk,)`.
* **`summarizer.l1_request`** joins prefix → marker → chunk → instruction. The prefix is sections
  1-3 in order; the retry rung passes an empty prefix and `source_only=True`.
* **The retry rung is derived, not queued.** `minisystem.step` computes it from the unit's own
  failure history — `retry = last_reason(log, target) == "refusal"` — exactly as the `overflow`
  reshape already was. `step()` still returns `(text | None, reason | None)`, the gate's reason
  strings and ordering are untouched, and the log is still the only store. The new `scope` value
  (`"six-section"` vs `"source-only"`) lives on the request dict beside `source` and `prefix`, and
  none of the three is a key of the dispatched payload, so the bound counts each exactly once.
* **Section 6 is doc-aware.** `formatReadingChunkInstruction` (`:250-266`) is implemented as
  `llm/prompts/reading_l1.txt`, verbatim, selected by `detectDocContext`'s own two L1-site guards: the
  chunk must BE one shard (production returns null unless every message in the chunk shares the
  group, `:9728-9735`) and the whole group must be at least twice the chunk (`:9746-9752`). The
  merge site applies neither guard (`:7166` fires on the leaf set alone); the two detections are
  kept as production has them rather than harmonized, and the asymmetry is recorded in `config.py`.
  This was the difference the previous round named by hand, and it is no longer outstanding.
* **Section 3 is implemented but not reached.** It falls out of the shared projection, and the
  merge's non-empty raw middle is already pinned by a test. The L1's is empty in any normal run
  because `work` derives an L1 for every chunk in order: the chunk before the current one is either
  already minted (so it arrives as its pair) or is the current one. It is non-empty only when an L1
  is taken out of order — the demand path's holdback bypass, or a terminal-debt clear — and that
  path is not pinned by a test. Recorded as such in `config.py` rather than claimed as exercised.

**The load-bearing order, pinned.** Three checks exist because the rules they encode were learned
from failures, and each fails under the mutation that reverts it:

| rule | why | pinned by | mutation tried |
|---|---|---|---|
| head precedes the pairs | the reverse order caused the "68 initiations" runaway false memories (`:5268-5278`) | `test_l1_request_is_the_six_sections` | head moved after the pairs → 9 checks fail, including the head-first one |
| pairs come from the unmerged frontier only | children plus parent doubles the prompt unboundedly; ~500 L1s blew a 200k window at chunk 118 of a 4000-message import (`:5305-5311`) | `test_l1_prior_pairs_are_the_unmerged_frontier` | frontier filter dropped → 3 checks fail, including "the six children do NOT" |
| no tail after the chunk | future information would leak into the KV state and corrupt as-of framing (`:5292-5294`) | `test_l1_request_is_the_six_sections` | (the head-after-pairs move fails it too, since the prefix lands after the directive) |
| newest-first cap, chronological re-sort | one oversized pair must not hide smaller siblings behind it (`:2489-2514`) | `test_l1_ladder_caps_newest_first_and_resorts` | — |

Both mutations were applied in scratch copies with the originals hashed before and after
(`llm/summarizer.py 50adb570…`, `minisystem.py 7224ae53…`) and re-checked byte-identical on revert;
`python3 test_summarizer.py` exits 0 with 323 checks on the restored tree.

**§1.2 rows this fix overwrites** (the inventory is the state as audited; these entries are no
longer true):

| §1.2 row | then | now |
|---|---|---|
| L1 request carries the source only | ours: present — "`llm/summarizer.py:50-52`", and the row's verdict was that this is a deliberate simplification | ours: no. The L1 carries sections 1-6; source-only is the refusal rung |
| Head window as a request section | ours: absent | ours: present, first, and pinned against being moved |
| Prior recall pairs and `capRecallPairs` on the L1 path | ours: absent — "moot while the request is source-only" | ours: present, capped ÷10, newest-first with `continue`-not-`break` |
| `formatReadingChunkInstruction` on the L1 | ours: absent, named as an unimplemented difference | ours: present, with both of production's L1-site guards |

**Left open by this fix** (each recorded in `config.py`, not merely here): production's
per-section participant turns — the head and raw middle as the original participants, the pairs as
`Context Manager` / the agent, the marker and directive as `Context Manager` (`:5392-5504`) — which
this boundary joins into one user turn; the rest of the L1 refusal ladder (the three recall-curve
variants, `:2893-3008`, and the split-stitch fallback, `:6186-6191`), both needing a provider that
actually refuses; the thinking-block stripping (`:5346`, `:5464`) and same-participant collapsing
(`:5503-5520`) production does on replayed messages, neither of which has anything to act on here;
the L1 raw middle as a *reachable* case; and, unchanged and independent of this fix, the merge
picker's missing contiguity/span/interior-run rules.

**One consequence measured, and one judgment call.** L1 requests now carry context, so their
pre-send bound grew. `Summarizer.admit` gates on the serialized dispatched payload plus the output
ceiling (1 600 here) against the 20 000-byte budget, so the honest unit is bytes:

| condition | L1 requests | worst bound | of budget | refused |
|---|---|---|---|---|
| shipped demo, 8 turns (mock digests) | 3 | 3 294 B | 16.5% | 0 |
| 40 turns, mock digests (7-token memories) | 31 | 5 953 B | 29.8% | 0 |
| 40 turns, memories at production's 200-token target, merges kept up | 31 | 16 033 B | 80.2% | 0 |
| 40 turns, same, through `run()` with one work item per message | 17 | 11 734 B | 58.7% | 0 |

The bound is set by the *frontier*, not by the recall budget: in a loop that drains merges the
frontier's steady state is the merge threshold (6 summaries) and `capRecallPairs` never binds.
The budget was **not** raised, and `config.py` states the truth rather than accommodating the
shape. It becomes a real constraint only if the frontier is allowed to fill its 10 000-token
recall budget, which happens when merges lag — our `tick` spends one work item per message where
production's compression loop drains, so under sustained load the frontier grows faster than
merges retire it; measured at that point, 48 of 59 L1 requests in one 120-turn run were over bound
(bound 19 155-59 057 against 18 400 of usable budget, frontier 36-50 summaries / ~10 000 priced
tokens, the cap binding). The demo never gets near that: the refuse-to-fit state needs the
frontier to fill the whole recall budget, which is 10 000 *tokens* of memories and therefore
roughly 40 000 *bytes* of prefix against a 20 000-byte budget — the structural mismatch being that
the recall budget is denominated in tokens while admission is denominated in fail-closed bytes,
and a recall pair costs about five bytes per priced token. Production has the same mismatch
(`compressionRequestInputBoundTokens` is bytes compared against a token-named budget,
`:2636-2649` against `:3015`); what keeps it from biting there is that its merges retire the
frontier faster than its compression loop grows it. So this is a finding about the frontier's
drain rate and about a ×10-scaled budget, not about the six-section shape — reported rather than
papered over, and left as the caller's call.

---

## 9. What may be compressed at all, and the wide-span merge guard

Two gaps, one of them the last `MISSING` row in `config.py`. Files touched: `config.py`
(723 -> 758), `minisystem.py` (707 -> 904), `test_summarizer.py` (1395 -> 1587), `README.md`
(524 -> 559), `AUDIT.md` (this section). `connectome_min.py` was not touched (159 lines before and
after) and its four documented runs are **byte-identical** to the four captures;
`llm/summarizer.py` was not touched either (382 lines) — the request builder already carried all six
sections, it was the derivation that never filled section 1.

**What production actually does** (re-read at the cited lines before any edit). The chunker is not
handed the store. `rebuildChunks` chunks `getCompressibleMessages(store).filter(m => !consumed)`
(`:10054-10056`), and that list is built as (`:9967-9980`):

```ts
for (let i = 0; i < recentStart; i++) {          // the recent window is excluded outright
  if (i >= headStart && i < headEnd) continue;   // the head window is excluded
  if (pinned.has(i)) continue;                   // pinned positions are excluded
  out.push(messages[i]);
}
```

So a head-window, recent-window or pinned message is never inside a chunk there: never compressed,
never carrying an L1, never available to a fold. The head builder then warns when a head message
turns out to be covered by a live summary anyway — *"head-window message(s) are covered by live
summaries; rendering them via recall pairs (ownership wins) … likely store head-boundary drift"*
(`:5394-5408`) — which is the drift case in production and the *normal* case here, because our
chunker runs over every message.

The merge side is `contiguousMergeCandidates` (`:6654-6688`). Two candidate filters sit in it: a
candidate whose `sourceRange` no longer resolves to live messages "can NEVER merge" and is warned
as `permanently unmergeable, frontier debt` rather than dropped silently (that silent drop was
itself a review finding, `:6660-6667`), and a candidate whose span in messages exceeds
`spanBase * mergeK ** max(0, level - 3)` is quarantined as wide for its level (`:6678-6686`). The
scaling is load-bearing and has its own incident behind it: *"mythos 2026-08 — every L4 spans
3.0k–6.9k messages > 1500, so an L5 was structurally impossible at any store state and the fold
floor sat ~23k above where one L5 puts it"* (`:6670-6677`).

**What changed.**

* **The compressible zone, as the policy's own boundary.** `minisystem.raw_zone`
  (`minisystem.py:318-332`) recomputes `plan_controlled_frontier`'s expression — `HEAD_CHUNKS`,
  `TAIL_TOKENS`, the pin ranges, the same accumulation — and `work()` (`:451`) refuses an L1 for
  every chunk in it. The frozen policy does not export its raw zone, so this is a second
  computation of one boundary, which is exactly the kind of copy that drifts silently; the test
  checks it against the policy's *behaviour* instead of its text: under a window no plan can fit,
  `fold_pass` folds every foldable chunk and a raw-zone chunk cannot move, because
  `fold_depth_cap` returns `-1` for it (`connectome_min.py:56-59`). Both directions are asserted,
  so the check is not vacuous.
* **The overlap guard became a span test.** `work()` used to ask `c.id not in a["l1_of"]`; the
  audit recorded what that cannot see (§1.3, `AUDIT.md:116`). `covered_by_l1`
  (`minisystem.py:356-381`) is production's three arms as one test — exact (`findExactL1`,
  `:5142-5155`), fully covered (`:5164-5181`) and partial overlap (`:5183-5203`) — and
  `overlap_blocked` (`:383-389`) is the loud half, printed by the demo beside the terminal debt
  instead of being a silent drop. The span is resolved at **mint time**: production stamps every
  summary with the `sourceRange` of its source messages (`:5238`) and keeps it, so coverage stays
  a fact about messages; our mint event records chunk ids, and the chunker is a pure projection of
  the log, so `l1_spans` (`:335-353`) re-derives `chunk_spans(log[:i])` at the mint's own index and
  recovers the boundaries as they stood when the memory was written.
* **A shard needs an ordinal.** `chunk_spans` (`:296-308`) gives every chunk `(first, last,
  ordinal)`. Body-group shards are slices of ONE message, so they share its message range, and
  without the ordinal each shard would "cover" its siblings and only the first of a document's
  three shards could ever be compressed. Production's shards are separate ingress chunks carrying
  a `shardIndex` (`:1578-1583`), so the ordinal is that field by another name.
* **The merge guard.** `span_limit` / `merge_span` / `merge_exclusions` (`:398-435`) implement both
  filters, over the same unmerged candidates `work()` selects from, with production's own two
  reason strings. The base is scaled `1500 -> 150` — the same /10 as `targetChunkTokens`,
  `summaryTargetTokens` and the two budgets — and the exponent is production's `max(0, level - 3)`
  with `k = mergeThreshold`. `config.py`'s row is no longer `MISSING`.

**The head section is no longer inert, probed rather than inferred.** `source()` builds section 1
from `uncovered(head chunks)`; while every closed chunk got an L1, the head chunk always had one,
so the section was empty in every live session and the head-before-pairs rule it exists to protect
was protected by nothing. `test_head_section_in_a_real_l1_request` drives three ticks of a real
session, takes the third request off the boundary and asserts that the head chunk's text is the
request's first prefix section, that the prior recall pairs are present after it, and that the same
order holds inside the assembled user turn. The old suite did pin the head-first property, but only
in a fixture that hand-emitted no mint for the head chunk (`test_request_scope_in_a_session`) —
which is why a live session could be inert without any test noticing.

**`python3 minisystem.py`, diffed hunk by hunk against the pre-fix capture.** The plans are
*identical* for turns 0-7 (`382, 764, 832, 1118, 1581, 1908, 2200, 2274`, same branches, same
frontier profiles) and differ at turn 8 only. Measured with a traced `plan_controlled_frontier` on
both trees:

| turn | W | before | after |
|---|---|---|---|
| 0-7 | 2300 | identical: `bootstrap 382`, `adopt-ideal 764/832/1118/1581/1908`, `hold 2200`, `adopt-ideal 2274` | identical |
| 8 | 1200 | `adopt-ideal 2188`, frontier `{L0:6, L2:3}` | `adopt-ideal 2280`, frontier `{L0:6, L1:3}` |

* **Four L1 units the derivation no longer spends** (`L1:c0`, `L1:c1`, `L1:c2`, `L1:c7`) and the
  L2 they fed are gone from the transcript: five units, seven calls. `c0` is the head chunk, so it
  is the one the old rule could never have folded at all. That they changed no plan for eight turns
  is the point — the old derivation was producing L1s nothing folded.
* **`compression calls spent: 14 -> 11`.** The derivation's own reduction is 14 - 7 = 7 calls;
  re-keying the scripted failure scenarios onto the chunks that now reach the boundary (below) adds
  4 back (`c3` and `c5` now take three attempts each instead of one). Net 11.
* **`REFUSED: turn 8: 2188 -> 2280 tokens` (+92, +4.2%), and the escalation turn did not move.**
  The refused plan is `raw:c0,c1,c2,c4,c7,c8` = 2142 priced tokens plus three L1 recalls of 46
  tokens each (production's `recallPairCost` = the 6-word label plus the mock's 40-word digest),
  `2142 + 138 = 2280`. Before, the same three chunks were covered by ONE L2 recall: `2142 + 46 =
  2188`. The +92 is exactly the two extra recall pairs, and it is the fold-floor consequence in
  the brief: the L2 that consolidated six L1s no longer exists, so the same stretch of history
  costs 92 more tokens at the ideal cut. It is not enough to escalate a turn earlier at this
  window (W = 1200 is crossed either way), and the demo's phase-1 plans are unaffected because the
  frontier never needed more than one folded chunk before the cut.
* **`model usage: 9963 -> 11834 in`, `249 -> 137 out`.** Input rose 18.8% on 21% fewer calls, i.e.
  **~51% more bytes per request** (712 -> 1076 tokens/call): every L1 request now carries the head
  chunk raw (354 tokens in this session) plus the prior recall pairs, which is what section 1 is
  for. Output fell with the mint count.
* **`mints=8 -> 3`, `fails=6 -> 8`, `events=59 -> 56`.** Eight mints were 7 L1s plus 1 L2; three
  are 3 L1s. `fails` rose because the re-keyed script adds the two `provider_error` attempts.
* **`frontier (L0 = raw): L0:6 L2:3 -> L0:6 L1:3`** — the same six raw chunks and the same three
  covered ones, one level shallower.
* **The demo no longer runs a merge at all.** This is a finding, not a detail: 36 messages over 16
  turns now bring only four chunks (`c3`-`c6`) to the boundary, and a base-6 merge needs six
  unmerged L1s, so `done L2:L1-c0+...` cannot appear. The merge request keeps its own tests
  (prefix, one-level-deeper target, instruction last, reading mode) and `work()`'s merge path is
  exercised by `test_merge_span_guard`'s fixture, but the *demo* no longer shows a merge end to
  end. The fix for that is to lengthen phase 1 (12 turns is enough: six L1s and one L2), and it was
  left undone on purpose — changing the session's length would make this before/after table
  incomparable, which is the measurement the brief asked for. It is the caller's call.
* **The scripted failure scenarios were re-keyed, and that is a judgment call.** `MockModel`'s plan
  is keyed by chunk id, and the ids that reach the boundary moved from `c0`-`c7` (all but the terminal `c4`) to
  `c3,c4,c5,c6`. Left alone, the demo would have silently stopped exercising the source-only rung
  and the transient provider path — two of the six policy rows — so the keys moved to `c3`
  (refusal x2), `c4` (tool_call + truncated), `c5` (provider_error x2) and `c6` (overflow), which
  keeps all four scripted classes firing. The line that says so is
  `failed attempts by reason: overflowx1, provider_errorx2, refusalx2, tool_callx1, truncatedx2`,
  and `test_demo_story` anchors it, so a future move that drops a scenario fails a test instead of
  quietly shrinking the demo.
* **Two new report lines, both about honesty rather than behaviour**: `compressible zone: 4 of 9
  chunks (5 protected by head/tail/pin: ...)`, `overlap-blocked chunks ...: none -- no
  re-derivation moved a chunk boundary here`, and `merge candidates excluded ...: none -- every
  candidate spans 150 msgs or fewer`. The last two say *why* they are empty, because "none" on its
  own reads as "not implemented".

**Two guards the demo cannot exercise, and why.**

* *The overlap guard.* The demo's own chunker does drift — image stripping re-prices old messages,
  and the demo's chunk list changes at message 25 for exactly that reason — but no L1 exists before
  that drift, so `overlap_blocked` is empty for the whole run. The fixture in
  `test_overlap_guard_is_a_span_test` uses the real mechanism rather than a staged one: a document
  message shards into three chunks while its image is live, an L1 is minted over the third slice,
  and once its depth passes `IMAGE_STRIP_DEPTH_TOKENS` the image is stripped, the message drops under
  `2 x targetChunkTokens`, the shard group collapses and every id after it moves by two. The L1
  then owns messages that `c1` -- a chunk with no L1 of its own -- now covers, which is precisely
  the state the id test called compressible.
* *The merge guard.* It cannot bind at demo scale, and that is arithmetic rather than a missing
  test: a demo chunk closes at 300 tokens over 4 messages, so a healthy L2 spans 24 messages and
  the demo's 9 chunks never build an L3 (144 messages) at all. Every candidate is 150+ messages
  inside its limit. The fixture builds the case on a 1004-message session: an L4 spanning 603
  messages is admitted (limit 900) while one spanning 1003 is quarantined, and a 603-message L2 is
  quarantined with no scaling involved. `config.py`, `README.md` and this section all say so.

**Mutations** (each in a scratch copy, `sha256` of `minisystem.py` recorded before and after the
edit, `__pycache__` cleared first, the original restored and re-checked byte-identical and green;
`python3 test_summarizer.py` exits 0 with 361 checks on the restored tree):

| mutation | edit | result |
|---|---|---|
| 1a | `raw_zone` keeps the head window but not the tail | 6 checks fail, incl. "the zone is the head chunk plus the tail window" and the demo anchors |
| 1b | the overlap test reverted to `c.id not in a["l1_of"]` | "so no L1 work is derived for it -- and c1 is the only compressible chunk here" fails (the span-overlap check) |
| 2a | `span_limit` returns the base for every level (scaling dropped) | 5 checks fail, incl. "so six eligible candidates do merge" -- the higher-level reachability check |
| 2b | the merge span filter dropped from `work()` | "so five eligible candidates are not a merge" fails (the wide-span check) |

**§1 rows this fix overwrites** (the inventory is the state as audited; these entries are no longer
true):

| §1 row | then | now |
|---|---|---|
| `mergeMaxSourceSpanMessages` (`AUDIT.md:48`) | ours: absent — "`config.py:159-164` (MISSING); no span check in `work()`" | ours: present, both filters, scaled base 150; `config.py` has no `MISSING` row left |
| head window raw, never chunked (`:61`) | ours: partial — "**ours still mints an L1 for the head chunk** (`work()` excludes only the holdback), so the archive holds a memory production would never write" | ours: no. The head chunk, the tail chunks and every pinned chunk get no L1, and the head section of a request is no longer inert |
| tail window raw and stable (`:62`) | ours: "scale only" | ours: the tail is in the same zone as the head, so it is not compressed either; the scale is still a scale |
| L1 overlap guard (`:116`) | ours: partial — "an id test, not a **span** test" | ours: the span test, resolved at mint time, with the loud half as a projection |
| §4 item 11 (the wide-span guard) | "not fixed … ~8 lines, as `config.py` estimates" | fixed; the estimate was ~8 lines for the filter; the filter plus its two projections are 44 lines here, before their tests |

**What could not be verified, and what is a judgment call.**

* The zone claim is a *reading* of production, not a run: no store, no provider, no chronicle host
  here. What is verified is the artifact's side — that its compressible set never overlaps what the
  frozen policy protects, checked against the policy's fold behaviour.
* The scale `1500 -> 150` is a judgment call with an argument, not a measurement: message-count
  limits do not scale by the same argument as token budgets, and 1500 messages would be unreachable
  at every level of a 36-message demo. It is the /10 used everywhere else here, stated as such in
  `config.py`. A caller who disagrees can move one constant.
* The brief's characterization of `_overlapBlocked` (`:5185-5198`, "a chunk whose messages are
  already covered by a live L1 is blocked and added to `_overlapBlocked`") merges two arms that
  production keeps apart: the fully-covered arm (`:5164-5181`) marks the chunk compressed and warns
  but does *not* add to `_overlapBlocked`, and the set is written only by the partial-overlap arm
  (`:5190-5203`). Both are implemented here, as one test with an `exact` flag, and only the
  non-exact arm is reported. `AUDIT.md:116`'s line range is short by the same two lines.
* "Never fold" is exactly true of the head chunk and of pinned chunks; a tail chunk is protected
  only while it is in the tail, and once it ages out it becomes foldable-by-cap but has no L1 until
  the next tick derives one. That is the state that moves the fold floor, and it is why the +92
  above is a real consequence rather than a bookkeeping artifact.
* Still not implemented, and unchanged by this section: the strictly-contiguous run split
  (`:6691-6705`) and the interior-run-at-2 escape (`:6718-6721`). `work()` still takes the first
  `mergeThreshold` *eligible* unmerged siblings at a level, which can bridge a hole; the two
  filters this section adds are applied before that choice, so a quarantined candidate no longer
  counts toward the six.
