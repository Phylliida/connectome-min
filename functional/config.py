"""Every knob in one place: ours, production's default, why it exists, and the audit status.

    python3 config.py            # the full table, grouped
    python3 config.py --gaps     # only what we do not implement, or implement differently

`connectome_min.py` (policy), `minisystem.py` (system) and `summarizer.py` (model boundary) import
their constants from here, so this file is both the tuning surface and the fidelity audit trail.
Production citations are the vendored `ref/context-manager` (library) and `ref/connectome-host`
(host overrides), both read at their defaults.

Status vocabulary:

    implemented   matches production's default path
    simplified    the same rule, coarser: demo scale, or one case where production has several
    MISSING       default-ON in production and absent here; `cost` says what that buys and what
                  closing it would take
    unmodelled    default-ON in production and deliberately not modelled here: the production
                  value is the whole entry, and `cost` says what the value does and why nothing in
                  our path can exercise it
    deviation     deliberately different from production's default
    ignored       default-OFF in production, so there is nothing to implement

`ignored` and `unmodelled` are the two answers to "what setting turns this on?" -- nothing, and
nothing (it is already on). Getting them backwards is the one error this table can make that a
reader cannot detect, so `audit_status_conflicts()` refuses it: a row claiming `ignored` must
record a production value that is OFF, and a row claiming `unmodelled` or `MISSING` must record
one that is ON. `report()` calls it, so a contradictory table makes `python3 config.py` exit
non-zero rather than print a falsehood.
"""
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    group: str
    name: str
    ours: object
    production: object
    status: str
    where: str
    why: str        # why the knob exists at all
    cost: str = ""  # for non-implemented: what the difference costs, and the size of a fix


KNOBS = (
    # ================= what the solver decides (adaptive resolution / kv-stable) =================
    Knob("solve", "mergeThreshold", 6, 6, "implemented", "types/strategy.ts:870",
         "the pyramid's branching factor sets the fold quantum: a summary is atomic over its "
         "span, so available fold depth arrives in groups of k. The value matches, and the run "
         "selection now does too: minisystem.work merges the first group of a STRICTLY CONTIGUOUS "
         "run of unmerged siblings (:6698), never bridges a hole, and consolidates an interior run "
         "-- one that can never grow, because summaries are produced at the live end -- at 2 "
         "instead of 6 (:6718-6721, the starvation fix). The wide-for-its-level and unresolved-"
         "source guards are their own row."),
    Knob("solve", "compressionSlackRatio", 0.1, 0.1, "implemented", "autobiographical.ts:925",
         "buys the dead band: target = W * (1 - slack). Without slack the solver would refold "
         "on every turn and never collect a cache hit."),
    Knob("solve", "qualityGapRatio", 0.35, 0.35, "implemented", "kv-control.ts:821",
         "lets a stuck, misallocated profile self-heal instead of fossilizing in the dead band: "
         "a big enough gap falls through to the ideal even though it is inside P."),
    Knob("solve", "reachTokens (P)", "= W", "unset -> W", "implemented", "kv-control.ts:819",
         "the trust region on per-turn perturbation -- the cost/continuity knob. Unset by "
         "default, so it never binds and the solver adopts the ideal outright."),
    Knob("solve", "foldAt / expandAt watermarks", "W / target", "goal ?? budget", "implemented",
         "kv-stable.ts:148-150",
         "the pair sets fold frequency and amplitude: fold above `foldAt`, shed to `target`, "
         "do nothing inside the band.",
         "identical under defaults; they diverge only when a host prepares a future window."),
    Knob("solve", "strictReach", False, False, "implemented", "kv-stable.ts:153",
         "turns P into a hard per-turn pace so that a window change never pays its whole "
         "perturbation bill in one turn.",
         "off unless the host calls the hot-settings API to prepare a window "
         "(autobiographical.ts:8842), which is why we can ignore it."),
    Knob("solve", "MAX_FOLD_LEVEL", 8, "8, capped by maxAvailableLevel(tree)", "implemented",
         "kv-control.ts:47,824",
         "bounds the shape prior's depth. The real cap is what production has BUILT: a level "
         "with no summary cannot be folded to, so phase A quietly floors at the tree's depth."),
    Knob("solve", "chunk salience", "computed per message", "computed per message", "implemented",
         "autobiographical.ts:8934-8995",
         "\"is the window the only copy?\" -- code, tool output, images and bare links fold "
         "cheap because the payload lives on disk/git/CDN; conversation exists nowhere else and "
         "stays expensive. It is the coefficient on information loss.",
         "IMPLEMENTED (minisystem.salience, ported line-for-line). One difference: production "
         "keys salience, caps and the frontier by MESSAGE; we aggregate a closed chunk and give "
         "it its cheapest message's salience -- equivalent for fold decisions, since a chunk's "
         "messages share an L1 and always move together."),

    # ============================ what the window looks like ============================
    Knob("layout", "targetChunkTokens", 300, 3000, "simplified", "types/strategy.ts:1479",
         "the unit of memory: a chunk exists only once it closes. It is never partial, with one "
         "exception: a sharded message closes whatever is pending first, as a partial chunk, "
         "because the alternative is dropping those messages from every chunk, silently "
         "(minisystem.spans).",
         "demo scale; the rule (close at the target, at least 4 messages) matches."),
    Knob("layout", "recentWindowTokens (tail)", 400, 30000, "simplified",
         "types/strategy.ts:1480",
         "the working-memory window: always raw, always stable, never folded -- the thing a "
         "fold must not disturb if the agent is to keep its footing.",
         "demo scale; both are token windows."),
    Knob("layout", "headWindowTokens", "1 chunk", "0 library / 4000 connectome-host",
         "simplified", "types/strategy.ts:1481, framework-strategy.ts:83",
         "the identity/preamble window: everything before it is history, this is not.",
         "we count chunks where production counts tokens. ~3 lines."),
    Knob("layout", "maxMessageTokens", 250, "0 library / 10000 connectome-host", "implemented",
         "types/strategy.ts:1500, framework-strategy.ts:90",
         "keeps one enormous message from dominating the accounting: its contribution to "
         "head/tail is capped at cap+50.",
         "IMPLEMENTED (scaled to 250). Production does both halves -- it TRUNCATES the content it "
         "emits (`truncateContent`, :11096, called behind `msgCap > 0` at :5041, :7613, :8154, "
         ":8172, :9148, :9505, :9574) and prices the message at `min(estimate, cap + 50)` "
         "(:9148) -- and so do we, in that order: minisystem.truncate_blocks cuts the text the "
         "archive renders, then spans() prices the cut blocks, so the plan can never price a "
         "window smaller than the text it stands for. The deliberate exception is body-group "
         "shards, which production excludes from this cap in as many words (:8051-8057) and we "
         "reach by sharding before the cap."),
    Knob("layout", "attachmentsIgnoreSize", "text-only pricing", True, "unmodelled",
         "types/strategy.ts:1483, autobiographical.ts:10013 and :10112",
         "attachment bytes are not re-sent, so counting them over-reserves the window: a chunk is "
         "priced from its text-only estimate (:10013) and the chunker closes on the same number "
         "(:10112) -- a live default-ON pricing rule, not an irrelevance.",
         "our messages carry no attachments, so the alternative branch (`store.estimateTokens`, "
         "which counts them) is the one we would never take: nothing to gate and nothing to "
         "scale."),
    Knob("layout", "maxLiveImages", 2, 6, "implemented", "types/strategy.ts:1501",
         "images are the most expensive and most externalized content, so they are the first "
         "thing worth shedding.",
         "IMPLEMENTED (scaled to 2). Images are stripped newest-first in minisystem.strip_images."),
    Knob("layout", "imageStripDepthTokens", 800, 30000, "implemented", "types/strategy.ts:1502",
         "the age half of the same rule: strip past this depth even if few images are live.",
         "IMPLEMENTED (scaled to 800 tokens). We strip on the post-strip estimate exactly "
         "as production budgets on it; the known tail-starvation bug is not reproduced."),
    Knob("layout", "bodyGroup sharding", "shards", "split messages > 2x targetChunkTokens",
         "simplified", "autobiographical.ts:1565-1572, 7978-8000",
         "a 500k-token document cannot be one chunk and cannot be several API messages without "
         "destroying the KV prefix: shards share a bodyGroupId and render concatenated.",
         "IMPLEMENTED at ingest (shard before the cap, as production does). Remaining "
         "difference: production renders a group's consecutive shards as ONE API message, so "
         "the unit is the group; we use adjacent per-shard units. Nearly identical cache cost, "
         "since the shards are adjacent. Our shards are SLICES of the message, as production's "
         "are (:1567-1580, combined at :7982-7990) -- a 234-token shard carries ~937 characters, "
         "not the whole document. That matters once a merge re-renders the raw layer: a merge over "
         "six shards of one 2000-token document is the largest request this artifact builds "
         "(bound 9104 of the 20000 budget, 46%) and it is admitted. Only the API-message shape "
         "above remains a difference."),
    Knob("layout", "positionedRecallPairs", True, True, "implemented", "types/strategy.ts:1503",
         "a recall pair sits where its span sat, so the rendered timeline still reads in order."),
    Knob("layout", "recallHeaderTemplate", "not emitted (unit identity)", "'[Recall {id}]'",
         "simplified", "types/strategy.ts:1504; emitted at autobiographical.ts:9468 (:9846)",
         "keeps a recall's identity in its rendered bytes, so a later pass recognises the fold "
         "instead of parsing prose back out of it.",
         "the identity duty is carried by our unit name (`recall:{summary}`), which is what a fold "
         "comparison reads; this live-view template's bytes are not materialised -- and, per the "
         "price row below, they are not priced either. ~2 lines to emit them, and no plan would "
         "move: production's picker does not weigh them. What a request DOES materialise is "
         "production's other, different header, `[CM] Recall memory <id>.`, in a merge's recall "
         "pairs (:7055, :7139; config's RECALL_MEMORY_HEADER)."),
    Knob("layout", "recall-pair price", "label + content", "label + content", "implemented",
         "autobiographical.ts:9006-9061 (recallPairCost), :9007",
         "what the fold planner weighs for one recall pair is the memory's own bytes: the "
         "question label plus the answer, and nothing else -- so the planner never pays for "
         "envelope it does not render.",
         "IMPLEMENTED term for term: `summaryContextLabel` + the answer content (minisystem."
         "archive). Two neighbouring numbers are deliberately NOT in it, and both were once "
         "charged here: the pair's Q-side header, which production EMITS (:9468) but does not "
         "price, and the +50 per summary in the prompt-side cap, which belongs to the "
         "`[CM] Recall memory <id>.` question turn that wraps a recall body (`capRecallPairs`, "
         ":2489). That +50 IS charged now, in the cap and nowhere else (minisystem.cap_recalls, "
         "config's RECALL_PAIR_OVERHEAD), so the fold floor still never pays it. Reasoning "
         "carriers (carrierPolicy 'full', :9029) are not modelled, so our answer side is prose "
         "alone."),

    # ============================ producing summaries ============================
    Knob("production", "hierarchical", True, True, "implemented", "autobiographical.ts:912",
         "the archive is a pyramid: deeper levels cover more raw content per recall token, "
         "which is what keeps the fold floor flat as a session grows."),
    Knob("production", "summaryTargetTokens", 200, 2000, "simplified", "autobiographical.ts:915",
         "the recall size target. The summarizer targets a size, not a ratio, so consolidation "
         "-- not compression -- is what buys context.",
         "scaled ÷10 like compressionContextBudgetTokens, and by that argument rather than an "
         "independent one: at demo scale a recall covers a few hundred raw tokens (the chunk "
         "target is 300), so production's 2000 would have the merge instruction ask for a memory "
         "larger than its whole source. IMPLEMENTED other than the scale: the number reaches the "
         "prompt as `{target_tokens}` (summarizer.TARGET_TOKENS) and the request ceiling derives "
         "from it as max(floor, target * 1.5) -- production's expression term for term, its "
         "16000 floor scaled /10 to 1600, so the FLOOR is what binds here too (see "
         "MAX_OUTPUT_TOKENS in summarizer.py). The memories are still the mock's "
         "40-word digest, because the demo's recall-pair prices are computed from their word "
         "count."),
    Knob("production", "l1HoldbackChunks", 1, 1, "implemented", "autobiographical.ts:3979",
         "the live edge is still in motion (edits, tool results landing, the episode "
         "unresolved), so the newest closed chunk waits for a newer one to close.",
         "IMPLEMENTED: the newest closed chunk is excluded from derived work until a newer one "
         "closes -- unless a picker produce op demanded it, which is the one way past the window "
         "in production too (a demanded chunk lands in `_demandedL1Chunks`, :998, and bypasses "
         "the holdback at :3992-3999, which is the window built at :3981-3985). Our demand path "
         "is the `produced` runs of connectome_min.plan_controlled_frontier, consumed by "
         "minisystem.work."),
    Knob("production", "L1 minting trigger", "pressure-gated (demo host)", "eager at chunk close",
         "deviation", "autobiographical.ts:9990",
         "when a memory is written: production mints an L1 the moment a chunk closes; the demo's "
         "host mints only while the last plan was over target (plus the demand path), because a "
         "memory nobody will fold this turn is a model call spent early -- the demand channel "
         "generalized from escalation to ordinary pressure.",
         "deliberate: lazy production makes model calls proportional to folding pressure instead of "
         "history length. The price is one turn of lag (the fold floor sits higher until the mint "
         "catches up) and the loss of production's freshness guarantee -- production wants the "
         "summary ready BEFORE the fold needs it, which is what speculativeProduction below is "
         "for. minisystem.work keeps eager as its default; the deviation is the host's choice of "
         "pressure=False, so the system supports both."),
    Knob("production", "speculativeProduction", "drained per tick", True, "implemented",
         "autobiographical.ts:926",
         "bottom-up pre-production: keep the forest deeper than the current budget needs, so a "
         "budget cut never wedges with no summaries to fold into.",
         "production's tick IS a drain: `checkMergeThresholdRecursive` cascades every level and "
         "recurses until a tick makes no progress, macrotask-paced so inbound events are not "
         "starved (:4149-4163). The demo's tick now drains too: one attempt per derivable unit, "
         "the work set re-derived between attempts, so a merge enqueued by a compress mints in "
         "the same tick and level availability no longer lags on a run. What stays unmodeled is "
         "the macrotask pacing (ours drains inline) and whether production retries a transient "
         "failure within one drain (ours waits a tick). Same single gap as "
         "`autoTickOnNewMessage` below, seen from the cascade side."),
    Knob("production", "autoTickOnNewMessage", "drained, both priorities per message",
         "true connectome-host",
         "simplified", "framework-strategy.ts:89, autobiographical.ts:4105",
         "memory formation rides the message cadence rather than the compile, so summaries are "
         "ready when a fold needs them.",
         "the CADENCE matches (the demo ticks after every message) and the TICK now does too -- "
         "a drain like `driveSpeculativeDrain` (:4105 -> :4120-4166), with one deliberate pacing "
         "difference: a unit gets ONE attempt per tick, so a retry waits a tick instead of "
         "tight-looping to terminal, and inbound events are never starved because the drain "
         "cannot spin. `simplified` for the pacing only."),
    Knob("production", "L1 prompt: marker + directive",
         "COMPRESSION_MARKER + formatInstruction", "COMPRESSION_MARKER + formatInstruction",
         "implemented", "autobiographical.ts:99-102, :5480, :6187, :192-206, :9683",
         "the summarizer is asked for a memory in two structural pieces: a marker turn saying a "
         "memory is about to be written, and a directive that names the voice, the details worth "
         "preserving and the anti-padding rule. Neither is text-searchable.",
         "IMPLEMENTED verbatim (prompts/marker.txt + prompts/l1_chunk.txt), and that is a "
         "correction: our L1 turn used to be `summaryUserPrompt` and our system turn "
         "`summarySystemPrompt`, two knobs with ZERO read sites in production -- `src/types/"
         "strategy.ts:1485-1486` declares them and nothing in `src/`, `bench/`, `test/` or "
         "`scripts/` sends them. What production sends is this marker plus "
         "`formatInstruction(targetTokens)`, wrapped around the chunk. One shape difference, "
         "forced by the source-only request: production pushes the marker as its own message "
         "ahead of the chunk, and our request has one user turn, so the marker is that turn's "
         "leading paragraph. Production ALSO serves the host's live identity prompt as the "
         "request's system turn (:5604, spread only when `ctx.systemPrompt` exists), which "
         "connectome-host/framework-strategy supplies; we have no live agent identity, so "
         "prompts/system.txt is a stand-in of our own and says so in its own header. NOT "
         "IMPLEMENTED: the reading-mode L1 instruction (`formatReadingChunkInstruction`, "
         ":250-266, reached from `detectDocContext` at :9721-9731), which fires when a chunk's "
         "messages are shards of one substantially larger message -- nothing in our L1 path "
         "consults body groups, so a summarizer handed a document shard here still adopts the "
         "document's voice. Its merge-side twin IS implemented (the `merge request scope` row)."),
    Knob("production", "wide-span merge guard", "150 x k^(level-3) messages",
         "1500 x k^(level-3) messages", "implemented",
         "autobiographical.ts:6654-6688",
         "a node that is wide FOR ITS LEVEL breaks what the level means (replay-era bridge "
         "summaries). The limit scales with level so L4/L5 stay reachable: a flat limit "
         "structurally forbids every consolidation above the level where it matches a healthy "
         "node's span, which is how the mythos store's L4s (3.0k-6.9k messages) made an L5 "
         "impossible and left the fold floor ~23k tokens too high.",
         "IMPLEMENTED (minisystem.span_limit, merge_span, merge_exclusions) with both of the "
         "function's candidate filters: the wide-for-its-level quarantine and the adjacent one "
         "for a span whose source no longer resolves -- `permanently unmergeable, frontier debt` "
         "(:6660-6667), a silent drop until a review made it loud. The base is scaled 1500 -> 150, "
         "the same /10 as every other size dimension here (targetChunkTokens, summaryTargetTokens, "
         "the context and recall budgets), and the scaling exponent is production's own "
         "`max(0, level - 3)` with k = mergeThreshold. IT CANNOT BIND IN THE SHIPPED DEMO, and "
         "that is arithmetic rather than a missing test: a demo chunk closes at 300 tokens over "
         "4 messages, so a healthy L2 spans 24 messages and the demo's 9 chunks never build an "
         "L3 (144 messages) at all -- every candidate is 150+ messages inside its limit. It is "
         "exercised by a synthetic fixture instead, on a 1004-message session where an L4 "
         "spanning 603 messages is admitted (limit 900) while one spanning 1003 is quarantined, which "
         "is also the check that fails if the level scaling is dropped. What remains open is the "
         "rest of the same function: the strictly contiguous run split (:6691-6705) and the "
         "interior-run 2-escape (:6718-6721) -- see the `mergeThreshold` row."),
    Knob("production", "compressible zone", "head/tail/pinned chunks excluded",
         "getCompressibleMessages fed to the chunker", "implemented",
         "autobiographical.ts:9967-9980, :10054-10056",
         "the head window, the recent window and every pinned position are not compressed at all "
         "-- the chunker is handed the compressible messages rather than the store, so a chunk "
         "there is never created, never compressed and never has an L1 for a fold to use.",
         "IMPLEMENTED (minisystem.raw_zone + work). Our chunker runs over every message, because "
         "the chunk projection is also what prices the window and what renders raw units, so the "
         "boundary is applied to the DERIVATION instead of to the chunker. The boundary itself is "
         "`connectome_min.plan_controlled_frontier`'s own expression -- HEAD_CHUNKS, TAIL_TOKENS "
         "and the pin ranges -- recomputed in the system layer because the frozen policy does not "
         "export its raw zone, and cross-checked against the policy's actual fold behaviour by a "
         "test rather than trusted as a copy. Two consequences, both visible in the demo and "
         "reported in AUDIT.md §9: the head chunk and the tail chunks never get L1s, and a tail "
         "chunk that later ages out of the tail is foldable-by-cap but has no L1 until the next "
         "tick derives one, which moves the fold floor and therefore the plans."),
    Knob("production", "tool_use pairing guard", True, True, "implemented",
         "autobiographical.ts:10137-10144",
         "a chunk must not close on a message holding tool_use: the matching tool_result rides "
         "in the next message, and the provider rejects a request where they are split.",
         "IMPLEMENTED: a chunk never closes on a message holding tool_use, so the matching "
         "tool_result rides in the same chunk."),
    Knob("production", "l1/l2/l3BudgetTokens", None, 30000, "unmodelled",
         "autobiographical.ts:916-920, read at :9196-9198",
         "the legacy hierarchical renderer carries its summary mass by level budget -- L3 first, "
         "then L2, then L1, 30000 tokens each -- so the oldest and deepest memories are the ones "
         "that survive a squeeze.",
         "not inert in production, and not default-OFF: `conditionalLibraryDefaults` sets all "
         "three whenever `hierarchical` is on, which is the default (:912), and "
         "`selectHierarchical` reads them at :9196-9198. Nothing in our path selects by level "
         "budget -- the adaptive picker solves a frontier instead -- and a level budget is a "
         "different selection mechanism, not a coarser one, so there is no scaled value to carry."),

    # ============================ when compression fails ============================
    Knob("failure", "failure classes", "ACTIONS table",
         "refusal / unusable_empty / provider_error / admission_rejected / incomplete",
         "simplified", "autobiographical.ts:473-486",
         "an LLM call fails in a handful of distinguishable ways, and each one has a different "
         "right answer: retry, reshape, count, or stop. Naming them is what makes one policy "
         "possible instead of one per call site.",
         "ours is a table with the same vocabulary minus durable receipts. ~0 lines to align the "
         "names; the receipts are the production-grade part."),
    Knob("failure", "request shape", "sections 1-6, both requests", "experience replay",
         "implemented", "autobiographical.ts:5266-5294 (the six sections, L1), :6867-6891 (merge), "
         "types/strategy.ts:966-994 (the refusal ladder)",
         "production's default request replays the agent's continuous experience -- head window, "
         "prior recall pairs, raw middle, then the marker, the target chunk and the instruction. "
         "The refusal ladder exists to route around the refusals that shape can cause, and every "
         "rung after the first is a variant of it.",
         "IMPLEMENTED as those six sections for BOTH requests, which closes the last wholesale "
         "deviation this artifact had. An L1 request now carries the head window raw (the identity "
         "anchor, and the chunk's own grounding), the unmerged frontier as `[CM] Recall memory "
         "<id>.` pairs in source order under `compressionRecallBudgetTokens`, the raw middle "
         "between them and the chunk, then the marker, the chunk and the instruction -- in that "
         "order, which is not cosmetic: the head MUST precede the pairs because the reverse order "
         "made the head read as the most recent live conversation and compounded into runaway "
         "false memories (:5268-5278, the \"68 initiations\" incident), and the frontier rule "
         "exists because children plus their parent double the prompt size unboundedly "
         "(:5279-5284). Both properties, the section order and the absence of any tail after the "
         "chunk (:5292-5294), are pinned by test_summarizer.py on a live session's request. "
         "Sections 1-3 are assembled by one projection (`minisystem.source`), shared with the "
         "merge request, which now differs only in what its leaves are. NOT MODELLED, and the "
         "carve-out: production pushes each section as its OWN participant turn -- head and raw "
         "messages as the original participants, the pairs as `Context Manager` / the agent, the "
         "marker and directive as `Context Manager` (:5392-5504) -- while this boundary sends one "
         "system turn and one user turn, joining the sections with a blank line. The section "
         "order, the section text and the section boundaries are production's; only the "
         "participant envelope and the multi-turn framing are simplified, the same trade the merge "
         "request already made. Also not modelled: production STRIPS thinking blocks from every "
         "replayed message (:5392, :5470, :5485) and collapses consecutive same-participant turns "
         "(:5503-5520) -- we store no thinking blocks and have no participants to collapse."),
    Knob("failure", "L1 request scope", "six sections + source-only retry rung",
         "six sections + a five-rung refusal ladder", "simplified",
         "autobiographical.ts:5411-5466 (the arm), :5466-5489 (the fallback), "
         "types/strategy.ts:966-994",
         "the canonical request is the six sections; when it is refused, production reshapes "
         "rather than retries -- recall-curve variants that drop pairs, then the source-only "
         "shape, then the split-stitch halves.",
         "we keep ONE rung of that ladder, and it is production's own last-before-split rung: "
         "`compressionSourceOnly` (types/strategy.ts:966-980) sends sections 4-6 only, which is "
         "exactly the marker, the chunk and the directive. Its justification is an incident rather "
         "than a theory: L1 compression was refused by the Fable safety classifier when the "
         "request compiled the raw recent-window room traffic alongside the target chunk, the "
         "failing contribution was localized to that block as a class, and \"handing the "
         "summarizer only the thing it is summarizing is the correct scope; the copied drain "
         "cleared all quarantined chunks first-try and the summaries passed a full fidelity "
         "audit\" (:966-969). The rung is DERIVED from the unit's failure history, not queued: a "
         "recorded refusal moves the next attempt onto it, exactly as an `overflow` moves the next "
         "attempt to the reshape (`minisystem.step`), so `step()` still returns only "
         "`(text | None, reason | None)` and the log stays the only store. NOT MODELLED: the "
         "recall-curve variants (`buildRecallCurveVariants`, :2893-3008) that precede it, and the "
         "split-stitch fallback that follows it (:992, :6186-6191) -- both are rungs we would need a "
         "provider that actually refuses to exercise. A merge is unaffected either way: L1 only "
         "(:976)."),
    Knob("failure", "L1 doc-aware instruction", "formatReadingChunkInstruction",
         "formatReadingChunkInstruction", "implemented", "autobiographical.ts:250-266 (the text), "
         ":9696-9701 (getReadingChunkInstruction), :9716-9757 (detectDocContext), :5497-5506 (site)",
         "when the chunk is a portion of a substantially larger sharded message, the instruction "
         "asks what reading was like and what was learned, instead of framing the shard as events "
         "to narrate. For content heavily first-person from someone other than the agent (a "
         "user-shared document) the standard framing leads the model to take on the content "
         "author's voice; the reading framing forces it back to its own vantage point.",
         "IMPLEMENTED as section 6's doc-aware branch, and this was the last difference the "
         "previous round named by hand. The detection reuses the body-group tag `spans` already "
         "puts on a shard, with both guards production's L1 site adds: the chunk must BE one shard "
         "(production walks the chunk's messages and returns null unless every one shares the "
         "group, :9728-9735) and the whole group must be at least twice this chunk (:9746-9752). "
         "Recorded rather than silently harmonized: the MERGE site (:7166) applies neither guard "
         "-- it fires on the leaf set alone. Ours keeps production's two detections as they are, "
         "so a merge over shards still reads the way production's merge reads. `total_tokens` is "
         "the WHOLE group's size, not the shard's. The prompt is `prompts/reading_l1.txt`, "
         "verbatim."),
    Knob("failure", "L1 raw middle", "raw between head and chunk, when uncovered",
         "raw between head and chunk, when uncovered", "implemented",
         "autobiographical.ts:5445-5457",
         "section 3 exists so the request is contiguous: the head is the permanent prefix, the "
         "pairs stand for what they cover, and whatever is left between the head window and the "
         "chunk is shown raw rather than silently dropped. \"usually empty\" in production, "
         "because chunking proceeds contiguously and summaries cover everything up to the chunk "
         "being processed.",
         "IMPLEMENTED, and it does fall out of the shared projection -- but it is the one section "
         "this artifact cannot exercise in a normal run: `work` derives an L1 for every chunk in "
         "order, so the chunk preceding the current one is either already minted (represented by "
         "its pair) or is the current one. It is NON-EMPTY only when an L1 is taken OUT of order "
         "-- the demand path's holdback bypass, where an escalated plan asks for the newest "
         "uncovered run while an older chunk is still unminted, and after a terminal-debt clear "
         "that makes an older chunk derivable again while a newer one is being compressed. The "
         "merge request's raw middle IS exercised in a normal run "
         "(`test_merge_prefix_is_the_prior_content_only` pins a non-empty one); this row records "
         "that the L1's is built by the same code and why a session does not reach it. It is NOT "
         "pinned by a test, which is the honest status."),
    Knob("failure", "mergeAttemptLimit", 3, 5, "simplified", "types/strategy.ts:880",
         "bounded retries: never infinite, never silent. The counter is persisted so a restart "
         "does not restart the count.",
         "value differs; the bounded rule, the reason and the terminal state all match."),
    Knob("failure", "MERGE_SERVER_ERROR_STREAK_LIMIT", 12, 12, "implemented",
         "autobiographical.ts:1023",
         "retryable provider faults are usually transient, so they must not burn attempts -- "
         "but a persistent one on an unchanged payload loops forever unless it is counted."),
    Knob("failure", "compressionContextBudgetTokens", 20000, 200000, "deviation",
         "types/strategy.ts:1013-1023, autobiographical.ts:3027-3068",
         "refuse to send a request that cannot fit: admission is checked BEFORE the call, so an "
         "oversized request costs nothing -- no provider round trip, no burned attempt, no "
         "operator waking up to a 400 that was knowable in advance.",
         "DEVIATION IN SCOPE, faithful in arithmetic. Production gates FALLBACK VARIANTS ONLY -- "
         "the knob's own doc: `The canonical request remains unchanged and is always attempted "
         "first; only fallback variants are gated` (:1013-1023); `compressionRefusalPlan` returns "
         "[] at a zero fallback limit and otherwise iterates `variants`, pricing each one "
         "(:3027-3042, :3065-3068), and the canonical path has no admission site at all. We have "
         "one rung and we gate it, so ours refuses a request production would have sent: that is "
         "a behavioural difference, not a simplification of production's rule, and it is kept "
         "deliberately -- a teaching artifact has no operator to widen a budget, and the reshape "
         "rung stands in for the fallback ladder. Everything below is production's, term for "
         "term. IMPLEMENTED as a pre-send check on the stateful boundary "
         "(`summarizer.Summarizer.admit`), so a refusal decided before dispatch lands in the same "
         "ledger as every other outcome. The input side is production's fail-closed BOUND, term "
         "for term: `request_input_bound` serializes the complete DISPATCHED request "
         "(`summarizer.wire_payload`: one system turn, one user turn -- the `source` field rides "
         "the request for inspectability and is already inside the user turn, so it is not "
         "serialized again), counts UTF-8 bytes, and adds 512 plus 128 per message for the role "
         "envelopes and special tokens the JSON does not represent "
         "(`compressionRequestInputBoundTokens`, :2636-2649, called at :2999, :3035, :3575). "
         "Bytes, not tokens: every token is at least one byte, so the bound needs no tokenizer and "
         "cannot under-count -- which is why the estimate is NOT the gate. The output side is the "
         "request's own max_tokens, production's `outputReserveTokens` term exactly (:3066). Over "
         "the budget the system sees `overflow` and nothing is dispatched, and the refusal "
         "carries both numbers on `Reply.detail` (bound and estimate): the reason alone cannot say "
         "which bound was crossed, or by how much the estimate misled. Scaled to the demo the way "
         "targetChunkTokens is (200k -> 20k), so the bound's room is 18400 bytes and the reserve "
         "1600 = 8% of the budget, production's share exactly (16000 of 200000, 8x its target as "
         "ours is 8x ours). What still differs: production rejects and waits for an operator to "
         "widen the request budget; we halve the source on the retry."),
    Knob("failure", "compressionRecallBudgetTokens", 10000, 100000, "simplified",
         "types/strategy.ts:917, read at autobiographical.ts:5416 and :7017",
         "the recall ladder INSIDE a compression request can itself overflow the window; the cap "
         "evicts newest-first, so proximate context survives -- 100k by default, and zeroed "
         "outright under source-only requests (`sourceOnly ? 0 : ...`).",
         "IMPLEMENTED for the merge prefix, absent exactly where production zeroes it: our L1 "
         "request is source-only, which IS production's `recallBudget = sourceOnly ? 0 : ...` "
         "branch (:5416), so the ladder it would cap does not exist there; the merge prefix reads "
         "it the way :7017 does, and `minisystem.cap_recalls` walks it newest-first exactly as "
         "`capRecallPairs` does (:2489-2514) -- `continue`, not `break`, on a pair too big for "
         "what is left, and the kept set re-sorted chronologically. This row used to say "
         "`unmodelled` because a source-only request has no ladder to cap; merges carry a prefix "
         "now, so that reasoning is gone. Scaled /10 (100k -> 10k), like compressionContextBudget "
         "and summaryTargetTokens: half of compressionContextBudgetTokens at both scales. Two "
         "differences remain: production prices a pair as `(s.tokens ?? chars/4) + 50 + envelope` "
         "and we price our recall-pair price (label + content, above) plus the same 50 -- the same "
         "rule on our own unit; and the per-attempt halving is NOT modelled, so a merge that "
         "overflows is reshaped by the one rung we have instead of retrying with a ladder halved "
         "`max(8000, configured * 0.5 ** min(mergeAttempts, 4))` (:7017-7020). The expansion is "
         "also what makes a merge request big: the demo's L2 merge goes from a 706-character "
         "source to 11694 characters of raw L0, so its pre-send bound goes 2263 -> 13399 bytes "
         "and its estimate 337 -> 3092 -- admitted, with 5001 bytes of budget left. A merge over "
         "many shards of one long document is the largest request here (bound 9104 of 20000) and "
         "is admitted too; the cap's unit is the prefix ladder, not the target, so it does not "
         "bound that growth."),
    Knob("failure", "compressionCacheMarkers / TTL", None, "true / '1h'", "unmodelled",
         "autobiographical.ts:929-930, read at :2553, sent at :5621 and :7309",
         "the mint request's prefix is stable and append-mostly, so its seams are worth an hour "
         "of prompt cache: the writer places them at end-of-head, the last L>=2 pair and the last "
         "pair, and suppresses them on a capped ladder.",
         "cost only, no layout effect, and nothing in our path places a seam -- but it is not "
         "default-OFF: `conditionalLibraryDefaults` sets both unconditionally, outside the "
         "adaptiveResolution gate (:929-930). Recorded for the reader who is reconstructing what "
         "the default configuration actually switches on."),
    Knob("failure", "overBudgetGraceRatio", 0.02, 0.02, "implemented", "types/strategy.ts:1507",
         "a small tolerance above W so the wall is not brittle, and a hard throw beyond it: "
         "the enforced wall is W * 1.02 and crossing it raises OverBudgetError.",
         "IMPLEMENTED: minisystem raises Refused past W*(1+grace); a plan between W and the "
         "wall renders over W, which the demo prints as OVER W (inside grace)."),
    Knob("failure", "retry-only no-tools line", "ours", "production's own sentence",
         "simplified", "autobiographical.ts:5908-5915",
         "a model that answers with a tool call cannot be told 'no tools' structurally (with "
         "tool_choice:none it emits nothing at all), so it has to be told in prose on the retry.",
         "we invented the wording; the mechanism matches. One placement difference: production "
         "appends its line to the INSTRUCTION text on a merge retry (:7183-7200), and we append "
         "ours to the shown source, which in a merge request means between the target and the "
         "instruction -- the model reads it either way, but the bytes are not production's."),
    Knob("failure", "quarantine alarm", "derived 'stalled work'", "15-minute repeating alarm",
         "deviation", "types/strategy.ts:603-608",
         "quarantine is deferred debt, never a resting state: it keeps a span raw and the fold "
         "floor climbing, so it must keep announcing itself.",
         "ours is a projection printed on demand; production persists records and repeats an "
         "alarm until an operator clears it."),
    Knob("failure", "quarantine clearing", "one log event", "clearCompressionRefusalQuarantine",
         "implemented", "autobiographical.ts:1263-1306, :3441-3459 (called at :6459, :6530)",
         "quarantine is deferred debt, never a resting state: a record that can never be cleared "
         "makes a span permanent raw and the fold floor permanent, so production ships an explicit "
         "operator escape hatch -- and an automatic one, firing when the same span later "
         "compresses successfully under a newer request shape.",
         "IMPLEMENTED, both halves: minisystem.clear_debt emits a `clear` event (the log is the "
         "only store, so the escape hatch is a record and not a mutation of the quarantine map), "
         "and `stalled` clears itself when the unit mints. Both reset what the debt is a "
         "projection OF -- the unit's failures since the last clear -- which is why a cleared unit "
         "is derivable work again."),


    # ============================ the model call ============================
    Knob("model", "compressionModel", "MockModel | HttpModel", "the session's routed model",
         "deviation", "packages/compaction/compaction-autobiographical/src/index.ts:453",
         "autobiographical memory is the agent writing about its own history, so the library "
         "refuses a substitute voice.",
         "still a stand-in, but a real boundary now: `summarizer.py` assembles the request, "
         "classifies provider outcomes and gates the disposition. production routes through the "
         "harness's normal lanes (credentials, retry, accounting); ours is a stdlib HTTP client "
         "where production has a whole provider membrane. `python3 minisystem.py` stays offline "
         "on MockModel, so the demo needs no credentials."),
    Knob("model", "compression request assembly", "one builder, one shape",
         "one builder per call site", "simplified", "autobiographical.ts:5132, :6824",
         "the request decides what a summary can know, so who assembles it is a real design "
         "surface: production has an L1 builder and an L_n merge builder, both ending in the same "
         "instruction formatter (autobiographical.ts:277).",
         "IMPLEMENTED as `summarizer.build_request`, which both paths call: system prompt + "
         "directive, tools=None, scope recorded in the request itself. tools=None is our "
         "deviation: production declares the agent's live tools because a tools-less request "
         "replaying tool_use/tool_result history reads to the safety classifier as a "
         "foreign agent trace being duplicated, a deterministic refusal (:5631-5640); we replay no "
         "such history, and the tool_call reason is answered in prose on the retry (:5907-5935). "
         "Production's builders also emit the head window, the recall ladder and the raw recent "
         "window (sections 1-3); the source-only deviation above is what removes them from an L1, "
         "and it is asserted in test_summarizer.py at the system level, not just on the assembled "
         "request. The merge builder's own sections are NOT removed: a merge request carries the "
         "prefix and the one-level-deeper target (`merge request scope` above), and the assembly "
         "of both shapes is the same function and the same three-placeholder instruction."),
    Knob("model", "model boundary", "Reply | complete(request)",
         "membrane NormalizedRequest / NormalizedResponse", "simplified",
         "autobiographical.ts:2758-2794; @animalabs/membrane",
         "one protocol so the system never learns what a provider is: a request goes out, a "
         "normalized reply comes back, and the gate decides whether it is a memory.",
         "stdlib only: one POST, one response, no streaming, no credential refresh, no cache "
         "markers (compressionCacheMarkers/TTL stay unmodelled above). The failure vocabulary is "
         "ours, mapped from HTTP: 429/408/5xx/timeout/connection -> provider_error retryable, "
         "any other 4xx -> provider_error non-retryable, tool call -> tool_call, truncating stop "
         "-> truncated, empty body -> empty, refusal -> refusal. A live 400 context_length is "
         "one of those non-retryable provider_errors, and that is faithful rather than a gap: it "
         "is what a provider-side rejection looks like when the pre-send admission check "
         "(compressionContextBudgetTokens, below) should have caught the request and did not. The "
         "byte bound cannot under-count tokens -- that is why it is the gate -- so a 400 here "
         "means the budget sits above the provider's real window, or a gateway adds tokens we do "
         "not model. Production classifies a provider-side rejection the same way. NOT SHIPPED, deliberately: a second dialect (Anthropic's `/v1/messages`; "
         "one OpenAI-compatible client is what a reference implementation needs, and a dialect is "
         "a change to `build_wire`/`parse` alone), and any cancellation hook (without streaming a "
         "cancel cannot interrupt a request already on the wire, so rather than offer one that "
         "can only check before dispatch and after the reply, the boundary offers none -- a "
         "caller discards a stale result by version stamp, as minisystem.step() does). A "
         "malformed 2xx body is a provider_error, as production's `malformed_response` is "
         "(:2766-2772)."),
    Knob("model", "terminal-disposition gate", "implemented", "end_turn + nonempty text",
         "implemented", "autobiographical.ts:2758-2794, :474-486",
         "a truncated or interrupted generation must never become a permanent memory: the gate "
         "exists because a 163-char refusal preamble was persisted as an L4 parent over six real "
         "L3 children when only text-nonemptiness was checked.",
         "IMPLEMENTED (`summarizer.gate`): complete stop + usable text + no tool call, or a "
         "failure with the reason the policy table routes. One ordering difference, deliberate: "
         "a tool call is checked BEFORE empty text (production's `unusable_empty` wins there, "
         ":2786 vs :2788), because our retry line is keyed to the tool_call reason. A refusal "
         "outranks even its own text, as production's does (:2784-2786), and every other stop is "
         "`truncated` -- production's `incomplete`, one class covering max_tokens, abort, "
         "stop_sequence and a missing stop (:473-486, :2788-2792)."),
    Knob("model", "compression input estimate + calibration", "payload bytes, EMA-calibrated",
         "same estimator, plus a multiplier fed by the live session", "simplified",
         "autobiographical.ts:2630-2634, :2636-2649, :8345-8400",
         "an estimate is what makes a budget comparable across models and a bound is what makes it "
         "safe: production keeps both, and only the bound may refuse a request.",
         "IMPLEMENTED for the compression request. `summarizer.request_input_tokens` is the wire "
         "payload's UTF-8 bytes, which production labels metadata in as many words and defers rendered "
         "accounting to the provider (`estimateCompressionRequestTokens`, :2630-2634); it is "
         "surfaced as `Reply.estimated_tokens` and totalled in `usage()`, and nothing gates on it. "
         "`Summarizer.observe` keeps the closed loop (`reportRealInputTokens`, :8345-8400): an EMA "
         "of provider-reported input over that estimate, alpha 0.2 and band [0.6, 1.8], applied to "
         "future ESTIMATES only -- the bound never takes it, which is the point of having a bound. "
         "Production must arm its sampler once per compile because a turn makes many calls that "
         "each grow the request, so only the first answers the estimate: ratios of 2.0-2.3 drove "
         "the multiplier 1.0 -> 2.37 in minutes, inflating the fold floor 62k -> 108k (:8351-8357). "
         "Our pairing is structural -- the estimate comes from the very request whose reply "
         "carries the usage -- so that trap cannot arise here and no arm is needed. NOT MIRRORED, "
         "deliberately: production's multiplier targets the WINDOW estimator (the rendered "
         "context, `_lastCompileEstimate`, :8327-8330) and is fed from the session's main-model "
         "usage event (packages/compaction/compaction-autobiographical/src/index.ts:396-409). This "
         "artifact has no main-model usage event and no window estimator, and production warns "
         "that confusing the two lanes is what let a miscalibrated sample inflate the window; our "
         "multiplier therefore stays on the compression-request lane and never touches the fold "
         "floor."),
    Knob("model", "usage accounting", "in/out tokens + calls, per session",
         "provider usage per attempt, priced into the store", "simplified",
         "autobiographical.ts:2668-2698",
         "a compression that cannot be priced cannot be budgeted: production reads the "
         "provider's own usage and prices recall pairs from it.",
         "IMPLEMENTED as totals (`Summarizer.usage()`), so minisystem prints them at the end of "
         "the demo. MockModel reports the request's own byte estimate; HttpModel reports what the "
         "provider "
         "billed. The retryable / non-retryable fault split is kept there too, since production's "
         "streak and its attempt counter are fed by different halves of that class."),
    Knob("model", "summaryParticipant", None, "the agent's own name", "unmodelled",
         "framework-strategy.ts:130-132, read at autobiographical.ts:5206, :6859, :9443",
         "recollections must speak as the agent, not as a stranger -- the library's own fallback "
         "is the literal 'Claude', and the host overrides it with the agent's name.",
         "voice only, no layout effect -- and default-ON under connectome-host, which is what "
         "'the agent's own name' records. We model neither the participant field nor the voice: a "
         "mock digest has nobody speaking."),
    Knob("model", "ladder rungs", "2 (six-section L1 + source-only retry; canonical merge)",
         "canonical + 3 curve variants + source-only + split-stitch", "simplified",
         "types/strategy.ts:981,984,992,1011,1505",
         "each rung reshapes the request rather than retrying it: fewer recall pairs, only the "
         "source, halves of the chunk stitched back together.",
         "the rung set as it now stands: the canonical six-section request, then the one rung "
         "production's own incident data credits with clearing the whole class -- the source-only "
         "shape, sections 4-6 of the builder. Still not production's five rungs, and the gap is "
         "named rather than implied: the three recall-curve variants (`compressionRefusalCurve"
         "Fallbacks`, :3009-3011) and the split-stitch fallback (:992) are absent, because both "
         "need a provider that actually refuses to be exercised and a rung nothing can reach is a "
         "rung that cannot be checked. `compressionSourceOnlyFallback` (types:983) is what makes "
         "the source-only shape a rung rather than the shape -- it is the fallback PAST the curve "
         "variants -- so keeping it alone is keeping production's last useful rung, not a "
         "degradation of the first. The boundary makes the equivalence checkable: the gate returns "
         "the reason a would-be rung would have retried on, and the rung is derived from that same "
         "record. A merge is unaffected: production's degraded merge rungs (the refusal fallback "
         "and the halved recall budget) are retry-only and we do not model them -- see "
         "`merge request scope`."),
)

# ============================ demo parameters (not fidelity knobs) ============================
DEMO = {
    "window": 2300,      # W for the scripted session; production would be 65k-200k
    "cut_window": 1200,  # phase 2: the operator cuts the window (this breaks the fold floor)
    "reach": 1250,       # P used in the documented cascade run; P = W unless set
    "turns": 8,          # phase 1: normal budget (see AUDIT: the demo cannot reach a
                     # merge at this size -- 5 unmerged L1s against a candidacy of six)
    "extra_turns": 8,    # phase 2: after the window cut
    "fork_turn": 7,      # a fork lands mid-call here, to exercise the stale-version rule
                         # (the first turn lazy production has L1 work in flight: pressure starts
                         # after turn 6's plan crosses target, and phase 2 ends REFUSED at turn 8)
    "msg_words": 90,     # synthetic message size; 4 messages close a 300-token chunk
}

# Runtime defaults for the HTTP client and the offline mock -- not fidelity knobs, so they carry
# no audit row. They live here because config.py is the single place constants are looked for.
DIGEST_WORDS = 40                      # the mock's memories are priced by their word count
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"      # rstrip'd; /chat/completions is appended
DEFAULT_TIMEOUT = 120.0

# Rules from `config.py --gaps` that are now implemented (values scaled for the demo, with the
# production value beside them):
MIN_MSGS = 4                      # production: targetChunkTokens 3000 with a 4-message floor
RECALL_LABEL = "What do you remember from earlier?"      # summaryContextLabel (types:1496)
RECALL_MEMORY_HEADER = "[CM] Recall memory {id}."        # a request's recall pair Q-side (:7055, :7139)
RECALL_PAIR_OVERHEAD = 50         # the `+50` per pair inside capRecallPairs (autobiographical.ts:2489)
TRUNCATION_NOTE = "\n\n[truncated — original was {tokens} tokens]"     # truncateContent (:11108)
IMAGE_CHARS = 400                 # scaled from the renderer's 6400-char flat image cost
IMAGE_TOKENS = IMAGE_CHARS // 4   # the same image in the token domain (production: ~1600)
MAX_LIVE_IMAGES = 2               # production: 6      (types:1501)
IMAGE_STRIP_DEPTH_TOKENS = 800    # production: 30000  (types:1502)
MAX_MESSAGE_TOKENS = 250          # production: 10000 (connectome-host); 0 disables (library)
MAX_MESSAGE_HEADROOM = 50         # the `cap + 50` in production's min(estimate, cap + 50) (:9148)
GRACE = 0.02                      # overBudgetGraceRatio (types:1507)
TAIL_HOLDBACK = 1                 # l1HoldbackChunks (autobiographical.ts:3979)
MERGE_MAX_SOURCE_SPAN_MESSAGES = 150   # production: 1500 (mergeMaxSourceSpanMessages, :6654)

# Ours, as importable constants -- the single source of truth for both other files.
_K = {k.name: k.ours for k in KNOBS}
MERGE_THRESHOLD = _K["mergeThreshold"]                    # 6
CHUNK_TOKENS = _K["targetChunkTokens"]                    # 300
TAIL_TOKENS = _K["recentWindowTokens (tail)"]             # 400
HEAD_CHUNKS = 1                                           # ours: a chunk count, not a token window
SLACK = _K["compressionSlackRatio"]                       # 0.1
GAP_RATIO = _K["qualityGapRatio"]                         # 0.35
MAX_FOLD_LEVEL = _K["MAX_FOLD_LEVEL"]                     # 8
MAX_ATTEMPTS = _K["mergeAttemptLimit"]                    # 3
PROVIDER_STREAK = _K["MERGE_SERVER_ERROR_STREAK_LIMIT"]   # 12
CONTEXT_BUDGET = _K["compressionContextBudgetTokens"]     # 20000 (production: 200000)
SUMMARY_TARGET_TOKENS = _K["summaryTargetTokens"]         # 200 (production: 2000)
RECALL_BUDGET = _K["compressionRecallBudgetTokens"]       # 10000 (production: 100000)
# Admission's fail-closed bound (compressionRequestInputBoundTokens, autobiographical.ts:2636-2649):
# UTF-8 bytes of the complete dispatched request, plus a fixed and a per-message reserve for the
# role envelopes and special tokens the JSON does not represent.
BOUND_RESERVE = 512
BOUND_PER_MESSAGE = 128
# Estimator calibration (reportRealInputTokens, autobiographical.ts:8345-8400): alpha is
# production's, and so is the band -- outside it a sample is a structural mismatch, not evidence.
CALIBRATION_ALPHA = 0.2
CALIBRATION_BAND = (0.6, 1.8)
NO_TOOLS = "Reply with the recall text only; do not call tools."
ACTIONS = {"refusal": "retry", "empty": "retry", "truncated": "retry", "tool_call": "retry",
           "overflow": "shrink", "provider_error": "transient"}

# Closed so far (each wave in the order it landed): chunk salience, recall-pair pricing,
# maxMessageTokens, overBudgetGraceRatio, l1HoldbackChunks, image stripping, bodyGroup sharding,
# tool_use pairing guard; then, over the audit: the five mislabelled `ignored` rows, the
# recall-pair price (header -> label + content), maxMessageTokens truncation, the demand path
# (`produced` runs + the holdback bypass), the real L1 prompt (marker + formatInstruction), the
# fatal coverage invariant, and the two quarantine clears; then the merge request's three parts
# (prefix under the recall budget + one-level-deeper target + instruction last, with reading mode);
# then the L1 request's six sections -- head, unmerged frontier, raw middle, marker, chunk,
# instruction -- with the source-only shape demoted to the refusal rung production keeps it as;
# then the compressible zone (the head/tail/pin boundary applied to the derivation, which is what
# makes the head section carry text) and the wide-span merge guard; then the odometer run rule
# (strict contiguity + the interior escape at 2), the draining tick, lazy production as the
# host's pressure gate, and one pricing unit (payload bytes for estimate and bound alike).
# What is left is deliberately open:
PRIORITY = ("request participants",      # per-section participant turns, not one joined user turn
            "failure receipts")          # durable, inspectable records instead of log events


# A status that names the production DEFAULT has to agree with the value recorded beside it,
# or the table lies in a way a reader cannot check: `ignored` says "default-OFF, nothing to
# implement", `unmodelled`/`MISSING` say "default-ON, and here is what it does". Judge the value
# itself, which is what a reader sees: an explicit off, a zero, or an "unset"/"none" spelling is
# OFF; anything else -- True, a positive number, a setting name, a default object -- is ON.
_OFF_SPELLINGS = ("none", "unset", "false", "off", "0", "no")


def _production_is_on(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return not value.strip().lower().startswith(_OFF_SPELLINGS)
    return True


def audit_status_conflicts(knobs=KNOBS):
    """Rows whose status contradicts its own definition. Exported so a test can doctor a table
    and watch it be refused; `report()` calls it on the real one."""
    conflicts = []
    for k in knobs:
        if k.status == "ignored" and _production_is_on(k.production):
            conflicts.append(f"{k.name}: status `ignored` means default-OFF in production, but the "
                             f"recorded production value {k.production!r} is on")
        if k.status in ("unmodelled", "MISSING") and not _production_is_on(k.production):
            conflicts.append(f"{k.name}: status `{k.status}` means default-ON in production, but "
                             f"the recorded production value {k.production!r} is off")
    return conflicts


def report(gaps_only=False, knobs=KNOBS):
    conflicts = audit_status_conflicts(knobs)
    if conflicts:
        raise SystemExit("config.py refuses this table -- a status contradicts its own "
                         "definition:\n  " + "\n  ".join(conflicts))
    marks = {"implemented": "ok  ", "simplified": "~   ", "MISSING": "MISS", "deviation": "DEV ",
             "unmodelled": "--  ", "ignored": "-   "}
    for group in ("solve", "layout", "production", "failure", "model"):
        rows = [k for k in knobs if k.group == group and
                (not gaps_only or k.status != "implemented")]
        if not rows:
            continue
        print(f"\n{group.upper()}")
        for k in rows:
            print(f"  [{marks[k.status]}] {k.name}")
            print(f"          ours={k.ours!s:<30} production={k.production!s}")
            print(f"          {k.where} -- {k.why}")
            if k.cost:
                print(f"          -> {k.cost}")
    counts = {s: sum(1 for k in knobs if k.status == s) for s in marks}
    print("\n" + "  ".join(f"{s}:{n}" for s, n in counts.items()))
    print("close-gap order: " + ", ".join(PRIORITY))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gaps", action="store_true", help="only what differs from production")
    report(ap.parse_args().gaps)
