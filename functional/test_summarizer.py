"""Tests for the boundary (`llm/summarizer.py`) and the system (`minisystem.py`) -- the functional
port of ../test_summarizer.py. No third-party dependencies, no network:

    python3 test_summarizer.py

The coverage is the original's, test for test and check for check; what changed is the API shape:

    S.Summarizer(S.MockModel(plan))        ->  S.summarizer(S.mock_model(plan))
    S.Summarizer(S.HttpModel(...))         ->  S.summarizer(S.http_model(...))
    engine(kind, target, source)           ->  (text, reason), engine = S.complete(engine, ...)
    engine.request(...)                    ->  request, engine = S.request(engine, ...)
    engine.estimate(request)               ->  S.estimate(engine, request)
    engine.usage()                         ->  S.usage(engine)
    engine.last_request / .last_reply      ->  fields of the returned Summarizer value
    M.emit(log, ...)                       ->  log = M.emit(log, ...)
    M.step(log, item, engine, v)           ->  log, engine, outcome = M.step(log, item, engine, v)
    M.clear_debt(log, ...)                 ->  log = M.clear_debt(log, ...)
    archive["chunks"] / ["summaries"] ...  ->  archive.chunks / archive.summaries ...

The one fixture that needed rethinking rather than renaming: the original's `RefusingOnce` class
recorded requests ON the provider object; here the recording rides the returned model's `state`
(`refusing_once` below), as do the mock's surviving queues and the HTTP client's resolved
HttpConfig -- so every assertion that peeked at provider internals now reads a value. The check
harness works the same way: `recorder()` hands each test a failures list it returns, so there is
no module-level mutable state anywhere in this file.

What is covered, and why each one earns a test:

    request scope      the L1's six sections (:5266-5294) -- head FIRST (the reverse order caused a
                       runaway false-memory incident), the prior pairs as the UNMERGED FRONTIER only
                       (children plus parent doubles unboundedly), then marker, chunk and directive,
                       with no tail -- asserted on the assembled request AND at the system level,
                       from a request built during a real minisystem session whose archive holds
                       other chunks; plus the source-only rung that replaces all of it after a
                       refusal (types/strategy.ts:966-980)
    the L1 prompt      production's real pair: COMPRESSION_MARKER (:99-102) plus
                       formatInstruction(targetTokens) (:192-206) -- or, for a document shard,
                       formatReadingChunkInstruction (:250-266) -- and NOT the two knobs
                       production declares and never sends (types/strategy.ts:1485-1486)
    merge request      production's three parts (:6867-6891): the PREFIX (head raw, prior recall
                       pairs under compressionRecallBudgetTokens, raw middle -- nothing after the
                       merge range), the TARGET expanded ONE LEVEL DEEPER than the sources (raw L0
                       for an L2 merge, the L1s under each L2 for an L3 merge), and the INSTRUCTION
                       last, describing the layer actually shown -- plus the reading-mode variant
                       when every leaf is a shard of one body group
    the gate           empty / truncated / tool_call / refusal / transport, and the one accepted
                       shape: a complete stop with usable text and no tool call
    classification     every failure class through the mock, and the HTTP outcomes (429, 408, 5xx,
                       4xx, malformed body, tool call, truncating stop, refusal) through the real
                       client with an injected transport
    usage              accumulation across calls, and the retryable/non-retryable fault split
    truncation         maxMessageTokens truncates the EMITTED content and the emitted bytes are
                       what get priced (truncateContent, :11096): priced == rendered
    demand path        an escalated plan produces L1 requests for the uncovered foldable runs, and
                       those chunks bypass the l1HoldbackChunks window
    coverage           the fatal invariant: a chunk a recall does not cover fails loudly
    terminal debt      the clear path -- explicit, and the automatic clear a later mint performs
    stale discards     a reply against a moved log is dropped and burns no attempt
    config integrity   a status that names the production default must agree with the value
                       recorded beside it, and `report()` refuses a table that disagrees
    the demo story     minisystem's printed story still contains its anchors

The real transport is disabled at import: `http_model` is exercised only through injected fake
transports returning canned bodies, so no test can reach a provider. The fakes are pure functions;
what a test would have recorded -- the dispatched request, the endpoint config, the dispatch count
-- it instead reads off the Summarizer value (`last_request`, `model.state`, `usage()`).
"""
import io
import json
import re
import sys
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from dataclasses import replace

import config as C
import connectome_min as cm
import demo as D
import minisystem as M
from llm import summarizer as S


def _no_network(*_args, **_kwargs):
    raise AssertionError("a test tried to reach the network")


urllib.request.urlopen = _no_network

def recorder():
    """One test's verdict ledger, as a value: `check` prints the verdict and appends a failure to
    a list that is born here and returned by the test -- the folder's local-scratch exception, so
    no global, no `_`. A test ends `return fails or None` so pytest sees None on a green test and
    the runner collects the list."""
    fails = []

    def check(name, got, want):
        """The one assertion: prints the verdict and records the failure."""
        if got == want:
            print(f"  ok    {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")

    def check_true(name, condition, note=""):
        check(name + (f" ({note})" if note and not condition else ""), bool(condition), True)

    return check, check_true, fails


def template(name):
    """What is actually SENT from a prompt file: `prompt_text` drops the `#` documentation block,
    which is for us and not for the model."""
    return S.prompt_text(name)


def transport_answering(*replies):
    """A fake transport: answers with the canned `(status, body)` reply, so no test needs a
    socket. What the client SENT is asserted on the Summarizer's own state instead -- the request
    rides `last_request`, the endpoint config rides `model.state` -- so the transport itself is a
    pure function and nothing here records."""
    def transport(url, headers, body, timeout):
        return replies[-1]

    return transport


# ================== request assembly ==================
# The two strings production actually sends for an L1 mint, pinned here as literals so a change to
# either prompt file is a test failure and not a silent drift: COMPRESSION_MARKER (:99-102) and the
# opening of formatInstruction (:192-206).
MARKER = ("System: You will soon form a new memory, get ready. The messages that follow are the "
          "slice of recent experience you are about to compress. After them, write the memory in "
          "your own voice.")
DIRECTIVE_OPEN = ("Write the memory of events since the most recent memory system notification. "
                  "Speak in the first person from your own perspective.")


def test_l1_request():
    check, check_true, fails = recorder()
    request = S.build_request("L1", "L1:c7", "SOURCE-TEXT", model="m")
    user = request["user"]
    marker, _, rest = user.partition("\n\n")
    source, _, directive = rest.partition("\n\n")
    check("L1 turn opens with production's COMPRESSION_MARKER", marker, MARKER)
    check("...then the chunk raw", source, "SOURCE-TEXT")
    check_true("...then production's directive, not a dead knob's text",
                directive.startswith(DIRECTIVE_OPEN), directive[:60])
    check("L1 directive fills {target_tokens}, as formatInstruction's argument does",
           directive, S.prompt_text("l1_chunk.txt", target_tokens=S.TARGET_TOKENS))
    check("L1 system prompt is our prompts/system.txt", request["system"], template("system.txt"))
    check("L1 kind/target/model", (request["kind"], request["target"], request["model"]),
           ("L1", "L1:c7", "m"))
    check("L1 declares no tools", request["tools"], None)
    check("L1 scope names the shape production compresses with",
           request["scope"], "six-section")
    check("L1 request is a plain dict with known keys", sorted(request),
           ["kind", "max_tokens", "model", "prefix", "scope", "source", "system", "target", "tools",
            "user"])
    check("an L1 with nothing before it carries no prefix", request["prefix"], "")
    check("the output ceiling rides the request (production reads it there too)",
           request["max_tokens"], S.MAX_OUTPUT_TOKENS)
    return fails or None


def test_l1_request_is_the_six_sections():
    """production's builder, section for section (:5266-5294): head (1), prior recall pairs (2), raw
    middle (3), marker (4), chunk (5), instruction (6). The ORDER is load-bearing rather than
    cosmetic, which is why the last check here is the one that walks it:

      * head before the pairs -- the reverse order was a production incident, not a preference:
        following them, the head read as the most recent live conversation, thin chunks narrated it
        as fresh events, and the error compounded across merges into runaway false memories
        (:5268-5278, the "68 initiations" incident).
      * the pairs before the marker, and the instruction LAST, so the directive governs the turn it
        follows.
      * NO tail: nothing after the chunk (:5292-5294). Adding one would leak future information into
        the model's KV state and corrupt the as-of framing of memory formation, so its ABSENCE is
        asserted rather than left to a reader to infer from the join.

    The third assertion is the one that catches a later edit reversing 1 and 2, which is the edit
    that caused the incident."""
    check, check_true, fails = recorder()
    prefix = "\n\n".join(["HEAD-TEXT", "[CM] Recall memory L1-a.\nMEMORY-A",
                          "[CM] Recall memory L1-b.\nMEMORY-B", "MIDDLE-TEXT"])
    request = S.build_request("L1", "L1:c7", "CHUNK", prefix=prefix, model="m",
                              target_tokens=1234)
    # A recall pair is one paragraph (its question and its body joined by a single newline), so the
    # blank-line split is: 0 head | 1 pair a | 2 pair b | 3 middle | 4 marker | 5 chunk | 6 directive.
    sections = request["user"].split("\n\n")
    check("the head material precedes the prior recall pairs", sections[0], "HEAD-TEXT")
    check("...both pairs follow it, in the order the prefix carries them",
           (sections[1], sections[2]), ("[CM] Recall memory L1-a.\nMEMORY-A",
                                        "[CM] Recall memory L1-b.\nMEMORY-B"))
    check("...and each pair is intact where it landed, question then body",
           [s.count("[CM] Recall memory") for s in sections[1:3]], [1, 1])
    check("...the raw middle comes after them, section 3", sections[3], "MIDDLE-TEXT")
    check("...then the marker, the chunk raw, then the directive, in that order",
           (sections[4], sections[5], sections[6].startswith("Write the memory of events")),
           (MARKER, "CHUNK", True))
    check("exactly the prefix, the marker, the chunk and the directive were joined, so nothing "
           "else rode along", len(sections), 7)
    check("the instruction is last, nothing after it",
           request["user"].rpartition("\n\n")[2].startswith(DIRECTIVE_OPEN), True)
    check("no tail: the request ends at the directive, which is section 6",
           request["user"].endswith(S.prompt_text("l1_chunk.txt", target_tokens=1234)), True)
    check("the prefix rides the request whole, unadorned", request["prefix"], prefix)
    check("the chunk rides it whole, unadorned -- it is still `source`", request["source"], "CHUNK")
    check("...and no section is serialized twice: the bound counts the dispatched payload once",
           S.request_input_bound(request) - S.BOUND_RESERVE - 2 * S.BOUND_PER_MESSAGE,
           len(S.wire_body(request)))
    return fails or None


def test_l1_doc_aware_instruction():
    """Section 6 is doc-aware (:5497-5506). `detectDocContext` (:9716-9757) fires when the chunk is
    one shard of a substantially larger message -- every message sharing one bodyGroupId (:9728-9735)
    and the whole group at least twice this chunk (:9746-9752) -- and then
    `formatReadingChunkInstruction` (:250-266) replaces `formatInstruction`, asking what reading was
    like instead of directing the model to narrate events it did not experience. Nothing else about
    the request changes; the marker, the chunk and the pairs stay exactly where they are."""
    check, check_true, fails = recorder()
    log = ()
    for i in range(4):
        log = M.emit(log, "msg", i + 1, [("text", "PROSE word " + "w " * 200)])
    log = M.emit(log, "msg", 5, [("text", "doc " * 700)])      # one long message, several shards
    a = M.archive(log)
    groups = M.groups_of(log)
    shards = [cid for cid, group in groups.items() if group]
    chunk = next(c for c in a.chunks if c.id == shards[0])
    whole = sum(c.tokens for c in a.chunks if groups.get(c.id) == groups[chunk.id])
    check("the long message shards into chunks sharing one body group", len(shards) > 1, True)
    check("...each shard is under half the whole, so it is a portion of it",
           chunk.tokens * 2 <= whole, True)
    shown = M.source(log, "L1", chunk.id)
    check("the projection hands the request the whole group's size, not the shard's",
           shown.reading_tokens, whole)
    request = S.build_request("L1", f"L1:{chunk.id}", shown.target, prefix=shown.prefix,
                              reading_tokens=shown.reading_tokens, model="m",
                              target_tokens=1234)
    body = request["user"]
    check_true("the reading instruction replaces the consolidation one",
                "Reflect on this reading: what was it like?" in body)
    check("...which is not sent at all", DIRECTIVE_OPEN in body, False)
    check_true("...and it is still last, still section 6",
                body.rpartition("\n\n")[2].startswith("Speak in your own voice as the one reading."))
    check_true("total_tokens is the whole group's size",
                f"approximately {whole} tokens total in this piece" in body)
    check_true("the marker and the chunk are untouched by the variant",
                (MARKER in body, shown.target in body), (True, True))
    ordinary = M.source(log, "L1", "c0")                     # a whole message, no body group
    check("a chunk that is not a shard gets the ordinary directive",
           ordinary.reading_tokens, None)
    check("...and a chunk with no prefix at all still gets it",
           S.build_request("L1", "L1:c0", ordinary.target, model="m")["user"].rpartition("\n\n")[2]
           .startswith(DIRECTIVE_OPEN), True)
    return fails or None


def test_source_only_is_the_refusal_rung():
    """`compressionSourceOnly` (types/strategy.ts:966-980) is an OPTION in production, and the one
    our ladder keeps -- as a rung with a reason to exist rather than a permanent shape. The reason is
    production's own incident: L1 compression was refused by the safety classifier when the request
    compiled the raw recent-window traffic alongside the target chunk, the failing contribution was
    localized to that block as a class, and "handing the summarizer only the thing it is summarizing
    is the correct scope; the copied drain cleared all quarantined chunks first-try and the summaries
    passed a full fidelity audit" (:966-969). So the rung sends sections 4-6 of the builder -- marker,
    chunk, directive -- which is what `sourceOnly` skips structurally 1-3 to produce (:5411-5466).

    It is DERIVED, not queued: a refusal recorded for the unit moves the next attempt onto the rung,
    exactly as an `overflow` moves it to the reshape. That is why `step()` still returns only
    `(text | None, reason | None)` alongside the threaded state, and why the log stays the only store."""
    check, check_true, fails = recorder()
    log = ()
    for turn in range(4):
        for i in range(4):
            log = M.emit(log, "msg", turn * 4 + i + 1, [("text", f"Z{turn} word " + "w " * 200)])
    log = mint(log, "L1:c0", "L1-c0", 1, ("c0",), "MEMORY-OF-c0")
    engine = S.summarizer(S.mock_model())
    target = "L1:c1"
    check("the unit has no failure history yet, so nothing reshapes it",
           M.last_reason(log, target), None)
    log, engine, outcome = M.step(log, (target, "L1", "c1"), engine, len(log))
    check("the first attempt mints through the boundary", outcome, "done")
    canonical = engine.last_request
    check("the canonical shape is what goes first", canonical["scope"], "six-section")
    check_true("...and it carries what precedes the chunk, as the recall pair that owns it",
                canonical["prefix"] == "[CM] Recall memory L1-c0.\nMEMORY-OF-c0")
    check_true("...including the recall pair for the minted chunk",
                "[CM] Recall memory L1-c0.\nMEMORY-OF-c0" in canonical["user"])

    log = ()
    for turn in range(4):
        for i in range(4):
            log = M.emit(log, "msg", turn * 4 + i + 1, [("text", f"Z{turn} word " + "w " * 200)])
    log = mint(log, "L1:c0", "L1-c0", 1, ("c0",), "MEMORY-OF-c0")
    engine = S.summarizer(refusing_once())
    log, engine, outcome = M.step(log, (target, "L1", "c1"), engine, len(log))
    check("a refusal is recorded and routed to retry", outcome, "retry")
    check("...and the unit now reads as refused, which is what the rung keys on",
           M.last_reason(log, target), "refusal")
    check("only the canonical shape has been sent so far", len(engine.model.state), 1)
    log, engine, outcome = M.step(log, (target, "L1", "c1"), engine, len(log))
    check("...and the next attempt is the retry, which succeeds here", outcome, "done")
    first, second = engine.model.state
    check("the refused attempt was the canonical shape", (first["scope"], len(first["prefix"]) > 0),
           ("six-section", True))
    check("the retry is the source-only rung", second["scope"], "source-only")
    check("...which sends the marker, the chunk and the directive, and nothing else",
           second["user"],
           "\n\n".join([MARKER, second["source"],
                        S.prompt_text("l1_chunk.txt", target_tokens=S.TARGET_TOKENS)]))
    check("...as production's own source-only arm does, sections 4-6 of the builder",
           second["prefix"], "")
    check("...and it is not description-dependent: no CM recall header leaks in",
           "[CM] Recall memory" in second["user"], False)
    check("...while the canonical attempt it degraded FROM did carry a pair",
           "[CM] Recall memory" in first["user"], True)
    check("the rung is smaller than the shape it degrades from, which is its point",
           S.request_input_bound(second) < S.request_input_bound(first), True)
    log, engine, outcome = M.step(log, (target, "L1", "c1"), engine, len(log))
    check("a retry that succeeds mints, on the same work item", outcome, "done")
    check("...and the unit is a memory, so no rung is pending for it",
           target in M.mints(log), True)
    return fails or None


# A recording provider as a VALUE: the original's `RefusingOnce` class appended each request to a
# list on the instance; here the requests tuple rides the returned model's `state`, so the test
# reads them off the Summarizer's `model` field after the threaded calls.
def refusing_once():
    """The provider boundary in the one state the rung exists for: the canonical shape refused, then
    the source-only shape accepted. It dispatches nothing and records exactly what it was handed --
    in the state of the model it returns."""
    def make(requests):
        def complete(request):
            reply = (S.Reply(text="I can't write a memory of this material.", stop="refusal",
                             input_tokens=1, output_tokens=1) if not requests else
                     S.Reply(text="A MEMORY", stop="end_turn", input_tokens=1, output_tokens=1))
            return reply, make(requests + (request,))
        return S.Model("RefusingOnce", "refusing-1", complete, requests)
    return make(())


def test_prompt_provenance():
    """The prompts are production's, and the ones it never sends are gone. `summarySystemPrompt`
    and `summaryUserPrompt` (types/strategy.ts:1485-1486) have zero read sites in production: the
    system turn is the host's live identity prompt (:5604) and the L1 turn is the marker above plus
    formatInstruction. Sending a copy of a declared-but-unread knob was the audit's D8a."""
    check, check_true, fails = recorder()
    sent = S.build_request("L1", "L1:c7", "S", model="m")
    body = sent["system"] + "\n" + sent["user"]
    for dead in ("What do you recall from this part of the conversation?",
                 "Capture what matters:", "Write naturally, as recollection of what you experienced."):
        check(f"the never-sent summaryUserPrompt is not in the request ({dead[:24]!r})",
               dead in body, False)
    check_true("the directive names what to preserve, as production's does",
                "file paths, exact values, decisions, unresolved questions" in body)
    check_true("...targets the configured size", f"Target ~{S.TARGET_TOKENS} tokens" in body)
    check_true("...and carries the anti-padding rule", "do not pad it by re-narrating" in body)
    check_true("no prompt file's documentation block is sent",
                all("# " not in part for part in (sent["system"], sent["user"])))
    return fails or None


def test_merge_request():
    """`merge.txt`'s three placeholders, filled: the level being minted from the work id, the token
    target, and a `{seen_description}` derived from the layer the merge is SHOWN -- which is one
    level BELOW its sources (production's `sourceLevelShown`, :7151-7154), not the sources
    themselves. The sentence is production's own `formatMergeInstruction` (:277-295)."""
    check, check_true, fails = recorder()
    request = S.build_request("merge", "L3:L2-a+L2-b", "CHILD-ONE\nCHILD-TWO", model="m",
                              target_tokens=1234)
    sources, _, instruction = request["user"].partition("\n\n")
    check("merge sources sit above the instruction", sources, "CHILD-ONE\nCHILD-TWO")
    check("merge instruction is production's sentence, filled",
           instruction,
           "You have just reviewed the L1 memories above, in chronological order. They cover the "
           "stretch of experience you are about to consolidate into a single L3 memory. Write a "
           "memory that preserves the through-line: what happened, what was decided, what remains "
           "open, what concrete details future you will want to reach for. Speak in the first "
           "person. Target ~1234 tokens. Output only the memory body — no preamble, no "
           "meta-commentary about summarizing.")
    check_true("merge leaves no unfilled placeholder", "{" not in instruction and "}" not in instruction)
    check_true("merge.txt's documentation block is not sent", "#" not in request["user"])
    check("merge system prompt is prompts/system.txt", request["system"], template("system.txt"))
    check("a merge carries no marker: production's merge request has none",
           MARKER in request["user"], False)
    check("merge target level comes from the work id",
           S.build_request("merge", "L5:L4-a+L4-b", "S", model="m")["user"].split("single ")[1][:2],
           "L5")
    raw = S.build_request("merge", "L2:L1-a+L1-b", "CHUNK-ONE\nCHUNK-TWO", model="m")
    check_true("an L2 merge is described as the raw conversation it is shown",
                "You have just reviewed the slices of recent experience above (raw conversation), "
                "in chronological order." in raw["user"])
    check("a merge is not source-only any more, and says what it is",
           raw["scope"], "prefix+one-level-deeper")
    return fails or None


def test_request_carries_source_only():
    """The unit-level half of the invariant, for the rung that still has that scope: the source-only
    L1 user turn is sections 4-6 of production's builder -- the marker, the source, the directive --
    and nothing from anywhere else. A merge's turn is the three parts, in production's order, and
    both `source` and `prefix` ride the request as their own fields without being serialized twice."""
    check, check_true, fails = recorder()
    request = S.build_request("L1", "L1:c7", "THE ONLY SOURCE", source_only=True, model="m")
    check("the source-only rung's user turn is exactly marker + source + directive",
           request["user"],
           "\n\n".join([MARKER, "THE ONLY SOURCE",
                        S.prompt_text("l1_chunk.txt", target_tokens=S.TARGET_TOKENS)]))
    check("the source rides the request as its own field, unadorned", request["source"],
           "THE ONLY SOURCE")
    merge = S.build_request("merge", "L2:L1-a+L1-b", "THE ONLY SOURCE", model="m")
    check("a merge with no prefix is still source + directive",
           merge["user"].partition("\n\n")[0], "THE ONLY SOURCE")
    prefixed = S.build_request("merge", "L2:L1-a+L1-b", "THE SOURCE", prefix="THE PREFIX",
                               model="m")
    parts = prefixed["user"].split("\n\n")
    check("with a prefix the turn is prefix, target, instruction -- production's order",
           (parts[0], parts[1], parts[-1].startswith("You have just reviewed")),
           ("THE PREFIX", "THE SOURCE", True))
    check("both fields ride the request for inspectability",
           (prefixed["prefix"], prefixed["source"]), ("THE PREFIX", "THE SOURCE"))
    check("...and neither is serialized on its own: the wire body has one system and one user turn",
           sorted(json.loads(S.wire_body(prefixed))), ["max_tokens", "messages", "model", "stream"])
    check("...so the bound counts each of them once", S.request_input_bound(prefixed),
           len(S.wire_body(prefixed)) + S.BOUND_RESERVE + 2 * S.BOUND_PER_MESSAGE)
    l1 = S.build_request("L1", "L1:c7", "THE SOURCE", prefix="THE PREFIX", model="m")
    check("an L1 with a prefix keeps the same turn shape, in the same order",
           l1["user"].partition("\n\n")[0], "THE PREFIX")
    check("...so a prefix is counted once there too", S.request_input_bound(l1),
           len(S.wire_body(l1)) + S.BOUND_RESERVE + 2 * S.BOUND_PER_MESSAGE)
    check("...and the prefix is not one of its own keys, only content inside the user turn",
           sorted(json.loads(S.wire_body(l1))), ["max_tokens", "messages", "model", "stream"])
    check("`scope` is not a dispatched field either, so it cannot be counted at all",
           "scope" in json.dumps(json.loads(S.wire_body(l1))), False)
    return fails or None


def test_request_scope_in_a_session():
    """The invariant at the system level, restated for the six-section shape: during a real session,
    the request built for one chunk carries the chunk plus exactly the material that PRECEDES it and
    is unrepresented -- never a chunk that a live summary already stands for, and never anything
    after the chunk. That second half is the as-of rule (:5292-5294), and it is the one that would
    regress silently, because including later material makes a request look richer, not broken.

    Six turns of four 103-token messages each, one marker per turn: the chunker closes one chunk
    per turn, so each chunk's bytes are identifiable."""
    check, check_true, fails = recorder()
    log = ()
    for turn in range(6):
        for i in range(4):
            log = M.emit(log, "msg", turn * 4 + i + 1, [("text", f"ZEBRA{turn} word " + "w " * 200)])
    chunks = {c.id: c.text for c in M.archive(log).chunks}
    check("six turns close six chunks", sorted(chunks), ["c0", "c1", "c2", "c3", "c4", "c5"])
    check_true("chunk c1 is marked ZEBRA1", "ZEBRA1" in chunks["c1"])
    check_true("chunk c2 is marked ZEBRA2", "ZEBRA2" in chunks["c2"])

    engine = S.summarizer(S.mock_model())
    log, engine, outcome = M.step(log, ("L1:c1", "L1", "c1"), engine, len(log))
    check("step() mints through the boundary", outcome, "done")
    request = engine.last_request
    body = request["system"] + "\n" + request["user"]
    check("the request carries the target source verbatim", request["source"], chunks["c1"])
    check_true("the target chunk is in the request", "ZEBRA1" in body)
    check("nothing after the chunk is visible, from any chunk in the session",
           [t for t in (2, 3, 4, 5) if f"ZEBRA{t}" in body], [])
    check("the head chunk is there raw and whole: nothing stands for it, so nothing replaces it",
           request["prefix"], chunks["c0"])
    check("...and the chunk being compressed does not ride the prefix as well",
           "ZEBRA1" in request["prefix"], False)

    # Now the same chunk again, with c0 represented by a memory: ownership wins over the head
    # boundary, so the covered stretch arrives as its recall pair and not as raw text (:6981-6990).
    log = ()
    for turn in range(6):
        for i in range(4):
            log = M.emit(log, "msg", turn * 4 + i + 1, [("text", f"ZEBRA{turn} word " + "w " * 200)])
    log = mint(log, "L1:c0", "L1-c0", 1, ("c0",), "MEMORY-OF-ZEBRA0")
    engine = S.summarizer(S.mock_model())
    log, engine, _outcome = M.step(log, ("L1:c1", "L1", "c1"), engine, len(log))
    body = engine.last_request["system"] + "\n" + engine.last_request["user"]
    check("a covered stretch arrives as its recall pair, not raw",
           ("ZEBRA0 word" in body, "[CM] Recall memory L1-c0.\nMEMORY-OF-ZEBRA0" in body),
           (False, True))
    return fails or None


def test_l1_prior_pairs_are_the_unmerged_frontier():
    """Section 2 draws from the UNMERGED FRONTIER only -- "any summary that has not yet been merged
    into a higher level", because "after merges run, the L_{k+1} replaces its L_k children -- using
    the children plus their parent doubles the prompt size unboundedly" (:5279-5284). Production got
    there by measurement, not by theory: the original "all L1s regardless of merge state" rule was a
    fidelity optimization that "scales catastrophically: a 4000-message import converged to ~500 L1s
    that never aged out, blowing the 200k window around chunk 118" (:5305-5311).

    So this test asserts both directions. A summary already merged into a parent must NOT appear,
    and the parent that replaced it MUST -- the first is the doubling, the second is the fidelity."""
    check, check_true, fails = recorder()
    log = memories(session(12), skip=("c0",))            # c1..c11 minted as L1s
    log = mint(log, "L2:A", "L2-A", 2, tuple(f"L1-c{i}" for i in range(1, 7)), "MEMORY-L2-A")
    a = M.archive(log)
    frontier = [s.id for s in a.summaries.values() if s.id not in a.parent]
    check("the six merged L1s are no longer on the frontier",
           [f"L1-c{i}" for i in range(1, 7) if f"L1-c{i}" in frontier], [])
    check("...and the L2 that replaced them is", "L2-A" in frontier, True)
    check("the frontier is exactly the unmerged remainder",
           sorted(frontier), ["L1-c10", "L1-c11", "L1-c7", "L1-c8", "L1-c9", "L2-A"])

    engine = S.summarizer(S.mock_model())
    log, engine, outcome, request = ask(log, "L1:c11", engine)
    prefix = request["prefix"]
    check("the L1 request projects the frontier, so it mints", outcome, "done")
    check("the head chunk is section 1, and it is raw: nothing covers it",
           prefix.split("\n\n")[0].startswith("T0 word"), True)
    check("the parent that replaced its six children rides the prefix",
           "MEMORY-L2-A" in prefix, True)
    check("the six children do NOT: children plus parent is the unbounded doubling",
           sorted(memories_in(prefix) & {f"c{i}" for i in range(1, 7)}), [])
    check("...nor does the parent itself appear more than once",
           prefix.count("[CM] Recall memory L2-A."), 1)
    check("everything else on the frontier is present as its own pair",
           sorted(memories_in(prefix)), ["c10", "c7", "c8", "c9"])
    check("the chunk being compressed is the target, never part of the prefix",
           memories_in(prefix) & {"c11"}, set())
    check("the chunk's own id appears in no pair, so it cannot be shown twice",
           "[CM] Recall memory L1-c11." in request["user"], False)

    # And the frontier rule is what a merge's prefix uses too: one projection, both requests.
    log = memories(session(12), skip=("c0",))
    log = mint(log, "L2:A", "L2-A", 2, tuple(f"L1-c{i}" for i in range(1, 7)), "MEMORY-L2-A")
    log = mint(log, "L2:B", "L2-B", 2, tuple(f"L1-c{i}" for i in range(7, 12)), "MEMORY-L2-B")
    engine = S.summarizer(S.mock_model())
    _log, _engine, _outcome, merge = ask(log, "L3:L2-A+L2-B", engine)
    check("a merge projects the same frontier, so the merged children are excluded there too",
           sorted(memories_in(merge["prefix"])), [])
    check("...while the parent that replaced each of them is not doubled into it either",
           [m for m in ("MEMORY-L2-A", "MEMORY-L2-B") if m in merge["prefix"]],
           [])                                        # both are the merge's SOURCES, not priors
    return fails or None


def test_l1_ladder_caps_newest_first_and_resorts():
    """The L1 prefix is capped by the same `capRecallPairs` a merge's is (:2489-2514, budget read at
    :5430-5439): walked NEWEST-first, each pair that still fits kept with `continue` rather than
    `break`, and the kept set put back into chronological order. That last step is what makes the
    section-2 ordering rule survive the cap -- and it is the step a hand-rolled cap gets wrong.

    The numbers are read from `archive`, so the check cannot drift from the pricing `recallPairCost`
    defines (:9006-9061): a pair costs the question label plus the content, and the cap adds the
    per-pair overhead."""
    check, check_true, fails = recorder()
    log = memories(session(12), skip=("c0", "c6"))       # frontier: c1..c5, c7..c11
    pair_tokens = M.archive(log).summaries["L1-c1"].tokens
    keep = M.RECALL_BUDGET
    M.RECALL_BUDGET = 3 * (pair_tokens + M.RECALL_PAIR_OVERHEAD)
    try:
        engine = S.summarizer(S.mock_model())
        _log, _engine, outcome, capped = ask(log, "L1:c11", engine)
    finally:
        M.RECALL_BUDGET = keep
    prefix = capped["prefix"]
    check("the capped L1 still mints", outcome, "done")
    check("the cap kept the three NEWEST pairs and dropped the older five",
           sorted(memories_in(prefix)), ["c10", "c8", "c9"])
    check("...the boundary pair is exactly the one the budget admits",
           (memories_in(prefix) & {"c7"}, memories_in(prefix) & {"c8"}), (set(), {"c8"}))
    check("...and the kept set is back in chronological order, so section 2 stays chronological",
           [prefix.index(f"MEMORY-OF-c{i}") for i in (8, 9, 10)],
           sorted(prefix.index(f"MEMORY-OF-c{i}") for i in (8, 9, 10)))
    check("the head is still section 1 after the cap, untouched by it",
           prefix.split("\n\n")[0].startswith("T0 word") or "T0 word" in prefix, True)
    check("the test left the configured budget where it found it", M.RECALL_BUDGET, keep)

    M.RECALL_BUDGET = 0                                  # production's source-only arm zeroes it
    try:
        engine = S.summarizer(S.mock_model())
        _log, _engine, _outcome, empty = ask(log, "L1:c11", engine)
    finally:
        M.RECALL_BUDGET = keep
    check("a zero recall budget leaves the head and the raw middle, and no pairs at all",
           ("[CM] Recall memory" in empty["prefix"], "T0 word" in empty["prefix"]), (False, True))
    return fails or None


# ================== what a merge request is shown ==================
def mint(log, target, sid, level, children, text):
    """A mint event as the system writes it: the span each child owns AT THIS MOMENT is stamped into the
    event (minisystem.step does the same), so the coverage guard stays a fact about mint time even when
    a later append moves the chunk boundaries. Returns the next log, as `emit` does."""
    a = M.archive(log)
    here = M.chunk_spans(log)
    leaves = tuple(l for c in children for l in (a.summaries[c].leaves if c in a.summaries else (c,)))
    return M.emit(log, "mint", target, sid, level, children, text,
                  tuple(here[l] for l in leaves if l in here))


def memories(log, skip=()):
    """Hand-written L1 memories over a session's chunks. The mock's digest is a PREFIX of its own
    source, so it cannot tell "shown raw" from "shown as a summary" -- which is the one distinction
    the merge tests are about."""
    for c in M.archive(log).chunks:
        if c.id not in skip:
            log = mint(log, f"L1:{c.id}", f"L1-{c.id}", 1, (c.id,), f"MEMORY-OF-{c.id}")
    return log


def memories_in(text, prefix="MEMORY-OF-"):
    """Which memories a request carries, by exact id: a substring test would read
    `MEMORY-OF-c1` out of `MEMORY-OF-c10`, which is exactly the boundary the recall cap and
    the frontier filter are about."""
    return {m.group(1) for m in re.finditer(re.escape(prefix) + r"(c\d+)", text)}


def ask(log, target, engine):
    """One attempt through the system, and the request it built -- for a merge or an L1 alike: the
    kind comes from the work id, exactly as `work` produces it. Returns the threaded state too:
    (log, engine, outcome, request)."""
    kind = "L1" if target.startswith("L1:") else "merge"
    log, engine, outcome = M.step(log, (target, kind, target.split(":", 1)[1]), engine, len(log))
    return log, engine, outcome, engine.last_request


def merge_target(*chunks):
    return "L2:" + "+".join(f"L1-{c}" for c in chunks)


def test_merge_target_is_one_level_deeper():
    """production's TARGET expansion (:7076-7148): the sources are expanded ONE LEVEL DEEPER than
    they themselves are, so the consolidation is grounded in the material rather than in summaries
    of it -- "the model sees the actual conversation that the 6 L1s consolidate" (:6879-6883).
    An L2 merge's sources are L1s, whose only leaf is their chunk, so the raw L0 chunk text is what
    is shown; an L3 merge's sources are L2s, so the L1s under each L2 are shown, as `[CM] Recall
    memory <id>.` pairs (:7112-7148). Shown the children's own recall text instead, our merges were
    consolidating summaries of summaries -- config.py's `merge request scope` row, now closed."""
    check, check_true, fails = recorder()
    log = memories(session(12))
    engine = S.summarizer(S.mock_model())
    log, engine, outcome, request = ask(log, merge_target(*(f"c{i}" for i in range(6))), engine)
    raw = M.archive(log).chunks
    check("the L2 merge mints", outcome, "done")
    check("the target is the raw L0 the L1s consolidate, in order",
           request["source"], "\n\n".join(c.text for c in raw[:6]))
    check_true("...so the raw conversation is in the request", "T0 word" in request["user"])
    check("...and NOT the six summaries it is consolidating",
           "MEMORY-OF-c0" in request["user"], False)

    log = memories(session(12))
    for name, span in (("L2-A", range(6)), ("L2-B", range(6, 12))):
        log = mint(log, f"L2:{name}", name, 2, tuple(f"L1-c{i}" for i in span), f"MEMORY-{name}")
    engine = S.summarizer(S.mock_model())
    _log, _engine, outcome, request = ask(log, "L3:L2-A+L2-B", engine)
    check("the L3 merge mints", outcome, "done")
    pairs = [f"[CM] Recall memory L1-c{i}.\nMEMORY-OF-c{i}" for i in range(12)]
    check("the layer under each source is shown, one recall pair per L1",
           [p for p in pairs if p not in request["user"]], [])
    check("...and not the two L2 memories themselves", "MEMORY-L2-A" in request["user"], False)
    check("...and every pair is present exactly once",
           request["user"].count("[CM] Recall memory L1-"), 12)
    return fails or None


def test_merge_prefix_is_the_prior_content_only():
    """production's PREFIX (:6921-7074): the head window raw, then the prior recall pairs, then the
    raw middle -- everything that comes chronologically BEFORE the merge range, and nothing after it
    ("No tail-after-merge: same as-of principle as L1 compression", :6889-6891). Every element is
    skipped when a live summary already covers that stretch (:6948-6965): ownership wins over the
    head boundary, so covered text cannot leak back in raw. The prior set is the UNMERGED FRONTIER
    whose span starts before the merge range (:6921-6946) -- the section comment still says "L1
    recall pairs", but the code walks the frontier and says why."""
    check, check_true, fails = recorder()
    log = memories(session(14), skip=("c0", "c5"))       # c0 = the head chunk, c5 = the raw middle
    log = mint(log, "L2:A", "L2-A", 2, ("L1-c1", "L1-c2", "L1-c3", "L1-c4"), "THE-FIRST-FOUR")
    engine = S.summarizer(S.mock_model())
    _log, _engine, outcome, request = ask(log, merge_target(*(f"c{i}" for i in range(6, 12))), engine)
    prefix, body = request["prefix"], request["user"]
    check("the merge mints", outcome, "done")
    check("the prefix is exactly head + prior pairs + raw middle, in that order",
           len(prefix.split("\n\n")), 3)
    check_true("the head window is raw: no summary covers that chunk", prefix.startswith("T0 word"))
    check_true("the raw middle is raw too: an uncovered chunk before the merge range",
                "T5 word" in prefix)
    check_true("the prior frontier rides it as a recall pair",
                "[CM] Recall memory L2-A.\nTHE-FIRST-FOUR" in prefix)
    check("a stretch a live summary owns does not leak back in raw",
           [t for t in (1, 2, 3, 4) if f"T{t} word" in prefix], [])
    check("the merge's own leaves are the target, not the prefix",
           [f"T{t} word" in prefix for t in range(6, 12)], [False] * 6)
    check("nothing after the merge range is visible, in either part",
           [t for t in (12, 13) if f"T{t} word" in body], [])
    check_true("the target starts where the merge range starts",
                request["source"].startswith("T6 word"))
    check_true("the instruction is still last",
                body.rpartition("\n\n")[2].startswith("You have just reviewed"))
    return fails or None


def test_merge_instruction_describes_what_is_shown():
    """`{seen_description}` must name the layer the model is ACTUALLY looking at, or the instruction
    is describing a request that was not sent. Production derives it one level below the sources
    (`sources[0].sourceLevel === 0 ? 0 : sources[0].level - 1`, :7151-7154; the two wordings are
    :283-285), so an L2 merge sees raw L0 and is told so, and an L3 merge sees L1s and is told that.
    The raw branch was unreachable here while a merge was shown its children -- a test for it was
    deleted as dead code, which was right for that implementation and wrong for this one -- so both
    branches are pinned here."""
    check, check_true, fails = recorder()
    log = memories(session(12))
    engine = S.summarizer(S.mock_model())
    _log, _engine, _outcome, l2 = ask(log, merge_target(*(f"c{i}" for i in range(6))), engine)
    check_true("an L2 merge is shown raw L0, and its instruction says so",
                "You have just reviewed the slices of recent experience above (raw conversation), "
                "in chronological order." in l2["user"])
    check_true("...and names the level it is minting", "into a single L2 memory" in l2["user"])
    log = memories(session(12))
    log = mint(log, "L2:A", "L2-A", 2, tuple(f"L1-c{i}" for i in range(6)), "MEMORY-L2-A")
    log = mint(log, "L2:B", "L2-B", 2, tuple(f"L1-c{i}" for i in range(6, 12)), "MEMORY-L2-B")
    engine = S.summarizer(S.mock_model())
    _log, _engine, _outcome, l3 = ask(log, "L3:L2-A+L2-B", engine)
    check_true("an L3 merge is shown the L1s, and its instruction says so",
                "You have just reviewed the L1 memories above, in chronological order."
                in l3["user"])
    check_true("...and names the level it is minting", "into a single L3 memory" in l3["user"])
    check("neither request tells the model it is looking at its own sources",
           [("the L2 memories above" in r["user"], "the L3 memories above" in r["user"])
            for r in (l2, l3)], [(False, False), (False, False)])
    return fails or None


def test_recall_budget_evicts_the_oldest_first():
    """`capRecallPairs` (:2489-2514): the ladder inside a request cannot itself overflow the window,
    so it is walked NEWEST-first and each pair that no longer fits is dropped -- proximate context
    survives, the oldest goes -- with `continue`, not `break`, so one oversized pair cannot hide the
    smaller siblings behind it, and the kept set re-sorted chronologically. Ours is
    `minisystem.cap_recalls` over `config.RECALL_BUDGET`: production's 100k scaled /10, the same
    share of its context budget as there."""
    check, check_true, fails = recorder()
    check("the budget is production's, scaled /10 like the other token knobs",
           (M.RECALL_BUDGET, C.CONTEXT_BUDGET), (10000, 20000))
    log = memories(session(12))
    target = merge_target(*(f"c{i}" for i in range(6, 12)))
    engine = S.summarizer(S.mock_model())
    _log, _engine, _outcome, roomy = ask(log, target, engine)
    check("at the default budget every prior pair is kept, chronologically",
           [t for t in range(6) if f"MEMORY-OF-c{t}" in roomy["prefix"]], [0, 1, 2, 3, 4, 5])
    keep = M.RECALL_BUDGET
    M.RECALL_BUDGET = 120                     # two pairs at 8 words + 50 each: 116 of 120
    try:
        engine = S.summarizer(S.mock_model())
        _log, _engine, _outcome, capped = ask(log, target, engine)
    finally:
        M.RECALL_BUDGET = keep
    check("a smaller budget keeps the newest pairs and drops the oldest",
           [t for t in range(6) if f"MEMORY-OF-c{t}" in capped["prefix"]], [4, 5])
    check_true("...and the kept set is put back into chronological order",
                capped["prefix"].startswith("[CM] Recall memory L1-c4."))
    check("the test left the configured budget where it found it", M.RECALL_BUDGET, keep)
    return fails or None


def test_merge_reading_mode_instruction():
    """Reading mode (`formatReadingMergeInstruction`, :340-365, detected at :7156-7175): when EVERY
    leaf under the merge is a shard of the same body group, the agent was reading one long document
    rather than conversing, and the merge asks what the reading was like instead of forcing an
    impersonal consolidation -- the drift into the document author's voice this variant exists to
    stop. `totalTokens` is the WHOLE group, every message carrying that bodyGroupId, not just the
    leaves this merge covers. Our chunks carry `spans`' group tag for shards, so the detection falls
    out of the target's leaf set; nothing else about it is ours."""
    check, check_true, fails = recorder()
    log = ()
    for i in range(4):
        log = M.emit(log, "msg", i + 1, [("text", "PROSE word " + "w " * 200)])
    log = M.emit(log, "msg", 5, [("text", "doc " * 2000)])     # 2000 tokens -> seven shards
    log = memories(log)
    a = M.archive(log)
    shards = [cid for cid, _f, _l, _t, _x, _s, group in M.spans(log) if group]
    whole = sum(c.tokens for c in a.chunks if c.id in shards)
    leaves = sum(c.tokens for c in a.chunks if c.id in shards[:6])
    check("the long message shards into seven chunks sharing one body group", len(shards), 7)
    check_true("...so the merge's leaves are a subset of one document", leaves < whole)
    # Each shard carries a SLICE of the document, as production's does (chunkMessage splits the
    # message by chars, :1567-1580). A merge over six shards of one 2000-token document is
    # therefore large but admissible: measured bound 9104 of the 20 000 budget, and it is the most
    # expensive request this artifact builds -- pinned by the last two checks.
    engine = S.summarizer(S.mock_model(), budget=200000)
    _log, _engine, outcome, request = ask(log, merge_target(*(f"c{i}" for i in range(2, 8))), engine)
    user = request["user"]
    check("the merge mints", outcome, "done")
    check_true("the reading instruction replaces the consolidation one",
                "Reflect across the stretch: what was it like, reading these portions together?"
                in user)
    check("...which is not sent at all", "Write a memory that preserves the through-line" in user,
           False)
    check_true("the description names what was shown: the passages just read",
                "You have just re-experienced the portions of text you read above (raw passages "
                "from a larger piece), in chronological order." in user)
    check_true("total_tokens is the whole group, not the six leaves in this merge",
                f"approximately {whole} tokens in total across all of it" in user)
    mixed = S.summarizer(S.mock_model(), budget=200000)
    _log, _engine, _outcome, other = ask(log, merge_target(*(f"c{i}" for i in range(6))), mixed)
    check_true("leaves from two groups keep the consolidation instruction",
                "Write a memory that preserves the through-line" in other["user"])
    small = S.summarizer(S.mock_model())
    _log, _engine, outcome, request = ask(log, merge_target(*(f"c{i}" for i in range(2, 8))), small)
    bound = S.request_input_bound(request)
    check("at the shipped budget that same reading merge is admitted", outcome, "done")
    check_true("...and it is the largest request this artifact builds: over 40% of the budget",
                bound > 0.4 * S.CONTEXT_BUDGET, f"bound {bound} of {S.CONTEXT_BUDGET}")
    return fails or None


# ================== the terminal-disposition gate ==================
def test_gate():
    check, check_true, fails = recorder()
    check("complete stop with text is accepted",
           S.gate(S.Reply(text="a memory", stop="end_turn")), ("a memory", None))
    check("whitespace-only body is empty",
           S.gate(S.Reply(text=" \n\t ", stop="end_turn")), (None, "empty"))
    check("empty body with no stop is empty",
           S.gate(S.Reply(text="", stop="end_turn")), (None, "empty"))
    check("truncating stop is truncated",
           S.gate(S.Reply(text="half a memory", stop="max_tokens")), (None, "truncated"))
    check("abort is truncated (production's incomplete)",
           S.gate(S.Reply(text="as far as it got", stop="abort")), (None, "truncated"))
    check("a missing stop is truncated, not a memory",
           S.gate(S.Reply(text="plausible text", stop=None)), (None, "truncated"))
    check("a tool call is rejected",
           S.gate(S.Reply(text="", stop="tool_use", tool_call=True)), (None, "tool_call"))
    check("a tool call outranks its own empty text",
           S.gate(S.Reply(text="  ", stop="tool_use", tool_call=True)), (None, "tool_call"))
    check("a refusal outranks the preamble it carried",
           S.gate(S.Reply(text="I can't write a memory of this material.", stop="refusal")),
           (None, "refusal"))
    check("a transport fault is a provider_error",
           S.gate(S.Reply(error="provider_error", retryable=False)), (None, "provider_error"))
    return fails or None


# ================== the mock ==================
def test_mock_classes():
    check, check_true, fails = recorder()
    for reason in ("refusal", "empty", "truncated", "tool_call", "provider_error", "overflow"):
        engine = S.summarizer(S.mock_model({"c1": [reason]}))
        got, engine = S.complete(engine, "L1", "L1:c1", "source words here")
        check(f"mock simulates {reason}", got, (None, reason))
        check(f"mock's {reason} is routed by the policy table", reason in M.ACTIONS, True)
    engine = S.summarizer(S.mock_model())          # empty plan: every call succeeds
    got, engine = S.complete(engine, "L1", "L1:c1", "one two three")
    check("an exhausted plan succeeds", got, ("one two three", None))
    engine = S.summarizer(S.mock_model({"c1": [S.Reply(error="provider_error", retryable=False)]}))
    got, engine = S.complete(engine, "L1", "L1:c1", "src")
    check("the mock can script a non-retryable fault", got, (None, "provider_error"))
    check("...and the session accounting keeps the split", S.usage(engine)["faults"],
           {"retryable": 0, "non_retryable": 1})
    return fails or None


def test_mock_digest():
    check, check_true, fails = recorder()
    source = " ".join(f"w{i}" for i in range(120))
    engine = S.summarizer(S.mock_model())
    (text, reason), engine = S.complete(engine, "L1", "L1:c1", source)
    check("mock success is a digest of the source", text, " ".join(f"w{i}" for i in range(40)))
    check("mock success is a complete disposition", reason, None)
    return fails or None


# ================== pre-send admission: overflow without a provider ==================
def boundary_sources(engine, kind, target):
    """Two sources that straddle the admission line exactly, built from the real templates: `at`
    lands the input BOUND -- the gate's input side -- on the last admissible byte, `over` one past
    it. The bound grows one-for-one with ASCII source bytes, so the fixture is exact -- measured on
    a ONE-byte source, because an empty one drops the paragraph separator a real source carries
    (`summarizer.build_request` joins the parts a merge request actually has)."""
    probe, _s = engine_request(engine, kind, target, "x")  # pure: builds the request, dispatches nothing
    room = engine.budget - probe["max_tokens"]             # the bound's room, given the reserve
    fixed = S.request_input_bound(probe) - 1               # the whole bound except the source's bytes
    return "x" * (room - fixed), "x" * (room - fixed + 1), room


def engine_request(engine, kind, target, source):
    """`S.request` with the Summarizer field update folded back in, as the original's method did."""
    return S.request(engine, kind, target, source)


def transport_true_cost(multiple):
    """A provider whose reported input is `multiple` x the UTF-8 bytes of the payload it was handed,
    computed per request -- so a sample paired with another request's estimate lands far out of
    band and cannot be learned. Pure: the test recomputes what it reported from
    `S.wire_body(engine.last_request)`."""
    def transport(url, headers, body, timeout):
        return (200, json.dumps({"choices": [{"message": {"content": "A MEMORY"},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": billed_tokens(body, multiple),
                                           "completion_tokens": 3}}))

    return transport


def billed_tokens(body, multiple):
    """What that transport reports for one dispatched payload: the payload's own UTF-8 bytes, times
    the multiple it was built with -- one unit with the estimate, so the ratio IS the multiple."""
    return round(len(body) * multiple)


def test_request_input_bound():
    """Production's fail-closed bound, over the DISPATCHED request: UTF-8 bytes of the wire payload
    plus 512 and 128 per message. `source` rides the request for inspectability and already sits
    inside the user turn, so serializing the request dict would count it twice."""
    check, check_true, fails = recorder()
    request = S.build_request("L1", "L1:c1", "SOURCE " * 50, model="m")
    payload = S.wire_payload(request)
    check("the bound is bytes + 512 + 128 per message", S.request_input_bound(request),
           len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
           + S.BOUND_RESERVE + S.BOUND_PER_MESSAGE * len(payload["messages"]))
    check("the reserves are exactly 512 + 128 per message",
           S.request_input_bound(request) - len(S.wire_body(request)),
           S.BOUND_RESERVE + S.BOUND_PER_MESSAGE * 2)
    # The two checks above pin the FORMULA against whatever the constants are. These pin the
    # VALUE: 512 and 128 are production's own literals (autobiographical.ts:2649), ported rather
    # than scaled, so a silent change to either is a fidelity regression, not a tuning choice.
    check("the reserves are production's literals, ported not scaled",
           (S.BOUND_RESERVE, S.BOUND_PER_MESSAGE), (512, 128))
    check("the dispatched payload is model/ceiling/stream/messages", sorted(payload),
           ["max_tokens", "messages", "model", "stream"])
    check("two turns are dispatched, and counted", len(payload["messages"]), 2)
    check("`source` is not among the dispatched fields", "source" in json.dumps(payload), False)
    check_true("...but the source is in the request, inside the user turn",
                "SOURCE" in payload["messages"][1]["content"])
    check_true("serializing the request dict instead would count it twice",
                len(json.dumps(request, ensure_ascii=False).encode("utf-8")) > len(S.wire_body(request)))
    check("the client sends exactly what the bound measured", json.loads(S.wire_body(request)),
           payload)
    check("the estimate is a different, smaller number -- metadata, not the gate",
           S.request_input_tokens(request) < S.request_input_bound(request), True)
    check("the estimate is the payload's UTF-8 bytes, before the reserves",
           S.request_input_tokens(request), len(S.wire_body(request)))
    return fails or None


def test_bound_refuses_what_the_estimate_admits():
    """The hole this closes, narrowed to its remainder: the estimate is the payload's own bytes now,
    so exactly the reserves (512 + 128 per message) sit between the two numbers -- a request the
    estimate calls the last admissible byte is still refused once the reserves are added."""
    check, check_true, fails = recorder()
    engine = S.summarizer(S.mock_model())
    probe, _s = engine_request(engine, "L1", "L1:c1", "x")   # the estimate grows 1:1 with source bytes
    room = engine.budget - probe["max_tokens"]
    source = "x" * (room - (S.request_input_tokens(probe) - 1))
    request, _s = engine_request(engine, "L1", "L1:c1", source)
    check("the estimate alone would admit it, exactly: the last admissible byte",
           S.request_input_tokens(request), room)
    check_true("the bound refuses it",
                S.request_input_bound(request) + request["max_tokens"] > engine.budget,
                f"bound {S.request_input_bound(request)}")
    got, engine = S.complete(engine, "L1", "L1:c1", source)
    check("and admission follows the bound, not the estimate", got, (None, "overflow"))
    check("nothing was dispatched for it",
           (S.usage(engine)["input_tokens"], S.usage(engine)["output_tokens"]), (0, 0))
    check_true("the receipt shows both numbers, so the gap is visible",
                "bound" in engine.last_reply.detail and "estimate" in engine.last_reply.detail,
                engine.last_reply.detail)
    return fails or None


def test_calibration_moves_and_applies_to_estimates():
    """An EMA of real/estimated over dispatched calls, applied to ESTIMATES only: the bound must not
    move, because a bound needs no calibrating -- that is the point of having one.

    The provider's reported input is derived from the request it was handed (`transport_true_cost`),
    so the sample's ratio is the fixture's own number and not a literal that the prompt's size can
    push out of band."""
    check, check_true, fails = recorder()
    multiple = 1.5
    transport = transport_true_cost(multiple)
    engine = S.summarizer(S.http_model(transport=transport))
    request, _s = engine_request(engine, "L1", "L1:c1", "src")
    raw, bound = S.request_input_tokens(request), S.request_input_bound(request)
    check("the multiplier starts at 1.0", S.usage(engine)["calibration"], 1.0)
    check("so the estimate is the raw payload bytes", S.estimate(engine, request), raw)
    _outcome, engine = S.complete(engine, "L1", "L1:c1", "src")
    billed = billed_tokens(S.wire_body(engine.last_request), multiple)   # what that transport reported
    ratio = billed / raw                                     # real over the CALIBRATED estimate
    learned = 1.0 + S.CALIBRATION_ALPHA * (ratio - 1.0)      # production's EMA, once
    check_true("the sample is in band, so it is learned", 0.6 <= ratio <= 1.8, f"ratio {ratio:.2f}")
    check("the multiplier moved toward it", S.usage(engine)["calibration"], round(learned, 3))
    check("the call's own estimate rides the reply, as it was when it was built",
           engine.last_reply.estimated_tokens, raw)
    check("the session totals that same number", S.usage(engine)["estimated_tokens"], raw)
    check("the next estimate is in calibrated units", S.estimate(engine, engine.last_request),
           round(raw * learned))
    check("the bound did NOT move with the multiplier", S.request_input_bound(engine.last_request),
           bound)
    check("provider usage stays authoritative for the token ledger",
           S.usage(engine)["input_tokens"], billed)
    _outcome, engine = S.complete(engine, "L1", "L1:c1", "src")   # the same request, one call later
    check("a later call's estimate carries the multiplier learned before it",
           engine.last_reply.estimated_tokens, round(raw * learned))
    check("...and the ledger still counts what the provider said",
           S.usage(engine)["input_tokens"], billed + billed_tokens(S.wire_body(engine.last_request),
                                                                  multiple))

    transport = transport_answering(
        (200, json.dumps({"choices": [{"message": {"content": "A MEMORY"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 100000, "completion_tokens": 3}})))
    engine = S.summarizer(S.http_model(transport=transport))
    _outcome, engine = S.complete(engine, "L1", "L1:c1", "src")
    check("an out-of-band sample is dropped, not learned", S.usage(engine)["calibration"], 1.0)

    engine = S.summarizer(S.mock_model())
    _outcome, engine = S.complete(engine, "L1", "L1:c1", "src")
    check("the mock reports its own estimate, so the demo learns nothing",
           S.usage(engine)["calibration"], 1.0)
    return fails or None


def test_calibration_pairs_each_reply_with_its_own_request():
    """The pairing is structural: a sample is (this request's estimate, this reply's usage). A reply
    from a much larger request paired with the first request's estimate would be far out of band and
    dropped, so the second sample moving the multiplier is what proves the pairing."""
    check, check_true, fails = recorder()
    transport = transport_true_cost(1.5)
    engine = S.summarizer(S.http_model(transport=transport))
    _outcome, engine = S.complete(engine, "L1", "L1:c1", "a small source")
    first = S.usage(engine)["calibration"]
    _outcome, engine = S.complete(engine, "L1", "L1:c2", "a much larger source " * 200)
    second = S.usage(engine)["calibration"]
    check("two replies were dispatched", S.usage(engine)["calls"], 2)
    check_true("the first sample was learned", first > 1.0, str(first))
    check_true("so was the second, though it answers a far bigger request", second > first,
                f"{first} -> {second}")
    check_true("and it is converging on the provider's true ratio, not drifting past it",
                1.0 < second <= 1.5, f"{second}")
    return fails or None


def test_admission_rejects_before_dispatch():
    """An oversized request is refused by the boundary, not by a provider: zero transport calls,
    nothing billed, and the reason is one the policy table already routes."""
    check, check_true, fails = recorder()
    canned = (200, json.dumps({"choices": [{"message": {"content": "A MEMORY"},
                                            "finish_reason": "stop"}]}))
    transport = transport_answering(canned)
    engine = S.summarizer(S.http_model(transport=transport))
    huge = "word " * (engine.budget * 4)                # ~400k estimated bytes: far over budget
    got, engine = S.complete(engine, "L1", "L1:c1", huge)
    check("an oversized source is refused as overflow", got, (None, "overflow"))
    check("the transport was never invoked",
           (S.usage(engine)["calls"], S.usage(engine)["input_tokens"]), (1, 0))
    check("nothing was billed",
           (S.usage(engine)["input_tokens"], S.usage(engine)["output_tokens"]), (0, 0))
    check("the attempt is still counted", S.usage(engine)["calls"], 1)
    check("the refusal is counted by reason", S.usage(engine)["reasons"], {"overflow": 1})
    check_true("the receipt carries the numbers, not just the word",
                "admission" in engine.last_reply.detail
                and str(engine.budget) in engine.last_reply.detail, engine.last_reply.detail)
    check("overflow stays a reason the policy routes", "overflow" in M.ACTIONS, True)

    engine = S.summarizer(S.mock_model({"c1": ["refusal"]}))
    refused, engine = S.complete(engine, "L1", "L1:c1", huge)
    check("the mock is bypassed too", refused, (None, "overflow"))
    check("...so its plan is untouched", engine.model.state["c1"], ("refusal",))
    return fails or None


def test_admission_dispatches_under_budget():
    check, check_true, fails = recorder()
    canned = (200, json.dumps({"choices": [{"message": {"content": "A MEMORY"},
                                            "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
    transport = transport_answering(canned)
    engine = S.summarizer(S.http_model(transport=transport))
    got, engine = S.complete(engine, "L1", "L1:c1", "a small source")
    check("an under-budget request dispatches", got, ("A MEMORY", None))
    check("the transport saw exactly one call",
           (S.usage(engine)["calls"], S.usage(engine)["input_tokens"]), (1, 7))
    check_true("and its bound never left the budget",
                S.request_input_bound(engine.last_request) + engine.last_request["max_tokens"]
                <= engine.budget)
    check("provider usage is billed as usual",
           (S.usage(engine)["input_tokens"], S.usage(engine)["output_tokens"]), (7, 3))
    check("the estimate is the payload bytes, as metadata",
           S.request_input_tokens(engine.last_request),
           len(S.wire_body(engine.last_request)))
    check("...and it is surfaced on the reply as such", engine.last_reply.estimated_tokens,
           S.request_input_tokens(engine.last_request))
    return fails or None


def test_admission_boundary_is_exact():
    """The line is the configured budget itself: input + reserve == budget dispatches, one byte
    more is refused -- for L1 and for merges alike."""
    check, check_true, fails = recorder()
    engine = S.summarizer(S.mock_model())
    check_true("the reserve stays a small share of the budget, as production's does",
                S.MAX_OUTPUT_TOKENS <= engine.budget // 10,
                f"reserve {S.MAX_OUTPUT_TOKENS} of budget {engine.budget}")
    check("the target and the budget are scaled together: ours * 10 == production's",
           (S.TARGET_TOKENS * 10, C.CONTEXT_BUDGET * 10), (2000, 200000))
    # The two checks above read config's constants. These read what the engine RUNS on, so a
    # default left at production's absolute scale (an unscaled budget, say) cannot slip through.
    check("the engine runs on the scaled budget, not production's absolute one",
           engine.budget, C.CONTEXT_BUDGET)
    check_true("the budget is the scaled one, not production's 200000",
                engine.budget < 200000, f"engine budget {engine.budget}")
    check("the target reaches the merge prompt",
           "Target ~200 tokens" in S.build_request("merge", "L2:L1-a+L1-b", "S", model="m")["user"],
           True)
    for kind, target in (("L1", "L1:c1"), ("merge", "L2:L1-a+L1-b")):
        at, over, room = boundary_sources(engine, kind, target)
        request, _s = engine_request(engine, kind, target, at)
        check(f"{kind}: the fixture lands on the last admissible bound",
               S.request_input_bound(request), room)
        check(f"{kind}: the admitted total lands exactly on the budget",
               S.request_input_bound(request) + request["max_tokens"], engine.budget)
        check(f"{kind}: bound + reserve == budget dispatches",
               S.complete(engine, kind, target, at)[0][1], None)
        check(f"{kind}: one byte past the bound is refused",
               S.complete(engine, kind, target, over)[0], (None, "overflow"))
    return fails or None


# ================== the system layer: truncation, demand, coverage, debt ==================
def session(turns, words=200):
    """A log whose chunker closes exactly one chunk per turn: four messages of ~words/4 tokens
    each, no tool block, so the chunk closes on the fourth message."""
    log = ()
    for turn in range(turns):
        for i in range(4):
            log = M.emit(log, "msg", turn * 4 + i + 1, [("text", f"T{turn} word " + "w " * words)])
    return log


def burn(log, engine, rounds=6):
    """Drive the derivation until nothing is due, one attempt at a time. Returns (log, engine)."""
    for _ in range(rounds):
        items = M.work(log)
        if not items:
            break
        log, engine, _outcome = M.step(log, items[0], engine, len(log))
    return log, engine


def test_truncation_prices_what_it_renders():
    """maxMessageTokens truncates the content that is EMITTED and prices the emitted bytes:
    production does both (`truncateContent`, :11096, called behind `msgCap > 0` at :5041, :7613,
    :8154, :8172, :9148, :9505, :9574; the price at :9148). Capping only the price is how a plan
    comes to stand for a window it does not render -- priced 333 for 424 tokens of text."""
    check, check_true, fails = recorder()
    blocks = [("text", "word " * 250)]                  # 1250 chars -> 312 tokens: over cap 250,
    log = ()                                            # under the 2x shard threshold
    for i in range(M.MIN_MSGS):
        log = M.emit(log, "msg", i + 1, blocks)
    chunk = M.archive(log).chunks[0]
    truncated = M.truncate_blocks(blocks)
    per_message = min(M.chars(truncated) // 4,
                      M.MAX_MESSAGE_TOKENS + M.MAX_MESSAGE_HEADROOM)
    check_true("the fixture sits in the cap's band: over the cap, under the shard threshold",
                M.MAX_MESSAGE_TOKENS < M.chars(blocks) // 4 <= 2 * cm.CHUNK_TOKENS,
                f"{M.chars(blocks) // 4} tokens")
    check_true("the over-cap message is truncated", len(chunk.text) < len(blocks[0][1]) * M.MIN_MSGS)
    check("every message carries the truncation mark, as production's does",
           chunk.text.count("[truncated"), M.MIN_MSGS)
    check("the mark names the ORIGINAL size, in tokens, ceiling as production's does",
           f"original was {-(-len(blocks[0][1]) // 4)} tokens" in chunk.text, True)
    check("priced == rendered: the chunk is priced from the blocks it renders",
           chunk.tokens, per_message * M.MIN_MSGS)
    check_true("...and not from the cap's ceiling, which is slack, not the price",
                chunk.tokens < M.MIN_MSGS * (M.MAX_MESSAGE_TOKENS + M.MAX_MESSAGE_HEADROOM))
    check("the old price-only rule would have charged the ceiling per message",
           M.MAX_MESSAGE_TOKENS + M.MAX_MESSAGE_HEADROOM, 300)
    check("a message under the cap is untouched, and priced as before",
           M.truncate_blocks([("text", "w " * 40)]), [("text", "w " * 40)])
    check("the cap can be switched off entirely, as the library default does",
           M.truncate_blocks([("text", "##")]), [("text", "##")])
    return fails or None


def test_demand_path_bypasses_the_holdback():
    """kv-stable's demand-side production (kv-stable.ts:165-198): when the solve ESCALATES -- even
    the ideal, fully-folded cut is over W -- the uncovered foldable chunks are asked for as L1
    requests, and the consumer lets a demanded chunk past the `l1HoldbackChunks` window (:998,
    :3992-3999). Emitting none is what wedged tight operating points permanently.

    The holdback is raised from its default here on purpose: our tail window (400 tokens) always
    covers the newest chunk, and demand never names a raw-zone chunk, so at the default the two
    windows coincide and the bypass has nothing to bypass. Production's deadlock repro is exactly
    the case where they diverge -- a holdback window wider than the raw zone."""
    check, check_true, fails = recorder()
    log = session(5, words=400)                          # ~800-token chunks: the tail holds ONE
    a = M.archive(log)
    check("five turns close five chunks", [c.id for c in a.chunks], [f"c{i}" for i in range(5)])
    keep = M.TAIL_HOLDBACK
    M.TAIL_HOLDBACK = 2
    try:
        engine = S.summarizer(S.mock_model())
        log, engine = burn(log, engine)
        a = M.archive(log)
        check("the two newest chunks are held back, and the head chunk is never compressed at all",
               [c.id for c in a.chunks if c.id not in a.l1_of], ["c0", "c3", "c4"])
        check("so the derivation derives nothing for them",
               [i.target for i in M.work(log) if i.kind == "L1"], [])
        plan = cm.plan_controlled_frontier(a, {}, 1200, 1200, ())
        check_true("the plan escalated: even the ideal cut is over W", plan["tokens"] > 1200,
                    f"ideal {plan['tokens']}")
        check("so it produced the uncovered foldable run -- and not the raw-zone chunk",
               plan["produced"], [("c3", "c3")])
        demanded = M.demanded_chunks(plan["produced"], a)
        check("the consumer maps the range back onto a closed chunk", sorted(demanded), ["c3"])
        check("...and demand is what puts it back into the derivation",
               [i.target for i in M.work(log, demanded) if i.kind == "L1"], ["L1:c3"])
        check("a plan that did NOT escalate produces nothing",
               cm.plan_controlled_frontier(a, {}, 100000, 100000, ())["produced"], [])
    finally:
        M.TAIL_HOLDBACK = keep
    check("the default holdback is what it was", M.TAIL_HOLDBACK, keep)
    bare = session(3, words=400)
    plan = cm.plan_controlled_frontier(M.archive(bare), {}, 100, 100, ())
    check("at the default, the newest chunk is raw-zone, so demand never names it",
           [run for run in plan["produced"] if run[1] == "c2"], [])
    return fails or None


def test_coverage_invariant_is_fatal():
    """Production's coverage invariant (`assertFullCoverage`, :4655; `UncoveredDropError` exported
    at src/index.ts:38): nothing is ever silently dropped from the window. It is checked on the
    EMITTED render, so a recall that stands for a chunk its leaves do not cover is caught -- the
    stale or duplicated mint over drifted chunk ids, which an id-based `work()` test cannot see."""
    check, check_true, fails = recorder()
    log = session(2)
    log = mint(log, "L1:c0", "L1-c0", 1, ("c0",), "the first memory of this span")
    log = mint(log, "L1:c1", "L1-c0", 1, ("c1",), "a re-mint that drifted onto another chunk")
    a = M.archive(log)
    check("both chunks resolve to the same summary",
           (a.l1_of["c0"], a.l1_of["c1"]), ("L1-c0", "L1-c0"))
    check("...whose leaves cover only the second", a.summaries["L1-c0"].leaves, ("c1",))
    check("a healthy frontier (everything raw) passes",
           M.assert_coverage(a, {c.id: 0 for c in a.chunks}), None)
    check("a healthy fold passes too",
           M.assert_coverage(a, {"c0": 0, "c1": 0}), None)
    try:
        M.assert_coverage(a, {"c0": 1, "c1": 1})
        check("a chunk its own recall does not cover fails loudly",
               "no error raised", "UncoveredDropError")
    except M.UncoveredDropError as dropped:
        check_true("a chunk its own recall does not cover fails loudly",
                    "c0" in str(dropped), str(dropped))
        check_true("...and the message says which units were emitted instead",
                    "recall:L1-c0" in str(dropped), str(dropped))
    return fails or None


def test_clear_restores_derivable_work():
    """Production ships two ways out of terminal debt (`clearCompressionRefusalQuarantine`, :1263;
    `clearQuarantineForCompressedChunk`, :3441): an explicit operator escape hatch, and an
    automatic clear when the same span later mints. Ours is the same pair over the log, and the
    point of both is that the unit becomes DERIVABLE WORK again rather than resting forever.

    The fixture is four turns rather than two because the unit under test has to be one the
    derivation will actually hand out: c0 is the head chunk and c3 the tail, so c1 is the first
    compressible span a session this short has."""
    check, check_true, fails = recorder()
    log = session(4)
    engine = S.summarizer(S.mock_model({"c1": ["refusal"] * M.MAX_ATTEMPTS}))
    log, engine = burn(log, engine)
    check("the unit burned every attempt", M.attempts(log, "L1:c1"), M.MAX_ATTEMPTS)
    check("so it is terminal debt", M.stalled(log, "L1:c1"), True)
    check("...and it is no longer derived work", [i.target for i in M.work(log)], [])
    check("the log says so as an event, not as an edited structure",
           [e for e in log if isinstance(e, M.Clear)], [])
    log = M.clear_debt(log, "L1:c1", "operator: request shape changed")
    check("the clear is one event in the log",
           [e for e in log if isinstance(e, M.Clear)], [M.Clear("L1:c1", "operator: request shape changed")])
    check("it resets the count the debt is a projection of", M.attempts(log, "L1:c1"), 0)
    check("so the unit is derivable work again", [i.target for i in M.work(log)], ["L1:c1"])
    check("a retry starts canonical: the reshape reason is gone with the debt",
           M.last_reason(log, "L1:c1"), None)
    check("everything before the clear is still in the log",
           len([e for e in log if isinstance(e, M.Fail)]), M.MAX_ATTEMPTS)

    later = session(4)                                   # the automatic half, no clear event
    engine = S.summarizer(S.mock_model({"c1": ["refusal"] * M.MAX_ATTEMPTS}))
    later, engine = burn(later, engine)
    check("the unit is terminal debt", M.stalled(later, "L1:c1"), True)
    later = mint(later, "L1:c1", "L1-c1", 1, ("c1",), "the memory a later shape produced")
    check("a later mint of the same span clears it with no event at all",
           (M.stalled(later, "L1:c1"), [e for e in later if isinstance(e, M.Clear)]), (False, []))
    return fails or None


def test_stale_discard_burns_no_attempt():
    """The stale-version rule: a reply computed against a log that moved under the in-flight call is
    discarded and recorded as `stale` -- the system's own reason, deliberately NOT one of the
    boundary's `ACTIONS` keys -- and it burns no attempt, because a fork landing mid-call is not the
    unit's fault (same category as a provider blip). Before this exclusion, three unlucky forks read
    as terminal debt. The demo fires this rule at `fork_turn`; this test pins it directly."""
    check, check_true, fails = recorder()
    log = session(4)
    engine = S.summarizer(S.mock_model())
    item = ("L1:c1", "L1", "c1")
    v = len(log)
    log = M.emit(log, "msg", 900_000, [("text", "a forked branch lands mid-flight")])
    log, engine, outcome = M.step(log, item, engine, v)
    check("a reply against a moved log is discarded", outcome, "discarded-stale")
    check("the discard is recorded as the system's own reason", M.fails(log), [("L1:c1", "stale")])
    check("stale is not a boundary reason", "stale" in M.ACTIONS, False)
    check("nothing was minted from the stale result", M.mints(log), set())
    check("...and the discard burns no attempt", M.attempts(log, "L1:c1"), 0)
    check("...so the unit is not stalled and stays derivable",
           (M.stalled(log, "L1:c1"), [i.target for i in M.work(log) if i.target == "L1:c1"]),
           (False, ["L1:c1"]))
    for _ in range(M.MAX_ATTEMPTS + 1):
        log = M.emit(log, "fail", "L1:c1", "stale")
    check("stale discards never accumulate into terminal debt", M.stalled(log, "L1:c1"), False)
    log, engine, outcome = M.step(log, item, engine, len(log))
    check("and the very next attempt mints, canonical shape", outcome, "done")
    return fails or None


# ================== the audit trail refuses to contradict itself ==================
def report_error(knobs):
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            C.report(knobs=knobs)
    except SystemExit as refused:
        return str(refused)
    return "no refusal"


def test_compressible_zone_is_the_policys_boundary():
    """Production does not chunk what it will never fold: `rebuildChunks` is handed
    `getCompressibleMessages` (:10054-10056), which is everything before the recent window minus
    the head window and every pinned position (:9967-9980) -- so a head, tail or pinned message is
    never inside a chunk, never compressed, and never carries an L1. Our chunker runs over every
    message (the chunk projection also prices the window and renders raw units), so the boundary is
    applied to the DERIVATION instead, and this test is about that boundary being the POLICY's.

    `connectome_min.raw_zone` is that boundary, and it is now the SAME function the planner and the
    derivation both read, so the two cannot drift by construction; what is checked here is the
    sharing -- the derivation refuses exactly the chunks the planner's own output leaves raw. A
    raw-zone chunk cannot fold because `fold_depth_cap` returns -1 for it
    (`connectome_min.py`), and every chunk outside the zone does fold, so the check is not
    vacuous."""
    check, check_true, fails = recorder()
    log = memories(session(12))                     # every chunk has an L1, so folding is possible
    a = M.archive(log)
    zone = cm.raw_zone(a, ())
    check("the zone is the head chunk plus the tail window, by tokens",
           sorted(zone, key=lambda c: int(c[1:])), ["c0", "c11"])
    check("twelve chunks of ~411 tokens put exactly the newest one in the tail",
           a.chunks[-1].tokens >= C.TAIL_TOKENS, True)
    folded = cm.plan_controlled_frontier(a, {}, 1, 1, ())["F"]
    check("under a plan that folds everything it can, every raw-zone chunk stays raw",
           [cid for cid in sorted(zone, key=lambda c: int(c[1:])) if folded[cid] != 0], [])
    check("...and every chunk outside the zone is folded, so the check above is not vacuous",
           [c.id for c in a.chunks if c.id not in zone and folded[c.id] == 0], [])

    bare = session(12)
    a2 = M.archive(bare)
    zone2 = cm.raw_zone(a2, ())
    derived = [i.payload for i in M.work(bare) if i.kind == "L1"]
    check("the derivation names every chunk outside the zone, and nothing inside it",
           (derived, [cid for cid in derived if cid in zone2]), (["c1", "c2", "c3", "c4", "c5", "c6",
                                                                  "c7", "c8", "c9", "c10"], []))
    check("...which is the set the plan's own output leaves raw, not the zone function twice",
           [cid for cid in derived if folded.get(cid, 0) == 0], [])

    pinned = session(12)
    pinned = M.emit(pinned, "protect", 9, 12, 0, 0, "operator pin")   # messages 9-12: the third chunk
    a3 = M.archive(pinned)
    check("the pin lands on the chunk that owns those messages",
           sorted(M.protected_ids(pinned, a3)), ["c2"])
    check("a pin joins the zone, which is what keeps it out of the derivation",
           sorted(cm.raw_zone(a3, M.protected_ids(pinned, a3)) - zone2), ["c2"])
    check("so no L1 work is derived for it",
           [i.payload for i in M.work(pinned) if i.kind == "L1" and i.payload in ("c0", "c2", "c11")], [])
    return fails or None


def test_head_section_in_a_real_l1_request():
    """The identity anchor of every L1 request (:5266-5278) is `uncovered(head chunks)`, so it is
    inert the moment the head chunk has an L1 of its own -- which, while the derivation compressed
    every closed chunk, was always. This probes a REAL session's request rather than inferring
    from `source()`: three steps mint three L1s, and the third request must carry the head chunk
    raw, first, ahead of the prior recall pairs.

    The order is the point: when the head followed the pairs it read as the most recent live
    conversation, thin chunks narrated it as fresh events, and the error compounded across merges
    into runaway false memories (:5268-5278). A section that is always empty cannot be ordered
    wrongly, which is how the rule was protected by nothing at all."""
    check, check_true, fails = recorder()
    log = session(8)
    engine = S.summarizer(S.mock_model())
    for cid in ("c1", "c2", "c3"):                   # three real L1 requests, through the system
        log, engine, _outcome = M.step(log, (f"L1:{cid}", "L1", cid), engine, len(log))
    a = M.archive(log)
    request, prefix = engine.last_request, engine.last_request["prefix"]
    head_text = a.chunks[0].text
    check("the head chunk still has no L1, which is what makes the section non-empty",
           [cid for cid in a.l1_of if cid == "c0"], [])
    check("the request's target is the third compressible chunk", request["target"], "L1:c3")
    check_true("the head chunk is the request's first prefix section", prefix.startswith(head_text),
                prefix[:60])
    check_true("...and the prior recall pairs are there too, so the order is a real constraint",
                "[CM] Recall memory L1-c1." in prefix and "[CM] Recall memory L1-c2." in prefix,
                prefix[:200])
    check_true("the head precedes the first prior recall pair",
                prefix.index(head_text) < prefix.index("[CM] Recall memory L1-c1."), prefix[:200])
    check_true("...and it precedes the pairs in the assembled turn, ahead of the marker",
                request["user"].index(head_text) < request["user"].index("[CM] Recall memory")
                < request["user"].index(MARKER), request["user"][:200])
    check("the head is raw and whole: nothing stands for it, so nothing replaces it",
           prefix.split("\n\n")[0], head_text)
    return fails or None


def test_overlap_guard_is_a_span_test():
    """Production's duplicate-formation guard is a test on MESSAGES, not on chunk ids
    (`_overlapBlocked`, :5183-5203; the fully-covered arm above it at :5164-5181). Ours was
    `c.id not in a.l1_of` -- an id test -- and AUDIT.md's §1.3 row recorded the consequence:
    "after any re-derivation that shifts chunk ids, a message can be compressed twice".

    The drift here is the real mechanism rather than a staged one: the chunker prices the
    POST-strip blocks, so once the document message's image is stripped it stops being sharded
    (>2x targetChunkTokens) and the three chunks it was collapse into one -- every id after it
    moves by two while the messages stay exactly where they were. The L1 minted over the third
    slice then owns messages that a chunk with no L1 of its own now covers."""
    check, check_true, fails = recorder()
    log = ()
    for n in range(1, 5):                                # 4 x 80 tokens: the head chunk
        log = M.emit(log, "msg", n, [("text", f"m{n} " + "w " * 160)])
    log = M.emit(log, "msg", 5, [("image",), ("text", "doc " * 590)])
    for n in range(6, 14):
        log = M.emit(log, "msg", n, [("text", f"m{n} " + "w " * 160)])
    before = M.archive(log)
    check("while its image is live the document message shards into three chunks",
           [c.id for c in before.chunks], ["c0", "c1", "c2", "c3", "c4", "c5"])
    check("...all three carrying the same message range, which is why the ordinal exists",
           [M.chunk_spans(log)[cid] for cid in ("c1", "c2", "c3")],
           [(5, 5, 0), (5, 5, 1), (5, 5, 2)])
    check("the derivation hands out all three",
           [i.target for i in M.work(log) if i.kind == "L1"], ["L1:c1", "L1:c2", "L1:c3"])
    log = mint(log, "L1:c3", "L1-c3", 1, ("c3",), "MEMORY-OF-THE-THIRD-SLICE")

    for n in range(14, 18):                              # depth past the strip threshold
        log = M.emit(log, "msg", n, [("text", f"m{n} " + "w " * 160)])
    after = M.archive(log)
    check("once the image is stripped the shard group collapses and every later id moves",
           [c.id for c in after.chunks], ["c0", "c1", "c2", "c3"])
    check("the chunk that now covers the document's messages is c1 -- a different id",
           M.chunk_spans(log)["c1"], (5, 8, 0))
    check("the live L1 still owns the messages it was minted over",
           M.l1_spans(log)["L1-c3"], {(5, 5, 2)})
    check("the id test would call c1 compressible", "c1" in after.l1_of, False)
    check("the span test sees that its messages are already owned, under other boundaries",
           M.overlap_blocked(log, after), [("c1", "L1-c3")])
    check("...and that the arm is not the exact one", M.covered_by_l1(log, after)["c1"][1], False)
    check("so no L1 work is derived for it -- and c1 is the only compressible chunk here",
           ([i.target for i in M.work(log)], [c.id for c in after.chunks
                                          if c.id not in cm.raw_zone(after, ())]), ([], ["c1"]))
    return fails or None


def test_merge_span_guard():
    """`contiguousMergeCandidates` refuses two kinds of candidate (:6654-6688): a span whose source
    no longer resolves to live messages, which "can NEVER merge" and is warned rather than dropped
    silently (a review finding in its own right), and a span that is wide FOR ITS LEVEL.

    The scaling is the load-bearing half, and it is checked here as REACHABILITY rather than as
    arithmetic: limits are `base x k^(max(0, level-3))`, so at level 4 the limit is six times the
    base. A 603-message L4 is therefore admissible, and it is exactly the case a flat limit
    forbids -- which is how "every mythos L4 spans 3.0k-6.9k messages > 1500, so an L5 was
    structurally impossible at any store state and the fold floor sat ~23k above where one L5 puts
    it" (:6670-6677). A 1003-message L4 is wide for its level and is quarantined."""
    check, check_true, fails = recorder()
    def candidates(span_to):
        """Six level-4 candidates over a 251-chunk session, CONTIGUOUS (the odometer merges no
        run with a hole): five narrow, one reaching from just past them to `span_to`."""
        log = session(251)                               # 251 chunks x 4 messages = 1004
        for i in range(5):
            log = mint(log, f"L4:n{i}", f"L4-n{i}", 4, (f"c{i}",), f"MEMORY-n{i}")
        return mint(log, "L4:x", "L4-x", 4, ("c5", span_to), "MEMORY-x")

    check("the session's chunks are four messages each", M.chunk_spans(session(4))["c0"], (1, 4, 0))
    check("the limit is the base for L1-L3 and base x k above it",
           [M.span_limit(level) for level in (1, 2, 3, 4, 5)],
           [M.MERGE_MAX_SOURCE_SPAN_MESSAGES * 6 ** max(0, level - 3) for level in (1, 2, 3, 4, 5)])
    check("...which is production's own expression with the demo's scaled base",
           (M.MERGE_MAX_SOURCE_SPAN_MESSAGES, [M.span_limit(level) for level in (4, 5)]), (150, [900, 5400]))

    wide = candidates("c250")                            # 983 messages: wide for an L4
    check("a candidate wide for its level is quarantined, with production's reason",
           [r for _sid, r in M.merge_exclusions(wide) if _sid == "L4-x"],
           ["wide-span quarantine -- span 983 msgs > limit 900 (base 150 x 6^1)"])
    check("the quarantined candidate no longer counts toward the six -- and the run it strands "
           "can never grow, so the interior escape consolidates it at 2 (:6718-6721)",
           [i.target for i in M.work(wide) if i.kind == "merge"],
           ["L5:L4-n0+L4-n1+L4-n2+L4-n3+L4-n4"])
    check("...while the L1 work for the session's own chunks is unaffected",
           len([i for i in M.work(wide) if i.kind == "L1"]) > 0, True)

    long = candidates("c150")                            # 583 messages: long, legal at an L4
    check("a 583-message L4 is NOT quarantined: that is what the level scaling buys",
           [r for _sid, r in M.merge_exclusions(long) if _sid == "L4-x"], [])
    check("so six eligible candidates do merge",
           [i.target for i in M.work(long) if i.kind == "merge"],
           ["L5:L4-n0+L4-n1+L4-n2+L4-n3+L4-n4+L4-x"])

    flat = candidates("c150")                            # the same fixture, one level down
    flat = mint(flat, "L2:w", "L2-w", 2, ("c0", "c150"), "MEMORY-w")
    check("at level 2 the same span IS wide for its level, with no scaling involved",
           [r for _sid, r in M.merge_exclusions(flat) if _sid == "L2-w"],
           ["wide-span quarantine -- span 603 msgs > limit 150 (base 150 x 6^0)"])

    gone = session(4)
    gone = mint(gone, "L2:orphan", "L2-orphan", 2, ("c1", "c99"), "MEMORY-orphan")
    check("a span whose source no longer resolves is frontier debt, not a silent drop",
           [r for _sid, r in M.merge_exclusions(gone) if _sid == "L2-orphan"],
           ["source position unresolved -- permanently unmergeable, frontier debt"])
    return fails or None


def test_merge_candidacy_is_an_odometer():
    """The run selection the mergeThreshold row now claims: the first STRICTLY CONTIGUOUS run of
    six unmerged siblings merges (:6698) and a hole is never bridged; a run that can never grow --
    a newer summary exists at its level, so the run is interior -- consolidates at 2 instead of
    waiting for a sixth that can never arrive (:6718-6721)."""
    check, check_true, fails = recorder()
    full = memories(session(7))                        # L1s c0..c6: one contiguous run of 7
    check("six contiguous siblings at the live end merge, the first group only",
           [i.target for i in M.work(full) if i.kind == "merge"],
           ["L2:L1-c0+L1-c1+L1-c2+L1-c3+L1-c4+L1-c5"])

    holed = memories(session(7), skip=("c3",))         # the same six ids, split 3 + 3 by one hole
    check("a hole breaks the run: no six-merge forms, and the half the hole strands "
           "consolidates at 3 instead",
           [i.target for i in M.work(holed) if i.kind == "merge"],
           ["L2:L1-c0+L1-c1+L1-c2"])

    stranded = memories(session(7), skip=("c4", "c5", "c6"))
    stranded = mint(stranded, "L1:c4", "L1-c4", 1, ("c4",), "MEMORY-c4")   # already merged into an L2:
    stranded = mint(stranded, "L1:c5", "L1-c5", 1, ("c5",), "MEMORY-c5")   # the run ahead of them can
    stranded = mint(stranded, "L2:x", "L2-x", 2, ("L1-c4", "L1-c5"), "MEMORY-x")  # never grow past them
    check("a four-run whose successor is already merged consolidates whole, below the six",
           [i.target for i in M.work(stranded) if i.kind == "merge"],
           ["L2:L1-c0+L1-c1+L1-c2+L1-c3"])

    lone = session(7)                                  # two isolated L1s, each a run of one --
    lone = mint(lone, "L1:c0", "L1-c0", 1, ("c0",), "MEMORY-c0")           # one interior, one at the
    lone = mint(lone, "L1:c2", "L1-c2", 1, ("c2",), "MEMORY-c2")           # live end
    check("but a run of one never merges, interior or not",
           [i.target for i in M.work(lone) if i.kind == "merge"], [])
    return fails or None


def test_pressure_gates_l1_derivation():
    """Lazy production (`work`'s `pressure`): a calm session mints nothing -- the demand channel,
    generalized from escalation to ordinary pressure -- while a demanded chunk bypasses both the
    holdback and the gate. Eager stays the default, so every fixture above reads unchanged."""
    check, check_true, fails = recorder()
    log = session(4)
    check("under pressure the derivation is production's eager one",
           [i.target for i in M.work(log)], ["L1:c1", "L1:c2"])
    check("a calm session derives nothing: no folding pressure, no model calls",
           [i.target for i in M.work(log, pressure=False)], [])
    check("...and demand opens the gate, as it opens the holdback",
           [i.target for i in M.work(log, ("c2",), pressure=False)], ["L1:c2"])
    return fails or None


def test_config_status_integrity():
    """A status that names the production DEFAULT has to agree with the value recorded beside it:
    `ignored` says default-OFF, `unmodelled`/`MISSING` say default-ON. The audit found five rows
    carrying `ignored` while production has them on (D1), which is a lie a reader cannot check --
    so the table refuses to describe itself that way."""
    check, check_true, fails = recorder()
    check("the shipped table has no contradictory status", C.audit_status_conflicts(), [])
    check("nothing claims default-OFF any more",
           [k.name for k in C.KNOBS if k.status == "ignored"], [])
    on = next(k for k in C.KNOBS if k.name == "attachmentsIgnoreSize")
    check("a live default-ON pricing rule, recorded as such", (on.production, on.status), (True, "unmodelled"))
    doctored = tuple(replace(k, status="ignored") if k is on else k for k in C.KNOBS)
    check_true("re-labelling a default-ON knob `ignored` is caught",
                any("attachmentsIgnoreSize" in c for c in C.audit_status_conflicts(doctored)),
                str(C.audit_status_conflicts(doctored)))
    check_true("report() refuses that table instead of printing it",
                "refuses" in report_error(doctored), report_error(doctored))
    off = next(k for k in C.KNOBS if k.name == "reachTokens (P)")
    doctored = tuple(replace(k, status="unmodelled") if k is off else k for k in C.KNOBS)
    check_true("...and a default-OFF knob called `unmodelled` is caught too",
                any("reachTokens" in c for c in C.audit_status_conflicts(doctored)))
    check("the real table reports", report_error(C.KNOBS), "no refusal")
    return fails or None


# ================== usage accounting ==================
def test_admission_feeds_the_system_policy():
    """The reason has to be reachable *and* routable: an admission refusal arrives at
    minisystem.step() as a normal failure and the existing table reshapes the source."""
    check, check_true, fails = recorder()
    log = ()
    for i in range(4):
        log = M.emit(log, "msg", i + 1, [("text", "ZEBRA word " + "w " * 200)])
    check("the session built the chunk under test", [c.id for c in M.archive(log).chunks], ["c0"])
    engine = S.summarizer(S.mock_model(), budget=1)     # nothing can be admitted
    log, engine, outcome = M.step(log, ("L1:c0", "L1", "c0"), engine, len(log))
    check("step() routes an admission refusal to the reshape rung", outcome, "shrink")
    check("the log recorded it as an ordinary overflow failure", M.fails(log), [("L1:c0", "overflow")])
    return fails or None


def test_usage_accumulation():
    check, check_true, fails = recorder()
    engine = S.summarizer(S.mock_model({"c1": ["refusal", "provider_error",
                                               S.Reply(error="provider_error", retryable=False)]}))
    for _ in range(3):
        _got, engine = S.complete(engine, "L1", "L1:c1", "source words " * 50)
    usage = S.usage(engine)
    check("three calls are counted", usage["calls"], 3)
    check_true("input tokens accumulate", usage["input_tokens"] > 0, str(usage))
    check_true("output tokens accumulate", usage["output_tokens"] > 0, str(usage))
    check("failures are counted by reason",
           usage["reasons"], {"refusal": 1, "provider_error": 2})
    check("provider faults keep the retryable split",
           usage["faults"], {"retryable": 1, "non_retryable": 1})
    check("last_request / last_reply stay inspectable",
           (engine.last_request["target"], engine.last_reply.error), ("L1:c1", "provider_error"))
    return fails or None


# ================== the HTTP client, no network ==================
def test_http_openai_body_and_parse():
    check, check_true, fails = recorder()
    transport = transport_answering(
        (200, json.dumps({"choices": [{"message": {"content": "A MEMORY"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 123, "completion_tokens": 45}})))
    engine = S.summarizer(S.http_model(model="test-model", base_url="https://example.invalid/v1",
                                       api_key="k", transport=transport))
    (text, reason), engine = S.complete(engine, "L1", "L1:c1", "SOURCE")
    check("openai: the memory is accepted", (text, reason), ("A MEMORY", None))
    # The client dispatches from the config on its `state` and serializes `last_request`; both are
    # values, so what the transport was handed is reconstructed rather than recorded.
    cfg = engine.model.state
    sent = S.wire_payload(engine.last_request)
    check("openai: url", f"{cfg.base_url.rstrip('/')}/chat/completions",
           "https://example.invalid/v1/chat/completions")
    check("openai: bearer auth", f"Bearer {cfg.api_key}", "Bearer k")
    check("openai: system then user, nothing else", sent["messages"],
           [{"role": "system", "content": engine.last_request["system"]},
            {"role": "user", "content": engine.last_request["user"]}])
    check("openai: model and ceiling", (sent["model"], sent["max_tokens"], sent["stream"]),
           ("test-model", S.MAX_OUTPUT_TOKENS, False))
    check("openai: the ceiling is production's max(floor, target * 1.5)",
           S.MAX_OUTPUT_TOKENS, max(S.OUTPUT_FLOOR_TOKENS, int(S.TARGET_TOKENS * 1.5)))
    check("openai: the floor binds at demo scale, as production's does at its own",
           S.MAX_OUTPUT_TOKENS, S.OUTPUT_FLOOR_TOKENS)
    check("openai: usage is accounted", (S.usage(engine)["input_tokens"],
                                          S.usage(engine)["output_tokens"]), (123, 45))
    return fails or None


def test_http_classification():
    check, check_true, fails = recorder()
    cases = [
        ("429 is retryable", (429, '{"error": {"message": "rate limited"}}'),
         "provider_error", True),
        ("503 is retryable", (503, "upstream unavailable"), "provider_error", True),
        ("408 is retryable", (408, "timeout"), "provider_error", True),
        ("400 context_length is NOT retryable",
         (400, json.dumps({"error": {"type": "invalid_request_error",
                                     "message": "context length exceeded"}})),
         "provider_error", False),
        ("401 is NOT retryable", (401, '{"error": {"message": "bad key"}}'),
         "provider_error", False),
    ]
    for name, reply, reason, retryable in cases:
        transport = transport_answering(reply)
        engine = S.summarizer(S.http_model(transport=transport))
        got, engine = S.complete(engine, "L1", "L1:c1", "src")
        check(f"http: {name}", got, (None, reason))
        check(f"http: {name} -- retryable flag", engine.last_reply.retryable, retryable)
    transport = transport_answering(
        (400, json.dumps({"error": {"type": "invalid_request_error",
                                    "message": "context length exceeded"}})))
    engine = S.summarizer(S.http_model(transport=transport))
    _got, engine = S.complete(engine, "L1", "L1:c1", "src")
    check_true("http: the receipt carries the provider's words",
                "context length exceeded" in engine.last_reply.detail, engine.last_reply.detail)
    return fails or None


def test_http_reply_classification():
    check, check_true, fails = recorder()
    cases = [
        ("length stop is truncated",
         (200, json.dumps({"choices": [{"message": {"content": "half"}, "finish_reason": "length"}]})),
         (None, "truncated")),
        ("tool_calls is a tool_call",
         (200, json.dumps({"choices": [{"message": {"content": "",
                                                    "tool_calls": [{"id": "t1"}]},
                                        "finish_reason": "tool_calls"}]})),
         (None, "tool_call")),
        ("empty content is empty",
         (200, json.dumps({"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]})),
         (None, "empty")),
        ("content_filter is a refusal",
         (200, json.dumps({"choices": [{"message": {"content": "no"},
                                        "finish_reason": "content_filter"}]})),
         (None, "refusal")),
        ("a missing finish_reason is malformed",
         (200, json.dumps({"choices": [{"message": {"content": "text"}}]})),
         (None, "provider_error")),
        ("no choices is malformed", (200, json.dumps({"choices": []})), (None, "provider_error")),
        ("a non-json body is malformed", (200, "<html>502</html>"), (None, "provider_error")),
    ]
    for name, reply, want in cases:
        transport = transport_answering(reply)
        engine = S.summarizer(S.http_model(transport=transport))
        got, _engine = S.complete(engine, "L1", "L1:c1", "src")
        check(f"reply: {name}", got, want)
    return fails or None


def test_http_transport_faults():
    check, check_true, fails = recorder()
    def failing(exception):
        def transport(*_args, **_kwargs):
            raise exception
        return transport

    for name, exception in (("connection error", urllib.error.URLError("no route to host")),
                            ("timeout", TimeoutError("timed out"))):
        engine = S.summarizer(S.http_model(transport=failing(exception)))
        got, engine = S.complete(engine, "L1", "L1:c1", "src")
        check(f"transport: {name} is a retryable provider_error",
               (got, engine.last_reply.retryable),
               ((None, "provider_error"), True))
    return fails or None


# ================== the demo's story ==================
def test_demo_story():
    check, check_true, fails = recorder()
    out = io.StringIO()
    with redirect_stdout(out):
        D.demo()
    printed = out.getvalue()
    anchors = ["refusal         -> retry",
               "      discarded-stale L1:c3",     # the fork fires, and burns no attempt
               "      terminal        L1:c4",
               "      shrink          L1:c6",
               "      done            L1:c3",
               "  REFUSED: turn 8: 2296 tokens > wall 1224",
               "  demand path: the escalated plan produced [('c4', 'c4'), ('c7', 'c7')]",
               "  stalled work (terminal debt): ['L1:c4']",
               "  compressible zone: 5 of 10 chunks (5 protected by head/tail/pin: "
               "['c0', 'c1', 'c2', 'c8', 'c9'])",
               "  overlap-blocked chunks (a live L1 already owns their messages): none",
               "  merge candidates excluded (resolved span / wide for its level): none",
               "  failed attempts by reason: overflowx1, provider_errorx2, refusalx1, stalex1, "
               "tool_callx1, truncatedx2",
               "  compression calls spent: 11 (lazy: only while the last plan was over target)",
               "clear_debt(L1:c4) -> attempts=0 stalled=False, derivable again: ['L1:c4']",
               "  model usage: "]
    for anchor in anchors:
        check_true(f"demo still prints {anchor.strip()[:46]!r}", anchor in printed)
    return fails or None


def main():
    tests = [test_l1_request, test_l1_request_is_the_six_sections, test_l1_doc_aware_instruction,
             test_source_only_is_the_refusal_rung,
             test_prompt_provenance, test_merge_request,
             test_request_carries_source_only,
             test_request_scope_in_a_session, test_l1_prior_pairs_are_the_unmerged_frontier,
             test_l1_ladder_caps_newest_first_and_resorts,
             test_merge_target_is_one_level_deeper,
             test_merge_prefix_is_the_prior_content_only,
             test_merge_instruction_describes_what_is_shown,
             test_recall_budget_evicts_the_oldest_first, test_merge_reading_mode_instruction,
             test_gate, test_mock_classes, test_mock_digest,
             test_request_input_bound, test_bound_refuses_what_the_estimate_admits,
             test_calibration_moves_and_applies_to_estimates,
             test_calibration_pairs_each_reply_with_its_own_request,
             test_admission_rejects_before_dispatch, test_admission_dispatches_under_budget,
             test_admission_boundary_is_exact, test_admission_feeds_the_system_policy,
             test_usage_accumulation, test_http_openai_body_and_parse,
             test_http_classification, test_http_reply_classification, test_http_transport_faults,
             test_truncation_prices_what_it_renders, test_demand_path_bypasses_the_holdback,
             test_compressible_zone_is_the_policys_boundary,
             test_head_section_in_a_real_l1_request, test_overlap_guard_is_a_span_test,
             test_merge_span_guard, test_merge_candidacy_is_an_odometer,
             test_coverage_invariant_is_fatal, test_clear_restores_derivable_work,
             test_stale_discard_burns_no_attempt, test_pressure_gates_l1_derivation,
             test_config_status_integrity, test_demo_story]
    failed = []
    for test in tests:
        print(f"\n{test.__name__}")
        failed += test() or []
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
