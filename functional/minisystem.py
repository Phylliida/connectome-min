"""Minimal Connectome *system* -- the purely functional port of ../minisystem.py: one log, work
DERIVED (never queued), one model boundary (`llm/summarizer.py`), one failure policy (a table),
one protection type, one compare-and-swap commit, plus the default-on accounting rules and the
six rules that are not accounting.

    python3 demo.py    # the demo that drives this system; `connectome_min.py` is the POLICY

The port in one paragraph: the original already SAID "the log is the only store", but its write
side still appended in place (`emit`/`commit`) and its model boundary kept counters on an
object. Here the log is an immutable TUPLE of events; `emit` returns the next log; `commit` --
the compare-and-swap -- returns `(next_log, committed)`; and `step` threads both the log and
the boundary: `(log, boundary, outcome) = step(log, item, boundary, v)`. Every projection
(messages, spans, archive, the failure ledger, work) was already a fold over the log and keeps
its name and shape. No function mutates caller-visible state, so the folder's `_` prefix goes
unused here too. Local scratch that never escapes (the chunker's buffer, the image walk) is
still written with plain loops -- rebinding and building FRESH structures is not the mutation
the convention forbids.

Rationale, citations and audit evidence: ../README.md `## minisystem.py` + its appendix,
../AUDIT.md §8, §9.
"""
from collections import namedtuple
from dataclasses import replace
from functools import reduce

import connectome_min as cm
from config import (ACTIONS, DEMO, GRACE, IMAGE_CHARS, IMAGE_STRIP_DEPTH_TOKENS,
                    IMAGE_TOKENS, MAX_ATTEMPTS, MAX_LIVE_IMAGES, MAX_MESSAGE_HEADROOM,
                    MAX_MESSAGE_TOKENS, MERGE_MAX_SOURCE_SPAN_MESSAGES, MIN_MSGS, NO_TOOLS,
                    PROVIDER_STREAK, RECALL_BUDGET, RECALL_LABEL, RECALL_MEMORY_HEADER,
                    RECALL_PAIR_OVERHEAD, TAIL_HOLDBACK, TRUNCATION_NOTE)
from llm import summarizer as S

# ================== the log: the only store. Messages carry blocks ==================
# Named event records (namedtuples: immutable, still unpack like tuples): the discriminated union
# the log appends. `kind` strings at emit() map to these; reads are isinstance + field names.
Msg = namedtuple("Msg", "seq blocks")            # blocks: ("text"|"tool_use"|"tool_result", s) | ("image",)
Mint = namedtuple("Mint", "target id level children text owned")   # owned: spans stamped at mint time
Fail = namedtuple("Fail", "target reason")
Clear = namedtuple("Clear", "target why")
Emission = namedtuple("Emission", "levels")      # the frontier the host last emitted
Protect = namedtuple("Protect", "first last lo hi why")
EVENTS = {"msg": Msg, "mint": Mint, "fail": Fail,
          "clear": Clear, "frontier": Emission, "protect": Protect}

# Records carried between projections (same drop-in discipline):
Span = namedtuple("Span", "id first last tokens text salience group")
Range = namedtuple("Range", "first last ordinal")
Work = namedtuple("Work", "target kind payload")
Cand = namedtuple("Cand", "id first last")


def emit(log, kind, *fields):
    """Append, as a pure function: the log is a tuple and the NEXT log is returned. The original
    returned the event and mutated `log` in place; callers that need the event read `log[-1]`
    off the returned value."""
    return log + (EVENTS[kind](*fields),)


def commit(log, v, kind, *fields):
    """The write side is compare-and-swap: extend only if the log still matches the version the
    attempt read before its slow call. A mid-flight fork is a failed CAS -- and a failed CAS is
    not an attempt, which is all "stale burns nothing" means. Returns (log', committed): the
    flag is None exactly when the CAS failed, and log' is then the log unchanged."""
    if len(log) != v:
        return log, None
    return emit(log, kind, *fields), True


def project(log, init, step):
    """Every view of the log is a LEFT FOLD over it; this is the fold. The stateful projections
    (the archive, the failure ledger) are stated as (init, step) pairs so that "the log is the
    only store" is structural rather than stylistic -- the pure filters above and below are
    folds too degenerate to need a name."""
    return reduce(step, log, init)


def messages(log):
    return [(e.seq, e.blocks) for e in log if isinstance(e, Msg)]


def chars(blocks):
    """Byte proxy, production's estimator flattened: 4 chars per token, image = flat cost."""
    return sum(IMAGE_CHARS if b[0] == "image" else len(b[1]) for b in blocks)


def text_of(blocks):
    return "\n".join(b[1] for b in blocks if b[0] != "image") or "[image]"


def salience(blocks):
    """Port of `computeStaticSalience`: how much of a message is re-derivable payload, floored at 0.2."""
    total = external = 0
    for kind, *rest in blocks:
        body = rest[0] if rest else ""
        if kind == "text":
            total += len(body)
            external += sum(len(f) for f in body.split("```")[1::2])       # fenced code
            external += sum(len(l.strip()) for l in body.split("\n")
                            if l.strip().startswith(("http://", "https://")))  # bare link drop
        elif kind in ("tool_use", "tool_result"):
            total += len(body)
            external += len(body)                                          # re-derivable
        elif kind == "image":
            total += IMAGE_CHARS
            external += IMAGE_CHARS                                        # file/CDN retains it
        else:
            total += len(body)
    return 1.0 if total <= 0 else max(0.2, 1 - 0.8 * min(1.0, external / total))


def strip_images(msgs):
    """An eviction policy over the image population, recency-ordered: images past `maxLiveImages` newest
    (the count cap) or `imageStripDepthTokens` deep (the depth budget) become a marker, before any budget.
    Written as a fold over newest-first messages carrying (depth, live) forward; the kept blocks
    are built fresh per message."""
    def visit(state, seq_blocks):
        depth, live, out = state
        seq, blocks = seq_blocks
        kept = []
        for block in reversed(blocks):
            if block[0] != "image":
                kept.append(block)
                continue
            live += 1
            if depth > IMAGE_STRIP_DEPTH_TOKENS or live > MAX_LIVE_IMAGES:
                kept.append(("text", "[image stripped]"))
            else:
                kept.append(block)
            depth += IMAGE_TOKENS
        depth += chars([b for b in blocks if b[0] != "image"]) // 4
        return depth, live, [(seq, list(reversed(kept)))] + out
    return reduce(visit, reversed(msgs), (0, 0, []))[2]


def truncate_blocks(blocks):
    """Port of `truncateContent`: truncate what is EMITTED first, then price the bytes that will render."""
    if MAX_MESSAGE_TOKENS <= 0 or chars(blocks) // 4 <= MAX_MESSAGE_TOKENS:
        return blocks
    remaining, out = MAX_MESSAGE_TOKENS * 4, []
    for block in blocks:
        if block[0] != "text":
            out.append(block)
            continue
        if remaining <= 0:
            continue
        if len(block[1]) <= remaining:
            out.append(block)
            remaining -= len(block[1])
        else:
            out.append(("text", block[1][:remaining]
                        + TRUNCATION_NOTE.format(tokens=-(-len(block[1]) // 4))))
            remaining = 0
    return out


def spans(log):
    """The chunker is first-fit bin packing, message order preserved: a bin (chunk) closes at
    targetChunkTokens, and an oversized item -- a message larger than 2x the target -- is SPLIT into
    pieces sharing a bodyGroupId. A sharded message closes whatever is pending first -- a partial chunk,
    under target and possibly under MIN_MSGS, because the alternative is dropping those messages from every
    chunk, silently. Production's picker is per MESSAGE, so a closed chunk is one unit whose salience is its
    cheapest message's. AUDIT.md §9.

    Stated as one fold over the stripped messages: the accumulator is (closed spans, the pending
    bin), and every branch REBINDS it -- the pending buffer is local scratch, rebuilt per
    message, never shared."""
    def feed(state, seq_blocks):
        out, buf, tok, first, last, least = state
        seq, blocks = seq_blocks
        n = chars(blocks) // 4
        if n > 2 * cm.CHUNK_TOKENS:                          # shard a long message
            if buf:                                          # close the pending chunk rather than drop it
                out = out + [Span(f"c{len(out)}", first, last, tok, "\n".join(buf), least, None)]
                buf, tok, first, least = [], 0, 0, 1.0
            pieces = -(-n // cm.CHUNK_TOKENS)                 # ceil: every shard fits the target
            share = n // pieces
            text = text_of(blocks)                           # production splits the MESSAGE by
            step = max(1, len(text) // pieces)               # chars (chunkMessage): a shard is a
            shards = [Span(f"c{len(out) + k}", seq, seq, share,      # slice, never the whole
                           text[k * step:(k + 1) * step] if k < pieces - 1 else text[k * step:],
                           salience(blocks), f"g{seq}")              # each shard its own chunk
                      for k in range(pieces)]
            return out + shards, buf, tok, first, last, least
        blocks = truncate_blocks(blocks)                     # the cap truncates, then we price
        n = chars(blocks) // 4                               # ...the bytes we will render
        if MAX_MESSAGE_TOKENS:                               # and nothing is priced above cap+50
            n = min(n, MAX_MESSAGE_TOKENS + MAX_MESSAGE_HEADROOM)
        first = first or seq
        last = seq
        buf = buf + [text_of(blocks)]
        tok += n
        least = min(least, salience(blocks))
        has_tool_use = any(b[0] == "tool_use" for b in blocks)
        if tok >= cm.CHUNK_TOKENS and len(buf) >= MIN_MSGS and not has_tool_use:
            return out + [Span(f"c{len(out)}", first, seq, tok, "\n".join(buf), least, None)], \
                [], 0, 0, last, 1.0
        return out, buf, tok, first, last, least
    return reduce(feed, strip_images(messages(log)), ([], [], 0, 0, 0, 1.0))[0]


def archive(log):
    """Projection: chunks and summaries from events alone. Recalls are PRICED as production prices them --
    question label plus content; the header a positioned pair EMITS is not, and the `+50` is the prompt cap.
    AUDIT.md D6. A fold building the next Archive value per mint event -- connectome_min.mint's
    own shape, with the event's text instead of the policy's `summarize`."""
    def minted(a, e):
        if not isinstance(e, Mint):
            return a
        leaves = tuple(l for c in e.children
                       for l in (a.summaries[c].leaves if c in a.summaries else (c,)))
        pair = f"{RECALL_LABEL} {e.text}"
        s = cm.Summary(e.id, e.level, leaves, e.text, max(1, len(pair.split())))
        return replace(a,
                       summaries={**a.summaries, e.id: s},
                       children={**a.children, e.id: tuple(e.children)},
                       l1_of={**a.l1_of, e.children[0]: e.id} if e.level == 1 else a.l1_of,
                       parent=a.parent if e.level == 1 else
                              {**a.parent, **{c: e.id for c in e.children}})
    chunks = tuple(cm.Chunk(cid, last, text, tokens, sal)
                   for cid, _first, last, tokens, text, sal, _group in spans(log))
    return project(log, cm.Archive(chunks=chunks), minted)


def frontier(log):
    return next((e.levels for e in reversed(log) if isinstance(e, Emission)), {})


def mints(log):
    return {e.target for e in log if isinstance(e, Mint)}


def fails(log):
    return [(e.target, e.reason) for e in log if isinstance(e, Fail)]


def ledger(log, target):
    """One unit's whole debt vocabulary as a SINGLE fold of its events: burned attempts, the
    transient streak, the last reason, and whether a clear or a mint ever landed (both of
    production's clears, :3441). Every predicate below reads this one fold, so the vocabulary
    cannot disagree with itself. `stale` is recorded but burns nothing: not an attempt, not a
    fault -- a failed commit."""
    def count(d, e):
        if isinstance(e, Clear) and e.target == target:
            return {"attempts": 0, "streak": 0, "last": None, "cleared": True}
        if isinstance(e, Mint) and e.target == target:
            return {**d, "cleared": True}        # the memory IS the clear; the attempts that
        if not isinstance(e, Fail) or e.target != target:   # produced it stay visible
            return d
        if e.reason == "stale":
            return {**d, "last": e.reason}
        return {**d, "attempts": d["attempts"] + (e.reason != "provider_error"),
                "streak": d["streak"] + 1 if e.reason == "provider_error" else 0,
                "last": e.reason}
    return project(log, {"attempts": 0, "streak": 0, "last": None, "cleared": False}, count)


def last_reason(log, target):
    return ledger(log, target)["last"]


def attempts(log, target):
    """Burned attempts: every failure except a transient provider fault or a stale discard --
    neither is the unit's fault, so neither counts against the bound."""
    return ledger(log, target)["attempts"]


def streak(log, target):
    """Consecutive transient provider faults; a stale discard is neither a fault nor a success."""
    return ledger(log, target)["streak"]


def clear_debt(log, target, why="operator"):
    """Production's escape hatch as one event: a clear is a record in the log, not an edit to a
    structure -- so the pure version just returns the extended log."""
    return emit(log, "clear", target, why)


def cleared(log, target):
    """Both of production's clears as one predicate: the explicit `clear` event, and a later mint."""
    return ledger(log, target)["cleared"]


def stalled(log, target):
    """Terminal debt: this unit's failures crossed the bound and nothing has cleared them."""
    d = ledger(log, target)
    return not d["cleared"] and (d["attempts"] >= MAX_ATTEMPTS or d["streak"] >= PROVIDER_STREAK)


def policy(log, target, reason):
    """The whole failure policy: one table and one bounded rule, for L1 and merge alike."""
    if reason == "provider_error" and streak(log, target) < PROVIDER_STREAK:
        return "transient"                   # a blip is not the chunk's fault: no attempt burned
    return "terminal" if attempts(log, target) >= MAX_ATTEMPTS else ACTIONS[reason]


# ================== what may be compressed at all ==================
def chunk_spans(log):
    """Every chunk's span: its message range plus an ordinal separating the SHARDS of one message.
    A fold carrying the per-range ordinal counter alongside the result map."""
    def visit(state, span):
        out, seen = state
        cid, first, last = span[0], span[1], span[2]
        ordinal = seen.get((first, last), 0)
        return {**out, cid: Range(first, last, ordinal)}, \
            {**seen, (first, last): ordinal + 1}
    return reduce(visit, spans(log), ({}, {}))[0]


def l1_spans(log):
    """The spans each live L1 OWNS, stamped into the mint event itself: coverage is a fact recorded at mint
    time (production's sourceRange, :5238), not a re-derivation of chunk_spans(log[:i]) per mint -- the
    mint event already stores the derived text, so it stores the derived span too."""
    return {e.id: set(e.owned) for e in log
            if isinstance(e, Mint) and e.level == 1 and e.owned}


def meets(a, b):
    """Plain interval overlap: do two message ranges share lived messages?"""
    return a.first <= b.last and b.first <= a.last


def iv_overlap(a, b):
    """Do two chunk spans share lived messages? Equal message ranges are the SHARDS of one message, so the
    ordinal decides (different shards are different slices, not duplicates); different ranges overlap as
    intervals. The one predicate behind both arms of the coverage guard; protections use the plain
    half (`meets`)."""
    return a == b if (a.first, a.last) == (b.first, b.last) else meets(a, b)


def covered_by_l1(log, a=None):
    """Every chunk a live L1 stands for: `{chunkId: (summaryId, exact)}`, production's three arms as one."""
    a = archive(log) if a is None else a
    live, here = l1_spans(log), chunk_spans(log)
    out = {}
    for c in a.chunks:
        span = here[c.id]
        hits = sorted(sid for sid, owned in live.items()
                      if any(iv_overlap(span, other) for other in owned))
        if hits:
            out[c.id] = (hits[0], any(span in live[sid] for sid in hits))
    return out


def overlap_blocked(log, a=None):
    """The loud half: chunks whose messages a live L1 covers under DIFFERENT boundaries."""
    return sorted((cid, sid) for cid, (sid, exact) in covered_by_l1(log, a).items() if not exact)


def span_limit(level):
    """`spanBase * mergeK ** max(0, level - 3)`: a FLAT limit is what made an L5 structurally impossible."""
    return MERGE_MAX_SOURCE_SPAN_MESSAGES * cm.MERGE_THRESHOLD ** max(0, level - 3)


def merge_candidates(log, a=None):
    """Free (unmerged) summaries by level as [(id, first, last)] in range order; `[(id, reason)]` for each
    candidate the guard removed; and each level's LIVE END -- the newest message position any summary at
    that level reaches, merged or not, which is what separates a growing run from a stranded one.
    One fold over the summaries carrying (free, excluded, live_end), each REBUILT per event."""
    a = archive(log) if a is None else a
    here = chunk_spans(log)

    def visit(state, s):
        free, out, live_end = state
        owned = [here[l] for l in s.leaves if l in here]   # the candidate's span in MESSAGES, or
        if not owned or len(owned) != len(s.leaves):       # frontier debt when it no longer
            if s.id not in a.parent:                       # resolves: "can NEVER merge"
                out = out + [(s.id, "source position unresolved -- permanently unmergeable, "
                                    "frontier debt")]
            return free, out, live_end
        first, last = min(o.first for o in owned), max(o.last for o in owned)
        live_end = {**live_end, s.level: max(last, live_end.get(s.level, -1))}
        if s.id in a.parent:
            return free, out, live_end
        span = last - first
        if span > span_limit(s.level):
            return free, out + [(s.id, f"wide-span quarantine -- span {span} msgs > limit "
                                       f"{span_limit(s.level)} (base "
                                       f"{MERGE_MAX_SOURCE_SPAN_MESSAGES} x "
                                       f"{cm.MERGE_THRESHOLD}^{max(0, s.level - 3)})")], live_end
        cands = sorted(free.get(s.level, []) + [Cand(s.id, first, last)],
                       key=lambda c: (c.first, c.last))
        return {**free, s.level: cands}, out, live_end

    return reduce(visit, a.summaries.values(), ({}, [], {}))


def merge_exclusions(log, a=None):
    """The excluded half alone: what the demo prints and what the tests pin."""
    return merge_candidates(log, a)[1]


# ================== work is DERIVED, never stored ==================
def demanded_chunks(produced, a):
    """kv-stable hands back chunk-id RANGES; one range is one coalesced run of uncovered foldable chunks."""
    order = [c.id for c in a.chunks]
    pos = {cid: i for i, cid in enumerate(order)}
    ids = set()
    for first, last in produced:
        if first in pos and last in pos:
            ids.update(order[pos[first]:pos[last] + 1])
    return ids


def work(log, demanded=(), pressure=True):
    """Chunks with no L1 (minus the holdback window), plus merges per the odometer rule; the log's
    only ledger state is failures, so nothing is enqueued or quarantined.

    An L1 is derived only for a chunk the POLICY can fold that no live L1 already stands for -- a SPAN test,
    because an id test cannot see a re-derivation that moves chunk ids while the messages stay put
    (AUDIT.md §1.3). `demanded` opens the holdback; `pressure` is the lazy-production gate -- BACKPRESSURE,
    in the established sense: the planner (the consumer of memories) signals the producer, and a calm
    session mints nothing. Eager is the default and is production's behavior; our host passes False.
    The ordering of the returned items is a topological order of the mint DAG: chunks before merges,
    each level before the next, so the tick's drain can mint children and derive their parent in one pass."""
    a = archive(log)
    demanded = set(demanded)
    cut = max(0, len(a.chunks) - TAIL_HOLDBACK)
    zone = cm.raw_zone(a, protected_ids(log, a))
    covered = covered_by_l1(log, a)
    fresh = [c for i, c in enumerate(a.chunks) if (i < cut and pressure) or c.id in demanded]
    out = [Work(f"L1:{c.id}", "L1", c.id) for c in fresh
           if c.id not in zone and c.id not in covered]
    free, _excluded, live_end = merge_candidates(log, a)
    for level in sorted(free):
        # the odometer: merges fire on STRICTLY CONTIGUOUS runs of unmerged siblings (:6698), never
        # bridging a hole; a run that isn't at the level's live end can never grow (summaries are
        # produced at the live end), so it consolidates at 2 instead of 6 (:6718-6721)
        for run in cm.runs(free[level], lambda p, c: c.first <= p.last + 1):
            interior = run[-1].last < live_end.get(level, -1)
            if len(run) >= cm.MERGE_THRESHOLD:
                ids = [c.id for c in run[: cm.MERGE_THRESHOLD]]
            elif interior and len(run) >= 2:
                ids = [c.id for c in run]
            else:
                continue
            out.append(Work(f"L{level + 1}:{'+'.join(ids)}", "merge", "+".join(ids)))
    done = mints(log)
    return [i for i in out if i.target not in done and not stalled(log, i.target)]


class UncoveredDropError(Exception):
    """Production's fatal coverage error: a plan that drops part of the agent's own history is never emitted."""


def assert_coverage(a, F):
    """The invariant, on the EMITTED render: the units TILE the chunk list -- every chunk is raw, or inside
    a recall whose leaves cover it. The tiling DP maintains this by construction; this asserts it on what
    would actually be emitted (`assertFullCoverage`, :4655)."""
    covered, units = set(), cm.render(a, F)
    for name, _tokens in units:
        if name.startswith("raw:"):
            covered.add(name[4:])
        else:
            covered.update(a.summaries[name[7:]].leaves)
    dropped = [c.id for c in a.chunks if c.id not in covered]
    if dropped:
        raise UncoveredDropError(
            f"{len(dropped)} of {len(a.chunks)} chunks would leave the window: {dropped}; "
            f"emitted units {[name for name, _t in units]}")


# What a request is SHOWN, which is not what its work item names: `target` is the sources expanded
# one level deeper, `prefix` the content preceding them, `reading_tokens` the whole document's size
# for a doc-reading request, `retry` the refusal rung (sections 4-6 only).
Shown = namedtuple("Shown", "prefix target reading_tokens retry", defaults=("", "", None, False))


# `detectDocContext`'s sharded-message threshold: the whole must be 2x this portion, i.e. the
# portion is at most half the original message.
READING_CHUNK_MULTIPLE = 2


def reading_stretch(a, leaves, groups, multiple=None):
    """Reading mode (`detectDocContext`): EVERY leaf under the merge is a shard of one body group, so the
    stretch was the agent reading one long document and the request asks what that was like instead of
    forcing a consolidation. Returns the WHOLE group's size, or None; `multiple` is the L1 site's guard."""
    found = {groups.get(leaf) for leaf in leaves}
    if len(found) != 1:
        return None
    group = next(iter(found))
    if not group:                                     # an untagged chunk is not a reading stretch
        return None
    whole = sum(c.tokens for c in a.chunks if groups.get(c.id) == group)
    portion = sum(c.tokens for c in a.chunks if c.id in set(leaves))
    return whole if not multiple or whole >= multiple * portion else None


def source(log, kind, payload, retry=False):
    """The projection: what the request for this work item is SHOWN, in production's parts. BOTH requests are the
    six-section shape; only what each one's LEAVES are differs.

    PREFIX (1-3): head window raw, prior recall pairs (the unmerged frontier starting BEFORE this range, since
    children alongside their parent double the prompt unboundedly), then the raw middle, each element skipped
    when a live summary covers it. TARGET: for a merge the sources expanded ONE LEVEL DEEPER, for an L1 the
    chunk itself. Nothing after the chunk or the merge range is visible, and `retry` is production's
    `sourceOnly` arm. Section order: README.md `## summarizer.py`."""
    a = archive(log)
    groups = groups_of(log)
    order = {c.id: i for i, c in enumerate(a.chunks)}
    if kind == "L1":
        leaves, kids = (payload,), None
        level, start = 0, order[payload]
        doc = reading_stretch(a, leaves, groups, multiple=READING_CHUNK_MULTIPLE)   # section 6
    else:
        kids = payload.split("+")
        level = a.summaries[kids[0]].level
        leaves = tuple(l for k in kids for l in a.summaries[k].leaves)
        start = min((order[l] for l in leaves if l in order), default=len(order))
        doc = reading_stretch(a, leaves, groups)
    if retry:
        return Shown(target=a.chunks[start].text, retry=True, reading_tokens=doc)
    leaf_set = set(leaves)

    def uncovered(chunks):
        """Raw, in order: a chunk this request will not show deeper (not a leaf) and no summary covers."""
        return [c.text for c in chunks if c.id not in leaf_set and c.id not in a.l1_of]

    def first_index(s):
        known = [order[l] for l in s.leaves if l in order]
        return min(known) if known else None

    prior = [s for s in a.summaries.values()
             if s.id not in a.parent and not leaf_set.intersection(s.leaves)]
    prior = [(i, s) for s in prior if (i := first_index(s)) is not None and i < start]
    prior = [s for _i, s in sorted(prior)]
    prefix = uncovered(a.chunks[:cm.HEAD_CHUNKS])                 # head window, raw
    prefix += [recall_memory(s) for s in cap_recalls(prior)]      # prior recall pairs, capped
    prefix += uncovered(a.chunks[cm.HEAD_CHUNKS:start])           # raw middle (usually empty)

    # TARGET: one level deeper than the sources -- the chunk itself, for an L1.
    if level == 0:                                    # an L1's leaf is the chunk being compressed
        target = [a.chunks[start].text]
    elif level == 1:                                  # sources are L1s: show the raw L0 they cover
        target = [next(c.text for c in a.chunks if c.id == leaf) for leaf in leaves]
    else:                                             # sources are L2+: show the L_{n-2} under each
        target = [recall_memory(a.summaries[c]) for k in kids for c in a.children[k]]
    return Shown(prefix="\n\n".join(prefix), target="\n\n".join(target), reading_tokens=doc)


def recall_memory(s):
    """One recall pair as a request emits it: the `[CM] Recall memory <id>.` turn, then the body."""
    return RECALL_MEMORY_HEADER.format(id=s.id) + "\n" + s.text


def cap_recalls(pairs, budget=None):
    """`capRecallPairs`: a knapsack in recency order -- walk NEWEST-first, keeping each pair that still
    fits and `continue`-ing rather than breaking, so one oversized pair cannot hide smaller siblings
    behind it; the kept set goes back into chronological order. `budget=None` reads the configured one,
    so a test can patch it. A fold carrying (kept, total); the kept list is prepended, so the fold
    leaves it chronological already."""
    budget = RECALL_BUDGET if budget is None else budget

    def keep(state, s):
        kept, total = state
        cost = s.tokens + RECALL_PAIR_OVERHEAD
        if total + cost > budget:
            return state
        return [s] + kept, total + cost

    return reduce(keep, reversed(pairs), ([], 0))[0]


def groups_of(log):
    """The bodyGroupId of each chunk, over the same `spans`: a shard carries `g<seq>`, a whole message none."""
    return {cid: group for cid, _f, _l, _t, _x, _s, group in spans(log)}


def step(log, item, boundary, v):
    """One attempt: the boundary assembles the request, scoped to what this work item is shown, and gates the
    reply, so all this side sees is `(text | None, reason | None)`; one stale-version rule, nothing about
    providers. A refusal derives the source-only rung, as `overflow` derives the reshape.

    The pure signature: (log', boundary', outcome) = step(log, item, boundary, v). The original
    mutated `log` through `commit`/`emit` and the boundary through its __call__; here both ride
    the return value, and the CAS compares against the input log's length, as before."""
    target, kind, payload = item
    last = last_reason(log, target)
    shown = source(log, kind, payload, retry=kind == "L1" and last == "refusal")
    src = shown.target
    if last == "overflow":
        words = src.split()                       # the only reshape we allow: halve it
        src = " ".join(words[len(words) // 2:])   # one rule, not a ladder
    if last == "tool_call":
        src += "\n\n" + NO_TOOLS
    (text, reason), boundary = S.complete(boundary, kind, target, src, prefix=shown.prefix,
                                          reading_tokens=shown.reading_tokens,
                                          source_only=shown.retry)
    if text:     # the gate guarantees usable text; a reason comes back as None
        a = archive(log)
        here = chunk_spans(log)
        owned = lambda leaves: tuple(here[l] for l in leaves if l in here)
        if kind == "L1":
            fields = (target, f"L1-{payload}", 1, (payload,), text, owned((payload,)))
        else:
            kids = tuple(payload.split("+"))
            level = a.summaries[kids[0]].level + 1
            leaves = tuple(l for k in kids for l in a.summaries[k].leaves)
            fields = (target, f"L{level}-{kids[0]}", level, kids, text, owned(leaves))
        log, ok = commit(log, v, "mint", *fields)
        if ok:
            return log, boundary, "done"
    else:
        log, ok = commit(log, v, "fail", target, reason or "empty")
        if ok:                                 # the label matches the state this very attempt
            return log, boundary, policy(log, target, reason or "empty")  # produced. The CAS
    return emit(log, "fail", target, "stale"), boundary, "discarded-stale"  # failed: record the
                                               # system's own reason -- not in ACTIONS, burning
                                               # nothing -- and discard


def protected_ids(log, a):
    """One predicate for every protection type: a pin is a range with level bounds, and "range meets
    chunk" is plain interval overlap -- no ordinal rule: a pin over a sharded message covers every
    shard, which is exactly what the coverage guard's `iv_overlap` must NOT say."""
    here = chunk_spans(log)
    return {c.id for c in a.chunks
            for p in protections(log)
            if meets(p, here[c.id])}


def protections(log):
    return [e for e in log if isinstance(e, Protect)]


class Refused(Exception):
    """The enforced wall, W * (1 + grace). Carries the plan that crossed it and the demand it produced --
    and, in the functional port, the (log, model) the run had reached, because an exception is the one
    way a value cannot ride a return."""

    def __init__(self, message, plan=None, demand=(), log=None, model=None):
        super().__init__(message)
        self.plan, self.demand, self.log, self.model = plan, demand, log, model


if __name__ == "__main__":
    print("this module is the SYSTEM; the demo that drives it lives in demo.py")
