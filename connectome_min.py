"""Connectome compaction, minimal model (@animalabs/context-manager 0.10.1).
    python3 connectome_min.py                # W=2300, P=W: bootstrap, hold, folds, overrides
    python3 connectome_min.py --reach 1250   # all five branches, incl. amortized repair
    python3 connectome_min.py --window 700   # fold floor above W -> escalation
Mirrors adaptive/kv-control.ts plus kv-stable's demand-side production; README.md maps code to source."""
import argparse, math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import takewhile
from config import (CHUNK_TOKENS, DEMO, GAP_RATIO, HEAD_CHUNKS, MAX_FOLD_LEVEL,
                    MERGE_THRESHOLD, SLACK, TAIL_TOKENS)   # config.py documents every knob
@dataclass
class Chunk: id: str; seq: int; text: str; tokens: int; salience: float = 1.0
@dataclass
class Summary: id: str; level: int; leaves: tuple; text: str; tokens: int

# ---- production: the append-only pyramid chunk -> L1 -> L2 (one model call per mint) ----
def summarize(text): return " ".join(text.split()[: max(40, len(text.split()) // 8)])
def max_level(a): return max((s.level for s in a["summaries"].values()), default=0)
def ancestor_at(a, cid, level):                      # the L(level) recollection covering cid:
    sid = a["l1_of"].get(cid)                        # a level-ancestor query by pointer climbing
    while sid and a["summaries"][sid].level < level: sid = a["parent"].get(sid)
    return a["summaries"][sid] if sid and a["summaries"][sid].level == level else None
def mint(a, sid, level, children):
    src = " ".join(a["summaries"][c].text if c in a["summaries"] else next(k.text for k in a["chunks"] if k.id == c) for c in children)
    leaves = tuple(l for c in children for l in (a["summaries"][c].leaves if c in a["summaries"] else (c,)))
    text = summarize(src)
    a["summaries"][sid] = Summary(sid, level, leaves, text, max(1, len(text.split())))
    if level == 1: a["l1_of"][children[0]] = sid
    else:
        for c in children: a["parent"][c] = sid      # setMergedInto; source keeps its text
def add_chunk(a, seq, sal=1.0):                      # chunk closes -> L1 -> merge a base run
    c = Chunk(f"c{len(a['chunks'])}", seq, "w " * CHUNK_TOKENS, CHUNK_TOKENS, sal)
    a["chunks"].append(c); mint(a, f"L1-{c.id}", 1, (c.id,))
    for level in range(1, max_level(a) + 1):
        run = [s.id for s in a["summaries"].values() if s.level == level and s.id not in a["parent"]]
        if len(run) >= MERGE_THRESHOLD: mint(a, f"L{level + 1}-{run[0]}", level + 1, tuple(run)); break

# ---- presentation: the live context is a projection of the archive ----
def render(a, F):                                    # the request the model sees, as units
    def unit(c):
        node = ancestor_at(a, c.id, F.get(c.id, 0)) if F.get(c.id, 0) > 0 else None
        return ("recall:" + node.id, node.tokens) if node else ("raw:" + c.id, c.tokens)
    return list(dict.fromkeys(unit(c) for c in a["chunks"]))   # stable dedupe: a node stands for
def tokens(a, F): return sum(t for _, t in render(a, F))       # its whole span, so it renders once
def kv_cost(prev, nxt):
    """Tokens re-read after the first divergence: the longest common PREFIX is free (the KV cache's
    whole premise), everything from the first different unit on is billed."""
    d = len(list(takewhile(lambda xy: xy[0][0] == xy[1][0], zip(prev, nxt))))
    return sum(t for _, t in nxt[d:])
def fold_depth_cap(chunk, now, raw_zone, threshold): # soft shape prior; -1 = hard protection
    if chunk.id in raw_zone: return -1
    ratio = max(0, now - chunk.seq) / max(1, len(raw_zone))
    # log base k of age: how many odometer digits this chunk's age buys it
    return 0 if ratio < 1 else max(0, min(MAX_FOLD_LEVEL, math.floor(math.log(ratio) / math.log(max(2, threshold))) + 1))
def runs(items, adjacent):
    """Maximal runs of a sequence, split wherever two neighbors fail `adjacent` -- run-length
    segmentation, the one loop behind both the demand path's coalescing and the merge odometer."""
    out = []
    for x in items:
        if out and adjacent(out[-1][-1], x):
            out[-1].append(x)
        else:
            out.append([x])
    return out
_VIOLATION = 1_000_000      # folding past a shape cap: phase-B territory, priced above any in-cap cut
def relevance_cut(a, caps, target, window):          # the ideal cut: P is never consulted
    """The ideal cut as a TILING of the timeline by pyramid nodes (kv-control.ts:268): each chunk raw, or
    covered by exactly one summary -- so group atomicity holds by construction and no repair pass
    (production's projectToValidCut) or fixpoint exists. Shortest path over chunk positions; edges are
    raw, or one node spanning its first..last foldable leaf, priced tokens + badness (salience x level,
    cap violations priced past any in-cap cut). Selection is lexicographic scalarization of that
    multiobjective path -- feasibility first, then distance to target, then badness -- mirroring the
    phase semantics it replaced: fit the window, prefer not violating caps, land near target. The
    phase-based solver this replaced is byte-identical on the documented runs and lives in
    legacy/relevance_cut.py."""
    chunks = a["chunks"]; n = len(chunks)
    pos = {c.id: i for i, c in enumerate(chunks)}
    edges = [[(i + 1, 0, c.tokens, 0.0, ((c.id, 0),))] for i, c in enumerate(chunks)]  # raw is always an option
    for s in a["summaries"].values():
        idx = sorted(pos[l] for l in s.leaves if l in pos)
        if len(idx) != len(s.leaves):
            continue                                   # a tiling needs every leaf to resolve
        # A node folds its UNPROTECTED leaves; protected ones punch holes and render raw (the
        # carve-out the legacy phase solver made with `foldable`). One edge spans first..last
        # foldable leaf, carrying the recall once plus the raw cost of the holes, so the group
        # stays atomic.
        foldable = [j for j in idx if caps[chunks[j].id] >= 0]
        if not foldable:
            continue                                   # every leaf protected: nothing to fold
        first, last = foldable[0], foldable[-1]
        assigns = tuple((chunks[j].id, 0 if caps[chunks[j].id] < 0 else s.level)
                        for j in range(first, last + 1))
        etok = s.tokens + sum(chunks[j].tokens for j in range(first, last + 1)
                              if caps[chunks[j].id] < 0)
        bad = sum((_VIOLATION if s.level > caps[chunks[j].id] else 0) +
                  chunks[j].salience * s.level * (1 + 0.01 * j / n)   # taste: fold older first
                  for j in foldable)
        edges[first].append((last + 1, s.level, etok, bad, assigns))
    dp = [[] for _ in range(n + 1)]                    # dp[i]: Pareto frontier of (tokens, badness, F) --
    dp[0] = [(0, 0.0, {})]                             # label-setting for a multicriteria shortest
    for i in range(n):                                 # path: extend every label, prune the dominated
        for tok, bad, F in dp[i]:
            for j, level, etok, ebad, assigns in edges[i]:
                cand = (tok + etok, bad + ebad, {**F, **dict(assigns)})
                if any(t <= cand[0] and b <= cand[1] for t, b, _ in dp[j]): continue   # dominated
                dp[j] = [(t, b, f) for t, b, f in dp[j] if not (cand[0] <= t and cand[1] <= b)]
                dp[j].append(cand)
    final = dp[n]
    feasible = [s for s in final if s[0] <= window]
    if not feasible:                                   # escalated: fold floor above W
        return min(final, key=lambda s: (s[0], s[1]))[2]
    return min(feasible, key=lambda s: (s[1] >= _VIOLATION, abs(s[0] - target), s[1]))[2]
def suffix_adopt(a, carried, carried_units, ideal, P):
    """Adopt the ideal's NEWEST changes only. Perturbation is prefix-based (kv_cost), so a later
    adoption boundary is monotonically cheaper -- the cheapest affordable partial is a bisect
    over the boundary, not a search."""
    changed = [c.seq for c in a["chunks"] if carried.get(c.id, 0) != ideal.get(c.id, 0)]
    bounds = changed + [math.inf]             # inf = adopt nothing: cost 0, always feasible
    build = lambda b: {c.id: (ideal.get(c.id, 0) if c.seq >= b else carried.get(c.id, 0))
                       for c in a["chunks"]}
    affordable = lambda i: kv_cost(carried_units, render(a, build(bounds[i]))) <= P
    best = build(bounds[bisect_left(range(len(bounds)), True, key=affordable)])
    return best, kv_cost(carried_units, render(a, best))
def demand_runs(a, zone, escalated):                 # kv-stable's demand-side production (:165-198)
    """An ESCALATED plan -- even the IDEAL cut is over W -- carries one request per CONTIGUOUS run of uncovered
    foldable chunks, coalesced as kv-stable does; `work` lets them bypass the holdback."""
    if not escalated: return []
    eligible = lambda c: c.id not in a["l1_of"] and c.id not in zone
    return [(seg[0].id, seg[-1].id)
            for seg in runs(sorted(a["chunks"], key=lambda c: c.seq),
                            lambda p, c: eligible(p) and eligible(c))
            if eligible(seg[0])]
def raw_zone(a, protected=()):                       # head window, tail window and every pin
    """The chunks neither the policy nor the derivation may touch: production never even chunks them
    (`getCompressibleMessages`), so `work` applies the same boundary to the DERIVATION. Shared on purpose."""
    zone, used = {c.id for c in a["chunks"][:HEAD_CHUNKS]} | set(protected), 0
    for c in reversed(a["chunks"]):
        zone.add(c.id); used += c.tokens
        if used >= TAIL_TOKENS: break
    return zone
def plan_controlled_frontier(a, prev, window, P, protected=()):   # W the only wall; P soft
    target = window - int(window * SLACK)
    zone = raw_zone(a, protected)
    caps = {c.id: fold_depth_cap(c, a["chunks"][-1].seq, zone, MERGE_THRESHOLD) for c in a["chunks"]}
    carried = {c.id: max(0, -1 if c.id in zone else prev.get(c.id, 0)) for c in a["chunks"]}
    carried_units, carried_tokens = render(a, carried), tokens(a, carried)
    loss = lambda F: sum(c.salience * max(0, F.get(c.id, 0) - caps[c.id])   # misallocation, not
                         for c in a["chunks"] if caps[c.id] >= 0)           # information loss
    ideal = relevance_cut(a, caps, target, window)
    escalated = tokens(a, ideal) > window            # kv-control.ts:925 `ideal.tokens > windowTokens`
    out = lambda F, br, pt, ov=None: dict(F=F, tokens=tokens(a, F), branch=br, perturbation=pt,
                                          override=ov, produced=demand_runs(a, zone, escalated))
    memo = {}                                        # guards and actions share computations
    def lazily(key, thunk):
        if key not in memo:
            memo[key] = thunk()
        return memo[key]
    gap = lambda: lazily("gap", lambda: GAP_RATIO * max(1.0, loss(ideal)))  # infinite under strictReach
    pert = lambda: lazily("pert", lambda: kv_cost(carried_units, render(a, ideal)))
    def suffix():                                    # (frontier, its perturbation, its tokens)
        if "suffix" not in memo:
            partial, ppert = suffix_adopt(a, carried, carried_units, ideal, P)
            memo["suffix"] = (partial, ppert, tokens(a, partial))
        return memo["suffix"]
    rules = (            # the cascade, first match wins: each rule is (name, guard, action)
        ("bootstrap",                                # 1. nothing carried
         lambda: not prev,
         lambda: out(ideal, "bootstrap", 0, "bootstrap")),
        ("hold",                                     # 2. dead band: inside [target, W], near ideal
         lambda: target <= carried_tokens <= window and loss(carried) - loss(ideal) <= gap(),
         lambda: out(carried, "hold", 0)),
        ("adopt-ideal",                              # 3. the whole move fits the trust region
         lambda: pert() <= P,
         lambda: out(ideal, "adopt-ideal", pert())),
        ("suffix-adopt",                             # 4. else the newest changes only, amortized
         lambda: suffix()[2] <= window and loss(suffix()[0]) - loss(ideal) <= gap()
                 and (carried_tokens <= window or suffix()[2] <= window or suffix()[1] > 0),
         lambda: out(suffix()[0], "suffix-adopt", suffix()[1])),
        ("override",                                 # 5. the trust region loses: the ideal, regardless
         lambda: True,
         lambda: out(ideal, "override", pert(),
                     "infeasible" if suffix()[2] > window else "quality-gap")),
    )
    return next(action() for _name, guard, action in rules if guard())
def replay(window, reach):
    a = {"chunks": [], "summaries": {}, "parent": {}, "l1_of": {}}
    prev, P = {}, (reach or window)
    print(f"W={window} target={window - int(window * SLACK)} P={P} base={MERGE_THRESHOLD} chunk={CHUNK_TOKENS}tok")
    print(f"{'turn':>4} {'chunks':>6} {'Lmax':>4} {'plan':>12} {'tokens':>7} {'dP':>6}  note")
    for t in range(12):
        add_chunk(a, a["chunks"][-1].seq + 4 if a["chunks"] else 4, 0.4 if t % 3 else 1.0)
        p = plan_controlled_frontier(a, prev, window, P); prev = p["F"]
        note = p["override"] or ("escalated: over W" if p["tokens"] > window else "")
        print(f"{t:>4} {len(a['chunks']):>6} {max_level(a):>4} {p['branch']:>12} {p['tokens']:>7} {p['perturbation']:>6}  {note}")
    prof = {k: sum(1 for c in a["chunks"] if prev.get(c.id, 0) == k) for k in set(prev.values())}
    print("final profile (L0 = raw): " + " ".join(f"L{k}:{v}" for k, v in sorted(prof.items())))
    print("rendered: " + " | ".join(k for k, _ in render(a, prev))[:120] + " ...")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=DEMO["window"])
    ap.add_argument("--reach", type=int, default=0, help="P, the trust region (default W)")
    x = ap.parse_args(); replay(x.window, x.reach)