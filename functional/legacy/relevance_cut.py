"""The phase-based ideal cut (kv-control.ts:268 as phases): the solver the tiling DP replaced --
the functional port of ../../../legacy/relevance_cut.py.

Kept as the readable record of the A/B/C phase structure -- group-atomic folds under the shape
prior (A), past it to fit the window (B), then un-fold youngest-first to spend headroom (C) -- and
as the comparison baseline for any future solver change. Validated equivalent to the DP on the
documented runs and 300 fuzzed archives before being retired; see ../__init__.py.

The original threaded one dict `F` through both passes and mutated it leaf by leaf (with a
roll-back write in `pack`); here each pass returns the next frontier and `pack`'s rollback is
simply keeping the old value. Frontier maps are rebuilt with `{**F, ...}`.

    python3 -c "from legacy.relevance_cut import relevance_cut"   # importable, unused by default
"""
from connectome_min import ancestor_at, max_level, tokens


def fold_pass(a, F, caps, priority, stop_at, ignore_shape_caps):   # A: prior, B: past it
    for level in range(1, max_level(a) + 1):
        for c in priority:
            node = ancestor_at(a, c.id, level)
            if tokens(a, F) <= stop_at or node is None or F.get(c.id, 0) >= level: continue
            foldable = [l for l in node.leaves if caps[l] >= 0]   # protected leaves are excluded
            if not foldable or (not ignore_shape_caps and any(caps[l] < level for l in foldable)): continue
            F = {**F, **{leaf: level for leaf in foldable}}
    return F


def pack(a, F, caps, target, window):                # C: un-fold newest-first, accept if closer
    for level in range(max_level(a), 0, -1):
        for c in reversed(a.chunks):
            if tokens(a, F) >= target: return F
            node = ancestor_at(a, c.id, level)
            group = [l for l in node.leaves if caps[l] >= 0] if node else []
            if F.get(c.id, 0) != level or not group or any(F.get(l, 0) != level for l in group): continue
            before = tokens(a, F)
            candidate = {**F, **{leaf: level - 1 for leaf in group}}
            after = tokens(a, candidate)
            if after <= window and abs(after - target) < abs(before - target):
                F = candidate                         # accept; the "rollback" is not assigning
    return F


def relevance_cut(a, caps, target, window):          # the ideal cut: P is never consulted
    F = {c.id: 0 for c in a.chunks}
    priority = sorted(a.chunks, key=lambda c: (c.salience, c.seq))   # cheapest information first
    for _ in range(3):
        if tokens(a, F) > target: F = fold_pass(a, F, caps, priority, target, False)
        if tokens(a, F) > window: F = fold_pass(a, F, caps, priority, target, True)
        if window >= tokens(a, F) < target: F = pack(a, F, caps, target, window)
        if tokens(a, F) <= window: break
    return F
