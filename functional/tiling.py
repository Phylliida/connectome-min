"""The ideal cut as a TILING of the timeline by pyramid nodes -- the functional port of
../tiling.py. Same solver, same selection, same output: plain data in, one dict out, and the
caller owns every structure it handed in afterwards (nothing shared is mutated).

The one concession to the machine: the DP tables are built as FRESH local dicts, updated in
place inside this call and never aliased outside it. A persistent-map version is O(states^2) in
CPython (no structural sharing for dicts) and minutes-slow at session scale; local scratch keeps
the function referentially transparent without that. This is the folder's one internal-
mutation site, which is why it is documented rather than `_`-prefixed: it mutates no state any
caller can observe.

_VIOLATION prices folding past a shape cap above any in-cap cut. Edges are raw, or one node
spanning its first..last foldable leaf, priced tokens + badness (salience x level). Selection is
lexicographic: fit the window, prefer not violating caps, land near target."""

_VIOLATION = 1_000_000      # folding past a shape cap: priced above any in-cap cut


def cut(chunks, summaries, caps, target, window):
    """chunks: [(id, tokens, salience)] in timeline order; summaries: [(level, leaves, tokens)]
    with leaves chunk ids; caps: {chunk id: max fold level, -1 = protected}. Returns
    {chunk id: level} -- the ideal cut. The trust region is the caller's business."""
    n = len(chunks)
    ids = [c[0] for c in chunks]                   # transpose to columns up front: every loop below
    toks = [c[1] for c in chunks]                  # reads positionally -- toks[j], cap[j] -- with no
    sals = [c[2] for c in chunks]                  # chunks[j][1]-style indexing
    cap = [caps[cid] for cid in ids]
    pos = {cid: i for i, cid in enumerate(ids)}
    edges = [[(i + 1, toks[i], 0.0, ((ids[i], 0),))] for i in range(n)]   # raw is always an option
    for level, leaves, stokens in summaries:                             # these are the folds
        idx = sorted(pos[l] for l in leaves if l in pos)
        if len(idx) != len(leaves):
            continue                                   # a tiling needs every leaf to resolve
        # A node folds its UNPROTECTED leaves; protected ones punch holes and render raw. One
        # edge spans first..last foldable leaf, carrying the recall once plus the raw cost of
        # the holes, so the group stays atomic.
        foldable = [j for j in idx if cap[j] >= 0]
        if not foldable:
            continue                                   # every leaf protected: nothing to fold
        first, last = foldable[0], foldable[-1]
        assigns = tuple((ids[j], 0 if cap[j] < 0 else level) for j in range(first, last + 1))
        etok = stokens + sum(toks[j] for j in range(first, last + 1) if cap[j] < 0)
        bad = sum((_VIOLATION if level > cap[j] else 0) +
                  sals[j] * level * (1 + 0.01 * j / n)        # taste: fold older first
                  for j in foldable)
        edges[first].append((last + 1, etok, bad, assigns))
    dp = [{0: (0.0, {})}] + [{} for _ in range(n)]   # dp[i][tokens] = (min badness, F) -- fresh
    for i in range(n):                               # local tables, updated in place (see module
        for t, (bad, F) in dp[i].items():            # docstring); F itself is REBUILT, never
            for j, etok, ebad, assigns in edges[i]:  # mutated, so a stored cut stays immutable
                nt, nb = t + etok, bad + ebad
                if nt not in dp[j] or nb < dp[j][nt][0]:
                    dp[j][nt] = (nb, {**F, **dict(assigns)})
    final = dp[n]
    feasible = [t for t in final if t <= window]
    if not feasible:                                   # escalated: fold floor above the window
        return final[min(final)][1]
    return final[min(feasible, key=lambda t: (final[t][0] >= _VIOLATION,
                                              abs(t - target), final[t][0]))][1]
