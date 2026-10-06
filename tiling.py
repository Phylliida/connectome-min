"""The ideal cut as a TILING of the timeline by pyramid nodes (kv-control.ts:268 in the production
system this models): each chunk stays raw, or is covered by exactly one summary -- so group
atomicity holds by construction and no repair pass (production's projectToValidCut) or fixpoint
exists.

Self-contained: plain data in, one dict out. One table, dp[position][tokens] = the minimum-badness
tiling at exactly that token count -- per exact token count there is nothing to dominate with, so
no Pareto frontier and no dominance pruning. Edges are raw, or one node spanning its first..last
foldable leaf, priced tokens + badness (salience x level, cap violations priced past any in-cap
cut). Selection is lexicographic: fit the window, prefer not violating caps, land near target."""

_VIOLATION = 1_000_000      # folding past a shape cap: priced above any in-cap cut


def cut(chunks, summaries, caps, target, window):
    """chunks: [(id, tokens, salience)] in timeline order; summaries: [(level, leaves, tokens)]
    with leaves chunk ids; caps: {chunk id: max fold level, -1 = protected}. Returns
    {chunk id: level} -- the ideal cut. The trust region is the caller's business."""
    n = len(chunks)
    pos = {cid: i for i, (cid, _t, _s) in enumerate(chunks)}
    edges = [[(i + 1, tokens, 0.0, ((cid, 0),))] for i, (cid, tokens, _s) in enumerate(chunks)]
    for level, leaves, stokens in summaries:         # raw is always an option; these are the folds
        idx = sorted(pos[l] for l in leaves if l in pos)
        if len(idx) != len(leaves):
            continue                                   # a tiling needs every leaf to resolve
        # A node folds its UNPROTECTED leaves; protected ones punch holes and render raw. One
        # edge spans first..last foldable leaf, carrying the recall once plus the raw cost of
        # the holes, so the group stays atomic.
        foldable = [j for j in idx if caps[chunks[j][0]] >= 0]
        if not foldable:
            continue                                   # every leaf protected: nothing to fold
        first, last = foldable[0], foldable[-1]
        assigns = tuple((chunks[j][0], 0 if caps[chunks[j][0]] < 0 else level)
                        for j in range(first, last + 1))
        etok = stokens + sum(chunks[j][1] for j in range(first, last + 1)
                             if caps[chunks[j][0]] < 0)
        bad = sum((_VIOLATION if level > caps[chunks[j][0]] else 0) +
                  chunks[j][2] * level * (1 + 0.01 * j / n)   # taste: fold older first
                  for j in foldable)
        edges[first].append((last + 1, etok, bad, assigns))
    dp = [{0: (0.0, {})}] + [{} for _ in range(n)]   # dp[i][tokens] = (min badness, F)
    for i in range(n):
        for t, (bad, F) in dp[i].items():
            for j, etok, ebad, assigns in edges[i]:
                nt, nb = t + etok, bad + ebad
                if nt not in dp[j] or nb < dp[j][nt][0]:
                    dp[j][nt] = (nb, {**F, **dict(assigns)})
    final = dp[n]
    feasible = [t for t in final if t <= window]
    if not feasible:                                   # escalated: fold floor above the window
        return final[min(final)][1]
    return final[min(feasible, key=lambda t: (final[t][0] >= _VIOLATION,
                                              abs(t - target), final[t][0]))][1]
