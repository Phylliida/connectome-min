"""The kv-stable fold planner of @animalabs/context-manager 0.10.1 (KvStableStrategy.solve,
planControlledFrontier and the picker's accounting), reproduced exactly in its default mode.

Each compile picks a level for every message between the head and tail windows: 0 renders it raw,
k renders its L_k ancestor's recall pair, once for all the messages that share it.
  1. The ideal cut, from scratch: fold whole groups in (salience, age) order, level by level,
     first no deeper than each message's log-age band, then deeper if the window W demands it;
     then unfold youngest-first while that brings the total closer to the target; again, up to three
     rounds in all, while the render still exceeds W (a negative head or tail can leave it there).
  2. Hold the carried frontier if its total is in [target, W] and its misallocation is not much
     worse than the ideal's.
  3. Otherwise adopt the ideal if switching to it re-reads at most P tokens (kv_cost).
  4. Otherwise adopt the ideal's newest changes only: the longest suffix costing at most P.
  5. Otherwise adopt the ideal anyway.

Default mode: no pins or locked messages and no prepared-window transition. Summaries must form
the library's nested pyramid: each L_k merges whole L_(k-1)s, so a message's ancestors have levels
1, 2, 3, ... The ideal cut then only ever moves whole groups and stays group-consistent, so the
library's projection, overlap tolerance and unfold eligibility never act on it;
the bootstrap branch needs an empty log. Carried frontiers and suffix candidates can cut through a
group (a message returning from the tail with a stale level, a boundary inside a group), and are
projected.

The cut operations this policy uses (what a cut renders and costs, projection) are cut.py's.
"""
import math
from bisect import bisect_left
from functools import cache, reduce
from itertools import groupby, product
from operator import add
from typing import NamedTuple

import chunking
from config import Config
from cut import Inputs, Leaf, Summary, kv_cost, size
from estimate import Priced


class Plan(NamedTuple):
    levels: dict                          # message id -> level, for every leaf
    tokens: float                         # head + middle + tail (the picker's finalTokens)
    perturbation: float                   # tokens the provider re-reads against the carried frontier
    branch: str                           # hold | ideal | suffix | infeasible | quality-gap
    escalated: bool                       # even the ideal exceeds W
    produced: list                        # when escalated: (first, last) id of each run of leaves with no L1
    units: list                           # the middle as it renders, in order: (('raw', seq) | ('recall', summary id), tokens)


def plan(view: Inputs, W: float, cfg: Config) -> Plan:
    leaves, target, P = view.leaves, W * (1 - cfg.slack), W if cfg.reach_tokens is None else cfg.reach_tokens
    cap = {leaf.id: band(view, leaf, cfg.merge_threshold) for leaf in leaves}
    adopt = lambda F, pert, branch, escalated, produced: Plan(F, view.total(F), pert, branch, escalated, produced, view.units(F))

    # 1. The ideal cut.
    ideal = ideal_cut(view, W, target, cap)
    escalated = size(view.layout(ideal)) > W
    produced = [(run[0].id, run[-1].id) for run in
                (list(run) for uncovered, run in groupby(leaves, lambda m: not m.l1) if uncovered and escalated)]

    # 2. Hold the carried frontier.
    carried = view.project({leaf.id: leaf.level for leaf in leaves})
    carried_units, ideal_loss = view.layout(carried), loss(view, ideal, cap)
    gap = cfg.quality_gap_ratio * max(1, ideal_loss)
    if target <= size(carried_units) <= W and loss(view, carried, cap) - ideal_loss <= gap:
        return adopt(carried, 0, 'hold', False, [])

    # 3. Adopt the ideal within the trust region.
    if (pert := kv_cost(carried_units, view.layout(ideal))) <= P:
        return adopt(ideal, pert, 'ideal', escalated, produced)

    # 4. Adopt the ideal from a boundary on: the oldest of its changes the trust region affords,
    #    found by the library's binary search (len(changes) adopts nothing).
    changes = [leaf.seq for leaf in leaves if carried[leaf.id] != ideal[leaf.id]]

    @cache
    def suffix(i: int) -> tuple:
        F = view.project({m.id: ideal[m.id] if i < len(changes) and m.seq >= changes[i] else carried[m.id] for m in leaves})
        return F, (units := view.layout(F)), kv_cost(carried_units, units)

    F, units, partial_pert = suffix(bisect_left(range(len(changes)), True, key=lambda i: suffix(i)[2] <= P))
    if (partial := size(units)) <= W and loss(view, F, cap) - ideal_loss <= gap:
        return adopt(F, partial_pert, 'suffix', False, [])

    # 5. The trust region loses.
    return adopt(ideal, pert, 'infeasible' if partial > W else 'quality-gap', escalated, produced)


def ideal_cut(view: Inputs, W: float, target: float, cap: dict) -> dict:
    """Step 1, in rounds: the ledger (view.total) steers the phases, but a round ends on the render,
    which leaves out a negative window; while that exceeds W and shrinks, another round, three at most.
      A  past the target: fold whole groups, level by level in (salience, age) order, within their bands.
      B  still past W: the same, past the bands.
      C  under the target: unfold each group at a level once, youngest-first, if that nears the target
         and stays within W. Unfolding a group moves the total by a fixed amount and only other groups'
         unfolds came before, so the library's retries of a refused group after each accept always refuse."""
    leaves, at, members, total = view.leaves, view.at, view.members, view.total
    ideal, rendered = dict.fromkeys((leaf.id for leaf in leaves), 0), math.inf
    priority = sorted(leaves, key=lambda m: (m.salience, m.seq))
    for _ in range(3):
        tokens = total(ideal)
        for within_bands, limit in ((True, target), (False, W)):    # phase A, and phase B only past W
            if tokens > limit:
                for level, leaf in product(range(1, view.top + 1), priority):
                    node = at(leaf, level) if tokens > target and ideal[leaf.id] < level else None
                    if node and (not within_bands or min(cap[m] for m in members(node.id)) >= level):
                        ideal |= dict.fromkeys(members(node.id), level)
                        tokens = total(ideal)
        for level in range(view.top, 0, -1):                          # phase C
            for node in dict.fromkeys(at(m, level) for m in reversed(leaves) if ideal[m.id] == level):
                if tokens < target:
                    trial = ideal | dict.fromkeys(members(node.id), level - 1)
                    if (t := total(trial)) <= W and abs(t - target) < abs(tokens - target):
                        ideal, tokens = trial, t
        if (now := size(view.layout(ideal))) <= W or now >= rendered:
            break
        rendered = now
    return ideal


def band(view: Inputs, leaf: Leaf, k: int) -> int:
    """foldDepthCap: how deep the message's age allows, counted in base-k digits. (The library caps it
    at the top level, which no fold passes anyway.)"""
    ratio = (len(view.prices) - 1 - leaf.seq) / max(1, len(view.prices) - len(view.leaves))   # age / zone size
    return 0 if ratio < 1 else math.floor(math.log(ratio) / math.log(k)) + 1


def loss(view: Inputs, F: dict, cap: dict) -> float:
    """Misallocation: salience-weighted depth past each band, added oldest first."""
    return reduce(add, (m.salience * (F[m.id] - cap[m.id]) for m in view.leaves if F[m.id] > cap[m.id]), 0)


def ceiling(W: float, cfg: Config) -> int:
    """The most a compile may render, W plus the grace: OverBudgetError's budget."""
    return math.floor(W * (1 + cfg.over_budget_grace))


def inputs(messages: list[Priced], summaries: list[Summary], levels: dict, multiplier: float,
           cfg: Config) -> Inputs:
    """The planner's view of the log (selectAdaptive): the messages between the windows at their
    post-strip prices, each with the first L1 that lists it and its carried level."""
    head, tail, prices = chunking.windows(messages, multiplier, cfg)    # a tail reaching into the head starts after it
    l1 = {m: s.id for s in summaries if s.level == 1 for m in s.sources}
    leaves = [Leaf(m.id, i, prices[i], levels.get(m.id, 0), l1.get(m.id), m.salience) for i, m in enumerate(messages[head:tail], head)]
    return Inputs(leaves, summaries, head, prices)
