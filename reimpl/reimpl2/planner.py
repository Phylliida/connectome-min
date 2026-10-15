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

This module is the policy: the targets, the age bands and misallocation that judge a cut, the order
and rounds of the folds, and the names of the outcomes. The search is optimize.py's (coarsen, refine,
adopt), over cut.py's operations.
"""
import math
from functools import reduce
from itertools import groupby
from operator import add
from typing import NamedTuple

import chunking
from config import Config
from cut import Inputs, Leaf, Summary, size
from estimate import Priced
from optimize import adopt, coarsen, refine


class Plan(NamedTuple):
    levels: dict                          # message id -> level, for every leaf
    tokens: float                         # head + middle + tail (the picker's finalTokens)
    perturbation: float                   # tokens the provider re-reads against the carried frontier
    branch: str                           # hold | ideal | suffix | infeasible | quality-gap
    escalated: bool                       # even the ideal exceeds W
    produced: list                        # when escalated: (first, last) id of each run of leaves with no L1
    units: list                           # the middle as it renders, in order: (('raw', seq) | ('recall', summary id), tokens)


def plan(view: Inputs, W: float, cfg: Config) -> Plan:
    target, P = W * (1 - cfg.slack), W if cfg.reach_tokens is None else cfg.reach_tokens
    cap = {leaf.id: band(view, leaf, cfg.merge_threshold) for leaf in view.leaves}
    ideal = ideal_cut(view, W, target, cap)
    escalated = size(view.layout(ideal)) > W
    produced = [(run[0].id, run[-1].id) for run in
                (list(run) for uncovered, run in groupby(view.leaves, lambda m: not m.l1) if uncovered and escalated)]

    ideal_loss = loss(view, ideal, cap)
    near = lambda F: loss(view, F, cap) - ideal_loss <= cfg.quality_gap_ratio * max(1, ideal_loss)   # not much worse
    step = adopt(view, view.project({leaf.id: leaf.level for leaf in view.leaves}), ideal, P,
                 holds=lambda F, units: target <= size(units) <= W and near(F), fits=lambda F, units: size(units) <= W and near(F))
    planned = lambda branch, escalated, produced: Plan(step.cut, view.total(step.cut), step.cost, branch, escalated, produced,
                                                       view.units(step.cut))
    match step.name:
        case 'hold' | 'suffix':
            return planned(step.name, False, [])
        case 'ideal':
            return planned('ideal', escalated, produced)
        case 'neither':
            return planned('infeasible' if size(step.tried) > W else 'quality-gap', escalated, produced)


def ideal_cut(view: Inputs, W: float, target: float, cap: dict) -> dict:
    """Step 1, in rounds: the ledger (view.total) steers the phases, but a round ends on the render,
    which leaves out a negative window; while that exceeds W and shrinks, another round, three at most.
      A  past the target: fold in (salience, age) order, within the bands, while past the target.
      B  still past W: the same, past the bands.
      C  under the target: unfold where that nears the target within W. Unfolding a group moves the
         total by a fixed amount and only other groups' unfolds came before, so the library's retries
         of a refused group after each accept always refuse; refine tries each group once."""
    ledger, priority = view.total, sorted(view.leaves, key=lambda m: (m.salience, m.seq))
    in_band = lambda group, level: min(cap[m] for m in view.members(group.id)) >= level
    ideal, rendered = dict.fromkeys((leaf.id for leaf in view.leaves), 0), math.inf
    for _ in range(3):
        ideal = coarsen(view, ideal, ledger, priority, lambda t: t > target, in_band)
        if ledger(ideal) > W:
            ideal = coarsen(view, ideal, ledger, priority, lambda t: t > target, lambda group, level: True)
        ideal = refine(view, ideal, ledger, lambda t: t < target, lambda t, now: t <= W and abs(t - target) < abs(now - target))
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
