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
"""
import math
from bisect import bisect_left
from functools import cache, reduce
from itertools import groupby, product, takewhile
from operator import add
from typing import Callable, NamedTuple

import chunking
from config import Config
from estimate import Priced, js_len


class Leaf(NamedTuple):
    """A message between the windows, as the planner sees it."""
    id: object
    seq: int                              # position in the log
    tokens: float                         # post-strip store price
    level: int                            # carried resolution
    l1: str | None                        # the L1 that lists it
    salience: float


class Summary(NamedTuple):
    id: str
    level: int
    tokens: float                         # its recall pair's price (recall_tokens)
    sources: tuple                        # message ids for an L1, child summary ids above
    parent: str | None = None


class Inputs(NamedTuple):
    leaves: list                          # Leaf, in log order: the messages between the windows
    summaries: list                       # Summary
    head: int                             # messages in the head window; the leaves follow them
    prices: list                          # every message's post-strip store price, in log order

    @property
    def head_tokens(self) -> float:
        return sum(self.prices[:self.head])

    @property
    def tail_tokens(self) -> float:
        return sum(self.prices[self.head + len(self.leaves):])


class Plan(NamedTuple):
    levels: dict                          # message id -> level, for every leaf
    tokens: float                         # head + middle + tail (the picker's finalTokens)
    perturbation: float                   # tokens the provider re-reads against the carried frontier
    branch: str                           # hold | ideal | suffix | infeasible | quality-gap
    escalated: bool                       # even the ideal exceeds W
    produced: list                        # when escalated: (first, last) id of each run of leaves with no L1


def size(units: list) -> float:
    return sum(tokens for _, tokens in units)


def kv_cost(before: list, after: list) -> float:
    """Tokens of `after` re-read from its first unit that differs from `before`'s."""
    return size(after[sum(1 for _ in takewhile(lambda pair: pair[0][0] == pair[1][0], zip(before, after))):])


def ancestry(summaries: list[Summary]) -> Callable[[Leaf, int], Summary | None]:
    """The summary that renders a leaf at a level, if any (SummaryTree.ancestorAt): its L1's chain
    of parents, read as a table of level -> summary."""
    nodes = {s.id: s for s in summaries}

    @cache
    def lineage(sid) -> dict:
        return {**lineage(s.parent), s.level: s} if (s := nodes.get(sid)) else {}

    return lambda leaf, level: lineage(leaf.l1).get(level)


def plan(inputs: Inputs, W: float, cfg: Config) -> Plan:
    leaves, nodes = inputs.leaves, {s.id: s for s in inputs.summaries}
    middle = {leaf.id for leaf in leaves}
    target, P = W * (1 - cfg.slack), W if cfg.reach_tokens is None else cfg.reach_tokens
    top, k = max((s.level for s in inputs.summaries), default=0), cfg.merge_threshold
    at, head_tokens, tail_tokens = ancestry(inputs.summaries), inputs.head_tokens, inputs.tail_tokens

    @cache
    def members(sid) -> tuple:
        """A summary's leaves; its head and tail messages render raw beside its recall."""
        s = nodes[sid]
        return tuple(m for m in s.sources if m in middle) if s.level == 1 else \
            tuple(m for child in s.sources for m in members(child))

    def middle_units(F: dict) -> list:
        """The middle's rendered units in order, as (identity, tokens): siblings share one recall."""
        unit = lambda m, node: (('raw', m.id), m.tokens) if node is None else (('recall', node.id), node.tokens)
        return list(dict.fromkeys(unit(m, at(m, F[m.id])) for m in leaves))

    def layout(F: dict) -> list:
        """renderLayout, which kv_cost and the hold, suffix and escalation tests measure: the head
        and tail join only when their totals are positive (stripped images can make them negative)."""
        return [('head', head_tokens)] * (head_tokens > 0) + middle_units(F) + [('tail', tail_tokens)] * (tail_tokens > 0)

    def accounted(F: dict) -> float:
        """The token ledger the fold phases steer by, and the picker's reported total: head and tail
        always count."""
        return head_tokens + tail_tokens + size(middle_units(F))

    def band(leaf: Leaf) -> int:
        """foldDepthCap: how deep the message's age allows, counted in base-k digits. (The library caps
        it at the top level, which no fold passes anyway.)"""
        ratio = (len(inputs.prices) - 1 - leaf.seq) / max(1, len(inputs.prices) - len(leaves))   # age / zone size
        return 0 if ratio < 1 else math.floor(math.log(ratio) / math.log(k)) + 1

    cap = {leaf.id: band(leaf) for leaf in leaves}

    def loss(F: dict) -> float:
        """Misallocation: salience-weighted depth past each band, added oldest first."""
        return reduce(add, (m.salience * (F[m.id] - cap[m.id]) for m in leaves if F[m.id] > cap[m.id]), 0)

    def project(F: dict) -> dict:
        """projectToValidCut: each pass lowers by one, in place, every leaf whose group disagrees
        about its level, deepest first, until a pass changes nothing. In place matters: a group a
        deeper member just joined can agree by the time its shallower members are judged."""
        changed = True
        while changed:
            changed = False
            for leaf in sorted(leaves, key=lambda m: -F[m.id]):
                level = F[leaf.id]
                if level > 0 and not ((node := at(leaf, level)) and all(F[m] == level for m in members(node.id))):
                    F[leaf.id], changed = level - 1, True
        return F

    # 1. The ideal cut, in rounds: the ledger steers the phases, but a round ends on the render, which
    #    leaves out a negative window. While that exceeds W and shrinks, another round, three at most.
    ideal, rendered = dict.fromkeys(middle, 0), math.inf
    priority = sorted(leaves, key=lambda m: (m.salience, m.seq))
    for _ in range(3):
        tokens = accounted(ideal)
        for within_bands, limit in ((True, target), (False, W)):    # phase A, and phase B only past W
            if tokens > limit:
                for level, leaf in product(range(1, top + 1), priority):
                    node = at(leaf, level) if tokens > target and ideal[leaf.id] < level else None
                    if node and (not within_bands or min(cap[m] for m in members(node.id)) >= level):
                        ideal |= dict.fromkeys(members(node.id), level)
                        tokens = accounted(ideal)
        # Phase C, when the total is under W: unfold each group at a level once, youngest-first, if that
        # nears the target. Unfolding a group moves the total by a fixed amount and only other groups'
        # unfolds came before, so the library's retries of a refused group after each accept always refuse.
        for level in range(top, 0, -1):
            for node in dict.fromkeys(at(m, level) for m in reversed(leaves) if ideal[m.id] == level):
                if tokens < target:
                    trial = ideal | dict.fromkeys(members(node.id), level - 1)
                    if (t := accounted(trial)) <= W and abs(t - target) < abs(tokens - target):
                        ideal, tokens = trial, t
        if (now := size(layout(ideal))) <= W or now >= rendered:
            break
        rendered = now
    escalated = size(layout(ideal)) > W
    produced = [(run[0].id, run[-1].id) for run in
                (list(run) for uncovered, run in groupby(leaves, lambda m: not m.l1) if uncovered and escalated)]

    # 2. Hold the carried frontier.
    carried = project({leaf.id: leaf.level for leaf in leaves})
    carried_units, ideal_loss = layout(carried), loss(ideal)
    gap = cfg.quality_gap_ratio * max(1, ideal_loss)
    if target <= (held := size(carried_units)) <= W and loss(carried) - ideal_loss <= gap:
        return Plan(carried, accounted(carried), 0, 'hold', False, [])

    # 3. Adopt the ideal within the trust region.
    if (pert := kv_cost(carried_units, layout(ideal))) <= P:
        return Plan(ideal, accounted(ideal), pert, 'ideal', escalated, produced)

    # 4. Adopt the ideal from a boundary on: the oldest of its changes the trust region affords,
    #    found by the library's binary search (len(changes) adopts nothing).
    changes = [leaf.seq for leaf in leaves if carried[leaf.id] != ideal[leaf.id]]

    @cache
    def suffix(i: int) -> tuple:
        F = project({m.id: ideal[m.id] if i < len(changes) and m.seq >= changes[i] else carried[m.id] for m in leaves})
        return F, (units := layout(F)), kv_cost(carried_units, units)

    F, units, partial_pert = suffix(bisect_left(range(len(changes)), True, key=lambda i: suffix(i)[2] <= P))
    if (partial := size(units)) <= W and loss(F) - ideal_loss <= gap:
        return Plan(F, accounted(F), partial_pert, 'suffix', False, [])

    # 5. The trust region loses.
    return Plan(ideal, accounted(ideal), pert, 'infeasible' if partial > W else 'quality-gap', escalated, produced)


def ceiling(W: float, cfg: Config) -> int:
    """The most a compile may render, W plus the grace: OverBudgetError's budget."""
    return math.floor(W * (1 + cfg.over_budget_grace))


def inputs(messages: list[Priced], summaries: list[Summary], levels: dict, multiplier: float,
           cfg: Config) -> Inputs:
    """The planner's view of the log (selectAdaptive): the messages between the windows at their
    post-strip prices, each with the first L1 that lists it and its carried level."""
    priced = chunking.post_strip_tokens(messages, multiplier, cfg)
    lo = chunking.head_end(messages, multiplier, cfg)
    hi = chunking.recent_start(messages, multiplier, cfg)                # a tail reaching into the head starts after it
    l1 = {m: s.id for s in summaries if s.level == 1 for m in s.sources}
    leaves = [Leaf(m.id, i, priced[i], levels.get(m.id, 0), l1.get(m.id), m.salience)
              for i, m in enumerate(messages[lo:hi], lo)]
    return Inputs(leaves, summaries, lo, priced)


def recall_tokens(answer, cfg: Config) -> int:
    """A recall pair's price (recallPairCost, default envelope, reasoning carriers stripped): its label
    plus each of its answer's blocks, all text, chars/4."""
    return math.ceil(js_len(cfg.summary_label) / 4) + sum(math.ceil(js_len(b['text']) / 4) for b in answer)
