"""Choosing a cut through a hierarchy (cut.py), generically: greedy coarsening and refinement of a cut
against its total, and adopting a better cut stably, only as far as switching to it re-reads few
enough tokens (kv_cost). What is better, and every limit, are the caller's.
"""
from bisect import bisect_left
from functools import cache
from itertools import product
from typing import Callable, NamedTuple

from cut import Inputs, Summary, kv_cost


class Step(NamedTuple):
    """Which step of adopt applied, with its cut."""
    name: str                             # hold | ideal | suffix | neither
    cut: dict
    cost: float                           # tokens the switch re-reads; 0 to hold
    tried: list | None = None             # the layout of the suffix candidate, once adopt tried one


def coarsen(h: Inputs, cut: dict, total: Callable[[dict], float], order: list, more: Callable[[float], bool],
            allowed: Callable[[Summary, int], bool]) -> dict:
    """Fold whole groups while more(total(cut)): level by level from the bottom, and within a level in
    `order`, each leaf's group at that level, if the leaf is below it and allowed(group, level)."""
    tokens = total(cut)
    for level, leaf in product(range(1, h.top + 1), order):
        group = h.at(leaf, level) if more(tokens) and cut[leaf.id] < level else None
        if group and allowed(group, level):
            cut = cut | dict.fromkeys(h.members(group.id), level)
            tokens = total(cut)
    return cut


def refine(h: Inputs, cut: dict, total: Callable[[dict], float], more: Callable[[float], bool],
           better: Callable[[float, float], bool]) -> dict:
    """Unfold whole groups one level while more(total(cut)): from the top level down, and within a level
    newest first, each group once, keeping each unfold better(new total, old total) approves."""
    tokens = total(cut)
    for level in range(h.top, 0, -1):
        for group in dict.fromkeys(h.at(m, level) for m in reversed(h.leaves) if cut[m.id] == level):
            if more(tokens):
                trial = cut | dict.fromkeys(h.members(group.id), level - 1)
                if better(t := total(trial), tokens):
                    cut, tokens = trial, t
    return cut


def adopt(h: Inputs, current: dict, ideal: dict, budget: float, holds: Callable[[dict, list], bool],
          fits: Callable[[dict, list], bool]) -> Step:
    """Move from the current cut toward the ideal one, the first step that applies:
      hold     keep the current cut, if holds(cut, its layout)
      ideal    switch, if that re-reads at most `budget` tokens
      suffix   adopt the ideal's changes from a boundary on, projected: the oldest boundary whose cut
               re-reads at most `budget`, by binary search over the changes (past the newest, nothing
               changes); if fits(cut, its layout)
      neither  the ideal anyway, at its full cost
    `current` must be a valid cut (Inputs.project)."""
    units = h.layout(current)
    if holds(current, units):
        return Step('hold', current, 0)
    if (cost := kv_cost(units, h.layout(ideal))) <= budget:
        return Step('ideal', ideal, cost)
    changes = [leaf.seq for leaf in h.leaves if current[leaf.id] != ideal[leaf.id]]

    @cache
    def suffix(i: int) -> tuple:
        F = h.project({m.id: ideal[m.id] if i < len(changes) and m.seq >= changes[i] else current[m.id] for m in h.leaves})
        return F, (layout := h.layout(F)), kv_cost(units, layout)

    F, layout, partial = suffix(bisect_left(range(len(changes)), True, key=lambda i: suffix(i)[2] <= budget))
    return Step('suffix', F, partial, layout) if fits(F, layout) else Step('neither', ideal, cost, layout)
