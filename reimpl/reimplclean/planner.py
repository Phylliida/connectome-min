"""The lazy counter: what the middle of the window shows, as a frontier.

The frontier lists, in log order, a node for every closed chunk: (0, j) shows chunk j raw; (n, p)
shows layer n's memory p, which covers chunks p*k^(n-1) .. (p+1)*k^(n-1) - 1. Levels fall toward
the present: old content coarse, new content fine. A change re-reads everything after it, so a plan
keeps the carried frontier and changes it only at the edges of a band, each time where the change
starts latest:
  over W         fold until the total is at most the target, W * (1 - slack): each time, the
                 lowest level's oldest complete group: raw chunk j into memory (1, j), or memories
                 (n, p*k) .. (n, p*k+k-1), adjacent in the frontier, into (n+1, p), once it exists
  under the low  after a fold, or below W * unfold_below: replace the newest memory with what it
  water          merged while the total stays within the target
Lower levels sit later in the window, so the lowest level's oldest group is the latest change that
keeps levels falling toward the present; a level changes only when the level below it fills, as in
a counter.
"""
from typing import Callable

from config import Config

Node = tuple[int, int]                    # (layer, id)


def plan(frontier: list[Node], cost: Callable[[Node], float], exists: Callable[[Node], bool], fixed: float,
         W: float, cfg: Config) -> list[Node]:
    """fixed: the tokens outside the frontier (the open run and the tail)."""
    total = lambda F: fixed + sum(map(cost, F))
    target, F, folded = W * (1 - cfg.slack), frontier, total(frontier) > W
    while folded and total(F) > target and (folds := fold(F, exists, cfg.merge_threshold)):
        F = folds
    if folded or total(F) < W * cfg.unfold_below:
        while (trial := unfold(F, cfg.merge_threshold)) and total(trial) <= target:
            F = trial
    return F


def fold(F: list[Node], exists: Callable[[Node], bool], k: int) -> list[Node] | None:
    """F with its lowest level's oldest complete group folded, or None if no group is complete."""
    groups = [(level, i, width, parent) for i, (level, j) in enumerate(F)
              for width, parent in [(1, (1, j)) if level == 0 else (k, (level + 1, j // k))]
              if j % width == 0 and F[i:i + width] == [(level, j + t) for t in range(width)] and exists(parent)]
    if not groups:
        return None
    _, i, width, parent = min(groups)
    return F[:i] + [parent] + F[i + width:]


def unfold(F: list[Node], k: int) -> list[Node] | None:
    """F with its newest memory replaced by what it merged, or None if it shows none."""
    i = next((i for i in reversed(range(len(F))) if F[i][0] > 0), None)
    if i is None:
        return None
    level, p = F[i]
    return F[:i] + ([(0, p)] if level == 1 else [(level - 1, p * k + t) for t in range(k)]) + F[i + 1:]


def chunks(node: Node, k: int) -> range:
    """The chunks a node covers."""
    level, p = node
    return range(p, p + 1) if level == 0 else range(p * k ** (level - 1), (p + 1) * k ** (level - 1))
