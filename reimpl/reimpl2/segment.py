"""Generic tools for an append-only sequence: where a running total first passes a budget, the runs
that close among the items between two bounds, a work queue that holds back its newest items until
they are wanted, and an admission budget for items taken newest first.
"""
from itertools import accumulate
from typing import Callable, Iterable


def crossing(prices, limit) -> int | None:
    """How many prices come before the running total first exceeds limit; None if it never does.
    A scan, not a bisection: a price can be negative."""
    return next((k for k, total in enumerate(accumulate(prices)) if total > limit), None)


def runs(start: int, stop: int, free: Callable[[int], bool], size: Callable[[int], float],
         full: Callable[[list[int], float], bool]) -> list[list[int]]:
    """The runs that close among the free items start..stop-1: each joins the current run, which closes
    once full(run, its total size), or when the next free item isn't adjacent to it. A last run that
    isn't full stays open, for a later call to extend."""
    closed, run, total, done = [], [], 0, False
    for i in filter(free, range(start, stop)):
        if run and (done or run[-1] != i - 1):
            closed.append(run)
            run, total = [], 0
        run.append(i)
        total += size(i)
        done = full(run, total)
    return closed + [run] * done


def budget(count: int, size: int) -> Callable[[int], bool]:
    """An admission budget: admit(n) says whether one more item of n bytes fits within `count` items
    and `size` bytes in all, 0 meaning no limit, and counts it if it does."""
    kept = kept_bytes = 0

    def admit(n: int) -> bool:
        nonlocal kept, kept_bytes
        if 0 < count <= kept or 0 < size < kept_bytes + n:
            return False
        kept, kept_bytes = kept + 1, kept_bytes + n
        return True
    return admit


def backlog(n: int, done: set[int], holdback: int, wanted) -> list[int]:
    """The queue items 0..n-1 make: those not done, in order, less the newest `holdback` unless wanted."""
    return [i for i in range(n) if i not in done and (i < n - holdback or i in wanted)]


def enqueue(queue: list[int], candidates: Iterable[int], done: set[int]) -> list[int]:
    """`queue` with each candidate neither done nor queued appended, in order."""
    return [*queue, *(c for c in dict.fromkeys(candidates) if c not in done and c not in queue)]
