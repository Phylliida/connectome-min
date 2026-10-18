"""Generic tools for an append-only sequence: the runs that close among the items between two bounds,
and an admission budget for items taken newest first.
"""
from typing import Callable


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
