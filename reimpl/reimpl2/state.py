"""The strategy's state between calls: AutobiographicalStrategy's fields and its message store's, as
the harness drives them. chunking.rebuild, emit.select and mint.tick each take one and return the
next; drive.py sequences them as the harness's pass does."""
from typing import NamedTuple

from estimate import Calibration


class State(NamedTuple):
    # the log
    messages: tuple = ()                  # {'id', 'participant', 'content'}, in order
    priced: tuple = ()                    # estimate.price of each
    chunks: tuple = ()                    # chunking.close_chunks's, in close order
    # memory
    entries: tuple = ()                   # mint.Entry, in creation order: the n of L<k>-n counts the ones before it
    queue: tuple = ()                     # chunk indices awaiting an L1: the last rebuild's, then demand's, less what ticks took
    quarantine: frozenset = frozenset()   # (chunk, request JSON) for each request no reply could answer
    merges: tuple = ()                    # mint.Merge, in the order they run
    merge_quarantine: frozenset = frozenset()   # source runs that exhausted their attempts
    # what compiles carry
    levels: dict = {}                     # carried level per message id
    demanded: dict = {}                   # chunk indices the planner asked an L1 for, keys in that order
    calibration: Calibration = Calibration()
    cache_keys: list | None = None        # the last compile's entries, for cache markers

    @property
    def summarized(self) -> set[int]:
        """Indices of the chunks with an L1."""
        l1s = {s.sources for s in self.entries if s.level == 1}
        return {i for i, chunk in enumerate(self.chunks) if chunk in l1s}
