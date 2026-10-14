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
    entries: tuple = ()                   # Entry, in creation order: the n of L<k>-n counts the ones before it
    queue: tuple = ()                     # chunk indices awaiting an L1: the last rebuild's, then demand's, less what ticks took
    quarantine: frozenset = frozenset()   # (chunk, request JSON) for each request no reply could answer
    merges: tuple = ()                    # Merge, in the order they run
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


class Entry(NamedTuple):
    """A SummaryEntry: one memory, as the strategy stores it."""
    id: str
    level: int
    content: str                          # the memory's text
    tokens: int                           # the summarizer's output tokens, else chars/3; a stub's 40
    sources: tuple                        # message ids for an L1, child summary ids above
    span: tuple                           # (first, last) message id: sourceRange
    parent: str | None = None             # mergedInto
    blocks: tuple | None = None           # responseContent: the reply's blocks, when they carried reasoning
    provenance: tuple | None = None       # (stopReason, requestHash, model) of the request that wrote it


class Merge(NamedTuple):
    """A merge queue entry. It records its failures; the library's lastStopReason, lastOutcome and
    hadRefusal are views of them."""
    level: int                            # the level it writes
    sources: tuple                        # the run's ids, in chunk order
    failures: tuple = ()                  # (outcome, stopReason) of each rejected attempt
    streak: int = 0                       # serverErrorStreak: its server errors so far, never reset

    @property
    def stop(self) -> str | None:
        """lastStopReason: the latest one a failure reported."""
        return next((stop for _, stop in reversed(self.failures) if stop is not None), None)

    @property
    def outcome(self) -> str | None:
        """lastOutcome."""
        return self.failures[-1][0] if self.failures else None

    @property
    def refused(self) -> bool:
        """hadRefusal: some failure reported a refusal."""
        return any(stop == 'refusal' for _, stop in self.failures)
