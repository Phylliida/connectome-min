"""Cuts through the summary pyramid, the planner's working material. A cut maps each leaf (a message
between the windows) to the level it renders at: 0 raw, k its L_k ancestor's recall pair, once for
all the leaves that share it. Inputs answers what a cut renders and costs, which leaves a summary
covers, and the nearest cut whose groups agree; kv_cost what switching cuts re-reads.
"""
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import takewhile
from typing import Callable, NamedTuple


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
    tokens: float                         # its recall pair's price (emit.recall_tokens)
    sources: tuple                        # message ids for an L1, child summary ids above
    parent: str | None = None


@dataclass(frozen=True)
class Inputs:
    """One compile's view for the planner: the leaves and the pyramid over them."""
    leaves: list                          # Leaf, in log order: the messages between the windows
    summaries: list                       # Summary
    head: int                             # messages in the head window; the leaves follow them
    prices: list                          # every message's post-strip store price, in log order

    @cached_property
    def head_tokens(self) -> float:
        return sum(self.prices[:self.head])

    @cached_property
    def tail_tokens(self) -> float:
        return sum(self.prices[self.head + len(self.leaves):])

    @cached_property
    def top(self) -> int:
        """The pyramid's highest level."""
        return max((s.level for s in self.summaries), default=0)

    @cached_property
    def at(self) -> Callable[[Leaf, int], Summary | None]:
        """at(leaf, level): the summary that renders the leaf at that level, if any (SummaryTree.ancestorAt)."""
        return ancestry(self.summaries)

    @cached_property
    def members(self) -> Callable[[str], tuple]:
        """members(summary id): its leaves; its head and tail messages render raw beside its recall."""
        nodes, middle = {s.id: s for s in self.summaries}, {leaf.id for leaf in self.leaves}

        @cache
        def members(sid) -> tuple:
            s = nodes[sid]
            return tuple(m for m in s.sources if m in middle) if s.level == 1 else \
                tuple(m for child in s.sources for m in members(child))
        return members

    def units(self, cut: dict) -> list:
        """The middle's rendered units in order, as (identity, tokens): siblings share one recall."""
        unit = lambda m, node: (('raw', m.seq), m.tokens) if node is None else (('recall', node.id), node.tokens)
        return list(dict.fromkeys(unit(m, self.at(m, cut[m.id])) for m in self.leaves))

    def layout(self, cut: dict) -> list:
        """renderLayout, which kv_cost and the planner's hold, suffix and escalation tests measure: the
        head and tail join only when their totals are positive (stripped images can make them negative)."""
        head, tail = self.head_tokens, self.tail_tokens
        return [('head', head)] * (head > 0) + self.units(cut) + [('tail', tail)] * (tail > 0)

    def total(self, cut: dict) -> float:
        """The token ledger the fold phases steer by, and the picker's reported total: head and tail
        always count."""
        return self.head_tokens + self.tail_tokens + size(self.units(cut))

    def project(self, cut: dict) -> dict:
        """projectToValidCut: each pass lowers by one, in place, every leaf whose group disagrees
        about its level, deepest first, until a pass changes nothing. In place matters: a group a
        deeper member just joined can agree by the time its shallower members are judged."""
        changed = True
        while changed:
            changed = False
            for leaf in sorted(self.leaves, key=lambda m: -cut[m.id]):
                level = cut[leaf.id]
                if level > 0 and not ((node := self.at(leaf, level)) and all(cut[m] == level for m in self.members(node.id))):
                    cut[leaf.id], changed = level - 1, True
        return cut


def size(units: list) -> float:
    return sum(tokens for _, tokens in units)


def kv_cost(before: list, after: list) -> float:
    """Tokens of `after` re-read from its first unit that differs from `before`'s."""
    return size(after[sum(1 for _ in takewhile(lambda pair: pair[0][0] == pair[1][0], zip(before, after))):])


def ancestry(summaries: list[Summary]) -> Callable[[Leaf, int], Summary | None]:
    """The summary that renders a leaf at a level, if any: its L1's chain of parents, read as a table
    of level -> summary."""
    nodes = {s.id: s for s in summaries}

    @cache
    def lineage(sid) -> dict:
        return {**lineage(s.parent), s.level: s} if (s := nodes.get(sid)) else {}

    return lambda leaf, level: lineage(leaf.l1).get(level)
