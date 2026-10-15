"""The recall-curve fallback of compressChunkHierarchical (buildRecallCurveVariants and
compressionRefusalPlan): when no direct attempt at an L1 is accepted, the same request with one
recalled summary above L1 replaced by its children's recall pairs, newest summary first, as many as
curve_fallbacks allows and the context budget admits."""
from itertools import islice, pairwise
from typing import Iterable, Iterator

from config import Config
from js import js_json, js_trim
from prompts import Context, answer, header, leaves, pairs
from state import Entry, State


def variants(ctx: Context, state: State, request: dict, kept: list[Entry]) -> Iterator[dict]:
    """buildRecallCurveVariants, lazily: for each recalled summary above L1, newest first, the request
    with its pair replaced by its children's. The summary's messages must ascend in the log, its pair
    must appear exactly once, unjoined, and the result must still answer each tool block in the
    adjacent turn.

    The library also sorts each node's children by position, then requires its messages to ascend and
    its range to name the first and last. A merge takes a run adjacent in chunk order and names its
    range after the run's ends, and chunk order departs from log order only where a shrinking head
    exposes earlier messages to later chunks, which then precede the run's first; so a run passes
    exactly when it is already in log order."""
    nodes, wire = {s.id: s for s in state.entries}, request['messages']
    for parent in sorted((s for s in kept if s.level > 1), key=lambda s: -ctx.position[s.span[1]]):
        at = [i for i in range(len(wire) - 1) if header(wire[i]) == parent.id and wire[i + 1]['content'] == answer(parent)]
        if len(at) == 1 and all(ctx.position[a] < ctx.position[b] for a, b in pairwise(leaves(parent, nodes))):
            turns = wire[:at[0]] + pairs((nodes[c] for c in parent.sources), wire[at[0] + 1]['participant']) + wire[at[0] + 2:]
            if well_paired(turns):
                yield {**request, 'messages': turns}


def well_paired(turns: list[dict]) -> bool:
    """validateRecallCurveRequest: no blank text, and every tool block's partner in the adjacent turn.
    Checking each result's call suffices: the request keeps only calls answered somewhere, and tool
    ids are unique. (Turns are never empty: the request drops those, and a recall pair has blocks.
    Nor does turn 0 need guarding: the last turn, the instruction, holds no call.)"""
    return all((b['type'] != 'text' or js_trim(b['text']))
               and (b['type'] != 'tool_result' or any(c.get('id') == b['toolUseId'] for c in turns[i - 1]['content']))
               for i, t in enumerate(turns) for b in t['content'])


def admitted(request: dict, options: Iterable[dict], provider: float | None, cfg: Config) -> Iterator[dict]:
    """compressionRefusalPlan: the first curve_fallbacks variants whose input plus output reserve fits the
    context budget. Input is the provider's count for the request plus the variant's growth, or without
    a count, the variant's serialized bytes plus a per-turn reserve."""
    bound = lambda r: len(js_json(r).encode()) + 512 + 128 * len(r['messages'])
    fits = lambda v: (bound(v) if provider is None else provider + max(1, bound(v) - bound(request))) \
        + request['config']['maxTokens'] <= cfg.context_budget_tokens
    return islice(filter(fits, options), cfg.curve_fallbacks)
