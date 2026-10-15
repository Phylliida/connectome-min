"""The compiled context of @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.select on the
adaptive path, then ContextManager.compile's tool-message split), reproduced exactly in default mode.

A compile plans the levels (planner.py), refuses if the plan exceeds W plus the grace, then renders:
the head raw; each middle message raw, or its ancestor's recall pair (a 'Context Manager' turn with
the summary label, then the memory as answer() renders it, as the agent) once, where its first
message would be; the tail raw. Passes over the result: trailing tool calls with no result are
dropped, tool pairing is repaired, images beyond the live limits become placeholders, and up to
three cache breakpoints are placed. Only a compile that completes commits its levels and arms
calibration.

The renderer checks its running total, not just the plan's: the middle in order, then the tail
newest-first, against W plus the grace. A message whose stripped image priced it below zero can
make a prefix exceed what the whole fits under, and then the compile is refused. Not reproduced: the
full-coverage assertion, which cannot fire when every unit renders as planned.
"""
import math
from itertools import takewhile
from typing import NamedTuple

import chunking
import planner
from blocks import REASONING, image_bytes
from config import Config
from cut import Summary
from estimate import Priced, store_tokens
from js import js_len, js_trim
from segment import budget, crossing, enqueue
from state import State
from turns import IMAGE_PLACEHOLDER, has, pair_tools, split

PICKER = 'Adaptive picker exhausted'
EMISSION = 'Emission overran the plan (planner/emitter estimator drift)'
TAIL = 'Tail emission dropped reserved recent-window messages'
NEWEST = 'Structural repair did not retain the newest turn'


class Refusal(NamedTuple):
    """An OverBudgetError: the compile renders nothing and commits nothing."""
    stage: str
    actual: float
    budget: int


def select(state: State, W: float, cfg: Config) -> tuple[State, list | Refusal]:
    """AutobiographicalStrategy.select: rebuild, plan, queue an L1 for each chunk the plan demands,
    render. Memories are state.entries, each rendered by answer()."""
    state = chunking.rebuild(state, cfg)
    messages, priced, multiplier = state.messages, state.priced, state.calibration.multiplier
    answers = {s.id: answer(s) for s in state.entries}
    summaries = [Summary(s.id, s.level, recall_tokens(answers[s.id], cfg), s.sources, s.parent) for s in state.entries]
    view = planner.inputs(priced, summaries, state.levels, multiplier, cfg)
    limit = planner.ceiling(W, cfg)
    if view.head_tokens + view.tail_tokens > limit:
        return state, Refusal(PICKER, view.head_tokens + view.tail_tokens, limit)
    plan = planner.plan(view, W, cfg)
    wanted = [c for first, last in plan.produced for c in chunking.demand(state.chunks, first, last)]
    state = state._replace(demanded=state.demanded | dict.fromkeys(wanted), queue=enqueue(state.queue, wanted, state.summarized))
    if plan.tokens > limit:
        return state, Refusal(PICKER, plan.tokens, limit)

    # Render, keeping the running total under the limit: middle first, then the tail newest-first.
    copy = lambda m: {'participant': m['participant'], 'content': m['content'], 'source': m['id']}
    recall = lambda sid: [{'participant': 'Context Manager', 'content': [{'type': 'text', 'text': cfg.summary_label}], 'source': None},
                          {'participant': cfg.summary_participant, 'content': answers[sid], 'source': None}]
    tail_start, entries, total, room = view.head + len(view.leaves), [], view.head_tokens, limit - view.tail_tokens
    for (kind, ref), cost in plan.units:
        if total + cost > room:
            return state, Refusal(EMISSION, total + cost + view.tail_tokens, limit)
        entries, total = entries + ([copy(messages[ref])] if kind == 'raw' else recall(ref)), total + cost
    tail = messages[tail_start:]
    evicted = crossing(reversed(view.prices[tail_start:]), limit - total) is not None
    if evicted or tail and has(tail[0], 'tool_result') and not has(tail[0], 'tool_use'):   # its call is not in the tail
        return state, Refusal(TAIL, total + view.tail_tokens, limit)
    entries = [copy(m) for m in messages[:view.head]] + entries + [copy(m) for m in tail]

    # Repair: drop trailing calls with no result, pair tools, strip images; the newest turn must survive.
    entries = entries[:len(entries) - sum(1 for _ in takewhile(
        lambda e: has(e, 'tool_use') and not has(e, 'tool_result'), reversed(entries)))]
    entries = strip_images(pair_tools(entries), priced, multiplier, cfg)
    if messages and not any(e['source'] == messages[-1]['id'] for e in entries):
        return state, Refusal(NEWEST, plan.tokens, limit)

    marks, keys = cache_marks(entries, {m['id'] for m in messages[:view.head]}, {m['id'] for m in tail}, state.cache_keys)
    state = state._replace(levels=state.levels | plan.levels, calibration=state.calibration.arm(plan.tokens), cache_keys=keys)
    return state, [part for i, e in enumerate(entries) for part in split(e, i in marks)]


def answer(memory) -> list[dict]:
    """A memory's recall answer as the live window renders it under the harness's carrierPolicy
    'live-strip' (liveWindowAnswerProse): the reply's stored blocks without their reasoning, block
    for block, when a text block among them has visible text; else the memory's text. Its pair's
    price follows this render (recallPairCost), so the summarizer's token count doesn't enter it."""
    stripped = [b for b in memory.blocks or () if b['type'] not in REASONING]
    return stripped if any(js_trim(b['text']) for b in stripped) else [{'type': 'text', 'text': memory.content}]   # all text


def recall_tokens(answer, cfg: Config) -> int:
    """A recall pair's price (recallPairCost, default envelope, reasoning carriers stripped): its label
    plus each of its answer's blocks, all text, chars/4."""
    return math.ceil(js_len(cfg.summary_label) / 4) + sum(math.ceil(js_len(b['text']) / 4) for b in answer)


def strip_images(entries: list[dict], priced: list[Priced], multiplier: float, cfg: Config) -> list[dict]:
    """applyImageStripping, in place: newest entry first, an image becomes a placeholder when its
    message is deeper than the strip depth (walked on store prices) or the kept images reach the
    count or byte limit."""
    position = {m.id: i for i, m in enumerate(priced)}
    deep = crossing(reversed([store_tokens(m, multiplier) for m in priced]), cfg.image_strip_depth_tokens)
    start = 0 if deep is None else len(priced) - deep
    pos, admit = lambda i: position.get(entries[i]['source'], -1), budget(cfg.max_live_images, cfg.max_live_image_bytes)
    for i in sorted(range(len(entries)), key=lambda i: -pos(i)):
        too_deep = cfg.image_strip_depth_tokens > 0 and pos(i) < start
        entries[i] = {**entries[i], 'content': [block if block['type'] != 'image' or not too_deep and admit(image_bytes(block))
                                                else IMAGE_PLACEHOLDER for block in entries[i]['content']]}
    return entries


def cache_marks(entries: list[dict], head: set, tail: set, previous: list | None) -> tuple[set, list | None]:
    """placeCacheMarkers: the last head entry, the end of the prefix unchanged since the previous
    compile (before any, the last entry ahead of the tail), and the last entry. Entries compare by
    participant and content, as the library compares their JSON."""
    if not entries:
        return set(), previous
    keys = [(e['participant'], e['content']) for e in entries]
    last_head = max((i for i, e in enumerate(entries) if e['source'] in head), default=-1)
    if previous is None:
        stable = next((i for i, e in enumerate(entries) if e['source'] in tail), len(entries)) - 1
    else:
        stable = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(keys, previous))) - 1
    return {len(entries) - 1} | {last_head} - {-1} | ({stable} if stable > last_head else set()), keys
