"""The compiled context of @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.select on the
adaptive path, then ContextManager.compile's tool-message split), reproduced exactly in default mode.

A compile plans the levels (fold.py), refuses if the plan exceeds W plus the grace, then renders:
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
from itertools import groupby, takewhile
from typing import NamedTuple

import chunking
import fold
from config import Config
from estimate import Priced, image_bytes, js_trim, store_tokens
from state import State

PICKER = 'Adaptive picker exhausted'
EMISSION = 'Emission overran the plan (planner/emitter estimator drift)'
TAIL = 'Tail emission dropped reserved recent-window messages'
NEWEST = 'Structural repair did not retain the newest turn'
OMITTED = {'type': 'text', 'text': '[tool call omitted]'}
STUB = '[tool result unavailable — omitted during context compression]'
IMAGE_PLACEHOLDER = {'type': 'text', 'text': '[image dropped from live context]'}
LOOKAHEAD = 6                             # entries searched for a displaced tool result
REASONING = ('thinking', 'redacted_thinking')


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
    summaries = [fold.Summary(s.id, s.level, fold.recall_tokens(answers[s.id], cfg), s.sources, s.parent) for s in state.entries]
    view = fold.inputs(priced, summaries, state.levels, multiplier, cfg)
    limit = fold.ceiling(W, cfg)
    if view.head_tokens + view.tail_tokens > limit:
        return state, Refusal(PICKER, view.head_tokens + view.tail_tokens, limit)
    plan = fold.plan(view, W, cfg)
    wanted = [c for first, last in plan.produced for c in chunking.demand(state.chunks, first, last)]
    state = state._replace(demanded=state.demanded | dict.fromkeys(wanted), queue=chunking.enqueue(state.queue, wanted, state.summarized))
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
    evicted = chunking.crossing(reversed(view.prices[tail_start:]), limit - total) is not None
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


def has(entry: dict, kind: str) -> bool:
    return any(block['type'] == kind for block in entry['content'])


def pair_tools(entries: list[dict]) -> list[dict]:
    """enforceToolPairing, in place: a tool_result must answer a tool_use in the entry just before it,
    once; others are dropped. A tool_use the next entry doesn't answer gets its result moved up from
    the next six entries, else a stub; results join that entry, or a new user entry before it."""
    uses, i = {}, 0
    while i < len(entries):
        entry, seen = entries[i], set()
        kept = [b for b in entry['content'] if b['type'] != 'tool_result'
                or (b['toolUseId'] in uses and b['toolUseId'] not in seen and not seen.add(b['toolUseId']))]
        if len(kept) != len(entry['content']):
            entry['content'] = kept or [OMITTED]
        answered = {b['toolUseId'] for b in entry['content'] if b['type'] == 'tool_result'}
        if missing := [u for u in uses if u not in answered]:
            results = [take_result(entries, i, u) for u in missing]
            if not answered:                   # the new entry is checked next, and answers them
                entries.insert(i, {'participant': 'user', 'content': results, 'source': None})
                continue
            entry['content'] = results + entry['content']
        uses = dict.fromkeys(b['id'] for b in entry['content'] if b['type'] == 'tool_use')
        i += 1
    if uses:
        entries.append({'participant': 'user', 'content': [stub(u) for u in uses], 'source': None})
    return entries


def take_result(entries: list[dict], i: int, use_id: str) -> dict:
    """relocateOrDropMissingResults: the result for `use_id` from the next LOOKAHEAD entries, else a
    stub. The library also removes it there; the entry left holding it drops it when checked."""
    return next((b for later in entries[i + 1:i + 1 + LOOKAHEAD] for b in later['content']
                 if b['type'] == 'tool_result' and b['toolUseId'] == use_id), stub(use_id))


def stub(use_id: str) -> dict:
    return {'type': 'tool_result', 'toolUseId': use_id, 'content': STUB}


def strip_images(entries: list[dict], priced: list[Priced], multiplier: float, cfg: Config) -> list[dict]:
    """applyImageStripping, in place: newest entry first, an image becomes a placeholder when its
    message is deeper than the strip depth (walked on store prices) or the kept images reach the
    count or byte limit."""
    position = {m.id: i for i, m in enumerate(priced)}
    deep = chunking.crossing(reversed([store_tokens(m, multiplier) for m in priced]), cfg.image_strip_depth_tokens)
    start = 0 if deep is None else len(priced) - deep
    pos = lambda i: position.get(entries[i]['source'], -1)
    kept = kept_bytes = 0
    for i in sorted(range(len(entries)), key=lambda i: -pos(i)):
        too_deep = cfg.image_strip_depth_tokens > 0 and pos(i) < start
        content = []
        for block in entries[i]['content']:
            size = image_bytes(block)
            if block['type'] != 'image':
                content.append(block)
            elif too_deep or 0 < cfg.max_live_images <= kept or 0 < cfg.max_live_image_bytes < kept_bytes + size:
                content.append(IMAGE_PLACEHOLDER)
            else:
                content.append(block)
                kept, kept_bytes = kept + 1, kept_bytes + size
        entries[i] = {**entries[i], 'content': content}
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


def split(entry: dict, marked: bool) -> list[dict]:
    """ContextManager.compile's splitMixedToolMessages: a non-user turn carrying tool results becomes
    runs of its own blocks and user turns of results; a marked entry's last part carries the marker."""
    user = entry['participant'].lower() == 'user' or not has(entry, 'tool_result')
    parts = [(entry['participant'], entry['content'])] if user else \
        [('user' if result else entry['participant'], list(run))
         for result, run in groupby(entry['content'], lambda b: b['type'] == 'tool_result')]
    return [{'participant': p, 'content': c, **({'cacheBreakpoint': True} if marked and k == len(parts) - 1 else {})}
            for k, (p, c) in enumerate(parts)]
