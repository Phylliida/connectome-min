"""The compiled context of @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.select on the
adaptive path, then ContextManager.compile's tool-message split), reproduced exactly in default mode.

A compile plans the levels (fold.py), refuses if the plan exceeds W plus the grace, then renders:
the head raw; each middle message raw, or its ancestor's recall pair (a 'Context Manager' turn with
the summary label, then the summary as the agent) once, where its first message would be; the tail
raw. Passes over the result: trailing tool calls with no result are dropped, tool pairing is
repaired, images beyond the live limits become placeholders, and up to three cache breakpoints are
placed. Only a compile that completes commits its levels and arms calibration.

The renderer checks its running total, not just the plan's: the middle in order, then the tail
newest-first, against W plus the grace. A message whose stripped image priced it below zero can
make a prefix exceed what the whole fits under, and then the compile is refused. Not reproduced: the
full-coverage assertion, which cannot fire when every unit renders as planned. Close chunks
(chunking.close_chunks) before each compile, as select's rebuild does.
"""
import math
from dataclasses import dataclass, field, replace
from itertools import groupby, takewhile
from typing import NamedTuple

import chunking
import fold
from config import Config
from estimate import Calibration, Priced, image_bytes, store_tokens

PICKER = 'Adaptive picker exhausted'
EMISSION = 'Emission overran the plan (planner/emitter estimator drift)'
TAIL = 'Tail emission dropped reserved recent-window messages'
NEWEST = 'Structural repair did not retain the newest turn'
OMITTED = {'type': 'text', 'text': '[tool call omitted]'}
STUB = '[tool result unavailable — omitted during context compression]'
IMAGE_PLACEHOLDER = {'type': 'text', 'text': '[image dropped from live context]'}
LOOKAHEAD = 6                             # entries searched for a displaced tool result


class Refusal(NamedTuple):
    """An OverBudgetError: the compile renders nothing and commits nothing."""
    stage: str
    actual: float
    budget: int


@dataclass(frozen=True)
class Session:
    """What compiles carry between them besides the log."""
    levels: dict = field(default_factory=dict)           # carried level per message id
    demanded: frozenset = frozenset()                     # chunk indices the planner asked an L1 for
    calibration: Calibration = Calibration()
    cache_keys: list | None = None                        # the last compile's entries, for cache markers


def select(messages: list[dict], priced: list[Priced], chunks: list, summaries: list[fold.Summary],
           session: Session, W: float, cfg: Config) -> tuple[Session, list | Refusal]:
    """messages: {'id', 'participant', 'content'} in log order; priced: estimate.price of each."""
    view = fold.inputs(priced, summaries, session.levels, session.calibration.multiplier, cfg)
    limit = math.floor(W * (1 + cfg.over_budget_grace))
    if view.head_tokens + view.tail_tokens > limit:
        return session, Refusal(PICKER, view.head_tokens + view.tail_tokens, limit)
    plan = fold.plan(view, W, cfg)
    session = replace(session, demanded=session.demanded.union(
        *(chunking.demand(chunks, first, last) for first, last in plan.produced)))
    if plan.tokens > limit:
        return session, Refusal(PICKER, plan.tokens, limit)

    # Render, keeping the running total under the limit: middle first, then the tail newest-first.
    copy = lambda m: {'participant': m['participant'], 'content': m['content'], 'source': m['id']}
    tail_start, at, shown, entries = view.head + len(view.leaves), fold.ancestry(summaries), set(), []
    total, room = view.head_tokens, limit - view.tail_tokens
    for leaf, m in zip(view.leaves, messages[view.head:tail_start]):
        if (node := at(leaf, plan.levels[leaf.id])) is None:
            unit, cost = [copy(m)], leaf.tokens
        elif node.id not in shown:
            shown.add(node.id)
            unit, cost = [{'participant': 'Context Manager', 'content': [{'type': 'text', 'text': cfg.summary_label}], 'source': None},
                          {'participant': cfg.summary_participant, 'content': [{'type': 'text', 'text': node.text}], 'source': None}], node.tokens
        else:
            continue
        if total + cost > room:
            return session, Refusal(EMISSION, total + cost + view.tail_tokens, limit)
        entries, total = entries + unit, total + cost
    tail = messages[tail_start:]
    evicted = chunking.crossing(reversed(view.prices[tail_start:]), limit - total) is not None
    if evicted or tail and has(tail[0], 'tool_result') and not has(tail[0], 'tool_use'):   # its call is not in the tail
        return session, Refusal(TAIL, total + view.tail_tokens, limit)
    entries = [copy(m) for m in messages[:view.head]] + entries + [copy(m) for m in tail]

    # Repair: drop trailing calls with no result, pair tools, strip images; the newest turn must survive.
    entries = entries[:len(entries) - sum(1 for _ in takewhile(
        lambda e: has(e, 'tool_use') and not has(e, 'tool_result'), reversed(entries)))]
    entries = strip_images(pair_tools(entries), priced, session.calibration.multiplier, cfg)
    if messages and not any(e['source'] == messages[-1]['id'] for e in entries):
        return session, Refusal(NEWEST, plan.tokens, limit)

    marks, keys = cache_marks(entries, {m['id'] for m in messages[:view.head]}, {m['id'] for m in tail}, session.cache_keys)
    session = replace(session, levels=session.levels | plan.levels, calibration=session.calibration.arm(plan.tokens),
                      cache_keys=keys)
    return session, [part for i, e in enumerate(entries) for part in split(e, i in marks)]


def has(entry: dict, kind: str) -> bool:
    return any(block['type'] == kind for block in entry['content'])


def pair_tools(entries: list[dict]) -> list[dict]:
    """enforceToolPairing: a tool_result must answer a tool_use in the entry just before it, once;
    others are dropped. A tool_use the next entry doesn't answer gets its result moved up from the
    next six entries, else a stub; results join that entry, or a new user entry before it."""
    entries, uses, i = [dict(e) for e in entries], {}, 0
    while i < len(entries):
        entry = entries[i]
        if has(entry, 'tool_result'):
            seen = set()
            kept = [b for b in entry['content'] if b['type'] != 'tool_result'
                    or (b['toolUseId'] in uses and b['toolUseId'] not in seen and not seen.add(b['toolUseId']))]
            if len(kept) != len(entry['content']):
                entry['content'] = kept or [OMITTED]
        answered = {b['toolUseId'] for b in entry['content'] if b['type'] == 'tool_result'}
        if missing := [u for u in uses if u not in answered]:
            results = [take_result(entries, i, u) for u in missing]
            if answered:
                entry['content'] = results + entry['content']
            else:
                entries.insert(i, {'participant': 'user', 'content': results, 'source': None})
                uses, i = {}, i + 1
                continue
        uses = dict.fromkeys(b['id'] for b in entry['content'] if b['type'] == 'tool_use')
        i += 1
    if uses:
        entries.append({'participant': 'user', 'content': [stub(u) for u in uses], 'source': None})
    return entries


def take_result(entries: list[dict], i: int, use_id: str) -> dict:
    """relocateOrDropMissingResults: the result for `use_id` from the next LOOKAHEAD entries, removed
    from there; else a stub."""
    for later in entries[i + 1:i + 1 + LOOKAHEAD]:
        for k, block in enumerate(later['content']):
            if block['type'] == 'tool_result' and block['toolUseId'] == use_id:
                later['content'] = later['content'][:k] + later['content'][k + 1:] or [OMITTED]
                return block
    return stub(use_id)


def stub(use_id: str) -> dict:
    return {'type': 'tool_result', 'toolUseId': use_id, 'content': STUB}


def strip_images(entries: list[dict], priced: list[Priced], multiplier: float, cfg: Config) -> list[dict]:
    """applyImageStripping: newest entry first, an image becomes a placeholder when its message is
    deeper than the strip depth (walked on store prices) or the kept images reach the count or byte
    limit."""
    if not (cfg.max_live_images or cfg.image_strip_depth_tokens or cfg.max_live_image_bytes):
        return entries
    position = {m.id: i for i, m in enumerate(priced)}
    deep = chunking.crossing(reversed([store_tokens(m, multiplier) for m in priced]), cfg.image_strip_depth_tokens)
    start = 0 if deep is None else len(priced) - deep
    pos = lambda i: position.get(entries[i]['source'], -1)
    entries, kept, kept_bytes = list(entries), 0, 0
    for i in sorted((i for i, e in enumerate(entries) if has(e, 'image')), key=lambda i: -pos(i)):
        too_deep = cfg.image_strip_depth_tokens > 0 and (pos(i) < 0 or pos(i) < start)
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
