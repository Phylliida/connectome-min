"""The compiled context (AutobiographicalStrategy.select, then ContextManager.compile's tool-message
split), from the session's logs. Chunks close, the counter plans the frontier (planner.py), and the
window renders: each frontier node's chunk raw or its memory's recall pair (a 'Context Manager' turn
with the summary label, then the memory as answer() renders it, as the agent); then the messages no
chunk holds yet, raw. Passes over the result: trailing tool calls with no result are dropped, tool
pairing is repaired, and up to three cache breakpoints are placed. A compile over W plus the grace is
refused; only a compile that completes commits its frontier and arms calibration, and a refused one
that couldn't fold enough marks the session escalated, so the next L1 doesn't wait for newer chunks.
"""
import hashlib
import json
import math
from functools import cache
from itertools import takewhile
from typing import NamedTuple

from chunking import close_chunks, live_price, open_start, priced, stripped_raw, tail_start
from config import Config
from estimate import Calibration
from js import js_len, js_trim
from blocks import REASONING, image_bytes
from planner import chunks, plan
from segment import budget
from store import Session
from turns import IMAGE_PLACEHOLDER, has, pair_tools, split

LABEL = 'Context Manager'


class Refusal(NamedTuple):
    """An OverBudgetError: the compile renders nothing and commits nothing."""
    stage: str
    actual: float
    budget: int


def select(session: Session, W: float, calibration: Calibration, cfg: Config) -> list | Refusal:
    m, last = calibration.multiplier, session.last('compiles') or {}
    close_chunks(session, m, cfg)
    count = lambda n: session.count(f'layer{n}')
    cost = cache(lambda node: round(session.get('layer0', node[1])['raw'] * m) if node[0] == 0 else session.get(f'layer{node[0]}', node[1])['recall'])
    shown = [tuple(node) for node in last.get('frontier', [])]
    covered = chunks(shown[-1], cfg.merge_threshold).stop if shown else 0
    frontier = shown + [(0, j) for j in range(covered, count(0))]
    tail, loose = tail_start(session, m, cfg), list(session.span('messages', open_start(session), session.count('messages') - 1))
    admit = budget(cfg.max_live_images, cfg.max_live_image_bytes)
    live = {msg['id'] for msg in loose if msg['id'] >= tail}
    fixed = sum(live_price(p, m, admit) if msg['id'] in live else round(stripped_raw(p) * m)
                for msg in reversed(loose) for p in [priced(session, msg)])
    frontier = plan(frontier, cost, cache(lambda node: node[1] < count(node[0])), fixed, W, cfg)
    total, limit = fixed + sum(map(cost, frontier)), math.floor(W * (1 + cfg.over_budget_grace))
    if total > limit:
        session.append('compiles', {**last, 'multiplier': m, 'armed': calibration.armed, 'escalated': True})
        return Refusal('Adaptive picker exhausted', total, limit)

    entries = [e for node in frontier for e in render(session, node, cfg)] + messages(loose, live, cfg)
    entries = entries[:len(entries) - sum(1 for _ in takewhile(lambda e: has(e, 'tool_use') and not has(e, 'tool_result'), reversed(entries)))]
    entries = pair_tools(entries)
    if loose and not any(e['source'] == loose[-1]['id'] for e in entries):
        return Refusal('Structural repair did not retain the newest turn', total, limit)
    keys = [hashlib.sha256(json.dumps([e['participant'], e['content']]).encode()).hexdigest() for e in entries]
    marks = cache_marks(keys, [e['source'] in live for e in entries], last.get('keys'))
    session.append('compiles', {'frontier': frontier, 'multiplier': m, 'armed': total, 'keys': keys, 'escalated': total > W})
    return [part for i, e in enumerate(entries) for part in split(e, i in marks)]


def render(session: Session, node, cfg: Config) -> list[dict]:
    """A frontier node's entries: its chunk's messages, or its memory's recall pair."""
    if node[0] == 0:
        chunk = session.get('layer0', node[1])
        return messages(list(session.span('messages', chunk['first'], chunk['last'])), set(), cfg)
    memory = session.get(f'layer{node[0]}', node[1])
    return [{'participant': LABEL, 'content': [{'type': 'text', 'text': cfg.summary_label}], 'source': None},
            {'participant': cfg.summary_participant, 'content': answer(memory), 'source': None}]


def messages(log: list[dict], live: set, cfg: Config) -> list[dict]:
    """Messages as entries: images past the live limits, newest first, or outside the tail, become
    placeholders."""
    admit = budget(cfg.max_live_images, cfg.max_live_image_bytes)
    keep = lambda msg, b: b['type'] != 'image' or msg['id'] in live and admit(image_bytes(b))
    out = [{'participant': msg['participant'], 'content': [b if keep(msg, b) else IMAGE_PLACEHOLDER for b in msg['content']],
            'source': msg['id']} for msg in reversed(log)]
    return out[::-1]


def answer(memory: dict) -> list[dict]:
    """A memory's recall answer under carrierPolicy 'live-strip' (liveWindowAnswerProse): the reply's
    stored blocks without their reasoning, when a text block among them has visible text; else the
    memory's text."""
    stripped = [b for b in memory.get('blocks') or () if b['type'] not in REASONING]
    return stripped if any(js_trim(b['text']) for b in stripped) else [{'type': 'text', 'text': memory['text']}]


def recall_tokens(answer: list[dict], cfg: Config) -> int:
    """A recall pair's price (recallPairCost): its label plus each of its answer's blocks, chars/4."""
    return math.ceil(js_len(cfg.summary_label) / 4) + sum(math.ceil(js_len(b['text']) / 4) for b in answer)


def cache_marks(keys: list[str], in_tail: list[bool], previous: list | None) -> set[int]:
    """placeCacheMarkers: the end of the prefix unchanged since the previous compile (before any, the
    last entry ahead of the tail), and the last entry."""
    if not keys:
        return set()
    stable = (next((i for i, t in enumerate(in_tail) if t), len(keys)) if previous is None
              else sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(keys, previous)))) - 1
    return {len(keys) - 1} | ({stable} if stable >= 0 else set())
