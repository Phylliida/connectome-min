"""The compiled context (AutobiographicalStrategy.select, then ContextManager.compile's tool-message
split), from the session's logs, in two passes.

select plans from what messages and memories cost and their blocks' shapes (estimate.Priced,
blocks.Block), never their content: chunks close, the counter plans the frontier (planner.py), and
the window is laid out as turns of blocks that say where their content is: each frontier node's chunk
raw or its memory's recall pair (a 'Context Manager' turn with the summary label, then the memory as
answer() renders it, as the agent); then the messages no chunk holds yet, raw. Passes over the layout:
trailing tool calls with no result are dropped, tool pairing is repaired, and up to three cache
breakpoints are placed. A compile over W plus the grace is refused; only a compile that completes
commits its frontier and arms calibration, and a refused one that couldn't fold enough marks the
session escalated, so the next L1 doesn't wait for newer chunks.

stream then reads each block's content as the caller writes the turns out, so the context is never
in memory, only one record of it at a time.
"""
import math
from functools import cache
from itertools import count, takewhile
from typing import Callable, Iterator, NamedTuple

from blocks import REASONING, Block, shape
from chunking import close_chunks, live_price, open_start, priced, stripped_raw, tail_start
from config import Config
from estimate import Calibration
from js import digest, js_len, js_trim
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


def select(session: Session, W: float, calibration: Calibration, cfg: Config) -> list[dict] | Refusal:
    """The plan: turns {'participant', 'content': [Block], 'cacheBreakpoint'?}."""
    m, last = calibration.multiplier, session.last('compiles') or {}
    close_chunks(session, m, cfg)
    held = lambda n: session.count(f'layer{n}')
    cost = cache(lambda node: round(session.get('layer0', node[1])['raw'] * m) if node[0] == 0 else session.get(f'layer{node[0]}', node[1])['recall'])
    shown = [tuple(node) for node in last.get('frontier', [])]
    covered = chunks(shown[-1], cfg.merge_threshold).stop if shown else 0
    frontier = shown + [(0, j) for j in range(covered, held(0))]
    tail, loose = tail_start(session, m, cfg), range(open_start(session), session.count('messages'))
    admit, live = budget(cfg.max_live_images, cfg.max_live_image_bytes), range(tail, loose.stop)
    fixed = sum(live_price(p, m, admit) if i in live else round(stripped_raw(p) * m) for i in reversed(loose) for p in [priced(session, i)])
    frontier = plan(frontier, cost, cache(lambda node: node[1] < held(node[0])), fixed, W, cfg)
    total, limit = fixed + sum(map(cost, frontier)), math.floor(W * (1 + cfg.over_budget_grace))
    if total > limit:
        session.append('compiles', {**last, 'multiplier': m, 'armed': calibration.armed, 'escalated': True})
        return Refusal('Adaptive picker exhausted', total, limit)

    entries = [e for node in frontier for e in render(session, node, cfg)] + messages(session, loose, live, cfg)
    entries = entries[:len(entries) - sum(1 for _ in takewhile(lambda e: has(e, 'tool_use') and not has(e, 'tool_result'), reversed(entries)))]
    entries = pair_tools(entries)
    if loose and not any(e['source'] == loose[-1] for e in entries):
        return Refusal('Structural repair did not retain the newest turn', total, limit)
    keys = [digest([e['participant'], [b.at for b in e['content']]]) for e in entries]
    marks = cache_marks(keys, [e['source'] is not None and e['source'] >= tail for e in entries], last.get('keys'))
    session.append('compiles', {'frontier': frontier, 'multiplier': m, 'armed': total, 'keys': keys, 'escalated': total > W})
    return [part for i, e in enumerate(entries) for part in split(e, i in marks)]


def render(session: Session, node, cfg: Config) -> list[dict]:
    """A frontier node's entries: its chunk's messages, or its memory's recall pair."""
    if node[0] == 0:
        chunk = session.get('layer0', node[1])
        return messages(session, range(chunk['first'], chunk['last'] + 1), range(0), cfg)
    return [{'participant': LABEL, 'content': [shape({'type': 'text', 'text': cfg.summary_label})], 'source': None},
            {'participant': cfg.summary_participant, 'content': recalled(session.get(f'layer{node[0]}', node[1]), node[0]), 'source': None}]


def messages(session: Session, ids: range, live: range, cfg: Config) -> list[dict]:
    """Messages as entries: images past the live limits, newest first, or outside the tail, become
    placeholders."""
    admit = budget(cfg.max_live_images, cfg.max_live_image_bytes)
    keep = lambda i, b: b.type != 'image' or i in live and admit(b.images[0])
    out = [{'participant': p.participant, 'content': [b if keep(i, b) else IMAGE_PLACEHOLDER for b in p.blocks], 'source': i}
           for i in reversed(ids) for p in [priced(session, i)]]
    return out[::-1]


def answer(memory: dict) -> list[dict]:
    """A memory's recall answer under carrierPolicy 'live-strip' (liveWindowAnswerProse): the reply's
    stored blocks without their reasoning, when a text block among them has visible text; else the
    memory's text."""
    stripped = [b for b in memory.get('blocks') or () if b['type'] not in REASONING]
    return stripped if any(js_trim(b['text']) for b in stripped) else [{'type': 'text', 'text': memory['text']}]


def recalled(memory: dict, level: int) -> list[Block]:
    """The blocks of a memory's answer, located in its layer."""
    return [shape(b, (f'layer{level}', memory['id'], j)) for j, b in enumerate(answer(memory))]


def recall_tokens(answer: list[dict], cfg: Config) -> int:
    """A recall pair's price (recallPairCost): its label plus each of its answer's blocks, chars/4."""
    return math.ceil(js_len(cfg.summary_label) / 4) + sum(math.ceil(js_len(b['text']) / 4) for b in answer)


def cache_marks(keys: list[str], in_tail: list[bool], previous: list | None) -> set[int]:
    """placeCacheMarkers: the end of the prefix unchanged since the previous compile (before any, the
    last entry ahead of the tail), and the last entry. An entry's key is where its blocks are, which
    says what they hold, since records never change."""
    if not keys:
        return set()
    stable = (next((i for i, t in enumerate(in_tail) if t), len(keys)) if previous is None
              else sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(keys, previous)))) - 1
    return {len(keys) - 1} | ({stable} if stable >= 0 else set())


def stream(session: Session, turns: list[dict], scrub: bool = False) -> Iterator[dict]:
    """The planned turns with their content: each block is read from disk as the turn's content is
    iterated. scrub drops the blocks' stale cache_control markers."""
    read = reader(session)
    return ({**t, 'content': (load(read, b, scrub) for b in t['content'])} for t in turns)


def reader(session: Session) -> Callable[[str, int], dict]:
    """session.read for records read mostly in order, as a stream reads them: per file, the record read
    last is kept, the next one continues its scan, and any other starts a scan."""
    scans = {}                                # file -> (the id read last, its record, its scan)

    def read(file: str, i: int) -> dict:
        last, record, scan = scans.get(file, (None, None, None))
        if i != last:
            if last is None or i != last + 1:
                scan = session.span(file, i, session.count(file) - 1)
            scans[file] = (i, record := next(scan), scan)
        return record
    return read


def load(read, b: Block, scrub: bool) -> dict:
    """A block's content: itself, or read from where it is, with the images it drops as placeholders."""
    if isinstance(b.at, dict):
        return b.at
    file, i, j, *dropped = b.at
    block = (read(file, i)['content'] if file == 'messages' else answer(read(file, i)))[j]
    if dropped:
        seen = count()
        block = {**block, 'content': [IMAGE_PLACEHOLDER.at if c['type'] == 'image' and next(seen) in dropped else c for c in block['content']]}
    return {k: v for k, v in block.items() if k != 'cache_control'} if scrub else block
