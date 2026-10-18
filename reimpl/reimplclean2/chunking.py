"""Chunks and the raw tail, read from the end of the log (the library's rebuildChunks, without its head
window). Messages outside the tail close into chunks, in log order: a run is full once it reaches
target_chunk_tokens of text-only tokens over at least four messages and doesn't end on a tool_use,
whose result comes next; a run that isn't full stays open. A message in a chunk never re-enters the
tail. Images count only in the tail, newest first within the live limits; elsewhere a placeholder
stands for each.
"""
import math
from functools import lru_cache

from config import Config
from estimate import Priced, price, store_tokens
from segment import budget, runs
from store import Session

PLACEHOLDER = math.ceil(len('[image dropped from live context]') / 4)


def tail_start(session: Session, multiplier: float, cfg: Config) -> int:
    """The tail's first message: walking back from the newest, the messages until one takes the total
    past recent_window_tokens, which is outside; a tool_result first in the tail brings its call in."""
    floor, admit, total, after = open_start(session), budget(cfg.max_live_images, cfg.max_live_image_bytes), 0, None
    for i in range(session.count('messages') - 1, floor - 1, -1):
        total += live_price(p := priced(session, i), multiplier, admit)
        if total > cfg.recent_window_tokens:
            return i + 1 - (after is not None and 'tool_result' in after.kinds)
        after = p
    return floor


def open_start(session: Session) -> int:
    """The first message no chunk holds."""
    return last['last'] + 1 if (last := session.last('layer0')) else 0


def close_chunks(session: Session, multiplier: float, cfg: Config) -> None:
    """Append the chunks that close now: the open run, from the last chunk to the tail, in full runs."""
    p = lambda i: priced(session, i)
    full = lambda run, total: total >= cfg.target_chunk_tokens and len(run) >= 4 and 'tool_use' not in p(run[-1]).kinds
    for run in runs(open_start(session), tail_start(session, multiplier, cfg), lambda i: True, lambda i: p(i).text_only, full):
        session.append('layer0', {'first': run[0], 'last': run[-1], 'raw': sum(stripped_raw(p(i)) for i in run)})


@lru_cache(maxsize=4096)
def priced(session: Session, i: int) -> Priced:
    """estimate.price of logged message i, kept while recently used (messages never change); the
    message itself isn't kept."""
    return price(session.read('messages', i))


def live_price(p: Priced, multiplier: float, admit) -> float:
    """A tail message's price: images past the live limits count as placeholders (uncalibrated, as in
    the library)."""
    return store_tokens(p, multiplier) - sum(max(0, raw - PLACEHOLDER) for raw, size in p.images if not admit(size))


def stripped_raw(p: Priced) -> float:
    """A message's uncalibrated price outside the tail, every image a placeholder."""
    return sum(p.raw) - sum(max(0, raw - PLACEHOLDER) for raw, _ in p.images)
