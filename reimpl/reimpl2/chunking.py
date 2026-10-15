"""Chunk formation and L1 scheduling in @animalabs/context-manager 0.10.1 (rebuildChunks and its
compression queue), reproduced exactly.

This module is the policy: the windows' budgets and their tool-pairing retreats, image stripping, when
a chunk is full, and the order demand queues chunks in. The scans, segmenting and queue are
segment.py's. Messages are estimate.Priced records in log order. A chunk is the tuple of ids of the consecutive
messages it owns; chunks are listed in close order and never change, so an index names one.

The library rebuilds (rebuild) after every appended message and at the start of every compile, with
the multiplier then in effect. Closed chunks are permanent, so rebuild timing decides which exist by
a given moment.

Not reproduced, because an append-only log with stored chunks never reaches them: the library's
repairs for records that lost their messages, chunks overlapping existing L1s, and crashes between
writing an L1 and marking its chunk.
"""
import math
from itertools import chain

from config import Config
from estimate import Priced, store_tokens
from segment import backlog, budget, crossing, runs

IMAGE_PLACEHOLDER = math.ceil(len('[image dropped from live context]') / 4)


def post_strip_tokens(messages: list[Priced], multiplier: float, cfg: Config) -> list[float]:
    """Store prices, less each top-level image the live window replaces with a placeholder. Walking newest
    first, an image is stripped once the unstripped total of strictly newer messages exceeds the depth, or
    the images already kept reach the count or byte limit. Its price comes off uncalibrated, as in the library."""
    out, depth, admit = [], 0, budget(cfg.max_live_images, cfg.max_live_image_bytes)
    for message in reversed(messages):
        out.append(full := store_tokens(message, multiplier))
        for raw, size in message.images:
            if 0 < cfg.image_strip_depth_tokens < depth or not admit(size):
                out[-1] -= max(0, raw - IMAGE_PLACEHOLDER)
        depth += full
    return out[::-1]


def head_end(messages: list[Priced], multiplier: float, cfg: Config) -> int:
    """Index of the first message after the head window. The crossing message is outside it, and so is
    a tool_use just before it, which stays with its result."""
    if cfg.head_window_tokens <= 0: return 0
    end = crossing((store_tokens(m, multiplier) for m in messages), cfg.head_window_tokens)
    return len(messages) if end is None else end - (end > 0 and 'tool_use' in messages[end - 1].kinds)


def windows(messages: list[Priced], multiplier: float, cfg: Config) -> tuple[int, int, list[float]]:
    """Where the raw windows fall: the index of the first message after the head window (head_end), of
    the first in the tail window, and each message's post-strip price, which sizes the tail. The tail
    window holds the newest messages after the one whose price takes their total past
    recent_window_tokens; a tool_result first in it brings its call in."""
    prices = post_strip_tokens(messages, multiplier, cfg)
    inside = crossing(reversed(prices), cfg.recent_window_tokens)
    tail = 0 if inside is None else len(messages) - inside
    return head_end(messages, multiplier, cfg), tail - (0 < tail < len(messages) and 'tool_result' in messages[tail].kinds), prices


def close_chunks(messages: list[Priced], chunks, multiplier: float, cfg: Config) -> list[tuple]:
    """The chunks that close now, to append to `chunks`: runs (segment.runs) of the messages between the
    windows that no chunk owns, sized text-only (or at store prices without attachments_ignore_size). A
    run is full once it reaches target_chunk_tokens over at least four messages and doesn't end on a
    tool_use, whose result comes next."""
    owned, (head, tail, _) = set(chain.from_iterable(chunks)), windows(messages, multiplier, cfg)
    size = lambda i: messages[i].text_only if cfg.attachments_ignore_size else store_tokens(messages[i], multiplier)
    full = lambda run, total: total >= cfg.target_chunk_tokens and len(run) >= 4 and 'tool_use' not in messages[run[-1]].kinds
    return [tuple(messages[i].id for i in run) for run in runs(head, tail, lambda i: messages[i].id not in owned, size, full)]


def rebuild(state, cfg: Config):
    """rebuildChunks, on a state.State: the chunks that can close do, and the L1 queue is cut afresh:
    every chunk without an L1, in close order, less the newest l1_holdback_chunks unless demanded.
    Until the next rebuild, demand appends to it (segment.enqueue)."""
    chunks = state.chunks + tuple(close_chunks(state.priced, state.chunks, state.calibration.multiplier, cfg))
    return state._replace(chunks=chunks, queue=backlog(len(chunks), state.summarized, cfg.l1_holdback_chunks, state.demanded))   # no new chunk has an L1


def demand(chunks, first_id, last_id) -> dict:
    """The chunks the folder demands an L1 for when it needs one over first_id..last_id
    (enqueueL1ForRange), as keys in the order it queues them: first_id's, last_id's, then those between. A
    last_id in the open run reaches the newest chunk; a first_id in no chunk leaves only last_id's."""
    owner = {mid: i for i, chunk in enumerate(chunks) for mid in chunk}
    first, last = owner.get(first_id), owner.get(last_id)
    if first is None: first = last
    elif last is None: last = len(chunks) - 1
    return {} if first is None else dict.fromkeys([first, last, *range(min(first, last), max(first, last) + 1)])
