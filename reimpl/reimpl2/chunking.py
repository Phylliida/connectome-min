"""Chunk formation and L1 scheduling in @animalabs/context-manager 0.10.1 (rebuildChunks and its
compression queue), reproduced exactly.

Messages are estimate.Priced records in log order. A chunk is the tuple of ids of the consecutive
messages it owns; chunks are listed in close order and never change, so an index names one.

Call close_chunks wherever the library rebuilds: after every appended message and at the start of
every compile, with the multiplier then in effect. Closed chunks are permanent, so rebuild timing
decides which exist by a given moment. The harness's agent step: append the step's messages,
Calibration.observe the last request's usage, compile (close_chunks, then Calibration.arm), then make
at most one summarizer call, for l1_queue's head or, when the queue is empty, a merge.

Not reproduced, because an append-only log with stored chunks never reaches them: the library's
repairs for records that lost their messages, chunks overlapping existing L1s, and crashes between
writing an L1 and marking its chunk.
"""
import math
from itertools import accumulate, chain

from config import Config
from estimate import Priced, store_tokens

IMAGE_PLACEHOLDER = math.ceil(len('[image dropped from live context]') / 4)


def post_strip_tokens(messages: list[Priced], multiplier: float, cfg: Config) -> list[float]:
    """Store prices, less each top-level image the live window replaces with a placeholder. Walking newest
    first, an image is stripped once the unstripped total of strictly newer messages exceeds the depth, or
    the images already kept reach the count or byte limit. Its price comes off uncalibrated, as in the library."""
    out, depth, kept, kept_bytes = [], 0, 0, 0
    for message in reversed(messages):
        out.append(full := store_tokens(message, multiplier))
        for raw, size in message.images:
            if (0 < cfg.image_strip_depth_tokens < depth or 0 < cfg.max_live_images <= kept
                    or 0 < cfg.max_live_image_bytes < kept_bytes + size):
                out[-1] -= max(0, raw - IMAGE_PLACEHOLDER)
            else:
                kept, kept_bytes = kept + 1, kept_bytes + size
        depth += full
    return out[::-1]


def crossing(prices, budget) -> int | None:
    """How many prices come before the running total first exceeds budget; None if it never does.
    A scan, not a bisection: a stripped image can make a message's price negative."""
    return next((k for k, total in enumerate(accumulate(prices)) if total > budget), None)


def head_end(messages: list[Priced], multiplier: float, cfg: Config) -> int:
    """Index of the first message after the head window. The crossing message is outside it, and so is
    a tool_use just before it, which stays with its result."""
    if cfg.head_window_tokens <= 0: return 0
    end = crossing((store_tokens(m, multiplier) for m in messages), cfg.head_window_tokens)
    return len(messages) if end is None else end - (end > 0 and 'tool_use' in messages[end - 1].kinds)


def recent_start(messages: list[Priced], multiplier: float, cfg: Config) -> int:
    """Index of the first message in the tail window. The crossing message is outside it, unless the
    first tail message is a tool_result, which keeps its call."""
    inside = crossing(reversed(post_strip_tokens(messages, multiplier, cfg)), cfg.recent_window_tokens)
    start = 0 if inside is None else len(messages) - inside
    return start - (0 < start < len(messages) and 'tool_result' in messages[start].kinds)


def close_chunks(messages: list[Priced], chunks, multiplier: float, cfg: Config) -> list[tuple]:
    """The chunks that close now, to append to `chunks`.

    Walks the messages between the windows that no chunk owns, adding each to a run. A run is full once
    it reaches target_chunk_tokens over at least four messages and doesn't end on a tool_use, whose
    result comes next. A run closes when it is full, or when the next message isn't adjacent to it
    (owned messages lie between). A last run that isn't full is left for the next call."""
    owned = set(chain.from_iterable(chunks))
    closed, run, total, full = [], [], 0, False
    for i in range(head_end(messages, multiplier, cfg), recent_start(messages, multiplier, cfg)):
        if (message := messages[i]).id in owned: continue
        if run and (full or run[-1] != i - 1):
            closed.append(run)
            run, total = [], 0
        run.append(i)
        total += message.text_only if cfg.attachments_ignore_size else store_tokens(message, multiplier)
        full = total >= cfg.target_chunk_tokens and len(run) >= 4 and 'tool_use' not in message.kinds
    return [tuple(messages[i].id for i in run) for run in closed + [run] * full]


def l1_queue(chunks, summarized: set[int], demanded: set[int], cfg: Config) -> list[int]:
    """Indices of the chunks awaiting an L1, in the order the summarizer takes them: close order, minus
    the newest l1_holdback_chunks unless demanded. This is the library's queue as of its latest rebuild;
    until the next one, it appends demanded chunks in its own order, which differs from this only when
    two or more held-back chunks are demanded at once (l1_holdback_chunks >= 2)."""
    cutoff = len(chunks) - cfg.l1_holdback_chunks
    return [i for i in range(len(chunks)) if i not in summarized and (i < cutoff or i in demanded)]


def demand(chunks, first_id, last_id) -> set[int]:
    """The chunks to add to `demanded` when the folder needs an L1 over first_id..last_id
    (enqueueL1ForRange): those at either end and between. A last_id in the open run reaches the newest
    chunk; a first_id in no chunk leaves only last_id's."""
    owner = {mid: i for i, chunk in enumerate(chunks) for mid in chunk}
    first, last = owner.get(first_id), owner.get(last_id)
    if first is None: first = last
    elif last is None: last = len(chunks) - 1
    return set() if first is None else set(range(min(first, last), max(first, last) + 1))
