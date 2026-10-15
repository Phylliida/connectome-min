"""Memory formation in @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.tick, with
compressChunkHierarchical for L1s), reproduced exactly in default mode.

A tick writes at most one memory: an L1 for the compression queue's next chunk or, when that queue is
empty, the merge queue's first merge (merges.py). Both ask the summarizer the same way (prompts.py).

An L1 (mint):
  1. A chunk of text blocks only, under min_chunk_chars of trimmed text, gets a fixed stub; no call.
  2. The request replays the head window, a recall pair for each unmerged summary the recall budget
     fits, and the raw messages from the head to the chunk; then a marker turn, the chunk and the
     instruction.
  3. The tick defers, without a call, while the host has declared no tools and the request carries
     tool blocks or the summarizer is a Fable/Mythos model.
  4. A chunk whose request failed before, or that has failed three requests, is skipped.
  5. The calls climb a ladder of rungs (ladder) until a reply may become the memory; when none may,
     the request is quarantined.
  6. The memory is stored as L1-n, and the chunk's quarantine clears.

Not reproduced, because the library's defaults and an append-only log with stored chunks never
reach them: the guards against chunks that overlap an L1 and merges whose sources are missing or
merged, source-only requests and their split fallback, witnessed and reading-mode instructions, the
identity reminder, the XML recall envelope, branch switches, concurrent producers, alerts, preimage
storage and telemetry. Nor is the L1 request's image byte cap: in 0.10.1 it rewrites the prompt's
turns after the request has copied them.
"""
import re

import chunking
from calls import Failure, Host, Reply, call, input_tokens, written
from config import Config
from js import js_json, js_len, js_trim
from merges import enqueue_merges, merge, sweep
from fallbacks import admitted, variants
from prompts import INSTRUCTION, MARKER, NO_TOOLS, PLAIN_PROSE, Context, ask, raw, say, with_line, without_tools
from state import Entry, State

QUIET = ('(A quiet stretch: {} messages of routine system traffic — heartbeats, empty turns, notices. '
         'Nothing happened worth remembering.)')
TOOL_BLOCKS = ('tool_use', 'tool_result')
SHAPE_CAP = 3                             # distinct failed requests per chunk before it is skipped for good
_TOOLS_LESS = re.compile(r'(?:^|/)claude-(?:fable|mythos)-')


def tick(state: State, host: Host, cfg: Config) -> tuple[State, Exception | None]:
    """AutobiographicalStrategy.tick. Rebuilds cut state.queue afresh (chunking.rebuild); ticks
    between rebuilds take its following entries. An error the library's tick throws comes back
    beside the state: an L1's chunk has left the queue and nothing else changed; a merge stays
    queued, with a server error counted."""
    ctx = Context({m['id']: i for i, m in enumerate(state.messages)},
                  chunking.head_end(state.priced, state.calibration.multiplier, cfg), host, cfg)
    state = sweep(ctx, state)
    if not state.queue:
        return merge(ctx, state) if state.merges else (state, None)
    index, state = state.queue[0], state._replace(queue=state.queue[1:])
    try:
        return mint(ctx, index, state)
    except Exception as error:            # reading a malformed reply; the library's tick catches that too
        return state, error


def mint(ctx: Context, index: int, state: State) -> tuple[State, Exception | None]:
    """compressChunkHierarchical for chunk `index`: a quiet chunk gets a stub without a call; a request
    the host can't take yet, or one already quarantined, waits; otherwise the ladder decides."""
    cfg, ids = ctx.cfg, state.chunks[index]
    chunk = [state.messages[ctx.position[i]] for i in ids]
    blocks = [b for m in chunk for b in m['content']]
    if all(b['type'] == 'text' for b in blocks) and sum(js_len(js_trim(b['text'])) for b in blocks) < cfg.min_chunk_chars:
        return remember(ctx, state, Entry(f'L1-{len(state.entries)}', 1, QUIET.format(len(ids)), 40, ids, (ids[0], ids[-1]))), None

    request, kept = ask(ctx, state, [s for s in state.entries if s.parent is None], cfg.recall_budget_tokens, ctx.position[ids[0]],
                        [say(MARKER), *raw(chunk), say(INSTRUCTION.format(cfg.summary_target_tokens))])
    if not ctx.host.tools and (_TOOLS_LESS.search(cfg.compression_model)
                               or any(b['type'] in TOOL_BLOCKS for t in request['messages'] for b in t['content'])):
        return state, None
    family = (ids, js_json(request))
    if family in state.quarantine or sum(q[0] == ids for q in state.quarantine) >= SHAPE_CAP:
        return state, None
    match ladder(ctx, state, request, kept):
        case Failure(error=error):
            return state, error
        case None:
            return state._replace(quarantine=state.quarantine | {family}), None
        case reply:
            return remember(ctx, state, written(state, 1, ids, (ids[0], ids[-1]), reply, cfg)), None


def ladder(ctx: Context, state: State, request: dict, kept: list[Entry]) -> Reply | Failure | None:
    """The L1's calls, one per rung, each result picking the next rung. Returns the reply that becomes
    the memory, the failure that ends the tick, or None when no rung is left to try.
      first          the request: accepted on end_turn; on tool_use, next 'no tools'
      no tools       the request with the no-tools line: accepted on end_turn; on tool_use, next 'tools dropped'
      tools dropped  that request without the tools: accepted on end_turn
      variant        each recall variant the first reply's input count admits, in turn: accepted if valid
      prose          an accepted reply was all <thinking>: the request with the plain-prose line, accepted
                     on end_turn with text
    A failure ends the tick, except a variant's, which counts as its rejection."""
    rung, result, options = 'first', call(ctx.host, request), None
    first = result
    while True:
        match rung, result:
            case _, Failure() if rung != 'variant':
                return result
            case 'prose', Reply(stop='end_turn') if js_trim(result.text):
                return result
            case 'prose', _:
                return None
            case ('first' | 'no tools' | 'tools dropped', Reply(stop='end_turn')) | ('variant', Reply(verdict=('valid', _))):
                if js_trim(result.text):
                    return result
                rung, result = 'prose', call(ctx.host, with_line(request, PLAIN_PROSE))
            case 'first', Reply(stop='tool_use'):
                rung, result = 'no tools', call(ctx.host, lined := with_line(request, NO_TOOLS))
            case 'no tools', Reply(stop='tool_use'):
                rung, result = 'tools dropped', call(ctx.host, without_tools(lined))
            case _:
                options = options or admitted(request, variants(ctx, state, request, kept), input_tokens(first.response), ctx.cfg)
                if (variant := next(options, None)) is None:
                    return None
                rung, result = 'variant', call(ctx.host, variant)


def remember(ctx: Context, state: State, entry: Entry) -> State:
    """Store an L1, clear its chunk's quarantine, and queue the merges it completes."""
    return enqueue_merges(ctx, state._replace(entries=state.entries + (entry,),
                                              quarantine=frozenset(q for q in state.quarantine if q[0] != entry.sources)))
