"""Memory formation in @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.tick, with
compressChunkHierarchical for L1s and executeMerge above), reproduced exactly in default mode.

A tick writes at most one memory: an L1 for the compression queue's next chunk or, when that queue is
empty, the merge queue's first merge. Both ask the summarizer the same way (ask): what the agent saw
before the slice in question, replayed, then the slice and an instruction.

An L1 (mint):
  1. A chunk of text blocks only, under min_chunk_chars of trimmed text, gets a fixed stub; no call.
  2. The request replays the head window, a recall pair for each unmerged summary the recall budget
     fits, and the raw messages from the head to the chunk; then a marker turn, the chunk and the
     instruction.
  3. The tick defers, without a call, while the host has declared no tools and the request carries
     tool blocks or the summarizer is a Fable/Mythos model.
  4. A chunk whose request failed before, or that has failed three requests, is skipped.
  5. The calls go down a ladder (attempts) until a reply may become the memory; when none may, the
     request is quarantined. Text that was all <thinking> asks once more for plain prose, and text
     still empty is quarantined too.
  6. The memory is stored as L1-n, and the chunk's quarantine clears.

A merge (merge) consolidates a run of unmerged summaries of one level into one a level up, with one
call per tick; its failures so far shape what the next attempt asks (merge_request). After every
memory each level queues at most one run (enqueue_merges); a run that reaches merge_attempt_limit is
quarantined until another merge takes one of its members (sweep).

Not reproduced, because the library's defaults and an append-only log with stored chunks never
reach them: the guards against chunks that overlap an L1 and merges whose sources are missing or
merged, source-only requests and their split fallback, witnessed and reading-mode instructions, the
identity reminder, the XML recall envelope, branch switches, concurrent producers, alerts, preimage
storage and telemetry. Nor is the L1 request's image byte cap: in 0.10.1 it rewrites the prompt's
turns after the request has copied them.
"""
import hashlib
import math
import re
from functools import reduce
from itertools import chain, groupby, islice, pairwise
from typing import Callable, Iterable, Iterator, NamedTuple

import chunking
from config import Config
from emit import IMAGE_PLACEHOLDER, OMITTED, REASONING, split
from estimate import _JS_SPACE, image_bytes, js_json, js_len, js_round, js_trim
from state import State

CM = 'Context Manager'
MARKER = ('System: You will soon form a new memory, get ready. The messages that follow are the slice of '
          'recent experience you are about to compress. After them, write the memory in your own voice.')
INSTRUCTION = ('Write the memory of events since the most recent memory system notification. Speak in the '
               'first person from your own perspective. Preserve concrete details — file paths, exact values, '
               "decisions, unresolved questions, the user's active asks. Target ~{} tokens. Output only the "
               'memory body — no preamble, no section headers unless they help preservation, no '
               'meta-commentary about summarizing. Memorize only what actually happened in that slice: if it '
               'holds little beyond routine system traffic (heartbeats, empty turns, failure notices), a short '
               'memory saying so is correct — do not pad it by re-narrating events you already remember from '
               'earlier as if they had just happened again.')
MERGE_INSTRUCTION = ('You have just reviewed {seen}, in chronological order. They cover the stretch of experience '
                     'you are about to consolidate into a single L{level} memory. Write a memory that preserves the '
                     'through-line: what happened, what was decided, what remains open, what concrete details '
                     'future you will want to reach for. Speak in the first person. Target ~{target} tokens. '
                     'Output only the memory body — no preamble, no meta-commentary about summarizing.')
NO_TOOLS = '\n\nWrite the memory as plain text directly in your response — do not call any tools for this.'
PLAIN_PROSE = ('\n\nWrite the memory directly as plain prose — do not wrap it in <thinking> or any other tags; '
               'begin immediately with the memory text itself.')
QUIET = ('(A quiet stretch: {} messages of routine system traffic — heartbeats, empty turns, notices. '
         'Nothing happened worth remembering.)')
SEPARATOR = {'type': 'text', 'text': '\n\n---\n\n'}
TOOL_BLOCKS = ('tool_use', 'tool_result')
STOPS = {'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'abort'}
SHAPE_CAP = 3                             # distinct failed requests per chunk before it is skipped for good
SERVER_STREAK = 12                        # server errors in a row that count as one failed merge attempt
_HEADER = re.compile(r'\[CM\] Recall memory ([^\n\r\u2028\u2029]+)\.')
_TOOLS_LESS = re.compile(r'(?:^|/)claude-(?:fable|mythos)-')
_WRAPPED = re.compile(rf'[{_JS_SPACE}]*<thinking>.*?</thinking>[{_JS_SPACE}]*', re.I | re.A | re.S)
_OPENS = re.compile(rf'[{_JS_SPACE}]*<thinking>', re.I | re.A)


class Entry(NamedTuple):
    """A SummaryEntry: one memory, as the strategy stores it."""
    id: str
    level: int
    content: str                          # the memory's text
    tokens: int                           # the summarizer's output tokens, else chars/3; a stub's 40
    sources: tuple                        # message ids for an L1, child summary ids above
    span: tuple                           # (first, last) message id: sourceRange
    parent: str | None = None             # mergedInto
    blocks: tuple | None = None           # responseContent: the reply's blocks, when they carried reasoning
    provenance: tuple | None = None       # (stopReason, requestHash, model) of the request that wrote it


class Merge(NamedTuple):
    """A merge queue entry. It records its failures; the library's lastStopReason, lastOutcome and
    hadRefusal are views of them."""
    level: int                            # the level it writes
    sources: tuple                        # the run's ids, in chunk order
    failures: tuple = ()                  # (outcome, stopReason) of each rejected attempt
    streak: int = 0                       # serverErrorStreak: its server errors so far, never reset

    @property
    def stop(self) -> str | None:
        """lastStopReason: the latest one a failure reported."""
        return next((stop for _, stop in reversed(self.failures) if stop is not None), None)

    @property
    def outcome(self) -> str | None:
        """lastOutcome."""
        return self.failures[-1][0] if self.failures else None

    @property
    def refused(self) -> bool:
        """hadRefusal: some failure reported a refusal."""
        return any(stop == 'refusal' for _, stop in self.failures)


class Host(NamedTuple):
    """The host's side of a call."""
    complete: Callable                    # request -> response, as membrane normalizes them; raises on failure,
                                          # with the HTTP status as .status, the kind as .type, and .retryable
    tools: list | None = None             # the agent's live tool definitions, once declared
    system: str | None = None             # the host's system prompt, once declared


class Context(NamedTuple):
    """What a tick reads besides the state (StrategyContext)."""
    position: dict                        # message id -> its index in the log
    head: int                             # messages in the head window (chunking.head_end)
    host: Host
    cfg: Config


class Reply(NamedTuple):
    """A call's result: the request the provider took, and its response."""
    sent: dict
    response: dict

    @property
    def stop(self):
        return self.response.get('stopReason')

    @property
    def verdict(self) -> tuple[str, str | None]:
        """assessFallbackCompressionResponse: (outcome, stopReason). Only 'valid', a complete end_turn of
        well-formed blocks with text, may become a memory."""
        stop, content = s if isinstance(s := self.stop, str) else None, self.response.get('content')
        if stop not in STOPS or not isinstance(content, list) or not all(map(well_formed, content)):
            return 'provider_error', stop
        if stop == 'refusal':
            return 'refusal', stop
        if not js_trim('\n'.join(b['text'] for b in content if b['type'] == 'text')):
            return 'unusable_empty', stop
        return 'valid' if stop == 'end_turn' else 'incomplete', stop

    @property
    def text(self) -> str:
        """The memory's text: the reply's text blocks, less a leading <thinking>…</thinking>
        (stripThinkingPreamble); a reply that opens one and never closes it was all thinking."""
        text = '\n'.join(b.get('text') or '' for b in self.response['content'] if b['type'] == 'text')
        if m := _WRAPPED.match(text):
            return text[m.end():]
        return '' if _OPENS.match(text) else text       # an opening _WRAPPED didn't match never closes


def sha256(value) -> str:
    """sha256Json."""
    return hashlib.sha256(js_json(value).encode()).hexdigest()


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
        return mint(ctx, index, state), None
    except Exception as error:
        return state, error


# --- L1s ---

def mint(ctx: Context, index: int, state: State) -> State:
    """compressChunkHierarchical for chunk `index`."""
    cfg, ids = ctx.cfg, state.chunks[index]
    chunk = [state.messages[ctx.position[i]] for i in ids]
    blocks = [b for m in chunk for b in m['content']]
    if all(b['type'] == 'text' for b in blocks) and sum(js_len(js_trim(b['text'])) for b in blocks) < cfg.min_chunk_chars:
        return remember(ctx, state, Entry(f'L1-{len(state.entries)}', 1, QUIET.format(len(ids)), 40, ids, (ids[0], ids[-1])))

    request, kept = ask(ctx, state, [s for s in state.entries if s.parent is None], cfg.recall_budget_tokens, ctx.position[ids[0]],
                        [say(MARKER), *raw(chunk), say(INSTRUCTION.format(cfg.summary_target_tokens))])
    if not ctx.host.tools and (_TOOLS_LESS.search(cfg.compression_model)
                               or any(b['type'] in TOOL_BLOCKS for t in request['messages'] for b in t['content'])):
        return state
    family = (ids, js_json(request))
    if family in state.quarantine or sum(q[0] == ids for q in state.quarantine) >= SHAPE_CAP:
        return state
    chosen = next((reply for reply, accepted in attempts(ctx, state, request, kept) if accepted), None)
    if chosen and not js_trim(chosen.text):
        retry = call(ctx.host, with_line(request, PLAIN_PROSE))
        chosen = retry if retry.stop == 'end_turn' and js_trim(retry.text) else None
    if chosen is None:
        return state._replace(quarantine=state.quarantine | {family})
    return remember(ctx, state, written(state, 1, ids, (ids[0], ids[-1]), chosen, cfg))


def attempts(ctx: Context, state: State, request: dict, kept: list[Entry]) -> Iterator[tuple[Reply, bool]]:
    """The L1 ladder, lazily: each reply in call order, paired with whether it may become the memory.
    First the request, accepted on end_turn. After a tool_use, the request again with the no-tools
    line and, on a second tool_use, without the tools: accepted on end_turn. Then the recall variants
    the first reply's input count admits: accepted if valid. Errors escape, except a variant's, which
    counts as its failure."""
    first = call(ctx.host, request)
    yield first, first.stop == 'end_turn'
    if first.stop == 'tool_use':
        retry = call(ctx.host, lined := with_line(request, NO_TOOLS))
        if retry.stop == 'tool_use':
            retry = call(ctx.host, without_tools(lined))
        yield retry, retry.stop == 'end_turn'
    for variant in admitted(request, variants(ctx, state, request, kept), input_tokens(first.response), ctx.cfg):
        try:
            reply = call(ctx.host, variant)
        except Exception:
            continue
        yield reply, reply.verdict[0] == 'valid'


def remember(ctx: Context, state: State, entry: Entry) -> State:
    """Store an L1, clear its chunk's quarantine, and queue the merges it completes."""
    return enqueue_merges(ctx, state._replace(entries=state.entries + (entry,),
                                              quarantine=frozenset(q for q in state.quarantine if q[0] != entry.sources)))


# --- merges ---

def merge(ctx: Context, state: State) -> tuple[State, Exception | None]:
    """executeMerge for the merge queue's head, with tick's accounting of an error: a provider error
    marked not retryable counts as a failed attempt, and so does each server error from the twelfth;
    any other escapes, the merge still queued."""
    job = state.merges[0]
    request = merge_request(ctx, state, job)
    try:
        reply = call(ctx.host, request)
    except Exception as error:
        retryable, kind = getattr(error, 'retryable', None), getattr(error, 'type', None)
        server = retryable is True and kind == 'server'
        state = state._replace(merges=(job := job._replace(streak=job.streak + server),) + state.merges[1:])
        if retryable is False and isinstance(kind, str) or server and job.streak >= SERVER_STREAK:
            return reject(state, 'provider_error', None, ctx.cfg), None
        return state, error
    outcome, stop = reply.verdict
    if outcome == 'valid' and not js_trim(reply.text):
        outcome, stop = 'unusable_empty', 'end_turn'
    if outcome != 'valid':
        return reject(state, outcome, stop, ctx.cfg), None
    nodes = {s.id: s for s in state.entries}
    entry = written(state, job.level, job.sources, (nodes[job.sources[0]].span[0], nodes[job.sources[-1]].span[1]), reply, ctx.cfg)
    entries = tuple(s._replace(parent=entry.id) if s.id in job.sources else s for s in state.entries) + (entry,)
    return enqueue_merges(ctx, state._replace(entries=entries, merges=state.merges[1:])), None


def merge_request(ctx: Context, state: State, job: Merge) -> dict:
    """executeMerge's request, shaped by the merge's failures so far. It recalls the summaries before
    the run whose sources aren't the run's messages (above L1 they are summaries, so only position
    leaves one out), on a budget halved for each failure, four times at most, to no less than 8000. It
    shows the run one level deeper (an L1's messages raw, a higher summary's children as recall
    pairs) or, once a reply was refused, the run itself; adds the no-tools line after a tool_use and
    the prose line after an empty reply; caps its images; and drops the tools once two attempts have
    failed and the latest stop reason was tool_use."""
    cfg, nodes, tries = ctx.cfg, {s.id: s for s in state.entries}, len(job.failures)
    sources = [nodes[i] for i in job.sources]
    under, start = {m for s in sources for m in leaves(s, nodes)}, ctx.position[sources[0].span[0]]
    prior = [s for s in state.entries if s.parent is None and under.isdisjoint(s.sources) and ctx.position[s.span[0]] < start]
    level, who = sources[0].level, cfg.summary_participant
    target = (pairs(sources, who) if job.refused else raw(state.messages[ctx.position[m]] for s in sources for m in s.sources) if level == 1
              else pairs((nodes[c] for s in sources for c in s.sources), who))
    shown = level if job.refused else level - 1
    instruction = MERGE_INSTRUCTION.format(level=job.level, target=cfg.summary_target_tokens, seen=f'the L{shown} memories above'
                                           if shown else 'the slices of recent experience above (raw conversation)') \
        + NO_TOOLS * (job.stop == 'tool_use') + PLAIN_PROSE * (job.outcome == 'unusable_empty')
    request, _ = ask(ctx, state, prior, max(8000, js_round(cfg.recall_budget_tokens * 0.5 ** min(tries, 4))), start,
                     target + [say(instruction)])
    request = {**request, 'messages': cap_images(request['messages'], cfg.compression_image_bytes)}
    return without_tools(request) if job.stop == 'tool_use' and tries >= 2 else request


def reject(state: State, outcome: str, stop: str | None, cfg: Config) -> State:
    """recordMergeRejection for the queue's head: record the failure; at merge_attempt_limit, quarantine
    the run."""
    job = (old := state.merges[0])._replace(failures=old.failures + ((outcome, stop),))
    if len(job.failures) >= cfg.merge_attempt_limit:          # a limit under 1 acts as 1
        return state._replace(merges=state.merges[1:], merge_quarantine=state.merge_quarantine | {job.sources})
    return state._replace(merges=(job,) + state.merges[1:])


def sweep(ctx: Context, state: State) -> State:
    """sweepPaidOffMergeQuarantine, before every tick: a quarantined run that another merge took a
    member of can never run again, so it is dropped and the merges are requeued."""
    unmerged = {s.id for s in state.entries if s.parent is None}
    stale = {run for run in state.merge_quarantine if not unmerged.issuperset(run)}
    return enqueue_merges(ctx, state._replace(merge_quarantine=state.merge_quarantine - stale)) if stale else state


def enqueue_merges(ctx: Context, state: State) -> State:
    """checkMergeThresholdRecursive: for each level, a merge over the run that contiguousMergeCandidates
    picks among its unmerged, unqueued summaries, unless that run is quarantined."""
    seq, queued = {m: k for k, m in enumerate(chain.from_iterable(state.chunks))}, {s for m in state.merges for s in m.sources}
    runs = (candidates([s for s in state.entries if s.level == level and s.parent is None and s.id not in queued], seq, ctx.cfg)
            for level in range(1, max((s.level for s in state.entries), default=0) + 1))
    return state._replace(merges=state.merges + tuple(
        Merge(level, run) for level, run in enumerate(runs, 2) if run and run not in state.merge_quarantine))


def candidates(unmerged: list[Entry], seq: dict, cfg: Config) -> tuple | None:
    """contiguousMergeCandidates: once merge_threshold summaries are unmerged, split them, ordered by
    chunk position, into runs that leave no gap, leaving out any whose span is too wide for its level.
    The first run that has merge_threshold members gives its first ones; a run with a newer run after
    it needs only two.

    The library orders each span's ends and extends a run to the furthest end so far. A summary covers
    a contiguous window of chunk order (a merge takes adjacent ones), and unmerged windows are
    disjoint, so neither ever acts."""
    k = cfg.merge_threshold
    if len(unmerged) < k:
        return None
    windows = sorted((seq[s.span[0]], seq[s.span[1]], s.id) for s in unmerged
                     if seq[s.span[1]] - seq[s.span[0]] <= cfg.merge_span_messages * k ** max(0, s.level - 3))
    gaps = [i for i, (a, b) in enumerate(pairwise(windows), 1) if b[0] != a[1] + 1]
    runs = [[sid for *_, sid in windows[i:j]] for i, j in pairwise([0, *gaps, len(windows)])]
    return next((tuple(run[:k]) for i, run in enumerate(runs) if len(run) >= k or i < len(runs) - 1 and len(run) >= 2), None)


def leaves(s: Entry, nodes: dict) -> Iterable:
    """The messages under a summary, in chunk order."""
    return s.sources if s.level == 1 else chain.from_iterable(leaves(nodes[c], nodes) for c in s.sources)


# --- requests ---

def ask(ctx: Context, state: State, prior: list[Entry], budget: float, start: int, task: list[dict]) -> tuple[dict, list[Entry]]:
    """The request for `task`, after what the agent saw before message `start`: the head window raw;
    recall pairs, oldest first, for the newest of `prior` that fit `budget`; the messages from the
    head that no L1 covers, raw. Mixed tool turns are split, a participant's consecutive turns
    joined, tool blocks whose partner is nowhere in the request dropped, then blank text, empty turns
    and stale cache_control; cache seams follow unless the budget left a summary out. Returns it with
    the summaries recalled."""
    cfg, covered = ctx.cfg, {m for s in state.entries if s.level == 1 for m in s.sources}
    prior = sorted(prior, key=lambda s: ctx.position[s.span[0]])
    kept = recall(prior, budget)
    seen = raw(state.messages[:ctx.head], covered) + pairs(kept, cfg.summary_participant) + raw(state.messages[ctx.head:start], covered)
    turns = drop_unpaired(collapse([part for t in seen + task for part in split(t, False)]))
    wire = [{'participant': t['participant'], 'content': c} for t in turns if (c := [
        {k: v for k, v in b.items() if k != 'cache_control'} for b in t['content'] if b['type'] != 'text' or js_trim(b['text'])])]
    marks = set() if len(kept) < len(prior) else seams(wire, {s.id: s.level for s in kept})
    cap = c if (c := cfg.compression_max_tokens or 0) > 0 else math.inf
    max_tokens = min(max(16000, js_round(cfg.summary_target_tokens * 1.5)), cap)
    return {**({'system': ctx.host.system} if ctx.host.system else {}), 'shedOversizeImages': True,
            'messages': [{**t, 'cacheBreakpoint': True} if i in marks else t for i, t in enumerate(wire)],
            **({'cacheTtl': '1h'} if marks else {}), 'config': {'model': cfg.compression_model, 'maxTokens': max_tokens},
            **({'tools': ctx.host.tools} if ctx.host.tools is not None else {})}, kept


def say(text: str) -> dict:
    return {'participant': CM, 'content': [{'type': 'text', 'text': text}]}


def raw(messages: Iterable[dict], covered=frozenset()) -> list[dict]:
    """Messages as the request replays them: without their reasoning, unless `covered` has them."""
    return [{'participant': m['participant'], 'content': strip_reasoning(m['content'])} for m in messages if m['id'] not in covered]


def recall(prior: list[Entry], budget: float) -> list[Entry]:
    """capRecallPairs: newest first, every summary that still fits the budget at its tokens plus 50."""
    fit = lambda acc, s: ([s, *acc[0]], acc[1] + s.tokens + 50) if acc[1] + s.tokens + 50 <= budget else acc
    return reduce(fit, reversed(prior), ([], 0))[0]


def pairs(summaries: Iterable[Entry], participant: str) -> list[dict]:
    """Recall pairs: a turn naming each summary, then its answer as `participant`."""
    return [t for s in summaries for t in (say(f'[CM] Recall memory {s.id}.'), {'participant': participant, 'content': answer(s)})]


def seams(wire: list[dict], levels: dict) -> set[int]:
    """applyMintCacheSeams: the turns a breakpoint follows: the one before the first recall pair, the
    newest pair above L1's answer, and the newest pair's answer. None without pairs. (The library
    scans all but the last turn, the instruction, which never names a summary.)"""
    found = [(i, levels[sid]) for i, t in enumerate(wire) if (sid := header(t)) in levels]
    return {found[0][0] - 1, found[-1][0] + 1, max((i + 1 for i, level in found if level >= 2), default=-1)} - {-1} if found else set()


def cap_images(turns: list[dict], cap: int) -> list[dict]:
    """capCompressionImageBytes: newest turn first, images, nested ones too, become placeholders once
    the bytes kept would pass `cap`. (The library caps before the request drops blank text, empty
    turns and cache_control, none of which an image is.)"""
    kept = 0

    def fit(b: dict) -> dict:
        nonlocal kept
        if b['type'] == 'image':
            if fits := kept + image_bytes(b) <= cap:
                kept += image_bytes(b)
            return b if fits else IMAGE_PLACEHOLDER
        return {**b, 'content': list(map(fit, b['content']))} if isinstance(b.get('content'), list) else b   # a tool result's
    return turns if cap <= 0 else [{**t, 'content': list(map(fit, t['content']))} for t in turns[::-1]][::-1]


def answer(s: Entry) -> list:
    """summaryAnswerContent: the reply's blocks, reasoning included, or the memory as one text block."""
    return list(s.blocks) if s.blocks else [{'type': 'text', 'text': s.content}]


def strip_reasoning(content: list) -> list:
    return [b for b in content if b['type'] not in REASONING]


def collapse(turns: list[dict]) -> list[dict]:
    """collapseConsecutiveMessages: a participant's consecutive turns join, a rule between each."""
    return [{'participant': p, 'content': [b for k, t in enumerate(run) for b in [SEPARATOR] * (k > 0) + t['content']]}
            for p, run in groupby(turns, lambda t: t['participant'])]


def drop_unpaired(turns: list[dict]) -> list[dict]:
    """stripUnpairedToolBlocks: tool blocks whose partner is nowhere in the request go; a turn they
    leave empty says a call was omitted."""
    uses = {b['id'] for t in turns for b in t['content'] if b['type'] == 'tool_use'}
    results = {b['toolUseId'] for t in turns for b in t['content'] if b['type'] == 'tool_result'}
    keep = lambda b: b['id'] in results if b['type'] == 'tool_use' else b['toolUseId'] in uses if b['type'] == 'tool_result' else True
    return [{'participant': t['participant'], 'content': [b for b in t['content'] if keep(b)] or [OMITTED] * bool(t['content'])}
            for t in turns]


def header(turn: dict) -> str | None:
    """The summary id a recall pair's question turn names."""
    match turn:
        case {'participant': 'Context Manager', 'content': [{'type': 'text', 'text': text}]}:
            return m[1] if (m := _HEADER.fullmatch(text)) else None


def with_line(request: dict, line: str) -> dict:
    """withAppendedInstruction: `line` appended to the last block, the instruction's text."""
    *before, last = request['messages']
    *rest, final = last['content']
    return {**request, 'messages': before + [{**last, 'content': rest + [{**final, 'text': final['text'] + line}]}]}


def without_tools(request: dict) -> dict:
    return {k: v for k, v in request.items() if k != 'tools'}


# --- fallbacks ---

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


def input_tokens(response: dict) -> float | None:
    """compressionResponseInputTokens: the provider's count of the request's input, if it gave one above
    zero. Anthropic and Bedrock count cache reads and writes apart from it."""
    obj = lambda value: value if isinstance(value, dict) else {}
    details = obj(response.get('details'))
    usage = details['usage'] if isinstance(details.get('usage'), dict) else obj(response.get('usage'))
    read = lambda key: v if type(v := usage.get(key)) in (int, float) and v >= 0 else None   # JSON carries no infinity
    provider = obj(details.get('model')).get('provider')
    if (n := read('inputTokens')) is not None and isinstance(provider, str) and provider.lower() in ('anthropic', 'bedrock'):
        n += (read('cacheCreationTokens') or 0) + (read('cacheReadTokens') or 0)
    return n or None


# --- calls and replies ---

def call(host: Host, request: dict) -> Reply:
    """runAttempt's transport. A request the provider rejects for its reasoning blocks
    (isCarrierTransportRejection) goes once more without them."""
    try:
        return Reply(request, host.complete(request))
    except Exception as error:
        message, kind = str(error).lower(), getattr(error, 'type', None)
        rejected = getattr(error, 'status', None) == 400 or isinstance(kind, str) and 'invalid' in kind.lower() or 'invalid_request' in message
        if not (rejected and ('thinking' in message or 'reasoning' in message)
                and any(b['type'] in REASONING for t in request['messages'] for b in t['content'])):
            raise
    stripped = {**request, 'messages': [{**t, 'content': strip_reasoning(t['content'])} for t in request['messages']]}   # none empties
    return Reply(stripped, host.complete(stripped))


def written(state: State, level: int, sources: tuple, span: tuple, reply: Reply, cfg: Config) -> Entry:
    """The memory a reply wrote. Its tokens are the reply's output tokens, else chars/3; it keeps the
    reply's reasoning and text blocks if it reasoned (captureResponseContent), and the hash of the
    request the provider took."""
    out, content, text = (reply.response.get('usage') or {}).get('outputTokens'), reply.response['content'], reply.text
    return Entry(f'L{level}-{len(state.entries)}', level, text, out if out and out > 0 else math.ceil(js_len(text) / 3), sources, span,
                 blocks=tuple(b for b in content if b['type'] in REASONING + ('text',)) if any(b['type'] in REASONING for b in content) else None,
                 provenance=('end_turn', sha256(reply.sent), cfg.compression_model))


def well_formed(block) -> bool:
    """fallbackContentBlockError finds nothing wrong."""
    source = lambda s, url: isinstance(s, dict) and (url and s.get('type') == 'url' and isinstance(s.get('url'), str) or s.get(
        'type') == 'base64' and isinstance(s.get('data'), str) and isinstance(s.get('mediaType'), str))
    match block:
        case {'type': 'text', 'text': str()} | {'type': 'redacted_thinking', 'data': str()}: return True
        case {'type': 'thinking', 'thinking': str()}: return isinstance(block.get('signature', ''), str)
        case {'type': 'tool_use', 'id': str(), 'name': str(), 'input': dict()}: return True
        case {'type': 'tool_result', 'toolUseId': str(), 'content': str()}: return True
        case {'type': 'tool_result', 'toolUseId': str(), 'content': list() as nested}: return all(map(well_formed, nested))
        case {'type': 'image'}: return source(block.get('source'), True)
        case {'type': 'document' | 'audio' | 'video'}: return source(block.get('source'), False)
        case {'type': 'generated_image', 'data': str(), 'mimeType': str()}: return True
    return False
