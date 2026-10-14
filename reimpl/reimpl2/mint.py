"""Memory formation in @animalabs/context-manager 0.10.1 (AutobiographicalStrategy.tick, with
compressChunkHierarchical for L1s and executeMerge above), reproduced exactly in default mode.

A tick writes one memory: an L1 for the compression queue's next chunk or, when that queue is empty,
the merge queue's first merge.

An L1:
  1. A chunk of text blocks only, under min_chunk_chars of trimmed text, gets a fixed stub; no call.
  2. The request replays what the agent saw before the chunk: the head window raw; every unmerged
     summary, oldest first, as a recall pair (a 'Context Manager' turn naming it, then its answer as
     the agent), less the oldest that don't fit the recall budget; the raw messages between the head
     and the chunk that no L1 covers; then a marker turn, the chunk, and the instruction. Mixed tool
     turns are split, consecutive turns by one participant joined, tool blocks whose partner is
     nowhere in the request dropped, then blank text and empty turns. Cache breakpoints go after the
     turn before the first recall pair, the newest pair above L1 and the newest pair, unless the
     budget left some out.
  3. The tick defers, without a call, while the host has declared no tools and the request carries
     tool blocks or the summarizer is a Fable/Mythos model.
  4. A chunk whose request failed before, or that has failed three requests, is skipped.
  5. The calls. On tool_use, again with a no-tools line, then without tools. On any other ending but
     end_turn, variants that each expand one recall pair into its children's, newest first: the first
     curve_fallbacks that fit the context budget, until one returns valid text; when none does, the
     request is quarantined. A call the transport refuses for its reasoning blocks is retried once
     without them. Any other error from a variant counts as its failure; from the request or its
     tool_use retries, it escapes the tick and the chunk waits for the next rebuild.
  6. Text that was all <thinking> asks once more for plain prose; text still empty is quarantined.
  7. The memory is stored as L1-n with the reply's blocks when they carry reasoning, and the chunk's
     quarantine clears.

A merge consolidates a run of unmerged summaries of one level into one a level up. After every
memory, each level queues at most one: once merge_threshold of its summaries are unmerged and not
already queued, the first run of them adjacent in chunk order that has merge_threshold members (its
first ones) or, with a newer run after it, two or more; never a quarantined run. The request is the head,
recall pairs for what precedes the run (on a budget of at least 8000), the raw middle, then the run
one level deeper (an L1's messages raw, a higher summary's children as recall pairs) and the
instruction; the image byte cap applies to it. One call per tick. A reply that isn't a complete
end_turn with text, or a non-retryable provider error, counts against the merge, and so does every
server error from the twelfth on; other errors escape the tick. The next attempt shows the run itself
if any reply was a refusal, adds the no-tools line after a tool_use and the prose line after an empty
reply, drops the tools once two attempts failed the last on a tool_use, and halves the recall budget
(four times at most). At merge_attempt_limit the run is quarantined until another merge takes one of
its members.

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
from dataclasses import dataclass, replace
from functools import reduce
from itertools import chain, groupby, islice, pairwise
from typing import Callable, NamedTuple

import chunking
from config import Config
from emit import IMAGE_PLACEHOLDER, OMITTED, split
from estimate import _JS_SPACE, image_bytes, js_json, js_len, js_round

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
REASONING = ('thinking', 'redacted_thinking')
STOPS = {'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'abort'}
SHAPE_CAP = 3                             # distinct failed requests per chunk before it is skipped for good
SERVER_STREAK = 12                        # server errors in a row that count as one failed merge attempt
_HEADER = re.compile(r'\[CM\] Recall memory ([^\n\r\u2028\u2029]+)\.')
_TOOLS_LESS = re.compile(r'(?:^|/)claude-(?:fable|mythos)-')
_WRAPPED = re.compile(rf'[{_JS_SPACE}]*<thinking>.*?</thinking>[{_JS_SPACE}]*', re.I | re.A | re.S)
_OPENS, _CLOSES = re.compile(rf'[{_JS_SPACE}]*<thinking>', re.I | re.A), re.compile('</thinking>', re.I | re.A)


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
    """A merge queue entry and its failed attempts so far."""
    level: int                            # the level it writes
    sources: tuple                        # the run's ids, in chunk order
    attempts: int = 0
    stop: str | None = None               # lastStopReason
    outcome: str | None = None            # lastOutcome
    refused: bool = False                 # hadRefusal: some attempt's reply was a refusal
    streak: int = 0                       # serverErrorStreak


@dataclass(frozen=True)
class Memory:
    entries: tuple = ()                   # Entry, in creation order: the n of L<k>-n counts the ones before it
    queue: tuple = ()                     # chunk indices: l1_queue as of the last rebuild, less what ticks took
    quarantine: frozenset = frozenset()   # (chunk, request JSON) for each request that exhausted its fallbacks
    merges: tuple = ()                    # Merge, in the order they run
    merge_quarantine: frozenset = frozenset()   # source runs that exhausted their attempts


class Host(NamedTuple):
    """The host's side of a call (StrategyContext)."""
    complete: Callable                    # request -> response, as membrane normalizes them; raises on failure,
                                          # with the HTTP status as .status, the kind as .type, and .retryable
    tools: list | None = None             # the agent's live tool definitions, once declared
    system: str | None = None             # the host's system prompt, once declared


def js_trim(s: str) -> str:
    return s.strip(_JS_SPACE)


def sha256(value) -> str:
    """sha256Json."""
    return hashlib.sha256(js_json(value).encode()).hexdigest()


def without_tools(request: dict) -> dict:
    return {k: v for k, v in request.items() if k != 'tools'}


def summarized(memory: Memory, chunks: list) -> set[int]:
    """Indices of the chunks with an L1, for chunking.l1_queue."""
    l1s = {s.sources for s in memory.entries if s.level == 1}
    return {i for i, chunk in enumerate(chunks) if chunk in l1s}


def tick(messages: list[dict], priced: list, chunks: list, memory: Memory, multiplier: float, host: Host,
         cfg: Config) -> tuple[Memory, Exception | None]:
    """AutobiographicalStrategy.tick. Rebuilds set memory.queue to chunking.l1_queue(chunks,
    summarized(memory, chunks), ...); ticks between rebuilds take its following entries. An error the
    library's tick throws comes back beside the memory: an L1's chunk has left the queue and nothing
    else changed; a merge stays queued, with a server error counted."""
    memory, head = sweep(memory, chunks, cfg), chunking.head_end(priced, multiplier, cfg)
    if not memory.queue:
        return merge(messages, head, chunks, memory, host, cfg) if memory.merges else (memory, None)
    index, memory = memory.queue[0], replace(memory, queue=memory.queue[1:])
    try:
        return mint(messages, head, chunks, index, memory, host, cfg), None
    except Exception as error:
        return memory, error


def mint(messages: list[dict], head: int, chunks: list, index: int, memory: Memory, host: Host, cfg: Config) -> Memory:
    """compressChunkHierarchical for chunk `index`; `head` is chunking.head_end."""
    position, ids = {m['id']: i for i, m in enumerate(messages)}, chunks[index]
    chunk = [messages[position[i]] for i in ids]
    blocks = [b for m in chunk for b in m['content']]
    if cfg.min_chunk_chars > 0 and all(b['type'] == 'text' for b in blocks) \
            and sum(js_len(js_trim(b['text'])) for b in blocks) < cfg.min_chunk_chars:
        return remember(memory, Entry(f'L1-{len(memory.entries)}', 1, QUIET.format(len(ids)), 40, ids, (ids[0], ids[-1])), chunks, cfg)

    # The request: what the agent saw, then the chunk.
    covered = {m for s in memory.entries if s.level == 1 for m in s.sources}
    prior = sorted((s for s in memory.entries if s.parent is None), key=lambda s: position[s.span[0]])
    kept = recall(prior, cfg.recall_budget_tokens)
    turns = prepare(raw(messages[:head], covered) + pairs(kept, cfg.summary_participant) + raw(messages[head:position[ids[0]]], covered)
                    + [say(MARKER)] + raw(chunk) + [say(INSTRUCTION.format(cfg.summary_target_tokens))])
    if not host.tools and (_TOOLS_LESS.search(cfg.compression_model) or any(
            b['type'] in ('tool_use', 'tool_result') for t in turns for b in t['content'])):
        return memory
    request = compose(turns, kept, len(kept) < len(prior), host, cfg)

    family = (ids, js_json(request))
    if family in memory.quarantine or sum(q[0] == ids for q in memory.quarantine) >= SHAPE_CAP:
        return memory
    sent, response = call(host, request)
    if response.get('stopReason') == 'tool_use':
        retry, reply = call(host, lined := with_line(request, NO_TOOLS))
        if reply.get('stopReason') == 'tool_use':
            retry, reply = call(host, without_tools(lined))
        if reply.get('stopReason') == 'end_turn':
            sent, response = retry, reply
    if response.get('stopReason') != 'end_turn':
        for variant in admitted(request, variants(request, kept, memory.entries, position), input_tokens(response), cfg):
            try:
                sent, response = call(host, variant)
            except Exception:
                continue
            if assess(response)[0] == 'valid':
                break
        else:
            return replace(memory, quarantine=memory.quarantine | {family})
    if not js_trim(text := extract(response)):
        retry, reply = call(host, with_line(request, PLAIN_PROSE))
        if reply.get('stopReason') != 'end_turn' or not js_trim(text := extract(reply)):
            return replace(memory, quarantine=memory.quarantine | {family})
        sent, response = retry, reply
    return remember(memory, written(memory, 1, ids, (ids[0], ids[-1]), sent, response, text, cfg), chunks, cfg)


def remember(memory: Memory, entry: Entry, chunks: list, cfg: Config) -> Memory:
    """Store an L1, clear its chunk's quarantine, and queue the merges it completes."""
    return enqueue_merges(replace(memory, entries=memory.entries + (entry,),
                                  quarantine=frozenset(q for q in memory.quarantine if q[0] != entry.sources)), chunks, cfg)


def merge(messages: list[dict], head: int, chunks: list, memory: Memory, host: Host, cfg: Config) -> tuple[Memory, Exception | None]:
    """executeMerge for the merge queue's head, and the accounting of its failure as tick does it."""
    job, position, nodes = memory.merges[0], {m['id']: i for i, m in enumerate(messages)}, {s.id: s for s in memory.entries}
    sources = [nodes[i] for i in job.sources]
    under, start = {m for s in sources for m in leaves(s, nodes)}, position[sources[0].span[0]]
    prior = sorted((s for s in memory.entries if s.parent is None and under.isdisjoint(s.sources) and position[s.span[0]] < start),
                   key=lambda s: position[s.span[0]])
    kept = recall(prior, max(8000, js_round(cfg.recall_budget_tokens * 0.5 ** min(job.attempts, 4))))
    covered = {m for s in memory.entries if s.level == 1 for m in s.sources}

    # The run one level deeper, or itself once a reply was refused. Only a rejection sets stop, outcome
    # and refused, so each implies a retry.
    level, who = sources[0].level, cfg.summary_participant
    target = (pairs(sources, who) if job.refused else raw(messages[position[m]] for s in sources for m in s.sources) if level == 1
              else pairs((nodes[c] for s in sources for c in s.sources), who))
    shown = level if job.refused else level - 1
    instruction = MERGE_INSTRUCTION.format(level=job.level, target=cfg.summary_target_tokens, seen=f'the L{shown} memories above'
                                           if shown else 'the slices of recent experience above (raw conversation)') \
        + NO_TOOLS * (job.stop == 'tool_use') + PLAIN_PROSE * (job.outcome == 'unusable_empty')
    turns = cap_images(prepare(raw(messages[:head], covered) + pairs(kept, who) + raw(messages[head:start], covered)
                               + target + [say(instruction)]), cfg.compression_image_bytes)
    request = compose(turns, kept, len(kept) < len(prior), host, cfg)
    if job.stop == 'tool_use' and job.attempts >= 2:
        request = without_tools(request)

    try:
        sent, response = call(host, request)
    except Exception as error:
        retryable, kind = getattr(error, 'retryable', None), getattr(error, 'type', None)
        server = retryable is True and kind == 'server'
        memory = replace(memory, merges=(job := job._replace(streak=job.streak + server),) + memory.merges[1:])
        if retryable is False and isinstance(kind, str) or server and job.streak >= SERVER_STREAK:
            return reject(memory, 'provider_error', None, cfg), None
        return memory, error
    outcome, stop = assess(response)
    if outcome == 'valid' and not js_trim(text := extract(response)):
        outcome, stop = 'unusable_empty', 'end_turn'
    if outcome != 'valid':
        return reject(memory, outcome, stop, cfg), None
    entry = written(memory, job.level, job.sources, (sources[0].span[0], sources[-1].span[1]), sent, response, text, cfg)
    entries = tuple(s._replace(parent=entry.id) if s.id in job.sources else s for s in memory.entries) + (entry,)
    return enqueue_merges(replace(memory, entries=entries, merges=memory.merges[1:]), chunks, cfg), None


def reject(memory: Memory, outcome: str, stop: str | None, cfg: Config) -> Memory:
    """recordMergeRejection for the queue's head: count the failed attempt; at merge_attempt_limit,
    quarantine the run."""
    job = (old := memory.merges[0])._replace(attempts=old.attempts + 1, stop=old.stop if stop is None else stop,
                                             outcome=outcome, refused=old.refused or stop == 'refusal')
    if job.attempts >= max(1, cfg.merge_attempt_limit):
        return replace(memory, merges=memory.merges[1:], merge_quarantine=memory.merge_quarantine | {job.sources})
    return replace(memory, merges=(job,) + memory.merges[1:])


def written(memory: Memory, level: int, sources: tuple, span: tuple, sent: dict, response: dict, text: str, cfg: Config) -> Entry:
    """The memory a reply wrote. Its tokens are the reply's output tokens, else chars/3; it keeps the
    reply's reasoning and text blocks if it reasoned (captureResponseContent), and the hash of the
    request the provider took."""
    out, content = (response.get('usage') or {}).get('outputTokens'), response['content']
    return Entry(f'L{level}-{len(memory.entries)}', level, text, out if out and out > 0 else math.ceil(js_len(text) / 3), sources, span,
                 blocks=tuple(b for b in content if b['type'] in REASONING + ('text',)) if any(b['type'] in REASONING for b in content) else None,
                 provenance=('end_turn', sha256(sent), cfg.compression_model))


def sweep(memory: Memory, chunks: list, cfg: Config) -> Memory:
    """sweepPaidOffMergeQuarantine, before every tick: a quarantined run that another merge took a
    member of can never run again, so it is dropped and the merges are requeued."""
    unmerged = {s.id for s in memory.entries if s.parent is None}
    stale = {run for run in memory.merge_quarantine if not unmerged.issuperset(run)}
    return enqueue_merges(replace(memory, merge_quarantine=memory.merge_quarantine - stale), chunks, cfg) if stale else memory


def enqueue_merges(memory: Memory, chunks: list, cfg: Config) -> Memory:
    """checkMergeThresholdRecursive: for each level, a merge over the run that contiguousMergeCandidates
    picks among its unmerged, unqueued summaries, unless that run is quarantined."""
    seq, queued = {m: k for k, m in enumerate(chain.from_iterable(chunks))}, {s for m in memory.merges for s in m.sources}
    runs = (candidates([s for s in memory.entries if s.level == level and s.parent is None and s.id not in queued], seq, cfg)
            for level in range(1, max((s.level for s in memory.entries), default=0) + 1))
    return replace(memory, merges=memory.merges + tuple(
        Merge(level, run) for level, run in enumerate(runs, 2) if run and run not in memory.merge_quarantine))


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


def leaves(s: Entry, nodes: dict) -> tuple:
    """The messages under a summary, in chunk order."""
    return s.sources if s.level == 1 else tuple(chain.from_iterable(leaves(nodes[c], nodes) for c in s.sources))


# --- building requests ---

def say(text: str) -> dict:
    return {'participant': CM, 'content': [{'type': 'text', 'text': text}]}


def raw(messages, covered=frozenset()) -> list[dict]:
    """Messages as the request replays them: without their reasoning, unless `covered` has them."""
    return [{'participant': m['participant'], 'content': strip_reasoning(m['content'])} for m in messages if m['id'] not in covered]


def recall(prior: list[Entry], budget: float) -> list[Entry]:
    """capRecallPairs: newest first, every summary that still fits the budget at its tokens plus 50."""
    fit = lambda acc, s: ([s, *acc[0]], acc[1] + s.tokens + 50) if acc[1] + s.tokens + 50 <= budget else acc
    return reduce(fit, reversed(prior), ([], 0))[0]


def pairs(summaries, participant: str) -> list[dict]:
    """Recall pairs: a turn naming each summary, then its answer as `participant`."""
    return [t for s in summaries for t in (say(f'[CM] Recall memory {s.id}.'), {'participant': participant, 'content': answer(s)})]


def prepare(turns: list[dict]) -> list[dict]:
    """splitMixedToolMessages, collapseConsecutiveMessages, stripUnpairedToolBlocks."""
    return drop_unpaired(collapse([part for t in turns for part in split(t, False)]))


def compose(turns: list[dict], kept: list[Entry], capped: bool, host: Host, cfg: Config) -> dict:
    """The request: blank text, empty turns and stale cache_control dropped, then the cache seams."""
    wire = [{'participant': t['participant'], 'content': c} for t in turns if (c := [
        {k: v for k, v in b.items() if k != 'cache_control'} for b in t['content'] if b['type'] != 'text' or js_trim(b['text'])])]
    marks = set() if capped else seams(wire, {s.id: s.level for s in kept})
    max_tokens = max(16000, js_round(cfg.summary_target_tokens * 1.5))
    if (cfg.compression_max_tokens or 0) > 0:
        max_tokens = min(max_tokens, cfg.compression_max_tokens)
    return {**({'system': host.system} if host.system else {}), 'shedOversizeImages': True,
            'messages': [{**t, 'cacheBreakpoint': True} if i in marks else t for i, t in enumerate(wire)],
            **({'cacheTtl': '1h'} if marks else {}), 'config': {'model': cfg.compression_model, 'maxTokens': max_tokens},
            **({'tools': host.tools} if host.tools is not None else {})}


def seams(wire: list[dict], levels: dict) -> set[int]:
    """applyMintCacheSeams: the turns a breakpoint follows: the one before the first recall pair, the
    newest pair above L1's answer, and the newest pair's answer. None without pairs. (The library
    scans all but the last turn, the instruction, which never names a summary.)"""
    found = [(i, levels[sid]) for i, t in enumerate(wire) if (sid := header(t)) in levels]
    return {found[0][0] - 1, found[-1][0] + 1, max((i + 1 for i, level in found if level >= 2), default=-1)} - {-1} if found else set()


def cap_images(turns: list[dict], cap: int) -> list[dict]:
    """capCompressionImageBytes: newest turn first, images, nested ones too, become placeholders once
    the bytes kept would pass `cap`."""
    kept = 0

    def fit(b: dict) -> dict:
        nonlocal kept
        if b['type'] == 'image':
            if fits := kept + image_bytes(b) <= cap:
                kept += image_bytes(b)
            return b if fits else IMAGE_PLACEHOLDER
        return {**b, 'content': list(map(fit, b['content']))} if b['type'] == 'tool_result' and isinstance(b.get('content'), list) else b
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


# --- fallbacks ---

def variants(request: dict, kept: list[Entry], entries, position: dict) -> list[dict]:
    """buildRecallCurveVariants: for each recalled summary above L1, newest first, the request with its
    pair replaced by its children's. The summary's messages must ascend in the log, its pair must
    appear exactly once, unjoined, and the result must still answer each tool block in the adjacent
    turn.

    The library also sorts each node's children by position, then requires its messages to ascend and
    its range to name the first and last. A merge takes a run adjacent in chunk order and names its
    range after the run's ends, and chunk order departs from log order only where a shrinking head
    exposes earlier messages to later chunks, which then precede the run's first; so a run passes
    exactly when it is already in log order."""
    nodes, wire, out = {s.id: s for s in entries}, request['messages'], []
    for parent in sorted((s for s in kept if s.level > 1), key=lambda s: -position[s.span[1]]):
        at = [i for i in range(len(wire) - 1) if header(wire[i]) == parent.id and wire[i + 1]['content'] == answer(parent)]
        if len(at) != 1 or any(position[a] > position[b] for a, b in pairwise(leaves(parent, nodes))):
            continue
        turns = wire[:at[0]] + pairs((nodes[c] for c in parent.sources), wire[at[0] + 1]['participant']) + wire[at[0] + 2:]
        if well_paired(turns):
            out.append({**request, 'messages': turns})
    return out


def well_paired(turns: list[dict]) -> bool:
    """validateRecallCurveRequest: no blank text, and every tool block's partner in the adjacent turn.
    (Turns are never empty: compose drops those, and a recall pair has blocks.)"""
    has = lambda j, kind, key, value: 0 <= j < len(turns) and any(b['type'] == kind and b.get(key) == value for b in turns[j]['content'])
    return all((b['type'] != 'text' or js_trim(b['text']))
               and (b['type'] != 'tool_use' or has(i + 1, 'tool_result', 'toolUseId', b['id']))
               and (b['type'] != 'tool_result' or has(i - 1, 'tool_use', 'id', b['toolUseId']))
               for i, t in enumerate(turns) for b in t['content'])


def admitted(request: dict, options: list[dict], provider: float | None, cfg: Config) -> list[dict]:
    """compressionRefusalPlan: the first curve_fallbacks variants whose input plus output reserve fits the
    context budget. Input is the provider's count for the request plus the variant's growth, or without
    a count, the variant's serialized bytes plus a per-turn reserve."""
    bound = lambda r: len(js_json(r).encode()) + 512 + 128 * len(r['messages'])
    fits = lambda v: (bound(v) if provider is None else provider + max(1, bound(v) - bound(request))) \
        + request['config']['maxTokens'] <= cfg.context_budget_tokens
    return list(islice(filter(fits, options), cfg.curve_fallbacks))


def input_tokens(response: dict) -> float | None:
    """compressionResponseInputTokens: the provider's count of the request's input, if it gave one above
    zero. Anthropic and Bedrock count cache reads and writes apart from it."""
    obj = lambda value: value if isinstance(value, dict) else {}
    details = obj(response.get('details'))
    usage = details['usage'] if isinstance(details.get('usage'), dict) else obj(response.get('usage'))
    read = lambda key: v if type(v := usage.get(key)) in (int, float) and 0 <= v < math.inf else None
    provider = obj(details.get('model')).get('provider')
    if (n := read('inputTokens')) is not None and isinstance(provider, str) and provider.lower() in ('anthropic', 'bedrock'):
        n += (read('cacheCreationTokens') or 0) + (read('cacheReadTokens') or 0)
    return n or None


def call(host: Host, request: dict) -> tuple[dict, dict]:
    """runAttempt's transport: (the request the provider took, its response)."""
    try:
        return request, host.complete(request)
    except Exception as error:                     # isCarrierTransportRejection, when the request carries any
        message, kind = str(error).lower(), getattr(error, 'type', None)
        rejected = getattr(error, 'status', None) == 400 or isinstance(kind, str) and 'invalid' in kind.lower() or 'invalid_request' in message
        if not (rejected and ('thinking' in message or 'reasoning' in message)
                and any(b['type'] in REASONING for t in request['messages'] for b in t['content'])):
            raise
    request = {**request, 'messages': [t for t in ({**t, 'content': strip_reasoning(t['content'])} for t in request['messages']) if t['content']]}
    return request, host.complete(request)


# --- replies ---

def assess(response: dict) -> tuple[str, str | None]:
    """assessFallbackCompressionResponse: (outcome, stopReason). Only 'valid', a complete end_turn of
    well-formed blocks with text, may become a memory."""
    stop, content = s if isinstance(s := response.get('stopReason'), str) else None, response.get('content')
    if stop not in STOPS or not isinstance(content, list) or not all(map(well_formed, content)):
        return 'provider_error', stop
    if stop == 'refusal':
        return 'refusal', stop
    if not js_trim('\n'.join(b['text'] for b in content if b['type'] == 'text')):
        return 'unusable_empty', stop
    return 'valid' if stop == 'end_turn' else 'incomplete', stop


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


def extract(response: dict) -> str:
    """The memory's text: the reply's text blocks, less a leading <thinking>…</thinking>
    (stripThinkingPreamble); a reply that opens one and never closes it was all thinking."""
    text = '\n'.join(b.get('text') or '' for b in response['content'] if b['type'] == 'text')
    if m := _WRAPPED.match(text):
        return text[m.end():]
    return '' if _OPENS.match(text) and not _CLOSES.search(text) else text
