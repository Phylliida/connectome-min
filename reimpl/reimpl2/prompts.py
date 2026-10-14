"""The summarizer's requests (ask), as compressChunkHierarchical and executeMerge build them: what
the agent saw before the slice in question, replayed, then the slice and an instruction; and the
recall-curve variants an L1 falls back to (variants, admitted)."""
import math
import re
from functools import reduce
from itertools import chain, islice, pairwise
from typing import Iterable, Iterator, NamedTuple

from calls import Host
from config import Config
from js import js_json, js_round, js_trim
from state import Entry, State
from turns import collapse, drop_unpaired, split, strip_reasoning

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
_HEADER = re.compile(r'\[CM\] Recall memory ([^\n\r\u2028\u2029]+)\.')


class Context(NamedTuple):
    """What a tick reads besides the state (StrategyContext)."""
    position: dict                        # message id -> its index in the log
    head: int                             # messages in the head window (chunking.head_end)
    host: Host
    cfg: Config


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


def answer(s: Entry) -> list:
    """summaryAnswerContent: the reply's blocks, reasoning included, or the memory as one text block."""
    return list(s.blocks) if s.blocks else [{'type': 'text', 'text': s.content}]


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


def leaves(s: Entry, nodes: dict) -> Iterable:
    """The messages under a summary, in chunk order."""
    return s.sources if s.level == 1 else chain.from_iterable(leaves(nodes[c], nodes) for c in s.sources)


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
