"""Memory formation (AutobiographicalStrategy.tick, simplified): at most one memory per tick, in order,
so the logs' tails say what is due:
  - the next chunk's L1, once l1_holdback_chunks newer chunks have closed, or at once when the last
    compile couldn't fold enough (escalated);
  - else the next merge at the lowest level: layer n+1's entry p, once layer n has entries p*k ..
    p*k+k-1.
A memory asks the summarizer what the agent saw before the stretch, replayed (a recall pair for each
unmerged memory the recall budget fits, newest first), then the stretch and an instruction. A
text-only chunk under min_chunk_chars gets a stub without a call. Requests are built to go through:
the agent's tools stay (its cache prefix) with toolChoice 'none', recall answers carry no reasoning,
and images past compression_image_bytes become placeholders. A request is planned as emit.select's
context is, and its messages stream to the host (emit.stream).

Two outcomes besides a memory: a refusal stops memory formation, all of it, until surgery (and so,
eventually, folding: the window then overflows, as it should); any other failure is retried, at most
attempt_limit times until the log grows.
"""
import math
import re
from functools import reduce
from itertools import count, takewhile
from typing import Callable, NamedTuple

from blocks import REASONING, Block, shape
from chunking import priced
from config import Config
from emit import LABEL, answer, recall_tokens, recalled, stream
from js import JS_SPACE, digest, js_len, js_round, js_trim
from store import Session
from turns import cap_images, normalize, strip_reasoning

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
QUIET = ('(A quiet stretch: {} messages of routine system traffic — heartbeats, empty turns, notices. '
         'Nothing happened worth remembering.)')
_HEADER = re.compile(r'\[CM\] Recall memory ([^\n\r  ]+)\.')
_WRAPPED = re.compile(rf'[{JS_SPACE}]*<thinking>.*?</thinking>[{JS_SPACE}]*', re.I | re.A | re.S)
_OPENS = re.compile(rf'[{JS_SPACE}]*<thinking>', re.I | re.A)


class Host(NamedTuple):
    """The host's side of a call."""
    complete: Callable                    # request -> response {stopReason, content, usage}; raises on failure.
                                          # The request's messages, and each one's content, are iterators
    tools: list | None = None             # the agent's live tool definitions, once declared
    system: str | None = None             # the host's system prompt, once declared


def tick(session: Session, host: Host, cfg: Config) -> tuple[bool, Exception | None]:
    """Form the memory that's due, if it may be tried: (whether a call was made, the error if it failed)."""
    if (target := due(session, cfg)) is None or blocked(session, target, cfg):
        return False, None
    level, p = target
    if level == 1 and (stub := quiet(session, p, cfg)):
        session.append('layer1', stub)
        return False, None
    if (request := l1_request(session, p, host, cfg) if level == 1 else merge_request(session, level, p, host, cfg)) is None:
        return False, None                                    # deferred until the host declares its tools
    failed = lambda outcome: session.append('attempts', {'target': target, 'messages': session.count('messages'), 'outcome': outcome})
    try:
        response = host.complete({**request, 'messages': stream(session, request['messages'], scrub=True)})
    except Exception as error:
        failed('retry')
        return True, error
    if response.get('stopReason') == 'refusal':
        failed('refusal')
    elif text := usable(response):
        session.append(f'layer{level}', written(session, level, p, request, response, text, cfg))
    else:
        failed('retry')
    return True, None


def due(session: Session, cfg: Config) -> tuple[int, int] | None:
    """The memory to form next: the next L1, else the lowest level's next merge."""
    k, closed, l1s = cfg.merge_threshold, session.count('layer0'), session.count('layer1')
    if l1s < closed - (0 if (session.last('compiles') or {}).get('escalated') else cfg.l1_holdback_chunks):
        return 1, l1s
    for n in count(1):
        if (have := session.count(f'layer{n}')) < k:
            return None
        if (p := session.count(f'layer{n + 1}')) * k + k <= have:
            return n + 1, p


def blocked(session: Session, target: tuple, cfg: Config) -> bool:
    """Memory formation stops at a refusal: nothing is tried after one, so it stays the attempts log's
    last record until surgery appends another. A target also waits for the log to grow once it failed
    attempt_limit times at the log's length."""
    if (last := session.last('attempts')) and last['outcome'] == 'refusal':
        return True
    now = session.count('messages')
    return sum(a['target'] == list(target) for a in takewhile(lambda a: a['messages'] == now, session.backwards('attempts'))) >= cfg.attempt_limit


def quiet(session: Session, p: int, cfg: Config) -> dict | None:
    """The stub for a chunk of text blocks only under min_chunk_chars of trimmed text."""
    first, last = span(session, 1, p, cfg)
    blocks = [b for m in session.span('messages', first, last) for b in m['content']]
    if all(b['type'] == 'text' for b in blocks) and sum(js_len(js_trim(b['text'])) for b in blocks) < cfg.min_chunk_chars:
        return memory(first, last, QUIET.format(last - first + 1), None, 40, None, cfg)
    return None


def l1_request(session: Session, p: int, host: Host, cfg: Config) -> dict | None:
    """compressChunkHierarchical's request for chunk p, or None to defer it while the host has declared
    no tools and the request carries tool blocks, which a provider takes only with their tools."""
    first, last = span(session, 1, p, cfg)
    task = [say(MARKER), *raw(session, first, last), say(INSTRUCTION.format(cfg.summary_target_tokens))]
    request = ask(session, first, task, host, cfg)
    if not host.tools and any(b.type in ('tool_use', 'tool_result') for t in request['messages'] for b in t['content']):
        return None
    return request


def merge_request(session: Session, level: int, p: int, host: Host, cfg: Config) -> dict:
    """executeMerge's request for layer `level`'s entry p: the run one level deeper (an L1 run's
    messages raw, a higher run's children as recall pairs), then the merge instruction."""
    (k, below), (first, last) = (cfg.merge_threshold, level - 1), span(session, level, p, cfg)
    shown = raw(session, first, last) if below == 1 else \
        pairs([(below - 1, m) for m in session.span(f'layer{below - 1}', p * k * k, (p + 1) * k * k - 1)], cfg)
    seen = f'the L{below - 1} memories above' if below > 1 else 'the slices of recent experience above (raw conversation)'
    return ask(session, first, shown + [say(MERGE_INSTRUCTION.format(seen=seen, level=level, target=cfg.summary_target_tokens))], host, cfg)


def ask(session: Session, before: int, task: list[dict], host: Host, cfg: Config) -> dict:
    """The request for `task`: recall pairs, oldest first, for the newest unmerged memories before
    message `before` that fit the recall budget at their tokens plus 50 (capRecallPairs); normalized
    for the provider, images capped; cache seams follow unless the budget left a memory out."""
    prior = unmerged(session, before, cfg)
    kept = reduce(lambda acc, m: ([m, *acc[0]], acc[1] + m[1]['tokens'] + 50) if acc[1] + m[1]['tokens'] + 50 <= cfg.recall_budget_tokens
                  else acc, reversed(prior), ([], 0))[0]
    wire = cap_images(normalize(pairs(kept, cfg) + task), cfg.compression_image_bytes)
    marks = seams(wire, {f'L{level}-{m["id"]}': level for level, m in kept}) if len(kept) == len(prior) else set()
    cap = c if (c := cfg.compression_max_tokens or 0) > 0 else math.inf
    return {**({'system': host.system} if host.system else {}),
            'messages': [{**t, 'cacheBreakpoint': True} if i in marks else t for i, t in enumerate(wire)],
            **({'cacheTtl': '1h'} if marks else {}),
            'config': {'model': cfg.compression_model, 'maxTokens': min(max(16000, js_round(cfg.summary_target_tokens * 1.5)), cap)},
            **({'tools': host.tools, 'toolChoice': 'none'} if host.tools is not None else {})}


def unmerged(session: Session, before: int, cfg: Config) -> list[tuple[int, dict]]:
    """(level, memory) for each memory no merge has taken that ends before message `before`, oldest first."""
    k, out = cfg.merge_threshold, []
    for n in count(1):
        if (have := session.count(f'layer{n}')) == 0:
            break
        out += [(n, m) for m in session.span(f'layer{n}', session.count(f'layer{n + 1}') * k, have - 1) if m['last'] < before]
    return sorted(out, key=lambda lm: lm[1]['first'])


def pairs(memories: list[tuple[int, dict]], cfg: Config) -> list[dict]:
    """Recall pairs: a turn naming each memory, then its answer as the agent."""
    return [t for level, m in memories for t in (say(f'[CM] Recall memory L{level}-{m["id"]}.'),
                                                  {'participant': cfg.summary_participant, 'content': recalled(m, level)})]


def seams(wire: list[dict], levels: dict) -> set[int]:
    """applyMintCacheSeams: the turns a breakpoint follows: the one before the first recall pair, the
    newest pair above L1's answer, and the newest pair's answer. None without pairs."""
    found = [(i, levels[mid]) for i, t in enumerate(wire) if (mid := header(t)) in levels]
    return {found[0][0] - 1, found[-1][0] + 1, max((i + 1 for i, level in found if level >= 2), default=-1)} - {-1} if found else set()


def header(turn: dict) -> str | None:
    """The memory a recall pair's question turn names."""
    match turn:
        case {'participant': 'Context Manager', 'content': [Block(type='text', at={'text': str() as text})]}:
            return m[1] if (m := _HEADER.fullmatch(text)) else None


def say(text: str) -> dict:
    return {'participant': LABEL, 'content': [shape({'type': 'text', 'text': text})]}


def raw(session: Session, first: int, last: int) -> list[dict]:
    """Messages first..last as a request replays them: without their reasoning."""
    return [{'participant': p.participant, 'content': strip_reasoning(p.blocks)} for p in (priced(session, i) for i in range(first, last + 1))]


def usable(response: dict) -> str:
    """A finished reply's memory text: its text blocks, less a leading <thinking>…</thinking>
    (stripThinkingPreamble); '' for a reply cut short, malformed, empty, or all thinking."""
    content = response.get('content')
    if response.get('stopReason') != 'end_turn' or not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
        return ''
    text = '\n'.join(b['text'] for b in content if b.get('type') == 'text' and isinstance(b.get('text'), str))
    text = text[m.end():] if (m := _WRAPPED.match(text)) else '' if _OPENS.match(text) else text
    return text if js_trim(text) else ''


def written(session: Session, level: int, p: int, request: dict, response: dict, text: str, cfg: Config) -> dict:
    """The memory a reply wrote: its text, its reasoning and text blocks if it reasoned
    (captureResponseContent), its output tokens (else chars/3), and the hash of its request as
    planned."""
    (first, last), content = span(session, level, p, cfg), response['content']
    out = usage.get('outputTokens') if isinstance(usage := response.get('usage'), dict) else None
    blocks = [b for b in content if b['type'] in REASONING + ('text',)] if any(b['type'] in REASONING for b in content) else None
    return memory(first, last, text, blocks, out if out and out > 0 else math.ceil(js_len(text) / 3),
                  {'stopReason': 'end_turn', 'requestHash': digest({**request, 'messages': [{**t, 'content': [b.at for b in t['content']]} for t in request['messages']]}),
                   'model': cfg.compression_model}, cfg)


def span(session: Session, level: int, p: int, cfg: Config) -> tuple[int, int]:
    """The first and last message of the stretch layer `level`'s entry p covers: chunk p's for an L1,
    else from its first child's to its last's."""
    if level == 1:
        return (chunk := session.get('layer0', p))['first'], chunk['last']
    k = cfg.merge_threshold
    return session.get(f'layer{level - 1}', p * k)['first'], session.get(f'layer{level - 1}', p * k + k - 1)['last']


def memory(first: int, last: int, text: str, blocks: list | None, tokens: int, provenance: dict | None, cfg: Config) -> dict:
    record = {'first': first, 'last': last, 'text': text, 'blocks': blocks, 'tokens': tokens, 'provenance': provenance}
    return {**record, 'recall': recall_tokens(answer(record), cfg)}
