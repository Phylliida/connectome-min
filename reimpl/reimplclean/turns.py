"""Transforms of conversation turns ({'participant', 'content': [blocks]}) that the compile and the
summarizer's requests share, each reproducing one of the library's: splitting mixed tool turns,
joining a participant's consecutive turns, repairing or dropping unpaired tool blocks, normalizing
turns for a provider, stripping reasoning, and replacing images past a byte cap."""
from itertools import groupby

from blocks import REASONING, image_bytes
from js import js_trim
from segment import budget

OMITTED = {'type': 'text', 'text': '[tool call omitted]'}
STUB = '[tool result unavailable — omitted during context compression]'
IMAGE_PLACEHOLDER = {'type': 'text', 'text': '[image dropped from live context]'}
LOOKAHEAD = 6                             # entries searched for a displaced tool result
SEPARATOR = {'type': 'text', 'text': '\n\n---\n\n'}


def has(entry: dict, kind: str) -> bool:
    return any(block['type'] == kind for block in entry['content'])


def split(entry: dict, marked: bool) -> list[dict]:
    """ContextManager.compile's splitMixedToolMessages: a non-user turn carrying tool results becomes
    runs of its own blocks and user turns of results; a marked entry's last part carries the marker."""
    user = entry['participant'].lower() == 'user' or not has(entry, 'tool_result')
    parts = [(entry['participant'], entry['content'])] if user else \
        [('user' if result else entry['participant'], list(run))
         for result, run in groupby(entry['content'], lambda b: b['type'] == 'tool_result')]
    return [{'participant': p, 'content': c, **({'cacheBreakpoint': True} if marked and k == len(parts) - 1 else {})}
            for k, (p, c) in enumerate(parts)]


def collapse(turns: list[dict]) -> list[dict]:
    """collapseConsecutiveMessages: a participant's consecutive turns join, a rule between each."""
    return [{'participant': p, 'content': [b for k, t in enumerate(run) for b in [SEPARATOR] * (k > 0) + t['content']]}
            for p, run in groupby(turns, lambda t: t['participant'])]


def normalize(turns: list[dict]) -> list[dict]:
    """Turns as a provider takes them: mixed tool turns split, a participant's consecutive turns
    joined, tool pairing repaired (pair_tools), then blank text, empty turns and stale cache_control
    markers dropped."""
    joined = pair_tools(collapse([part for t in turns for part in split(t, False)]))
    return [{'participant': t['participant'], 'content': c} for t in joined if (c := [
        {k: v for k, v in b.items() if k != 'cache_control'} for b in t['content'] if b['type'] != 'text' or js_trim(b['text'])])]


def pair_tools(entries: list[dict]) -> list[dict]:
    """enforceToolPairing, in place: a tool_result must answer a tool_use in the entry just before it,
    once; others are dropped. A tool_use the next entry doesn't answer gets its result moved up from
    the next six entries, else a stub; results join that entry, or a new user entry before it."""
    uses, i = {}, 0
    while i < len(entries):
        entry, seen = entries[i], set()
        kept = [b for b in entry['content'] if b['type'] != 'tool_result'
                or (b['toolUseId'] in uses and b['toolUseId'] not in seen and not seen.add(b['toolUseId']))]
        if len(kept) != len(entry['content']):
            entry['content'] = kept or [OMITTED]
        answered = {b['toolUseId'] for b in entry['content'] if b['type'] == 'tool_result'}
        if missing := [u for u in uses if u not in answered]:
            results = [take_result(entries, i, u) for u in missing]
            if not answered:                   # the new entry is checked next, and answers them
                entries.insert(i, {'participant': 'user', 'content': results, 'source': None})
                continue
            entry['content'] = results + entry['content']
        uses = dict.fromkeys(b['id'] for b in entry['content'] if b['type'] == 'tool_use')
        i += 1
    if uses:
        entries.append({'participant': 'user', 'content': [stub(u) for u in uses], 'source': None})
    return entries


def take_result(entries: list[dict], i: int, use_id: str) -> dict:
    """relocateOrDropMissingResults: the result for `use_id` from the next LOOKAHEAD entries, else a
    stub. The library also removes it there; the entry left holding it drops it when checked."""
    return next((b for later in entries[i + 1:i + 1 + LOOKAHEAD] for b in later['content']
                 if b['type'] == 'tool_result' and b['toolUseId'] == use_id), stub(use_id))


def stub(use_id: str) -> dict:
    return {'type': 'tool_result', 'toolUseId': use_id, 'content': STUB}


def strip_reasoning(content: list) -> list:
    return [b for b in content if b['type'] not in REASONING]


def cap_images(turns: list[dict], cap: int) -> list[dict]:
    """capCompressionImageBytes: newest turn first, images, nested ones too, become placeholders once
    the bytes kept would pass `cap`; a cap of 0 or less keeps them all. (The library caps before the request drops blank text, empty
    turns and cache_control, none of which an image is.)"""
    admit = budget(0, cap)

    def fit(b: dict) -> dict:
        if b['type'] == 'image':
            return b if admit(image_bytes(b)) else IMAGE_PLACEHOLDER
        return {**b, 'content': list(map(fit, b['content']))} if isinstance(b.get('content'), list) else b   # a tool result's
    return [{**t, 'content': list(map(fit, t['content']))} for t in turns[::-1]][::-1]
