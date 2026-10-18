"""Every rule of @animalabs/context-manager 0.10.1 that turns on a block's type, one table per
question: a block's store and text-only prices, and its image bytes. Blocks are the library's dicts
({'type': 'text', 'text': ...}); lengths are JavaScript's (js.py)."""
import math
import re
from functools import reduce
from operator import add

from js import js_len, js_round, json_len

PROSE, DENSE = 2.9, 2.3             # chars per token; DENSE also prices all tool JSON
HIDDEN_THINKING = 600               # signed thinking whose text was removed; bare redacted_thinking
_PLAIN = re.compile(r'[^\x7f-\U0010ffff{}\[\]":_/=`]')   # code units the density probe counts as prose
REASONING = ('thinking', 'redacted_thinking')


def text_only_tokens(block) -> int:
    match block:
        case {'type': 'text', 'text': s} | {'type': 'thinking', 'thinking': s} | {'type': 'tool_result', 'content': str() as s}:
            return math.ceil(js_len(s) / 4)
        case {'type': 'tool_use', 'input': value}: return math.ceil(json_len(value) / 4) + 20
    return 0


def raw_tokens(block) -> float:
    """The uncalibrated store price."""
    stamped, thinking = block.get('tokenEstimate'), block.get('thinking')
    match block:
        case {'type': 'thinking' | 'redacted_thinking'} if type(stamped) in (int, float): return stamped
        case {'type': 'thinking', 'signature': str() as signature} if signature and not thinking: return HIDDEN_THINKING
        case {'type': 'thinking'}: return density_tokens(thinking)
        case {'type': 'text', 'text': text}: return density_tokens(text)
        case {'type': 'redacted_thinking', 'data': str() as data} if data: return js_round(js_len(data) / 6)
        case {'type': 'redacted_thinking'}: return HIDDEN_THINKING
        case {'type': 'tool_use', 'input': value}: return math.ceil(json_len(value) / DENSE) + 20   # + name, id
        case {'type': 'tool_result', 'content': str() as text}: return math.ceil(js_len(text) / DENSE)
        case {'type': 'tool_result', 'content': list() as blocks}: return reduce(add, map(raw_tokens, blocks), 0)
        case {'type': 'image'}: return 1600 if stamped is None else stamped
        case {'type': 'document' | 'audio' | 'video'}: return 1000
    return 0


def density_tokens(text) -> int:
    """2.3 chars/token when over 12% of the first 2000 code units are above U+007E or JSON/code
    punctuation, otherwise 2.9."""
    if not text: return 0
    units = text.encode('utf-16-le', 'surrogatepass')
    n = min(len(units) // 2, 2000)
    prose = len(_PLAIN.findall(units[:4000].decode('utf-16-le', 'ignore')))   # ASCII: a split pair can't count
    return math.ceil(len(units) // 2 / (DENSE if (n - prose) / n > 0.12 else PROSE))


def image_bytes(block) -> int:
    """An image block's base64 length (imageBlockBytes); 0 for anything else."""
    match block:
        case {'source': {'data': str() as data}}: return js_len(data)
    return 0
