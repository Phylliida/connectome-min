"""What a message costs in @animalabs/context-manager 0.10.1, reproduced to the integer.

Blocks are the library's dicts ({'type': 'text', 'text': ...}). Lengths are JavaScript string lengths
(UTF-16 code units) and rounding is Math.round, because the library's numbers are defined in those terms.
A message has two prices:
  text_only  sizes chunks (estimateTextOnlyTokens): chars/4, uncalibrated, blind to tool_result block lists.
  store      prices the windows (MessageStore.estimateTokens): density-aware chars/token, each top-level
             block scaled by the calibration multiplier and rounded.
"""
import json
import math
import re
from decimal import Decimal
from functools import reduce
from operator import add
from typing import NamedTuple

PROSE, DENSE = 2.9, 2.3             # chars per token; DENSE also prices all tool JSON
HIDDEN_THINKING = 600               # signed thinking whose text was removed; bare redacted_thinking
_PLAIN = re.compile(r'[^\x7f-\U0010ffff{}\[\]":_/=`]')   # code units the density probe counts as prose
_LONE_SURROGATE = re.compile(r'[\ud800-\udfff]')         # json.dumps keeps these raw; JS writes \uXXXX
_INDEX = re.compile(r'0|[1-9][0-9]*')                    # an array-index key, if under 2**32 - 1
_JS_SPACE = '\t\n\v\f\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
_LINK = re.compile(f'https?://[^{_JS_SPACE}]+')          # a bare link line, after JS's trim()


# --- JavaScript arithmetic ---

def js_len(s: str) -> int:
    """String.prototype.length: an astral character is two code units."""
    return len(s.encode('utf-16-le', 'surrogatepass')) // 2


def js_round(x: float) -> int:
    """Math.round: halves go up, where Python's round() goes to even (12.5 -> 13, not 12)."""
    return (whole := math.floor(x)) + (x - whole >= 0.5)


def js_number(x: float) -> str:
    """Number::toString, as JSON.stringify writes it: 1.0 is '1', 1e-7 is '1e-7', 1e21 is '1e+21'."""
    if not math.isfinite(x := float(x)): return 'null'
    if x == 0: return '0'
    _, digits, exponent = Decimal(repr(abs(x))).normalize().as_tuple()   # repr's digits are JS's
    d, k = ''.join(map(str, digits)), len(digits)
    n = exponent + k                                                     # |x| = 0.d1..dk * 10^n
    return '-' * (x < 0) + (d + '0' * (n - k) if k <= n <= 21 else      # 1200
                            d[:n] + '.' + d[n:] if 0 < n <= 21 else      # 1.2
                            '0.' + '0' * -n + d if -6 < n <= 0 else      # 0.0012
                            d[0] + '.' * (k > 1) + d[1:] + 'e' + '-+'[n > 1] + str(abs(n - 1)))   # 1.2e-7


def js_json(value) -> str:
    """JSON.stringify(value) for a JSON value. JavaScript lists an object's array-index keys first, in
    numeric order, and escapes lone surrogates."""
    match value:
        case None: return 'null'
        case True: return 'true'
        case False: return 'false'
        case str(): return _LONE_SURROGATE.sub(lambda c: f'\\u{ord(c[0]):04x}', json.dumps(value, ensure_ascii=False))
        case int() | float(): return js_number(value)
        case list() | tuple(): return f'[{",".join(map(js_json, value))}]'
        case dict():
            index = lambda key: (0, int(key)) if _INDEX.fullmatch(key) and int(key) < 2**32 - 1 else (1, 0)
            return f'{{{",".join(js_json(k) + ":" + js_json(value[k]) for k in sorted(value, key=index))}}}'
    raise TypeError(f'not a JSON value: {value!r}')


def json_len(value) -> int:
    return js_len(js_json(value))


# --- block prices ---

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


# --- messages ---

class Priced(NamedTuple):
    """What the windows and the chunker read from a message. Messages never change, so this is computed
    once; the multiplier applies on read, as in the library's per-block cache."""
    id: object
    kinds: frozenset                      # top-level block types
    raw: tuple                            # raw_tokens of each top-level block
    images: tuple                         # (raw_tokens, base64 length) of each top-level image
    text_only: int
    salience: float                       # static_salience, the planner's fold order


def price(message) -> Priced:
    """message: {'id': ..., 'content': [blocks]}."""
    content = message['content']
    raw = tuple(map(raw_tokens, content))
    images = tuple((r, image_bytes(b)) for r, b in zip(raw, content) if b['type'] == 'image')
    return Priced(message['id'], frozenset(b['type'] for b in content), raw, images,
                  sum(map(text_only_tokens, content)), static_salience(content))


def store_tokens(message: Priced, multiplier: float = 1.0) -> int:
    """Rounded per block, then summed: rounding the sum instead gives different totals."""
    return sum(js_round(r * multiplier) for r in message.raw)


def static_salience(content) -> float:
    """1 for conversation, down to 0.2 for content that exists elsewhere: fenced code, bare link
    lines, tool payloads, images (computeStaticSalience)."""
    total = external = 0
    for block in content:
        match block:
            case {'type': 'text'}:
                text = block.get('text') or ''
                total += js_len(text)
                external += sum(js_len(fence) for fence in text.split('```')[1::2])
                external += sum(js_len(s) for line in text.split('\n') if _LINK.fullmatch(s := line.strip(_JS_SPACE)))
            case {'type': 'tool_use'}:
                n = json_len({} if block.get('input') is None else block['input'])
                total, external = total + n, external + n
            case {'type': 'tool_result', 'content': str() as text}:
                total, external = total + js_len(text), external + js_len(text)
            case {'type': 'tool_result'}:
                n = json_len('' if block.get('content') is None else block['content'])
                total, external = total + n, external + n
            case {'type': 'image'}:
                total, external = total + 6400, external + 6400     # 1600 tokens' worth
            case _:
                total += js_len(block.get('text') or '')
    return 1 if total <= 0 else max(0.2, 1 - 0.8 * min(1, external / total))


def image_bytes(block) -> int:
    """An image block's base64 length (imageBlockBytes); 0 for anything else."""
    match block:
        case {'source': {'data': str() as data}}: return js_len(data)
    return 0


class Calibration(NamedTuple):
    """The store multiplier, learned from usage (reportRealInputTokens). A committed compile arms one
    sample with the window total it rendered; only the next usage report counts, since later calls in
    the turn send a grown request. The harness keeps the multiplier in memory, so it restarts at 1.0;
    Connectome persists it, clamps it to 0.6..1.8 on load, and applies it once the first compile has
    rebuilt its chunks."""
    multiplier: float = 1.0
    armed: float = 0

    def arm(self, window_tokens: float) -> 'Calibration':
        return self._replace(armed=window_tokens)

    def observe(self, real_tokens: float) -> 'Calibration':
        """real_tokens: input + cache-read + cache-write. A ratio outside 0.6..1.8 is a request the
        compile didn't build: it disarms without learning."""
        if not (math.isfinite(real_tokens) and real_tokens > 0 and self.armed > 0): return self
        ratio, m = real_tokens / self.armed, self.multiplier
        if not 0.6 <= ratio <= 1.8: return Calibration(m)
        return Calibration(min(1.8, max(0.6, m + 0.2 * (ratio * m - m))))   # JS's operation order
