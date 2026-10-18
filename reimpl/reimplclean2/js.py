"""JavaScript's semantics, where the library's numbers depend on them: lengths are UTF-16 code units,
and trim() and Math.round are JavaScript's. JSON is compact json.dumps, which matches JSON.stringify
except on odd floats, integer-like keys and lone surrogates."""
import json
import math

JS_SPACE = '\t\n\v\f\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'


def js_len(s: str) -> int:
    """String.prototype.length: an astral character is two code units."""
    return len(s.encode('utf-16-le', 'surrogatepass')) // 2


def js_trim(s: str) -> str:
    """String.prototype.trim, which keeps U+0085 and the other spaces Python's strip() takes."""
    return s.strip(JS_SPACE)


def js_round(x: float) -> int:
    """Math.round: halves go up, where Python's round() goes to even (12.5 -> 13, not 12)."""
    return (whole := math.floor(x)) + (x - whole >= 0.5)


def compact(value) -> str:
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def json_len(value) -> int:
    return js_len(compact(value))


def digest(value) -> str:
    """FNV-1a, 64 bits, of the value's compact JSON in UTF-8."""
    h = 0xcbf29ce484222325
    for byte in compact(value).encode('utf-8', 'surrogatepass'):
        h = (h ^ byte) * 0x100000001b3 & 0xffffffffffffffff
    return f'{h:016x}'
