"""JavaScript's semantics, where the library's numbers depend on them: lengths are UTF-16 code units,
trim() and Math.round are JavaScript's, and JSON is JSON.stringify's text, which the library prices,
hashes and compares."""
import hashlib
import json
import math
import re
from decimal import Decimal

_LONE_SURROGATE = re.compile(r'[\ud800-\udfff]')         # json.dumps keeps these raw; JS writes \uXXXX
_INDEX = re.compile(r'0|[1-9][0-9]*')                    # an array-index key, if under 2**32 - 1
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


def js_number(x: float) -> str:
    """Number::toString, as JSON.stringify writes a JSON number: 1.0 is '1', 1e-7 is '1e-7', 1e21 is
    '1e+21', and an integer past 2**53 rounds as a double."""
    _, digits, exponent = Decimal(repr(x := float(x))).normalize().as_tuple()   # repr's digits are JS's
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


def sha256(value) -> str:
    """sha256Json."""
    return hashlib.sha256(js_json(value).encode()).hexdigest()
