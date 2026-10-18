//! JavaScript's semantics, where the library's numbers depend on them: lengths are UTF-16 code units,
//! and trim() and Math.round are JavaScript's. JSON is json.rs's compact form.

use crate::json::{to_string, Value};

/// String.prototype.length: an astral character is two code units.
pub fn js_len(s: &str) -> usize {
    s.bytes().map(|b| (b & 0xc0 != 0x80) as usize + (b >= 0xf0) as usize).sum()
}

pub fn is_js_space(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}'
        | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}

/// String.prototype.trim, which keeps U+0085 and the other spaces Rust's trim() takes.
pub fn js_trim(s: &str) -> &str {
    s.trim_matches(is_js_space)
}

/// Math.round: halves go up (12.5 -> 13, -12.5 -> -12).
pub fn js_round(x: f64) -> f64 {
    let whole = x.floor();
    whole + if x - whole >= 0.5 { 1.0 } else { 0.0 }
}

/// Python's round(): halves go to even.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// The UTF-16 length of the value's compact JSON.
pub fn json_len(v: &Value) -> usize {
    js_len(&to_string(v))
}

/// FNV-1a, 64 bits, of the value's compact JSON in UTF-8. (What's hashed is small: blocks' locations,
/// not their content.)
pub fn digest(v: &Value) -> String {
    let h = to_string(v).bytes().fold(0xcbf29ce484222325, |h: u64, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    #[test]
    fn rounding() {
        // Math.round and Python's round(), where they part: halves
        assert_eq!([12.5, -12.5, 0.5, 1.5, 2.4999].map(js_round), [13.0, -12.0, 1.0, 2.0, 2.0]);
        assert_eq!([12.5, -12.5, 0.5, 1.5, 2.4999].map(py_round), [12.0, -12.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn lengths_and_hashes() {
        // as the Python port's js_len, json_len and digest give them
        assert_eq!(js_len("é漢😀"), 4);
        let v = parse("{\"p\":\"a\\u00e9\\n\\ud83d\\ude00\",\"n\":[1,1.5,-0.25,null,true]}".as_bytes()).unwrap();
        assert_eq!(to_string(&v), "{\"p\":\"aé\\n😀\",\"n\":[1,1.5,-0.25,null,true]}");
        assert_eq!(json_len(&v), 42);
        assert_eq!(digest(&v), "c0b0cd2aa565bc3b");
    }
}
