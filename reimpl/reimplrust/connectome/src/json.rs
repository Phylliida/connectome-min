//! JSON: serde_json's values, keys in insertion order, the reads this crate makes of them, and a
//! compact writer that writes numbers as JavaScript does (1.0 as 1). serde_json escapes as Python's
//! json.dumps with ensure_ascii=False does (only `"`, `\` and control characters), so the lengths and
//! hashes taken over what the writer writes match the Python port's.

use serde::Serialize;
pub use serde_json::{Map, Value};
use std::io;

/// Reads that take whatever is there, as the library's JavaScript does.
pub trait Json {
    /// The items of an array; none for anything else.
    fn arr(&self) -> &[Value];
    /// A block's or record's "type".
    fn kind(&self) -> &str;
    fn u64(&self) -> u64;
    /// A record's id.
    fn id(&self) -> u64;
}

impl Json for Value {
    fn arr(&self) -> &[Value] {
        self.as_array().map_or(&[], Vec::as_slice)
    }

    fn kind(&self) -> &str {
        self["type"].as_str().unwrap_or("")
    }

    fn u64(&self) -> u64 {
        self.as_f64().unwrap_or(0.0) as u64
    }

    fn id(&self) -> u64 {
        self["id"].u64()
    }
}

/// An object, its keys in the order given, its values moved in: `obj! {"type" => "text", "text" => s}`.
#[macro_export]
macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {
        $crate::json::Value::Object($crate::json::Map::from_iter([$(($k.to_string(), $crate::json::Value::from($v))),*]))
    };
}

/// An object without its null fields: an optional field given as None is left out.
pub fn sparse(mut v: Value) -> Value {
    if let Value::Object(fields) = &mut v {
        fields.retain(|_, x| !x.is_null());
    }
    v
}

pub fn parse(bytes: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}

pub fn write(v: &(impl Serialize + ?Sized), out: impl io::Write) -> io::Result<()> {
    Ok(v.serialize(&mut serde_json::Serializer::with_formatter(out, Js))?)
}

pub fn to_string(v: &(impl Serialize + ?Sized)) -> String {
    let mut out = vec![];
    write(v, &mut out).expect("writing to memory");
    String::from_utf8(out).expect("JSON is UTF-8")
}

/// serde_json's compact form with Number.prototype.toString's numbers: decimals from 1e-6 up to
/// 1e21, exponents outside. (serde_json writes a value that isn't finite as null itself.)
pub struct Js;

impl serde_json::ser::Formatter for Js {
    fn write_f64<W: ?Sized + io::Write>(&mut self, out: &mut W, n: f64) -> io::Result<()> {
        let text = match n.abs() {
            0.0 => "0".into(),
            a if (1e-6..1e21).contains(&a) => format!("{n}"),
            a => format!("{n:e}").replacen('e', if a >= 1.0 { "e+" } else { "e" }, 1),
        };
        out.write_all(text.as_bytes())
    }
}
