//! Every rule of @animalabs/context-manager 0.10.1 that turns on a block's type: a block's store and
//! text-only prices, its image bytes, and its shape. Blocks are the library's JSON objects
//! ({"type": "text", "text": ...}); lengths are JavaScript's (js.rs).

use crate::js::{js_len, js_round, js_trim, json_len};
use crate::json::{Json, Value};

pub const PROSE: f64 = 2.9; // chars per token
pub const DENSE: f64 = 2.3; // also prices all tool JSON
pub const HIDDEN_THINKING: f64 = 600.0; // signed thinking whose text was removed; bare redacted_thinking

pub fn reasoning(kind: &str) -> bool {
    kind == "thinking" || kind == "redacted_thinking"
}

pub fn text_only_tokens(b: &Value) -> f64 {
    let chars = |s: &Value| (js_len(s.as_str().unwrap_or("")) as f64 / 4.0).ceil();
    match (b.kind(), b.get("text"), b.get("thinking"), b.get("content")) {
        ("text", Some(s), _, _) | ("thinking", _, Some(s), _) | ("tool_result", _, _, Some(s @ Value::String(_))) => chars(s),
        ("tool_use", ..) if b.get("input").is_some() => (json_len(&b["input"]) as f64 / 4.0).ceil() + 20.0,
        _ => 0.0,
    }
}

/// The uncalibrated store price.
pub fn raw_tokens(b: &Value) -> f64 {
    let stamped = b["tokenEstimate"].as_f64();
    let thinking = &b["thinking"];
    match b.kind() {
        "thinking" | "redacted_thinking" if stamped.is_some() => stamped.unwrap(),
        "thinking" if b["signature"].as_str().is_some_and(|s| !s.is_empty()) && thinking.as_str().is_none_or(str::is_empty) => HIDDEN_THINKING,
        "thinking" => density_tokens(thinking.as_str().unwrap_or("")),
        "text" if b.get("text").is_some() => density_tokens(b["text"].as_str().unwrap_or("")),
        "redacted_thinking" => match b["data"].as_str() {
            Some(data) if !data.is_empty() => js_round(js_len(data) as f64 / 6.0),
            _ => HIDDEN_THINKING,
        },
        "tool_use" if b.get("input").is_some() => (json_len(&b["input"]) as f64 / DENSE).ceil() + 20.0, // + name, id
        "tool_result" => match &b["content"] {
            Value::String(text) => (js_len(text) as f64 / DENSE).ceil(),
            Value::Array(blocks) => blocks.iter().map(raw_tokens).fold(0.0, |a, x| a + x),
            _ => 0.0,
        },
        "image" => b["tokenEstimate"].as_f64().unwrap_or(1600.0),
        "document" | "audio" | "video" => 1000.0,
        _ => 0.0,
    }
}

/// 2.3 chars/token when over 12% of the first 2000 code units are above U+007E or JSON/code
/// punctuation, otherwise 2.9.
pub fn density_tokens(text: &str) -> f64 {
    let prose = |units: usize, c: char| (units < 2000 && c < '\u{7f}' && !"{}[]\":_/=`".contains(c)) as usize;
    let (units, prose) = text.chars().fold((0, 0), |(units, n), c| (units + c.len_utf16(), n + prose(units, c)));
    let n = units.min(2000); // none: 0/0 isn't over 0.12, and no units cost nothing
    (units as f64 / if (n - prose) as f64 / n as f64 > 0.12 { DENSE } else { PROSE }).ceil()
}

/// An image block's base64 length (imageBlockBytes); 0 for anything else.
pub fn image_bytes(b: &Value) -> usize {
    b["source"]["data"].as_str().map_or(0, js_len)
}

/// Where a block's content is: in a record on disk (file, record id, block index, and the images
/// in it that stream as placeholders), or the block itself, for a block a plan makes.
#[derive(Clone, Debug, PartialEq)]
pub enum At {
    Disk(String, u64, usize, Vec<usize>),
    Itself(Value),
}

impl At {
    /// As the Python port writes it: [file, id, index, *dropped] or the block.
    pub fn value(&self) -> Value {
        match self {
            At::Disk(file, id, j, dropped) => [file.as_str().into(), Value::from(*id), Value::from(*j)].into_iter().chain(dropped.iter().map(|&k| k.into())).collect(),
            At::Itself(block) => block.clone(),
        }
    }
}

/// A block as a plan sees it: what the plan's rules read, and where the content is, which a
/// Reader reads only as the turns are written out.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub kind: String,
    pub tool: Option<String>, // a tool_use's id, a tool_result's toolUseId
    pub blank: bool,          // text with nothing visible
    pub images: Vec<usize>,   // base64 lengths: the image's own, or those a tool result holds
    pub at: At,
}

/// The Block for a block found `at` on disk, or for a block the plan makes, which carries itself.
pub fn shape(block: &Value, at: Option<At>) -> Block {
    let kind = block.kind();
    let images = match kind {
        "image" => vec![image_bytes(block)],
        _ => block["content"].arr().iter().filter(|b| b.kind() == "image").map(image_bytes).collect(),
    };
    let tool = match kind {
        "tool_use" => block["id"].as_str(),
        "tool_result" => block["toolUseId"].as_str(),
        _ => None,
    };
    Block {
        kind: kind.to_string(),
        tool: tool.map(str::to_string),
        blank: kind == "text" && js_trim(block["text"].as_str().unwrap_or("")).is_empty(),
        images,
        at: at.unwrap_or_else(|| At::Itself(block.clone())),
    }
}
