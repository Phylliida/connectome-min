//! What a message costs in @animalabs/context-manager 0.10.1, reproduced to the integer. A message
//! has two prices, each summed over its top-level blocks (blocks.rs):
//!   text_only  sizes chunks (estimateTextOnlyTokens): chars/4, uncalibrated, blind to tool_result block lists.
//!   store      prices the windows (MessageStore.estimateTokens): density-aware chars/token, each top-level
//!              block scaled by the calibration multiplier and rounded.

use crate::blocks::{raw_tokens, shape, text_only_tokens, At, Block};
use crate::js::js_round;
use crate::json::{Json, Value};

/// What plans read from a message: its prices and its blocks' shapes, never its content. Messages
/// never change, so this is computed once; the multiplier applies on read, as in the library's
/// per-block cache.
pub struct Priced {
    pub participant: String,
    pub blocks: Vec<Block>, // each top-level block, located in the log
    pub raw: Vec<f64>,      // raw_tokens of each top-level block
    pub text_only: f64,
}

impl Priced {
    pub fn has(&self, kind: &str) -> bool {
        self.blocks.iter().any(|b| b.kind == kind)
    }

    /// (raw_tokens, base64 length) of each top-level image.
    pub fn images(&self) -> impl Iterator<Item = (f64, usize)> + '_ {
        self.raw.iter().zip(&self.blocks).filter(|(_, b)| b.kind == "image").map(|(&r, b)| (r, b.images[0]))
    }
}

/// message: a log record, {id, participant, content: [blocks]}.
pub fn price(message: &Value) -> Priced {
    let content = message["content"].arr();
    Priced {
        participant: message["participant"].as_str().unwrap_or("").to_string(),
        blocks: content.iter().enumerate().map(|(j, b)| shape(b, Some(At::Disk("messages".into(), message.id(), j, vec![])))).collect(),
        raw: content.iter().map(raw_tokens).collect(),
        text_only: content.iter().map(text_only_tokens).fold(0.0, |a, x| a + x),
    }
}

/// Rounded per block, then summed: rounding the sum instead gives different totals.
pub fn store_tokens(p: &Priced, multiplier: f64) -> f64 {
    p.raw.iter().map(|r| js_round(r * multiplier)).fold(0.0, |a, x| a + x)
}

/// The store multiplier, learned from usage (reportRealInputTokens). A committed compile arms one
/// sample with the window total it rendered; only the next usage report counts, since later calls in
/// the turn send a grown request.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub multiplier: f64,
    pub armed: f64,
}

impl Calibration {
    /// real_tokens: input + cache-read + cache-write. A ratio outside 0.6..1.8 is a request the
    /// compile didn't build: it disarms without learning.
    pub fn observe(self, real_tokens: f64) -> Calibration {
        if !(real_tokens > 0.0 && self.armed > 0.0) {
            return self;
        }
        let (ratio, m) = (real_tokens / self.armed, self.multiplier);
        if !(0.6..=1.8).contains(&ratio) {
            return Calibration { multiplier: m, armed: 0.0 };
        }
        Calibration { multiplier: (m + 0.2 * (ratio * m - m)).clamp(0.6, 1.8), armed: 0.0 }
    }
}
