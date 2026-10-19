//! Chunks and the raw tail, read from the end of the log (the library's rebuildChunks, without its head
//! window). Messages outside the tail close into chunks, in log order: a run is full once it reaches
//! target_chunk_tokens of text-only tokens over at least four messages and doesn't end on a tool_use,
//! whose result comes next; a run that isn't full stays open. A message in a chunk never re-enters the
//! tail. Images count only in the tail, newest first within the live limits; elsewhere a placeholder
//! stands for each.

use crate::{config::Config, estimate::{price, store_tokens, Priced}, obj, segment::{budget, runs}, store::Session};
use std::rc::Rc;

/// ceil(len('[image dropped from live context]') / 4)
pub const PLACEHOLDER: f64 = 9.0;

/// The tail's first message: walking back from the newest, the messages until one takes the total
/// past recent_window_tokens, which is outside; a tool_result first in the tail brings its call in.
pub fn tail_start(session: &Session, multiplier: f64, cfg: &Config) -> u64 {
    let (floor, n, mut admit, mut total) = (open_start(session), session.count("messages"), budget(cfg.max_live_images, cfg.max_live_image_bytes), 0.0);
    let outside = (floor..n).rev().find(|&i| {
        total += live_price(&priced(session, i), multiplier, &mut admit);
        total > cfg.recent_window_tokens
    });
    outside.map_or(floor, |i| i + 1 - (i + 1 < n && priced(session, i + 1).has("tool_result")) as u64)
}

/// The first message no chunk holds.
pub fn open_start(session: &Session) -> u64 {
    session.last("layer0").map_or(0, |last| last["last"].as_f64().unwrap() as u64 + 1)
}

/// Append the chunks that close now: the open run, from the last chunk to the tail, in full runs.
pub fn close_chunks(session: &Session, multiplier: f64, cfg: &Config) {
    let p = |i| priced(session, i);
    let full = |run: &[u64], total: f64| total >= cfg.target_chunk_tokens && run.len() >= 4 && !p(*run.last().unwrap()).has("tool_use");
    for run in runs(open_start(session), tail_start(session, multiplier, cfg), |i| p(i).text_only, full) {
        let raw = run.iter().map(|&i| stripped_raw(&p(i))).fold(0.0, |a, x| a + x);
        session.append("layer0", obj! {"first" => run[0], "last" => *run.last().unwrap(), "raw" => raw});
    }
}

/// estimate::price of logged message i, kept while recently used (messages never change); the
/// message itself isn't kept.
pub fn priced(session: &Session, i: u64) -> Rc<Priced> {
    let mut memo = session.priced.borrow_mut();
    if memo.len() >= 4096 && !memo.contains_key(&i) {
        memo.clear();
    }
    memo.entry(i).or_insert_with(|| Rc::new(price(&session.read("messages", i)))).clone()
}

/// A tail message's price: images past the live limits count as placeholders (uncalibrated, as in
/// the library).
pub fn live_price(p: &Priced, multiplier: f64, admit: &mut impl FnMut(usize) -> bool) -> f64 {
    let dropped: f64 = p.images().filter(|&(_, size)| !admit(size)).map(|(raw, _)| (raw - PLACEHOLDER).max(0.0)).fold(0.0, |a, x| a + x);
    store_tokens(p, multiplier) - dropped
}

/// A message's uncalibrated price outside the tail, every image a placeholder.
pub fn stripped_raw(p: &Priced) -> f64 {
    p.raw.iter().fold(0.0, |a, x| a + x) - p.images().map(|(raw, _)| (raw - PLACEHOLDER).max(0.0)).fold(0.0, |a, x| a + x)
}
