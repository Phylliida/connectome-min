//! The compiled context (AutobiographicalStrategy.select, then ContextManager.compile's tool-message
//! split), from the session's logs, in two passes.
//!
//! select plans from what messages and memories cost and their blocks' shapes (estimate::Priced,
//! blocks::Block), never their content: chunks close, the counter plans the frontier (planner.rs),
//! and the window is laid out as turns of blocks that say where their content is: each frontier
//! node's chunk raw or its memory's recall pair (a 'Context Manager' turn with the summary label, then
//! the memory as answer() renders it, as the agent); then the messages no chunk holds yet, raw. Passes
//! over the layout: trailing tool calls with no result are dropped, tool pairing is repaired, and up
//! to three cache breakpoints are placed. A compile over W plus the grace is refused; only a compile
//! that completes commits its frontier and arms calibration, and a refused one that couldn't fold
//! enough marks the session escalated, so the next L1 doesn't wait for newer chunks.
//!
//! A Reader then reads each block's content as the caller writes the turns out, so the context is
//! never in memory, only one record of it at a time.

use crate::{blocks::{reasoning, shape, At, Block}, chunking::{close_chunks, live_price, open_start, priced, stripped_raw, tail_start}, config::Config};
use crate::{estimate::Calibration, js::{digest, js_len, js_trim, py_round}, json::{Json, Value}, obj, planner::{chunks, plan, Node}, segment::budget};
use crate::{memory::Recorded, store::Session, turns::{pair_tools, say, split, text, Turn, DROPPED}};
use std::{cell::RefCell, collections::HashMap, ops::Range};

/// What a compile fits the context to, in tokens: past `over` it folds, down to `to` where it can, and
/// at its most folded past `limit` it's refused; and if `reach`, refused as well when it folds and can't
/// get down to `to`.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub over: f64,
    pub to: f64,
    pub limit: f64,
    pub reach: bool,
}

impl Fit {
    /// The library's for a budget W: folding past W, down to W * (1 - slack), refused past W plus the
    /// grace.
    pub fn library(w: f64, cfg: &Config) -> Fit {
        Fit { over: w, to: w * (1.0 - cfg.slack), limit: (w * (1.0 + cfg.over_budget_grace)).floor(), reach: false }
    }
}

/// An OverBudgetError: the compile renders nothing and commits nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub stage: &'static str,
    pub actual: f64,
    pub budget: f64,
    /// What the context at its most folded holds, how many and their tokens: memories, and chunks
    /// raw that have no L1 yet; then the tokens of the messages no chunk holds yet, outside the tail
    /// and in it.
    pub memories: (usize, f64),
    pub unformed: (usize, f64),
    pub open: f64,
    pub tail: f64,
}

/// The plan: the turns of the context, their blocks located.
pub fn select(session: &Session, fit: &Fit, calibration: Calibration, cfg: &Config) -> Result<Vec<Turn>, Refusal> {
    let (m, last) = (calibration.multiplier, session.last("compiles").unwrap_or(obj! {}));
    close_chunks(session, m, cfg);
    let held = |n: u64| session.count(&format!("layer{n}"));
    let memo = RefCell::new(HashMap::new());
    let cost = |node: Node| {
        *memo.borrow_mut().entry(node).or_insert_with(|| match node {
            (0, j) => py_round(session.read("layer0", j)["raw"].as_f64().unwrap() * m),
            (n, p) => session.read(&format!("layer{n}"), p)["recall"].as_f64().unwrap(),
        })
    };
    let shown: Vec<Node> = last["frontier"].arr().iter().map(|n| (n.arr()[0].u64(), n.arr()[1].u64())).collect();
    let groups = Recorded::new(session, cfg);
    let covered = shown.last().map_or(0, |&node| chunks(node, &groups).end);
    let frontier: Vec<Node> = shown.into_iter().chain((covered..held(0)).map(|j| (0, j))).collect();
    let (tail, loose) = (tail_start(session, m, cfg), open_start(session)..session.count("messages"));
    let (mut admit, live) = (budget(cfg.max_live_images, cfg.max_live_image_bytes), tail..loose.end);
    let price = |i| {
        let p = priced(session, i);
        if live.contains(&i) { live_price(&p, m, &mut admit) } else { py_round(stripped_raw(&p) * m) }
    };
    let prices: Vec<f64> = loose.clone().rev().map(price).collect(); // newest first, as the live image limits count
    let fixed = prices.iter().fold(0.0, |a, x| a + x);
    let carried = fixed + frontier.iter().map(|&n| cost(n)).fold(0.0, |a, x| a + x);
    let frontier = plan(frontier, cost, |(n, p)| p < held(n), fixed, fit.over, fit.to, &groups, cfg);
    let total = fixed + frontier.iter().map(|&n| cost(n)).fold(0.0, |a, x| a + x);
    let limit = if fit.reach && carried > fit.over { fit.to } else { fit.limit };
    let refusal = |stage| {
        let held = |raw: bool| frontier.iter().filter(|n| (n.0 == 0) == raw).fold((0, 0.0), |(k, t), &n| (k + 1, t + cost(n)));
        let tail = prices[..(loose.end - tail) as usize].iter().fold(0.0, |a, x| a + x);
        Refusal { stage, actual: total, budget: limit, memories: held(false), unformed: held(true), open: fixed - tail, tail }
    };
    if total > limit {
        let mut record = last.clone();
        record["multiplier"] = m.into();
        record["armed"] = calibration.armed.into();
        record["escalated"] = true.into();
        session.append("compiles", record);
        return Err(refusal("Adaptive picker exhausted"));
    }

    let mut entries: Vec<Turn> = frontier.iter().flat_map(|&node| render(session, node, cfg)).chain(messages(session, loose.clone(), &live, cfg)).collect();
    let trailing = entries.iter().rev().take_while(|e| e.has("tool_use") && !e.has("tool_result")).count();
    entries.truncate(entries.len() - trailing);
    let entries = pair_tools(entries);
    if !loose.is_empty() && !entries.iter().any(|e| e.source == Some(loose.end - 1)) {
        return Err(refusal("Structural repair did not retain the newest turn"));
    }
    let keys: Vec<String> = entries.iter().map(|e| digest(&Value::Array(vec![e.participant.as_str().into(), located(&e.content)]))).collect();
    // placeCacheMarkers: the end of the prefix unchanged since the previous compile (before any, the last
    // entry ahead of the tail), and the last entry. An entry's key is where its blocks are, which says
    // what they hold, since records never change.
    let stable = match last["keys"].as_array() {
        None => entries.iter().position(|e| e.source.is_some_and(|s| s >= tail)).unwrap_or(keys.len()),
        Some(previous) => keys.iter().zip(previous).take_while(|(a, b)| b.as_str() == Some(a)).count(),
    };
    let marks = [keys.len().checked_sub(1), stable.checked_sub(1)];
    let nodes: Vec<Value> = frontier.iter().map(|&(n, p)| vec![n, p].into()).collect();
    session.append("compiles", obj! {"frontier" => nodes, "multiplier" => m, "armed" => total, "keys" => keys, "escalated" => total > fit.over});
    Ok(entries.iter().enumerate().flat_map(|(i, e)| split(e, marks.contains(&Some(i)))).collect())
}

/// Blocks as where they are: what keys and request hashes are taken over.
pub fn located(content: &[Block]) -> Value {
    Value::Array(content.iter().map(|b| b.at.value()).collect())
}

/// A frontier node's entries: its chunk's messages, or its memory's recall pair.
fn render(session: &Session, (n, p): Node, cfg: &Config) -> Vec<Turn> {
    if n == 0 {
        let chunk = session.read("layer0", p);
        return messages(session, chunk["first"].u64()..chunk["last"].u64() + 1, &(0..0), cfg);
    }
    vec![
        say(&cfg.summary_label),
        Turn::new(&cfg.summary_participant, recalled(&session.read(&format!("layer{n}"), p), n), None),
    ]
}

/// Messages as entries: images past the live limits, newest first, or outside the tail, become
/// placeholders.
fn messages(session: &Session, ids: Range<u64>, live: &Range<u64>, cfg: &Config) -> Vec<Turn> {
    let mut admit = budget(cfg.max_live_images, cfg.max_live_image_bytes);
    let mut turns: Vec<Turn> = ids.rev().map(|i| {
        let p = priced(session, i);
        let content = p.blocks.iter().map(|b| if b.kind != "image" || live.contains(&i) && admit(b.images[0]) { b.clone() } else { text(DROPPED) });
        Turn::new(&p.participant, content.collect(), Some(i))
    }).collect();
    turns.reverse(); // read newest first, as the live image limits count
    turns
}

/// A memory's recall answer under carrierPolicy 'live-strip' (liveWindowAnswerProse): the reply's
/// stored blocks without their reasoning, when a text block among them has visible text; else the
/// memory's text.
pub fn answer(memory: &Value) -> Vec<Value> {
    let stripped: Vec<Value> = memory["blocks"].arr().iter().filter(|b| !reasoning(b.kind())).cloned().collect();
    if stripped.iter().any(|b| !js_trim(b["text"].as_str().unwrap_or("")).is_empty()) {
        return stripped;
    }
    vec![obj! {"type" => "text", "text" => memory["text"].clone()}]
}

/// The blocks of a memory's answer, located in its layer.
pub fn recalled(memory: &Value, level: u64) -> Vec<Block> {
    answer(memory).iter().enumerate().map(|(j, b)| shape(b, Some(At::Disk(format!("layer{level}"), memory.id(), j, vec![])))).collect()
}

/// A recall pair's price (recallPairCost): its label plus each of its answer's blocks, chars/4.
pub fn recall_tokens(answer: &[Value], cfg: &Config) -> f64 {
    let quarter = |s: &str| (js_len(s) as f64 / 4.0).ceil();
    quarter(&cfg.summary_label) + answer.iter().map(|b| quarter(b["text"].as_str().unwrap_or(""))).fold(0.0, |a, x| a + x)
}

/// Per file: the id read last, its record, and the scan it came from.
type Scan<'s> = (Option<u64>, Value, Box<dyn Iterator<Item = Value> + 's>);

/// Reads planned blocks' content from disk, for records read mostly in order, as a stream reads them:
/// per file, the record read last is kept, the next one continues its scan, and any other starts a
/// scan. scrub drops the blocks' stale cache_control markers.
pub struct Reader<'s> {
    session: &'s Session,
    scrub: bool,
    scans: HashMap<String, Scan<'s>>,
}

impl<'s> Reader<'s> {
    pub fn new(session: &'s Session, scrub: bool) -> Reader<'s> {
        Reader { session, scrub, scans: HashMap::new() }
    }

    fn read(&mut self, file: &str, i: u64) -> &Value {
        let session = self.session;
        let (last, record, scan) = self.scans.entry(file.to_string()).or_insert_with(|| (None, Value::Null, Box::new(std::iter::empty())));
        if *last != Some(i) {
            if last.map(|l| l + 1) != Some(i) {
                *scan = session.span(file, i, session.count(file) - 1);
            }
            *record = scan.next().unwrap_or_else(|| panic!("no record {i} in {file}"));
            *last = Some(i);
        }
        record
    }

    /// A block's content: itself, or read from where it is, with the images it drops as placeholders.
    pub fn load(&mut self, b: &Block) -> Value {
        let (file, i, j, dropped) = match &b.at {
            At::Itself(block) => return block.clone(),
            At::Disk(file, i, j, dropped) => (file.as_str(), *i, *j, dropped),
        };
        let record = self.read(file, i);
        let mut block = if file == "messages" { record["content"].arr()[j].clone() } else { answer(record).swap_remove(j) };
        if let Some(Value::Array(nested)) = block.get_mut("content") {
            for (k, c) in nested.iter_mut().filter(|c| c.kind() == "image").enumerate() {
                if dropped.contains(&k) {
                    *c = obj! {"type" => "text", "text" => DROPPED};
                }
            }
        }
        if let (true, Some(fields)) = (self.scrub, block.as_object_mut()) {
            fields.shift_remove("cache_control");
        }
        block
    }
}
