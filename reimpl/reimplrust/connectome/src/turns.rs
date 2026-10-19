//! Transforms of planned turns that the compile and the summarizer's requests share, each reproducing
//! one of the library's: splitting mixed tool turns, joining a participant's consecutive turns,
//! repairing or dropping unpaired tool blocks, normalizing turns for a provider, stripping reasoning,
//! and replacing images past a byte cap. They move blocks by where they are and add blocks of their
//! own; none reads content.

use crate::{blocks::{reasoning, shape, At, Block}, json::{sparse, Value}, obj, segment::budget};
use itertools::Itertools;

pub const LABEL: &str = "Context Manager";
pub const STUB: &str = "[tool result unavailable — omitted during context compression]";
pub const DROPPED: &str = "[image dropped from live context]";
pub const LOOKAHEAD: usize = 6; // entries searched for a displaced tool result

/// A text block the plan makes.
pub fn text(s: &str) -> Block { shape(&obj! {"type" => "text", "text" => s}, None) }

/// A turn of the Context Manager's.
pub fn say(s: &str) -> Turn { Turn::new(LABEL, vec![text(s)], None) }

/// A turn of a plan: whose it is, its blocks, the message it came from, and whether a cache
/// breakpoint follows it.
#[derive(Clone, Debug)]
pub struct Turn {
    pub participant: String,
    pub content: Vec<Block>,
    pub source: Option<u64>,
    pub mark: bool,
}

impl Turn {
    pub fn new(participant: &str, content: Vec<Block>, source: Option<u64>) -> Turn {
        Turn { participant: participant.to_string(), content, source, mark: false }
    }

    pub fn has(&self, kind: &str) -> bool { self.content.iter().any(|b| b.kind == kind) }

    /// As the library writes it: {participant, content, cacheBreakpoint?}, each block as `block` gives it.
    pub fn value(&self, block: impl FnMut(&Block) -> Value) -> Value {
        let content: Vec<Value> = self.content.iter().map(block).collect();
        sparse(obj! {"participant" => self.participant.as_str(), "content" => content, "cacheBreakpoint" => self.mark.then_some(true)})
    }
}

/// ContextManager.compile's splitMixedToolMessages: a non-user turn carrying tool results becomes
/// runs of its own blocks and user turns of results; a marked entry's last part carries the marker.
pub fn split(entry: &Turn, marked: bool) -> Vec<Turn> {
    let result = |b: &Block| b.kind == "tool_result";
    let mut parts: Vec<Turn> = if entry.participant.eq_ignore_ascii_case("user") || !entry.has("tool_result") {
        vec![Turn::new(&entry.participant, entry.content.clone(), None)]
    } else {
        let run = |run: &[Block]| Turn::new(if result(&run[0]) { "user" } else { &entry.participant }, run.to_vec(), None);
        entry.content.chunk_by(|a, b| result(a) == result(b)).map(run).collect()
    };
    parts.last_mut().expect("a part").mark = marked;
    parts
}

/// collapseConsecutiveMessages: a participant's consecutive turns join, a rule between each.
pub fn collapse(turns: &[Turn]) -> Vec<Turn> {
    let joined = |run: &[Turn]| run.iter().map(|t| t.content.as_slice()).collect::<Vec<_>>().join(&text("\n\n---\n\n"));
    turns.chunk_by(|a, b| a.participant == b.participant).map(|run| Turn::new(&run[0].participant, joined(run), None)).collect()
}

/// Turns as a provider takes them: mixed tool turns split, a participant's consecutive turns
/// joined, tool pairing repaired (pair_tools), then blank text and empty turns dropped. (Stale
/// cache_control markers in the blocks go as they're read.)
pub fn normalize(turns: Vec<Turn>) -> Vec<Turn> {
    let joined = pair_tools(collapse(&turns.iter().flat_map(|t| split(t, false)).collect::<Vec<_>>()));
    let unblank = |t: Turn| Turn::new(&t.participant, t.content.into_iter().filter(|b| !b.blank).collect(), None);
    joined.into_iter().map(unblank).filter(|t| !t.content.is_empty()).collect()
}

/// enforceToolPairing: a tool_result must answer a tool_use in the entry just before it, once;
/// others are dropped, and an entry left empty says so. A tool_use the next entry doesn't answer gets
/// its result moved up from the next six entries, else a stub; results join that entry, or a new user
/// entry before it if it answers none.
pub fn pair_tools(entries: Vec<Turn>) -> Vec<Turn> {
    let (mut out, mut uses): (Vec<Turn>, Vec<Option<String>>) = (vec![], vec![]);
    for (i, entry) in entries.iter().enumerate() {
        let mut answered = vec![];
        let answers = |b: &&Block| b.kind != "tool_result" || uses.contains(&b.tool) && !answered.contains(&b.tool) && { answered.push(b.tool.clone()); true };
        let mut content: Vec<Block> = entry.content.iter().filter(answers).cloned().collect();
        if content.is_empty() && !entry.content.is_empty() {
            content = vec![text("[tool call omitted]")];
        }
        let results: Vec<Block> = uses.iter().filter(|u| !answered.contains(u)).map(|u| take_result(&entries, i, u)).collect();
        if answered.is_empty() && !results.is_empty() {
            out.push(Turn::new("user", results, None));
        } else {
            content.splice(0..0, results);
        }
        uses = content.iter().filter(|b| b.kind == "tool_use").map(|b| b.tool.clone()).unique().collect();
        out.push(Turn { content, ..entry.clone() });
    }
    if !uses.is_empty() {
        out.push(Turn::new("user", uses.iter().map(stub).collect(), None));
    }
    out
}

/// relocateOrDropMissingResults: the result for `use_id` from the next LOOKAHEAD entries, else a
/// stub. The library also removes it there; the entry left holding it drops it when checked. (The
/// library edits entries in place, but only up to the one it checks, so the entries after it are
/// the input's.)
fn take_result(entries: &[Turn], i: usize, use_id: &Option<String>) -> Block {
    let mut later = entries[i + 1..entries.len().min(i + 1 + LOOKAHEAD)].iter().flat_map(|later| &later.content);
    later.find(|b| b.kind == "tool_result" && b.tool == *use_id).cloned().unwrap_or_else(|| stub(use_id))
}

pub fn stub(use_id: &Option<String>) -> Block { shape(&obj! {"type" => "tool_result", "toolUseId" => use_id.as_deref(), "content" => STUB}, None) }

pub fn strip_reasoning(content: &[Block]) -> Vec<Block> { content.iter().filter(|b| !reasoning(&b.kind)).cloned().collect() }

/// capCompressionImageBytes: newest turn first, images, a tool result's too, become placeholders
/// once the bytes kept would pass `cap`; a cap of 0 keeps them all. A tool result keeps its place
/// and says which of its images the reader drops. (The library caps before the request drops blank
/// text, empty turns and cache_control, none of which an image is.)
pub fn cap_images(mut turns: Vec<Turn>, cap: usize) -> Vec<Turn> {
    let mut admit = budget(0, cap);
    for b in turns.iter_mut().rev().flat_map(|t| t.content.iter_mut()) {
        let dropped: Vec<usize> = (0..b.images.len()).filter(|&k| !admit(b.images[k])).collect();
        if b.kind == "image" && !dropped.is_empty() {
            *b = text(DROPPED);
        } else if let (false, At::Disk(.., at)) = (b.kind == "image", &mut b.at) {
            at.extend(dropped);
        }
    }
    turns
}
