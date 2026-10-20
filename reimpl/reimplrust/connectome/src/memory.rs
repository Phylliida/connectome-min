//! Memory formation (AutobiographicalStrategy.tick, simplified): at most one memory per tick, in order,
//! so the logs' tails say what is due:
//!   - the next chunk's L1, once l1_holdback_chunks newer chunks have closed, or at once when the last
//!     compile couldn't fold enough (escalated);
//!   - else the next merge at the lowest level: layer n+1's next entry, once the entries of layer n no
//!     memory has merged yet have closed a group (closed): the fewest, at least two, that hold
//!     target_chunk_tokens, as chunks close; or under merge_by_count, as the library merges, k of
//!     them (k = merge_threshold). A merged memory's record says what it merged (Recorded).
//!
//! A memory asks the summarizer what the agent saw before the stretch, replayed (a recall pair for each
//! unmerged memory the recall budget fits, newest first), then the stretch and an instruction. A
//! text-only chunk under min_chunk_chars gets a stub without a call. Requests are built to go through:
//! the agent's tools stay (its cache prefix) with toolChoice 'none', recall answers carry no reasoning,
//! and images past compression_image_bytes become placeholders. A request is planned as emit::select's
//! context is, and its messages are read from disk as the host writes them out (emit::Reader).
//!
//! Two outcomes besides a memory: a refusal stops memory formation, all of it, until surgery (and so,
//! eventually, folding: the window then overflows, as it should); any other failure is retried, at most
//! attempt_limit times until the log grows. A request its host says was stopped (INTERRUPTED, not the
//! library's) is neither: as if it wasn't made.

use crate::{blocks::{reasoning, At, Block}, chunking::priced, config::Config, emit::{answer, recall_tokens, recalled, Reader}, js::{digest, is_js_space, js_len, js_round, js_trim}};
use crate::{json::{sparse, Json, Value}, obj, planner::{chunks, Fixed, Groups, Node}, store::Session, turns::{cap_images, normalize, say, strip_reasoning, Turn, LABEL}};
use std::{cell::RefCell, collections::HashMap, ops::Range};

const MARKER: &str = "System: You will soon form a new memory, get ready. The messages that follow are the slice of \
recent experience you are about to compress. After them, write the memory in your own voice.";
const INSTRUCTION: &str = "Write the memory of events since the most recent memory system notification. Speak in the \
first person from your own perspective. Preserve concrete details — file paths, exact values, \
decisions, unresolved questions, the user's active asks. Target ~{target} tokens. Output only the \
memory body — no preamble, no section headers unless they help preservation, no \
meta-commentary about summarizing. Memorize only what actually happened in that slice: if it \
holds little beyond routine system traffic (heartbeats, empty turns, failure notices), a short \
memory saying so is correct — do not pad it by re-narrating events you already remember from \
earlier as if they had just happened again.";
const MERGE_INSTRUCTION: &str = "You have just reviewed {seen}, in chronological order. They cover the stretch of experience \
you are about to consolidate into a single L{level} memory. Write a memory that preserves the \
through-line: what happened, what was decided, what remains open, what concrete details \
future you will want to reach for. Speak in the first person. Target ~{target} tokens. \
Output only the memory body — no preamble, no meta-commentary about summarizing.";
const QUIET: &str = "(A quiet stretch: {n} messages of routine system traffic — heartbeats, empty turns, notices. \
Nothing happened worth remembering.)";

/// The host's side of a call.
pub trait Host {
    /// The agent's live tool definitions, once declared.
    fn tools(&self) -> Option<&Value> { None }
    /// The host's system prompt, once declared.
    fn system(&self) -> Option<&str> { None }
    /// The summarizer's response {stopReason, content, usage}, or why the call failed. The request's
    /// blocks are read through `reader` as the host writes them out.
    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String>;
}

/// A summarizer request as planned: its messages' blocks say where their content is.
pub struct Request {
    pub system: Option<String>,
    pub messages: Vec<Turn>,
    pub cache_ttl: bool,
    pub model: String,
    pub max_tokens: f64,
    pub tools: Option<Value>,
}

impl Request {
    /// As the library writes it, each block as `block` gives it: {system?, messages, cacheTtl?,
    /// config, tools?, toolChoice?}.
    pub fn value(&self, mut block: impl FnMut(&Block) -> Value) -> Value {
        let messages: Vec<Value> = self.messages.iter().map(|t| t.value(&mut block)).collect();
        let config = obj! {"model" => self.model.as_str(), "maxTokens" => self.max_tokens};
        let (cache_ttl, tool_choice) = (self.cache_ttl.then_some("1h"), self.tools.as_ref().map(|_| "none"));
        sparse(obj! {"system" => self.system.as_deref(), "messages" => messages, "cacheTtl" => cache_ttl, "config" => config, "tools" => self.tools.clone(), "toolChoice" => tool_choice})
    }
}

/// What a host answers for a request it was told to stop before its answer came (its user stopped
/// it): no attempt is recorded, as if the request wasn't made.
pub const INTERRUPTED: &str = "interrupted";

/// driveTick: the memories that are due form, in order (a stub without a call), until one makes a
/// call or none may be tried; the error a failed call gave, if one did.
pub fn drive(session: &Session, host: &mut dyn Host, cfg: &Config) -> Option<String> {
    loop {
        let (level, p) = due(session, cfg).filter(|&target| !blocked(session, target, cfg))?;
        if level == 1
            && let Some(stub) = quiet(session, p, cfg)
        {
            session.append("layer1", stub);
            continue;
        }
        // an L1 waits while it carries tool blocks and the host has declared no tools
        let request = if level == 1 { l1_request(session, p, host, cfg)? } else { merge_request(session, level, p, host, cfg) };
        let (outcome, error) = match host.complete(&request, &mut Reader::new(session, true)) {
            Err(error) if error == INTERRUPTED => return None,
            Err(error) => ("retry", Some(error)),
            Ok(response) if response["stopReason"] == "refusal" => ("refusal", None),
            Ok(response) => match usable(&response) {
                None => ("retry", None),
                Some(text) => {
                    session.append(&format!("layer{level}"), written(session, level, p, &request, &response, text, cfg));
                    return None;
                }
            },
        };
        session.append("attempts", obj! {"target" => vec![level, p], "messages" => session.count("messages"), "outcome" => outcome});
        return error;
    }
}

/// The memory to form next: the next L1, else the lowest level's next merge.
fn due(session: &Session, cfg: &Config) -> Option<(u64, u64)> {
    let (groups, count) = (Recorded::new(session, cfg), |n: u64| session.count(&format!("layer{n}")));
    let holdback = if session.last("compiles").is_some_and(|c| c["escalated"] == true) { 0 } else { cfg.l1_holdback_chunks };
    let merge = || (1..).take_while(|&n| count(n) >= 2).find(|&n| closed(session, n, &groups, cfg).is_some()).map(|n| (n + 1, count(n + 1)));
    (count(1) < count(0).saturating_sub(holdback)).then(|| (1, count(1))).or_else(merge)
}

/// The session's memories as their records group them: a merged memory's `merges`, the first and
/// last entry it merged of the layer below; one without merged by count (merge_by_count, as the
/// library does), its group of merge_threshold. Records never change, so what it reads it keeps.
pub struct Recorded<'s> {
    session: &'s Session,
    k: u64,
    read: RefCell<HashMap<Node, Range<u64>>>,
}

impl<'s> Recorded<'s> {
    pub fn new(session: &'s Session, cfg: &Config) -> Recorded<'s> {
        Recorded { session, k: cfg.merge_threshold, read: RefCell::default() }
    }
}

impl Groups for Recorded<'_> {
    fn members(&self, (n, p): Node) -> Range<u64> {
        if let Some(read) = self.read.borrow().get(&(n, p)) {
            return read.clone();
        }
        let merges = self.session.read(&format!("layer{n}"), p)["merges"].clone();
        let members = match merges.arr() { [first, last] => first.u64()..last.u64() + 1, _ => Fixed(self.k).members((n, p)) };
        self.read.borrow_mut().insert((n, p), members.clone());
        members
    }

    fn parent(&self, (n, j): Node) -> Option<Node> {
        let (mut low, mut high) = (0, self.session.count(&format!("layer{}", n + 1))); // to the first group that starts after j
        while low < high {
            let mid = (low + high) / 2;
            if self.members((n + 1, mid)).start <= j { low = mid + 1 } else { high = mid }
        }
        (low > 0 && self.members((n + 1, low - 1)).start == j).then_some((n + 1, low - 1))
    }
}

/// The first entry of layer n no memory of layer n+1 has merged.
fn open(session: &Session, n: u64, groups: &Recorded) -> u64 {
    let merged = session.count(&format!("layer{}", n + 1));
    if merged == 0 { 0 } else { groups.members((n + 1, merged - 1)).end }
}

/// The group layer n+1's next memory merges, once it has closed: of the entries of layer n no memory
/// has merged, the fewest, at least two, that hold target_chunk_tokens; under merge_by_count,
/// merge_threshold of them.
fn closed(session: &Session, n: u64, groups: &Recorded, cfg: &Config) -> Option<Range<u64>> {
    let (start, count) = (open(session, n, groups), session.count(&format!("layer{n}")));
    if cfg.merge_by_count {
        return (start + cfg.merge_threshold <= count).then(|| start..start + cfg.merge_threshold);
    }
    let mut held = 0.0;
    let full = |i: &u64| {
        held += session.read(&format!("layer{n}"), *i)["tokens"].as_f64().unwrap();
        *i > start && held >= cfg.target_chunk_tokens
    };
    (start..count).find(full).map(|last| start..last + 1)
}

/// The entries of layer level-1 that layer level's entry p merged, or, the next, merges.
fn merged(session: &Session, (level, p): Node, groups: &Recorded, cfg: &Config) -> Range<u64> {
    if p < session.count(&format!("layer{level}")) {
        groups.members((level, p))
    } else {
        closed(session, level - 1, groups, cfg).expect("its group has closed")
    }
}

/// Memory formation stops at a refusal: nothing is tried after one, so it stays the attempts log's
/// last record until surgery appends another. A target also waits for the log to grow once it failed
/// attempt_limit times at the log's length.
fn blocked(session: &Session, (level, p): (u64, u64), cfg: &Config) -> bool {
    let (now, target) = (session.count("messages"), Value::from(vec![level, p]));
    let recent = (0..session.count("attempts")).rev().map(|i| session.read("attempts", i)).take_while(|a| a["messages"].u64() == now);
    session.last("attempts").is_some_and(|a| a["outcome"] == "refusal") || recent.filter(|a| a["target"] == target).count() >= cfg.attempt_limit
}

/// The stub for a chunk of text blocks only under min_chunk_chars of trimmed text.
fn quiet(session: &Session, p: u64, cfg: &Config) -> Option<Value> {
    let (first, last) = span(session, 1, p, cfg);
    let text = |b: &Value| (b.kind() == "text").then(|| js_len(js_trim(b["text"].as_str().unwrap_or(""))));
    let chars: usize = session.span("messages", first, last).map(|m| m["content"].arr().iter().map(text).sum::<Option<usize>>()).sum::<Option<usize>>()?;
    (chars < cfg.min_chunk_chars).then(|| memory(first, last, &QUIET.replace("{n}", &(last - first + 1).to_string()), Value::Null, 40.0, Value::Null, cfg))
}

/// compressChunkHierarchical's request for chunk p, or None to defer it while the host has declared
/// no tools and the request carries tool blocks, which a provider takes only with their tools.
fn l1_request(session: &Session, p: u64, host: &dyn Host, cfg: &Config) -> Option<Request> {
    let (first, last) = span(session, 1, p, cfg);
    let task = [vec![say(MARKER)], raw(session, first, last), vec![say(&INSTRUCTION.replace("{target}", &number(cfg.summary_target_tokens)))]];
    let request = ask(session, first, task.concat(), host, cfg);
    let tools_used = request.messages.iter().flat_map(|t| &t.content).any(|b| b.kind == "tool_use" || b.kind == "tool_result");
    (host.tools().is_some_and(|tools| !tools.arr().is_empty()) || !tools_used).then_some(request)
}

/// executeMerge's request for layer `level`'s entry p: the run one level deeper (an L1 run's
/// messages raw, a higher run's children as recall pairs), then the merge instruction.
fn merge_request(session: &Session, level: u64, p: u64, host: &dyn Host, cfg: &Config) -> Request {
    let (groups, below, (first, last)) = (Recorded::new(session, cfg), level - 1, span(session, level, p, cfg));
    let (mut shown, seen) = if below == 1 {
        (raw(session, first, last), "the slices of recent experience above (raw conversation)".to_string())
    } else {
        let run = merged(session, (level, p), &groups, cfg); // its entries of layer below, then theirs
        let children: Vec<Node> = (groups.members((below, run.start)).start..groups.members((below, run.end - 1)).end).map(|id| (below - 1, id)).collect();
        (pairs(session, &children, cfg), format!("the L{} memories above", below - 1))
    };
    let instruction = MERGE_INSTRUCTION.replace("{seen}", &seen).replace("{level}", &level.to_string()).replace("{target}", &number(cfg.summary_target_tokens));
    shown.push(say(&instruction));
    ask(session, first, shown, host, cfg)
}

/// The request for `task`: recall pairs, oldest first, for the newest unmerged memories before
/// message `before` that fit the recall budget at their tokens plus 50 (capRecallPairs); normalized
/// for the provider, images capped; cache seams follow unless the budget left a memory out.
fn ask(session: &Session, before: u64, task: Vec<Turn>, host: &dyn Host, cfg: &Config) -> Request {
    let (prior, mut used) = (unmerged(session, before, cfg), 0.0);
    let fits = |&&(_, tokens): &&(Node, f64)| used + tokens + 50.0 <= cfg.recall_budget_tokens && { used += tokens + 50.0; true };
    let mut kept: Vec<Node> = prior.iter().rev().filter(fits).map(|&(node, _)| node).collect();
    kept.reverse();
    let mut wire = cap_images(normalize(pairs(session, &kept, cfg).into_iter().chain(task).collect()), cfg.compression_image_bytes);
    let marks = if kept.len() == prior.len() { seams(&wire, &kept) } else { vec![] };
    for (i, t) in wire.iter_mut().enumerate() {
        t.mark = marks.contains(&i);
    }
    let cap = cfg.compression_max_tokens.filter(|&c| c > 0.0).unwrap_or(f64::INFINITY);
    Request {
        system: host.system().filter(|s| !s.is_empty()).map(str::to_string),
        messages: wire,
        cache_ttl: !marks.is_empty(),
        model: cfg.compression_model.clone(),
        max_tokens: 16000f64.max(js_round(cfg.summary_target_tokens * 1.5)).min(cap),
        tools: host.tools().cloned(),
    }
}

/// Each memory no merge has taken that ends before message `before`, and its tokens, oldest first: a
/// level's are all older than the next level down's, since merges take each level's oldest.
fn unmerged(session: &Session, before: u64, cfg: &Config) -> Vec<(Node, f64)> {
    let (groups, count) = (Recorded::new(session, cfg), |n: u64| session.count(&format!("layer{n}")));
    let levels: Vec<u64> = (1..).take_while(|&n| count(n) > 0).collect();
    let level = |n: u64| session.span(&format!("layer{n}"), open(session, n, &groups), count(n) - 1);
    let ended = |m: &Value| m["last"].as_f64().unwrap() < before as f64;
    levels.into_iter().rev().flat_map(|n| level(n).filter(ended).map(move |m| ((n, m.id()), m["tokens"].as_f64().unwrap()))).collect()
}

/// Recall pairs: a turn naming each memory, then its answer as the agent.
fn pairs(session: &Session, memories: &[Node], cfg: &Config) -> Vec<Turn> {
    let pair = |&(level, id): &Node| {
        let answer = recalled(&session.read(&format!("layer{level}"), id), level);
        [say(&format!("[CM] Recall memory L{level}-{id}.")), Turn::new(&cfg.summary_participant, answer, None)]
    };
    memories.iter().flat_map(pair).collect()
}

/// applyMintCacheSeams: the turns a breakpoint follows: the one before the first recall pair, the
/// newest pair above L1's answer, and the newest pair's answer. None without pairs.
fn seams(wire: &[Turn], kept: &[Node]) -> Vec<usize> {
    let level_of = |name: &str| kept.iter().find(|(level, id)| format!("L{level}-{id}") == name).map(|&(level, _)| level);
    let found: Vec<(usize, u64)> = wire.iter().enumerate().filter_map(|(i, t)| header(t).and_then(level_of).map(|level| (i, level))).collect();
    let (Some(&(first, _)), Some(&(last, _))) = (found.first(), found.last()) else { return vec![] };
    let above_l1 = found.iter().filter(|&&(_, level)| level >= 2).map(|&(i, _)| i + 1).max();
    [first.checked_sub(1), Some(last + 1), above_l1].into_iter().flatten().collect()
}

/// The memory a recall pair's question turn names.
fn header(turn: &Turn) -> Option<&str> {
    let [Block { kind, at: At::Itself(block), .. }] = turn.content.as_slice() else { return None };
    let name = block["text"].as_str()?.strip_prefix("[CM] Recall memory ")?.strip_suffix('.')?;
    (turn.participant == LABEL && kind == "text" && !name.is_empty() && !name.contains(['\n', '\r', '\u{2028}', '\u{2029}'])).then_some(name)
}

/// Messages first..=last as a request replays them: without their reasoning.
fn raw(session: &Session, first: u64, last: u64) -> Vec<Turn> {
    (first..=last).map(|i| priced(session, i)).map(|p| Turn::new(&p.participant, strip_reasoning(&p.blocks), None)).collect()
}

/// A finished reply's memory text: its text blocks, less a leading <thinking>…</thinking>
/// (stripThinkingPreamble); empty for a reply cut short, malformed, empty, or all thinking.
pub fn usable(response: &Value) -> Option<String> {
    let content = response["content"].as_array().filter(|c| response["stopReason"] == "end_turn" && c.iter().all(Value::is_object))?;
    let joined = content.iter().filter(|b| b.kind() == "text").filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("\n");
    Some(strip_thinking(&joined)).filter(|text| !js_trim(text).is_empty()).map(str::to_string)
}

/// The text after a leading <thinking>…</thinking> and the spaces around it; none if it opens one
/// it never closes (ASCII case ignored).
fn strip_thinking(text: &str) -> &str {
    let rest = text.trim_start_matches(is_js_space);
    let Some(inside) = rest.get(..10).filter(|open| open.eq_ignore_ascii_case("<thinking>")).map(|_| &rest[10..]) else { return text };
    inside.to_ascii_lowercase().find("</thinking>").map_or("", |k| inside[k + 11..].trim_start_matches(is_js_space))
}

/// The memory a reply wrote: its text, its reasoning and text blocks if it reasoned
/// (captureResponseContent), its output tokens (else chars/3), the hash of its request as planned,
/// and, merged by tokens, the first and last entry it merged.
fn written(session: &Session, level: u64, p: u64, request: &Request, response: &Value, text: String, cfg: &Config) -> Value {
    let ((first, last), content) = (span(session, level, p, cfg), response["content"].arr());
    let out = response["usage"]["outputTokens"].as_f64();
    let blocks = content.iter().any(|b| reasoning(b.kind())).then(|| content.iter().filter(|b| reasoning(b.kind()) || b.kind() == "text").cloned().collect::<Vec<_>>());
    let tokens = out.filter(|&o| o > 0.0).unwrap_or_else(|| (js_len(&text) as f64 / 3.0).ceil());
    let provenance = obj! {"stopReason" => "end_turn", "requestHash" => digest(&request.value(|b| b.at.value())), "model" => cfg.compression_model.as_str()};
    let mut record = memory(first, last, &text, blocks.into(), tokens, provenance, cfg);
    if level >= 2 && !cfg.merge_by_count {
        let run = merged(session, (level, p), &Recorded::new(session, cfg), cfg);
        record["merges"] = vec![run.start, run.end - 1].into();
    }
    record
}

/// The first and last message of the stretch layer `level`'s entry p covers (or, the next, will):
/// from its first chunk's to its last's.
fn span(session: &Session, level: u64, p: u64, cfg: &Config) -> (u64, u64) {
    let groups = Recorded::new(session, cfg);
    let covered = if level <= 1 {
        p..p + 1
    } else {
        let run = merged(session, (level, p), &groups, cfg);
        chunks((level - 1, run.start), &groups).start..chunks((level - 1, run.end - 1), &groups).end
    };
    (session.read("layer0", covered.start)["first"].u64(), session.read("layer0", covered.end - 1)["last"].u64())
}

fn memory(first: u64, last: u64, text: &str, blocks: Value, tokens: f64, provenance: Value, cfg: &Config) -> Value {
    let mut record = obj! {"first" => first, "last" => last, "text" => text, "blocks" => blocks, "tokens" => tokens, "provenance" => provenance};
    record["recall"] = recall_tokens(&answer(&record), cfg).into();
    record
}

/// A setting as Python's str() writes an int.
fn number(n: f64) -> String {
    crate::json::to_string(&Value::from(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::append;

    /// A host that gives every request the same answer, and counts them.
    struct Answering(Result<Value, String>, usize);

    impl Host for Answering {
        fn complete(&mut self, _: &Request, _: &mut Reader) -> Result<Value, String> {
            self.1 += 1;
            self.0.clone()
        }
    }

    #[test]
    fn an_interrupted_request_is_no_attempt() {
        let dir = std::env::temp_dir().join(format!("connectome-interrupted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (session, cfg) = (Session::open(&dir), Config { target_chunk_tokens: 50.0, recent_window_tokens: 50.0, l1_holdback_chunks: 0, ..Config::default() });
        for (i, who) in ["user", "assistant"].iter().cycle().take(6).enumerate() {
            append(&session, &obj! {"participant" => *who, "content" => vec![obj! {"type" => "text", "text" => format!("message {i}: {}", "words ".repeat(60))}]}, &cfg);
        }
        assert!(due(&session, &cfg).is_some());
        let mut stopped = Answering(Err(INTERRUPTED.into()), 0);
        assert_eq!((drive(&session, &mut stopped, &cfg), stopped.1, session.count("attempts")), (None, 1, 0));
        let mut failing = Answering(Err("down".into()), 0);
        assert_eq!((drive(&session, &mut failing, &cfg), session.count("attempts")), (Some("down".into()), 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memories_merged_by_count_then_by_tokens() {
        let dir = std::env::temp_dir().join(format!("connectome-merges-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let session = Session::open(&dir);
        let by_count = Config { target_chunk_tokens: 50.0, recent_window_tokens: 50.0, l1_holdback_chunks: 0, merge_threshold: 4, merge_by_count: true, ..Config::default() };
        let reply = obj! {"stopReason" => "end_turn", "content" => vec![obj! {"type" => "text", "text" => "a memory"}], "usage" => obj! {"outputTokens" => 30.0}};
        let mut host = Answering(Ok(reply), 0);
        let mut said = 0;
        let mut grow = |chunks: u64, cfg: &Config| {
            let goal = session.count("layer0") + chunks;
            while session.count("layer0") < goal {
                append(&session, &obj! {"participant" => if said % 2 == 0 { "user" } else { "assistant" }, "content" => vec![obj! {"type" => "text", "text" => format!("message {said}: {}", "words ".repeat(60))}]}, cfg);
                said += 1;
                while due(&session, cfg).is_some() {
                    assert_eq!(drive(&session, &mut host, cfg), None);
                }
            }
        };
        let count = |n: u64| session.count(&format!("layer{n}"));
        let merges = |n: u64| (0..count(n)).map(|p| session.read(&format!("layer{n}"), p)["merges"].clone()).collect::<Vec<_>>();
        let runs = |n: u64, from: usize| merges(n)[from..].iter().map(|m| (m.arr()[0].u64(), m.arr()[1].u64())).collect::<Vec<_>>();
        grow(11, &by_count); // by count: fours, records that say nothing of them, three L1s waiting
        assert_eq!((count(1), count(2), count(3)), (11, 2, 0));
        assert!(merges(2).iter().all(Value::is_null));
        // by tokens: memories of 30 tokens merge two at a time to hold 50, from the first that waited
        let by_tokens = Config { merge_by_count: false, ..by_count.clone() };
        grow(5, &by_tokens);
        assert_eq!((runs(2, 2), runs(3, 0)), (vec![(8, 9), (10, 11), (12, 13), (14, 15)], vec![(0, 1), (2, 3), (4, 5)]));
        let groups = Recorded::new(&session, &by_tokens);
        assert_eq!((groups.members((2, 1)), groups.members((3, 1)), groups.parent((2, 2)), groups.parent((2, 3))), (4..8, 2..4, Some((3, 1)), None));
        assert_eq!((chunks((3, 0), &groups), chunks((3, 1), &groups)), (0..8, 8..12));
        let l3 = session.read("layer3", 1);
        assert_eq!((l3["first"].u64(), l3["last"].u64()), (session.read("layer0", 8)["first"].u64(), session.read("layer0", 11)["last"].u64()));
        // a group holds two at least, though one memory holds the target alone
        let tiny = Config { target_chunk_tokens: 10.0, ..by_tokens.clone() };
        let before = count(2) as usize;
        grow(4, &tiny);
        assert_eq!(runs(2, before), [(16, 17), (18, 19)]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
