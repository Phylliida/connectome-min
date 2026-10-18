//! Memory formation (AutobiographicalStrategy.tick, simplified): at most one memory per tick, in order,
//! so the logs' tails say what is due:
//!   - the next chunk's L1, once l1_holdback_chunks newer chunks have closed, or at once when the last
//!     compile couldn't fold enough (escalated);
//!   - else the next merge at the lowest level: layer n+1's entry p, once layer n has entries p*k ..
//!     p*k+k-1.
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
//! attempt_limit times until the log grows.

use crate::blocks::{reasoning, At, Block};
use crate::chunking::priced;
use crate::config::Config;
use crate::emit::{answer, recall_tokens, recalled, Reader};
use crate::js::{digest, is_js_space, js_len, js_round, js_trim};
use crate::json::{sparse, Json, Value};
use crate::obj;
use crate::planner::chunks;
use crate::store::Session;
use crate::turns::{cap_images, normalize, say, strip_reasoning, Turn, LABEL};

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
    fn tools(&self) -> Option<&Value>;
    /// The host's system prompt, once declared.
    fn system(&self) -> Option<&str>;
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

/// Form the memory that's due, if it may be tried: (whether a call was made, the error if it failed).
pub fn tick(session: &Session, host: &mut dyn Host, cfg: &Config) -> (bool, Option<String>) {
    let Some((level, p)) = due(session, cfg).filter(|&target| !blocked(session, target, cfg)) else { return (false, None) };
    if level == 1
        && let Some(stub) = quiet(session, p, cfg)
    {
        session.append("layer1", stub);
        return (false, None);
    }
    let request = if level == 1 { l1_request(session, p, host, cfg) } else { Some(merge_request(session, level, p, host, cfg)) };
    let Some(request) = request else { return (false, None) }; // deferred until the host declares its tools
    let (outcome, error) = match host.complete(&request, &mut Reader::new(session, true)) {
        Err(error) => ("retry", Some(error)),
        Ok(response) if response["stopReason"] == "refusal" => ("refusal", None),
        Ok(response) => match usable(&response) {
            text if text.is_empty() => ("retry", None),
            text => {
                session.append(&format!("layer{level}"), written(session, level, p, &request, &response, text, cfg));
                return (true, None);
            }
        },
    };
    session.append("attempts", obj! {"target" => vec![level, p], "messages" => session.count("messages"), "outcome" => outcome});
    (true, error)
}

/// The memory to form next: the next L1, else the lowest level's next merge.
fn due(session: &Session, cfg: &Config) -> Option<(u64, u64)> {
    let (k, closed, l1s) = (cfg.merge_threshold, session.count("layer0"), session.count("layer1"));
    let escalated = session.last("compiles").is_some_and(|c| c["escalated"].truthy());
    if l1s < closed.saturating_sub(if escalated { 0 } else { cfg.l1_holdback_chunks }) {
        return Some((1, l1s));
    }
    for n in 1.. {
        let have = session.count(&format!("layer{n}"));
        if have < k {
            return None;
        }
        let p = session.count(&format!("layer{}", n + 1));
        if p * k + k <= have {
            return Some((n + 1, p));
        }
    }
    unreachable!()
}

/// Memory formation stops at a refusal: nothing is tried after one, so it stays the attempts log's
/// last record until surgery appends another. A target also waits for the log to grow once it failed
/// attempt_limit times at the log's length.
fn blocked(session: &Session, (level, p): (u64, u64), cfg: &Config) -> bool {
    if session.last("attempts").is_some_and(|a| a["outcome"].as_str() == Some("refusal")) {
        return true;
    }
    let (now, target) = (session.count("messages"), Value::from(vec![level, p]));
    let recent = (0..session.count("attempts")).rev().map(|i| session.read("attempts", i)).take_while(|a| a["messages"].u64() == now);
    recent.filter(|a| a["target"] == target).count() >= cfg.attempt_limit
}

/// The stub for a chunk of text blocks only under min_chunk_chars of trimmed text.
fn quiet(session: &Session, p: u64, cfg: &Config) -> Option<Value> {
    let (first, last) = span(session, 1, p, cfg);
    let mut chars = 0;
    for message in session.span("messages", first, last) {
        for b in message["content"].arr() {
            if b.kind() != "text" {
                return None;
            }
            chars += js_len(js_trim(b["text"].as_str().unwrap_or("")));
        }
    }
    (chars < cfg.min_chunk_chars).then(|| memory(first, last, &QUIET.replace("{n}", &(last - first + 1).to_string()), Value::Null, 40.0, Value::Null, cfg))
}

/// compressChunkHierarchical's request for chunk p, or None to defer it while the host has declared
/// no tools and the request carries tool blocks, which a provider takes only with their tools.
fn l1_request(session: &Session, p: u64, host: &dyn Host, cfg: &Config) -> Option<Request> {
    let (first, last) = span(session, 1, p, cfg);
    let mut task = vec![say(MARKER)];
    task.extend(raw(session, first, last));
    task.push(say(&INSTRUCTION.replace("{target}", &number(cfg.summary_target_tokens))));
    let request = ask(session, first, task, host, cfg);
    let tools_used = request.messages.iter().flat_map(|t| &t.content).any(|b| b.kind == "tool_use" || b.kind == "tool_result");
    (host.tools().is_some_and(Value::truthy) || !tools_used).then_some(request)
}

/// executeMerge's request for layer `level`'s entry p: the run one level deeper (an L1 run's
/// messages raw, a higher run's children as recall pairs), then the merge instruction.
fn merge_request(session: &Session, level: u64, p: u64, host: &dyn Host, cfg: &Config) -> Request {
    let (k, below, (first, last)) = (cfg.merge_threshold, level - 1, span(session, level, p, cfg));
    let mut shown = if below == 1 {
        raw(session, first, last)
    } else {
        let layer = format!("layer{}", below - 1);
        let children: Vec<Recall> = session.span(&layer, p * k * k, (p + 1) * k * k - 1).map(|m| Recall::of(below - 1, &m)).collect();
        pairs(session, &children, cfg)
    };
    let seen = if below > 1 { format!("the L{} memories above", below - 1) } else { "the slices of recent experience above (raw conversation)".into() };
    let instruction = MERGE_INSTRUCTION.replace("{seen}", &seen).replace("{level}", &level.to_string()).replace("{target}", &number(cfg.summary_target_tokens));
    shown.push(say(&instruction));
    ask(session, first, shown, host, cfg)
}

/// The request for `task`: recall pairs, oldest first, for the newest unmerged memories before
/// message `before` that fit the recall budget at their tokens plus 50 (capRecallPairs); normalized
/// for the provider, images capped; cache seams follow unless the budget left a memory out.
fn ask(session: &Session, before: u64, task: Vec<Turn>, host: &dyn Host, cfg: &Config) -> Request {
    let prior = unmerged(session, before, cfg);
    let (mut kept, mut used) = (vec![], 0.0);
    for m in prior.iter().rev() {
        if used + m.tokens + 50.0 <= cfg.recall_budget_tokens {
            kept.insert(0, m.clone());
            used += m.tokens + 50.0;
        }
    }
    let mut turns = pairs(session, &kept, cfg);
    turns.extend(task);
    let mut wire = cap_images(normalize(turns), cfg.compression_image_bytes);
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

/// What a request needs of a memory it replays, without its text.
#[derive(Clone)]
struct Recall {
    level: u64,
    id: u64,
    first: f64,
    tokens: f64,
}

impl Recall {
    fn of(level: u64, m: &Value) -> Recall {
        Recall { level, id: m.id(), first: m["first"].as_f64().unwrap(), tokens: m["tokens"].as_f64().unwrap() }
    }
}

/// Each memory no merge has taken that ends before message `before`, oldest first.
fn unmerged(session: &Session, before: u64, cfg: &Config) -> Vec<Recall> {
    let mut out = vec![];
    for n in (1..).take_while(|n| session.count(&format!("layer{n}")) > 0) {
        let (layer, have) = (format!("layer{n}"), session.count(&format!("layer{n}")));
        let start = session.count(&format!("layer{}", n + 1)) * cfg.merge_threshold;
        out.extend(session.span(&layer, start, have - 1).filter(|m| m["last"].as_f64().unwrap() < before as f64).map(|m| Recall::of(n, &m)));
    }
    out.sort_by(|a, b| a.first.total_cmp(&b.first));
    out
}

/// Recall pairs: a turn naming each memory, then its answer as the agent.
fn pairs(session: &Session, memories: &[Recall], cfg: &Config) -> Vec<Turn> {
    let pair = |m: &Recall| {
        let answer = recalled(&session.read(&format!("layer{}", m.level), m.id), m.level);
        [say(&format!("[CM] Recall memory L{}-{}.", m.level, m.id)), Turn::new(&cfg.summary_participant, answer, None)]
    };
    memories.iter().flat_map(pair).collect()
}

/// applyMintCacheSeams: the turns a breakpoint follows: the one before the first recall pair, the
/// newest pair above L1's answer, and the newest pair's answer. None without pairs.
fn seams(wire: &[Turn], kept: &[Recall]) -> Vec<usize> {
    let level_of = |name: &str| kept.iter().find(|m| format!("L{}-{}", m.level, m.id) == name).map(|m| m.level);
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
pub fn usable(response: &Value) -> String {
    let Value::Array(content) = &response["content"] else { return String::new() };
    if response["stopReason"].as_str() != Some("end_turn") || !content.iter().all(|b| b.is_object()) {
        return String::new();
    }
    let texts: Vec<&str> = content.iter().filter(|b| b["type"].as_str() == Some("text")).filter_map(|b| b["text"].as_str()).collect();
    let joined = texts.join("\n");
    let text = strip_thinking(&joined);
    if js_trim(text).is_empty() { String::new() } else { text.to_string() }
}

/// The text after a leading <thinking>…</thinking> and the spaces around it; none if it opens one
/// it never closes (ASCII case ignored).
fn strip_thinking(text: &str) -> &str {
    let rest = text.trim_start_matches(is_js_space);
    if !rest.get(..10).is_some_and(|open| open.eq_ignore_ascii_case("<thinking>")) {
        return text;
    }
    match rest[10..].to_ascii_lowercase().find("</thinking>") {
        Some(k) => rest[10 + k + 11..].trim_start_matches(is_js_space),
        None => "",
    }
}

/// The memory a reply wrote: its text, its reasoning and text blocks if it reasoned
/// (captureResponseContent), its output tokens (else chars/3), and the hash of its request as
/// planned.
fn written(session: &Session, level: u64, p: u64, request: &Request, response: &Value, text: String, cfg: &Config) -> Value {
    let ((first, last), content) = (span(session, level, p, cfg), response["content"].arr());
    let out = response["usage"]["outputTokens"].as_f64();
    let blocks = if content.iter().any(|b| reasoning(b.kind())) {
        Value::Array(content.iter().filter(|b| reasoning(b.kind()) || b.kind() == "text").cloned().collect())
    } else {
        Value::Null
    };
    let tokens = out.filter(|&o| o > 0.0).unwrap_or_else(|| (js_len(&text) as f64 / 3.0).ceil());
    let provenance = obj! {"stopReason" => "end_turn", "requestHash" => digest(&request.value(|b| b.at.value())), "model" => cfg.compression_model.as_str()};
    memory(first, last, &text, blocks, tokens, provenance, cfg)
}

/// The first and last message of the stretch layer `level`'s entry p covers: from its first chunk's
/// to its last's.
fn span(session: &Session, level: u64, p: u64, cfg: &Config) -> (u64, u64) {
    let covered = chunks((level, p), cfg.merge_threshold);
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
