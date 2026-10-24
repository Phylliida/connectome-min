//! Memory formation (AutobiographicalStrategy.tick, simplified): at most one memory per tick, in order,
//! so the logs' tails say what is due:
//!   - the next chunk's L1, once l1_holdback_chunks newer chunks have closed, or at once when the last
//!     compile couldn't fold enough (escalated);
//!   - else the next merge at the lowest level: layer n+1's next entry, once the entries of layer n no
//!     memory has merged yet have closed a group (closed): the fewest, at least min_merge_memories,
//!     that hold target_chunk_tokens, as chunks close; or under merge_by_count, as the library merges, k of
//!     them (k = merge_threshold). A merged memory's record says what it merged (Recorded);
//!   - else, when the last compile couldn't fold enough and not under merge_by_count, a lifted merge,
//!     which the library doesn't make (it merges only within a layer). Each layer keeps the entries
//!     no memory has merged, short of a group, and with enough layers those and the tail outgrow the
//!     window, with nothing left to fold. A lifted merge takes the oldest of them past the top
//!     layer's, across layers, the fewest, at least min_merge_memories, that hold target_chunk_tokens, into a memory
//!     one layer above the highest it takes, so it waits with that layer's (lifted);
//!   - else, when the last compile couldn't fold enough, the same merge or lift of two at least, so
//!     folding never waits on a third memory (escalated: what was due at min_merge_memories first).
//!   A memory held back (blocked) holds back those after it, but not a lift, which goes ahead past it.
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
use std::{cell::RefCell, collections::HashMap};

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
/// call or none may be tried: whether any formed, or the error a failed call gave.
pub fn drive(session: &Session, host: &mut dyn Host, cfg: &Config) -> Result<bool, String> {
    let mut formed = false;
    loop {
        let Some(((level, p), members)) = next(session, cfg) else { return Ok(formed) };
        if level == 1
            && let Some(stub) = quiet(session, p, cfg)
        {
            session.append("layer1", stub);
            formed = true;
            continue;
        }
        // an L1 waits while it carries tool blocks and the host has declared no tools
        let Some(request) = (if level == 1 { l1_request(session, p, host, cfg) } else { Some(merge_request(session, level, &members, host, cfg)) }) else { return Ok(formed) };
        let (outcome, error) = match host.complete(&request, &mut Reader::new(session, true)) {
            Err(error) if error == INTERRUPTED => return Ok(formed),
            Err(error) => ("retry", Some(error)),
            Ok(response) if response["stopReason"] == "refusal" => ("refusal", None),
            Ok(response) => match usable(&response) {
                None => ("retry", None),
                Some(text) => {
                    session.append(&format!("layer{level}"), written(session, level, &members, &request, &response, text, cfg));
                    return Ok(true);
                }
            },
        };
        session.append("attempts", obj! {"target" => vec![level, p], "messages" => session.count("messages"), "outcome" => outcome});
        return error.map_or(Ok(formed), Err);
    }
}

/// The memory to form now: the one due, unless it's held back; then a lift past it, if one is due.
fn next(session: &Session, cfg: &Config) -> Option<(Node, Vec<Node>)> {
    let first = due(session, cfg)?;
    if !blocked(session, first.0, cfg) {
        return Some(first);
    }
    [cfg.min_merge_memories, 2].into_iter().find_map(|least| lift(session, least, cfg)).filter(|&(target, _)| !blocked(session, target, cfg))
}

/// Why the memory due doesn't form, if it's held back: memories stopped forming at a refusal, or it
/// failed attempt_limit times since the log last grew.
pub fn stalled(session: &Session, cfg: &Config) -> Option<String> {
    let ((level, p), _) = due(session, cfg).filter(|&(target, _)| blocked(session, target, cfg))?;
    let which = if level == 1 { format!("the memory of chunk {p}") } else { format!("the next L{level} memory") };
    Some(match session.last("attempts").is_some_and(|a| a["outcome"] == "refusal") {
        true => format!("memories stopped forming when the summarizer refused one, and {which} waits for that to be mended"),
        false => format!("{which} failed {} times since the log last grew", cfg.attempt_limit),
    })
}

/// The memory to form next, and what it holds: the next L1 (its chunk), else the lowest level's next
/// merge, else a lift, of min_merge_memories at least; else, when the last compile couldn't fold
/// enough, the same of two at least.
fn due(session: &Session, cfg: &Config) -> Option<(Node, Vec<Node>)> {
    let (groups, count) = (Recorded::new(session, cfg), |n: u64| session.count(&format!("layer{n}")));
    let holdback = if escalated(session) { 0 } else { cfg.l1_holdback_chunks };
    let merge = |least: u64| (1..).take_while(|&n| count(n) > 0).find_map(|n| closed(session, n, &groups, least, cfg)).map(|m| into(session, m));
    let either = |least: u64| merge(least).or_else(|| lift(session, least, cfg));
    let pairs = || (escalated(session) && cfg.min_merge_memories > 2).then(|| either(2)).flatten();
    (count(1) < count(0).saturating_sub(holdback)).then(|| into(session, vec![(0, count(1))])).or_else(|| either(cfg.min_merge_memories)).or_else(pairs)
}

/// A lifted merge of `least` memories at least, when the last compile couldn't fold enough, not under
/// merge_by_count.
fn lift(session: &Session, least: u64, cfg: &Config) -> Option<(Node, Vec<Node>)> {
    (escalated(session) && !cfg.merge_by_count).then(|| lifted(session, least, cfg)).flatten().map(|m| into(session, m))
}

/// Whether the last compile couldn't fold enough.
fn escalated(session: &Session) -> bool {
    session.last("compiles").is_some_and(|c| c["escalated"] == true)
}

/// The memory `members` make: the next of the layer above the highest of them.
fn into(session: &Session, members: Vec<Node>) -> (Node, Vec<Node>) {
    let level = members[0].0 + 1;
    ((level, session.count(&format!("layer{level}"))), members)
}

/// The session's memories as their records group them: a merged memory's `merges`, the first and
/// last entry it merged of the layer below, or, lifted, [layer, first, last] for each layer's run it
/// merged, the highest first; one without merged by count (merge_by_count, as the library does), its
/// group of merge_threshold. Records never change, so what it reads it keeps.
pub struct Recorded<'s> {
    session: &'s Session,
    k: u64,
    read: RefCell<HashMap<Node, Vec<Node>>>,
}

impl<'s> Recorded<'s> {
    pub fn new(session: &'s Session, cfg: &Config) -> Recorded<'s> {
        Recorded { session, k: cfg.merge_threshold, read: RefCell::default() }
    }
}

impl Groups for Recorded<'_> {
    fn members(&self, (n, p): Node) -> Vec<Node> {
        if let Some(read) = self.read.borrow().get(&(n, p)) {
            return read.clone();
        }
        let merges = self.session.read(&format!("layer{n}"), p)["merges"].clone();
        let run = |level: u64, first: &Value, last: &Value| (first.u64()..=last.u64()).map(move |j| (level, j));
        let members = match merges.arr() {
            [] => Fixed(self.k).members((n, p)),
            [first, last] if !first.is_array() => run(n - 1, first, last).collect(),
            runs => runs.iter().flat_map(|r| run(r.arr()[0].u64(), &r.arr()[1], &r.arr()[2])).collect(),
        };
        self.read.borrow_mut().insert((n, p), members.clone());
        members
    }

    fn parent(&self, (n, j): Node) -> Option<Node> {
        let (mut low, mut high) = (0, self.session.count(&format!("layer{}", n + 1))); // to the first group that starts after j
        while low < high {
            let mid = (low + high) / 2;
            if self.members((n + 1, mid))[0].1 <= j { low = mid + 1 } else { high = mid }
        }
        (low > 0 && self.members((n + 1, low - 1))[0].1 == j).then(|| (n + 1, low - 1))
    }
}

/// The first entry of layer n no memory has merged: the first past the chunks the layers above cover,
/// since each merge takes the oldest entries no memory has merged (a lifted one, of several layers).
fn open(session: &Session, n: u64, groups: &Recorded) -> u64 {
    let count = |m: u64| session.count(&format!("layer{m}"));
    let covered = (n + 1..).take_while(|&m| count(m) > 0).map(|m| chunks((m, count(m) - 1), groups).end).max().unwrap_or(0);
    let (mut low, mut high) = (0, count(n));
    while low < high {
        let mid = (low + high) / 2;
        if chunks((n, mid), groups).start < covered { low = mid + 1 } else { high = mid }
    }
    low
}

/// The group layer n+1's next memory merges, once it has closed: of the entries of layer n no memory
/// has merged, the fewest, at least `least`, that hold target_chunk_tokens; under merge_by_count,
/// merge_threshold of them.
fn closed(session: &Session, n: u64, groups: &Recorded, least: u64, cfg: &Config) -> Option<Vec<Node>> {
    let (start, count) = (open(session, n, groups), session.count(&format!("layer{n}")));
    if cfg.merge_by_count {
        return (start + cfg.merge_threshold <= count).then(|| (start..start + cfg.merge_threshold).map(|j| (n, j)).collect());
    }
    let mut held = 0.0;
    let full = |i: &u64| {
        held += session.read(&format!("layer{n}"), *i)["tokens"].as_f64().unwrap();
        *i + 1 >= start + least && held >= cfg.target_chunk_tokens
    };
    (start..count).find(full).map(|last| (start..=last).map(|j| (n, j)).collect())
}

/// The group a lifted memory merges: of the memories no memory has merged, oldest first, past the top
/// layer's, the fewest, at least `least`, that hold target_chunk_tokens.
fn lifted(session: &Session, least: u64, cfg: &Config) -> Option<Vec<Node>> {
    let waiting = unmerged(session, u64::MAX, cfg);
    let top = waiting.first()?.0.0;
    let below: Vec<(Node, f64)> = waiting.into_iter().skip_while(|&((level, _), _)| level == top).collect();
    let mut held = 0.0;
    let last = below.iter().enumerate().position(|(i, &(_, tokens))| {
        held += tokens;
        i as u64 + 1 >= least && held >= cfg.target_chunk_tokens
    })?;
    Some(below[..=last].iter().map(|&(node, _)| node).collect())
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
    let (first, last) = span(session, &[(0, p)], cfg);
    let text = |b: &Value| (b.kind() == "text").then(|| js_len(js_trim(b["text"].as_str().unwrap_or(""))));
    let chars: usize = session.span("messages", first, last).map(|m| m["content"].arr().iter().map(text).sum::<Option<usize>>()).sum::<Option<usize>>()?;
    (chars < cfg.min_chunk_chars).then(|| memory(first, last, &QUIET.replace("{n}", &(last - first + 1).to_string()), Value::Null, 40.0, Value::Null, cfg))
}

/// compressChunkHierarchical's request for chunk p, or None to defer it while the host has declared
/// no tools and the request carries tool blocks, which a provider takes only with their tools.
fn l1_request(session: &Session, p: u64, host: &dyn Host, cfg: &Config) -> Option<Request> {
    let (first, last) = span(session, &[(0, p)], cfg);
    let task = [vec![say(MARKER)], raw(session, first, last), vec![say(&INSTRUCTION.replace("{target}", &number(cfg.summary_target_tokens)))]];
    let request = ask(session, first, task.concat(), host, cfg);
    let tools_used = request.messages.iter().flat_map(|t| &t.content).any(|b| b.kind == "tool_use" || b.kind == "tool_result");
    (host.tools().is_some_and(|tools| !tools.arr().is_empty()) || !tools_used).then_some(request)
}

/// executeMerge's request for a memory of layer `level` that merges `members`: what they hold, one
/// level deeper (an L1's chunk's messages raw, a higher memory's members as recall pairs), then the
/// merge instruction.
fn merge_request(session: &Session, level: u64, members: &[Node], host: &dyn Host, cfg: &Config) -> Request {
    let groups = Recorded::new(session, cfg);
    let deeper: Vec<Node> = members.iter().flat_map(|&(n, id)| if n == 1 { vec![(0, id)] } else { groups.members((n, id)) }).collect();
    let mut shown: Vec<Turn> = deeper.chunk_by(|a, b| (a.0 == 0) == (b.0 == 0)).flat_map(|run| match run[0].0 {
        0 => { let (first, last) = span(session, run, cfg); raw(session, first, last) }
        _ => pairs(session, run, cfg),
    }).collect();
    let mut names: Vec<String> = deeper.iter().filter(|&&(n, _)| n > 0).map(|&(n, _)| format!("L{n}")).collect();
    names.dedup();
    let names = match names.split_last() { Some((last, [])) => last.clone(), Some((last, rest)) => format!("{} and {last}", rest.join(", ")), None => String::new() };
    let seen = match (names.is_empty(), deeper.iter().any(|&(n, _)| n == 0)) {
        (true, _) => "the slices of recent experience above (raw conversation)".to_string(),
        (false, false) => format!("the {names} memories above"),
        (false, true) => format!("the {names} memories and the slices of recent experience (raw conversation) above"),
    };
    let instruction = MERGE_INSTRUCTION.replace("{seen}", &seen).replace("{level}", &level.to_string()).replace("{target}", &number(cfg.summary_target_tokens));
    shown.push(say(&instruction));
    ask(session, span(session, members, cfg).0, shown, host, cfg)
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
    Request {
        system: host.system().filter(|s| !s.is_empty()).map(str::to_string),
        messages: wire,
        cache_ttl: !marks.is_empty(),
        model: cfg.compression_model.clone(),
        max_tokens: cfg.compression_max_tokens.filter(|&c| c > 0.0).unwrap_or_else(|| 16000f64.max(js_round(cfg.summary_target_tokens * 1.5))),
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
/// and, merged by tokens, what it merged: the first and last entry of the layer below, or, lifted,
/// [layer, first, last] for each layer's run.
fn written(session: &Session, level: u64, members: &[Node], request: &Request, response: &Value, text: String, cfg: &Config) -> Value {
    let ((first, last), content) = (span(session, members, cfg), response["content"].arr());
    let out = response["usage"]["outputTokens"].as_f64();
    let blocks = content.iter().any(|b| reasoning(b.kind())).then(|| content.iter().filter(|b| reasoning(b.kind()) || b.kind() == "text").cloned().collect::<Vec<_>>());
    let tokens = out.filter(|&o| o > 0.0).unwrap_or_else(|| (js_len(&text) as f64 / 3.0).ceil());
    let provenance = obj! {"stopReason" => "end_turn", "requestHash" => digest(&request.value(|b| b.at.value())), "model" => cfg.compression_model.as_str()};
    let mut record = memory(first, last, &text, blocks.into(), tokens, provenance, cfg);
    if level >= 2 && !cfg.merge_by_count {
        let runs: Vec<&[Node]> = members.chunk_by(|a, b| a.0 == b.0).collect();
        record["merges"] = match runs.as_slice() {
            [run] => vec![run[0].1, run[run.len() - 1].1].into(),
            runs => Value::Array(runs.iter().map(|run| vec![run[0].0, run[0].1, run[run.len() - 1].1].into()).collect()),
        };
    }
    record
}

/// The first and last message of the stretch `members` cover: from its first chunk's to its last's.
fn span(session: &Session, members: &[Node], cfg: &Config) -> (u64, u64) {
    let groups = Recorded::new(session, cfg);
    let (first, last) = (chunks(members[0], &groups).start, chunks(members[members.len() - 1], &groups).end - 1);
    (session.read("layer0", first)["first"].u64(), session.read("layer0", last)["last"].u64())
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
    use crate::{drive::{append, current}, emit::{select, Fit}};
    use serde_json::json;

    /// A host that gives every request the same answer, and counts them.
    struct Answering(Result<Value, String>, usize);

    impl Host for Answering {
        fn complete(&mut self, _: &Request, _: &mut Reader) -> Result<Value, String> {
            self.1 += 1;
            self.0.clone()
        }
    }

    /// Messages of 60 words, the speakers taking turns, until `chunks` more chunks have closed, each
    /// memory due forming as they do if `form`.
    fn grow(session: &Session, host: &mut Answering, chunks: u64, form: bool, cfg: &Config) {
        let goal = session.count("layer0") + chunks;
        while session.count("layer0") < goal {
            let said = session.count("messages");
            append(session, &obj! {"participant" => if said % 2 == 0 { "user" } else { "assistant" }, "content" => vec![obj! {"type" => "text", "text" => format!("message {said}: {}", "words ".repeat(60))}]}, cfg);
            while form && due(session, cfg).is_some() {
                assert_eq!(drive(session, host, cfg), Ok(true));
            }
        }
    }

    fn memory_of_30_tokens() -> Answering {
        Answering(Ok(obj! {"stopReason" => "end_turn", "content" => vec![obj! {"type" => "text", "text" => "a memory"}], "usage" => obj! {"outputTokens" => 30.0}}), 0)
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
        assert_eq!((drive(&session, &mut stopped, &cfg), stopped.1, session.count("attempts")), (Ok(false), 1, 0));
        let mut failing = Answering(Err("down".into()), 0);
        assert_eq!((drive(&session, &mut failing, &cfg), session.count("attempts")), (Err("down".into()), 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memories_merged_by_count_then_by_tokens() {
        let dir = std::env::temp_dir().join(format!("connectome-merges-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let session = Session::open(&dir);
        let by_count = Config { target_chunk_tokens: 50.0, recent_window_tokens: 50.0, l1_holdback_chunks: 0, merge_threshold: 4, merge_by_count: true, min_merge_memories: 2, ..Config::default() };
        let mut host = memory_of_30_tokens();
        let mut grow = |chunks: u64, cfg: &Config| grow(&session, &mut host, chunks, true, cfg);
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
        let nodes = |level: u64, ids: std::ops::Range<u64>| ids.map(|j| (level, j)).collect::<Vec<Node>>();
        assert_eq!((groups.members((2, 1)), groups.members((3, 1)), groups.parent((2, 2)), groups.parent((2, 3))), (nodes(1, 4..8), nodes(2, 2..4), Some((3, 1)), None));
        assert_eq!((chunks((3, 0), &groups), chunks((3, 1), &groups)), (0..8, 8..12));
        let l3 = session.read("layer3", 1);
        assert_eq!((l3["first"].u64(), l3["last"].u64()), (session.read("layer0", 8)["first"].u64(), session.read("layer0", 11)["last"].u64()));
        // a group holds two at least, though one memory holds the target alone
        let tiny = Config { target_chunk_tokens: 10.0, ..by_tokens.clone() };
        let before = count(2) as usize;
        grow(4, &tiny);
        assert_eq!(runs(2, before), [(16, 17), (18, 19)]);
        // and min_merge_memories at least, which may change as it goes
        let threes = Config { min_merge_memories: 3, ..tiny.clone() };
        let before = count(2) as usize;
        grow(6, &threes);
        assert_eq!(runs(2, before), [(20, 22), (23, 25)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_context_that_cant_fold_enough_lifts_the_oldest_memories() {
        let dir = std::env::temp_dir().join(format!("connectome-lifted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let session = Session::open(&dir);
        let cfg = Config { target_chunk_tokens: 50.0, recent_window_tokens: 50.0, l1_holdback_chunks: 0, min_merge_memories: 2, ..Config::default() };
        let mut host = memory_of_30_tokens();
        let count = |n: u64| session.count(&format!("layer{n}"));
        // memories of 30 tokens merge two at a time, so eleven chunks leave L4 0, L2 4 and L1 10 unmerged
        grow(&session, &mut host, 11, true, &cfg);
        assert_eq!((1..6).map(count).collect::<Vec<_>>(), [11, 5, 2, 1, 0]);
        assert_eq!(due(&session, &cfg), None);
        // a window nothing fits: the compile folds all it can and escalates
        let floor = || select(&session, &Fit::library(1.0, &cfg), current(&session), &cfg).expect_err("over the window").actual;
        let before = floor();
        assert_eq!(due(&session, &Config { merge_by_count: true, ..cfg.clone() }), None);   // as the library: nothing to merge
        assert_eq!(due(&session, &cfg), Some(((3, 2), vec![(2, 4), (1, 10)])));   // past the top layer's, into one above the highest
        assert_eq!(lifted(&session, 2, &Config { target_chunk_tokens: 10.0, ..cfg.clone() }), Some(vec![(2, 4), (1, 10)]));   // two at least
        assert_eq!(lifted(&session, 3, &cfg), None);   // three at least, and two wait
        assert_eq!(due(&session, &Config { min_merge_memories: 3, ..cfg.clone() }), Some(((3, 2), vec![(2, 4), (1, 10)])));   // so two, to fold
        // its request: what each merged, one level deeper (L2 4's L1s as recall pairs, L1 10's chunk raw)
        let request = merge_request(&session, 3, &[(2, 4), (1, 10)], &host, &cfg).value(|b| b.at.value()).to_string();
        let at = |needle: &str| request.find(needle).unwrap_or_else(|| panic!("{needle} is not in {request}"));
        let chunk = format!("[\"messages\",{},0]", session.read("layer0", 10)["first"].u64());
        let order = [at("Recall memory L1-8."), at("Recall memory L1-9."), at(&chunk), at("the L1 memories and the slices of recent experience (raw conversation) above")];
        assert!(order.is_sorted() && at("a single L3 memory") > order[3], "{request}");
        assert_eq!(drive(&session, &mut host, &cfg), Ok(true));
        assert_eq!(session.read("layer3", 2)["merges"], json!([[2, 4, 4], [1, 10, 10]]));
        let groups = Recorded::new(&session, &cfg);
        assert_eq!((groups.members((3, 2)), chunks((3, 2), &groups), groups.parent((2, 4))), (vec![(2, 4), (1, 10)], 8..11, Some((3, 2))));
        let after = floor();
        assert!(after < before, "lifting freed nothing: {before} tokens, then {after}");
        assert!(select(&session, &Fit::library(after, &cfg), current(&session), &cfg).is_ok());
        assert_eq!((session.last("compiles").unwrap()["frontier"].clone(), due(&session, &cfg)), (json!([[4, 0], [3, 2]]), None));
        // escalated again, two chunks on: their L1s and the merge they close come first, from the first
        // L1 the lift didn't take; then a lift, which closes a merge of the top layer's
        grow(&session, &mut host, 2, false, &cfg);
        floor();
        let mut formed = vec![];
        while let Some(memory) = due(&session, &cfg) {
            formed.push(memory);
            assert_eq!(drive(&session, &mut host, &cfg), Ok(true));
        }
        let l1 = |p: u64| ((1, p), vec![(0, p)]);
        assert_eq!(formed, [l1(11), l1(12), ((2, 5), vec![(1, 11), (1, 12)]), ((4, 1), vec![(3, 2), (2, 5)]), ((5, 0), vec![(4, 0), (4, 1)])]);
        assert_eq!((session.read("layer2", 5)["merges"].clone(), Recorded::new(&session, &cfg).parent((1, 11))), (json!([11, 12]), Some((2, 5))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_lift_goes_past_a_memory_that_keeps_failing() {
        let dir = std::env::temp_dir().join(format!("connectome-stalled-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let session = Session::open(&dir);
        let cfg = Config { target_chunk_tokens: 50.0, recent_window_tokens: 50.0, l1_holdback_chunks: 0, min_merge_memories: 2, ..Config::default() };
        let mut host = memory_of_30_tokens();
        grow(&session, &mut host, 11, true, &cfg);
        grow(&session, &mut host, 1, false, &cfg); // chunk 11, its memory not yet formed
        assert!(select(&session, &Fit::library(1.0, &cfg), current(&session), &cfg).is_err());
        let mut failing = Answering(Err("down".into()), 0);
        for _ in 0..cfg.attempt_limit {
            assert_eq!(drive(&session, &mut failing, &cfg), Err("down".into()));
        }
        assert_eq!(stalled(&session, &cfg), Some("the memory of chunk 11 failed 3 times since the log last grew".into()));
        assert_eq!(next(&session, &Config { merge_by_count: true, ..cfg.clone() }), None);   // the library waits
        assert_eq!(next(&session, &cfg), Some(((3, 2), vec![(2, 4), (1, 10)])));
        assert_eq!((drive(&session, &mut host, &cfg), session.count("layer3")), (Ok(true), 3));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
