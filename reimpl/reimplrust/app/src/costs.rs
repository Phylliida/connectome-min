//! What a session's requests used: its costs log (costs.jsonl), a record for each step's reply and each
//! round of memories (Cost). A reply's record also says which message it is and which compile its
//! request was built from: so the reply's prompt, as its provider counted it, calibrates the compile
//! after it, and the page shows how the context folded between replies (Fold). A record the app can't
//! read (a hand edit) is passed over.

use crate::{claude::Limits, llm::Usage};
use connectome::{js::py_round, store::Session};
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// What requests were for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum For { Reply, Memories }

/// A record of the costs log: what requests `used`, and the session's running totals by what they
/// were for; a reply's also says which message it is and which compile its request was built from,
/// and, for a reply that took several requests (a Claude Code turn's, whose `used` sums them), the
/// last one's prompt, the context it ended with; and, from a run of Claude Code on a plan, the plan's
/// limits as it said them.
#[derive(Serialize, Deserialize)]
struct Cost {
    #[serde(rename = "for")] what: For,
    #[serde(default, skip_serializing_if = "Option::is_none")] message: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] compile: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] prompt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] limits: Option<Limits>,
    used: Usage,
    total: BTreeMap<For, Usage>,
}

/// What the session's requests have used, as the page shows it: by what they were for, and in all.
#[derive(Clone, Serialize)]
pub struct Spent { #[serde(flatten)] by: BTreeMap<For, Usage>, total: Usage }

impl Spent {
    /// None before any request.
    fn of(by: BTreeMap<For, Usage>) -> Option<Spent> {
        (!by.is_empty()).then(|| Spent { total: by.values().copied().sum(), by })
    }
}

/// A node of a frontier: [layer, id].
pub type Node = (u64, u64);

/// What the app reads of a compile record: the frontier it showed, and its calibration.
#[derive(Clone, Serialize, Deserialize)]
pub struct Compile { pub frontier: Vec<Node>, pub multiplier: f64, pub armed: f64 }

/// What the app reads of a chunk's or memory's record: the messages it covers, and its price (a
/// chunk's raw tokens, which the multiplier scales, or a memory's recall).
#[derive(Deserialize)]
struct Covering { first: u64, last: u64, #[serde(default)] raw: f64, #[serde(default)] recall: f64 }

/// How the context folded between two compiles: the nodes it no longer shows (`from`) and those it
/// shows instead (`into`), and the tokens that saves (below 0 when it unfolds).
#[derive(Clone, Serialize)]
pub struct Fold { from: Vec<Node>, into: Vec<Node>, saved: f64 }

/// Record `i` of a file, if the app can read it.
fn read<T: DeserializeOwned>(s: &Session, file: &str, i: u64) -> Option<T> {
    serde_json::from_value(s.read(file, i)).ok()
}

/// The costs log's records, newest first, as they're read.
fn costs(s: &Session) -> impl Iterator<Item = Cost> + '_ {
    (0..s.count("costs")).rev().filter_map(|i| read(s, "costs", i))
}

/// The records of the session's replies, newest first.
fn replies(s: &Session) -> impl Iterator<Item = Cost> + '_ {
    costs(s).filter(|c| c.what == For::Reply)
}

/// What requests `used` for `what` joins the costs log, if they used anything, with the running
/// totals, for a reply of several requests the last one's `prompt`, and the plan's `limits` Claude Code
/// said; and the totals as the page shows them.
pub fn spend(s: &Session, what: For, used: Usage, prompt: Option<f64>, limits: Option<Limits>) -> Option<Spent> {
    if used.input + used.output == 0.0 { return None }
    let mut total = costs(s).next().map(|c| c.total).unwrap_or_default();
    total.entry(what).and_modify(|t| *t = *t + used).or_insert(used);
    let reply = what == For::Reply;
    let (message, compile) = (reply.then(|| s.count("messages")), reply.then(|| s.count("compiles").checked_sub(1)).flatten());
    let cost = Cost { what, message, compile, prompt, limits, used, total };
    s.append("costs", serde_json::to_value(&cost).expect("a cost is JSON"));
    Spent::of(cost.total)
}

/// What the session's requests have used; None before any.
pub fn spent(s: &Session) -> Option<Spent> {
    Spent::of(costs(s).next()?.total)
}

/// The plan's limits as the session's last run of Claude Code said them; None without any.
pub fn limits(s: &Session) -> Option<Limits> {
    costs(s).next()?.limits
}

/// The last reply's prompt in tokens, as its provider counted it (its last request's); None before any.
pub fn prompted(s: &Session) -> Option<f64> {
    replies(s).next().map(|c| c.prompt.unwrap_or(c.used.input))
}

/// The last reply's prompt (its last request's), unless a compile has learnt from it: the one its
/// request was built from is no longer the last.
pub fn unlearnt(s: &Session) -> Option<f64> {
    replies(s).next().filter(|c| c.compile.is_some_and(|i| i + 1 == s.count("compiles"))).map(|c| c.prompt.unwrap_or(c.used.input))
}

/// The compile a reply's request was built from.
fn compile(s: &Session, reply: &Cost) -> Option<Compile> {
    read(s, "compiles", reply.compile?)
}

/// How the context folded before each reply from message `from` on, by the reply's message: its
/// compile against the reply's before it (folded). Replies logged without their compile have none.
pub fn folds(s: &Session, from: u64) -> HashMap<u64, Fold> {
    let linked = replies(s).take_while(|c| c.compile.is_some()).take_while_inclusive(|c| c.message.unwrap_or(0) >= from);
    let compiles = linked.map(|c| (c.message.unwrap_or(0), compile(s, &c)));
    compiles.tuple_windows().filter_map(|((at, after), (_, before))| Some((at, folded(s, &before?, &after?)?))).collect()
}

/// How the context folds for the reply to come: the last compile against the last reply's.
pub fn fold(s: &Session) -> Option<Fold> {
    let last = serde_json::from_value(s.last("compiles")?).ok()?;
    folded(s, &compile(s, &replies(s).next()?)?, &last)
}

/// What a compile's frontier (`after`) shows differently from an earlier one's (`before`) where both
/// reach (chunks closed since aren't a change), priced as the planner prices them at `after`'s
/// multiplier; None when it shows the same.
fn folded(s: &Session, before: &Compile, after: &Compile) -> Option<Fold> {
    let covering = |&(layer, id): &Node| read::<Covering>(s, &format!("layer{layer}"), id);
    let reach = match before.frontier.last() { // the last message the earlier one shows
        Some(n) => covering(n)?.last,
        None => 0,
    };
    let from: Vec<Node> = before.frontier.iter().copied().filter(|n| !after.frontier.contains(n)).collect();
    let into: Vec<Node> = after.frontier.iter().copied().filter(|n| !before.frontier.contains(n) && covering(n).is_some_and(|c| c.first <= reach)).collect();
    let price = |n: &Node| covering(n).map_or(0.0, |c| if n.0 == 0 { py_round(c.raw * after.multiplier) } else { c.recall });
    let saved = from.iter().map(price).sum::<f64>() - into.iter().map(price).sum::<f64>();
    (!from.is_empty()).then_some(Fold { from, into, saved })
}
