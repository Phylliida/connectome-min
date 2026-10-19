//! What a session's requests used: its costs log (costs.jsonl), a record for each step's reply and each
//! round of memories, {id, for, message?, compile?, used, total}. `used` is what the requests used
//! (llm::Usage), and `total` the session's running totals by what they were for ({reply, memories}). A
//! reply's record also says which message it is and which compile its request was built from: so the
//! reply's prompt, as its provider counted it, calibrates the compile after it, and the page shows
//! how the context folded between replies.

use crate::llm::Usage;
use connectome::{js::py_round, json::{sparse, Json, Value}, obj, store::Session};
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};

/// What requests `used` for `what` ("reply" or "memories") joins the costs log, if they used anything,
/// with the running totals; and the event that tells the pages: the totals as they show them
/// ({"spent"}), and a reply's prompt ({"prompt"}).
pub fn spend(s: &Session, what: &str, used: Usage) -> Option<Value> {
    if used.input + used.output == 0.0 { return None }
    let mut totals = totals(s);
    totals.entry(what.into()).and_modify(|total| *total = *total + used).or_insert(used);
    let reply = what == "reply";
    let (message, compile) = (reply.then(|| s.count("messages")), reply.then(|| s.count("compiles").saturating_sub(1)));
    let total = serde_json::to_value(&totals).expect("usage is JSON");
    s.append("costs", sparse(obj! {"for" => what, "message" => message, "compile" => compile, "used" => used, "total" => total}));
    Some(sparse(obj! {"spent" => shown(&totals), "prompt" => reply.then_some(used.input)}))
}

/// The records of the session's replies, newest first, as they're read.
fn replies(s: &Session) -> impl Iterator<Item = Value> + '_ {
    (0..s.count("costs")).rev().map(|i| s.read("costs", i)).filter(|c| c["for"] == "reply")
}

/// The last reply's prompt in tokens, as its provider counted it; None before any.
pub fn prompted(s: &Session) -> Option<f64> {
    replies(s).next().and_then(|c| c["used"]["input"].as_f64())
}

/// The last reply's prompt, unless a compile has learnt from it: the one its request was built from is
/// no longer the last.
pub fn unlearnt(s: &Session) -> Option<f64> {
    replies(s).next().filter(|c| c["compile"].as_u64().is_some_and(|i| i + 1 == s.count("compiles"))).and_then(|c| c["used"]["input"].as_f64())
}

/// What the session's requests have used, by what they were for: the log's last running totals.
fn totals(s: &Session) -> BTreeMap<String, Usage> {
    s.last("costs").and_then(|c| serde_json::from_value(c["total"].clone()).ok()).unwrap_or_default()
}

/// What the session's requests have used, as the page shows it (shown); None before any.
pub fn spent(s: &Session) -> Option<Value> {
    shown(&totals(s))
}

/// Totals as the page shows them, with all of them ("total"); None without any.
fn shown(totals: &BTreeMap<String, Usage>) -> Option<Value> {
    let mut shown = totals.clone();
    shown.insert("total".into(), totals.values().copied().sum());
    (!totals.is_empty()).then(|| serde_json::to_value(shown).expect("usage is JSON"))
}

/// The compile a reply's request was built from.
fn compile(s: &Session, reply: &Value) -> Value {
    s.read("compiles", reply["compile"].u64())
}

/// How the context folded before each reply from message `from` on, by the reply's message: its
/// compile against the reply's before it (folded). Replies logged without their compile have none.
pub fn folds(s: &Session, from: u64) -> HashMap<u64, Value> {
    let linked = replies(s).take_while(|c| c.get("compile").is_some()).take_while_inclusive(|c| c["message"].u64() >= from);
    let compiles = linked.map(|c| (c["message"].u64(), compile(s, &c)));
    compiles.tuple_windows().filter_map(|((at, after), (_, before))| Some((at, folded(s, &before, &after)?))).collect()
}

/// How the context folds for the reply to come: the last compile against the last reply's.
pub fn fold(s: &Session) -> Option<Value> {
    let reply = replies(s).next().filter(|c| c.get("compile").is_some())?;
    folded(s, &compile(s, &reply), &s.last("compiles")?)
}

/// What a compile's frontier (`after`) shows differently from an earlier one's (`before`) where both
/// reach (chunks closed since aren't a change): the nodes it no longer shows ("from") and those it
/// shows instead ("into"), each [layer, id], and the tokens that saves ("saved", below 0 when it
/// unfolds), as the planner prices them at `after`'s multiplier; None when it shows the same.
fn folded(s: &Session, before: &Value, after: &Value) -> Option<Value> {
    let nodes = |c: &Value| -> Vec<(u64, u64)> { c["frontier"].arr().iter().map(|n| (n[0].u64(), n[1].u64())).collect() };
    let record = |&(layer, id): &(u64, u64)| s.read(&format!("layer{layer}"), id);
    let (old, new, m) = (nodes(before), nodes(after), after["multiplier"].as_f64().unwrap_or(1.0));
    let reach = old.last().map_or(0, |n| record(n)["last"].u64()); // the last message the earlier one shows
    let from: Vec<_> = old.iter().copied().filter(|n| !new.contains(n)).collect();
    let into: Vec<_> = new.iter().copied().filter(|n| !old.contains(n) && record(n)["first"].u64() <= reach).collect();
    let price = |n: &(u64, u64)| if n.0 == 0 { py_round(record(n)["raw"].as_f64().unwrap_or(0.0) * m) } else { record(n)["recall"].as_f64().unwrap_or(0.0) };
    let saved = from.iter().map(price).sum::<f64>() - into.iter().map(price).sum::<f64>();
    let json = |nodes: Vec<(u64, u64)>| serde_json::to_value(nodes).expect("nodes are JSON");
    (!from.is_empty()).then(|| obj! {"from" => json(from), "into" => json(into), "saved" => saved})
}
