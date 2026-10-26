//! The gateway's fold of a Claude Code session's requests (claude.rs, gateway.rs). Claude Code keeps
//! its whole conversation and sends all of it with every request. The session's context is compiled
//! from its log as a step's is (Worker::compile: past fold_above it folds, down to fold_to, as many
//! memories forming first as that takes, while Claude Code waits for its answer), and once memories
//! stand for the oldest part of it, the request goes upstream with the context up to its last memory in
//! place of the messages of Claude Code's that part holds, then Claude Code's own messages as they are,
//! from where they take up the log (kept_from): so its system prompt, tools, reminders and the system
//! messages it adds mid-session, which the log doesn't have, go as Claude Code sent them, and what of
//! those its folded messages held goes ahead of the memories. A request folds at a turn's start;
//! mid-turn, only once the context wouldn't fit otherwise; and what goes ahead changes only as it folds.
//!
//! A reasoning block a request carries back is bound by its signature to the messages before it, as
//! they were when it was written (Claude's preserved thinking: an edit before it is a 400), so each
//! time what goes before Claude Code's messages changes, the reasoning they hold from before then goes
//! (their text and calls stay), the same on every request after: prefixes.jsonl records each change,
//! with the newest reasoning block Claude Code's messages had then. What comes after it was written
//! with what goes before it now, so it stays until the next change. Before any memory, a request goes
//! as it came.

use crate::{llm, work::{Setup, Worker}};
use connectome::{blocks::{density_tokens, At}, emit::{located, Reader}, js::digest, json::{self, Json, Value}, obj, turns::Turn};

impl Worker<'_> {
    /// Claude Code's request `body` as it goes upstream, compiled; None to send it as it came. `reported`
    /// is the last request's prompt, by the API's count, unless a compile has learnt from it.
    pub fn fold(&self, body: &Value, reported: Option<f64>) -> Result<Option<Vec<u8>>, String> {
        let messages = body["messages"].arr();
        let mut setup = Setup::new(self.app, self.name)?;
        // what the request carries besides the context compiled from the log: Claude Code's system prompt and
        // tools, and what its messages hold that the log doesn't, its reminders and system messages (a few
        // thousand tokens, which in a small context would keep calibration from ever learning)
        let mut unlogged = 0.0;
        for m in messages {
            for b in content(m) {
                let text = b["text"].as_str().unwrap_or("");
                if m["role"] == "system" || text.starts_with("<system-reminder>") { unlogged += density_tokens(&json::to_string(&b)) }
            }
        }
        setup.overhead = density_tokens(&json::to_string(&body["system"])) + density_tokens(&json::to_string(&body["tools"])) + unlogged;
        // what goes ahead changes only as it folds, where it unfolds back to fold_to: every other change would cost the
        // cache and the reasoning (unfolding below unfold_below, as calibration moved its prices, would change it often)
        setup.settings.config.unfold_below = 0.0;
        // 1) the context, folding past fold_above at a turn's start, and mid-turn only once it wouldn't fit
        let fold_above = if starts_turn(messages) { setup.settings.fold_above } else { f64::INFINITY };
        let turns = self.compile(&setup, reported, fold_above)?;
        // 2) the part of it up to its last memory, and where Claude Code's messages take up the log after it
        let remembered = |t: &Turn| t.content.iter().any(|b| matches!(&b.at, At::Disk(file, ..) if file.starts_with("layer")));
        let (ahead, cut) = match turns.iter().rposition(remembered) {
            None => (&turns[..0], 0),
            Some(last) => {
                let n = self.s.count("messages");
                let mut from = n;
                for b in turns[last + 1..].iter().flat_map(|t| &t.content) {
                    if let At::Disk(file, i, ..) = &b.at && file == "messages" { from = from.min(*i) }
                }
                let logged: Vec<Value> = if from < n { self.s.span("messages", from, n - 1).collect() } else { vec![] };
                let Some(cut) = kept_from(&logged, messages) else {
                    eprintln!("{}: Claude Code's request has none of the log's messages from {from} on, so it goes unfolded", self.name);
                    return Ok(None);
                };
                (&turns[..=last], cut)
            }
        };
        // 3) the reasoning Claude Code's messages hold from before what goes ahead of them last changed: a change
        //    is recorded with the newest reasoning block they had then (or again, if that has gone), and the reasoning
        //    goes from its message and those before it
        let shape = digest(&obj! {"ahead" => ahead.iter().map(|t| located(&t.content)).collect::<Vec<_>>(), "cut" => cut});
        let last = self.s.last("prefixes");
        let gone = |record: &Value| record["reasoning"].as_str().is_some_and(|mark| !reasoning(messages).any(|(_, d)| d == mark));
        let changed = last.as_ref().map_or(!ahead.is_empty(), |record| record["shape"] != shape.as_str() || gone(record));
        let mark = if changed {
            let newest = reasoning(messages).last().map(|(_, d)| d);
            self.s.append("prefixes", obj! {"shape" => shape, "reasoning" => newest.clone()});
            newest
        } else {
            last.and_then(|record| record["reasoning"].as_str().map(String::from))
        };
        let until = mark.and_then(|mark| reasoning(messages).find(|(_, d)| *d == mark)).map(|(i, _)| i).filter(|&i| i >= cut);
        if ahead.is_empty() && until.is_none() { return Ok(None) }
        // 4) what goes ahead as the API's messages, a role's consecutive turns joined (its reasoning out, and no
        //    cache breakpoints, as Claude Code's own fill the four a request may have)
        let mut reader = Reader::new(&self.s, true);
        let mut rendered: Vec<Value> = vec![];
        for t in ahead {
            let role = if t.participant == "assistant" { "assistant" } else { "user" };
            let blocks: Vec<Value> = llm::carried(t, false).into_iter().map(|b| llm::to_api(reader.load(b), false)).collect();
            match rendered.last_mut() {
                _ if blocks.is_empty() => {}
                Some(last) if last["role"] == role => last["content"].as_array_mut().expect("a message's blocks").extend(blocks),
                _ => rendered.push(obj! {"role" => role, "content" => blocks}),
            }
        }
        // 5) with what Claude Code's folded messages hold that the log doesn't, and so no memory does: the
        //    reminders it began its session with (CLAUDE.md, the user's context), in the first message, and its
        //    system messages (where it runs, its MCP servers' instructions, the tools it added or took away, its
        //    effort), after it; but those that wrap a message said mid-turn, which the log has, and turn-scoped ones
        let folded = &messages[..cut];
        let reminders: Vec<Value> = folded.first().filter(|m| m["role"] == "user").map_or(vec![], |m| {
            m["content"].arr().iter().filter(|b| b["text"].as_str().is_some_and(|t| t.starts_with("<system-reminder>"))).map(uncached).collect()
        });
        let (mut standing, mut effort) = (vec![], None);
        for m in folded.iter().filter(|m| m["role"] == "system" && m.get("clear_at").is_none()) {
            let blocks: Vec<Value> = content(m).iter().map(uncached).collect();
            let wraps = blocks.first().and_then(|b| b["text"].as_str()).and_then(|t| t.lines().next()).is_some_and(|line| line.ends_with("while you were working:"));
            if wraps { continue }
            standing.extend(blocks);
            effort = m.get("output_config").cloned().or(effort);
        }
        if let Some(first) = rendered.first_mut() && !reminders.is_empty() {
            first["content"].as_array_mut().expect("a message's blocks").splice(0..0, reminders);
        }
        // a system message follows the user's and comes before the assistant's: after the first message, which a reply follows
        if (!standing.is_empty() || effort.is_some()) && rendered.get(1).is_some_and(|m| m["role"] == "assistant") {
            rendered.insert(1, json::sparse(obj! {"role" => "system", "content" => standing, "output_config" => effort}));
        }
        // 6) then Claude Code's messages from the cut, the reasoning that's gone out
        let mut sent = messages[cut..].to_vec();
        if let Some(i) = until {
            for message in sent.iter_mut().take(i + 1 - cut) {
                if let Some(blocks) = message["content"].as_array_mut() {
                    blocks.retain(|b| !matches!(b.kind(), "thinking" | "redacted_thinking"));
                }
            }
            sent.retain(|m| !m["content"].as_array().is_some_and(Vec::is_empty)); // a reply that was all reasoning
        }
        // 7) the request: Claude Code's fields, those messages in place of its own
        let mut request = serde_json::Map::new();
        for (k, v) in body.as_object().into_iter().flatten() {
            request.insert(k.clone(), if k == "messages" { Value::Array(std::mem::take(&mut rendered)) } else { v.clone() });
        }
        request["messages"].as_array_mut().expect("the messages").extend(sent);
        Ok(Some(json::to_string(&request).into_bytes()))
    }
}

/// Where Claude Code's `messages` take up the `logged` messages (the log's from where the memories end):
/// the message of theirs that holds the earliest of them they have, each found walking back from the
/// last, before the match of the one after it, by a block's key, else its text; one not found is
/// passed over (of a reply the log has in parts, or results it has one by one, which are one message
/// of Claude Code's, the last is found); then back to a message that starts a turn or a reply, not
/// results, whose calls would be left behind, nor a system message, which follows another. None if
/// they have none of them.
fn kept_from(logged: &[Value], messages: &[Value]) -> Option<usize> {
    let (mut end, mut matched) = (messages.len(), None); // `end`: the match of the message after
    for message in logged.iter().rev() {
        let assistant = message["participant"] == "assistant";
        let found = message["content"].arr().iter().find_map(key);
        let text = message["content"].arr().iter().find(|b| b.kind() == "text").and_then(|b| b["text"].as_str()).filter(|t| !t.trim().is_empty());
        let at = (0..end).rev().find(|&j| {
            let sent = &messages[j];
            (sent["role"] == "assistant") == assistant && match (found, text) {
                (Some(found), _) => sent["content"].arr().iter().any(|b| key(b) == Some(found)),
                (None, Some(text)) => says(sent, text),
                (None, None) => false,
            }
        });
        if let Some(j) = at {
            (end, matched) = (j, Some(j));
        }
    }
    let mut cut = matched?;
    while cut > 0 && (messages[cut]["role"] == "system" || messages[cut]["content"].arr().iter().any(|b| b.kind() == "tool_result")) {
        cut -= 1;
    }
    Some(cut)
}

/// Whether a request starts a turn: its last message but Claude Code's system ones is the user's,
/// not a call's results.
fn starts_turn(messages: &[Value]) -> bool {
    let last = messages.iter().rev().find(|m| m["role"] != "system");
    last.is_some_and(|m| m["role"] == "user" && !m["content"].arr().iter().any(|b| b.kind() == "tool_result"))
}

/// Whether a message of Claude Code's says `text`: a text block of it, or its text, is it, or ends with
/// it on a line of its own, as a message the user sends mid-turn does in the system message Claude
/// Code wraps it in ("The user sent a new message while you were working:\nTEXT").
pub fn says(sent: &Value, text: &str) -> bool {
    let ends = |s: &str| s == text || s.ends_with(&format!("\n{text}"));
    match &sent["content"] {
        Value::String(s) => ends(s),
        content => content.arr().iter().any(|b| b.kind() == "text" && b["text"].as_str().is_some_and(ends)),
    }
}

/// What a block is known by in both the log and Claude Code's messages, and what it is: a call's id,
/// a result's call's (the API's tool_use_id, the log's toolUseId), reasoning's signature (or its data,
/// redacted).
fn key(b: &Value) -> Option<(&'static str, &str)> {
    match b.kind() {
        "tool_use" => b["id"].as_str().map(|id| ("call", id)),
        "tool_result" => b["tool_use_id"].as_str().or(b["toolUseId"].as_str()).map(|id| ("result", id)),
        "thinking" => b["signature"].as_str().filter(|s| !s.is_empty()).map(|s| ("reasoning", s)),
        "redacted_thinking" => b["data"].as_str().map(|d| ("reasoning", d)),
        _ => None,
    }
}

/// A message's blocks: a string's is one of text.
fn content(m: &Value) -> Vec<Value> {
    match &m["content"] {
        Value::String(text) => vec![obj! {"type" => "text", "text" => text.as_str()}],
        content => content.arr().to_vec(),
    }
}

/// A block of Claude Code's without its cache breakpoint, for what goes ahead, which has none.
fn uncached(b: &Value) -> Value {
    let mut b = b.clone();
    if let Some(fields) = b.as_object_mut() { fields.shift_remove("cache_control"); }
    b
}

/// Each reasoning block of Claude Code's messages, in order: its message, and its key's digest, which
/// prefixes.jsonl records (a signature runs to kilobytes).
fn reasoning(messages: &[Value]) -> impl Iterator<Item = (usize, String)> + '_ {
    messages.iter().enumerate().flat_map(|(i, m)| m["content"].arr().iter().filter_map(move |b| match key(b) {
        Some(("reasoning", k)) => Some((i, digest(&k.into()))),
        _ => None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(t: &str) -> Value { obj! {"type" => "text", "text" => t} }
    fn call(id: &str) -> Value { obj! {"type" => "tool_use", "id" => id, "name" => "Bash", "input" => obj! {}} }
    fn logged(participant: &str, content: Vec<Value>) -> Value { obj! {"participant" => participant, "content" => content} }
    fn sent(role: &str, content: Value) -> Value { obj! {"role" => role, "content" => content} }

    #[test]
    fn claude_codes_messages_take_up_the_log_where_they_hold_its_first() {
        // a reply the log has in two parts (asked about between its calls), their results one by one, a
        // message said mid-turn, which Claude Code wraps in a system message, and a message said three times
        let thinking = obj! {"type" => "thinking", "thinking" => "", "signature" => "s1"};
        let result = |id: &str, field: &str| obj! {"type" => "tool_result", field => id, "content" => "ok"};
        let messages = vec![
            sent("user", vec![text("<system-reminder>\nHi.\n</system-reminder>"), text("go")].into()),
            sent("system", "# Environment".into()),
            sent("assistant", vec![thinking.clone(), text("Calling."), call("a"), call("b")].into()),
            sent("user", vec![result("a", "tool_use_id"), result("b", "tool_use_id")].into()),
            sent("system", "The user sent a new message while you were working:\nalso this".into()),
            sent("assistant", vec![text("Done.")].into()),
            sent("user", vec![text("go")].into()),
            sent("assistant", vec![text("Again.")].into()),
            sent("user", vec![text("go")].into()),
        ];
        let log = [
            logged("user", vec![text("go")]),
            logged("assistant", vec![thinking, text("Calling."), call("a")]),
            logged("assistant", vec![call("b")]),
            logged("user", vec![result("a", "toolUseId")]),
            logged("user", vec![result("b", "toolUseId")]),
            logged("user", vec![text("also this")]),
            logged("assistant", vec![text("Done.")]),
            logged("user", vec![text("go")]),
            logged("assistant", vec![text("Again.")]),
            logged("user", vec![text("go")]),
        ];
        // from each of the log's messages, where Claude Code's take it up: the results and the message said
        // mid-turn from the reply whose calls they answer, which starts with its first part
        let cuts = [0, 2, 2, 2, 2, 2, 5, 6, 7, 8];
        for (from, expected) in cuts.into_iter().enumerate() {
            assert_eq!(kept_from(&log[from..], &messages), Some(expected), "from the log's message {from}");
        }
        // a reply's later part alone, and its results alone, by their calls' ids, which aren't first in Claude Code's message
        assert_eq!((kept_from(&log[2..3], &messages), kept_from(&log[3..5], &messages)), (Some(2), Some(2)));
        // a message of the log's from before Claude Code's session (on another provider) that it hasn't: from
        // the first it has, though an assistant's reply of Claude Code's says the same
        let earlier = [vec![logged("user", vec![text("Calling.")])], log[6..].to_vec()].concat();
        assert_eq!(kept_from(&earlier, &messages), Some(5));
        assert_eq!(kept_from(&[logged("user", vec![text("never said")])], &messages), None);
        // the same said twice in a row, each its own message of Claude Code's
        let twice = [sent("user", vec![text("yes")].into()), sent("user", vec![text("yes")].into())];
        assert_eq!(kept_from(&[logged("user", vec![text("yes")]), logged("user", vec![text("yes")])], &twice), Some(0));
    }

    #[test]
    fn a_turn_starts_with_what_the_user_says() {
        let results = sent("user", vec![obj! {"type" => "tool_result", "tool_use_id" => "a", "content" => "ok"}].into());
        let said = sent("user", vec![text("go")].into());
        let system = sent("system", "# MCP Server Instructions".into());
        assert!(starts_turn(&[said.clone()]) && starts_turn(&[said.clone(), system.clone()]));
        assert!(!starts_turn(&[said, sent("assistant", vec![call("a")].into()), results.clone(), system]) && !starts_turn(&[results]));
    }
}
