//! A session's companion: another session, made with the user's goal for the first one's work, that
//! talks with it as a collaborator. Each has the message tool, which tells the other something now
//! (and can wait for the answer), and the companion has finish, which turns Auto off. A turn's end
//! goes to the other, which takes a turn on it, while Auto is on; and also when the other messaged it
//! during the turn (so a message always gets its answer), or when the turn called finish (so the
//! worker hears the last word). What a session hears from the other joins its log as the other's
//! message (its participant the other's name), headed by how it came: the other ended its turn (then
//! with what it ran), sends a message while it works, or asks and waits for the answer. It hears it
//! between steps of a turn running, or else takes a turn on it.

use crate::{hub::{Job, Msg, Routed}, settings::{self, SessionSettings}, tools::{Tool, Toolbox, POLL}, App};
use connectome::{json::{Json, Value}, obj, store::Session};
use itertools::Itertools;
use std::{collections::{BTreeSet, HashSet}, sync::mpsc::{channel, RecvTimeoutError}, time::Instant};

/// What a session hears from its partner: who from, and the text as its log keeps it.
pub struct Heard {
    pub from: String,
    pub text: String,
}

/// How a message came: its sender ended its turn, sends it while it works, or asks and waits.
#[derive(Clone, Copy, PartialEq)]
pub enum How { Ended, Sends, Asks }

impl How {
    fn head(self, from: &str) -> String {
        match self {
            How::Ended => format!("{from} ended its turn:"),
            How::Sends => format!("{from} sends you a message while it works:"),
            How::Asks => format!("{from} asks you, and waits for your answer:"),
        }
    }
}

/// What `from` tells its partner, `how`, as the partner's log keeps it.
pub fn heard(from: &str, how: How, text: &str) -> Heard {
    Heard { from: from.into(), text: format!("{}\n\n{text}", how.head(from)) }
}

/// The tools most turns call, which a turn's end lists; message and finish aren't listed, since the
/// partner hears those.
const OWN: [&str; 2] = ["message", "finish"];

/// The calls a turn's end lists, the rest counted.
const LISTED: usize = 20;

/// A turn's end as `name`'s partner hears it: the last text its replies have, then what it ran:
/// bash's commands, edit's paths, web_search's queries and other tools' names, each failed one so
/// marked.
pub fn ended(name: &str, turn: &[Value]) -> String {
    let replies = || turn.iter().filter(|m| m["participant"] == "assistant");
    let said = replies().rev().map(text).find(|t| !t.trim().is_empty()).unwrap_or_else(|| {
        if turn.is_empty() { "(it hasn't said anything yet)" } else { "(it ended the turn without saying anything)" }.into()
    });
    let failed: HashSet<&str> = turn.iter().flat_map(|m| m["content"].arr()).filter(|b| b.kind() == "tool_result" && b["isError"] == true).filter_map(|b| b["toolUseId"].as_str()).collect();
    let calls = replies().flat_map(|m| m["content"].arr()).filter(|b| b.kind() == "tool_use" && !OWN.contains(&b["name"].as_str().unwrap_or("")));
    let calls: Vec<String> = calls.map(|b| shown(b) + if failed.contains(b["id"].as_str().unwrap_or("")) { " (failed)" } else { "" }).collect();
    if calls.is_empty() {
        return said;
    }
    let more = if calls.len() > LISTED { format!(" · and {} more", calls.len() - LISTED) } else { String::new() };
    format!("{said}\n\nWhat {name} ran this turn: {}{more}", calls.iter().take(LISTED).join(" · "))
}

/// A message's text, its text blocks joined.
fn text(m: &Value) -> String {
    m["content"].arr().iter().filter(|b| b.kind() == "text").filter_map(|b| b["text"].as_str()).join("\n\n")
}

/// A call as a turn's end lists it.
fn shown(call: &Value) -> String {
    let (name, input) = (call["name"].as_str().unwrap_or("?"), &call["input"]);
    let field = |k: &str| input[k].as_str().unwrap_or("").to_string();
    match name {
        "bash" => {
            let line = field("command").lines().next().unwrap_or("").to_string();
            format!("$ {}", if line.chars().count() > 80 { line.chars().take(80).collect::<String>() + "…" } else { line })
        }
        "edit" => format!("edit {}", field("path")),
        "web_search" => format!("web_search “{}”", field("query")),
        _ => name.into(),
    }
}

/// Whether a turn's end goes to `partner` though Auto is off: the turn heard `partner` send or ask
/// something (a message in it, or a waiting call's answer), or it called finish.
pub fn owed(turn: &[Value], partner: &str) -> bool {
    let asking = [How::Sends.head(partner), How::Asks.head(partner)];
    let asks = |text: &str| asking.iter().any(|head| text.starts_with(head.as_str()));
    turn.iter().any(|m| {
        let blocks = m["content"].arr();
        (m["participant"] == partner && asks(&text(m)))
            || blocks.iter().any(|b| b.kind() == "tool_result" && b["content"].as_str().is_some_and(asks))
            || blocks.iter().any(|b| b.kind() == "tool_use" && b["name"] == "finish")
    })
}

/// The session's last turn: its messages from the last that started one (the user's or the partner's,
/// not a call's results), among its newest 400.
fn last_turn(s: &Session) -> Vec<Value> {
    let n = s.count("messages");
    let recent: Vec<Value> = if n == 0 { vec![] } else { s.span("messages", n.saturating_sub(400), n - 1).collect() };
    let start = recent.iter().rposition(|m| m["participant"] != "assistant" && m["content"].arr().iter().any(|b| b.kind() == "text"));
    recent[start.unwrap_or(0)..].to_vec()
}

/// What the companion is told, its system prompt (which the user can edit then, like any session's).
fn prompt(me: &str, worker: &str, goal: &str) -> String {
    format!("You're {me}, working alongside {worker}: another AI, in a session of its own, doing a long piece of work for the user. The user has given you both this goal:

{goal}

You're {worker}'s companion on it: a collaborator, not its supervisor. While the user has Auto on, each time {worker} ends a turn, what it said comes to you, with what it ran, and your answer goes back to it as its next message. Long work tends to stall when a turn ends early, with steps left or results unchecked; you're here so it doesn't, and so neither of you works alone. Read what it reports with care. Say what you notice, suggest what to try next, ask about what it hasn't checked, share ideas, disagree when you see it differently, and say so when something is good. When it's on track, a short answer that keeps it going is enough.

Either of you can message the other at any time with the message tool; with wait, the call returns the other's answer. You see only what {worker} tells you (you have no workspace unless the user gives you one), so ask for what you want to see. The user may write to either of you; their messages come as the user's.

When the goal is fully done, or the work needs the user (a decision, access, something only they can give), call finish: Auto goes off, your answer after it is the last that goes to {worker}, and the user reads both your sessions.")
}

/// The tools a session has with its partner, if it has one: message, and for a companion finish.
/// Neither asks the user first.
pub fn tools(app: &App, me: &str, settings: &SessionSettings) -> Toolbox {
    let Some(partner) = settings.partner() else { return Toolbox::default() };
    let who = if settings.companion.is_some() {
        format!("{partner} is your companion on this work: another AI, in a session of its own, which keeps the user's goal for it in mind; talk to it as a collaborator.")
    } else {
        format!("{partner} is the AI doing the work toward the goal you share, in a session of its own.")
    };
    let description = format!("Send {partner} a message now, and keep working, or with wait, wait for its answer. {who} It hears you between its steps, or takes a turn to answer; \
        its messages to you come as its, between your steps. When you end your turn, what you said (and what you ran) goes to {partner} if the user has Auto on, or if it messaged you during the turn.");
    let wait = format!("Wait for its answer, which is then this call's result (for up to {} s; an answer that comes later reaches you as a message). Default false.", app.limit.as_secs());
    let properties = obj! {"text" => obj! {"type" => "string", "description" => "What to tell it"}, "wait" => obj! {"type" => "boolean", "description" => wait}};
    let definition = obj! {"name" => "message", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["text"]}};
    let (hub, from, to, limit) = (app.hub.clone(), me.to_string(), partner.to_string(), app.limit);
    let message = move |input: &Value, stopped: &dyn Fn() -> bool| {
        let text = input["text"].as_str().filter(|t| !t.trim().is_empty()).ok_or("message needs text")?;
        let (wait, (answer, answered), (numbered, _)) = (input["wait"] == true, channel(), channel());
        if wait { // before the message goes, so the answer can't miss it
            hub.send(Routed::To(from.clone(), Msg::Await(answer))).expect("the hub runs");
        }
        let told = heard(&from, if wait { How::Asks } else { How::Sends }, text);
        hub.send(Routed::To(to.clone(), Msg::Job(Job::Hear(told), numbered))).expect("the hub runs");
        if !wait {
            return Ok(format!("{to} has your message.").into());
        }
        let by = Instant::now() + limit;
        loop {
            match answered.recv_timeout(POLL) {
                Ok(heard) => return Ok(heard.text.into()),
                Err(RecvTimeoutError::Timeout) if stopped() => return Err("stopped by the user".into()),
                Err(RecvTimeoutError::Timeout) if Instant::now() < by => continue,
                Err(_) => return Ok(format!("{to} hasn't answered yet; its answer will reach you as a message when it comes.").into()),
            }
        }
    };
    let mut tools = vec![Tool { definition, asks: false, run: Box::new(message) }];
    if settings.accompanies.is_some() {
        let description = format!("Turn Auto off, when the goal is fully done or the work needs the user. Your answer after this is the last that goes to {partner}.");
        let definition = obj! {"name" => "finish", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => obj! {}}};
        let (hub, me, partner) = (app.hub.clone(), me.to_string(), partner.to_string());
        let finish = move |_: &Value, _: &dyn Fn() -> bool| {
            let (reply, answer) = channel();
            hub.send(Routed::To(me.clone(), Msg::Auto(false, reply))).expect("the hub runs");
            answer.recv().expect("the hub answers")?;
            Ok(format!("Auto is off. What you say now is the last that goes to {partner}; the user reads both your sessions.").into())
        };
        tools.push(Tool { definition, asks: false, run: Box::new(finish) });
    }
    Toolbox(tools)
}

/// `worker`'s companion made, as `v` gives it: its name and the user's goal. It starts with the
/// worker's provider, model, window and library settings, no workspace, and the goal in its system
/// prompt; the two are paired with Auto on, and the back-and-forth starts (kick_off).
pub fn make(app: &App, worker: &str, name: &str, goal: &str) -> Result<(), String> {
    let own = app.profile(worker)?;
    if let Some(partner) = own.partner() {
        return Err(format!("{worker} is paired with {partner} already"));
    }
    std::fs::create_dir(app.root.join(name)).map_err(|e| format!("can't make {name}: {e}"))?;
    let first = SessionSettings { workspace: None, allow: BTreeSet::new(), system: Some(prompt(name, worker, goal)), companion: None, accompanies: Some(worker.into()), auto: true, ..own };
    settings::write(&app.profile_path(name), &first)?;
    app.ask_session(worker, |reply| Msg::Pair(name.into(), reply))?;
    kick_off(app, worker)
}

/// Auto just on: unless a turn runs in either session (its end then goes to the other), the worker's
/// last turn goes to its companion, which so starts the back-and-forth.
pub fn kick_off(app: &App, name: &str) -> Result<(), String> {
    let me = app.profile(name)?;
    let (worker, companion) = match (&me.companion, &me.accompanies) {
        (Some(companion), _) => (name.to_string(), companion.clone()),
        (_, Some(worker)) => (worker.clone(), name.to_string()),
        _ => return Err(format!("{name} has no companion")),
    };
    if app.ask_session(&worker, Msg::Turning) || app.ask_session(&companion, Msg::Turning) {
        return Ok(());
    }
    let told = heard(&worker, How::Ended, &ended(&worker, &last_turn(&app.session(&worker))));
    app.ask_session(&companion, |numbered| Msg::Job(Job::Hear(told), numbered));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(who: &str, content: Value) -> Value { obj! {"participant" => who, "content" => content} }

    #[test]
    fn a_turns_end_says_what_it_ran() {
        let call = |id: &str, name: &str, input: Value| obj! {"type" => "tool_use", "id" => id, "name" => name, "input" => input};
        let long = "x".repeat(90);
        let turn = [
            said("pal", vec![obj! {"type" => "text", "text" => "pal ended its turn:\n\ngo on"}].into()),
            said("assistant", vec![obj! {"type" => "text", "text" => "On it."}, call("a", "bash", obj! {"command" => "cargo test\necho done"}), call("b", "edit", obj! {"path" => "src/x.rs"}), call("c", "message", obj! {"text" => "hi"})].into()),
            said("user", vec![obj! {"type" => "tool_result", "toolUseId" => "a", "content" => "fail", "isError" => true}, obj! {"type" => "tool_result", "toolUseId" => "b", "content" => "ok"}].into()),
            said("assistant", vec![call("d", "web_search", obj! {"query" => "bees"}), call("e", "bash", obj! {"command" => long.as_str()}), call("f", "add", obj! {})].into()),
            said("assistant", vec![obj! {"type" => "thinking", "thinking" => "hm"}, obj! {"type" => "text", "text" => "Tests fail; looking."}].into()),
        ];
        let x80 = "x".repeat(80);
        assert_eq!(ended("main", &turn), format!("Tests fail; looking.\n\nWhat main ran this turn: $ cargo test (failed) · edit src/x.rs · web_search “bees” · $ {x80}… · add"));
        assert_eq!(ended("main", &turn[..1]), "(it ended the turn without saying anything)");
        assert_eq!(ended("main", &[]), "(it hasn't said anything yet)");
        let many: Vec<Value> = (0..23).map(|i| call(&i.to_string(), "add", obj! {})).collect();
        assert!(ended("main", &[said("assistant", many.into())]).ends_with("add · add · and 3 more"));
    }

    #[test]
    fn a_message_is_owed_an_answer() {
        let from = |how: How| said("pal", vec![obj! {"type" => "text", "text" => heard("pal", how, "hi").text}].into());
        assert!(owed(&[from(How::Sends)], "pal") && owed(&[from(How::Asks)], "pal"));
        assert!(!owed(&[from(How::Ended)], "pal") && !owed(&[from(How::Sends)], "other"));
        let answered = |how: How| said("user", vec![obj! {"type" => "tool_result", "toolUseId" => "a", "content" => heard("pal", how, "hi").text}].into());
        assert!(owed(&[answered(How::Sends)], "pal") && !owed(&[answered(How::Ended)], "pal"));
        assert!(owed(&[said("assistant", vec![obj! {"type" => "tool_use", "id" => "f", "name" => "finish", "input" => obj! {}}].into())], "pal"));
    }
}
