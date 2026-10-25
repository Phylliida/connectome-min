//! Channels: named logs of posts that sessions take part in. A post is the user's, or a session's,
//! made with its message_channel tool in a channel it's in; it goes at once to each session in the
//! channel that has inputs on but the one that posted it, as a message headed by the channel and who
//! posted it, the way a session hears its companion (companion.rs): between the steps of a turn
//! running, or else in a turn of its own. A session's settings say which channels it's in and whether
//! it has inputs on in each (`channels`), and the most posts a day it may make in each (`per_day`, over
//! any 24 hours: past it, message_channel is refused), which only the user sets. Each post a session hears ends with
//! a reminder of how to answer in the channel, since an answer there must be posted, then the
//! channel's description, if the user gave it one (say, how sparingly to post). A channel is
//! ROOT/.channels/NAME: posts.jsonl, each post {id, from, text, at}, `from` the session that posted it
//! (null: the user) and `at` when, in seconds since 1970, and channel.json, its settings (About); the hub alone writes both, as it routes
//! each post (hub's Routed::Post) and each change of settings. No session can be named .channels.

use crate::{companion::Heard, hub::Routed, settings::{self, Membership, SessionSettings}, tools::{Tool, Toolbox}, App};
use connectome::{json::Value, obj, store::Session};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::mpsc::channel, time::{SystemTime, UNIX_EPOCH}};

/// The window a limit of posts a day counts them over, in seconds.
const DAY: u64 = 86_400;

/// A channel's settings (channel.json): its description, which each post its sessions hear ends with.
#[derive(Default, Serialize, Deserialize)]
pub struct About {
    #[serde(default, skip_serializing_if = "Option::is_none")] pub description: Option<String>,
}

/// A post on its way to a channel's log: the channel, the session that posted it (None: the user), and
/// its text.
pub struct Post {
    pub channel: String,
    pub from: Option<String>,
    pub text: String,
}

/// A post as logged, the sessions that heard it, and, if its session has a limit there, how many posts
/// it has left of it: (left, limit).
pub struct Posted {
    pub post: Value,
    pub heard: Vec<String>,
    pub left: Option<(u64, u64)>,
}

impl App {
    fn channels_path(&self) -> PathBuf { self.root.join(".channels") }

    /// A channel's log; it must exist (channel_exists), since opening one makes its directory.
    pub fn channel(&self, name: &str) -> Session { Session::open(self.channels_path().join(name)) }

    pub fn channel_exists(&self, name: &str) -> bool { self.channels_path().join(name).is_dir() }

    fn about_path(&self, name: &str) -> PathBuf { self.channels_path().join(name).join("channel.json") }

    /// A channel's settings, as its file has them (none yet: the defaults).
    pub fn about(&self, name: &str) -> About { settings::read(&self.about_path(name)).unwrap_or_default() }
}

/// A new channel, named `name`, unless there's one.
pub fn make(app: &App, name: &str) -> Result<(), String> {
    std::fs::create_dir_all(app.channels_path()).map_err(|e| format!("can't make the channels' directory: {e}"))?;
    std::fs::create_dir(app.channels_path().join(name)).map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => format!("there's a channel {name} already"),
        _ => format!("can't make {name}: {e}"),
    })
}

/// Channel `name`'s description set (blank: none). Only the hub calls this, as it writes the settings
/// files.
pub fn describe(app: &App, name: &str, description: &str) -> Result<(), String> {
    if !app.channel_exists(name) {
        return Err(format!("no channel {name}"));
    }
    let description = Some(description.trim()).filter(|d| !d.is_empty()).map(str::to_string);
    settings::write(&app.about_path(name), &About { description })
}

/// Every channel: its name, its description, its posts, and the sessions in it, each with whether it
/// has inputs on.
pub fn listing(app: &App) -> Value {
    let names = std::fs::read_dir(app.channels_path()).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).sorted();
    let profiles = app.profiles();
    Value::Array(names.map(|name| {
        let members: Vec<Value> = profiles.iter().filter_map(|(member, s)| s.channels.get(&name).map(|m| obj! {"name" => member.as_str(), "inputs" => m.inputs, "per_day" => m.limit()})).collect();
        obj! {"name" => name.as_str(), "description" => app.about(&name).description, "posts" => app.channel(&name).count("posts"), "members" => members}
    }).collect())
}

/// The sessions that hear channel `name`'s posts: those in it with inputs on.
pub fn hearing(app: &App, name: &str) -> Vec<String> {
    app.profiles().into_iter().filter(|(_, s)| s.channels.get(name).is_some_and(|m| m.inputs)).map(|(member, _)| member).collect()
}

/// A post joins its channel's log, stamped with the time, unless its session has made the most posts
/// its settings allow there in the last 24 hours: the record it's logged as, and, under a limit, how
/// many posts the session has left of it. Only the hub calls this.
pub fn append(app: &App, post: &Post) -> Result<(Value, Option<(u64, u64)>), String> {
    if !app.channel_exists(&post.channel) {
        return Err(format!("no channel {}", post.channel));
    }
    let at = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let limited = post.from.as_deref().and_then(|me| Some((me, app.profile(me).ok()?.channels.get(&post.channel)?.limit()?)));
    let left = match limited {
        Some((me, limit)) => {
            let made = recent(app, &post.channel, me, at);
            if made.len() as u64 >= limit {
                let frees = made[made.len() - limit as usize] + DAY; // when the oldest post that keeps it at its limit is a day old
                return Err(format!("Rate limit exceeded: you've posted {} times in #{} in the last 24 hours, the most the user allows there; your next post can go in {}.",
                    made.len(), post.channel, how_long(frees.saturating_sub(at))));
            }
            Some((limit - made.len() as u64 - 1, limit))
        }
        None => None,
    };
    Ok((app.channel(&post.channel).append("posts", obj! {"from" => post.from.as_deref(), "text" => post.text.as_str(), "at" => at}), left))
}

/// When session `me`'s posts in channel `name` in the 24 hours up to `now` were made, oldest first
/// (posts are logged in time order, so they're the newest; one logged before posts had times counts as
/// long ago).
fn recent(app: &App, name: &str, me: &str, now: u64) -> Vec<u64> {
    let (log, since) = (app.channel(name), now.saturating_sub(DAY));
    let (mut low, mut high) = (0, log.count("posts")); // to the first post since then
    while low < high {
        let mid = (low + high) / 2;
        if log.read("posts", mid)["at"].as_u64().unwrap_or(0) < since { low = mid + 1 } else { high = mid }
    }
    let n = log.count("posts");
    if low >= n { return vec![] }
    log.span("posts", low, n - 1).filter(|p| p["from"] == me).map(|p| p["at"].as_u64().unwrap_or(0)).collect()
}

/// A wait, as a person says it: hours and minutes, or under a minute.
fn how_long(seconds: u64) -> String {
    match (seconds / 3600, seconds % 3600 / 60) {
        (0, 0) => "under a minute".into(),
        (0, m) => format!("{m} min"),
        (h, m) => format!("{h} h {m} min"),
    }
}

/// A post as a session that hears its channel gets it: from the channel, headed by who posted it, and
/// ending with how to answer there, then the channel's `description`, if it has one.
pub fn heard(post: &Post, description: Option<&str>) -> Heard {
    let (who, c) = (post.from.as_deref().map_or("The user".to_string(), str::to_string), &post.channel);
    let how = format!("(To reply in #{c}, use message_channel with channel \"{c}\"; what you say otherwise stays in your session.)");
    let about = description.map_or(String::new(), |d| format!("\n\nAbout #{c}: {d}"));
    Heard { from: format!("#{c}"), text: format!("{who} posted in #{c}:\n\n{}\n\n{how}{about}", post.text) }
}

/// Session `name`'s place in channel `channel` changed as `change` says: {inputs?, per_day?} (joining
/// with inputs off and no limit, if it isn't in the channel), or null to leave. Only the hub calls this, as it writes the
/// settings files.
pub fn member(app: &App, name: &str, channel: &str, change: &Value) -> Result<(), String> {
    let there = app.channel_exists(channel);
    settings::change(&app.profile_path(name), |s: &mut SessionSettings| {
        if change.is_null() {
            s.channels.remove(channel);
            return Ok(());
        }
        if !there {
            return Err(format!("no channel {channel}"));
        }
        let m = s.channels.entry(channel.to_string()).or_default();
        if let Some(on) = change["inputs"].as_bool() { m.inputs = on }
        if let Some(n) = change["per_day"].as_f64() { m.per_day = n as i64 }
        Ok(())
    }).map(drop)
}

/// The session's tool for its channels: message_channel, while it's in any. It doesn't ask the user
/// first.
pub fn tools(app: &App, me: &str, settings: &SessionSettings) -> Toolbox {
    let ins: Vec<&str> = settings.channels.keys().map(String::as_str).collect();
    if ins.is_empty() {
        return Toolbox::default();
    }
    let limit = |m: &Membership| m.limit().map_or(String::new(), |n| format!(" (at most {n} posts a day)"));
    let list = settings.channels.iter().map(|(c, m)| format!("#{c}{}", limit(m))).join(", ");
    let description = format!("Post a message in a channel you're in ({list}). Every other session there that listens gets it at once, as a message headed by your name \
        and the channel, and the user reads the channel. Posts there come to you the same way, if the user has you listen; what you say otherwise stays in your session.");
    let properties = obj! {"channel" => obj! {"type" => "string", "enum" => ins, "description" => "The channel, without its #"}, "text" => obj! {"type" => "string", "description" => "What to post"}};
    let definition = obj! {"name" => "message_channel", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["channel", "text"]}};
    let (hub, me, profile) = (app.hub.clone(), me.to_string(), app.profile_path(me));
    let message = move |input: &Value, _: &dyn Fn() -> bool| {
        let named = input["channel"].as_str().map(|c| c.trim_start_matches('#')).ok_or("message_channel needs the channel")?;
        let text = input["text"].as_str().filter(|t| !t.trim().is_empty()).ok_or("message_channel needs text")?;
        let own: SessionSettings = settings::read(&profile)?;
        if !own.channels.contains_key(named) {
            return Err(format!("you're not in #{named}").into());
        }
        let (reply, answer) = channel();
        hub.send(Routed::Post(Post { channel: named.into(), from: Some(me.clone()), text: text.into() }, reply)).expect("the hub runs");
        let Posted { heard, left, .. } = answer.recv().expect("the hub answers")?;
        let heard = match heard.as_slice() {
            [] => format!("Posted in #{named}; no one else there listens now, but the user reads it."),
            [one] => format!("Posted in #{named}; {one} heard it."),
            [rest @ .., last] => format!("Posted in #{named}; {} and {last} heard it.", rest.join(", ")),
        };
        let left = left.map_or(String::new(), |(left, limit)| format!(" You have {left} of your {limit} posts a day left there."));
        Ok(format!("{heard}{left}").into())
    };
    Toolbox(vec![Tool { definition, asks: false, run: Box::new(message) }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_read_as_hours_and_minutes() {
        assert_eq!([0, 59, 60, 3599, 3600, 86_399].map(how_long), ["under a minute", "under a minute", "1 min", "59 min", "1 h 0 min", "23 h 59 min"]);
        let limit = |per_day| Membership { inputs: false, per_day }.limit();
        assert_eq!([limit(5), limit(0), limit(-3)], [Some(5), None, None]);
    }

    #[test]
    fn a_post_is_heard_from_its_channel() {
        let post = |from: Option<&str>| heard(&Post { channel: "lounge".into(), from: from.map(str::to_string), text: "hi".into() }, None);
        let how = "(To reply in #lounge, use message_channel with channel \"lounge\"; what you say otherwise stays in your session.)";
        assert_eq!((post(Some("onion")).from, post(Some("onion")).text), ("#lounge".to_string(), format!("onion posted in #lounge:\n\nhi\n\n{how}")));
        assert_eq!(post(None).text, format!("The user posted in #lounge:\n\nhi\n\n{how}"));
        let described = heard(&Post { channel: "lounge".into(), from: None, text: "hi".into() }, Some("Post only when it matters."));
        assert_eq!(described.text, format!("The user posted in #lounge:\n\nhi\n\n{how}\n\nAbout #lounge: Post only when it matters."));
    }
}
