//! Channels: named logs of posts that sessions take part in. A post is the user's, or a session's,
//! made with its message_channel tool in a channel it's in; it goes at once to each session in the
//! channel that has inputs on but the one that posted it, as a message headed by the channel and who
//! posted it, the way a session hears its companion (companion.rs): between the steps of a turn
//! running, or else in a turn of its own. A session's settings say which channels it's in and whether
//! it has inputs on in each (`channels`), which only the user sets. Each post a session hears ends with
//! a reminder of how to answer in the channel, since an answer there must be posted, then the
//! channel's description, if the user gave it one (say, how sparingly to post). A channel is
//! ROOT/.channels/NAME: posts.jsonl, each post {id, from, text}, `from` the session that posted it
//! (null: the user), and channel.json, its settings (About); the hub alone writes both, as it routes
//! each post (hub's Routed::Post) and each change of settings. No session can be named .channels.

use crate::{companion::Heard, hub::Routed, settings::{self, SessionSettings}, tools::{Tool, Toolbox}, App};
use connectome::{json::Value, obj, store::Session};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::mpsc::channel};

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
    let profiles = profiles(app);
    Value::Array(names.map(|name| {
        let members: Vec<Value> = profiles.iter().filter_map(|(member, s)| s.channels.get(&name).map(|m| obj! {"name" => member.as_str(), "inputs" => m.inputs})).collect();
        obj! {"name" => name.as_str(), "description" => app.about(&name).description, "posts" => app.channel(&name).count("posts"), "members" => members}
    }).collect())
}

/// Every session's settings, by name, those that have them.
fn profiles(app: &App) -> Vec<(String, SessionSettings)> {
    let names = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok()).filter(|n| !n.starts_with('.'));
    names.sorted().filter_map(|name| app.profile(&name).ok().map(|s| (name, s))).collect()
}

/// The sessions that hear channel `name`'s posts: those in it with inputs on.
pub fn hearing(app: &App, name: &str) -> Vec<String> {
    profiles(app).into_iter().filter(|(_, s)| s.channels.get(name).is_some_and(|m| m.inputs)).map(|(member, _)| member).collect()
}

/// A post joins its channel's log, as the record it's logged as. Only the hub calls this.
pub fn append(app: &App, post: &Post) -> Result<Value, String> {
    if !app.channel_exists(&post.channel) {
        return Err(format!("no channel {}", post.channel));
    }
    Ok(app.channel(&post.channel).append("posts", obj! {"from" => post.from.as_deref(), "text" => post.text.as_str()}))
}

/// A post as a session that hears its channel gets it: from the channel, headed by who posted it, and
/// ending with how to answer there, then the channel's `description`, if it has one.
pub fn heard(post: &Post, description: Option<&str>) -> Heard {
    let (who, c) = (post.from.as_deref().map_or("The user".to_string(), str::to_string), &post.channel);
    let how = format!("(To reply in #{c}, use message_channel with channel \"{c}\"; what you say otherwise stays in your session.)");
    let about = description.map_or(String::new(), |d| format!("\n\nAbout #{c}: {d}"));
    Heard { from: format!("#{c}"), text: format!("{who} posted in #{c}:\n\n{}\n\n{how}{about}", post.text) }
}

/// Session `name`'s place in channel `channel` changed as `change` says: {inputs?} (joining with
/// inputs off, if it isn't in the channel), or null to leave. Only the hub calls this, as it writes the
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
    let list = ins.iter().map(|c| format!("#{c}")).join(", ");
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
        let (_, heard) = answer.recv().expect("the hub answers")?;
        Ok(match heard.as_slice() {
            [] => format!("Posted in #{named}; no one else there listens now, but the user reads it."),
            [one] => format!("Posted in #{named}; {one} heard it."),
            [rest @ .., last] => format!("Posted in #{named}; {} and {last} heard it.", rest.join(", ")),
        }.into())
    };
    Toolbox(vec![Tool { definition, asks: false, run: Box::new(message) }])
}

#[cfg(test)]
mod tests {
    use super::*;

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
