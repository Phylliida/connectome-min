//! The settings: JSON files a user may edit, read again at every use, so an edit counts at once, and
//! only this user can read, since settings.json holds the keys. README.md describes their fields:
//! ROOT/settings.json is a Settings, each session's ROOT/NAME/session_settings.json a SessionSettings.

use connectome::{config::Config, json::{sparse, Value}, obj};
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use smart_default::SmartDefault;
use std::{collections::{BTreeMap, BTreeSet}, io::Write, path::{Path, PathBuf}};

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub providers: BTreeMap<String, Provider>,
    pub defaults: SessionSettings, // what a new session starts with
    pub sandbox: Sandbox,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub search: Option<String>, // a SearXNG server, which web_search asks
}

impl Settings {
    /// What settings.json starts as: a provider named default, Anthropic's API without a key yet, and
    /// new sessions on it (on the one model it lists) with a window of 180k tokens, 30k of it the
    /// reply's, a prompt that folds past 90k down to 65k, a raw tail of 20k, chunks of 3k (the
    /// library's), merges of 3 memories at least, and memories of up to 20k tokens.
    pub fn first() -> Settings {
        let (defaults, base) = (SessionSettings::default(), "https://api.anthropic.com".into());
        let config = Config { summary_participant: "assistant".into(), recent_window_tokens: 20000.0, target_chunk_tokens: 3000.0, min_merge_memories: 3, compression_max_tokens: Some(20000.0), ..Config::default() };
        let providers = BTreeMap::from([("default".into(), Provider { base, key: String::new(), models: vec![defaults.model.clone()], command: None })]);
        let defaults = SessionSettings { fold_above: 90000.0, fold_to: 65000.0, config, ..defaults };
        Settings { providers, defaults, sandbox: Sandbox::default(), search: None }
    }

    /// The providers as the page sees them: never a key, only whether there is one.
    pub fn providers(&self) -> Value {
        let shown = self.providers.iter().map(|(name, p)| sparse(obj! {"name" => name.as_str(), "base" => p.base.as_str(), "key" => !p.key.is_empty(), "models" => p.models.clone(), "command" => p.command.clone()}));
        obj! {"providers" => shown.collect::<Vec<_>>()}
    }

    /// A provider added, or its base URL or key changed, by what `v` gives that isn't blank, and its
    /// models, if `v` lists them, those it lists (each once, in their order); or, if `v` says delete,
    /// deleted, unless new sessions start on it (sessions on it then need another).
    pub fn provide(&mut self, v: &Value) -> Result<(), String> {
        let given = |k: &str| v[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        let name = given("name").ok_or("a provider's name")?;
        if v["delete"] == true {
            if name == self.defaults.provider {
                return Err(format!("new sessions start on {name}: name another in settings.json's defaults first"));
            }
            return self.providers.remove(&name).map(drop).ok_or(format!("no provider {name}"));
        }
        let p = self.providers.entry(name).or_default();
        (p.base, p.key) = (given("base").unwrap_or(std::mem::take(&mut p.base)), given("key").unwrap_or(std::mem::take(&mut p.key)));
        if let Some(models) = v.get("models") {
            let listed = models.as_array().ok_or("a provider's models, a list")?.iter().filter_map(Value::as_str).map(str::trim);
            p.models = listed.filter(|m| !m.is_empty()).unique().map(String::from).collect();
        }
        if p.base.is_empty() && p.command.is_none() { Err("a provider's base URL".into()) } else { Ok(()) }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Provider {
    pub base: String, // e.g. https://api.anthropic.com
    pub key: String,
    #[serde(default)] pub models: Vec<String>, // what the page offers a session on it
    // a Claude Code provider: how to run Claude Code, say ["claude"]; its sessions take their turns in
    // it (claude.rs), and `base` and `key` go unused
    #[serde(default, skip_serializing_if = "Option::is_none")] pub command: Option<Vec<String>>,
}

/// What a sandboxed command sees besides its workspace: the paths in `ro` that exist, read-only, and
/// in `rw`, read-write (a leading ~/ is the home directory); and the network, if `network`.
#[derive(Clone, Serialize, Deserialize, SmartDefault)]
pub struct Sandbox {
    #[default(["/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc", "/nix", "/run", "/sys"].map(String::from).into())] pub ro: Vec<String>,
    #[default(vec!["/tmp".into()])] pub rw: Vec<String>,
    #[default(true)] pub network: bool,
}

/// A session's settings; new sessions start from settings.json's defaults, which start as these.
#[derive(Clone, Serialize, Deserialize, SmartDefault)]
pub struct SessionSettings {
    #[default("default")] pub provider: String,
    #[default("claude-sonnet-5-5")] pub model: String,
    pub workspace: Option<PathBuf>,
    #[default(cfg!(target_os = "linux"))] pub sandbox: bool, // where bubblewrap runs
    // whether the model can ask the user questions: ask_user (tools/ask.rs), or Claude Code's AskUserQuestion
    #[default(true)] #[serde(default = "on")] pub questions: bool,
    pub allow: BTreeSet<String>,          // the tools that run without asking
    #[default(180000.0)] pub window: f64, // tokens a request may take, the reply's included
    #[default(30000.0)] pub reserve: f64, // tokens for the reply (its reasoning too)
    // a prompt (the system prompt and tools with the context) past fold_above folds, down to fold_to: at
    // most what the window leaves the reply, which is fold_above's default, and fold_to's config.slack (10%) under that
    #[default(150000.0)] pub fold_above: f64,
    #[default(135000.0)] pub fold_to: f64,
    pub system: Option<String>,
    pub config: Config,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub companion: Option<String>,   // the session that is this one's companion (companion.rs)
    #[serde(default, skip_serializing_if = "Option::is_none")] pub accompanies: Option<String>, // the session this one is the companion of
    #[serde(default, skip_serializing_if = "std::ops::Not::not")] pub auto: bool,               // each turn's end goes to the other
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")] pub channels: BTreeMap<String, Membership>, // the channels it's in, by name (channels.rs)
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")] pub boards: BTreeSet<String>,  // the boards it's on (boards.rs), which take a workspace, and not being a companion
    #[serde(default, skip_serializing_if = "Option::is_none")] pub claude_session: Option<String>, // on a Claude Code provider, the session Claude Code keeps, once started (claude.rs)
}

/// A session's place in a channel (channels.rs): whether it hears the channel's posts (`inputs`), and
/// the most posts it may make there in any 24 hours (`per_day`; 0 or less, no limit).
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct Membership {
    #[serde(default)] pub inputs: bool,
    #[serde(default, skip_serializing_if = "no_limit")] pub per_day: i64,
}

/// Whether a limit of posts a day is none (0 or less), which settings leave out.
fn no_limit(per_day: &i64) -> bool { *per_day <= 0 }

impl Membership {
    /// Its limit, if it has one.
    pub fn limit(&self) -> Option<u64> { u64::try_from(self.per_day).ok().filter(|&n| n > 0) }
}

impl SessionSettings {
    /// The session this one talks with: its companion, or the one it's the companion of.
    pub fn partner(&self) -> Option<&str> { self.companion.as_deref().or(self.accompanies.as_deref()) }

    /// These settings as what `v` gives changes them: any of their fields, and any of config's. The
    /// provider must be one of `all`'s, a workspace given a directory that exists (absolute, a leading
    /// ~/ the home directory, no trailing /), fold_to at most fold_above and that at most what the window
    /// leaves the reply, chunks of more than 0 tokens, merges of 2 memories or more, and once memories have `merged`, merge_threshold as it was and merge_by_count
    /// not turned on (merging by count after merging by tokens would misread them).
    pub fn tuned(&self, v: &Value, merged: bool, all: &Settings) -> Result<SessionSettings, String> {
        let mut new = serde_json::to_value(self).expect("settings are JSON");
        for (k, x) in v.as_object().into_iter().flatten() {
            match (k.as_str(), x) {
                ("config", Value::Object(fields)) => new["config"].as_object_mut().expect("config").extend(fields.clone()),
                _ => new[k] = x.clone(),
            }
        }
        let mut new: SessionSettings = serde_json::from_value(new).map_err(|e| format!("the settings {}: {e}", connectome::json::to_string(v)))?;
        if let Some(path) = v["workspace"].as_str().map(str::trim) {
            let dir: PathBuf = std::path::absolute(expand(path)).map_err(|e| format!("{path}: {e}"))?.components().collect();
            if !dir.is_dir() {
                return Err(format!("no directory {path}"));
            }
            new.workspace = Some(dir);
        }
        if !all.providers.contains_key(&new.provider) {
            return Err(format!("no provider {}", new.provider));
        }
        let room = new.window - new.reserve;
        if !(0.0..=room).contains(&new.fold_above) {
            return Err(format!("fold_above ({}) must be at most what the window leaves the reply ({room})", new.fold_above));
        }
        if !(0.0..=new.fold_above).contains(&new.fold_to) {
            return Err(format!("fold_to ({}) must be at most fold_above ({})", new.fold_to, new.fold_above));
        }
        if new.config.target_chunk_tokens <= 0.0 {
            return Err(format!("a chunk ({} tokens) must hold more than 0", new.config.target_chunk_tokens));
        }
        if new.config.min_merge_memories < 2 {
            return Err(format!("a merge ({} memories at least) must take 2 or more", new.config.min_merge_memories));
        }
        if merged && new.config.merge_threshold != self.config.merge_threshold {
            return Err("merge_threshold stays as it is once memories have merged".into());
        }
        if merged && new.config.merge_by_count && !self.config.merge_by_count {
            return Err("merge_by_count can't go on once memories have merged by tokens".into());
        }
        Ok(new)
    }
}

/// A setting that's on unless a file says otherwise.
fn on() -> bool { true }

/// A path, a leading ~/ the home directory.
pub fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// A settings file as `f` changes it, written if `f` succeeds. The hub changes every settings file
/// (once whoever made a session wrote its first).
pub fn change<T: Serialize + DeserializeOwned>(path: &Path, f: impl FnOnce(&mut T) -> Result<(), String>) -> Result<T, String> {
    let mut settings = read(path)?;
    f(&mut settings)?;
    write(path, &settings).map(|()| settings)
}

/// Writes a settings file whole, through a new file that then takes the old one's place, so no one
/// reads half of it.
pub fn write(path: &Path, settings: &impl Serialize) -> Result<(), String> {
    let new = path.with_extension("json.new");
    let _ = std::fs::remove_file(&new); // left by a write that stopped halfway
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let text = serde_json::to_vec_pretty(settings).expect("settings are JSON");
    let written = options.open(&new).and_then(|mut file| file.write_all(&text)).and_then(|()| std::fs::rename(&new, path));
    written.map_err(|e| format!("saving {}: {e}", path.display()))
}
