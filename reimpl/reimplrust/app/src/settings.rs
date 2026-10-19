//! The settings: JSON files a user may edit, read again at every use, so an edit counts at once, and
//! only this user can read, since settings.json holds the keys. README.md describes their fields:
//! ROOT/settings.json is a Settings, each session's ROOT/NAME/session_settings.json a SessionSettings.

use connectome::{config::Config, json::Value, obj};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use smart_default::SmartDefault;
use std::{collections::{BTreeMap, BTreeSet}, io::Write, path::{Path, PathBuf}};

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub providers: BTreeMap<String, Provider>,
    pub defaults: SessionSettings, // what a new session starts with
    pub sandbox: Sandbox,
}

impl Settings {
    /// What settings.json starts as: a provider named default, Anthropic's API without a key yet, and
    /// new sessions on it with a window of 60k tokens, `overhead` of which every request carries (the
    /// system prompt and tool definitions), the library's settings sized for the rest.
    pub fn first(overhead: f64) -> Settings {
        let (defaults, base) = (SessionSettings::default(), "https://api.anthropic.com".into());
        let left = defaults.window - overhead; // the library's sizes below are its defaults, for windows of 120k and more
        let (recent_window_tokens, target_chunk_tokens) = ((left / 4.0).min(30000.0), (left / 16.0).min(3000.0));
        let config = Config { summary_participant: "assistant".into(), recent_window_tokens, target_chunk_tokens, compression_max_tokens: Some(8000.0), ..Config::default() };
        let providers = BTreeMap::from([("default".into(), Provider { base, key: String::new() })]);
        Settings { providers, defaults: SessionSettings { config, ..defaults }, sandbox: Sandbox::default() }
    }

    /// The providers as the page sees them: never a key, only whether there is one.
    pub fn providers(&self) -> Value {
        let shown = self.providers.iter().map(|(name, p)| obj! {"name" => name.as_str(), "base" => p.base.as_str(), "key" => !p.key.is_empty()});
        obj! {"providers" => shown.collect::<Vec<_>>()}
    }

    /// A provider added, or its base URL or key changed, by what `v` gives that isn't blank.
    pub fn provide(&mut self, v: &Value) -> Result<(), String> {
        let given = |k: &str| v[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        let p = self.providers.entry(given("name").ok_or("a provider's name")?).or_default();
        (p.base, p.key) = (given("base").unwrap_or(std::mem::take(&mut p.base)), given("key").unwrap_or(std::mem::take(&mut p.key)));
        if p.base.is_empty() { Err("a provider's base URL".into()) } else { Ok(()) }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Provider {
    pub base: String, // e.g. https://api.anthropic.com
    pub key: String,
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
    pub sandbox: bool,
    pub allow: BTreeSet<String>,          // the tools that run without asking
    #[default(60000.0)] pub window: f64,  // tokens a request may take, the reply's included
    #[default(4000.0)] pub reserve: f64,  // tokens for the reply
    pub system: Option<String>,
    pub config: Config,
}

impl SessionSettings {
    /// These settings as what `v` gives changes them: any of their fields, and any of config's. The
    /// provider must be one of `all`'s, a workspace given a directory that exists (absolute, a leading
    /// ~/ the home directory, no trailing /), and merge_threshold as it was once memories have `merged`.
    pub fn tuned(&self, v: &Value, merged: bool, all: &Settings) -> Result<SessionSettings, String> {
        let mut new = serde_json::to_value(self).expect("settings are JSON");
        for (k, x) in v.as_object().into_iter().flatten() {
            match (k.as_str(), x) {
                ("config", Value::Object(fields)) => new["config"].as_object_mut().expect("config").extend(fields.clone()),
                _ => new[k] = x.clone(),
            }
        }
        let mut new: SessionSettings = serde_json::from_value(new).map_err(|e| e.to_string())?;
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
        if merged && new.config.merge_threshold != self.config.merge_threshold {
            return Err("merge_threshold stays as it is once memories have merged".into());
        }
        Ok(new)
    }
}

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
