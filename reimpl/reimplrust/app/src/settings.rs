//! The settings: JSON files a user may edit, read again at every use, so an edit counts at once, and
//! only this user can read, since settings.json holds the keys. README.md describes their fields:
//! ROOT/settings.json is a Settings, each session's ROOT/NAME/session_settings.json a SessionSettings.

use connectome::config::Config;
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
