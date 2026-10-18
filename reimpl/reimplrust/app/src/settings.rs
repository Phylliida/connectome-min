//! The settings: JSON files a user may edit, read again at every use, so an edit counts at once.
//!
//! ROOT/settings.json holds what sessions share: the providers by name, each a base URL and its key
//! (so only this user can read the file); what a new session starts with (`defaults`); and what a
//! sandboxed command sees (`sandbox`). Each session's ROOT/NAME/session_settings.json holds what it
//! runs with: its provider (by name) and model, its workspace (where bash runs; a session without one
//! has no bash), whether bash runs in the sandbox, the tools it always allows, its window, reply
//! reserve and system prompt, and the library's settings (`config`). A new session's file starts as a
//! copy of `defaults` with the workspace it's given, a fork's as a copy of its parent's.

use connectome::config::Config;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub providers: BTreeMap<String, Provider>,
    pub defaults: SessionSettings,
    pub sandbox: Sandbox,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Provider {
    pub base: String, // e.g. https://api.anthropic.com
    pub key: String,
}

/// What a sandboxed command sees besides its workspace: the paths in `ro` that exist, read-only, and
/// in `rw`, read-write (a leading ~/ is the home directory); and the network, if `network`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Sandbox {
    pub ro: Vec<String>,
    pub rw: Vec<String>,
    pub network: bool,
}

impl Default for Sandbox {
    fn default() -> Sandbox {
        let ro = ["/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc", "/nix", "/run", "/sys"];
        Sandbox { ro: ro.map(String::from).into(), rw: vec!["/tmp".into()], network: true }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    pub provider: String,
    pub model: String,
    pub workspace: Option<PathBuf>,
    pub sandbox: bool,
    pub allow: BTreeSet<String>, // the tools that run without asking
    pub window: f64,             // tokens a request may take, the reply's included
    pub reserve: f64,            // tokens for the reply
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

/// A settings file, or None if there's none.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, String> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Writes a settings file whole, through a new file that then takes the old one's place, so no one
/// reads half of it; a `private` one only this user can read.
pub fn write(path: &Path, settings: &impl Serialize, private: bool) -> Result<(), String> {
    let new = path.with_extension("json.new");
    let _ = std::fs::remove_file(&new); // left by a write that stopped halfway, maybe with another mode
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, if private { 0o600 } else { 0o644 });
    let text = serde_json::to_vec_pretty(settings).expect("settings are JSON");
    let written = options.open(&new).and_then(|mut file| file.write_all(&text)).and_then(|()| std::fs::rename(&new, path));
    written.map_err(|e| format!("saving {}: {e}", path.display()))
}
