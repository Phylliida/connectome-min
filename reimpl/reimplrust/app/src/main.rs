//! connectome-web: chat with a model whose context the connectome library manages, in a web page that
//! shows each session's memory pyramid and what its context shows, and forks a session at any message.
//! One binary that needs nothing installed: the page is built in, and so is TLS (rustls, ring and
//! Mozilla's roots). README.md describes its flags, its settings files (settings.rs) and its tools
//! (tools/).
//!
//! Whatever changes has one thread that owns it, and threads tell each other things through channels,
//! sharing nothing that changes:
//! - each request's thread (api.rs) reads files and tells the hub what it wants of a session;
//! - the hub (hub.rs) owns what changes quickly: the settings files, the channels' logs, the changes the
//!   page makes to boards and what it last saw of them, which sessions have a thread, and the pages
//!   following each session or channel and what they hear;
//! - each session's thread (work.rs) does its jobs one at a time, the only one to write its log.
//!
//! A session's files are its only record, what its requests used among them (costs.rs): the hub keeps
//! beside them only what they don't hold yet.

mod api;
mod boards;
mod channels;
mod claude;
mod companion;
mod costs;
mod gateway;
mod hub;
mod llm;
mod settings;
mod tools;
mod work;

use crate::{settings::{SessionSettings, Settings}, tools::mcp};
use connectome::{json, store::Session};
use std::{net::TcpListener, path::PathBuf, sync::mpsc::{channel, Sender}, time::Duration};

/// What every thread reads, set once at the start and never changed.
pub struct App {
    root: PathBuf,
    addr: String,
    agent: ureq::Agent,
    upstream: String,      // where the gateway forwards requests (gateway.rs)
    mcp: Vec<mcp::Server>, // every session's MCP servers
    mcp_config: Option<String>, // their config's JSON, which Claude Code sessions are given (claude.rs)
    bash: bool,            // whether a session with a workspace has bash
    limit: Duration,       // a tool call's time
    hub: Sender<hub::Routed>,
}

impl App {
    fn session(&self, name: &str) -> Session { Session::open(self.root.join(name)) }

    fn settings_path(&self) -> PathBuf { self.root.join("settings.json") }

    fn settings(&self) -> Result<Settings, String> { settings::read(&self.settings_path()) }

    fn profile_path(&self, name: &str) -> PathBuf { self.root.join(name).join("session_settings.json") }

    fn profile(&self, name: &str) -> Result<SessionSettings, String> { settings::read(&self.profile_path(name)) }

    /// Every session's settings, by name, of those that have them.
    fn profiles(&self) -> Vec<(String, SessionSettings)> {
        let names = std::fs::read_dir(&self.root).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok()).filter(|n| !n.starts_with('.'));
        let mut names: Vec<String> = names.collect();
        names.sort();
        names.into_iter().filter_map(|name| self.profile(&name).ok().map(|s| (name, s))).collect()
    }
}

fn main() {
    // glibc raises its mmap threshold after a large block is freed, and keeps later large blocks in
    // per-thread arenas: a message of a megabyte read on many connection threads then stays resident.
    // A fixed threshold returns each large block to the system once it's freed.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe { libc::mallopt(libc::M_MMAP_THRESHOLD, 128 * 1024) };
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.into());
    let flag = |name: &str| args.iter().any(|a| a == name);
    let ca = flag("--ca").then(|| std::fs::read(arg("--ca", "")).expect("--ca"));
    let root = PathBuf::from(arg("--root", "sessions"));
    std::fs::create_dir_all(&root).expect("--root");
    let limit = Duration::from_secs_f64(arg("--tool-timeout", "600").parse().expect("--tool-timeout"));
    let mcp_config = std::fs::read_to_string(arg("--mcp", &root.join("mcp.json").to_string_lossy())).ok();
    let mcp = mcp_config.as_ref().map_or_else(Vec::new, |config| mcp::servers(&json::parse(config.as_bytes()).expect("--mcp: JSON"), limit));
    let (told, inbox) = channel();
    let (addr, agent) = (arg("--addr", "127.0.0.1:8080"), llm::agent(ca.as_deref()).expect("--ca"));
    let upstream = arg("--upstream", "https://api.anthropic.com");
    if !ureq::http::Uri::try_from(&upstream).is_ok_and(|u| u.scheme().is_some() && u.host().is_some()) {
        panic!("--upstream {upstream}: not a URL like https://api.anthropic.com");
    }
    let app: &'static App = Box::leak(Box::new(App { root, addr, agent, upstream, mcp, mcp_config, bash: !flag("--no-bash"), limit, hub: told }));
    if !app.settings_path().exists() {
        settings::write(&app.settings_path(), &Settings::first()).expect("settings.json");
    }
    std::thread::spawn(move || hub::hub(app, inbox));
    let listener = TcpListener::bind(&app.addr).expect("--addr");
    eprintln!("connectome-web on http://{}, sessions in {}, MCP tools: {:?}", app.addr, app.root.display(), mcp::toolbox(&app.mcp).names());
    if !flag("--no-open") { let _ = open::that_detached(format!("http://{}", app.addr)); }
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || api::handle(app, stream).unwrap_or_else(|e| eprintln!("{e}")));
    }
}
