//! connectome-web: chat with a model whose context the connectome library manages, in a web page that
//! shows each session's memory pyramid and what its context shows, and forks a session at any message.
//! One binary that needs nothing installed: the page is built in, and so is TLS (rustls, ring and
//! Mozilla's roots).
//!
//!     connectome-web [--root DIR] [--addr 127.0.0.1:8080] [--base URL] [--model NAME] [--window TOKENS]
//!                    [--reserve TOKENS] [--system TEXT] [--ca PEM] [--mcp FILE] [--tool-timeout SECONDS]
//!                    [--no-bash] [--no-open]
//!
//! The settings are files in DIR (settings.rs): DIR/settings.json holds the providers, their keys and
//! what a new session starts with, which --base, --model, --window, --reserve, --system and API_KEY,
//! ANTHROPIC_API_KEY or DEEPSEEK_API_KEY give when there's no such file; each session holds its own.
//! --ca trusts the root certificates in a PEM file instead of Mozilla's (a company proxy's, a test
//! server's). The model can call bash in its session's workspace (bash.rs; unless --no-bash), in a
//! sandbox if the session says so, and the tools of the MCP servers that --mcp names (by default
//! DIR/mcp.json, if there is one; mcp.rs), each call once the user allows it on the page, or always
//! allows its tool for the session, and within --tool-timeout (600 s by default). The system prompt and
//! the tool definitions come out of the window the context compiles to, since every request carries
//! them. A chat message joins the log, then the turn takes steps until a reply calls no tools: the
//! context compiles and streams from disk into the request, the reply joins the log, and the tools it
//! calls run, their results joining the log. Each step reads the settings again. Memories form between
//! steps and, once the page has the turn, after it, the session locked throughout. The page opens in
//! the browser unless --no-open.

mod bash;
mod llm;
mod mcp;
mod settings;
mod tools;

use connectome::blocks::density_tokens;
use connectome::config::Config;
use connectome::drive::{append, compile, current, drive};
use connectome::emit::Reader;
use connectome::estimate::Calibration;
use connectome::json::{self, sparse, Json, Value};
use connectome::memory::{Host, Request};
use connectome::obj;
use connectome::store::Session;
use llm::Llm;
use serde::de::DeserializeOwned;
use serde::Serialize;
use settings::{Provider, Sandbox, SessionSettings, Settings};
use tools::Toolbox;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PAGE: &str = include_str!("ui.html");

struct App {
    root: PathBuf,
    addr: String,
    agent: ureq::Agent,
    mcp: Toolbox,                                   // every session's MCP tools
    bash: bool,                                     // whether a session with a workspace has bash
    limit: Duration,                                // a tool call's time
    writing: Mutex<()>,                             // settings files change one at a time
    asking: Mutex<HashMap<String, Sender<String>>>, // the calls waiting for the user's answer, by id
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    usage: Mutex<HashMap<String, f64>>, // each session's last prompt size, for calibration
    forming: Mutex<HashSet<String>>,    // sessions forming memories
    live: Mutex<HashMap<String, Arc<Mutex<Live>>>>, // the turns running, by session
}

/// A turn as it runs: how many messages the log had once its message joined, its events so far,
/// which a page that comes in late plays first, and the pages following it.
struct Live {
    from: u64,
    events: Vec<String>,
    pages: Vec<TcpStream>,
}

/// The head of a turn's events, a JSON line each.
const EVENTS: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n";

/// How long a page following a turn may keep it waiting before it's left behind.
const PAGE_WAIT: Duration = Duration::from_secs(30);

/// What a session's step runs with: its settings, the model client and tools they give, and the
/// tokens the system prompt and tool definitions take, which every request carries.
struct Setup {
    s: SessionSettings,
    llm: Llm,
    tools: Toolbox,
    overhead: f64,
}

/// The summarizer's host. It declares the agent's tools, which the library sends with tool choice
/// "none" so a memory's request can carry tool blocks, and no system prompt.
struct Summarizer<'a>(&'a Llm, Option<Value>);

impl Host for Summarizer<'_> {
    fn tools(&self) -> Option<&Value> { self.1.as_ref() }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.0.model } else { &request.model };
        let none = request.tools.as_ref().map(|_| obj! {"type" => "none"});
        let head = sparse(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens, "tools" => request.tools.clone(), "tool_choice" => none});
        self.0.messages(head, &request.messages, reader, |_, _| {})
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
    let key = ["API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"].iter().find_map(|k| std::env::var(k).ok()).unwrap_or_default();
    let extra = args.iter().any(|a| a == "--ca").then(|| std::fs::read(arg("--ca", "")).expect("--ca"));
    let agent = llm::agent(extra.as_deref()).expect("--ca");
    let root = PathBuf::from(arg("--root", "sessions"));
    let addr = arg("--addr", "127.0.0.1:8080");
    std::fs::create_dir_all(&root).expect("--root");
    let limit = Duration::from_secs_f64(arg("--tool-timeout", "600").parse().expect("--tool-timeout"));
    let bash = !args.iter().any(|a| a == "--no-bash");
    let mcp = std::fs::read(arg("--mcp", &root.join("mcp.json").to_string_lossy())).ok();
    let mcp = mcp.map_or_else(Toolbox::default, |config| mcp::toolbox(&json::parse(&config).expect("--mcp: JSON"), limit));
    let path = root.join("settings.json");
    if settings::read::<Settings>(&path).expect("settings.json").is_none() {
        let (window, system): (f64, _) = (arg("--window", "60000").parse().expect("--window"), args.iter().any(|a| a == "--system").then(|| arg("--system", "")));
        let tools = if bash { bash::tool(PathBuf::new(), limit, None) } else { Toolbox::default() } + mcp.clone();
        let left = window - overhead(system.as_deref(), &tools); // what the context compiles to
        let config = Config {
            summary_participant: "assistant".into(),
            recent_window_tokens: (left / 4.0).min(30000.0), // the library's defaults, for windows of 120k and more
            target_chunk_tokens: (left / 16.0).min(3000.0),
            compression_max_tokens: Some(8000.0),
            ..Config::default()
        };
        let (model, reserve) = (arg("--model", "claude-sonnet-5-5"), arg("--reserve", "4000").parse().expect("--reserve"));
        let defaults = SessionSettings { provider: "default".into(), model, workspace: None, sandbox: false, allow: Default::default(), window, reserve, system, config };
        let providers = BTreeMap::from([("default".to_string(), Provider { base: arg("--base", "https://api.anthropic.com"), key })]);
        settings::write(&path, &Settings { providers, defaults, sandbox: Sandbox::default() }, true).expect("settings.json");
    }
    let app = Arc::new(App {
        root: root.clone(),
        addr: addr.clone(),
        agent,
        mcp,
        bash,
        limit,
        writing: Mutex::default(),
        asking: Mutex::default(),
        locks: Mutex::default(),
        usage: Mutex::default(),
        forming: Mutex::default(),
        live: Mutex::default(),
    });
    let listener = TcpListener::bind(&addr).expect("--addr");
    eprintln!("connectome-web on http://{addr}, sessions in {}, MCP tools: {:?}", root.display(), app.mcp.names());
    if !args.iter().any(|a| a == "--no-open") {
        let _ = open::that_detached(format!("http://{addr}"));
    }
    for stream in listener.incoming().flatten() {
        let app = app.clone();
        std::thread::spawn(move || handle(&app, stream).unwrap_or_else(|e| eprintln!("{e}")));
    }
}

/// One HTTP/1.1 request, then the connection closes. Only this server's own page gets an answer: a
/// request must name this server as its Host (another site's page reaching it through a DNS name it
/// controls doesn't), and a POST must carry an x-connectome header (another site's page can't send one
/// without a CORS preflight, which this server never approves), since a POST can change where the key
/// goes.
fn handle(app: &App, mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut lines = (&mut reader).lines().map_while(Result::ok);
    let line = lines.next().unwrap_or_default();
    let mut words = line.split_whitespace();
    let (method, target) = (words.next().unwrap_or("").to_string(), words.next().unwrap_or("/").to_string());
    let field = |h: String| h.split_once(':').map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()));
    let headers: Vec<(String, String)> = lines.take_while(|h| !h.is_empty()).filter_map(field).collect();
    let header = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    let length = header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0usize).min(1 << 20);
    let port = app.addr.rsplit(':').next().unwrap_or("");
    if ![app.addr.as_str(), &format!("localhost:{port}")].contains(&header("host").unwrap_or("")) || method == "POST" && header("x-connectome").is_none() {
        return send(&mut stream, "403 Forbidden", "text/plain", b"this server answers only its own page");
    }
    let body = {
        let mut raw = vec![0; length];
        reader.read_exact(&mut raw)?;
        json::parse(&raw).unwrap_or(Value::Null)
    };
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let param = |name: &str| query.split('&').find_map(|kv| kv.strip_prefix(name)?.strip_prefix('=')?.parse::<u64>().ok());
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let result = match (method.as_str(), parts.as_slice()) {
        ("GET", []) => return send(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        ("GET", ["api", "sessions"]) => Ok(sessions(app)),
        ("GET", ["api", "settings"]) => global(app).map(|all| providers(&all)),
        ("POST", ["api", "settings"]) => change(app, &app.root.join("settings.json"), true, |all| provide(all, &body)).map(|all| providers(&all)),
        ("POST", ["api", "approve"]) => match app.asking.lock().unwrap().remove(body["id"].as_str().unwrap_or("")) {
            Some(answer) => Ok(obj! {"answered" => answer.send(body["answer"].as_str().unwrap_or("deny").to_string()).is_ok()}),
            None => Err("no call waits for that answer".into()),
        },
        ("POST", ["api", "sessions"]) => create(app, &body),
        (_, ["api", "s", name, rest @ ..]) if named(name) && app.root.join(name).is_dir() => session(app, &method, name, rest, &body, param, &mut stream),
        (_, ["api", "s", name, ..]) => Err(format!("no session {name}")),
        _ => Err("not found".into()),
    };
    match result {
        Ok(Value::Null) => Ok(()), // already answered
        Ok(v) => send(&mut stream, "200 OK", "application/json", json::to_string(&v).as_bytes()),
        Err(e) => send(&mut stream, "400 Bad Request", "application/json", json::to_string(&obj! {"error" => e}).as_bytes()),
    }
}

/// A session's name: letters, digits, - and _.
fn named(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn send(stream: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\ncontent-type: {kind}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len())?;
    stream.write_all(body)
}

fn session(app: &App, method: &str, name: &str, rest: &[&str], body: &Value, param: impl Fn(&str) -> Option<u64>, stream: &mut TcpStream)
    -> Result<Value, String> {
    let s = Session::open(app.root.join(name));
    match (method, rest) {
        ("GET", []) => Ok(state(app, name, &s)),
        ("GET", ["messages"]) => Ok(listing(&s, "messages", 200, param).map(|m| shown(&m)).collect()),
        ("GET", ["layer", n]) => Ok(listing(&s, &format!("layer{n}"), u64::MAX, param).collect()),
        ("POST", ["fork"]) => {
            let (at, fork) = (body["at"].as_f64().ok_or("fork at which message?")? as u64, body["name"].as_str().unwrap_or(""));
            let fork = Some(fork).filter(|f| named(f)).ok_or("a fork's name")?;
            if app.root.join(fork).exists() || at >= s.count("messages") {
                return Err(format!("can't fork {name} at {at} as {fork}"));
            }
            let lock = lock(app, name);
            let _held = lock.lock().unwrap();
            let child = Session::fork(&s, at, app.root.join(fork));
            profile(app, name)?;
            std::fs::copy(profile_path(app, name), profile_path(app, fork)).map_err(|e| format!("copying the settings: {e}"))?;
            Ok(state(app, fork, &child))
        }
        ("POST", ["settings"]) => tune(app, name, &s, body),
        ("POST", ["say"]) => say(app, name, &s, Some(body["text"].as_str().unwrap_or("")), stream),
        ("POST", ["continue"]) => say(app, name, &s, None, stream),
        ("GET", ["live"]) => follow(app, name, stream),
        _ => Err("not found".into()),
    }
}

/// Records `from` to `to` of a file (by default, its newest `newest`), as they're read.
fn listing<'s>(s: &'s Session, file: &str, newest: u64, param: impl Fn(&str) -> Option<u64>) -> Box<dyn Iterator<Item = Value> + 's> {
    let n = s.count(file);
    let (from, to) = (param("from").unwrap_or(n.saturating_sub(newest)), param("to").unwrap_or(n).min(n));
    if from < to { s.span(file, from, to - 1) } else { Box::new(std::iter::empty()) }
}

fn lock(app: &App, name: &str) -> Arc<Mutex<()>> {
    app.locks.lock().unwrap().entry(name.to_string()).or_default().clone()
}

/// Steps a turn may take, a model request each, before it stops.
const MAX_STEPS: usize = 64;

/// A chat turn: the message joins the log, then steps run until a reply calls no tools. Without a
/// message, the turn continues the log as it is: after an interruption the context ends with the
/// user's message or the tool results, and after a reply cut short with that reply, which the
/// Messages API continues (a prefill); a session without messages has nothing to continue. Every page
/// following the turn (follow) gets one JSON line per event as it happens: the pieces of a reply's
/// reasoning and text ({"thinking"}, {"text"}), each tool call ({"tool"}: the tool_use block), the
/// question whether it may run and that it was answered ({"approve"}, {"answered"}: its id), and its
/// result ({"result"}: {id, text, error}); then {"ok": true} once the turn has joined the log, or
/// {"error"}. Memories form between steps and after the turn, the session locked throughout.
fn say(app: &App, name: &str, s: &Session, text: Option<&str>, stream: &mut TcpStream) -> Result<Value, String> {
    if text.map_or(s.count("messages") == 0, |text| text.trim().is_empty()) {
        return Err("say something".into());
    }
    let lock = lock(app, name);
    let _held = lock.lock().unwrap();
    let cfg = profile(app, name)?.config;
    // a page that has gone away still gets its turn logged
    let _ = stream.write_all(EVENTS);
    if let Some(text) = text {
        append(s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &cfg);
    }
    let first = stream.try_clone().map_err(|e| e.to_string())?;
    let _ = first.set_write_timeout(Some(PAGE_WAIT));
    let live = Arc::new(Mutex::new(Live { from: s.count("messages"), events: vec![], pages: vec![first] }));
    app.live.lock().unwrap().insert(name.to_string(), live.clone());
    let mut line = |event: Value| {
        let (text, mut live) = (format!("{}\n", json::to_string(&event)), live.lock().unwrap());
        live.pages.retain_mut(|page| page.write_all(text.as_bytes()).is_ok());
        live.events.push(text);
    };
    let stopped = (0..MAX_STEPS).map(|_| step(app, name, s, &mut line)).find(|called| *called != Ok(true));
    line(match stopped {
        Some(Ok(_)) => obj! {"ok" => true},
        Some(Err(e)) => obj! {"error" => e},
        None => obj! {"error" => format!("the turn stopped after {MAX_STEPS} steps")},
    });
    app.live.lock().unwrap().remove(name);
    for page in &live.lock().unwrap().pages {
        let _ = page.shutdown(std::net::Shutdown::Both);
    }
    if let Ok(setup) = setup(app, name) {
        form(app, name, s, &setup);
    }
    Ok(Value::Null)
}

/// The turn running in a session, for a page that comes in late: its events so far, then each as it
/// happens, until it ends; none if no turn runs.
fn follow(app: &App, name: &str, stream: &mut TcpStream) -> Result<Value, String> {
    let live = app.live.lock().unwrap().get(name).cloned();
    let mut page = stream.try_clone().map_err(|e| e.to_string())?;
    let _ = page.set_write_timeout(Some(PAGE_WAIT));
    let _ = page.write_all(EVENTS);
    if let Some(live) = live {
        let mut live = live.lock().unwrap();
        if live.events.iter().all(|e| page.write_all(e.as_bytes()).is_ok()) {
            live.pages.push(page);
        }
    }
    Ok(Value::Null)
}

/// One step: the context compiles and streams into a request, the reply streams to the page and joins
/// the log, and the tools it calls run, their results joining the log; then memories form. Whether it
/// called any.
fn step(app: &App, name: &str, s: &Session, line: &mut impl FnMut(Value)) -> Result<bool, String> {
    let setup = setup(app, name)?;
    let (cfg, reserve) = (&setup.s.config, setup.s.reserve);
    let (usage, last) = (app.usage.lock().unwrap().remove(name), current(s));
    let calibration = usage.into_iter().fold(last, Calibration::observe);
    // how much bigger the last request's context was than its compile thought, in a sample calibration
    // learns from: calibration takes steps to catch up, and the context compiles that much smaller meanwhile
    let lag = usage.map(|u| u / last.armed).filter(|r| (1.0..=1.8).contains(r)).unwrap_or(1.0);
    // what the context compiles to: the window less what every request carries, smaller by the lag, so
    // the request and its answer fit even at the library's grace over budget
    let budget = (setup.s.window - setup.overhead - reserve) / lag / (1.0 + cfg.over_budget_grace) + reserve;
    let attempt = |calibration| compile(s, budget, reserve, budget, calibration, cfg);
    // a refusal escalates the session, so the memory the context waits for forms at once; then it compiles again, up to 3 times
    let compiled = (0..3).fold(attempt(calibration), |compiled, _| compiled.or_else(|_| { form(app, name, s, &setup); attempt(current(s)) }));
    let turns = compiled.map_err(|r| format!("{}: {} tokens over a budget of {}", r.stage, r.actual, r.budget))?;
    let head = sparse(obj! {"model" => setup.s.model.as_str(), "max_tokens" => reserve, "system" => setup.s.system.as_deref(), "tools" => setup.tools.definitions()});
    let reply = setup.llm.messages(head, &turns, &mut Reader::new(s, false), |kind, piece| line(obj! {kind => piece}))?;
    app.usage.lock().unwrap().insert(name.to_string(), reply["usage"]["inputTokens"].as_f64().unwrap_or(0.0) - setup.overhead);
    append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, cfg);
    let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
    let results: Vec<Value> = calls.map(|call| {
        line(obj! {"tool" => call.clone()});
        let result = approve(app, name, &setup, call, line).map_or_else(|why| tools::result(call, Err(why.into())), |()| setup.tools.call(call));
        line(obj! {"result" => obj! {"id" => call["id"].clone(), "text" => content(&result["content"]), "error" => result["isError"] == true}});
        result
    }).collect();
    if results.is_empty() {
        return Ok(false);
    }
    append(s, &obj! {"participant" => "user", "content" => results}, cfg);
    form(app, name, s, &setup);
    Ok(true)
}

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// Whether the user lets a call run. A tool that doesn't ask, or that the session always allows, runs
/// at once (and a call to no tool fails at once); otherwise the page is asked ({"approve": the call's id}) and the call runs if
/// the answer (POST /api/approve) is allow, or always, which allows its tool in the session from then
/// on. A denial, or no answer within APPROVAL_WAIT, is the call's failed result.
fn approve(app: &App, name: &str, setup: &Setup, call: &Value, line: &mut impl FnMut(Value)) -> Result<(), String> {
    let (id, tool) = (call["id"].as_str().unwrap_or(""), call["name"].as_str().unwrap_or(""));
    if !setup.tools.0.iter().any(|t| t.asks && t.definition["name"] == tool) || profile(app, name)?.allow.contains(tool) {
        return Ok(());
    }
    let (answer, answered) = channel();
    app.asking.lock().unwrap().insert(id.to_string(), answer);
    line(obj! {"approve" => id});
    let answer = answered.recv_timeout(APPROVAL_WAIT).unwrap_or_default();
    app.asking.lock().unwrap().remove(id);
    line(obj! {"answered" => id});
    match answer.as_str() {
        "always" => change(app, &profile_path(app, name), false, |s: &mut SessionSettings| { s.allow.insert(tool.into()); Ok(()) }).map(drop),
        "allow" => Ok(()),
        _ => Err("the user didn't allow this call".into()),
    }
}

/// Memories form (memory::drive), the page told that the session is forming them.
fn form(app: &App, name: &str, s: &Session, setup: &Setup) {
    app.forming.lock().unwrap().insert(name.to_string());
    if let Some(error) = drive(s, &mut Summarizer(&setup.llm, setup.tools.definitions()), &setup.s.config) {
        eprintln!("{name}: forming a memory: {error}");
    }
    app.forming.lock().unwrap().remove(name);
}

/// What a session's step runs with, from the settings as they are now.
fn setup(app: &App, name: &str) -> Result<Setup, String> {
    let (all, s) = (global(app)?, profile(app, name)?);
    let p = all.providers.get(&s.provider).ok_or_else(|| format!("no provider {} in settings.json", s.provider))?;
    let llm = Llm { base: p.base.clone(), key: p.key.clone(), model: s.model.clone(), agent: app.agent.clone() };
    let bash = match &s.workspace {
        Some(dir) if app.bash => bash::tool(dir.clone(), app.limit, s.sandbox.then_some(all.sandbox)),
        None if app.bash => bash::unavailable("it has no workspace; the user can set one in the session's settings"),
        _ => Toolbox::default(),
    };
    let tools = bash + app.mcp.clone();
    Ok(Setup { overhead: overhead(s.system.as_deref(), &tools), s, llm, tools })
}

/// The tokens a system prompt and tool definitions take.
fn overhead(system: Option<&str>, tools: &Toolbox) -> f64 {
    density_tokens(system.unwrap_or("")) + tools.definitions().map_or(0.0, |d| density_tokens(&json::to_string(&d)))
}

fn global(app: &App) -> Result<Settings, String> {
    let path = app.root.join("settings.json");
    settings::read(&path)?.ok_or_else(|| format!("no {}", path.display()))
}

fn profile_path(app: &App, name: &str) -> PathBuf {
    app.root.join(name).join("session_settings.json")
}

/// A session's settings, its file written first, as a copy of the defaults, if it has none.
fn profile(app: &App, name: &str) -> Result<SessionSettings, String> {
    let path = profile_path(app, name);
    match settings::read(&path)? {
        Some(s) => Ok(s),
        None => global(app).and_then(|all| settings::write(&path, &all.defaults, false).map(|()| all.defaults)),
    }
}

/// A settings file as `f` changes it, written if `f` succeeds; one change at a time.
fn change<T: Serialize + DeserializeOwned>(app: &App, path: &Path, private: bool, f: impl FnOnce(&mut T) -> Result<(), String>) -> Result<T, String> {
    let _one = app.writing.lock().unwrap();
    let mut settings = settings::read(path)?.ok_or_else(|| format!("no {}", path.display()))?;
    f(&mut settings)?;
    settings::write(path, &settings, private).map(|()| settings)
}

/// The providers as the page sees them: never a key, only whether there is one.
fn providers(all: &Settings) -> Value {
    let shown = all.providers.iter().map(|(name, p)| obj! {"name" => name.as_str(), "base" => p.base.as_str(), "key" => !p.key.is_empty()});
    obj! {"providers" => shown.collect::<Vec<_>>()}
}

/// A provider added, or its base URL or key changed, by what `v` gives that isn't blank.
fn provide(all: &mut Settings, v: &Value) -> Result<(), String> {
    let given = |k: &str| v[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let p = all.providers.entry(given("name").ok_or("a provider's name")?).or_default();
    for (field, slot) in [("base", &mut p.base), ("key", &mut p.key)] {
        if let Some(s) = given(field) {
            *slot = s;
        }
    }
    if p.base.is_empty() { Err("a provider's base URL".into()) } else { Ok(()) }
}

/// The workspace `v` names: none for null or blank, else a directory that exists, made absolute.
fn workspace(v: &Value) -> Result<Option<PathBuf>, String> {
    let Some(path) = v.as_str().map(str::trim).filter(|p| !p.is_empty()) else { return Ok(None) };
    let dir = std::path::absolute(settings::expand(path)).map_err(|e| format!("{path}: {e}"))?;
    if dir.is_dir() { Ok(Some(dir)) } else { Err(format!("no directory {path}")) }
}

/// A session as the page shows it, its settings, if it's new, the defaults with the workspace `v` gives.
fn create(app: &App, v: &Value) -> Result<Value, String> {
    let name = v["name"].as_str().filter(|name| named(name)).ok_or("a session's name is letters, digits, - and _")?;
    let dir = app.root.join(name);
    if !profile_path(app, name).exists() {
        let first = SessionSettings { workspace: workspace(&v["workspace"])?, ..global(app)?.defaults };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{name}: {e}"))?;
        settings::write(&profile_path(app, name), &first, false)?;
    }
    Ok(state(app, name, &Session::open(dir)))
}

/// A session's settings changed by what `v` gives: any of its fields, and any of config's. Its
/// provider must be one settings.json has, and its merge_threshold stays once memories have merged.
fn tune(app: &App, name: &str, s: &Session, v: &Value) -> Result<Value, String> {
    let known = global(app)?.providers;
    profile(app, name)?;
    change(app, &profile_path(app, name), false, |old: &mut SessionSettings| {
        let mut new = serde_json::to_value(&*old).expect("settings are JSON");
        for (k, x) in v.as_object().into_iter().flatten() {
            match (k.as_str(), x) {
                ("config", Value::Object(fields)) => new["config"].as_object_mut().expect("config").extend(fields.clone()),
                ("workspace", x) => new[k] = workspace(x)?.map(|dir| dir.to_string_lossy().into_owned()).into(),
                _ => new[k] = x.clone(),
            }
        }
        let new: SessionSettings = serde_json::from_value(new).map_err(|e| e.to_string())?;
        if !known.contains_key(&new.provider) {
            return Err(format!("no provider {}", new.provider));
        }
        if new.config.merge_threshold != old.config.merge_threshold && s.count("layer2") > 0 {
            return Err("merge_threshold stays as it is once memories have merged".into());
        }
        *old = new;
        Ok(())
    })?;
    Ok(state(app, name, s))
}

/// A message as the page shows it: its reasoning apart, the rest as notes.
fn shown(m: &Value) -> Value {
    let (thinking, rest): (Vec<&Value>, Vec<&Value>) = m["content"].arr().iter().partition(|b| b.kind() == "thinking");
    let thinking: Vec<&str> = thinking.iter().filter_map(|b| b["thinking"].as_str()).collect();
    obj! {"id" => m.id(), "participant" => m["participant"].clone(), "thinking" => thinking.join("\n"), "text" => rest.into_iter().map(note).collect::<Vec<_>>().join("\n")}
}

/// A block as the page shows it: text as it is, images, tool calls and their results as short notes.
fn note(b: &Value) -> String {
    match b.kind() {
        "text" => b["text"].as_str().unwrap_or("").to_string(),
        "image" => format!("[image, {} KB]", b["source"]["data"].as_str().map_or(0, str::len) / 1024),
        "tool_use" => format!("[tool call {} {}]", b["name"].as_str().unwrap_or(""), json::to_string(&b["input"])),
        "tool_result" => format!("[tool result{}] {}", if b["isError"] == true { ", failed" } else { "" }, content(&b["content"])),
        kind => format!("[{kind}]"),
    }
}

/// A tool result's content as the page shows it: a string, or its blocks' notes.
fn content(c: &Value) -> String {
    c.as_str().map_or_else(|| c.arr().iter().map(note).collect::<Vec<_>>().join("\n"), str::to_string)
}

/// Every session: its name, messages, the session it forked from, and its workspace.
fn sessions(app: &App) -> Value {
    let mut names: Vec<String> = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect();
    names.sort();
    Value::Array(names.iter().map(|name| {
        let s = Session::open(app.root.join(name));
        let workspace = settings::read::<SessionSettings>(&profile_path(app, name)).ok().flatten().and_then(|p| p.workspace);
        obj! {"name" => name.as_str(), "messages" => s.count("messages"), "parent" => parent(&s), "workspace" => workspace.map(|w| w.to_string_lossy().into_owned())}
    }).collect())
}

/// The session a fork was made from, and at which message.
fn parent(s: &Session) -> Value {
    let name = |path: &str| std::path::Path::new(path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    s.fork["parent"].as_str().map_or(Value::Null, |path| obj! {"name" => name(path), "at" => s.fork["at"].clone()})
}

/// What the page shows of a session: counts, each layer's size, chunk spans, the last compile, and
/// the turn running, if one is ("running": the messages the log had once its message joined).
fn state(app: &App, name: &str, s: &Session) -> Value {
    let layers: Vec<Value> = (0..).map(|n| s.count(&format!("layer{n}"))).take_while(|&c| c > 0).map(Value::from).collect();
    let spans = (s.count("layer0") > 0).then(|| s.span("layer0", 0, s.count("layer0") - 1));
    let chunks: Vec<Value> = spans.into_iter().flatten().map(|c| Value::Array(vec![c["first"].clone(), c["last"].clone()])).collect();
    let last = s.last("compiles").unwrap_or(obj! {});
    let refused = s.last("attempts").is_some_and(|a| a["outcome"].as_str() == Some("refusal"));
    let (setup, live) = (setup(app, name), app.live.lock().unwrap().get(name).cloned());
    let asking = setup.iter().flat_map(|x| x.tools.0.iter().filter(|t| t.asks).filter_map(|t| t.definition["name"].as_str()).map(move |t| (x, t)));
    let tools: Vec<Value> = asking.map(|(x, t)| obj! {"name" => t, "allowed" => x.s.allow.contains(t)}).collect();
    obj! {
        "name" => name, "messages" => s.count("messages"), "layers" => layers, "chunks" => chunks,
        "k" => setup.as_ref().map_or(Config::default().merge_threshold, |x| x.s.config.merge_threshold),
        "frontier" => last["frontier"].clone(), "armed" => last["armed"].clone(), "multiplier" => last["multiplier"].clone(),
        "escalated" => last["escalated"].clone(), "refused" => refused,
        "forming" => app.forming.lock().unwrap().contains(name), "parent" => parent(s), "running" => live.map(|l| l.lock().unwrap().from),
        "settings" => setup.as_ref().ok().map(|x| serde_json::to_value(&x.s).expect("settings are JSON")), "problem" => setup.as_ref().err().map(String::as_str),
        "tools" => tools, "overhead" => setup.as_ref().map_or(0.0, |x| x.overhead),
    }
}
