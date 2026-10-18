//! connectome-web: chat with a model whose context the connectome library manages, in a web page that
//! shows each session's memory pyramid and what its context shows, and forks a session at any message.
//! One binary that needs nothing installed: the page is built in, and so is TLS (rustls, ring and
//! Mozilla's roots).
//!
//!     connectome-web [--root DIR] [--addr 127.0.0.1:8080] [--base URL] [--model NAME] [--window TOKENS]
//!                    [--reserve TOKENS] [--system TEXT] [--ca PEM] [--no-open]
//!
//! The provider, model and key are the page's settings, kept in DIR/settings.json; until they're set,
//! --base, --model and API_KEY, ANTHROPIC_API_KEY or DEEPSEEK_API_KEY give them. --ca trusts the
//! root certificates in a PEM file instead of Mozilla's (a company proxy's, a test server's). A chat message joins
//! the log, the context compiles and streams from disk into the request, the reply joins the log, and
//! then, after the page has the reply, memories form while the session is locked. The page opens in
//! the browser unless --no-open.

mod llm;

use connectome::config::Config;
use connectome::drive::{append, compile, current, drive};
use connectome::emit::Reader;
use connectome::json::{self, sparse, Json, Value};
use connectome::memory::{Host, Request};
use connectome::obj;
use connectome::store::Session;
use llm::Llm;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const PAGE: &str = include_str!("ui.html");

struct App {
    root: PathBuf,
    addr: String,
    llm: Mutex<Llm>,
    cfg: Config,
    window: f64,
    reserve: f64,
    system: Option<String>,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    usage: Mutex<HashMap<String, f64>>, // each session's last prompt size, for calibration
    forming: Mutex<HashSet<String>>,    // sessions forming memories
}

struct Summarizer<'a>(&'a Llm);

impl Host for Summarizer<'_> {
    fn tools(&self) -> Option<&Value> {
        None
    }

    fn system(&self) -> Option<&str> {
        None
    }

    /// The request carries no system prompt or tools: this host declares neither.
    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.0.model } else { &request.model };
        self.0.messages(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens}, &request.messages, reader, |_, _| {})
    }
}

/// glibc raises its mmap threshold after a large block is freed, and keeps later large blocks in
/// per-thread arenas: a message of a megabyte read on many connection threads then stays resident.
/// A fixed threshold returns each large block to the system once it's freed.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn return_large_blocks() {
    unsafe extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
    }
    const M_MMAP_THRESHOLD: i32 = -3;
    unsafe { mallopt(M_MMAP_THRESHOLD, 128 * 1024) };
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn return_large_blocks() {}

fn main() {
    return_large_blocks();
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.into());
    let key = ["API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"].iter().find_map(|k| std::env::var(k).ok()).unwrap_or_default();
    let window: f64 = arg("--window", "60000").parse().expect("--window");
    let cfg = Config {
        summary_participant: "assistant".into(),
        recent_window_tokens: (window / 4.0).min(30000.0), // the library's defaults, for windows of 120k and more
        target_chunk_tokens: (window / 16.0).min(3000.0),
        compression_max_tokens: Some(8000.0),
        ..Config::default()
    };
    let extra = args.iter().any(|a| a == "--ca").then(|| std::fs::read(arg("--ca", "")).expect("--ca"));
    let agent = llm::agent(extra.as_deref()).expect("--ca");
    let root = PathBuf::from(arg("--root", "sessions"));
    let addr = arg("--addr", "127.0.0.1:8080");
    std::fs::create_dir_all(&root).expect("--root");
    let mut llm = Llm { base: arg("--base", "https://api.anthropic.com"), key, model: arg("--model", "claude-sonnet-5-5"), agent };
    if let Some(saved) = std::fs::read(root.join("settings.json")).ok().and_then(|b| json::parse(&b).ok()) {
        apply(&mut llm, &saved);
    }
    let app = Arc::new(App {
        root: root.clone(),
        addr: addr.clone(),
        llm: Mutex::new(llm),
        cfg,
        window,
        reserve: arg("--reserve", "4000").parse().expect("--reserve"),
        system: args.iter().any(|a| a == "--system").then(|| arg("--system", "")),
        locks: Mutex::default(),
        usage: Mutex::default(),
        forming: Mutex::default(),
    });
    let listener = TcpListener::bind(&addr).expect("--addr");
    eprintln!("connectome-web on http://{addr}, sessions in {}", app.root.display());
    if !args.iter().any(|a| a == "--no-open") {
        let _ = open::that_detached(format!("http://{addr}"));
    }
    for stream in listener.incoming().flatten() {
        let app = app.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(&app, stream) {
                eprintln!("{e}");
            }
        });
    }
}

/// One HTTP/1.1 request, then the connection closes. Only this server's own page gets an answer: a
/// request must name this server as its Host (another site's page reaching it through a DNS name it
/// controls doesn't), and a POST must carry an x-connectome header (another site's page can't send one
/// without a CORS preflight, which this server never approves), since a POST can change where the key
/// goes.
fn handle(app: &App, mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut words = line.split_whitespace();
    let (method, target) = (words.next().unwrap_or("").to_string(), words.next().unwrap_or("/").to_string());
    let (mut length, mut host, mut marked) = (0, String::new(), false);
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let Some((k, v)) = header.trim_end().split_once(':') else { break };
        match k.to_ascii_lowercase().as_str() {
            "content-length" => length = v.trim().parse().unwrap_or(0).min(1 << 20),
            "host" => host = v.trim().to_string(),
            "x-connectome" => marked = true,
            _ => {}
        }
    }
    let port = app.addr.rsplit(':').next().unwrap_or("");
    if ![app.addr.clone(), format!("localhost:{port}")].contains(&host) || method == "POST" && !marked {
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
        ("GET", ["api", "settings"]) => Ok(settings(&app.llm.lock().unwrap())),
        ("POST", ["api", "settings"]) => {
            let mut llm = app.llm.lock().unwrap();
            apply(&mut llm, &body);
            save(&app.root, &llm).map(|_| settings(&llm)).map_err(|e| format!("saving settings: {e}"))
        }
        ("POST", ["api", "sessions"]) => match body["name"].as_str().filter(|name| named(name)) {
            Some(name) => Ok(state(app, name, &Session::open(app.root.join(name)))),
            None => Err("a session's name is letters, digits, - and _".into()),
        },
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
            Ok(state(app, fork, &child))
        }
        ("POST", ["say"]) => say(app, name, &s, body["text"].as_str().unwrap_or(""), stream),
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

/// A chat turn: the message joins the log, the context compiles and streams into the request, and
/// the reply streams to the page as it comes, one JSON line per piece of text or reasoning ({"text"}
/// or {"thinking"}), then
/// {"ok": true} once it has joined the log, or {"error"}. Then memories form, the session still locked.
fn say(app: &App, name: &str, s: &Session, text: &str, stream: &mut TcpStream) -> Result<Value, String> {
    if text.trim().is_empty() {
        return Err("say something".into());
    }
    let lock = lock(app, name);
    let _held = lock.lock().unwrap();
    // a page that has gone away still gets its reply logged
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n");
    let mut line = |event: Value| {
        let _ = stream.write_all(format!("{}\n", json::to_string(&event)).as_bytes());
    };
    append(s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &app.cfg);
    let mut calibration = current(s);
    if let Some(usage) = app.usage.lock().unwrap().remove(name) {
        calibration = calibration.observe(usage);
    }
    match compile(s, app.window, app.reserve, app.window, calibration, &app.cfg) {
        Err(refusal) => line(obj! {"error" => format!("{}: {} tokens over a budget of {}", refusal.stage, refusal.actual, refusal.budget)}),
        Ok(turns) => {
            let mut reader = Reader::new(s, false);
            let llm = app.llm.lock().unwrap().clone();
            let head = sparse(obj! {"model" => llm.model.as_str(), "max_tokens" => app.reserve, "system" => app.system.as_deref()});
            match llm.messages(head, &turns, &mut reader, |kind, piece| line(obj! {kind => piece})) {
                Err(e) => line(obj! {"error" => e}),
                Ok(r) => {
                    app.usage.lock().unwrap().insert(name.to_string(), r["usage"]["inputTokens"].as_f64().unwrap_or(0.0));
                    append(s, &obj! {"participant" => "assistant", "content" => r["content"].clone()}, &app.cfg);
                    line(obj! {"ok" => true});
                }
            }
        }
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
    app.forming.lock().unwrap().insert(name.to_string());
    let llm = app.llm.lock().unwrap().clone();
    if let Some(error) = drive(s, &mut Summarizer(&llm), &app.cfg) {
        eprintln!("{name}: forming a memory: {error}");
    }
    app.forming.lock().unwrap().remove(name);
    Ok(Value::Null)
}

/// The settings the page shows: never the key, only whether there is one.
fn settings(llm: &Llm) -> Value {
    obj! {"base" => llm.base.as_str(), "model" => llm.model.as_str(), "key" => !llm.key.is_empty()}
}

/// The settings `v` gives (base, model, key) that aren't blank.
fn apply(llm: &mut Llm, v: &Value) {
    for (key, field) in [("base", &mut llm.base), ("model", &mut llm.model), ("key", &mut llm.key)] {
        if let Some(s) = v[key].as_str().map(str::trim).filter(|s| !s.is_empty()) {
            *field = s.to_string();
        }
    }
}

/// Writes ROOT/settings.json, which holds the key: only this user can read it.
fn save(root: &std::path::Path, llm: &Llm) -> std::io::Result<()> {
    let path = root.join("settings.json");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let text = json::to_string(&obj! {"base" => llm.base.as_str(), "model" => llm.model.as_str(), "key" => llm.key.as_str()});
    options.open(path)?.write_all(text.as_bytes())
}

/// A message as the page shows it: its reasoning apart, images and tool calls as short notes.
fn shown(m: &Value) -> Value {
    let note = |b: &Value| match b.kind() {
        "text" => b["text"].as_str().unwrap_or("").to_string(),
        "image" => format!("[image, {} KB]", b["source"]["data"].as_str().map_or(0, str::len) / 1024),
        "tool_use" => format!("[tool call {} {}]", b["name"].as_str().unwrap_or(""), json::to_string(&b["input"])),
        "tool_result" => format!("[tool result] {}", b["content"].as_str().unwrap_or("")),
        kind => format!("[{kind}]"),
    };
    let (thinking, rest): (Vec<&Value>, Vec<&Value>) = m["content"].arr().iter().partition(|b| b.kind() == "thinking");
    let thinking: Vec<&str> = thinking.iter().filter_map(|b| b["thinking"].as_str()).collect();
    obj! {"id" => m.id(), "participant" => m["participant"].clone(), "thinking" => thinking.join("\n"), "text" => rest.into_iter().map(note).collect::<Vec<_>>().join("\n")}
}

/// Every session: its name, messages, and the session it forked from.
fn sessions(app: &App) -> Value {
    let mut names: Vec<String> = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect();
    names.sort();
    Value::Array(names.iter().map(|name| {
        let s = Session::open(app.root.join(name));
        obj! {"name" => name.as_str(), "messages" => s.count("messages"), "parent" => parent(&s)}
    }).collect())
}

/// The session a fork was made from, and at which message.
fn parent(s: &Session) -> Value {
    let name = |path: &str| std::path::Path::new(path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    s.fork["parent"].as_str().map_or(Value::Null, |path| obj! {"name" => name(path), "at" => s.fork["at"].clone()})
}

/// What the page shows of a session: counts, each layer's size, chunk spans, and the last compile.
fn state(app: &App, name: &str, s: &Session) -> Value {
    let layers: Vec<Value> = (0..).map(|n| s.count(&format!("layer{n}"))).take_while(|&c| c > 0).map(Value::from).collect();
    let spans = (s.count("layer0") > 0).then(|| s.span("layer0", 0, s.count("layer0") - 1));
    let chunks: Vec<Value> = spans.into_iter().flatten().map(|c| Value::Array(vec![c["first"].clone(), c["last"].clone()])).collect();
    let last = s.last("compiles").unwrap_or(obj! {});
    let refused = s.last("attempts").is_some_and(|a| a["outcome"].as_str() == Some("refusal"));
    obj! {
        "name" => name, "messages" => s.count("messages"), "layers" => layers, "chunks" => chunks, "k" => app.cfg.merge_threshold,
        "frontier" => last["frontier"].clone(), "armed" => last["armed"].clone(), "multiplier" => last["multiplier"].clone(),
        "escalated" => last["escalated"].clone(), "window" => app.window, "refused" => refused,
        "forming" => app.forming.lock().unwrap().contains(name), "parent" => parent(s), "model" => app.llm.lock().unwrap().model.as_str(),
    }
}
