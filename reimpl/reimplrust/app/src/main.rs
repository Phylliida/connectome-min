//! connectome-web: chat with a model whose context the connectome library manages, in a web page that
//! shows each session's memory pyramid and what its context shows, and forks a session at any message.
//! One binary that needs nothing installed: the page is built in, and so is TLS (rustls, ring and
//! Mozilla's roots).
//!
//!     connectome-web [--root DIR] [--addr 127.0.0.1:8080] [--base URL] [--model NAME] [--window TOKENS]
//!                    [--reserve TOKENS] [--system TEXT] [--ca PEM] [--mcp FILE] [--workdir DIR]
//!                    [--tool-timeout SECONDS] [--no-bash] [--no-open]
//!
//! The provider, model and key are the page's settings, kept in DIR/settings.json; until they're set,
//! --base, --model and API_KEY, ANTHROPIC_API_KEY or DEEPSEEK_API_KEY give them. --ca trusts the
//! root certificates in a PEM file instead of Mozilla's (a company proxy's, a test server's). The model
//! can call bash (bash.rs; in --workdir, by default the current directory; unless --no-bash) and the
//! tools of the MCP servers that --mcp names (by default DIR/mcp.json, if there is one; mcp.rs), each
//! call once the user allows it on the page, or always allows its tool, and within --tool-timeout
//! (600 s by default). The system prompt and the tool definitions come out of the window the context
//! compiles to, since every request carries them. A chat message joins the log, then the turn takes steps
//! until a reply calls no tools: the context compiles and streams from disk into the request, the
//! reply joins the log, and the tools it calls run, their results joining the log. Memories form
//! between steps and, once the page has the turn, after it, the session locked throughout. The page
//! opens in the browser unless --no-open.

mod bash;
mod llm;
mod mcp;
mod tools;

use connectome::blocks::density_tokens;
use connectome::config::Config;
use connectome::drive::{append, compile, current, drive};
use connectome::emit::Reader;
use connectome::json::{self, sparse, Json, Value};
use connectome::memory::{Host, Request};
use connectome::obj;
use connectome::store::Session;
use llm::Llm;
use tools::Toolbox;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PAGE: &str = include_str!("ui.html");

struct App {
    root: PathBuf,
    addr: String,
    llm: Mutex<Llm>,
    cfg: Config,
    window: f64,
    reserve: f64,
    system: Option<String>,
    tools: Toolbox,
    overhead: f64,                                  // the system prompt's and tool definitions' tokens, which every request carries
    allowed: Mutex<BTreeSet<String>>,               // the tools the user always allows
    asking: Mutex<HashMap<String, Sender<String>>>, // the calls waiting for the user's answer, by id
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    usage: Mutex<HashMap<String, f64>>, // each session's last prompt size, for calibration
    forming: Mutex<HashSet<String>>,    // sessions forming memories
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
    let window: f64 = arg("--window", "60000").parse().expect("--window");
    let extra = args.iter().any(|a| a == "--ca").then(|| std::fs::read(arg("--ca", "")).expect("--ca"));
    let agent = llm::agent(extra.as_deref()).expect("--ca");
    let root = PathBuf::from(arg("--root", "sessions"));
    let addr = arg("--addr", "127.0.0.1:8080");
    std::fs::create_dir_all(&root).expect("--root");
    let limit = Duration::from_secs_f64(arg("--tool-timeout", "600").parse().expect("--tool-timeout"));
    let bash = if args.iter().any(|a| a == "--no-bash") { Toolbox::default() } else { bash::tool(std::path::absolute(arg("--workdir", ".")).expect("--workdir"), limit) };
    let mcp = std::fs::read(arg("--mcp", &root.join("mcp.json").to_string_lossy())).ok();
    let tools = bash + mcp.map_or_else(Toolbox::default, |config| mcp::toolbox(&json::parse(&config).expect("--mcp: JSON"), limit));
    let system = args.iter().any(|a| a == "--system").then(|| arg("--system", ""));
    let overhead = density_tokens(system.as_deref().unwrap_or("")) + tools.definitions().map_or(0.0, |d| density_tokens(&json::to_string(&d)));
    let left = window - overhead; // what the context compiles to
    let cfg = Config {
        summary_participant: "assistant".into(),
        recent_window_tokens: (left / 4.0).min(30000.0), // the library's defaults, for windows of 120k and more
        target_chunk_tokens: (left / 16.0).min(3000.0),
        compression_max_tokens: Some(8000.0),
        ..Config::default()
    };
    let app = Arc::new(App {
        root: root.clone(),
        addr: addr.clone(),
        llm: Mutex::new(Llm { base: arg("--base", "https://api.anthropic.com"), key, model: arg("--model", "claude-sonnet-5-5"), agent }),
        cfg,
        window,
        reserve: arg("--reserve", "4000").parse().expect("--reserve"),
        system,
        tools,
        overhead,
        allowed: Mutex::default(),
        asking: Mutex::default(),
        locks: Mutex::default(),
        usage: Mutex::default(),
        forming: Mutex::default(),
    });
    apply(&app, &std::fs::read(root.join("settings.json")).ok().and_then(|b| json::parse(&b).ok()).unwrap_or_default());
    let listener = TcpListener::bind(&addr).expect("--addr");
    eprintln!("connectome-web on http://{addr}, sessions in {}, tools: {:?} ({overhead} tokens with the system prompt)", app.root.display(), app.tools.names());
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
        ("GET", ["api", "settings"]) => Ok(settings(&app.llm.lock().unwrap())),
        ("POST", ["api", "settings"]) => {
            apply(app, &body);
            save(app).map(|()| settings(&app.llm.lock().unwrap()))
        }
        ("POST", ["api", "approve"]) => match app.asking.lock().unwrap().remove(body["id"].as_str().unwrap_or("")) {
            Some(answer) => Ok(obj! {"answered" => answer.send(body["answer"].as_str().unwrap_or("deny").to_string()).is_ok()}),
            None => Err("no call waits for that answer".into()),
        },
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

/// Steps a turn may take, a model request each, before it stops.
const MAX_STEPS: usize = 64;

/// A chat turn: the message joins the log, then steps run until a reply calls no tools. The page gets
/// one JSON line per event as it happens: the pieces of a reply's reasoning and text ({"thinking"},
/// {"text"}), each tool call ({"tool"}: the tool_use block) and its result ({"result"}: {id, text,
/// error}); then {"ok": true} once the turn has joined the log, or {"error"}. Memories form between
/// steps and after the turn, the session locked throughout.
fn say(app: &App, name: &str, s: &Session, text: &str, stream: &mut TcpStream) -> Result<Value, String> {
    if text.trim().is_empty() {
        return Err("say something".into());
    }
    let lock = lock(app, name);
    let _held = lock.lock().unwrap();
    // a page that has gone away still gets its turn logged
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n");
    let mut line = |event: Value| drop(stream.write_all(format!("{}\n", json::to_string(&event)).as_bytes()));
    append(s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &app.cfg);
    let llm = app.llm.lock().unwrap().clone();
    let stopped = (0..MAX_STEPS).map(|_| step(app, name, s, &llm, &mut line)).find(|called| *called != Ok(true));
    line(match stopped {
        Some(Ok(_)) => obj! {"ok" => true},
        Some(Err(e)) => obj! {"error" => e},
        None => obj! {"error" => format!("the turn stopped after {MAX_STEPS} steps")},
    });
    let _ = stream.shutdown(std::net::Shutdown::Both);
    form(app, name, s, &llm);
    Ok(Value::Null)
}

/// One step: the context compiles and streams into a request, the reply streams to the page and joins
/// the log, and the tools it calls run, their results joining the log; then memories form. Whether it
/// called any.
fn step(app: &App, name: &str, s: &Session, llm: &Llm, line: &mut impl FnMut(Value)) -> Result<bool, String> {
    let calibration = app.usage.lock().unwrap().remove(name).into_iter().fold(current(s), |c, usage| c.observe(usage));
    // what the context compiles to: the window less what every request carries, so the request and its
    // answer fit even at the library's grace over budget
    let budget = (app.window - app.overhead - app.reserve) / (1.0 + app.cfg.over_budget_grace) + app.reserve;
    let attempt = |calibration| compile(s, budget, app.reserve, budget, calibration, &app.cfg);
    // a refusal escalates the session, so the memory the context waits for forms at once; then it compiles again, up to 3 times
    let compiled = (0..3).fold(attempt(calibration), |compiled, _| compiled.or_else(|_| { form(app, name, s, llm); attempt(current(s)) }));
    let turns = compiled.map_err(|r| format!("{}: {} tokens over a budget of {}", r.stage, r.actual, r.budget))?;
    let head = sparse(obj! {"model" => llm.model.as_str(), "max_tokens" => app.reserve, "system" => app.system.as_deref(), "tools" => app.tools.definitions()});
    let reply = llm.messages(head, &turns, &mut Reader::new(s, false), |kind, piece| line(obj! {kind => piece}))?;
    app.usage.lock().unwrap().insert(name.to_string(), reply["usage"]["inputTokens"].as_f64().unwrap_or(0.0) - app.overhead);
    append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, &app.cfg);
    let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
    let results: Vec<Value> = calls.map(|call| {
        line(obj! {"tool" => call.clone()});
        let result = approve(app, call, line).map_or_else(|why| tools::result(call, Err(why.into())), |()| app.tools.call(call));
        line(obj! {"result" => obj! {"id" => call["id"].clone(), "text" => content(&result["content"]), "error" => result["isError"] == true}});
        result
    }).collect();
    if results.is_empty() {
        return Ok(false);
    }
    append(s, &obj! {"participant" => "user", "content" => results}, &app.cfg);
    form(app, name, s, llm);
    Ok(true)
}

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// Whether the user lets a call run. A tool always allowed runs at once (and a call to no tool fails
/// at once); otherwise the page is asked ({"approve": the call's id}) and the call runs if the answer
/// (POST /api/approve) is allow, or always, which allows its tool from then on. A denial, or no answer
/// within APPROVAL_WAIT, is the call's failed result.
fn approve(app: &App, call: &Value, line: &mut impl FnMut(Value)) -> Result<(), String> {
    let (id, tool) = (call["id"].as_str().unwrap_or(""), call["name"].as_str().unwrap_or(""));
    if !app.tools.names().contains(&tool) || app.allowed.lock().unwrap().contains(tool) {
        return Ok(());
    }
    let (answer, answered) = channel();
    app.asking.lock().unwrap().insert(id.to_string(), answer);
    line(obj! {"approve" => id});
    let answer = answered.recv_timeout(APPROVAL_WAIT).unwrap_or_default();
    app.asking.lock().unwrap().remove(id);
    match answer.as_str() {
        "always" => { app.allowed.lock().unwrap().insert(tool.to_string()); save(app) }
        "allow" => Ok(()),
        _ => Err("the user didn't allow this call".into()),
    }
}

/// Memories form (memory::drive), the page told that the session is forming them.
fn form(app: &App, name: &str, s: &Session, llm: &Llm) {
    app.forming.lock().unwrap().insert(name.to_string());
    if let Some(error) = drive(s, &mut Summarizer(llm, app.tools.definitions()), &app.cfg) {
        eprintln!("{name}: forming a memory: {error}");
    }
    app.forming.lock().unwrap().remove(name);
}

/// The settings the page shows: never the key, only whether there is one.
fn settings(llm: &Llm) -> Value {
    obj! {"base" => llm.base.as_str(), "model" => llm.model.as_str(), "key" => !llm.key.is_empty()}
}

/// The settings `v` gives: base, model and key, those that aren't blank, and the tools always allowed.
fn apply(app: &App, v: &Value) {
    let llm = &mut *app.llm.lock().unwrap();
    for (key, field) in [("base", &mut llm.base), ("model", &mut llm.model), ("key", &mut llm.key)] {
        if let Some(s) = v[key].as_str().map(str::trim).filter(|s| !s.is_empty()) {
            *field = s.to_string();
        }
    }
    if let Some(allow) = v["allow"].as_array() {
        *app.allowed.lock().unwrap() = allow.iter().filter_map(Value::as_str).map(str::to_string).collect();
    }
}

/// Writes ROOT/settings.json, which holds the key: only this user can read it.
fn save(app: &App) -> Result<(), String> {
    let (llm, allow) = (app.llm.lock().unwrap().clone(), app.allowed.lock().unwrap().clone());
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let text = json::to_string(&obj! {"base" => llm.base.as_str(), "model" => llm.model.as_str(), "key" => llm.key.as_str(), "allow" => Vec::from_iter(allow)});
    options.open(app.root.join("settings.json")).and_then(|mut file| file.write_all(text.as_bytes())).map_err(|e| format!("saving settings: {e}"))
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
        "tools" => app.tools.names().into_iter().map(|t| obj! {"name" => t, "allowed" => app.allowed.lock().unwrap().contains(t)}).collect::<Vec<_>>(),
        "overhead" => app.overhead,
    }
}
