//! connectome-web: chat with a model whose context the connectome library manages, in a web page that
//! shows each session's memory pyramid and what its context shows, and forks a session at any message.
//! One binary that needs nothing installed: the page is built in, and so is TLS (rustls, ring and
//! Mozilla's roots). README.md describes its flags, its settings files (settings.rs) and its tools
//! (bash.rs, mcp.rs).
//!
//! Whatever changes has one thread that owns it, and threads tell each other things through channels,
//! sharing nothing that changes. A request's thread reads files and tells the hub what it wants of a
//! session. The hub owns what changes quickly: the settings files, which sessions have a thread, and
//! what the pages following each session's turn see. Each session's thread (work) does its jobs one
//! at a time, the only one to write its log: a turn's chat message joins the log, then the turn takes
//! steps until a reply calls no tools, each reading the settings again: the context compiles and
//! streams from disk into the request, the reply joins the log, and the tools it calls run, each once
//! the user allows it, their results joining the log. Memories form between steps and after the turn.
//! A session's files are its only record: the hub keeps beside them only what they don't hold yet.

mod bash;
mod llm;
mod mcp;
mod settings;
mod tools;

use crate::{llm::Llm, settings::{SessionSettings, Settings}, tools::Toolbox};
use connectome::{blocks::density_tokens, drive::{append, current, drive}, emit::{select, Reader}, json::{self, sparse, Json, Value}, obj};
use connectome::{memory::{Host, Request}, store::Session};
use serde::{de::DeserializeOwned, Serialize, Serializer};
use std::collections::{BTreeSet, HashMap};
use std::{io::{BufRead, BufReader, BufWriter, Read, Write}, net::{TcpListener, TcpStream}, panic::{catch_unwind, AssertUnwindSafe}};
use std::{path::{Path, PathBuf}, sync::mpsc::{channel, Receiver, Sender}, time::Duration};

const PAGE: &str = include_str!("ui.html");

/// The page's libraries, as published: marked, which turns the assistant's markdown into HTML, and
/// DOMPurify, which cleans that HTML (vendor/).
const LIBS: &str = concat!(include_str!("../vendor/marked.umd.js"), "\n;", include_str!("../vendor/purify.min.js"));

/// What every thread reads, set once at the start and never changed.
struct App {
    root: PathBuf,
    addr: String,
    agent: ureq::Agent,
    mcp: Vec<mcp::Server>, // every session's MCP servers
    bash: bool,            // whether a session with a workspace has bash
    limit: Duration,       // a tool call's time
    hub: Sender<Routed>,
}

/// What the hub is told: a message about a session, or a change to settings.json, answered with the
/// settings it leaves.
enum Routed {
    To(String, Msg),
    Provide(Value, Sender<Result<Settings, String>>),
}

/// A job for a session's thread: a turn with a message, or without (None) one that continues the log,
/// which `page` follows; or a fork at message `at` as `name`.
enum Job {
    Say { text: Option<String>, page: Sender<String> },
    Fork { at: u64, name: String, reply: Sender<Result<(), String>> },
}

/// What the hub is told about a session, by requests' threads and by the session's.
enum Msg {
    /// A job for the session's thread, which does them in the order they come.
    Job(Job),
    /// A page that follows the job running; with none, its stream ends at once.
    Follow(Sender<String>),
    /// A page's answer whether call `id` may run: allow, always (its tool, from then on) or deny.
    Approve { id: String, answer: String, reply: Sender<Result<(), String>> },
    /// Whether a job runs.
    Status(Sender<bool>),
    /// A change to the session's settings (tune), and whether its memories have merged.
    Tune(Value, bool, Sender<Result<(), String>>),
    /// From the session's thread: a turn starts, which `page` follows.
    Start(Sender<String>),
    /// From the session's thread: the log has grown to this many messages.
    Logged(u64),
    /// From the session's thread: an event for the pages: a piece of a reply, a call's result, whether
    /// memories are forming, the turn's end.
    Event(Value),
    /// From the session's thread: a call's question, and where to answer it (Ok: it runs).
    Ask { id: String, tool: String, answer: Sender<Result<(), String>> },
    /// From the session's thread: the job is done, its memories formed; no call waits any more.
    Done,
}

/// The hub's thread. It owns what changes quickly, so each is told one thing at a time: settings.json
/// and the sessions' settings files, which sessions have a thread (started the first time one is
/// told something), and what the pages following each session's turn see (Live).
fn hub(app: &'static App, inbox: Receiver<Routed>) {
    let mut sessions: HashMap<String, Live> = HashMap::new();
    for routed in inbox {
        match routed {
            Routed::To(name, msg) => sessions.entry(name.clone()).or_insert_with(|| Live::start(app, &name)).handle(app, &name, msg),
            Routed::Provide(v, reply) => drop(reply.send(change(&global_path(app), |all| provide(all, &v)))),
        }
    }
}

/// The answer to what the hub is told, which says where to answer.
fn ask<T>(app: &App, told: impl FnOnce(Sender<T>) -> Routed) -> T {
    let (reply, answer) = channel();
    app.hub.send(told(reply)).expect("the hub runs");
    answer.recv().expect("the thread told answers")
}

/// The answer to what the hub is told about a session.
fn ask_session<T>(app: &App, name: &str, msg: impl FnOnce(Sender<T>) -> Msg) -> T {
    ask(app, |reply| Routed::To(name.into(), msg(reply)))
}

/// Tells the hub something about a session that wants no answer.
fn tell_session(app: &App, name: &str, msg: Msg) {
    app.hub.send(Routed::To(name.into(), msg)).expect("the hub runs");
}

/// A session as the hub knows it: where its thread takes jobs, and what the pages following its job see:
/// whether one runs (a turn, then its memories forming), the events since the log last grew (the first its length; then a reply streaming in, the questions whether the step's
/// calls may run, the answers, the calls' results, the memories forming, the turn's end), the pages,
/// and the calls waiting for an answer. What the log holds it drops, so it holds a step at most.
struct Live {
    jobs: Sender<Job>,
    running: bool,
    events: Vec<String>, // JSON lines
    pages: Vec<Sender<String>>,
    asked: HashMap<String, (String, Sender<Result<(), String>>)>, // by call id: its tool, where to answer
}

impl Live {
    /// A session, its thread started.
    fn start(app: &'static App, name: &str) -> Live {
        let (jobs, queue) = channel();
        let (name, hub) = (name.to_string(), app.hub.clone());
        std::thread::spawn(move || work(app, &name, queue, &|msg| drop(hub.send(Routed::To(name.clone(), msg)))));
        Live { jobs, running: false, events: vec![], pages: vec![], asked: HashMap::new() }
    }

    fn handle(&mut self, app: &App, name: &str, msg: Msg) {
        match msg {
            Msg::Job(job) => drop(self.jobs.send(job)),
            Msg::Follow(page) => if self.running && self.events.iter().all(|line| page.send(line.clone()).is_ok()) { self.pages.push(page) },
            Msg::Approve { id, answer, reply } => {
                let Some((tool, waiting)) = self.asked.remove(&id) else { return drop(reply.send(Err("no call waits for that answer".into()))) };
                let decided = match answer.as_str() {
                    "always" => change(&profile_path(app, name), |s: &mut SessionSettings| { s.allow.insert(tool); Ok(()) }).map(drop),
                    "allow" => Ok(()),
                    _ => Err(DENIED.into()),
                };
                let _ = waiting.send(decided);
                self.send(&obj! {"answered" => id, "answer" => answer});
                let _ = reply.send(Ok(()));
            }
            Msg::Status(reply) => drop(reply.send(self.running)),
            Msg::Tune(v, merged, reply) => drop(reply.send(tune(app, name, &v, merged))),
            Msg::Start(page) => (self.running, self.events, self.pages) = (true, vec![], vec![page]),
            Msg::Logged(n) => {
                self.events.clear();
                self.send(&obj! {"logged" => n});
            }
            Msg::Event(event) => self.send(&event),
            Msg::Ask { id, tool, answer } => {
                self.send(&obj! {"approve" => id.as_str()});
                self.asked.insert(id, (tool, answer));
            }
            Msg::Done => (self.running, self.pages, self.asked) = (false, vec![], HashMap::new()),
        }
    }

    /// An event to the pages following the job, kept for those that come later.
    fn send(&mut self, event: &Value) {
        let line = format!("{}\n", json::to_string(event));
        self.pages.retain(|page| page.send(line.clone()).is_ok());
        self.events.push(line);
    }
}

/// A job to a page as it happens, a JSON line each (Live), until the hub stops sending to the channel
/// `follow` gives it.
fn show(page: &mut TcpStream, follow: impl FnOnce(Sender<String>)) -> Route {
    let (sender, lines) = channel();
    follow(sender);
    let _ = page.set_write_timeout(Some(PAGE_WAIT));
    let _ = std::iter::once(head("200 OK", "application/x-ndjson", None)).chain(lines).try_for_each(|line| page.write_all(line.as_bytes()));
    Ok(None)
}

/// How long a page may keep a write waiting before it's left behind.
const PAGE_WAIT: Duration = Duration::from_secs(30);

/// What a session's step runs with: its settings, the model client and tools they give, the tools'
/// definitions, and the tokens the system prompt and those take, which every request carries.
struct Setup {
    settings: SessionSettings,
    llm: Llm,
    tools: Toolbox,
    definitions: Option<Value>,
    overhead: f64,
}

/// The summarizer's host: the session's model unless the library names one, and the agent's tools,
/// which the library sends with tool choice "none" so a memory's request can carry tool blocks; no
/// system prompt.
impl Host for &Setup {
    fn tools(&self) -> Option<&Value> { self.definitions.as_ref() }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.settings.model } else { &request.model };
        let none = request.tools.as_ref().map(|_| obj! {"type" => "none"});
        let head = sparse(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens, "tools" => request.tools.clone(), "tool_choice" => none});
        self.llm.messages(head, &request.messages, reader, |_, _| {})
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
    let mcp = std::fs::read(arg("--mcp", &root.join("mcp.json").to_string_lossy())).ok();
    let mcp = mcp.map_or_else(Vec::new, |config| mcp::servers(&json::parse(&config).expect("--mcp: JSON"), limit));
    let (told, inbox) = channel();
    let (addr, agent) = (arg("--addr", "127.0.0.1:8080"), llm::agent(ca.as_deref()).expect("--ca"));
    let app: &'static App = Box::leak(Box::new(App { root, addr, agent, mcp, bash: !flag("--no-bash"), limit, hub: told }));
    if !global_path(app).exists() {
        let tools = if app.bash { bash::tool(PathBuf::new(), limit, None) } else { Toolbox::default() } + mcp::toolbox(&app.mcp);
        settings::write(&global_path(app), &Settings::first(overhead(None, &tools.definitions()))).expect("settings.json");
    }
    std::thread::spawn(move || hub(app, inbox));
    let listener = TcpListener::bind(&app.addr).expect("--addr");
    eprintln!("connectome-web on http://{}, sessions in {}, MCP tools: {:?}", app.addr, app.root.display(), mcp::toolbox(&app.mcp).names());
    if !flag("--no-open") { let _ = open::that_detached(format!("http://{}", app.addr)); }
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || handle(app, stream).unwrap_or_else(|e| eprintln!("{e}")));
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
    let headers: HashMap<String, String> = lines.take_while(|h| !h.is_empty()).filter_map(field).collect();
    let header = |name: &str| headers.get(name).map_or("", String::as_str);
    let length = header("content-length").parse().unwrap_or(0u64);
    let mut raw = vec![];
    (&mut reader).take(length.min(BODY_LIMIT)).read_to_end(&mut raw)?;
    // read whole even when refused: a socket closed with a request unread resets, and the refusal can be lost
    std::io::copy(&mut (&mut reader).take(length.saturating_sub(BODY_LIMIT)), &mut std::io::sink())?;
    let port = app.addr.rsplit(':').next().unwrap_or("");
    if ![app.addr.as_str(), &format!("localhost:{port}")].contains(&header("host")) || method == "POST" && !headers.contains_key("x-connectome") {
        return fail(&mut stream, "403 Forbidden", "this server answers only its own page");
    }
    if length > BODY_LIMIT {
        return fail(&mut stream, "413 Content Too Large", &format!("a request's body takes {} MiB at most", BODY_LIMIT >> 20));
    }
    let body = match json::parse(&raw) {
        Ok(body) => body,
        Err(_) if raw.is_empty() => Value::Null,
        Err(e) => return fail(&mut stream, "400 Bad Request", &format!("the request's body: {e}")),
    };
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let param = |name: &str| query.split('&').find_map(|kv| kv.strip_prefix(name)?.strip_prefix('=')?.parse::<u64>().ok());
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let reply = match (method.as_str(), parts.as_slice()) {
        ("GET", []) => return send(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        ("GET", ["libs.js"]) => return send(&mut stream, "200 OK", "text/javascript; charset=utf-8", LIBS.as_bytes()),
        ("GET", ["api", "sessions"]) => Ok(Some(sessions(app))),
        ("GET", ["api", "settings"]) => global(app).map(|all| Some(providers(&all))),
        ("POST", ["api", "settings"]) => ask(app, |reply| Routed::Provide(body.clone(), reply)).map(|all| Some(providers(&all))),
        ("POST", ["api", "sessions"]) => create(app, &body).map(Some),
        (_, ["api", "s", name, rest @ ..]) if named(name) && app.root.join(name).is_dir() => session(app, &method, name, rest, &body, param, &mut stream),
        (_, ["api", "s", name, ..]) => Err(format!("no session {name}")),
        _ => Err("not found".into()),
    };
    match reply {
        Ok(None) => Ok(()),
        Ok(Some(v)) => send(&mut stream, "200 OK", "application/json", json::to_string(&v).as_bytes()),
        Err(e) => fail(&mut stream, "400 Bad Request", &e),
    }
}

/// The largest request body: a message pasted whole.
const BODY_LIMIT: u64 = 32 << 20;

/// A route's answer: the JSON to answer with, or None once the route has answered, streaming.
type Route = Result<Option<Value>, String>;

/// A session's name: letters, digits, - and _.
fn named(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// An HTTP response's head; without a length, the body runs until the connection closes.
fn head(status: &str, kind: &str, length: Option<usize>) -> String {
    let length = length.map_or(String::new(), |n| format!("content-length: {n}\r\n"));
    format!("HTTP/1.1 {status}\r\ncontent-type: {kind}\r\n{length}cache-control: no-cache\r\nconnection: close\r\n\r\n")
}

fn send(stream: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> std::io::Result<()> {
    stream.write_all(head(status, kind, Some(body.len())).as_bytes())?;
    stream.write_all(body)
}

/// A refusal the page shows: {"error": why}.
fn fail(stream: &mut TcpStream, status: &str, why: &str) -> std::io::Result<()> {
    send(stream, status, "application/json", json::to_string(&obj! {"error" => why}).as_bytes())
}

/// A JSON array to the page as its items are read, never held whole; past the head, a failure can only
/// end it early.
fn send_all(stream: &mut TcpStream, items: impl Iterator<Item = Value>) -> Route {
    if stream.write_all(head("200 OK", "application/json", None).as_bytes()).is_ok() {
        let _ = serde_json::Serializer::with_formatter(BufWriter::new(stream), json::Js).collect_seq(items);
    }
    Ok(None)
}

/// A session's routes.
fn session(app: &App, method: &str, name: &str, rest: &[&str], body: &Value, param: impl Fn(&str) -> Option<u64>, stream: &mut TcpStream) -> Route {
    let s = Session::open(app.root.join(name));
    match (method, rest) {
        ("GET", []) => Ok(Some(state(app, name, &s))),
        ("GET", ["messages"]) => {
            let (from, messages) = listing(&s, "messages", 200, param);
            let class = shown_as(&s, from);
            send_all(stream, messages.map(|m| shown(&m, class(m.id()))))
        }
        ("GET", ["layer", n]) => send_all(stream, listing(&s, &format!("layer{n}"), u64::MAX, param).1),
        ("POST", ["fork"]) => {
            let at = body["at"].as_f64().ok_or("fork at which message?")? as u64;
            let fork = body["name"].as_str().filter(|f| named(f)).ok_or("a fork's name")?;
            if at >= s.count("messages") {
                return Err(format!("can't fork {name} at {at}: it has {} messages", s.count("messages")));
            }
            ask_session(app, name, |reply| Msg::Job(Job::Fork { at, name: fork.into(), reply })).map(|()| Some(state(app, fork, &Session::open(app.root.join(fork)))))
        }
        ("POST", ["settings"]) => {
            let merged = s.count("layer2") > 0;
            ask_session(app, name, |reply| Msg::Tune(body.clone(), merged, reply)).map(|()| Some(state(app, name, &s)))
        }
        ("POST", [said @ ("say" | "continue")]) => {
            let text = (*said == "say").then(|| body["text"].as_str().unwrap_or("").to_string());
            if text.as_ref().map_or_else(|| s.count("messages") == 0, |text| text.trim().is_empty()) {
                return Err("say something".into());
            }
            profile(app, name)?;
            show(stream, |page| tell_session(app, name, Msg::Job(Job::Say { text, page })))
        }
        ("GET", ["live"]) => show(stream, |page| tell_session(app, name, Msg::Follow(page))),
        ("POST", ["approve"]) => {
            let (id, answer) = (body["id"].as_str().unwrap_or("").into(), body["answer"].as_str().unwrap_or("deny").into());
            ask_session(app, name, |reply| Msg::Approve { id, answer, reply }).map(|()| Some(obj! {}))
        }
        _ => Err("not found".into()),
    }
}

/// Records `from` to `to` of a file (by default, its newest `newest`), as they're read, and `from`.
fn listing<'s>(s: &'s Session, file: &str, newest: u64, param: impl Fn(&str) -> Option<u64>) -> (u64, Box<dyn Iterator<Item = Value> + 's>) {
    let to = param("to").map_or(s.count(file), |to| to.min(s.count(file)));
    let from = param("from").unwrap_or(to.saturating_sub(newest));
    (from, if from < to { s.span(file, from, to - 1) } else { Box::new(std::iter::empty()) })
}

/// How the last compile's context shows each message from `from` on: in the raw tail, past the chunks
/// ("tail"), through a memory the context shows ("recalled"), or in a raw chunk ("raw"). Only the
/// memories that reach `from` are read, from the end.
fn shown_as(s: &Session, from: u64) -> impl Fn(u64) -> &'static str {
    let chunked = s.last("layer0").map_or(0, |c| c["last"].u64() + 1);
    let frontier = s.last("compiles").map_or(vec![], |c| c["frontier"].arr().iter().map(|n| (n[0].u64(), n[1].u64())).collect());
    let memories = frontier.into_iter().rev().filter(|&(level, _)| level > 0).map(|(level, p)| s.read(&format!("layer{level}"), p));
    let recalled: Vec<(u64, u64)> = memories.map(|m| (m["first"].u64(), m["last"].u64())).take_while(|&(_, last)| last >= from).collect();
    move |i| if i >= chunked { "tail" } else if recalled.iter().any(|&(first, last)| (first..=last).contains(&i)) { "recalled" } else { "raw" }
}

/// A session's thread: it does its jobs one at a time, in the order they come (a message or fork told
/// during a turn waits for it), the only one to write the session's log, and keeps the last request's
/// prompt size, which calibrates the next compile. A job that panics (a file edited into something the
/// library can't read) fails, and the next runs.
fn work(app: &App, name: &str, jobs: Receiver<Job>, tell: &dyn Fn(Msg)) {
    let mut usage = None;
    for job in jobs {
        let done = catch_unwind(AssertUnwindSafe(|| {
            let s = Session::open(app.root.join(name));
            match job {
                Job::Say { text, page } => {
                    tell(Msg::Start(page));
                    tell(Msg::Event(turn(app, name, &s, text.as_deref(), &mut usage, tell).map_or_else(|e| obj! {"error" => e}, |()| obj! {"ok" => true})));
                    if let Ok(setup) = setup(app, name) { form(tell, name, &s, &setup) } // the turn's memories
                }
                Job::Fork { at, name: fork, reply } => {
                    let path = app.root.join(&fork); // claimed here, when the fork is made
                    let made = std::fs::create_dir(&path).map_err(|e| format!("can't fork {name} as {fork}: {e}"));
                    let _ = reply.send(made.and_then(|()| {
                        Session::fork(&s, at, path);
                        settings::write(&profile_path(app, &fork), &profile(app, name)?)
                    }));
                }
            }
        }));
        if done.is_err() {
            tell(Msg::Event(obj! {"forming" => false}));
            tell(Msg::Event(obj! {"error" => "the session's thread failed; the server's output says why"}));
        }
        tell(Msg::Done);
    }
}

/// A chat turn: the message joins the log, then steps run until a reply calls no tools. Without a
/// message, the turn continues the log as it is: after an interruption the context ends with the
/// user's message or the tool results, and after a reply cut short with that reply, which the Messages
/// API continues (a prefill). The turn tells the hub (`tell`) what the pages following it see: the
/// log's length each time it grows ({"logged"}), the pieces of a reply's reasoning and text
/// ({"thinking"}, {"text"}), the question whether each call may run ({"approve"}: its id; the answers
/// come from the pages: {"answered": the id, "answer"}), and each call's result ({"result"}: the
/// tool_result block, brief). Memories form between steps; the session's thread tells the turn's end
/// ({"ok": true} once it has joined the log, or {"error"}), then forms the memories after it.
fn turn(app: &App, name: &str, s: &Session, text: Option<&str>, usage: &mut Option<f64>, tell: &dyn Fn(Msg)) -> Result<(), String> {
    if let Some(text) = text {
        append(s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &profile(app, name)?.config);
    }
    tell(Msg::Logged(s.count("messages")));
    for _ in 0..MAX_STEPS {
        if !step(app, name, s, usage, tell)? { return Ok(()) }
    }
    Err(format!("the turn stopped after {MAX_STEPS} steps"))
}

/// Steps a turn may take, a model request each, before it stops.
const MAX_STEPS: usize = 64;

/// One step: the context compiles and streams into a request, the reply streams to the pages and joins
/// the log, and the tools it calls run, their results joining the log; then memories form. Whether it
/// called any.
fn step(app: &App, name: &str, s: &Session, usage: &mut Option<f64>, tell: &dyn Fn(Msg)) -> Result<bool, String> {
    let setup = setup(app, name)?;
    let (cfg, reserve) = (&setup.settings.config, setup.settings.reserve);
    let (reported, last) = (usage.take(), current(s));
    // how much bigger the last request's context was than its compile thought, in a sample calibration
    // learns from: calibration takes steps to catch up, and the context compiles that much smaller meanwhile
    let lag = reported.map(|u| u / last.armed).filter(|r| (1.0..=1.8).contains(r)).unwrap_or(1.0);
    // what the context compiles to: the window less what every request carries and the reply, smaller by
    // the lag, so the request and its answer fit even at the library's grace over budget
    let w = (setup.settings.window - setup.overhead - reserve) / lag / (1.0 + cfg.over_budget_grace);
    let mut compiled = select(s, w, reported.map_or(last, |u| last.observe(u)), cfg);
    // a refusal escalates the session, so the memory the context waits for forms at once; then it compiles again, up to 3 times
    for _ in 0..3 {
        if compiled.is_ok() { break }
        form(tell, name, s, &setup);
        compiled = select(s, w, current(s), cfg);
    }
    let turns = compiled.map_err(|r| format!("{}: {} tokens over a budget of {}", r.stage, r.actual, r.budget))?;
    let head = sparse(obj! {"model" => setup.settings.model.as_str(), "max_tokens" => reserve, "system" => setup.settings.system.as_deref(), "tools" => setup.definitions.clone()});
    let reply = setup.llm.messages(head, &turns, &mut Reader::new(s, false), |kind, piece| tell(Msg::Event(obj! {kind => piece})))?;
    *usage = Some(reply["usage"]["inputTokens"].as_f64().unwrap_or(0.0) - setup.overhead);
    append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, cfg);
    tell(Msg::Logged(s.count("messages")));
    let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
    let results = calls.map(|call| approve(app, name, &setup, tell, call).map_or_else(|why| tools::result(call, Err(why.into())), |()| setup.tools.call(call)));
    let results: Vec<Value> = results.inspect(|result| tell(Msg::Event(obj! {"result" => brief(result)}))).collect();
    if results.is_empty() { return Ok(false) }
    append(s, &obj! {"participant" => "user", "content" => results}, cfg);
    tell(Msg::Logged(s.count("messages")));
    form(tell, name, s, &setup);
    Ok(true)
}

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// A denied call's result.
const DENIED: &str = "the user didn't allow this call";

/// Whether the user lets a call run. A tool that doesn't ask, or that the session always allows, runs
/// at once (and a call to no tool fails at once); otherwise the turn asks the hub, which asks the pages
/// and decides by a page's answer (Live). No answer within APPROVAL_WAIT is a denial.
fn approve(app: &App, name: &str, setup: &Setup, tell: &dyn Fn(Msg), call: &Value) -> Result<(), String> {
    let (id, tool) = (call["id"].as_str().unwrap_or(""), call["name"].as_str().unwrap_or(""));
    if !setup.tools.asking().contains(&tool) || profile(app, name)?.allow.contains(tool) {
        return Ok(());
    }
    let (answer, answered) = channel();
    tell(Msg::Ask { id: id.into(), tool: tool.into(), answer });
    answered.recv_timeout(APPROVAL_WAIT).unwrap_or_else(|_| Err(DENIED.into()))
}

/// Memories form (memory::drive), the session's thread told while they do.
fn form(tell: &dyn Fn(Msg), name: &str, s: &Session, setup: &Setup) {
    tell(Msg::Event(obj! {"forming" => true}));
    if let Some(error) = drive(s, &mut &*setup, &setup.settings.config) {
        eprintln!("{name}: forming a memory: {error}");
    }
    tell(Msg::Event(obj! {"forming" => false}));
}

/// What a session's step runs with, from the settings as they are now.
fn setup(app: &App, name: &str) -> Result<Setup, String> {
    let (all, s) = (global(app)?, profile(app, name)?);
    let p = all.providers.get(&s.provider).ok_or_else(|| format!("no provider {} in settings.json", s.provider))?;
    let llm = Llm { base: p.base.clone(), key: p.key.clone(), agent: app.agent.clone() };
    let bash = match &s.workspace {
        Some(dir) if app.bash => bash::tool(dir.clone(), app.limit, s.sandbox.then_some(all.sandbox)),
        None if app.bash => bash::unavailable("it has no workspace; the user can set one in the session's settings"),
        _ => Toolbox::default(),
    };
    let tools = bash + mcp::toolbox(&app.mcp);
    let definitions = tools.definitions();
    Ok(Setup { overhead: overhead(s.system.as_deref(), &definitions), settings: s, llm, tools, definitions })
}

/// The tokens a system prompt and tool definitions take.
fn overhead(system: Option<&str>, definitions: &Option<Value>) -> f64 {
    density_tokens(system.unwrap_or("")) + definitions.as_ref().map_or(0.0, |d| density_tokens(&json::to_string(d)))
}

fn global_path(app: &App) -> PathBuf { app.root.join("settings.json") }

fn global(app: &App) -> Result<Settings, String> { settings::read(&global_path(app)) }

fn profile_path(app: &App, name: &str) -> PathBuf { app.root.join(name).join("session_settings.json") }

fn profile(app: &App, name: &str) -> Result<SessionSettings, String> { settings::read(&profile_path(app, name)) }

/// A settings file as `f` changes it, written if `f` succeeds. The hub changes every settings file
/// (once whoever made a session wrote its first).
fn change<T: Serialize + DeserializeOwned>(path: &Path, f: impl FnOnce(&mut T) -> Result<(), String>) -> Result<T, String> {
    let mut settings = settings::read(path)?;
    f(&mut settings)?;
    settings::write(path, &settings).map(|()| settings)
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
    (p.base, p.key) = (given("base").unwrap_or(std::mem::take(&mut p.base)), given("key").unwrap_or(std::mem::take(&mut p.key)));
    if p.base.is_empty() { Err("a provider's base URL".into()) } else { Ok(()) }
}

/// A session as the page shows it, its settings, if it's new, the defaults with the workspace `v` gives.
fn create(app: &App, v: &Value) -> Result<Value, String> {
    let name = v["name"].as_str().filter(|name| named(name)).ok_or("a session's name is letters, digits, - and _")?;
    if !profile_path(app, name).exists() {
        let first = tuned(app, &global(app)?.defaults, &obj! {"workspace" => v["workspace"].clone()}, false)?;
        std::fs::create_dir_all(app.root.join(name)).map_err(|e| format!("{name}: {e}"))?;
        settings::write(&profile_path(app, name), &first)?;
    }
    Ok(state(app, name, &Session::open(app.root.join(name))))
}

/// A session's settings changed by what `v` gives (tuned).
fn tune(app: &App, name: &str, v: &Value, merged: bool) -> Result<(), String> {
    change(&profile_path(app, name), |old: &mut SessionSettings| tuned(app, old, v, merged).map(|new| *old = new)).map(drop)
}

/// Session settings as what `v` gives changes them: any of their fields, and any of config's. The
/// provider must be one settings.json has, a workspace given a directory that exists (absolute, a
/// leading ~/ the home directory, no trailing /), and merge_threshold as it was once memories have
/// `merged`.
fn tuned(app: &App, old: &SessionSettings, v: &Value, merged: bool) -> Result<SessionSettings, String> {
    let mut new = serde_json::to_value(old).expect("settings are JSON");
    for (k, x) in v.as_object().into_iter().flatten() {
        match (k.as_str(), x) {
            ("config", Value::Object(fields)) => new["config"].as_object_mut().expect("config").extend(fields.clone()),
            _ => new[k] = x.clone(),
        }
    }
    let mut new: SessionSettings = serde_json::from_value(new).map_err(|e| e.to_string())?;
    if let Some(path) = v["workspace"].as_str().map(str::trim) {
        let dir: PathBuf = std::path::absolute(settings::expand(path)).map_err(|e| format!("{path}: {e}"))?.components().collect();
        if !dir.is_dir() {
            return Err(format!("no directory {path}"));
        }
        new.workspace = Some(dir);
    }
    if !global(app)?.providers.contains_key(&new.provider) {
        return Err(format!("no provider {}", new.provider));
    }
    if merged && new.config.merge_threshold != old.config.merge_threshold {
        return Err("merge_threshold stays as it is once memories have merged".into());
    }
    Ok(new)
}

/// A message as the page shows it: how the context shows it, and its blocks, brief.
fn shown(m: &Value, class: &str) -> Value {
    obj! {"id" => m.id(), "participant" => m["participant"].clone(), "shown" => class, "content" => m["content"].arr().iter().map(brief).collect::<Vec<_>>()}
}

/// A block as the page gets it: text and tool calls as they are, reasoning without its signature, a
/// tool result with its content's blocks brief, an image as its size, and anything else as its type.
fn brief(b: &Value) -> Value {
    match b.kind() {
        "text" | "tool_use" => b.clone(),
        "thinking" => obj! {"type" => "thinking", "thinking" => b["thinking"].clone()},
        "tool_result" => {
            let content = b["content"].as_array().map_or_else(|| b["content"].clone(), |blocks| blocks.iter().map(brief).collect());
            sparse(obj! {"type" => "tool_result", "toolUseId" => b["toolUseId"].clone(), "content" => content, "isError" => b["isError"].clone()})
        }
        "image" => obj! {"type" => "image", "kb" => b["source"]["data"].as_str().map_or(0, str::len) / 1024},
        kind => obj! {"type" => kind},
    }
}

/// Every session: its name, messages, the session it forked from, and its workspace.
fn sessions(app: &App) -> Value {
    let names: BTreeSet<String> = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect();
    Value::Array(names.iter().map(|name| {
        let s = Session::open(app.root.join(name));
        let workspace = profile(app, name).ok().and_then(|p| p.workspace);
        obj! {"name" => name.as_str(), "messages" => s.count("messages"), "parent" => parent(&s), "workspace" => workspace.map(|w| w.to_string_lossy().into_owned())}
    }).collect())
}

/// The session a fork was made from, and at which message.
fn parent(s: &Session) -> Value {
    let name = |path: &str| Path::new(path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    s.fork["parent"].as_str().map_or(Value::Null, |path| obj! {"name" => name(path), "at" => s.fork["at"].clone()})
}

/// What the page shows of a session: counts, each layer's size, the last compile, its settings as its
/// file has them, the tools that ask before they run and the tokens every request carries (or the
/// problem that keeps it from running), and whether a job runs (a turn, then its memories forming).
fn state(app: &App, name: &str, s: &Session) -> Value {
    let layers: Vec<Value> = (0..).map(|n| s.count(&format!("layer{n}"))).take_while(|&c| c > 0).map(Value::from).collect();
    let last = s.last("compiles").unwrap_or(obj! {});
    let refused = s.last("attempts").is_some_and(|a| a["outcome"].as_str() == Some("refusal"));
    let (setup, running) = (setup(app, name), ask_session(app, name, Msg::Status));
    obj! {
        "messages" => s.count("messages"), "layers" => layers,
        "frontier" => last["frontier"].clone(), "armed" => last["armed"].clone(), "multiplier" => last["multiplier"].clone(), "refused" => refused,
        "running" => running,
        "settings" => profile(app, name).ok().map(|p| serde_json::to_value(p).expect("settings are JSON")),
        "tools" => setup.as_ref().map_or(vec![], |x| x.tools.asking()), "overhead" => setup.as_ref().map_or(0.0, |x| x.overhead),
        "problem" => setup.as_ref().err().map(String::as_str),
    }
}
