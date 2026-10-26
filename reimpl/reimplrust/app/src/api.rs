//! The server: the page, its libraries and its API, one HTTP/1.1 request per connection, each on a
//! thread of its own. Only this server's own page gets an answer: a request must name this server as
//! its Host (another site's page reaching it through a DNS name it controls doesn't), and a POST must
//! carry an x-connectome header (another site's page can't send one without a CORS preflight, which
//! this server never approves), since a POST can change where the key goes.

use crate::{boards, channels::{self, Post}, claude::{self, Limits}, companion, costs::{self, Compile, Fold, Spent}, gateway, hub::{Event, Job, Msg, Queued, Routed}, settings::{self, SessionSettings}, work::Setup, App};
use connectome::{config::Config, json::{self, sparse, Json, Value}, memory::Recorded, obj, planner::chunks, store::Session};
use serde::{Serialize, Serializer};
use std::{collections::{BTreeSet, HashMap}, net::TcpStream, path::Path, sync::mpsc::{channel, RecvTimeoutError}, time::Duration};
use std::{any::Any, panic::{catch_unwind, AssertUnwindSafe}};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

const PAGE: &str = include_str!("ui.html");

/// The page's libraries, as published (vendor/): Preact, which draws the page from what it shows, and
/// htm, its templates; marked, which turns the assistant's markdown into HTML, DOMPurify, which cleans
/// that HTML, KaTeX, which sets its math, jsdiff, which finds the lines an edit changes, and
/// highlight.js, which colors code; with KaTeX's stylesheet and the fonts it names (fonts/NAME.woff2).
const LIBS: &str = concat!(
    include_str!("../vendor/preact/preact.min.umd.js"), "\n;", include_str!("../vendor/htm/htm.umd.js"), "\n;",
    include_str!("../vendor/marked.umd.js"), "\n;", include_str!("../vendor/purify.min.js"), "\n;", include_str!("../vendor/katex/katex.min.js"),
    "\n;", include_str!("../vendor/diff/diff.min.js"), "\n;", include_str!("../vendor/highlight/highlight.min.js"),
);
const KATEX_CSS: &str = include_str!("../vendor/katex/katex.min.css");
static FONTS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/vendor/katex/fonts");

/// The largest request body: a message pasted whole.
const BODY_LIMIT: u64 = 32 << 20;

/// How long a page may keep a write waiting before it's left behind.
const PAGE_WAIT: Duration = Duration::from_secs(30);

/// How long a page following a session goes without a line before it gets a blank one, which keeps
/// the connection open and finds a page that has gone.
const HEARTBEAT: Duration = Duration::from_secs(5);

/// A route's answer: the JSON to answer with, or None once the route has answered, streaming.
type Route = Result<Option<Value>, String>;

/// A request's numeric query parameters.
type Query<'a> = HashMap<&'a str, u64>;

pub fn handle(app: &App, mut stream: TcpStream) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut lines = (&mut reader).lines().map_while(Result::ok);
    let line = lines.next().unwrap_or_default();
    let mut words = line.split_whitespace();
    let (method, target) = (words.next().unwrap_or(""), words.next().unwrap_or("/"));
    let field = |h: String| h.split_once(':').map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()));
    let headers: HashMap<String, String> = lines.take_while(|h| !h.is_empty()).filter_map(field).collect();
    let header = |name: &str| headers.get(name).map_or("", String::as_str);
    let length = header("content-length").parse().unwrap_or(0u64);
    let mut raw = vec![];
    (&mut reader).take(length.min(BODY_LIMIT)).read_to_end(&mut raw)?;
    // read whole even when refused: a socket closed with a request unread resets, and the refusal can be lost
    io::copy(&mut (&mut reader).take(length.saturating_sub(BODY_LIMIT)), &mut io::sink())?;
    let port = app.addr.rsplit(':').next().unwrap_or("");
    if ![app.addr.as_str(), &format!("localhost:{port}")].contains(&header("host")) || method == "POST" && !headers.contains_key("x-connectome") {
        return fail(&mut stream, "403 Forbidden", "this server answers only its own page");
    }
    if length > BODY_LIMIT {
        return fail(&mut stream, "413 Content Too Large", &format!("a request's body takes {} MiB at most", BODY_LIMIT >> 20));
    }
    if let Some(forwarded) = target.strip_prefix("/gateway/") { // its body goes on as it came, unparsed
        return gateway::forward(app, method, &format!("/{forwarded}"), &headers, &raw, &mut stream);
    }
    let body = match json::parse(&raw) {
        Ok(body) => body,
        Err(_) if raw.is_empty() => Value::Null,
        Err(e) => return fail(&mut stream, "400 Bad Request", &format!("the request's body: {e}")),
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let query: Query = query.split('&').filter_map(|kv| kv.split_once('=')).filter_map(|(k, v)| Some((k, v.parse().ok()?))).collect();
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let reply = match (method, parts.as_slice()) {
        ("GET", []) => return send(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        ("GET", ["libs.js"]) => return send(&mut stream, "200 OK", "text/javascript; charset=utf-8", LIBS.as_bytes()),
        ("GET", ["katex.css"]) => return send(&mut stream, "200 OK", "text/css; charset=utf-8", KATEX_CSS.as_bytes()),
        ("GET", ["fonts", font]) => match FONTS.get_file(font) {
            Some(file) => return send(&mut stream, "200 OK", "font/woff2", file.contents()),
            None => Err("not found".into()),
        },
        ("GET", ["api", "sessions"]) => Ok(Some(sessions(app))),
        ("GET", ["api", "settings"]) => app.settings().map(|all| Some(all.providers())),
        ("POST", ["api", "settings"]) => app.ask(|reply| Routed::Provide(body.clone(), reply)).map(|all| Some(all.providers())),
        ("POST", ["api", "sessions"]) => create(app, &body).map(Some),
        ("GET", ["api", "channels"]) => Ok(Some(channels::listing(app))),
        ("POST", ["api", "channels"]) => founded(app, &body).map(Some),
        ("GET", ["api", "boards"]) => Ok(Some(boards::listing(app))),
        ("POST", ["api", "boards"]) => set_up(app, &body).map(Some),
        (_, ["api", "s", name, rest @ ..]) if named(name) && app.root.join(name).is_dir() => session(app, method, name, rest, &body, &query, &mut stream),
        (_, ["api", "s", name, ..]) => Err(format!("no session {name}")),
        (_, ["api", "c", name, rest @ ..]) if named(name) && app.channel_exists(name) => in_channel(app, method, name, rest, &body, &query, &mut stream),
        (_, ["api", "c", name, ..]) => Err(format!("no channel {name}")),
        (_, ["api", "b", name, rest @ ..]) if named(name) && app.board_exists(name) => on_board(app, method, name, rest, &body),
        (_, ["api", "b", name, ..]) => Err(format!("no board {name}")),
        _ => Err("not found".into()),
    };
    match reply {
        Ok(None) => Ok(()),
        Ok(Some(v)) => send(&mut stream, "200 OK", "application/json", json::to_string(&v).as_bytes()),
        Err(e) => fail(&mut stream, "400 Bad Request", &e),
    }
}

/// A session's routes.
fn session(app: &App, method: &str, name: &str, rest: &[&str], body: &Value, query: &Query, stream: &mut TcpStream) -> Route {
    let s = app.session(name);
    match (method, rest) {
        ("GET", []) => Ok(Some(state(app, name, &s))),
        ("GET", ["messages"]) => {
            let (from, messages) = listing(&s, "messages", 200, query);
            let folds = costs::folds(&s, from);
            send_all(stream, messages.map(|m| Shown::of(m, &folds)))
        }
        ("GET", ["layer", n]) => send_all(stream, listing(&s, &format!("layer{n}"), u64::MAX, query).1),
        ("POST", ["fork"]) => {
            if claude::command(app, name)?.is_some() { // its fork would resume the same conversation, carrying on both
                return Err(format!("{name} is a Claude Code session, which can't fork yet"));
            }
            let at = body["at"].as_f64().ok_or("fork at which message?")? as u64;
            let fork = body["name"].as_str().filter(|f| named(f)).ok_or("a fork's name")?;
            if at >= s.count("messages") {
                return Err(format!("can't fork {name} at {at}: it has {} messages", s.count("messages")));
            }
            let (reply, made) = channel();
            app.ask_session(name, |job| Msg::Job(Job::Fork { at, name: fork.into(), reply }, job));
            made.recv().expect("the session's thread answers").map(|()| Some(state(app, fork, &app.session(fork))))
        }
        ("POST", ["settings"]) => {
            let merged = s.count("layer2") > 0;
            app.ask_session(name, |reply| Msg::Tune(body.clone(), merged, reply)).map(|()| Some(state(app, name, &s)))
        }
        ("POST", [said @ ("say" | "continue")]) => {
            let text = (*said == "say").then(|| body["text"].as_str().unwrap_or("").to_string());
            if text.as_ref().map_or_else(|| s.count("messages") == 0, |text| text.trim().is_empty()) {
                return Err("say something".into());
            }
            app.profile(name)?;
            Ok(Some(obj! {"job" => app.ask_session(name, |job| Msg::Job(Job::Say { text }, job))}))
        }
        ("GET", ["stream"]) => follow(app, name, query, stream),
        ("POST", ["stop"]) => app.ask_session(name, Msg::Stop).map(|()| Some(obj! {})),
        ("POST", ["companion"]) => {
            if claude::command(app, name)?.is_some() {
                return Err(format!("{name} is a Claude Code session, which has no message tool to talk with a companion yet"));
            }
            let companion = body["name"].as_str().filter(|c| named(c)).ok_or("a companion's name is letters, digits, - and _")?;
            let goal = body["goal"].as_str().map(str::trim).filter(|g| !g.is_empty()).ok_or("a companion needs the goal")?;
            companion::make(app, name, companion, goal).map(|()| Some(state(app, name, &s)))
        }
        ("POST", ["auto"]) => {
            if app.ask(|reply| Routed::Auto(name.into(), body["on"] == true, reply))? {
                companion::kick_off(app, name)?;
            }
            Ok(Some(state(app, name, &s)))
        }
        ("POST", ["queued"]) => {   // {id, text}: a queued message changed; {id, cancel: true}: taken back
            let id = body["id"].as_f64().ok_or("which queued message?")? as u64;
            let text = match body["cancel"] == true {
                true => None,
                false => Some(body["text"].as_str().map(str::trim).filter(|t| !t.is_empty()).ok_or("say something")?.to_string()),
            };
            app.ask_session(name, |reply| Msg::Queued { id, text, reply }).map(|()| Some(state(app, name, &s)))
        }
        ("POST", ["approve"]) => {
            let (id, answer) = (body["id"].as_str().unwrap_or("").into(), body["answer"].as_str().unwrap_or("deny").into());
            app.ask_session(name, |reply| Msg::Approve { id, answer, reply }).map(|()| Some(obj! {}))
        }
        ("POST", ["board"]) => {   // {board}, or {board, leave: true}
            let (board, on) = (body["board"].as_str().ok_or("which board?")?.to_string(), body["leave"] != true);
            app.ask_session(name, |reply| Msg::OnBoard { board, on, reply }).map(|()| Some(state(app, name, &s)))
        }
        ("POST", ["channel"]) => {   // {channel, inputs?, per_day?}, or {channel, leave: true}
            let channel = body["channel"].as_str().ok_or("which channel?")?.to_string();
            let change = if body["leave"] == true { Value::Null } else { obj! {"inputs" => body["inputs"].as_bool(), "per_day" => body["per_day"].as_f64()} };
            app.ask_session(name, |reply| Msg::Member { channel, change, reply }).map(|()| Some(state(app, name, &s)))
        }
        _ => Err("not found".into()),
    }
}

/// A channel's routes: its posts, a page following it, a post of the user's, and its description set
/// (answered with the channels).
fn in_channel(app: &App, method: &str, name: &str, rest: &[&str], body: &Value, query: &Query, stream: &mut TcpStream) -> Route {
    match (method, rest) {
        ("GET", ["posts"]) => send_all(stream, listing(&app.channel(name), "posts", 200, query).1),
        ("GET", ["stream"]) => watch(app, name, query, stream),
        ("POST", ["post"]) => {
            let text = body["text"].as_str().map(str::trim).filter(|t| !t.is_empty()).ok_or("say something")?.to_string();
            app.ask(|reply| Routed::Post(Post { channel: name.into(), from: None, text }, reply)).map(|posted| Some(posted.post))
        }
        ("POST", ["describe"]) => {
            let description = body["description"].as_str().unwrap_or("").to_string();
            app.ask(|reply| Routed::Describe(name.into(), description, reply)).map(|()| Some(channels::listing(app)))
        }
        _ => Err("not found".into()),
    }
}

/// A board's routes: the board, a task, and a change to either (boards::apply) or the board's
/// description, answered with the board as it is after it.
fn on_board(app: &App, method: &str, name: &str, rest: &[&str], body: &Value) -> Route {
    let change = |change: Value| app.ask(|reply| Routed::Board(name.into(), change, reply)).map(|()| Some(boards::view(app, name)));
    match (method, rest) {
        ("GET", []) => Ok(Some(boards::view(app, name))),
        ("GET", ["t", file]) => boards::task(app, name, &unescaped(file)).map(Some),
        ("POST", ["change"]) => change(body.clone()),
        ("POST", ["describe"]) => change(obj! {"op" => "describe", "description" => body["description"].clone()}),
        _ => Err("not found".into()),
    }
}

/// A URL's path segment, its %XX escapes undone (a task's file a session named may have spaces).
fn unescaped(segment: &str) -> String {
    let (bytes, mut out, mut i) = (segment.as_bytes(), vec![], 0);
    while i < bytes.len() {
        let hex = bytes.get(i + 1..i + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok());
        match (bytes[i], hex) {
            (b'%', Some(b)) => { out.push(b); i += 3 }
            (b, _) => { out.push(b); i += 1 }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The boards, once the one `v` names is made.
fn set_up(app: &App, v: &Value) -> Result<Value, String> {
    let name = v["name"].as_str().filter(|name| named(name)).ok_or("a board's name is letters, digits, - and _")?;
    boards::make(app, name).map(|()| boards::listing(app))
}

/// The channels, once the one `v` names is made.
fn founded(app: &App, v: &Value) -> Result<Value, String> {
    let name = v["name"].as_str().filter(|name| named(name)).ok_or("a channel's name is letters, digits, - and _")?;
    channels::make(app, name).map(|()| channels::listing(app))
}

/// A session as the page shows it, its settings, if it's new, the defaults with the workspace `v` gives.
fn create(app: &App, v: &Value) -> Result<Value, String> {
    let name = v["name"].as_str().filter(|name| named(name)).ok_or("a session's name is letters, digits, - and _")?;
    if !app.profile_path(name).exists() {
        let all = app.settings()?;
        let first = all.defaults.tuned(&obj! {"workspace" => v["workspace"].clone()}, false, &all)?;
        std::fs::create_dir_all(app.root.join(name)).map_err(|e| format!("{name}: {e}"))?;
        settings::write(&app.profile_path(name), &first)?;
    }
    Ok(state(app, name, &app.session(name)))
}

/// What the page shows of a session: counts, each layer's size, the last compile's frontier and
/// calibration, its settings as its file has them, the tools that ask before they run and the tokens
/// every request carries (or the problem that keeps it from running), whether a turn runs (with its
/// memories forming after it), the turns the user asked for that wait for it, what its requests have used, and the last reply's prompt as its
/// provider counted it.
#[derive(Serialize)]
struct State<'a> {
    messages: u64,
    layers: Vec<u64>,
    #[serde(flatten)] last: Option<Compile>,
    refused: bool,
    running: bool,
    queued: Vec<Queued>,
    spent: Option<Spent>,
    prompt: Option<f64>,
    limits: Option<Limits>,
    settings: Option<SessionSettings>,
    tools: Vec<String>,
    overhead: f64,
    problem: Option<&'a str>,
    covers: Vec<Vec<(u64, u64)>>,
}

/// A session's State, as JSON.
fn state(app: &App, name: &str, s: &Session) -> Value {
    let (setup, (running, queued)) = (Setup::new(app, name), app.ask_session(name, Msg::Status));
    let state = State {
        messages: s.count("messages"),
        layers: (0..).map(|n| s.count(&format!("layer{n}"))).take_while(|&c| c > 0).collect(),
        last: s.last("compiles").and_then(|c| serde_json::from_value(c).ok()),
        refused: s.last("attempts").is_some_and(|a| a["outcome"] == "refusal"),
        running,
        queued,
        spent: costs::spent(s),
        prompt: costs::prompted(s),
        limits: costs::limits(s),
        settings: app.profile(name).ok(),
        // Claude Code's tools are its own, which it asks about as it calls them: those the session always allows
        tools: match (claude::command(app, name), app.profile(name)) {
            (Ok(Some(_)), Ok(own)) => own.allow.into_iter().collect(),
            _ => setup.as_ref().map_or(vec![], |x| x.tools.asking().into_iter().map(String::from).collect()),
        },
        overhead: setup.as_ref().map_or(0.0, |x| x.overhead),
        problem: setup.as_ref().err().map(String::as_str),
        covers: app.profile(name).map_or(vec![], |own| covers(s, &own.config)),
    };
    serde_json::to_value(state).expect("a state is JSON")
}

/// The chunks each memory above L1 covers, by layer from 2: its first, and the one after its last.
fn covers(s: &Session, cfg: &Config) -> Vec<Vec<(u64, u64)>> {
    let groups = Recorded::new(s, cfg);
    let layer = |n: u64| (0..s.count(&format!("layer{n}"))).map(|p| chunks((n, p), &groups)).map(|c| (c.start, c.end)).collect::<Vec<_>>();
    (2..).map(layer).take_while(|covered| !covered.is_empty()).collect()
}

/// Every session: its name, messages, the session it forked from, its workspace, and its companion or
/// the session it accompanies.
fn sessions(app: &App) -> Value {
    let names: BTreeSet<String> = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).filter(|n| named(n)).collect();
    Value::Array(names.iter().map(|name| {
        let (s, own) = (app.session(name), app.profile(name).ok());
        let workspace = own.as_ref().and_then(|p| p.workspace.as_ref()).map(|w| w.to_string_lossy().into_owned());
        let (companion, accompanies) = own.map_or((None, None), |p| (p.companion, p.accompanies));
        obj! {"name" => name.as_str(), "messages" => s.count("messages"), "parent" => parent(&s), "workspace" => workspace, "companion" => companion, "accompanies" => accompanies}
    }).collect())
}

/// The session a fork was made from, and at which message.
fn parent(s: &Session) -> Value {
    let name = |path: &str| Path::new(path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    s.fork["parent"].as_str().map_or(Value::Null, |path| obj! {"name" => name(path), "at" => s.fork["at"].clone()})
}

/// A file's records from the query's `from` to its `to` (by default, its newest `newest`), as they're
/// read, and `from`.
fn listing<'s>(s: &'s Session, file: &str, newest: u64, query: &Query) -> (u64, Box<dyn Iterator<Item = Value> + 's>) {
    let to = query.get("to").map_or(s.count(file), |&to| to.min(s.count(file)));
    let from = query.get("from").copied().unwrap_or(to.saturating_sub(newest));
    (from, if from < to { s.span(file, from, to - 1) } else { Box::new(std::iter::empty()) })
}

/// A message as the page shows it: its blocks brief, and how the context folded before it (costs::folds).
#[derive(Clone, Serialize)]
pub struct Shown {
    id: u64,
    participant: Value,
    content: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")] fold: Option<Fold>,
}

impl Shown {
    /// Message `m`, its content moved, not copied.
    fn of(mut m: Value, folds: &HashMap<u64, Fold>) -> Shown {
        let content = m["content"].as_array_mut().map_or(vec![], |blocks| blocks.drain(..).map(brief).collect());
        Shown { id: m.id(), participant: m["participant"].take(), content, fold: folds.get(&m.id()).cloned() }
    }
}

/// A block as the page gets it, moved: text and tool calls as they are, reasoning without its
/// signature, a tool result with its content's blocks brief, an image as its size, and anything else
/// as its type.
pub fn brief(mut b: Value) -> Value {
    match b.kind().to_string().as_str() {
        "text" | "tool_use" => b,
        "thinking" => obj! {"type" => "thinking", "thinking" => b["thinking"].take()},
        "tool_result" => {
            let content = match b["content"].take() {
                Value::Array(blocks) => blocks.into_iter().map(brief).collect(),
                content => content,
            };
            sparse(obj! {"type" => "tool_result", "toolUseId" => b["toolUseId"].take(), "content" => content, "isError" => b["isError"].take()})
        }
        "image" => obj! {"type" => "image", "kb" => b["source"]["data"].as_str().map_or(0, str::len) / 1024},
        kind => obj! {"type" => kind},
    }
}

/// A session's name: letters, digits, - and _.
pub fn named(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// An HTTP response's head; without a length, the body runs until the connection closes.
fn head(status: &str, kind: &str, length: Option<usize>) -> String {
    let length = length.map_or(String::new(), |n| format!("content-length: {n}\r\n"));
    format!("HTTP/1.1 {status}\r\ncontent-type: {kind}\r\n{length}cache-control: no-cache\r\nconnection: close\r\n\r\n")
}

pub fn send(stream: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> io::Result<()> {
    stream.write_all(head(status, kind, Some(body.len())).as_bytes())?;
    stream.write_all(body)
}

/// A refusal the page shows: {"error": why}.
fn fail(stream: &mut TcpStream, status: &str, why: &str) -> io::Result<()> {
    send(stream, status, "application/json", json::to_string(&obj! {"error" => why}).as_bytes())
}

/// A JSON array to the page as its items are read, never held whole; past the head, a failure can only
/// end it early.
fn send_all(stream: &mut TcpStream, items: impl Iterator<Item = impl Serialize>) -> Route {
    if stream.write_all(head("200 OK", "application/json", None).as_bytes()).is_ok() {
        let _ = serde_json::Serializer::with_formatter(BufWriter::new(stream), json::Js).collect_seq(items);
    }
    Ok(None)
}

/// A session as it goes on, to a page that follows it until it goes, a JSON line each (hub's Event):
/// the log's messages from the query's `from` (by default its newest 200), the session, then what the
/// hub tells the page: as the log grows, the messages it gains, read from disk, and after each job the
/// session again. The page follows the session before its messages are read, so it misses none: one
/// the log gains meanwhile it reads, or hears of, or both, and a page takes a message once and a turn's
/// events only for messages it doesn't have. Files the library can't read (a hand edit) the page hears
/// of ({"broken"}), and they're read again at the next event, as they may have been mended.
fn follow(app: &App, name: &str, query: &Query, page: &mut TcpStream) -> Route {
    let (sender, events) = channel();
    app.ask_session(name, |followed| Msg::Follow(sender, followed));
    let s = app.session(name);
    let _ = (page.set_write_timeout(Some(PAGE_WAIT)), page.set_nodelay(true)); // each batch of events out at once
    let mut out = BufWriter::new(page);
    let mut sent = None; // the messages the page has
    let mut catch_up = |out: &mut BufWriter<&mut TcpStream>, with_state: bool| {
        let read = catch_unwind(AssertUnwindSafe(|| -> io::Result<()> {
            let to = s.count("messages");
            messages(&s, sent.unwrap_or_else(|| query.get("from").copied().unwrap_or(to.saturating_sub(200))), to, &mut *out)?;
            sent = Some(to);
            if with_state { line(&Event::State(state(app, name, &s)), &mut *out)? }
            Ok(())
        }));
        read.unwrap_or_else(|why| line(&Event::Broken(format!("the session's files can't be read as they are: {}", said(why))), out))
    };
    let _ = (|| -> io::Result<()> {
        out.write_all(head("200 OK", "application/x-ndjson", None).as_bytes())?;
        catch_up(&mut out, true)?;
        loop {
            out.flush()?;
            let event = match events.recv_timeout(HEARTBEAT) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => { out.write_all(b"\n")?; continue }
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            };
            for event in std::iter::once(event).chain(events.try_iter()) {
                if let Event::Logged(_) = event { catch_up(&mut out, false)? }
                line(&event, &mut out)?;
                if let Event::Done(_) = event { catch_up(&mut out, true)? }
            }
        }
    })();
    Ok(None)
}

/// A channel as it goes on, to a page that follows it until it goes, a JSON line each: its posts from
/// the query's `from` (by default its newest 200), then each as it's logged ({"post"}). The page follows
/// the channel before its posts are read, so it misses none, and it takes each once.
fn watch(app: &App, name: &str, query: &Query, page: &mut TcpStream) -> Route {
    let (sender, posts) = channel();
    app.ask(|followed| Routed::Watch(name.into(), sender, followed));
    let log = app.channel(name);
    let _ = (page.set_write_timeout(Some(PAGE_WAIT)), page.set_nodelay(true));
    let mut out = BufWriter::new(page);
    let mut next = 0; // the first post the page doesn't have
    let mut send = |post: Value, out: &mut BufWriter<&mut TcpStream>| -> io::Result<()> {
        if post.id() < next { return Ok(()) }
        next = post.id() + 1;
        json::write(&obj! {"post" => post}, &mut *out).and_then(|()| out.write_all(b"\n"))
    };
    let _ = (|| -> io::Result<()> {
        out.write_all(head("200 OK", "application/x-ndjson", None).as_bytes())?;
        listing(&log, "posts", 200, query).1.try_for_each(|post| send(post, &mut out))?;
        loop {
            out.flush()?;
            match posts.recv_timeout(HEARTBEAT) {
                Ok(post) => send(post, &mut out)?,
                Err(RecvTimeoutError::Timeout) => out.write_all(b"\n")?,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    })();
    Ok(None)
}

/// What a panic said.
fn said(panic: Box<dyn Any + Send>) -> String {
    panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default()
}

/// The log's messages `from` up to `to`, each a line.
fn messages(s: &Session, from: u64, to: u64, out: &mut impl Write) -> io::Result<()> {
    if from >= to { return Ok(()) }
    let folds = costs::folds(s, from);
    s.span("messages", from, to - 1).try_for_each(|m| line(&Event::Message(Shown::of(m, &folds)), &mut *out))
}

/// An event as a line of JSON.
fn line(event: &Event, out: &mut impl Write) -> io::Result<()> {
    json::write(event, &mut *out).and_then(|()| out.write_all(b"\n"))
}
