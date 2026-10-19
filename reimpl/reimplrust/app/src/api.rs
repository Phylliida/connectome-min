//! The server: the page, its libraries and its API, one HTTP/1.1 request per connection, each on a
//! thread of its own. Only this server's own page gets an answer: a request must name this server as
//! its Host (another site's page reaching it through a DNS name it controls doesn't), and a POST must
//! carry an x-connectome header (another site's page can't send one without a CORS preflight, which
//! this server never approves), since a POST can change where the key goes.

use crate::{costs::{self, Compile, Fold, Spent}, hub::{Job, Msg, Routed}, settings::{self, SessionSettings}, work::Setup, App};
use connectome::{json::{self, sparse, Json, Value}, obj, store::Session};
use serde::{Serialize, Serializer};
use std::{collections::{BTreeSet, HashMap}, net::TcpStream, path::Path, sync::mpsc::{channel, Sender}, time::Duration};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

const PAGE: &str = include_str!("ui.html");

/// The page's libraries, as published (vendor/): marked, which turns the assistant's markdown into
/// HTML, DOMPurify, which cleans that HTML, KaTeX, which sets its math, jsdiff, which finds the lines
/// an edit changes, and highlight.js, which colors code; with KaTeX's stylesheet and the fonts it
/// names (fonts/NAME.woff2).
const LIBS: &str = concat!(
    include_str!("../vendor/marked.umd.js"), "\n;", include_str!("../vendor/purify.min.js"), "\n;", include_str!("../vendor/katex/katex.min.js"),
    "\n;", include_str!("../vendor/diff/diff.min.js"), "\n;", include_str!("../vendor/highlight/highlight.min.js"),
);
const KATEX_CSS: &str = include_str!("../vendor/katex/katex.min.css");
static FONTS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/vendor/katex/fonts");

/// The largest request body: a message pasted whole.
const BODY_LIMIT: u64 = 32 << 20;

/// How long a page may keep a write waiting before it's left behind.
const PAGE_WAIT: Duration = Duration::from_secs(30);

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
        (_, ["api", "s", name, rest @ ..]) if named(name) && app.root.join(name).is_dir() => session(app, method, name, rest, &body, &query, &mut stream),
        (_, ["api", "s", name, ..]) => Err(format!("no session {name}")),
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
            send_all(stream, messages.map(|mut m| Shown { id: m.id(), content: m["content"].arr().iter().map(brief).collect(), fold: folds.get(&m.id()), participant: m["participant"].take() }))
        }
        ("GET", ["layer", n]) => send_all(stream, listing(&s, &format!("layer{n}"), u64::MAX, query).1),
        ("POST", ["fork"]) => {
            let at = body["at"].as_f64().ok_or("fork at which message?")? as u64;
            let fork = body["name"].as_str().filter(|f| named(f)).ok_or("a fork's name")?;
            if at >= s.count("messages") {
                return Err(format!("can't fork {name} at {at}: it has {} messages", s.count("messages")));
            }
            app.ask_session(name, |reply| Msg::Job(Job::Fork { at, name: fork.into(), reply })).map(|()| Some(state(app, fork, &app.session(fork))))
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
            show(stream, |page| app.tell(name, Msg::Job(Job::Say { text, page })))
        }
        ("GET", ["live"]) => show(stream, |page| app.tell(name, Msg::Follow(page))),
        ("POST", ["approve"]) => {
            let (id, answer) = (body["id"].as_str().unwrap_or("").into(), body["answer"].as_str().unwrap_or("deny").into());
            app.ask_session(name, |reply| Msg::Approve { id, answer, reply }).map(|()| Some(obj! {}))
        }
        _ => Err("not found".into()),
    }
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
/// memories forming after it), what its requests have used, and the last reply's prompt as its
/// provider counted it.
#[derive(Serialize)]
struct State<'a> {
    messages: u64,
    layers: Vec<u64>,
    #[serde(flatten)] last: Option<Compile>,
    refused: bool,
    running: bool,
    spent: Option<Spent>,
    prompt: Option<f64>,
    settings: Option<SessionSettings>,
    tools: Vec<&'a str>,
    overhead: f64,
    problem: Option<&'a str>,
}

/// A session's State, as JSON.
fn state(app: &App, name: &str, s: &Session) -> Value {
    let setup = Setup::new(app, name);
    let state = State {
        messages: s.count("messages"),
        layers: (0..).map(|n| s.count(&format!("layer{n}"))).take_while(|&c| c > 0).collect(),
        last: s.last("compiles").and_then(|c| serde_json::from_value(c).ok()),
        refused: s.last("attempts").is_some_and(|a| a["outcome"] == "refusal"),
        running: app.ask_session(name, Msg::Status),
        spent: costs::spent(s),
        prompt: costs::prompted(s),
        settings: app.profile(name).ok(),
        tools: setup.as_ref().map_or(vec![], |x| x.tools.asking()),
        overhead: setup.as_ref().map_or(0.0, |x| x.overhead),
        problem: setup.as_ref().err().map(String::as_str),
    };
    serde_json::to_value(state).expect("a state is JSON")
}

/// Every session: its name, messages, the session it forked from, and its workspace.
fn sessions(app: &App) -> Value {
    let names: BTreeSet<String> = std::fs::read_dir(&app.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect();
    Value::Array(names.iter().map(|name| {
        let s = app.session(name);
        let workspace = app.profile(name).ok().and_then(|p| p.workspace);
        obj! {"name" => name.as_str(), "messages" => s.count("messages"), "parent" => parent(&s), "workspace" => workspace.map(|w| w.to_string_lossy().into_owned())}
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
#[derive(Serialize)]
struct Shown<'a> {
    id: u64,
    participant: Value,
    content: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")] fold: Option<&'a Fold>,
}

/// A block as the page gets it: text and tool calls as they are, reasoning without its signature, a
/// tool result with its content's blocks brief, an image as its size, and anything else as its type.
pub fn brief(b: &Value) -> Value {
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

/// A session's name: letters, digits, - and _.
fn named(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// An HTTP response's head; without a length, the body runs until the connection closes.
fn head(status: &str, kind: &str, length: Option<usize>) -> String {
    let length = length.map_or(String::new(), |n| format!("content-length: {n}\r\n"));
    format!("HTTP/1.1 {status}\r\ncontent-type: {kind}\r\n{length}cache-control: no-cache\r\nconnection: close\r\n\r\n")
}

fn send(stream: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> io::Result<()> {
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

/// A job to a page as it happens, a JSON line each (hub's Live), until the hub stops sending to the
/// channel `follow` gives it.
fn show(page: &mut TcpStream, follow: impl FnOnce(Sender<String>)) -> Route {
    let (sender, lines) = channel();
    follow(sender);
    let _ = page.set_write_timeout(Some(PAGE_WAIT));
    let _ = std::iter::once(head("200 OK", "application/x-ndjson", None)).chain(lines).try_for_each(|line| page.write_all(line.as_bytes()));
    Ok(None)
}
