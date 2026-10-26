//! The gateway: Claude Code (or anything else that speaks the Messages API) sent here, with
//! ANTHROPIC_BASE_URL=http://ADDR/gateway, has each request forwarded to --upstream with its headers
//! but this connection's, so its own credentials (an API key, or a Claude login's token and the
//! anthropic-beta that goes with it) go with it; the answer streams back as it arrives. A request of
//! the conversation a Claude Code session's turn runs (x-connectome-session names the session; not a
//! subagent's, x-claude-code-agent-id, nor one of another class, x-claude-code-request-class) waits
//! while the session's thread compiles it (claude_fold.rs), and goes with its context folded; any
//! other goes as it came, its body byte for byte. Each exchange, credentials left out, joins
//! ROOT/.gateway/SESSION/exchanges.jsonl, SESSION being the request's x-claude-code-session-id, so
//! what a session's requests were can be read.

use crate::{api, claude, hub::{Job, Msg, Routed}, App};
use connectome::{json, obj, store::Session};
use serde::Serialize;
use std::{borrow::Cow, collections::{BTreeMap, HashMap}, io::{self, Read, Write}, net::TcpStream, sync::mpsc::channel, time::{SystemTime, UNIX_EPOCH}};

/// Headers that belong to a connection, not to what it carries (RFC 9110 §7.6.1): ureq and this server
/// set their own. x-connectome is this server's check, and accept-encoding goes so that the answer
/// comes uncompressed, as the recording needs it.
const CONNECTION_HEADERS: [&str; 11] = ["host", "connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "content-length", "accept-encoding", "x-connectome"];

/// Headers that carry credentials, which the recording leaves out.
const CREDENTIALS: [&str; 2] = ["authorization", "x-api-key"];

/// The size of each read of the answer: a server-sent event goes back as soon as it's read.
const PIECE: usize = 16 * 1024;

/// A request the gateway forwarded, and its answer as far as it got.
#[derive(Serialize)]
struct Exchange {
    at: u64, // when it came, in seconds since 1970
    method: String,
    path: String, // as forwarded, with its query
    headers: BTreeMap<String, String>,
    body: json::Value, // as forwarded: its JSON, or its text if it isn't JSON
    #[serde(skip_serializing_if = "std::ops::Not::not")] folded: bool, // whether its context was folded (claude_fold.rs)
    status: Option<u16>, // none if the upstream never answered
    answer_headers: BTreeMap<String, String>,
    answer: String, // as it came: server-sent events, for a stream
    failed: Option<String>, // why it ended short: the upstream unreached or cut off, or the client gone
}

impl App {
    /// The recording of what Claude Code session `id` sent through the gateway.
    pub fn gateway(&self, id: &str) -> Session { Session::open(self.root.join(".gateway").join(id)) }
}

/// The request `method path` with `headers` and `body` forwarded to --upstream, compiled if it's a
/// Claude Code session's (folding), its answer streamed back to `client`, and the exchange recorded (by
/// the hub, the recording's only writer).
pub fn forward(app: &App, method: &str, path: &str, headers: &HashMap<String, String>, body: &[u8], client: &mut TcpStream) -> io::Result<()> {
    let at = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let compiled = match folding(app, method, path, headers) {
        Some(name) => fold(app, name, headers, body),
        None => Ok(None),
    };
    let folded = matches!(compiled, Ok(Some(_)));
    let body: Cow<[u8]> = match &compiled { Ok(Some(compiled)) => Cow::Borrowed(compiled.as_slice()), _ => Cow::Borrowed(body) };
    let mut recorded = BTreeMap::new();
    let url = format!("{}{path}", app.upstream.trim_end_matches('/'));
    let mut request = ureq::http::Request::builder().method(method).uri(&url);
    for (name, value) in headers {
        if CONNECTION_HEADERS.contains(&name.as_str()) { continue }
        request = request.header(name, value);
        let shown = if CREDENTIALS.contains(&name.as_str()) { "(left out)" } else { value };
        recorded.insert(name.clone(), shown.to_string());
    }
    let body_shown = json::parse(&body).unwrap_or_else(|_| String::from_utf8_lossy(&body).into());
    let mut exchange = Exchange { at, method: method.into(), path: path.into(), headers: recorded, body: body_shown, folded, status: None, answer_headers: BTreeMap::new(), answer: String::new(), failed: None };
    let sent = match compiled {
        // its context can't be compiled (it doesn't fit, or the turn was stopped while its memories formed): the
        // API's own answer to a request it won't take, so Claude Code doesn't send it again
        Err(why) => {
            let _ = api::send(client, "400 Bad Request", "application/json", error_json("invalid_request_error", &why).as_bytes());
            exchange.failed = Some(why);
            None
        }
        Ok(_) => Some(request.body(&*body).map(|r| app.agent.run(r))), // a GET's empty body goes as content-length: 0
    };
    // the exchange is recorded whether or not the client is still there to hear the answer
    match sent {
        None => {}
        Some(Err(e)) => { // a header that isn't one
            let why = format!("the request can't be forwarded: {e}");
            let _ = api::send(client, "400 Bad Request", "application/json", error_json("api_error", &why).as_bytes());
            exchange.failed = Some(why);
        }
        Some(Ok(Err(e))) => {
            let why = format!("{url}: {e}");
            let _ = api::send(client, "502 Bad Gateway", "application/json", error_json("api_error", &why).as_bytes());
            exchange.failed = Some(why);
        }
        Some(Ok(Ok(response))) => {
            let status = response.status();
            exchange.status = Some(status.as_u16());
            // the answer's length is where the connection closes, as this server's answers' is
            let mut head = format!("HTTP/1.1 {} {}\r\n", status.as_u16(), status.canonical_reason().unwrap_or("")).into_bytes();
            for (name, value) in response.headers() {
                if CONNECTION_HEADERS.contains(&name.as_str()) { continue }
                head.extend_from_slice(format!("{name}: ").as_bytes());
                head.extend_from_slice(value.as_bytes());
                head.extend_from_slice(b"\r\n");
                exchange.answer_headers.insert(name.to_string(), String::from_utf8_lossy(value.as_bytes()).into());
            }
            head.extend_from_slice(b"connection: close\r\n\r\n");
            let _ = client.set_nodelay(true); // each event out as it's written
            let mut answer = vec![];
            let mut upstream = response.into_body().into_reader();
            let mut piece = vec![0; PIECE];
            if let Err(e) = client.write_all(&head) {
                exchange.failed = Some(format!("the client went away: {e}"));
            }
            while exchange.failed.is_none() {
                let n = match upstream.read(&mut piece) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => {
                        exchange.failed = Some(format!("the answer was cut off: {e}"));
                        break;
                    }
                };
                answer.extend_from_slice(&piece[..n]);
                if let Err(e) = client.write_all(&piece[..n]) {
                    exchange.failed = Some(format!("the client went away: {e}"));
                }
            }
            // one conversion at the end: a character split between two reads stays whole
            exchange.answer = String::from_utf8_lossy(&answer).into();
        }
    }
    let session = headers.get("x-claude-code-session-id").map(String::as_str).filter(|id| api::named(id)).unwrap_or("unnamed");
    let record = serde_json::to_value(&exchange).expect("an exchange is JSON");
    app.hub.send(Routed::Exchange(session.into(), record)).expect("the hub runs");
    Ok(())
}

/// A failure as the Messages API words its own, of its `kind` (api_error, which a client tries again, or
/// invalid_request_error, which it doesn't), so the client shows why.
fn error_json(kind: &str, why: &str) -> String {
    json::to_string(&obj! {"type" => "error", "error" => obj! {"type" => kind, "message" => format!("connectome's gateway: {why}")}})
}

/// The Claude Code session whose request this is, if it's one to fold: a POST to /v1/messages of the
/// session's conversation, not a subagent's nor of another class, from a session whose provider is
/// Claude Code (the session's thread then says whether it's from the Claude Code its turn runs).
fn folding<'h>(app: &App, method: &str, path: &str, headers: &'h HashMap<String, String>) -> Option<&'h str> {
    let name = headers.get("x-connectome-session").map(String::as_str).filter(|name| api::named(name))?;
    let main = headers.get("x-claude-code-request-class").is_none_or(|class| class == "main") && !headers.contains_key("x-claude-code-agent-id");
    let claude_code = || claude::command(app, name).is_ok_and(|command| command.is_some());
    (method == "POST" && path.split('?').next() == Some("/v1/messages") && main && claude_code()).then_some(name)
}

/// The request's `body` compiled by session `name`'s thread, which holds it while memories form; None to
/// send it as it came (not JSON, or not of the conversation its turn runs).
fn fold(app: &App, name: &str, headers: &HashMap<String, String>, body: &[u8]) -> Result<Option<Vec<u8>>, String> {
    let Ok(body) = json::parse(body) else { return Ok(None) };
    let claude_session = headers.get("x-claude-code-session-id").cloned().unwrap_or_default();
    let (answer, compiled) = channel();
    app.ask_session(name, |numbered| Msg::Job(Job::Fold { claude_session, body, answer }, numbered));
    compiled.recv().unwrap_or(Ok(None)) // the turn ended first
}
