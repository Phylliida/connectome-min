//! The gateway, for now a recorder: Claude Code (or anything else that speaks the Messages API) sent
//! here, with ANTHROPIC_BASE_URL=http://ADDR/gateway, has each request forwarded to --upstream as it
//! came, its body byte for byte and its headers but this connection's, so its own credentials (an API
//! key, or a Claude login's token and the anthropic-beta that goes with it) go with it; the answer
//! streams back as it arrives. Each exchange, credentials left out, joins
//! ROOT/.gateway/SESSION/exchanges.jsonl, SESSION being the request's x-claude-code-session-id, so a
//! real session's requests can be read before the gateway changes any of them.

use crate::{api, hub::Routed, App};
use connectome::{json, obj, store::Session};
use serde::Serialize;
use std::{collections::{BTreeMap, HashMap}, io::{self, Read, Write}, net::TcpStream, time::{SystemTime, UNIX_EPOCH}};

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
    body: json::Value, // its JSON, or its text if it isn't JSON
    status: Option<u16>, // none if the upstream never answered
    answer_headers: BTreeMap<String, String>,
    answer: String, // as it came: server-sent events, for a stream
    failed: Option<String>, // why it ended short: the upstream unreached or cut off, or the client gone
}

impl App {
    /// The recording of what Claude Code session `id` sent through the gateway.
    pub fn gateway(&self, id: &str) -> Session { Session::open(self.root.join(".gateway").join(id)) }
}

/// The request `method path` with `headers` and `body` forwarded to --upstream, its answer streamed
/// back to `client`, and the exchange recorded (by the hub, the recording's only writer).
pub fn forward(app: &App, method: &str, path: &str, headers: &HashMap<String, String>, body: &[u8], client: &mut TcpStream) -> io::Result<()> {
    let at = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let mut recorded = BTreeMap::new();
    let url = format!("{}{path}", app.upstream.trim_end_matches('/'));
    let mut request = ureq::http::Request::builder().method(method).uri(&url);
    for (name, value) in headers {
        if CONNECTION_HEADERS.contains(&name.as_str()) { continue }
        request = request.header(name, value);
        let shown = if CREDENTIALS.contains(&name.as_str()) { "(left out)" } else { value };
        recorded.insert(name.clone(), shown.to_string());
    }
    let body_shown = json::parse(body).unwrap_or_else(|_| String::from_utf8_lossy(body).into());
    let mut exchange = Exchange { at, method: method.into(), path: path.into(), headers: recorded, body: body_shown, status: None, answer_headers: BTreeMap::new(), answer: String::new(), failed: None };
    let sent = request.body(body).map(|r| app.agent.run(r)); // a GET's empty body goes as content-length: 0
    // the exchange is recorded whether or not the client is still there to hear the answer
    match sent {
        Err(e) => { // a header that isn't one
            let why = format!("the request can't be forwarded: {e}");
            let _ = api::send(client, "400 Bad Request", "application/json", error_json(&why).as_bytes());
            exchange.failed = Some(why);
        }
        Ok(Err(e)) => {
            let why = format!("{url}: {e}");
            let _ = api::send(client, "502 Bad Gateway", "application/json", error_json(&why).as_bytes());
            exchange.failed = Some(why);
        }
        Ok(Ok(response)) => {
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

/// A failure as the Messages API words its own, so the client shows why.
fn error_json(why: &str) -> String {
    json::to_string(&obj! {"type" => "error", "error" => obj! {"type" => "api_error", "message" => format!("connectome's gateway: {why}")}})
}
