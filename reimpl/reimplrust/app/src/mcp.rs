//! Tools from MCP servers (the Model Context Protocol, https://modelcontextprotocol.io): each server
//! the config names is a child process speaking JSON-RPC 2.0 over its stdin and stdout, a message per
//! line. The client says hello (initialize), lists the server's tools (tools/list, page by page) and
//! calls them (tools/call). Each server has a thread that owns its process (Owner), told through one
//! channel both the requests to send, from any session's turn, and what a second thread reads from the
//! server's stdout. It hands each response to the request waiting for it, answers the server's pings,
//! refuses its other requests and passes over its notifications. A request unanswered within the
//! timeout fails, and the server is told to cancel it; a server that has stopped starts again at the
//! next request. The config is the usual one:
//!
//!     {"mcpServers": {"NAME": {"command": "...", "args": ["..."], "env": {"KEY": "..."}}}}

use crate::tools::{Outcome, Tool, Toolbox};
use connectome::json::{self, sparse, Json, Value};
use connectome::obj;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

type Answer = Result<Value, String>;

/// A server as the app knows it: its thread, and its tools' definitions as the Messages API takes them.
pub struct Server {
    link: Link,
    tools: Vec<Value>,
}

/// Where a server's thread is told its requests.
#[derive(Clone)]
struct Link {
    name: String,
    told: Sender<Told>,
}

/// What a server's thread is told: a request (its method and params, and where to answer), or what the
/// thread reading the nth process started reads from its stdout: a message, or that it closed.
enum Told {
    Request(String, Value, Sender<Answer>),
    Read(u64, Value),
    Stopped(u64),
}

/// The servers the config names, each with its tools listed; one that can't start is reported and left
/// out.
pub fn servers(config: &Value, timeout: Duration) -> Vec<Server> {
    config["mcpServers"].as_object().into_iter().flatten().filter_map(|(name, spec)| {
        let (told, inbox) = channel();
        let mut owner = Owner { spec: spec.clone(), timeout, me: told.clone(), stdin: None, started: 0, ids: 0, waiting: HashMap::new() };
        std::thread::spawn(move || owner.run(&inbox));
        let link = Link { name: name.clone(), told };
        link.tools().inspect_err(|e| eprintln!("{e}")).ok().map(|tools| Server { link, tools })
    }).collect()
}

/// Every server's tools, each calling its server, a later server's shadowing an earlier one's.
pub fn toolbox(servers: &[Server]) -> Toolbox {
    servers.iter().fold(Toolbox::default(), |tools, server| {
        tools + Toolbox(server.tools.iter().map(|definition| {
            let (link, name) = (server.link.clone(), definition["name"].as_str().unwrap_or("").to_string());
            Tool { definition: definition.clone(), asks: true, run: Box::new(move |input| link.call(&name, input)) }
        }).collect())
    })
}

impl Link {
    /// A request's answer from the server.
    fn request(&self, method: &str, params: Value) -> Answer {
        let (answer, answered) = channel();
        self.told.send(Told::Request(method.into(), params, answer)).expect("the server's thread runs");
        answered.recv().expect("the server's thread answers").map_err(|e| format!("MCP server {}: {e}", self.name))
    }

    /// Every tool the server lists.
    fn tools(&self) -> Result<Vec<Value>, String> {
        let (mut tools, mut cursor) = (vec![], Some(Value::Null));
        while let Some(at) = cursor {
            let page = self.request("tools/list", sparse(obj! {"cursor" => at}))?;
            let listed = page["tools"].arr().iter().map(|t| obj! {"name" => t["name"].as_str().unwrap_or(""), "description" => t["description"].clone(), "input_schema" => t["inputSchema"].clone()});
            tools.extend(listed);
            cursor = Some(page["nextCursor"].clone()).filter(|c| !c.is_null());
        }
        Ok(tools)
    }

    /// tools/call: the result's content blocks as a tool result's, failed if the server says so.
    fn call(&self, name: &str, input: &Value) -> Outcome {
        let result = self.request("tools/call", obj! {"name" => name, "arguments" => input.clone()}).map_err(Value::from)?;
        let content: Value = result["content"].arr().iter().map(block).collect();
        if result["isError"] == true { Err(content) } else { Ok(content) }
    }
}

/// A server's thread: it owns the server's process (started at the first request, and again at the
/// first after it stops; requests told meanwhile wait in its channel), its stdin, and the requests
/// waiting for an answer, each with its deadline.
struct Owner {
    spec: Value,
    timeout: Duration,
    me: Sender<Told>,
    stdin: Option<ChildStdin>,
    started: u64, // processes started, the last the one running
    ids: u64,
    waiting: HashMap<u64, (Sender<Answer>, Instant, String)>, // by id (from 1): where to answer, by when, the method
}

impl Owner {
    fn run(&mut self, inbox: &Receiver<Told>) {
        loop {
            if let Some(told) = self.next(inbox) {
                self.handle(told);
            }
        }
    }

    /// The next thing told, or None once a deadline passes, failing the requests it passes.
    fn next(&mut self, inbox: &Receiver<Told>) -> Option<Told> {
        let soonest = self.waiting.values().map(|(_, by, _)| *by).min();
        if let Ok(told) = inbox.recv_timeout(soonest.map_or(Duration::MAX, |by| by.saturating_duration_since(Instant::now()))) {
            return Some(told);
        }
        let now = Instant::now();
        for (id, (answer, _, method)) in self.waiting.extract_if(|_, (_, by, _)| *by <= now).collect::<Vec<_>>() {
            let _ = self.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/cancelled", "params" => obj! {"requestId" => id, "reason" => "timed out"}});
            let _ = answer.send(Err(format!("{method} timed out after {} s", self.timeout.as_secs_f64())));
        }
        None
    }

    fn handle(&mut self, told: Told) {
        match told {
            Told::Request(method, params, answer) => {
                if self.stdin.is_none() && let Err(e) = self.start() {
                    return drop(answer.send(Err(e)));
                }
                self.ids += 1;
                match self.send(&obj! {"jsonrpc" => "2.0", "id" => self.ids, "method" => method.as_str(), "params" => params}) {
                    Ok(()) => drop(self.waiting.insert(self.ids, (answer, Instant::now() + self.timeout, method))),
                    Err(e) => drop(answer.send(Err(e))),
                }
            }
            Told::Read(n, message) if n == self.started => {
                let reply = |field: &str, value: Value| obj! {"jsonrpc" => "2.0", "id" => message["id"].clone(), field => value};
                match (message["method"].as_str(), message.get("id")) {
                    (None, Some(id)) => {
                        if let Some((answer, ..)) = id.as_u64().and_then(|id| self.waiting.remove(&id)) {
                            let _ = answer.send(message.get("error").map_or(Ok(message["result"].clone()), |e| Err(e["message"].as_str().unwrap_or("failed").into())));
                        }
                    }
                    (Some("ping"), Some(_)) => drop(self.send(&reply("result", obj! {}))),
                    (Some(_), Some(_)) => drop(self.send(&reply("error", obj! {"code" => -32601, "message" => "not supported"}))),
                    _ => {} // a notification
                }
            }
            Told::Stopped(n) if n == self.started => {
                self.stdin = None;
                for (_, (answer, ..)) in self.waiting.drain() {
                    let _ = answer.send(Err("the server stopped".into()));
                }
            }
            Told::Read(..) | Told::Stopped(_) => {} // from a process that has gone
        }
    }

    /// The server's process started and initialized (initialize, id 0). The thread that reads its
    /// stdout hands over the answer, then tells the messages after it, and those before, to this one.
    fn start(&mut self) -> Result<(), String> {
        let mut command = Command::new(self.spec["command"].as_str().ok_or("no command")?);
        command.args(self.spec["args"].arr().iter().filter_map(Value::as_str));
        command.envs(self.spec["env"].as_object().into_iter().flatten().map(|(k, v)| (k, v.as_str().unwrap_or(""))));
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        self.started += 1;
        let (n, me, (hello, helloed)) = (self.started, self.me.clone(), channel());
        let mut messages = BufReader::new(child.stdout.take().expect("piped")).lines().map_while(Result::ok).filter_map(|line| json::parse(line.as_bytes()).ok()); // a line that isn't JSON-RPC passed over
        self.stdin = child.stdin.take();
        std::thread::spawn(move || {
            let answer = |m: &Value| m["id"] == 0 && m.get("method").is_none();
            let _ = hello.send(messages.by_ref().find(|m| answer(m) || me.send(Told::Read(n, m.clone())).is_err()));
            messages.for_each(|message| drop(me.send(Told::Read(n, message))));
            let _ = me.send(Told::Stopped(n));
            let _ = child.wait();
        });
        let hello = obj! {"name" => "connectome-web", "version" => env!("CARGO_PKG_VERSION")};
        let initialize = obj! {"protocolVersion" => "2025-06-18", "capabilities" => obj! {}, "clientInfo" => hello};
        let ready = self.send(&obj! {"jsonrpc" => "2.0", "id" => 0, "method" => "initialize", "params" => initialize}).and_then(|()| match helloed.recv_timeout(self.timeout) {
            Ok(Some(m)) => m.get("error").map_or(Ok(()), |e| Err(e["message"].as_str().unwrap_or("failed").into())),
            Err(RecvTimeoutError::Timeout) => Err(format!("initialize timed out after {} s", self.timeout.as_secs_f64())),
            _ => Err("the server stopped".into()),
        });
        let ready = ready.and_then(|()| self.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/initialized"}));
        if ready.is_err() {
            self.stdin = None;
        }
        ready
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("the server stopped")?;
        stdin.write_all(format!("{}\n", json::to_string(message)).as_bytes()).and_then(|()| stdin.flush()).map_err(|e| e.to_string())
    }
}

/// An MCP content block as the library logs a tool result's: text and images as they are, anything
/// else (audio, resources) as its JSON in text.
fn block(c: &Value) -> Value {
    match c.kind() {
        "text" => obj! {"type" => "text", "text" => c["text"].clone()},
        "image" => obj! {"type" => "image", "source" => obj! {"type" => "base64", "mediaType" => c["mimeType"].clone(), "data" => c["data"].clone()}},
        _ => obj! {"type" => "text", "text" => json::to_string(c)},
    }
}
