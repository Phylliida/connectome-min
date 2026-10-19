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

use super::{Outcome, Tool, Toolbox};
use connectome::{json::{self, sparse, Json, Value}, obj};
use std::{collections::HashMap, io::{BufRead, BufReader, Write}, process::{Child, ChildStdin, Command, Stdio}, time::{Duration, Instant}};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};

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

/// What a server's thread is told: a request (its method and params, and where to answer); what the
/// thread reading the nth process started reads from its stdout: a message, or that it closed; or to
/// stop its process and itself.
enum Told {
    Request(String, Value, Sender<Answer>),
    Read(u64, Value),
    Stopped(u64),
    Quit,
}

/// The servers the config names, each with its tools listed, all at once; one whose tools can't be
/// listed is reported and stopped.
pub fn servers(config: &Value, timeout: Duration) -> Vec<Server> {
    let links = config["mcpServers"].as_object().into_iter().flatten().map(|(name, spec)| {
        let (told, inbox) = channel();
        let owner = Owner { spec: spec.clone(), timeout, me: told.clone(), running: None, started: 0, ids: 0, waiting: HashMap::new() };
        std::thread::spawn(move || owner.run(inbox));
        Link { name: name.clone(), told }
    });
    std::thread::scope(|scope| {
        let listing: Vec<_> = links.map(|link| scope.spawn(move || (link.tools(), link))).collect();
        listing.into_iter().filter_map(|listing| match listing.join().expect("a listing") {
            (Ok(tools), link) => Some(Server { link, tools }),
            (Err(e), link) => {
                eprintln!("{e}");
                let _ = link.told.send(Told::Quit);
                None
            }
        }).collect()
    })
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

/// A server's thread: it owns the server's process and its stdin (started at the first request, and
/// again at the first after it stops; requests told meanwhile wait in its channel), and the requests
/// waiting for an answer, each with its deadline.
struct Owner {
    spec: Value,
    timeout: Duration,
    me: Sender<Told>,
    running: Option<(Child, ChildStdin)>,
    started: u64, // processes started, the last the one running
    ids: u64,
    waiting: HashMap<u64, (Sender<Answer>, Instant, String)>, // by id (from 1): where to answer, by when, the method
}

impl Owner {
    fn run(mut self, inbox: Receiver<Told>) {
        loop {
            let soonest = self.waiting.values().map(|(_, by, _)| *by).min();
            match inbox.recv_timeout(soonest.map_or(Duration::MAX, |by| by.saturating_duration_since(Instant::now()))) {
                Ok(Told::Quit) => return self.stop(),
                Ok(told) => self.handle(told),
                Err(_) => self.expire(), // a deadline passed (the channel never closes: it holds `me`)
            }
        }
    }

    /// The requests past their deadline failed, the server told to cancel each.
    fn expire(&mut self) {
        let now = Instant::now();
        for (id, (answer, _, method)) in self.waiting.extract_if(|_, (_, by, _)| *by <= now).collect::<Vec<_>>() {
            let _ = self.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/cancelled", "params" => obj! {"requestId" => id, "reason" => "timed out"}});
            let _ = answer.send(Err(format!("{method} timed out after {} s", self.timeout.as_secs_f64())));
        }
    }

    fn handle(&mut self, told: Told) {
        match told {
            Told::Request(method, params, answer) => {
                self.ids += 1;
                let started = if self.running.is_some() { Ok(()) } else { self.start() };
                match started.and_then(|()| self.send(&obj! {"jsonrpc" => "2.0", "id" => self.ids, "method" => method.as_str(), "params" => params})) {
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
                self.stop();
                self.waiting.drain().for_each(|(_, (answer, ..))| drop(answer.send(Err("the server stopped".into()))));
            }
            Told::Read(..) | Told::Stopped(_) | Told::Quit => {} // from a process that has gone (Quit: run's)
        }
    }

    /// The server's process killed, if it runs, and waited for.
    fn stop(&mut self) {
        if let Some((mut child, _)) = self.running.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// The server's process started and initialized (initialize, id 0). The thread that reads its
    /// stdout tells this one the messages before the answer, hands over the answer, then tells the rest.
    fn start(&mut self) -> Result<(), String> {
        let (args, env) = (self.spec["args"].arr().iter().filter_map(Value::as_str), self.spec["env"].as_object().into_iter().flatten());
        let mut command = Command::new(self.spec["command"].as_str().ok_or("no command")?);
        let mut child = command.args(args).envs(env.map(|(k, v)| (k, v.as_str().unwrap_or("")))).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        self.started += 1;
        let (n, me, (hello, helloed)) = (self.started, self.me.clone(), channel());
        let messages = BufReader::new(child.stdout.take().expect("piped")).lines().map_while(Result::ok).filter_map(|line| json::parse(line.as_bytes()).ok()); // a line that isn't JSON-RPC passed over
        let stdin = child.stdin.take().expect("piped");
        self.running = Some((child, stdin));
        std::thread::spawn(move || {
            let mut hello = Some(hello);
            for message in messages {
                match hello.take_if(|_| message["id"] == 0 && message.get("method").is_none()) {
                    Some(hello) => drop(hello.send(message)),
                    None => drop(me.send(Told::Read(n, message))),
                }
            }
            let _ = me.send(Told::Stopped(n));
        });
        let hello = obj! {"name" => "connectome-web", "version" => env!("CARGO_PKG_VERSION")};
        let initialize = obj! {"protocolVersion" => "2025-06-18", "capabilities" => obj! {}, "clientInfo" => hello};
        let ready = self.send(&obj! {"jsonrpc" => "2.0", "id" => 0, "method" => "initialize", "params" => initialize}).and_then(|()| match helloed.recv_timeout(self.timeout) {
            Ok(m) => m.get("error").map_or(Ok(()), |e| Err(e["message"].as_str().unwrap_or("failed").into())),
            Err(RecvTimeoutError::Timeout) => Err(format!("initialize timed out after {} s", self.timeout.as_secs_f64())),
            Err(RecvTimeoutError::Disconnected) => Err("the server stopped".into()),
        });
        ready.and_then(|()| self.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/initialized"})).inspect_err(|_| self.stop())
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        let (_, stdin) = self.running.as_mut().ok_or("the server stopped")?;
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
