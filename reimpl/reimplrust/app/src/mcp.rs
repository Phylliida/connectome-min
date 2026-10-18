//! Tools from MCP servers (the Model Context Protocol, https://modelcontextprotocol.io): each server
//! the config names is a child process speaking JSON-RPC 2.0 over its stdin and stdout, a message per
//! line. The client says hello (initialize), lists the server's tools (tools/list, page by page) and
//! calls them (tools/call). A thread per server reads its messages: it hands each response to the
//! request waiting for it, answers the server's pings, refuses its other requests and passes over its
//! notifications. A request unanswered within the timeout fails, and the server is told to cancel it;
//! a server that has stopped starts again at the next request. The config is the usual one:
//!
//!     {"mcpServers": {"NAME": {"command": "...", "args": ["..."], "env": {"KEY": "..."}}}}

use crate::tools::{Outcome, Tool, Toolbox};
use connectome::json::{self, sparse, Json, Value};
use connectome::obj;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Answer = Result<Value, String>;

pub struct Server {
    name: String,
    spec: Value,
    timeout: Duration,
    running: Mutex<Option<Arc<Process>>>,
}

/// A server's process: requests out, and the requests waiting for their answers (one that timed out
/// until its answer comes), None once it has stopped.
struct Process {
    stdin: Mutex<ChildStdin>,
    waiting: Mutex<Option<HashMap<u64, Sender<Answer>>>>,
}

/// The tools of every server the config names, a later server's shadowing an earlier one's; a server
/// that can't start is reported and left out.
pub fn toolbox(config: &Value, timeout: Duration) -> Toolbox {
    config["mcpServers"].as_object().into_iter().flatten().fold(Toolbox::default(), |tools, (name, spec)| {
        let server = Arc::new(Server { name: name.clone(), spec: spec.clone(), timeout, running: Mutex::default() });
        tools + server.tools().inspect_err(|e| eprintln!("{e}")).unwrap_or_default()
    })
}

impl Server {
    /// Every tool the server lists, each calling it.
    fn tools(self: Arc<Server>) -> Result<Toolbox, String> {
        let (mut tools, mut cursor) = (vec![], Some(Value::Null));
        while let Some(at) = cursor {
            let page = self.request("tools/list", sparse(obj! {"cursor" => at}))?;
            for t in page["tools"].arr() {
                let (server, name) = (self.clone(), t["name"].as_str().unwrap_or("").to_string());
                let definition = obj! {"name" => name.as_str(), "description" => t["description"].clone(), "input_schema" => t["inputSchema"].clone()};
                tools.push(Arc::new(Tool { definition, run: Box::new(move |input| server.call(&name, input)) }));
            }
            cursor = Some(page["nextCursor"].clone()).filter(|c| !c.is_null());
        }
        Ok(Toolbox(tools))
    }

    /// tools/call: the result's content blocks as a tool result's, failed if the server says so.
    fn call(&self, name: &str, input: &Value) -> Outcome {
        let result = self.request("tools/call", obj! {"name" => name, "arguments" => input.clone()}).map_err(Value::from)?;
        let content: Value = result["content"].arr().iter().map(block).collect();
        if result["isError"] == true { Err(content) } else { Ok(content) }
    }

    /// The running process's answer, the process started first if it isn't running.
    fn request(&self, method: &str, params: Value) -> Answer {
        let process = {
            let mut running = self.running.lock().unwrap();
            match running.as_ref().filter(|p| p.waiting.lock().unwrap().is_some()) {
                Some(p) => Ok(p.clone()),
                None => self.start().map(|p| running.insert(p).clone()),
            }
        };
        process.and_then(|p| p.request(method, params, self.timeout)).map_err(|e| format!("MCP server {}: {e}", self.name))
    }

    fn start(&self) -> Result<Arc<Process>, String> {
        let spec = &self.spec;
        let mut command = Command::new(spec["command"].as_str().ok_or("no command")?);
        command.args(spec["args"].arr().iter().filter_map(Value::as_str));
        command.envs(spec["env"].as_object().into_iter().flatten().map(|(k, v)| (k, v.as_str().unwrap_or(""))));
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        let stdout = BufReader::new(child.stdout.take().expect("piped"));
        let process = Arc::new(Process { stdin: Mutex::new(child.stdin.take().expect("piped")), waiting: Mutex::new(Some(HashMap::new())) });
        let reader = process.clone();
        std::thread::spawn(move || { reader.read(stdout); let _ = child.wait(); });
        let hello = obj! {"name" => "connectome-web", "version" => env!("CARGO_PKG_VERSION")};
        process.request("initialize", obj! {"protocolVersion" => "2025-06-18", "capabilities" => obj! {}, "clientInfo" => hello}, self.timeout)?;
        process.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/initialized"})?;
        Ok(process)
    }
}

impl Process {
    fn request(&self, method: &str, params: Value, timeout: Duration) -> Answer {
        static IDS: AtomicU64 = AtomicU64::new(1);
        let (id, (answer, answered)) = (IDS.fetch_add(1, Ordering::Relaxed), channel());
        self.waiting.lock().unwrap().as_mut().ok_or("the server stopped")?.insert(id, answer);
        self.send(&obj! {"jsonrpc" => "2.0", "id" => id, "method" => method, "params" => params})?;
        answered.recv_timeout(timeout).unwrap_or_else(|_| {
            let _ = self.send(&obj! {"jsonrpc" => "2.0", "method" => "notifications/cancelled", "params" => obj! {"requestId" => id, "reason" => "timed out"}});
            Err(format!("{method} timed out after {} s", timeout.as_secs_f64()))
        })
    }

    fn send(&self, message: &Value) -> Result<(), String> {
        let mut stdin = self.stdin.lock().unwrap();
        stdin.write_all(format!("{}\n", json::to_string(message)).as_bytes()).and_then(|_| stdin.flush()).map_err(|e| e.to_string())
    }

    /// The server's messages until it stops; then every request still waiting fails.
    fn read(&self, stdout: impl BufRead) {
        for line in stdout.lines().map_while(Result::ok) {
            let Ok(message) = json::parse(line.as_bytes()) else { continue }; // not JSON-RPC
            let reply = |field: &str, value: Value| drop(self.send(&obj! {"jsonrpc" => "2.0", "id" => message["id"].clone(), field => value}));
            match (message["method"].as_str(), message.get("id")) {
                (None, Some(id)) => {
                    if let Some(waiting) = id.as_u64().and_then(|id| self.waiting.lock().unwrap().as_mut()?.remove(&id)) {
                        let _ = waiting.send(message.get("error").map_or(Ok(message["result"].clone()), |e| Err(e["message"].as_str().unwrap_or("failed").into())));
                    }
                }
                (Some("ping"), Some(_)) => reply("result", obj! {}),
                (Some(_), Some(_)) => reply("error", obj! {"code" => -32601, "message" => "not supported"}),
                _ => {} // a notification
            }
        }
        for (_, waiting) in self.waiting.lock().unwrap().take().into_iter().flatten() {
            let _ = waiting.send(Err("the server stopped".into()));
        }
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
