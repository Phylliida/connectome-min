//! Claude Code sessions: a session whose provider names a `command`, how to run Claude Code, takes
//! each turn in Claude Code rather than in requests of its own. connectome-web runs that command
//! headless, speaking JSON lines on its stdin and stdout as the Agent SDK does (stream-json, and the
//! SDK's control requests, which ask before a tool runs), so all it needs is a Claude Code install:
//!
//!   claude --print --input-format stream-json --output-format stream-json --verbose
//!          --include-partial-messages --replay-user-messages --permission-prompt-tool stdio --setting-sources ""
//!          --settings '{"sandbox":{"enabled":false}}' --strict-mcp-config [--mcp-config MCP]
//!          --model MODEL [--append-system-prompt SYSTEM]
//!          (--session-id ID | --resume ID)
//!
//! in the session's workspace, its requests sent to this server's gateway (ANTHROPIC_BASE_URL), with
//! x-connectome-session naming the session, and in the session's sandbox if it has one: bubblewrap
//! around all of Claude Code, the same as bash's, with Claude Code's own files (its command's,
//! ~/.claude and ~/.claude.json) and the network, which reaching the gateway takes. Claude Code's own
//! sandbox stays off. A turn sends what was said, then streams the reply to the pages,
//! logs it, block by block as Claude Code finishes them, asks the pages before a call Claude Code
//! asks about (unless the session always allows its tool; a subagent's call, which stays in Claude
//! Code and out of the log, the pages are shown whole), logs the results, and ends at Claude
//! Code's result, whose cost joins the session's costs. What the user, the session's partner, a channel
//! or a board says meanwhile goes to Claude Code at once, which takes it in after the step it's on (a
//! call's result), or once its reply is done, as a turn of its own in the same run; it echoes each as
//! it does, and the log has it there, so the turn ends once Claude Code has taken in all it was told.
//! Claude Code keeps the conversation (its session, ID, in ~/.claude), so each turn starts it again,
//! resuming that session.

use crate::{api::brief, boards, companion::Heard, costs::{self, For}, hub::{Asking, Event, Job, Msg, Routed}, llm::{Content, Delta, Piece, Streamed, Usage}, settings::Sandbox, tools::{bash, POLL}, work::{said, Worker, APPROVAL_WAIT, DENIED, STOPPED}, App};
use connectome::{drive::append, json::{self, sparse, Json, Value}, obj};
use serde::Deserialize;
use std::{collections::VecDeque, hash::{BuildHasher, Hasher}, io::{BufRead, BufReader, Read, Write}, path::Path, process::{ChildStdin, Command, Stdio}, sync::mpsc::{channel, RecvTimeoutError}, time::Duration};
use wait_timeout::ChildExt;

/// Why a Claude Code session can't go on without a message: Claude Code takes a turn from one.
pub const NO_CONTINUE: &str = "a Claude Code session goes on from a message: say what it should do";

/// Claude Code's own settings for a session: its sandbox off, whatever the defaults, since the
/// session's sandbox is bubblewrap around all of Claude Code (with its own on, Claude Code would also
/// run Bash without asking, sandbox.autoAllowBashIfSandboxed). The user's settings files don't load
/// (--setting-sources "").
const SETTINGS: &str = r#"{"sandbox":{"enabled":false}}"#;

/// How long Claude Code has to exit once its input closes after a turn.
const EXIT_WAIT: Duration = Duration::from_secs(10);

/// The characters kept of what Claude Code writes on stderr, its end, for a turn that fails.
const STDERR_KEPT: usize = 4000;

/// How long a turn waits for the rest of Claude Code's stderr once it has exited.
const STDERR_WAIT: Duration = Duration::from_secs(1);

/// How session `name` runs Claude Code, if its provider is a Claude Code one.
pub fn command(app: &App, name: &str) -> Result<Option<Vec<String>>, String> {
    let (all, own) = (app.settings()?, app.profile(name)?);
    let provider = all.providers.get(&own.provider).ok_or(format!("no provider {} in settings.json", own.provider))?;
    Ok(provider.command.clone())
}

/// What Claude Code writes, a JSON line each; what connectome-web doesn't use passes (Other). A
/// subagent's messages and stream (parent_tool_use_id) stay in Claude Code.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Said {
    System { #[serde(default)] subtype: String, #[serde(default)] claude_code_version: String },
    StreamEvent { event: Value, parent_tool_use_id: Option<String> },
    Assistant { message: Message, parent_tool_use_id: Option<String> },
    User { message: Message, parent_tool_use_id: Option<String>, #[serde(default, rename = "isReplay")] is_replay: bool },
    ControlRequest { request_id: String, request: Value },
    Result { #[serde(default)] subtype: String, #[serde(default)] is_error: bool, result: Option<String>, total_cost_usd: Option<f64>, #[serde(default)] usage: Used },
    #[serde(other)] Other,
}

/// An API message as Claude Code passes it on: an assistant's holds one block (its content), each
/// block of a reply coming once it's finished, all with the reply's id; a user's holds tool results.
#[derive(Deserialize)]
struct Message { #[serde(default)] id: String, content: Value }

/// A control request of Claude Code's that connectome-web answers; others get an error.
#[derive(Deserialize)]
#[serde(tag = "subtype", rename_all = "snake_case")]
enum Asked {
    CanUseTool { tool_name: String, input: Value, #[serde(default)] tool_use_id: String, agent_id: Option<String> },
    #[serde(other)] Other,
}

/// A turn's tokens, all its requests', as Claude Code's result counts them.
#[derive(Deserialize, Default)]
struct Used {
    #[serde(default)] input_tokens: f64,
    #[serde(default)] cache_read_input_tokens: f64,
    #[serde(default)] cache_creation_input_tokens: f64,
    #[serde(default)] output_tokens: f64,
}

impl Worker<'_> {
    /// A turn in Claude Code on `message`, which the log has: Claude Code started on the session
    /// it keeps (claude_session; a new one, the first time, kept once Claude Code has started it), told
    /// it, and heard until its result, or until the user stops the turn, when it's killed and what came
    /// of the reply joins the log as it is.
    pub fn claude_turn(&self, command: &[String], message: &Value) -> Result<(), String> {
        let (all, own) = (self.app.settings()?, self.app.profile(self.name)?);
        let workspace = own.workspace.clone().ok_or("a Claude Code session runs in its workspace: set one in the session's settings")?;
        let boards = boards::of(self.app, &own);
        // 1) Claude Code, as the provider's command runs it
        let (id, resuming) = match &own.claude_session {
            Some(id) => (id.clone(), true),
            None => (new_uuid(), false),
        };
        let mut args: Vec<String> = ["--print", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--replay-user-messages", "--permission-prompt-tool", "stdio", "--setting-sources", "", "--settings", SETTINGS, "--strict-mcp-config"].map(String::from).into();
        if let Some(mcp) = &self.app.mcp_config {
            args.extend(["--mcp-config".into(), mcp.clone()]);
        }
        args.extend(["--model".into(), own.model.clone()]);
        let system: Vec<String> = [own.system.clone().filter(|s| !s.is_empty()), boards::note(self.app, &boards, self.name)].into_iter().flatten().collect();
        if !system.is_empty() {
            args.extend(["--append-system-prompt".into(), system.join("\n\n")]);
        }
        args.extend([if resuming { "--resume" } else { "--session-id" }.into(), id.clone()]);
        let mut claude = if own.sandbox {
            if !cfg!(target_os = "linux") {
                return Err("the session's sandbox is bubblewrap, which runs only on Linux: turn the sandbox off in its settings".into());
            }
            let mut sandbox: Sandbox = all.sandbox.clone();
            sandbox.rw.extend(boards.iter().map(|b| self.app.board_path(b).to_string_lossy().into_owned()));
            sandbox.ro.extend(command.iter().filter(|c| Path::new(c).is_absolute()).cloned()); // the program, and a script or loader it names
            sandbox.rw.extend(["~/.claude".into(), "~/.claude.json".into()]); // its sessions, settings and login
            sandbox.network = true; // the gateway is this server, on the network's loopback
            let mut wrapped = Command::new("bwrap");
            wrapped.args(bash::bwrap(&workspace, &sandbox)).args(command);
            wrapped
        } else {
            let mut plain = Command::new(&command[0]);
            plain.args(&command[1..]);
            plain
        };
        claude.args(&args).current_dir(&workspace).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        claude.env("ANTHROPIC_BASE_URL", format!("http://{}/gateway", self.app.addr));
        claude.env("ANTHROPIC_CUSTOM_HEADERS", format!("x-connectome: 1\nx-connectome-session: {}", self.name));
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut claude, 0); // so a stop kills what it started too
        let provider = &own.provider;
        let mut child = claude.spawn().map_err(|e| format!("can't start Claude Code ({}): {e}; provider {provider}'s command in settings.json says how to run it", command.join(" ")))?;
        // its output a line at a time, and the end of its stderr, each read by a thread of its own, so
        // neither pipe fills while the other is read
        let (stdout, mut stderr) = (child.stdout.take().expect("piped"), child.stderr.take().expect("piped"));
        let (line_sender, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if line_sender.send(line).is_err() { break }
            }
        });
        let (error_sender, errors) = channel();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text); // till every process holding the pipe has gone
            let cut = text.chars().count().saturating_sub(STDERR_KEPT);
            let _ = error_sender.send(text.chars().skip(cut).collect::<String>());
        });
        // 2) what was said, then what Claude Code says, until its result
        let mut stdin = child.stdin.take().expect("piped");
        let initialize = obj! {"type" => "control_request", "request_id" => "initialize", "request" => obj! {"subtype" => "initialize"}};
        let started = send(&mut stdin, &initialize).and_then(|()| send(&mut stdin, &told_line(message)));
        let mut told = VecDeque::from([Told { message: message.clone(), logged: true, job: None }]);
        self.tell(Msg::Taking); // what the user says from now on comes here, at once
        let mut version = String::new();
        let mut reply = Reply::default();
        let mut last_prompt = None; // the latest request's prompt
        let ended = match started { Err(e) => Err(format!("telling Claude Code what was said: {e}")), Ok(()) => 'turn: loop {
            if self.stopped() { break Err(STOPPED.into()) }
            // what's said meanwhile goes to Claude Code as it comes; a fork, or a turn that continues
            // the log, waits for the turn
            while let Ok((job, heard)) = self.jobs.try_recv() {
                self.taken.set(self.taken.get() + 1);
                let (message, users) = match heard {
                    Job::Say { text: Some(text) } => (said("user", text), Some(job)),
                    Job::Hear(Heard { from, text }) | Job::Notice(Heard { from, text }) => (said(&from, text), None),
                    other => {
                        self.later.borrow_mut().push_back((job, other));
                        continue;
                    }
                };
                if let Err(e) = send(&mut stdin, &told_line(&message)) {
                    break 'turn Err(format!("telling Claude Code what was said: {e}"));
                }
                told.push_back(Told { message, logged: false, job: users });
            }
            let line = match lines.recv_timeout(POLL) {
                Ok(Ok(line)) => line,
                Ok(Err(e)) => break Err(format!("reading Claude Code's output: {e}")),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break Err("Claude Code stopped before its turn ended".into()),
            };
            if line.trim().is_empty() { continue }
            let said: Said = match serde_json::from_str(&line) {
                Ok(said) => said,
                Err(e) => break Err(format!("Claude Code {version} wrote a line connectome-web can't read ({e}): {}", line.chars().take(300).collect::<String>())),
            };
            match said {
                Said::System { subtype, claude_code_version } if subtype == "init" => {
                    version = claude_code_version;
                    if !resuming { // Claude Code has started the session: later turns resume it
                        let merged = self.s.count("layer2") > 0;
                        if let Err(e) = self.app.ask_session(self.name, |answer| Msg::Tune(obj! {"claude_session" => id.as_str()}, merged, answer)) {
                            break Err(format!("keeping Claude Code's session {id}: {e}"));
                        }
                    }
                }
                Said::StreamEvent { event, parent_tool_use_id: None } => match serde_json::from_value(event) {
                    // a request starts: its prompt is the context Claude Code has now (the result's tokens sum the turn's requests)
                    Ok(Streamed::MessageStart { message }) => {
                        let counted = |n: Option<f64>| n.unwrap_or(0.0);
                        let prompt = counted(message.usage.input_tokens) + counted(message.usage.cache_read_input_tokens) + counted(message.usage.cache_creation_input_tokens);
                        self.event(Event::Prompt(prompt));
                        last_prompt = Some(prompt);
                    }
                    Ok(event) => {
                        if let Some(piece) = piece(event) { self.event(Event::Piece(piece)) }
                    }
                    Err(_) => {} // an event the stream's types don't cover passes
                },
                Said::Assistant { message, parent_tool_use_id: None } => {
                    if reply.id != message.id { self.log_reply(&mut reply) }
                    reply.id = message.id;
                    reply.blocks.extend(message.content.arr().iter().cloned());
                }
                Said::User { parent_tool_use_id: None, is_replay: true, .. } => {
                    // Claude Code takes in what it was told: the log has it here, where Claude Code's conversation does
                    self.log_reply(&mut reply);
                    if let Some(taken) = told.pop_front() && !taken.logged {
                        append(&self.s, &taken.message, &own.config);
                        self.logged();
                        if let Some(job) = taken.job { self.tell(Msg::Taken(job)) }
                    }
                }
                Said::User { message, parent_tool_use_id: None, .. } => {
                    self.log_reply(&mut reply);
                    let results: Vec<Value> = message.content.arr().iter().filter(|b| b.kind() == "tool_result").map(logged_result).collect();
                    if results.is_empty() { continue }
                    for result in &results { self.event(Event::Result(brief(result.clone()))) }
                    append(&self.s, &obj! {"participant" => "user", "content" => results}, &own.config);
                    self.logged();
                    if !boards.is_empty() { // what its calls changed on its boards is its doing
                        self.app.hub.send(Routed::Stepped(self.name.into())).expect("the hub runs");
                    }
                }
                Said::ControlRequest { request_id, request } => {
                    self.log_reply(&mut reply); // the call Claude Code asks about is in the log, where the pages answer it
                    let response = match serde_json::from_value(request) {
                        Ok(Asked::CanUseTool { tool_name, input, tool_use_id, agent_id }) => {
                            let call = Asking { id: tool_use_id.clone(), tool: tool_name.clone(), input: input.clone(), agent: agent_id };
                            let by_subagent = call.agent.is_some();
                            let decided = match self.allowed(call) {
                                Ok(()) => {
                                    if !by_subagent { // a subagent's call runs in it, out of the page's sight
                                        self.event(Event::Running { id: tool_use_id.clone(), tool: tool_name });
                                    }
                                    obj! {"behavior" => "allow", "updatedInput" => input, "toolUseID" => tool_use_id}
                                }
                                Err(why) => obj! {"behavior" => "deny", "message" => why, "toolUseID" => tool_use_id},
                            };
                            obj! {"subtype" => "success", "request_id" => request_id, "response" => decided}
                        }
                        _ => obj! {"subtype" => "error", "request_id" => request_id, "error" => "connectome-web doesn't answer this request"},
                    };
                    if let Err(e) = send(&mut stdin, &obj! {"type" => "control_response", "response" => response}) {
                        break Err(format!("answering Claude Code: {e}"));
                    }
                }
                Said::Result { subtype, is_error, result, total_cost_usd, usage } => {
                    self.log_reply(&mut reply);
                    let prompts = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
                    let used = Usage { cost: total_cost_usd, input: prompts, cached: usage.cache_read_input_tokens, output: usage.output_tokens };
                    if let Some(spent) = costs::spend(&self.s, For::Reply, used, last_prompt) { self.event(Event::Spent(spent)) }
                    if is_error { break Err(result.unwrap_or(subtype)) }
                    // what Claude Code was told after its reply ends it takes in next, as a turn of its own in the same run
                    if told.iter().all(|t| t.logged) { break Ok(()) }
                }
                _ => {}
            }
        } };
        self.log_reply(&mut reply); // a reply cut short joins the log as far as it came
        // what Claude Code was told but didn't take in, the session's thread takes next, each a turn of its own
        for left in told.into_iter().filter(|t| !t.logged) {
            let text = left.message["content"][0]["text"].as_str().unwrap_or("").to_string();
            let job = match left.message["participant"].as_str() {
                Some("user") => Job::Say { text: Some(text) },
                from => Job::Notice(Heard { from: from.unwrap_or("").into(), text }),
            };
            self.later.borrow_mut().push_back((left.job.unwrap_or(0), job));
        }
        // 3) Claude Code done with: its input closed, so it exits; killed at once if the turn failed
        drop(stdin);
        if ended.is_err() { kill(&mut child) }
        if !matches!(child.wait_timeout(EXIT_WAIT), Ok(Some(_))) { kill(&mut child) }
        // a process Claude Code started outside a sandbox may hold its stderr open after it exits
        let errors = errors.recv_timeout(STDERR_WAIT).unwrap_or_default();
        match ended {
            Err(why) if why != STOPPED && !errors.trim().is_empty() => Err(format!("{why}. Claude Code said: {}", errors.trim())),
            ended => ended,
        }
    }

    /// The reply streaming in, its blocks so far, joins the log, if it has any.
    fn log_reply(&self, reply: &mut Reply) {
        if reply.blocks.is_empty() { return }
        let Ok(own) = self.app.profile(self.name) else { return };
        append(&self.s, &obj! {"participant" => "assistant", "content" => std::mem::take(&mut reply.blocks)}, &own.config);
        self.logged();
    }

    /// Whether the user lets `call` run: at once if the session always allows its tool; otherwise the
    /// hub asks the pages and decides by a page's answer, no answer within APPROVAL_WAIT a denial.
    fn allowed(&self, call: Asking) -> Result<(), String> {
        if self.app.profile(self.name)?.allow.contains(&call.tool) { return Ok(()) }
        let (answer, answered) = channel();
        self.tell(Msg::Ask { call, answer });
        answered.recv_timeout(APPROVAL_WAIT).unwrap_or_else(|_| Err(DENIED.into()))
    }
}

/// A message Claude Code was told and hasn't taken in yet (it echoes each as it does): the message as
/// the log keeps it, whether the log has it already (the turn's own, logged before Claude Code starts),
/// and, for the user's, its job, which the pages show as handed over until then.
struct Told { message: Value, logged: bool, job: Option<u64> }

/// A message as Claude Code takes it on its stdin.
fn told_line(message: &Value) -> Value {
    obj! {"type" => "user", "message" => obj! {"role" => "user", "content" => message["content"].clone()}, "parent_tool_use_id" => Value::Null, "session_id" => ""}
}

/// The reply streaming in: its id, and the blocks it has that the log doesn't yet. A call Claude Code
/// asks about, or a call's result, comes before the reply's later blocks, which then join the log
/// after it, as a message of their own.
#[derive(Default)]
struct Reply { id: String, blocks: Vec<Value> }

/// A piece for the pages from an event of the reply's stream.
fn piece(event: Streamed) -> Option<Piece> {
    match event {
        Streamed::ContentBlockStart { content_block: Content::ToolUse { name, .. } } => Some(Piece::Calling(name)),
        Streamed::ContentBlockDelta { delta: Delta::Text { text } } => Some(Piece::Text(text)),
        Streamed::ContentBlockDelta { delta: Delta::Thinking { thinking } } => Some(Piece::Thinking(thinking)),
        Streamed::ContentBlockDelta { delta: Delta::InputJson { partial_json } } => Some(Piece::Input(partial_json)),
        _ => None,
    }
}

/// A tool result as the API has it (Claude Code's), as the library logs it: toolUseId, isError only
/// when true, an image's source as {mediaType, data}.
fn logged_result(b: &Value) -> Value {
    let content = match &b["content"] {
        Value::Array(parts) => {
            let mut logged = vec![];
            for part in parts {
                if part.kind() == "image" {
                    logged.push(obj! {"type" => "image", "source" => obj! {"mediaType" => part["source"]["media_type"].clone(), "data" => part["source"]["data"].clone()}});
                } else {
                    logged.push(part.clone());
                }
            }
            Value::Array(logged)
        }
        content => content.clone(),
    };
    sparse(obj! {"type" => "tool_result", "toolUseId" => b["tool_use_id"].clone(), "content" => content, "isError" => b["is_error"].as_bool().filter(|&e| e)})
}

/// A line to Claude Code.
fn send(stdin: &mut ChildStdin, line: &Value) -> std::io::Result<()> {
    let mut text = json::to_string(line);
    text.push('\n');
    stdin.write_all(text.as_bytes()).and_then(|()| stdin.flush())
}

/// Claude Code killed, with what it started (its process group; bubblewrap's PID namespace goes with it).
fn kill(child: &mut std::process::Child) {
    #[cfg(unix)]
    unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) };
    let _ = child.kill();
    let _ = child.wait();
}

/// A random UUID (version 4), a new Claude Code session's id: from the standard library's hasher
/// keys, which it seeds from the system's randomness.
fn new_uuid() -> String {
    let half = || std::collections::hash_map::RandomState::new().build_hasher().finish() as u128;
    let mut n = (half() << 64) | half();
    n = (n & !(0xf << 76)) | (0x4 << 76); // version 4: byte 6's high nibble
    n = (n & !(0x3 << 62)) | (0x2 << 62); // the RFC 9562 variant: byte 8's two high bits 10
    let h = format!("{n:032x}");
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_id_is_a_version_4_uuid() {
        let (a, b) = (new_uuid(), new_uuid());
        assert_ne!(a, b);
        let parts: Vec<&str> = a.split('-').collect();
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), [8, 4, 4, 4, 12]);
        assert!(parts[2].starts_with('4') && "89ab".contains(&parts[3][..1]), "{a}");
    }

    #[test]
    fn results_are_logged_as_the_library_logs_them() {
        let image = obj! {"type" => "image", "source" => obj! {"type" => "base64", "media_type" => "image/png", "data" => "iVBO"}};
        let api = obj! {"type" => "tool_result", "tool_use_id" => "t1", "content" => vec![obj! {"type" => "text", "text" => "hi"}, image], "is_error" => false};
        let logged = obj! {"type" => "tool_result", "toolUseId" => "t1", "content" => vec![obj! {"type" => "text", "text" => "hi"}, obj! {"type" => "image", "source" => obj! {"mediaType" => "image/png", "data" => "iVBO"}}]};
        assert_eq!(logged_result(&api), logged);
        assert_eq!(logged_result(&obj! {"type" => "tool_result", "tool_use_id" => "t2", "content" => "no", "is_error" => true}), obj! {"type" => "tool_result", "toolUseId" => "t2", "content" => "no", "isError" => true});
    }
}
