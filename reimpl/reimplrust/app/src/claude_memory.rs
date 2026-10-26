//! A Claude Code session's memories (claude.rs): the library asks for each (memory::Host), and a run
//! of Claude Code of its own writes it, on the session's model, so a memory is the same model's as the
//! session's turns. Claude Code takes only user messages, so the request goes as one: a transcript of
//! its turns (recall pairs of earlier memories, the stretch of conversation, the instruction), each
//! headed by who says it, calls and their results written out, reasoning left out as the library's
//! requests leave it out. The run has no tools, Claude Code's least system prompt (--system-prompt
//! ""), keeps no session, and may write the memory's max tokens (CLAUDE_CODE_MAX_OUTPUT_TOKENS); it
//! runs as the session's turns do (Worker::start_claude): in its workspace and sandbox, through the
//! gateway, but not naming the session, so its requests go as they came (the session's thread, which
//! folds a turn's, waits on the run). What a run used joins the session's costs, for memories.

use crate::{claude::{told_line, Said, SETTINGS}, costs::{self, For}, hub::Event, llm::Usage, tools::POLL, work::Worker};
use connectome::{emit::Reader, json::{Json, Value}, memory::{Host, Request, INTERRUPTED}, obj};
use std::{collections::HashMap, sync::mpsc::RecvTimeoutError};

/// What the transcript starts with: how it's laid out. The library's own messages in it say what
/// it's for.
const PREFACE: &str = "This is part of your conversation in this session, written out as a transcript: each part is headed by who says it \
([assistant] is you; [Context Manager] is the system that keeps your memories), and tool calls and their results are written in brackets.";

/// The library's summarizer for a Claude Code session, which runs `command` (its provider's); how
/// its last reply ended, and the tokens it could write, for why a memory didn't form.
pub struct Remembering<'a, 'w> {
    pub worker: &'a Worker<'w>,
    pub command: Vec<String>,
    pub ended: Option<(String, f64)>,
}

impl Host for Remembering<'_, '_> {
    fn carries_tool_blocks(&self) -> bool { true } // they go as text

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let own = self.worker.app.profile(self.worker.name)?;
        let model = if request.model.is_empty() { own.model } else { request.model.clone() };
        let args: Vec<String> = ["--print", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--setting-sources", "", "--settings", SETTINGS,
            "--strict-mcp-config", "--tools", "", "--system-prompt", "", "--no-session-persistence", "--model", &model].map(String::from).into();
        let mut claude = self.worker.start_claude(&self.command, &args, &[("CLAUDE_CODE_MAX_OUTPUT_TOKENS", request.max_tokens.to_string())])?;
        let asked = obj! {"content" => vec![obj! {"type" => "text", "text" => transcript(request, reader)}]};
        let initialize = obj! {"type" => "control_request", "request_id" => "initialize", "request" => obj! {"subtype" => "initialize"}};
        let told = claude.send(&initialize).and_then(|()| claude.send(&told_line(&asked)));
        let (mut written, mut limits) = (vec![], None);
        let reply = match told { Err(e) => Err(format!("telling Claude Code what to remember: {e}")), Ok(()) => loop {
            if self.worker.stopped() { break Err(INTERRUPTED.into()) }
            let line = match claude.lines.recv_timeout(POLL) {
                Ok(Ok(line)) => line,
                Ok(Err(e)) => break Err(format!("reading Claude Code's output: {e}")),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break Err("Claude Code stopped before it wrote the memory".into()),
            };
            match serde_json::from_str(&line) {
                Ok(Said::Assistant { message, parent_tool_use_id: None }) => {
                    written.extend(message.content.arr().iter().filter(|b| b.kind() == "text").cloned()); // its reasoning stays out
                }
                Ok(Said::ControlRequest { request_id, .. }) => { // it has no tools, so nothing to ask about
                    let refused = obj! {"type" => "control_response", "response" => obj! {"subtype" => "error", "request_id" => request_id, "error" => "a memory's run answers no requests"}};
                    if let Err(e) = claude.send(&refused) { break Err(format!("answering Claude Code: {e}")) }
                }
                Ok(Said::RateLimitEvent { rate_limit_info }) => {
                    limits = rate_limit_info.heard();
                    if let Some(heard) = &limits { self.worker.event(Event::Limits(heard.clone())) }
                }
                Ok(Said::Result { subtype, is_error, result, stop_reason, total_cost_usd, usage }) => {
                    let prompts = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
                    let used = Usage { cost: total_cost_usd, input: prompts, cached: usage.cache_read_input_tokens, output: usage.output_tokens };
                    if let Some(spent) = costs::spend(&self.worker.s, For::Memories, used, None, limits.take()) { self.worker.event(Event::Spent(spent)) }
                    if is_error { break Err(result.unwrap_or(subtype)) }
                    let stop = stop_reason.unwrap_or_else(|| "end_turn".into());
                    break Ok(obj! {"stopReason" => stop.as_str(), "content" => std::mem::take(&mut written), "usage" => obj! {"inputTokens" => prompts, "outputTokens" => usage.output_tokens}});
                }
                Ok(_) => {}
                Err(e) => break Err(format!("Claude Code wrote a line connectome-web can't read ({e}): {}", line.chars().take(300).collect::<String>())),
            }
        } };
        let errors = claude.finish(reply.is_err());
        self.ended = reply.as_ref().ok().map(|r| (r["stopReason"].as_str().unwrap_or("nothing").to_string(), request.max_tokens));
        match reply {
            Err(why) if why != INTERRUPTED && !errors.trim().is_empty() => Err(format!("{why}. Claude Code said: {}", errors.trim())),
            reply => reply,
        }
    }
}

/// The request as one message: PREFACE, then its turns, each headed by who says it.
fn transcript(request: &Request, reader: &mut Reader) -> String {
    let mut text = PREFACE.to_string();
    let mut called = HashMap::new(); // each call's tool, by its id, for its result
    for turn in &request.messages {
        text += &format!("\n\n[{}]", turn.participant);
        for block in &turn.content {
            let b = reader.load(block); // in order, as Reader reads them
            match b.kind() {
                "text" => text += &format!("\n{}", b["text"].as_str().unwrap_or("")),
                "tool_use" => {
                    let tool = b["name"].as_str().unwrap_or("a tool").to_string();
                    text += &format!("\n[called {tool} with {}]", connectome::json::to_string(&b["input"]));
                    called.insert(b["id"].as_str().unwrap_or("").to_string(), tool);
                }
                "tool_result" => {
                    let tool = called.get(b["toolUseId"].as_str().unwrap_or("")).map_or("a call", String::as_str);
                    let failed = if b["isError"] == true { ", failed" } else { "" };
                    text += &format!("\n[{tool}'s result{failed}: {}]", said(&b["content"]));
                }
                "image" => text += "\n[an image]",
                _ => {} // reasoning, which the library's requests leave out too
            }
        }
    }
    text
}

/// A tool result's content as text: itself, or its blocks' text, an image named.
fn said(content: &Value) -> String {
    match content {
        Value::Array(parts) => parts.iter().map(|p| if p.kind() == "image" { "[an image]" } else { p["text"].as_str().unwrap_or("") }).collect::<Vec<_>>().join("\n"),
        content => content.as_str().unwrap_or("").to_string(),
    }
}
