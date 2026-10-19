//! An Anthropic Messages API client, for Anthropic or any provider that serves the format (DeepSeek's
//! https://api.deepseek.com/anthropic, OpenRouter's https://openrouter.ai/api/v1, a local mock), over
//! ureq with rustls and Mozilla's roots compiled in. The request body is written as it's produced and
//! goes out in HTTP chunks, so a context streams from the session's files to the socket one block at a
//! time; the reply streams back as server-sent events.

use connectome::{blocks::Block, emit::Reader, json::{self, sparse, Json, Value}, obj, turns::Turn};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, io::{self, BufRead, BufReader, BufWriter, Write}, ops::Add};
use ureq::tls::{PemItem, TlsConfig};

pub struct Llm {
    pub base: String, // e.g. https://api.anthropic.com
    pub key: String,
    pub agent: ureq::Agent,
    pub spent: Cell<Usage>, // what its replies used since it was last taken
}

/// What replies used, as their providers report it: dollars, if a provider prices them (OpenRouter
/// does), and tokens: the prompts', of those the ones read from the provider's cache, and the replies'.
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub cost: Option<f64>,
    pub input: f64,
    pub cached: f64,
    pub output: f64,
}

impl Add for Usage {
    type Output = Usage;

    fn add(self, more: Usage) -> Usage {
        let cost = self.cost.map_or(more.cost, |cost| Some(cost + more.cost.unwrap_or(0.0)));
        Usage { cost, input: self.input + more.input, cached: self.cached + more.cached, output: self.output + more.output }
    }
}

impl From<Usage> for Value {
    fn from(used: Usage) -> Value {
        serde_json::to_value(used).expect("usage is JSON")
    }
}

/// An HTTP client that trusts Mozilla's roots, or instead only the certificates in `ca` (PEM): a
/// company proxy's, a test server's.
pub fn agent(ca: Option<&[u8]>) -> Result<ureq::Agent, String> {
    let certs = ca.map(|pem| ureq::tls::parse_pem(pem).filter_map(|item| match item { Ok(PemItem::Certificate(c)) => Some(c), _ => None }).collect::<Vec<_>>());
    if certs.as_ref().is_some_and(Vec::is_empty) {
        return Err("no certificates in the PEM".into());
    }
    let tls = certs.map_or(TlsConfig::builder(), |certs| TlsConfig::builder().root_certs(certs.into()));
    // buffers of a request's chunks and the response's headers, below ureq's 128 KB each per connection
    let config = ureq::Agent::config_builder().http_status_as_error(false).input_buffer_size(64 * 1024).output_buffer_size(16 * 1024);
    Ok(config.tls_config(tls.build()).build().into())
}

impl Llm {
    /// POSTs to {base}/v1/messages (or {base}/messages for a base that ends in /v1) `head`'s fields
    /// and `turns` as the messages, read through `reader` as they're written out, asking for a stream;
    /// hands `piece` each piece of the reply's text or reasoning as it arrives ("text" or "thinking",
    /// and the piece), and each call's tool as it starts and its input as it comes ("calling" and the
    /// tool's name, "input" and a piece of its JSON), and gives the whole reply as the library takes it: {stopReason, content, usage:
    /// {inputTokens, outputTokens}}.
    pub fn messages(&self, mut head: Value, turns: &[Turn], reader: &mut Reader, mut piece: impl FnMut(&str, &str)) -> Result<Value, String> {
        let base = self.base.trim_end_matches('/');
        let url = if base.ends_with("/v1") { format!("{base}/messages") } else { format!("{base}/v1/messages") };
        let request = self.agent.post(&url).header("x-api-key", &self.key).header("anthropic-version", "2023-06-01").header("content-type", "application/json");
        head["stream"] = true.into();
        // the reader holds the session, so the body is written on this thread while ureq sends it on another
        let (body, out) = io::pipe().map_err(|e| e.to_string())?;
        let response = std::thread::scope(|scope| {
            let sent = scope.spawn(|| request.send(ureq::SendBody::from_owned_reader(body)));
            let written = write_body(BufWriter::new(out), &head, turns, reader); // the pipe closes as it returns
            match (sent.join().expect("the sending thread"), written) {
                (Err(e), _) => Err(format!("{url}: {e}")),
                (_, Err(e)) => Err(format!("{url}: writing the request: {e}")),
                (Ok(response), _) => Ok(response),
            }
        })?;
        let status = response.status().as_u16();
        let mut body = response.into_body();
        if status != 200 {
            let text = body.read_to_string().unwrap_or_default();
            return Err(format!("HTTP {status}: {}", text.chars().take(500).collect::<String>()));
        }
        // each event's data, until OpenRouter's [DONE] (it ends the stream as OpenAI's does, past message_stop)
        let lines = BufReader::new(body.into_reader()).lines().map(|line| line.map_err(|e| e.to_string()));
        let data = lines.filter_map(|line| line.map(|l| l.strip_prefix("data:").map(|d| d.trim().to_string())).transpose());
        let mut events = data.take_while(|d| d.as_deref() != Ok("[DONE]"));
        let reply = events.try_fold(Reply::default(), |reply, d| reply.event(&json::parse(d?.as_bytes())?, &mut piece))?;
        self.spent.set(self.spent.get() + reply.used);
        reply.done()
    }
}

/// A streamed reply as its events build it, a fold over them.
#[derive(Default)]
struct Reply {
    blocks: Vec<Value>,
    stop: Value,
    used: Usage,
}

impl Reply {
    fn event(mut self, e: &Value, show: &mut impl FnMut(&str, &str)) -> Result<Reply, String> {
        match e.kind() {
            "message_start" => self.used(&e["message"]["usage"]),
            "content_block_start" => {
                if e["content_block"].kind() == "tool_use" {
                    show("calling", e["content_block"]["name"].as_str().unwrap_or(""));
                }
                self.blocks.push(e["content_block"].clone());
            }
            "content_block_delta" => {
                let block = self.blocks.last_mut().ok_or("a delta before its block")?;
                let key = match e["delta"].kind().strip_suffix("_delta") {
                    Some(key @ ("text" | "thinking" | "signature")) => key,
                    Some("input_json") => "partial_json", // a tool call's input, parsed once whole
                    _ => return Ok(self),
                };
                let more = e["delta"][key].as_str().unwrap_or("");
                match &mut block[key] {
                    Value::String(s) => s.push_str(more),
                    slot => *slot = more.into(),
                }
                match key {
                    "partial_json" => show("input", more),
                    "signature" => {}
                    _ => show(key, more),
                }
            }
            "content_block_stop" => {
                if let Some(block) = self.blocks.last_mut()
                    && block.kind() == "tool_use"
                    && let Some(input) = block["partial_json"].as_str().filter(|j| !j.is_empty())
                {
                    block["input"] = json::parse(input.as_bytes())?;
                }
            }
            "message_delta" => {
                self.stop = e["delta"]["stop_reason"].clone();
                self.used(&e["usage"]);
            }
            "error" => return Err(format!("{}: {}", e["error"]["type"].as_str().unwrap_or("error"), e["error"]["message"].as_str().unwrap_or(""))),
            _ => {} // ping, message_stop
        }
        Ok(self)
    }

    /// The usage an event reports. Counts only grow over a stream, which reports them at its start
    /// (Anthropic), at its end (OpenRouter, 0 at the start), or at both; the cost comes at its end.
    fn used(&mut self, usage: &Value) {
        let n = |k: &str| usage[k].as_f64().unwrap_or(0.0);
        let (prompt, cached) = (n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"), n("cache_read_input_tokens"));
        let used = &mut self.used;
        (used.input, used.cached, used.output) = (used.input.max(prompt), used.cached.max(cached), used.output.max(n("output_tokens")));
        used.cost = usage["cost"].as_f64().or(used.cost);
    }

    /// The reply, once it has stopped: a stream that ends before its stop reason was cut off.
    fn done(self) -> Result<Value, String> {
        if self.stop.is_null() {
            return Err("the reply's stream ended before the reply did".into());
        }
        Ok(obj! {
            "stopReason" => self.stop,
            "content" => self.blocks.iter().filter_map(from_api).collect::<Vec<_>>(),
            "usage" => obj! {"inputTokens" => self.used.input, "outputTokens" => self.used.output},
        })
    }
}

/// A block of the API's as the library logs it.
fn from_api(b: &Value) -> Option<Value> {
    Some(match b.kind() {
        "text" => obj! {"type" => "text", "text" => b["text"].clone()},
        "thinking" => obj! {"type" => "thinking", "thinking" => b["thinking"].clone(), "signature" => b["signature"].clone()},
        "redacted_thinking" => obj! {"type" => "redacted_thinking", "data" => b["data"].clone()},
        "tool_use" => obj! {"type" => "tool_use", "id" => b["id"].clone(), "name" => b["name"].clone(), "input" => b["input"].clone()},
        _ => return None,
    })
}

/// A block of the library's as the API takes it, its content moved, not copied.
fn to_api(mut b: Value, cache: bool) -> Value {
    let mut out = match b.kind() {
        "tool_use" => obj! {"type" => "tool_use", "id" => b["id"].take(), "name" => b["name"].take(), "input" => b["input"].take()},
        "tool_result" => {
            let content = match b["content"].take() {
                Value::Array(blocks) => blocks.into_iter().filter(|c| SENT.contains(&c.kind())).map(|c| to_api(c, false)).collect(),
                content => content,
            };
            sparse(obj! {"type" => "tool_result", "tool_use_id" => b["toolUseId"].take(), "content" => content, "is_error" => b["isError"].take()})
        }
        "image" => {
            let source = &mut b["source"];
            obj! {"type" => "image", "source" => obj! {"type" => "base64", "media_type" => source["mediaType"].take(), "data" => source["data"].take()}}
        }
        _ => obj! {"type" => "text", "text" => b["text"].take()},
    };
    if cache {
        out["cache_control"] = obj! {"type" => "ephemeral"};
    }
    out
}

/// The block types a request carries; reasoning, documents and the rest stay in the log.
const SENT: [&str; 4] = ["text", "tool_use", "tool_result", "image"];

/// `head`'s fields, then the turns as the API's messages: the assistant's turns speak as the assistant,
/// everyone else's (the user's, the Context Manager's) as the user; a role's consecutive turns join;
/// blocks the request doesn't carry and blank text are left out, and a turn left empty goes. A turn's
/// cache breakpoint marks its last block.
fn write_body(mut out: impl Write, head: &Value, turns: &[Turn], reader: &mut Reader) -> io::Result<()> {
    let sent = turns.iter().map(|t| (t, t.content.iter().filter(|b| SENT.contains(&b.kind.as_str()) && !b.blank).collect::<Vec<_>>()));
    let turns: Vec<(&Turn, Vec<&Block>)> = sent.filter(|(_, blocks)| !blocks.is_empty()).collect();
    let head = json::to_string(head);
    write!(out, "{},\"messages\":[", &head[..head.len() - 1])?; // the head's fields without its closing brace
    for (i, run) in turns.chunk_by(|(a, _), (b, _)| (a.participant == "assistant") == (b.participant == "assistant")).enumerate() {
        let role = if run[0].0.participant == "assistant" { "assistant" } else { "user" };
        write!(out, "{}{{\"role\":\"{role}\",\"content\":[", if i > 0 { "," } else { "" })?;
        let blocks = run.iter().flat_map(|(t, blocks)| blocks.iter().enumerate().map(move |(j, &b)| (b, t.mark && j + 1 == blocks.len())));
        for (j, (b, mark)) in blocks.enumerate() {
            out.write_all(if j > 0 { b"," } else { b"" })?;
            json::write(&to_api(reader.load(b), mark), &mut out)?;
        }
        out.write_all(b"]}")?;
    }
    out.write_all(b"]}").and_then(|()| out.flush())
}
