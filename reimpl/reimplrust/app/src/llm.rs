//! An Anthropic Messages API client, for Anthropic or any provider that serves the format (DeepSeek's
//! https://api.deepseek.com/anthropic, OpenRouter's https://openrouter.ai/api/v1, a local mock), over
//! ureq with rustls and Mozilla's roots compiled in. The request body is written as it's produced and
//! goes out in HTTP chunks, so a context streams from the session's files to the socket one block at a
//! time; the reply streams back as server-sent events.

use connectome::{blocks::Block, emit::Reader, json::{self, sparse, Json, Value}, obj, turns::Turn};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, io::{self, BufRead, BufReader, BufWriter, Write}, iter::Sum, ops::Add};
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
pub struct Usage { pub cost: Option<f64>, pub input: f64, pub cached: f64, pub output: f64 }

impl Add for Usage {
    type Output = Usage;

    fn add(self, more: Usage) -> Usage {
        let cost = self.cost.map_or(more.cost, |cost| Some(cost + more.cost.unwrap_or(0.0)));
        Usage { cost, input: self.input + more.input, cached: self.cached + more.cached, output: self.output + more.output }
    }
}

impl Sum for Usage {
    fn sum<I: Iterator<Item = Usage>>(all: I) -> Usage {
        all.fold(Usage::default(), Usage::add)
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
    /// and `turns` as the messages, read through `reader` as they're written out, asking for a stream.
    /// Hands `piece` each piece of the reply as it arrives (Piece), and gives the whole reply as the
    /// library takes it: {stopReason, content, usage: {inputTokens, outputTokens}}.
    pub fn messages(&self, mut head: Value, turns: &[Turn], reader: &mut Reader, mut piece: impl FnMut(Piece)) -> Result<Value, String> {
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
        let reply = events.try_fold(Reply::default(), |reply, d| reply.event(serde_json::from_str(&d?).map_err(|e| e.to_string())?, &mut piece))?;
        self.spent.set(self.spent.get() + reply.used);
        reply.done()
    }
}

/// A piece of a reply as it streams in: of its reasoning or its text, a call's tool as the call starts,
/// or a piece of the call's input (JSON). Serialized as {"thinking": piece} and so on.
#[derive(Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Piece { Thinking(String), Text(String), Calling(String), Input(String) }

/// The server-sent events that build a reply; any other passes.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Streamed {
    MessageStart { message: Started },
    ContentBlockStart { content_block: Content },
    ContentBlockDelta { delta: Delta },
    ContentBlockStop {},
    MessageDelta { delta: Stopped, #[serde(default)] usage: Counted },
    Error { error: Failure },
    #[serde(other)] Other, // ping, message_stop
}

#[derive(Deserialize)]
struct Started { #[serde(default)] usage: Counted }

#[derive(Deserialize)]
struct Stopped { stop_reason: Option<String> }

#[derive(Deserialize)]
struct Failure { #[serde(rename = "type", default)] kind: String, #[serde(default)] message: String }

/// The tokens and dollars an event reports; any of them may be missing.
#[derive(Deserialize, Default)]
struct Counted {
    input_tokens: Option<f64>,
    cache_read_input_tokens: Option<f64>,
    cache_creation_input_tokens: Option<f64>,
    output_tokens: Option<f64>,
    cost: Option<f64>,
}

/// A content block as its events build it: started, then extended by deltas of its kind.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Content {
    Text { text: String },
    Thinking { thinking: String, signature: Option<String> },
    RedactedThinking { data: Value },
    ToolUse { id: String, name: String, input: Value, #[serde(skip)] json: String }, // the input's JSON, parsed once whole
    #[serde(other)] Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")] Text { text: String },
    #[serde(rename = "thinking_delta")] Thinking { thinking: String },
    #[serde(rename = "signature_delta")] Signature { signature: String },
    #[serde(rename = "input_json_delta")] InputJson { partial_json: String },
    #[serde(other)] Other,
}

/// A streamed reply as its events build it, a fold over them.
#[derive(Default)]
struct Reply { blocks: Vec<Content>, stop: Option<String>, used: Usage }

impl Reply {
    fn event(mut self, e: Streamed, show: &mut impl FnMut(Piece)) -> Result<Reply, String> {
        match e {
            Streamed::MessageStart { message } => self.used(&message.usage),
            Streamed::ContentBlockStart { content_block } => {
                if let Content::ToolUse { name, .. } = &content_block {
                    show(Piece::Calling(name.clone()));
                }
                self.blocks.push(content_block);
            }
            Streamed::ContentBlockDelta { delta } => match (self.blocks.last_mut().ok_or("a delta before its block")?, delta) {
                (Content::Text { text }, Delta::Text { text: more }) => {
                    text.push_str(&more);
                    show(Piece::Text(more));
                }
                (Content::Thinking { thinking, .. }, Delta::Thinking { thinking: more }) => {
                    thinking.push_str(&more);
                    show(Piece::Thinking(more));
                }
                (Content::Thinking { signature, .. }, Delta::Signature { signature: more }) => signature.get_or_insert_default().push_str(&more),
                (Content::ToolUse { json, .. }, Delta::InputJson { partial_json }) => {
                    json.push_str(&partial_json);
                    show(Piece::Input(partial_json));
                }
                _ => {} // a delta this app doesn't keep
            },
            Streamed::ContentBlockStop {} => {
                if let Some(Content::ToolUse { input, json, .. }) = self.blocks.last_mut()
                    && !json.is_empty()
                {
                    *input = json::parse(json.as_bytes())?;
                }
            }
            Streamed::MessageDelta { delta, usage } => {
                self.stop = delta.stop_reason;
                self.used(&usage);
            }
            Streamed::Error { error } => return Err(format!("{}: {}", if error.kind.is_empty() { "error" } else { &error.kind }, error.message)),
            Streamed::Other => {}
        }
        Ok(self)
    }

    /// The usage an event reports. Counts only grow over a stream, which reports them at its start
    /// (Anthropic), at its end (OpenRouter, 0 at the start), or at both; the cost comes at its end.
    fn used(&mut self, c: &Counted) {
        let n = |count: Option<f64>| count.unwrap_or(0.0);
        let (prompt, cached) = (n(c.input_tokens) + n(c.cache_read_input_tokens) + n(c.cache_creation_input_tokens), n(c.cache_read_input_tokens));
        let used = &mut self.used;
        (used.input, used.cached, used.output) = (used.input.max(prompt), used.cached.max(cached), used.output.max(n(c.output_tokens)));
        used.cost = c.cost.or(used.cost);
    }

    /// The reply, once it has stopped: a stream that ends before its stop reason was cut off.
    fn done(self) -> Result<Value, String> {
        let stop = self.stop.ok_or("the reply's stream ended before the reply did")?;
        Ok(obj! {
            "stopReason" => stop,
            "content" => self.blocks.into_iter().filter_map(Content::logged).collect::<Vec<_>>(),
            "usage" => obj! {"inputTokens" => self.used.input, "outputTokens" => self.used.output},
        })
    }
}

impl Content {
    /// The block as the library logs it; None for a kind it doesn't keep.
    fn logged(self) -> Option<Value> {
        Some(match self {
            Content::Text { text } => obj! {"type" => "text", "text" => text},
            Content::Thinking { thinking, signature } => obj! {"type" => "thinking", "thinking" => thinking, "signature" => signature},
            Content::RedactedThinking { data } => obj! {"type" => "redacted_thinking", "data" => data},
            Content::ToolUse { id, name, input, .. } => obj! {"type" => "tool_use", "id" => id, "name" => name, "input" => input},
            Content::Other => return None,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply built from a stream's events, and the pieces it showed.
    fn streamed(events: &[&str]) -> (Result<Reply, String>, Vec<String>) {
        let mut pieces = vec![];
        let reply = events.iter().try_fold(Reply::default(), |reply, e| reply.event(serde_json::from_str(e).expect("an event"), &mut |p| pieces.push(json::to_string(&p))));
        (reply, pieces)
    }

    #[test]
    fn a_stream_builds_its_reply() {
        let (reply, pieces) = streamed(&[
            r#"{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"cache_read_input_tokens":5,"cache_creation_input_tokens":null,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hm."}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2ln"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"","citations":null}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":{"cited_text":"x"}}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"server_tool_use","id":"s1","name":"web_search","input":{}}}"#,
            r#"{"type":"content_block_stop","index":2}"#,
            r#"{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#,
            r#"{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#,
            r#"{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}"#,
            r#"{"type":"content_block_stop","index":3}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":7,"cost":0.5}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let reply = reply.expect("a reply");
        assert_eq!((reply.used.cost, reply.used.input, reply.used.cached, reply.used.output), (Some(0.5), 15.0, 5.0, 7.0));
        assert_eq!(pieces, [r#"{"thinking":"Hm."}"#, r#"{"text":"Hi"}"#, r#"{"calling":"bash"}"#, r#"{"input":"{\"command\":"}"#, r#"{"input":"\"ls\"}"}"#]);
        let content = vec![
            obj! {"type" => "thinking", "thinking" => "Hm.", "signature" => "c2ln"},
            obj! {"type" => "text", "text" => "Hi"},
            obj! {"type" => "tool_use", "id" => "t1", "name" => "bash", "input" => obj! {"command" => "ls"}},
        ];
        assert_eq!(reply.done(), Ok(obj! {"stopReason" => "tool_use", "content" => content, "usage" => obj! {"inputTokens" => 15.0, "outputTokens" => 7.0}}));
    }

    #[test]
    fn a_stream_can_fail() {
        let start = r#"{"type":"message_start","message":{"usage":{"input_tokens":0}}}"#;
        let (reply, _) = streamed(&[start, r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#]);
        assert_eq!(reply.err(), Some("overloaded_error: Overloaded".into()));
        let (reply, _) = streamed(&[start, r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#]);
        assert_eq!(reply.expect("a reply so far").done().err(), Some("the reply's stream ended before the reply did".into()));
    }
}
