//! An Anthropic Messages API client, for Anthropic or any provider that serves the format (DeepSeek's
//! https://api.deepseek.com/anthropic, OpenRouter's https://openrouter.ai/api/v1, a local mock), over
//! ureq with rustls and Mozilla's roots compiled in. The request body is written as it's produced and
//! goes out in HTTP chunks, so a context streams from the session's files to the socket one block at a
//! time; the reply streams back as server-sent events.

use connectome::blocks::Block;
use connectome::emit::Reader;
use connectome::json::{self, sparse, Json, Value};
use connectome::obj;
use connectome::turns::Turn;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use ureq::tls::{PemItem, TlsConfig};

pub struct Llm {
    pub base: String, // e.g. https://api.anthropic.com
    pub key: String,
    pub agent: ureq::Agent,
}

/// An HTTP client that trusts Mozilla's roots, or instead only the certificates in `ca` (PEM): a
/// company proxy's, a test server's.
pub fn agent(ca: Option<&[u8]>) -> Result<ureq::Agent, String> {
    let mut tls = TlsConfig::builder();
    if let Some(pem) = ca {
        let certs: Vec<_> = ureq::tls::parse_pem(pem).filter_map(|item| match item { Ok(PemItem::Certificate(c)) => Some(c), _ => None }).collect();
        if certs.is_empty() {
            return Err("no certificates in the PEM".into());
        }
        tls = tls.root_certs(certs.into());
    }
    // buffers of a request's chunks and the response's headers, below ureq's 128 KB each per connection
    let config = ureq::Agent::config_builder().http_status_as_error(false).input_buffer_size(64 * 1024).output_buffer_size(16 * 1024);
    Ok(config.tls_config(tls.build()).build().into())
}

impl Llm {
    /// POSTs to {base}/v1/messages (or {base}/messages for a base that ends in /v1) `head`'s fields
    /// and `turns` as the messages, read through `reader` as they're written out, asking for a stream;
    /// hands `piece` each piece of the reply's text or reasoning as it arrives ("text" or "thinking",
    /// and the piece), and gives the whole reply as the library takes it: {stopReason, content, usage:
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
            let mut out = BufWriter::new(out);
            let written = write_body(&mut out, &head, turns, reader).and_then(|()| out.flush());
            drop(out);
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
        let data = BufReader::new(body.into_reader()).lines().filter_map(|line| match line {
            Ok(line) => Some(Ok(line.strip_prefix("data:")?.trim().to_string())),
            Err(e) => Some(Err(e.to_string())),
        });
        let mut events = data.take_while(|d| d.as_deref() != Ok("[DONE]"));
        events.try_fold(Reply::default(), |reply, d| reply.event(&json::parse(d?.as_bytes())?, &mut piece)).and_then(Reply::done)
    }
}

/// A streamed reply as its events build it, a fold over them.
#[derive(Default)]
struct Reply {
    blocks: Vec<Value>,
    stop: Value,
    input: f64,
    output: f64,
}

impl Reply {
    fn event(mut self, e: &Value, show: &mut impl FnMut(&str, &str)) -> Result<Reply, String> {
        let tokens = |usage: &Value, k: &str| usage[k].as_f64().unwrap_or(0.0);
        match e.kind() {
            "message_start" => {
                let usage = &e["message"]["usage"];
                self.input = tokens(usage, "input_tokens") + tokens(usage, "cache_read_input_tokens") + tokens(usage, "cache_creation_input_tokens");
                self.output = tokens(usage, "output_tokens");
            }
            "content_block_start" => self.blocks.push(e["content_block"].clone()),
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
                if key == "text" || key == "thinking" {
                    show(key, more);
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
                self.output = e["usage"]["output_tokens"].as_f64().unwrap_or(self.output);
            }
            "error" => return Err(format!("{}: {}", e["error"]["type"].as_str().unwrap_or("error"), e["error"]["message"].as_str().unwrap_or(""))),
            _ => {} // ping, message_stop
        }
        Ok(self)
    }

    /// The reply, once it has stopped: a stream that ends before its stop reason was cut off.
    fn done(self) -> Result<Value, String> {
        if self.stop.is_null() {
            return Err("the reply's stream ended before the reply did".into());
        }
        Ok(obj! {
            "stopReason" => self.stop,
            "content" => self.blocks.iter().filter_map(from_api).collect::<Vec<_>>(),
            "usage" => obj! {"inputTokens" => self.input, "outputTokens" => self.output},
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
                Value::Array(blocks) => Value::Array(blocks.into_iter().filter(|c| SENT.contains(&c.kind())).map(|c| to_api(c, false)).collect()),
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
fn write_body(out: &mut impl Write, head: &Value, turns: &[Turn], reader: &mut Reader) -> io::Result<()> {
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
            json::write(&to_api(reader.load(b), mark), &mut *out)?;
        }
        out.write_all(b"]}")?;
    }
    out.write_all(b"]}")
}
