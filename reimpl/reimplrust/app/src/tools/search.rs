//! The web_search tool: a query asked of a SearXNG server (https://docs.searxng.org) through its JSON
//! API (GET {base}/search?q=…&format=json, which the server's settings.yml allows by listing json in
//! search.formats), its results given back as text: each page's title, address, date when known, and
//! excerpt.

use super::{Outcome, Tool, Toolbox};
use connectome::{json::{self, Json, Value}, obj};
use std::time::Duration;

/// The most results a call may ask for: SearXNG's first page.
const MOST: u64 = 20;

/// web_search on the SearXNG server at `base`, asked through `agent`, a call given `limit`.
pub fn tool(base: String, agent: ureq::Agent, limit: Duration) -> Toolbox {
    let count = format!("How many results, at most (10 by default, up to {MOST})");
    let properties = obj! {"query" => obj! {"type" => "string", "description" => "What to search for"}, "count" => obj! {"type" => "integer", "description" => count}};
    let description = "Search the web, and get the pages found: each one's title, address, date when known, and an excerpt.";
    let definition = obj! {"name" => "web_search", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["query"]}};
    Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |input, _| search(&base, &agent, limit, input)) }])
}

fn search(base: &str, agent: &ureq::Agent, limit: Duration, input: &Value) -> Outcome {
    let query = input["query"].as_str().filter(|q| !q.trim().is_empty()).ok_or("web_search needs a query")?;
    let count = input["count"].as_u64().unwrap_or(10).clamp(1, MOST) as usize;
    let url = format!("{}/search", base.trim_end_matches('/'));
    let failed = |why: String| Value::from(format!("web_search at {url}: {why}"));
    let asked = agent.get(&url).query("q", query).query("format", "json").config().timeout_global(Some(limit)).build();
    let response = asked.call().map_err(|e| failed(e.to_string()))?;
    let status = response.status().as_u16();
    let text = response.into_body().read_to_string().map_err(|e| failed(e.to_string()))?;
    match status {
        200 => {}
        403 => return Err(failed("the server doesn't answer in JSON: its settings.yml must list json in search.formats".into())),
        _ => return Err(failed(format!("HTTP {status}"))),
    }
    let found = json::parse(text.as_bytes()).map_err(failed)?;
    let pages: Vec<String> = found["results"].arr().iter().take(count).enumerate().map(|(i, page)| {
        let date = page["publishedDate"].as_str().map(|d| format!("{} · ", d.get(..10).unwrap_or(d))).unwrap_or_default();
        format!("{}. {}\n{}\n{date}{}", i + 1, page["title"].as_str().unwrap_or(""), page["url"].as_str().unwrap_or(""), page["content"].as_str().unwrap_or("").trim())
    }).collect();
    Ok(if pages.is_empty() { format!("nothing found for {query}") } else { pages.join("\n\n") }.into())
}
