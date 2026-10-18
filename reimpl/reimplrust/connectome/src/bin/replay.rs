//! Replays a scripted session for the differential test against the Python port (tests/test_rust.py):
//! from the session `start` names (main by default), each pass's fork, messages, usage and budgets,
//! and the summarizer's replies in call order. Prints what the Python trace records: each summarizer
//! request and each compile, read out in full, and each pass's error.
//!
//!     replay SCRIPT.json SESSION_DIR
//!
//! With --plan, it plans instead each case of a JSON array on stdin ({frontier, have, salt, fixed, W,
//! k, slack, unfold_below}; costs from cost() below) and prints the frontiers.
//!
//!     replay --plan < CASES.json

use connectome::config::Config;
use connectome::drive::step;
use connectome::emit::Reader;
use connectome::json::{self, Json, Value};
use connectome::memory::{Host, Request};
use connectome::planner::{plan, Node};
use connectome::store::Session;
use std::io::Read;
use std::path::Path;

struct Scripted<'a> {
    tools: Option<Value>,
    replies: std::slice::Iter<'a, Value>,
    events: Vec<Value>,
    now: u64,
}

impl Host for Scripted<'_> {
    fn tools(&self) -> Option<&Value> {
        self.tools.as_ref()
    }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        self.events.push(Value::Array(vec![format!("request {}", self.now).into(), request.value(|b| reader.load(b))]));
        let reply = self.replies.next().expect("a scripted reply");
        match reply["raise"].as_str() {
            Some(error) => Err(error.to_string()),
            None => Ok(reply["response"].clone()),
        }
    }
}

/// A made-up price: raw chunks 800..3000, memories 50..600.
fn cost((level, id): Node, salt: u64) -> f64 {
    (if level == 0 { 800 + (id * 7919 + salt) % 2201 } else { 50 + (level * 104729 + id * 7919 + salt) % 551 }) as f64
}

fn plans() {
    let mut text = vec![];
    std::io::stdin().read_to_end(&mut text).expect("stdin");
    let cases = json::parse(&text).expect("cases");
    let out = cases.arr().iter().map(|case| {
        let nodes = |v: &Value| v.arr().iter().map(|n| (n.arr()[0].u64(), n.arr()[1].u64())).collect::<Vec<Node>>();
        let have: Vec<u64> = case["have"].arr().iter().map(Value::u64).collect();
        let cfg = Config { merge_threshold: case["k"].u64(), slack: case["slack"].as_f64().unwrap(), unfold_below: case["unfold_below"].as_f64().unwrap(), ..Config::default() };
        let salt = case["salt"].u64();
        let exists = |(level, id): Node| have.get(level as usize).is_some_and(|&n| id < n);
        let planned = plan(nodes(&case["frontier"]), |n| cost(n, salt), exists, case["fixed"].as_f64().unwrap(), case["W"].as_f64().unwrap(), &cfg);
        Value::Array(planned.into_iter().map(|(l, i)| Value::Array(vec![l.into(), i.into()])).collect())
    });
    println!("{}", json::to_string(&Value::Array(out.collect())));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1] == "--plan" {
        return plans();
    }
    let script = json::parse(&std::fs::read(&args[1]).expect("script")).expect("script JSON");
    let dir = Path::new(&args[2]);
    let cfg: Config = serde_json::from_value(script["config"].clone()).expect("config");
    let tools = Some(script["tools"].clone()).filter(|t| *t != Value::Null);
    let mut host = Scripted { tools, replies: script["replies"].arr().iter(), events: vec![], now: 0 };
    let mut session = Session::open(dir.join(script["start"].as_str().unwrap_or("main")));
    for pass in script["passes"].arr() {
        let t = pass["t"].u64();
        host.now = t;
        if let [at, name] = pass["fork"].arr() {
            session = Session::fork(&session, at.u64(), dir.join(name.as_str().unwrap()));
        }
        let num = |key| pass[key].as_f64().unwrap();
        let (compiled, error) = step(&session, pass["messages"].arr(), pass["usage"].as_f64(), num("budget"), num("reserve"), num("window"), &mut host, &cfg);
        let compiled = match compiled {
            Ok(turns) => {
                let mut reader = Reader::new(&session, false);
                Value::Array(turns.iter().map(|t| t.value(|b| reader.load(b))).collect())
            }
            Err(r) => Value::Array(vec![r.stage.into(), r.actual.into(), r.budget.into()]),
        };
        host.events.push(Value::Array(vec![format!("compile {t}").into(), compiled]));
        host.events.push(Value::Array(vec![format!("error {t}").into(), error.unwrap_or("None".into()).into()]));
    }
    println!("{}", json::to_string(&Value::Array(host.events)));
}
