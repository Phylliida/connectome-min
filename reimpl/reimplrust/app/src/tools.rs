//! Tools the model can call, as plugins. A tool is its definition, as the Messages API takes it
//! ({name, description, input_schema}), whether a call waits for the user's yes, and a function from
//! its input to its outcome: Ok(content), or
//! Err(content) for a failure, the content being a tool result's (a string, or text and image blocks
//! as the library logs them). A Toolbox is a monoid under `+`: the union of two, a tool on the right
//! shadowing a same-named one on the left, with the empty toolbox as identity. So tool sources compose
//! in order: the tools built into the binary (bash.rs), then each MCP server's, in the config's order
//! (mcp.rs). A tool built in is a Tool whose `run` does the work in-process.

use connectome::json::{sparse, Value};
use connectome::obj;
use std::ops::Add;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pub type Outcome = Result<Value, Value>;

pub struct Tool {
    pub definition: Value,
    pub asks: bool,
    pub run: Box<dyn Fn(&Value) -> Outcome + Send + Sync>,
}

#[derive(Clone, Default)]
pub struct Toolbox(pub Vec<Arc<Tool>>);

impl Toolbox {
    /// The tools a request declares; None without any.
    pub fn definitions(&self) -> Option<Value> {
        (!self.0.is_empty()).then(|| self.0.iter().map(|t| t.definition.clone()).collect())
    }

    /// A tool_use block's result. A call to no tool, or one whose tool panics, fails.
    pub fn call(&self, call: &Value) -> Value {
        let tool = self.0.iter().find(|t| t.definition["name"] == call["name"]);
        let run = |t: &Arc<Tool>| catch_unwind(AssertUnwindSafe(|| (t.run)(&call["input"]))).unwrap_or_else(|_| Err("the tool crashed".into()));
        result(call, tool.map_or_else(|| Err(format!("no tool {}", call["name"].as_str().unwrap_or("?")).into()), run))
    }

    pub fn names(&self) -> Vec<&str> {
        self.0.iter().filter_map(|t| t.definition["name"].as_str()).collect()
    }
}

impl Add for Toolbox {
    type Output = Toolbox;

    fn add(self, right: Toolbox) -> Toolbox {
        let kept: Vec<Arc<Tool>> = self.0.into_iter().filter(|t| !right.0.iter().any(|r| r.definition["name"] == t.definition["name"])).collect();
        Toolbox(kept.into_iter().chain(right.0).collect())
    }
}

/// A call's result as the library logs it: {type, toolUseId, content, isError?}.
pub fn result(call: &Value, outcome: Outcome) -> Value {
    let (content, failed) = match outcome {
        Ok(content) => (content, None),
        Err(content) => (content, Some(true)),
    };
    sparse(obj! {"type" => "tool_result", "toolUseId" => call["id"].clone(), "content" => content, "isError" => failed})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, answer: &'static str) -> Toolbox {
        let definition = obj! {"name" => name, "description" => "", "input_schema" => obj! {"type" => "object"}};
        Toolbox(vec![Arc::new(Tool { definition, asks: true, run: Box::new(move |_| if answer == "panic" { panic!("boom") } else { Ok(answer.into()) }) })])
    }

    fn answers(tools: &Toolbox) -> Vec<String> {
        tools.names().iter().map(|&name| tools.call(&obj! {"id" => "t", "name" => name})["content"].as_str().unwrap_or("").to_string()).collect()
    }

    #[test]
    fn a_monoid_of_shadowing_unions() {
        let (a, b, a2) = (tool("a", "a1"), tool("b", "b1"), tool("a", "a2"));
        assert_eq!(answers(&(a.clone() + Toolbox::default())), answers(&a));
        assert_eq!(answers(&(Toolbox::default() + a.clone())), answers(&a));
        assert_eq!(answers(&((a.clone() + b.clone()) + a2.clone())), answers(&(a.clone() + (b.clone() + a2.clone()))));
        assert_eq!(answers(&(a + b + a2)), ["b1", "a2"]); // the later `a` shadows the earlier
    }

    #[test]
    fn failures_are_results() {
        let tools = tool("bad", "panic");
        assert_eq!(tools.call(&obj! {"id" => "t1", "name" => "bad"}), obj! {"type" => "tool_result", "toolUseId" => "t1", "content" => "the tool crashed", "isError" => true});
        assert_eq!(tools.call(&obj! {"id" => "t2", "name" => "gone"})["content"], "no tool gone");
        assert_eq!(tool("ok", "fine").call(&obj! {"id" => "t3", "name" => "ok"}), obj! {"type" => "tool_result", "toolUseId" => "t3", "content" => "fine"});
    }
}
