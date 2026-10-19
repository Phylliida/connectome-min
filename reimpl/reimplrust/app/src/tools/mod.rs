//! Tools the model can call, as plugins. A tool is its definition, as the Messages API takes it
//! ({name, description, input_schema}), whether a call waits for the user's yes, and a function from
//! its input to its outcome: Ok(content), or Err(content) for a failure, the content being a tool
//! result's (a string, or text and image blocks as the library logs them). A Toolbox is a monoid under
//! `+`: the union of two, a tool on the right shadowing a same-named one on the left, with the empty
//! toolbox as identity. So tool sources compose in order: the tools built into the binary (bash.rs,
//! edit.rs), whose `run` does the work in-process, then each MCP server's, in the config's order
//! (mcp.rs). A toolbox is made for each step, from the settings as they are then, and owns its tools.

use connectome::{json::{sparse, Value}, obj};
use std::{ops::Add, panic::{catch_unwind, AssertUnwindSafe}, time::Duration};

pub mod bash;
pub mod edit;
pub mod mcp;

pub type Outcome = Result<Value, Value>;

/// Whether the user has stopped the turn a call runs in, which a call that takes long asks every POLL
/// and ends at once if so.
pub type Stopped<'a> = &'a dyn Fn() -> bool;

pub const POLL: Duration = Duration::from_millis(100);

/// A tool's work: from a call's input to its outcome.
pub type Run = Box<dyn Fn(&Value, Stopped) -> Outcome>;

pub struct Tool {
    pub definition: Value,
    pub asks: bool,
    pub run: Run,
}

impl Tool {
    pub fn name(&self) -> &str { self.definition["name"].as_str().unwrap_or("") }
}

#[derive(Default)]
pub struct Toolbox(pub Vec<Tool>);

impl Toolbox {
    /// The tools a request declares; None without any.
    pub fn definitions(&self) -> Option<Value> { (!self.0.is_empty()).then(|| self.0.iter().map(|t| t.definition.clone()).collect()) }

    /// A tool_use block's result. A call to no tool, or one whose tool panics, fails.
    pub fn call(&self, call: &Value, stopped: Stopped) -> Value {
        let tool = self.0.iter().find(|t| call["name"] == t.name());
        let run = |t: &Tool| catch_unwind(AssertUnwindSafe(|| (t.run)(&call["input"], stopped))).unwrap_or_else(|_| Err("the tool crashed".into()));
        result(call, tool.map_or_else(|| Err(format!("no tool {}", call["name"].as_str().unwrap_or("?")).into()), run))
    }

    pub fn names(&self) -> Vec<&str> { self.0.iter().map(Tool::name).collect() }

    /// The tools whose calls wait for the user's yes.
    pub fn asking(&self) -> Vec<&str> { self.0.iter().filter(|t| t.asks).map(Tool::name).collect() }
}

impl Add for Toolbox {
    type Output = Toolbox;

    fn add(self, right: Toolbox) -> Toolbox {
        let kept: Vec<Tool> = self.0.into_iter().filter(|t| !right.names().contains(&t.name())).collect();
        Toolbox(kept.into_iter().chain(right.0).collect())
    }
}

/// A tool where it can't run, declared all the same, so that a session whose history calls it still
/// declares it (as the Messages API requires, and so a call comes back as one, not as text): what it
/// `does`, but not here, and why; each call fails at once, saying `why`, without asking.
pub fn unavailable(name: &str, does: &str, why: &'static str) -> Toolbox {
    let definition = obj! {"name" => name, "description" => format!("{does}, but not in this session: {why}."), "input_schema" => obj! {"type" => "object"}};
    Toolbox(vec![Tool { definition, asks: false, run: Box::new(move |_, _| Err(why.into())) }])
}

/// A call's result as the library logs it: {type, toolUseId, content, isError?}.
pub fn result(call: &Value, outcome: Outcome) -> Value {
    let failed = outcome.is_err().then_some(true);
    let (Ok(content) | Err(content)) = outcome;
    sparse(obj! {"type" => "tool_result", "toolUseId" => call["id"].clone(), "content" => content, "isError" => failed})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, answer: &'static str) -> Toolbox {
        let definition = obj! {"name" => name, "description" => "", "input_schema" => obj! {"type" => "object"}};
        Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |_, _| if answer == "panic" { panic!("boom") } else { Ok(answer.into()) }) }])
    }

    fn answers(tools: &Toolbox) -> Vec<String> {
        tools.names().iter().map(|&name| tools.call(&obj! {"id" => "t", "name" => name}, &|| false)["content"].as_str().unwrap_or("").to_string()).collect()
    }

    #[test]
    fn a_monoid_of_shadowing_unions() {
        let (a, b, a2) = (|| tool("a", "a1"), || tool("b", "b1"), || tool("a", "a2"));
        assert_eq!(answers(&(a() + Toolbox::default())), answers(&a()));
        assert_eq!(answers(&(Toolbox::default() + a())), answers(&a()));
        assert_eq!(answers(&((a() + b()) + a2())), answers(&(a() + (b() + a2()))));
        assert_eq!(answers(&(a() + b() + a2())), ["b1", "a2"]); // the later `a` shadows the earlier
    }

    #[test]
    fn failures_are_results() {
        let tools = tool("bad", "panic");
        assert_eq!(tools.call(&obj! {"id" => "t1", "name" => "bad"}, &|| false), obj! {"type" => "tool_result", "toolUseId" => "t1", "content" => "the tool crashed", "isError" => true});
        assert_eq!(tools.call(&obj! {"id" => "t2", "name" => "gone"}, &|| false)["content"], "no tool gone");
        assert_eq!(tool("ok", "fine").call(&obj! {"id" => "t3", "name" => "ok"}, &|| false), obj! {"type" => "tool_result", "toolUseId" => "t3", "content" => "fine"});
    }
}
