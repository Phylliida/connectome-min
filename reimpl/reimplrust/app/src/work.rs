//! A session's thread: it does its jobs one at a time, in the order they come (a message or fork told
//! during a turn waits for it), the only one to write the session's log. A turn's chat message joins
//! the log, then the turn takes steps until a reply calls no tools, each reading the settings again:
//! the context compiles and streams from disk into the request, the reply joins the log, and the tools
//! it calls run, each once the user allows it, their results joining the log. Memories form between
//! steps and after the turn. The thread tells the hub what the pages following the job see (Msg).

use crate::{api::brief, costs, hub::{Job, Msg}, llm::Llm, settings::{self, Sandbox, SessionSettings}, tools::{self, bash, edit, mcp, Toolbox}, App};
use connectome::{blocks::density_tokens, drive::{append, current, drive}, emit::{select, Reader}, json::{self, sparse, Json, Value}, obj};
use connectome::{memory::{Host, Request}, store::Session};
use std::{panic::{catch_unwind, AssertUnwindSafe}, path::Path, sync::mpsc::{channel, Receiver}, time::Duration};

/// Steps a turn may take, a model request each, before it stops.
const MAX_STEPS: usize = 64;

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// A denied call's result.
pub const DENIED: &str = "the user didn't allow this call";

/// What a session's step runs with: its settings, the model client and tools they give, the tools'
/// definitions, and the tokens the system prompt and those take, which every request carries.
pub struct Setup {
    pub settings: SessionSettings,
    llm: Llm,
    pub tools: Toolbox,
    definitions: Option<Value>,
    pub overhead: f64,
}

impl Setup {
    /// What a session's step runs with, from the settings as they are now.
    pub fn new(app: &App, name: &str) -> Result<Setup, String> {
        let (all, settings) = (app.settings()?, app.profile(name)?);
        let p = all.providers.get(&settings.provider).ok_or_else(|| format!("no provider {} in settings.json", settings.provider))?;
        let llm = Llm { base: p.base.clone(), key: p.key.clone(), agent: app.agent.clone(), spent: Default::default() };
        let tools = toolbox(app, settings.workspace.as_deref(), settings.sandbox.then_some(all.sandbox));
        let definitions = tools.definitions();
        Ok(Setup { overhead: overhead(settings.system.as_deref(), &definitions), settings, llm, tools, definitions })
    }
}

/// The summarizer's host: the session's model unless the library names one, and the agent's tools,
/// which the library sends with tool choice "none" so a memory's request can carry tool blocks; no
/// system prompt.
impl Host for &Setup {
    fn tools(&self) -> Option<&Value> { self.definitions.as_ref() }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.settings.model } else { &request.model };
        let none = request.tools.as_ref().map(|_| obj! {"type" => "none"});
        let head = sparse(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens, "tools" => request.tools.clone(), "tool_choice" => none});
        self.llm.messages(head, &request.messages, reader, |_, _| {})
    }
}

/// A session's tools: bash (unless --no-bash) and edit in its `workspace`, bash in `sandbox` if given,
/// both declared but failing without a workspace; then the MCP servers'.
pub fn toolbox(app: &App, workspace: Option<&Path>, sandbox: Option<Sandbox>) -> Toolbox {
    let why = "it has no workspace; the user can set one in the session's settings";
    let (bash, edit) = match workspace {
        Some(dir) => (bash::tool(dir.into(), app.limit, sandbox), edit::tool(dir.into())),
        None => (bash::unavailable(why), edit::unavailable(why)),
    };
    let bash = if app.bash { bash } else { Toolbox::default() };
    bash + edit + mcp::toolbox(&app.mcp)
}

/// The tokens a system prompt and tool definitions take.
pub fn overhead(system: Option<&str>, definitions: &Option<Value>) -> f64 {
    density_tokens(system.unwrap_or("")) + definitions.as_ref().map_or(0.0, |d| density_tokens(&json::to_string(d)))
}

/// A session's thread, which does its `jobs` (Worker). A job that panics (a file edited into something
/// the library can't read) fails, and the next runs.
pub fn work(app: &App, name: &str, jobs: Receiver<Job>) {
    for job in jobs {
        let done = catch_unwind(AssertUnwindSafe(|| Worker { app, name, s: app.session(name) }.run(job)));
        if done.is_err() {
            app.tell(name, Msg::Event(obj! {"forming" => false}));
            app.tell(name, Msg::Event(obj! {"error" => "the session's thread failed; the server's output says why"}));
        }
        app.tell(name, Msg::Done);
    }
}

/// A job in a session, as it reads the session's files again.
struct Worker<'a> {
    app: &'a App,
    name: &'a str,
    s: Session,
}

impl Worker<'_> {
    fn tell(&self, msg: Msg) { self.app.tell(self.name, msg) }

    /// An event for the pages following the job.
    fn event(&self, event: Value) { self.tell(Msg::Event(event)) }

    /// The pages told that the log has grown, and to what length.
    fn logged(&self) { self.tell(Msg::Logged(self.s.count("messages"))) }

    fn run(&self, job: Job) {
        match job {
            Job::Say { text, page } => {
                self.tell(Msg::Start(page));
                self.event(self.turn(text.as_deref()).map_or_else(|e| obj! {"error" => e}, |()| obj! {"ok" => true}));
                if let Ok(setup) = Setup::new(self.app, self.name) { self.form(&setup) } // the turn's memories
            }
            Job::Fork { at, name: fork, reply } => {
                let path = self.app.root.join(&fork); // claimed here, when the fork is made
                let made = std::fs::create_dir(&path).map_err(|e| format!("can't fork {} as {fork}: {e}", self.name));
                let _ = reply.send(made.and_then(|()| {
                    Session::fork(&self.s, at, path);
                    settings::write(&self.app.profile_path(&fork), &self.app.profile(self.name)?)
                }));
            }
        }
    }

    /// A chat turn: the message joins the log, then steps run until a reply calls no tools. Without a
    /// message, the turn continues the log as it is: after an interruption the context ends with the
    /// user's message or the tool results, and after a reply cut short with that reply, which the
    /// Messages API continues (a prefill). The pages hear:
    /// - the log's length each time it grows ({"logged"});
    /// - how the context folded before a reply ({"fold"}, costs::fold);
    /// - the pieces of a reply's reasoning and text ({"thinking"}, {"text"}), each call's tool as the
    ///   reply starts it and its input as it comes ({"calling"}, {"input"});
    /// - what the session's requests have used after each, and a reply's prompt as its provider
    ///   counted it ({"spent", "prompt"}, costs::spend);
    /// - the question whether each call may run ({"approve"}: its id; the hub tells the answers,
    ///   {"answered": the id, "answer"}), and each call's result ({"result"}: the block, brief);
    /// - memories forming ({"forming"}: true, then false).
    ///
    /// Then run tells the turn's end ({"ok": true} once it has joined the log, or {"error"}), and the
    /// memories after it form.
    fn turn(&self, text: Option<&str>) -> Result<(), String> {
        if let Some(text) = text {
            append(&self.s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &self.app.profile(self.name)?.config);
        }
        self.logged();
        for _ in 0..MAX_STEPS {
            if !self.step()? { return Ok(()) }
        }
        Err(format!("the turn stopped after {MAX_STEPS} steps"))
    }

    /// One step: the context compiles and streams into a request, the reply streams to the pages and
    /// joins the log, and the tools it calls run, their results joining the log; then memories form.
    /// Whether it called any.
    fn step(&self) -> Result<bool, String> {
        let (s, setup) = (&self.s, Setup::new(self.app, self.name)?);
        let (cfg, reserve) = (&setup.settings.config, setup.settings.reserve);
        // the context the last request carried, by its provider's count, unless a compile has learnt from it
        let (reported, last) = (costs::unlearnt(s).map(|prompt| prompt - setup.overhead), current(s));
        // how much bigger that context was than its compile thought, in a sample calibration learns
        // from: calibration takes steps to catch up, and the context compiles that much smaller meanwhile
        let lag = reported.map(|u| u / last.armed).filter(|r| (1.0..=1.8).contains(r)).unwrap_or(1.0);
        // what the context compiles to: the window less what every request carries and the reply, smaller by
        // the lag, so the request and its answer fit even at the library's grace over budget
        let w = (setup.settings.window - setup.overhead - reserve) / lag / (1.0 + cfg.over_budget_grace);
        let mut compiled = select(s, w, reported.map_or(last, |u| last.observe(u)), cfg);
        // a refusal escalates the session, so the memory the context waits for forms at once; then it compiles again, up to 3 times
        for _ in 0..3 {
            if compiled.is_ok() { break }
            self.form(&setup);
            compiled = select(s, w, current(s), cfg);
        }
        let turns = compiled.map_err(|r| format!("{}: {} tokens over a budget of {}", r.stage, r.actual, r.budget))?;
        if let Some(fold) = costs::fold(s) { self.event(obj! {"fold" => fold}) }
        let head = sparse(obj! {"model" => setup.settings.model.as_str(), "max_tokens" => reserve, "system" => setup.settings.system.as_deref(), "tools" => setup.definitions.clone()});
        let reply = setup.llm.messages(head, &turns, &mut Reader::new(s, false), |kind, piece| self.event(obj! {kind => piece}))?;
        self.spend(&setup, "reply");
        append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, cfg);
        self.logged();
        let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
        let results = calls.map(|call| self.approve(&setup, call).map_or_else(|why| tools::result(call, Err(why.into())), |()| setup.tools.call(call)));
        let results: Vec<Value> = results.inspect(|result| self.event(obj! {"result" => brief(result)})).collect();
        if results.is_empty() { return Ok(false) }
        append(s, &obj! {"participant" => "user", "content" => results}, cfg);
        self.logged();
        self.form(&setup);
        Ok(true)
    }

    /// Whether the user lets a call run. A tool that doesn't ask, or that the session always allows,
    /// runs at once (and a call to no tool fails at once); otherwise the hub asks the pages and decides
    /// by a page's answer. No answer within APPROVAL_WAIT is a denial.
    fn approve(&self, setup: &Setup, call: &Value) -> Result<(), String> {
        let (id, tool) = (call["id"].as_str().unwrap_or(""), call["name"].as_str().unwrap_or(""));
        if !setup.tools.asking().contains(&tool) || self.app.profile(self.name)?.allow.contains(tool) {
            return Ok(());
        }
        let (answer, answered) = channel();
        self.tell(Msg::Ask { id: id.into(), tool: tool.into(), answer });
        answered.recv_timeout(APPROVAL_WAIT).unwrap_or_else(|_| Err(DENIED.into()))
    }

    /// Memories form (memory::drive), the pages told while they do.
    fn form(&self, setup: &Setup) {
        self.event(obj! {"forming" => true});
        if let Some(error) = drive(&self.s, &mut &*setup, &setup.settings.config) {
            eprintln!("{}: forming a memory: {error}", self.name);
        }
        self.spend(setup, "memories");
        self.event(obj! {"forming" => false});
    }

    /// What the step's requests used since this was last told (Llm::spent), for `what`, joins the
    /// session's costs log, which the pages hear.
    fn spend(&self, setup: &Setup, what: &str) {
        if let Some(spent) = costs::spend(&self.s, what, setup.llm.spent.take()) { self.event(spent) }
    }
}
