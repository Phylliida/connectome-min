//! A session's thread: it does its jobs one at a time, in the order they come (a message or fork told
//! during a turn waits for it), the only one to write the session's log. A turn's chat message joins
//! the log, then the turn takes steps until a reply calls no tools, each reading the settings again:
//! the context compiles and streams from disk into the request, the reply joins the log, and the tools
//! it calls run, each once the user allows it, their results joining the log. Memories form between
//! steps and after the turn. The thread tells the hub what the pages following the job see (Msg).
//! The user can stop a turn: it ends where it is (Worker::stopped).

use crate::{api::brief, costs::{self, For}, hub::{Event, Job, Msg}, llm::Llm, settings::{self, Sandbox, SessionSettings}, tools::{self, bash, edit, mcp, Toolbox}, App};
use connectome::{blocks::density_tokens, drive::{append, current, drive}, emit::{select, Reader, Refusal}, json::{self, sparse, Json, Value}, obj};
use connectome::{memory::{Host, Request, INTERRUPTED}, store::Session};
use std::{panic::{catch_unwind, AssertUnwindSafe}, path::Path, sync::mpsc::{channel, Receiver, Sender, TryRecvError}, time::Duration};

/// Steps a turn may take, a model request each, before it stops.
const MAX_STEPS: usize = 64;

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// A denied call's result.
pub const DENIED: &str = "the user didn't allow this call";

/// A stopped turn's end, and the result of each of its calls that didn't run.
pub const STOPPED: &str = "Stopped by the user.";

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

/// The summarizer, the library's host: the session's model unless the library names one, and the
/// agent's tools, which the library sends with tool choice "none" so a memory's request can carry tool
/// blocks; no system prompt. A request its turn's user stops ends at once, as never made (INTERRUPTED).
struct Summarizer<'a> {
    setup: &'a Setup,
    stopped: tools::Stopped<'a>,
}

impl Host for Summarizer<'_> {
    fn tools(&self) -> Option<&Value> { self.setup.definitions.as_ref() }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.setup.settings.model } else { &request.model };
        let none = request.tools.as_ref().map(|_| obj! {"type" => "none"});
        let head = sparse(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens, "tools" => request.tools.clone(), "tool_choice" => none});
        let reply = self.setup.llm.messages(head, &request.messages, reader, |_| !(self.stopped)())?;
        if (self.stopped)() { Err(INTERRUPTED.into()) } else { Ok(reply) }
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

/// Why the context can't compile, as the user can do something about it: at its most folded it's
/// bigger than what the window leaves it, or (rarer) structural repair dropped the newest message.
fn refused(r: &Refusal, setup: &Setup) -> String {
    let s = &setup.settings;
    if r.actual <= r.budget {
        return format!("{} ({:.0} tokens, of {:.0})", r.stage, r.actual, r.budget);
    }
    format!("The context doesn't fit: folded as far as it can be, it takes {:.0} tokens, and the window leaves it {:.0} ({:.0}, less {:.0} for the reply and {:.0} for the tools and system prompt). \
        Raise the window, lower the reply reserve, or keep fewer recent tokens raw (config.recent_window_tokens, now {:.0}).", r.actual, r.budget, s.window, s.reserve, setup.overhead, s.config.recent_window_tokens)
}

/// The tokens a system prompt and tool definitions take.
pub fn overhead(system: Option<&str>, definitions: &Option<Value>) -> f64 {
    density_tokens(system.unwrap_or("")) + definitions.as_ref().map_or(0.0, |d| density_tokens(&json::to_string(d)))
}

/// A session's thread, which does its `jobs` (Worker). A job that panics (a file edited into something
/// the library can't read) fails, and the next runs.
pub fn work(app: &App, name: &str, jobs: Receiver<(u64, Job)>) {
    for (id, job) in jobs {
        let (stop, halt) = channel();
        let done = catch_unwind(AssertUnwindSafe(|| Worker { app, name, s: app.session(name), halt }.run(id, job, stop)));
        if done.is_err() {
            app.tell(name, Msg::Event(Event::Forming(false)));
            app.tell(name, Msg::Event(Event::End(Some("the session's thread failed; the server's output says why".into()))));
        }
        app.tell(name, Msg::Done(id));
    }
}

/// A job in a session, as it reads the session's files again; and where it hears that the user has
/// stopped its turn: the hub holds the other end while the turn runs, and lets go of it to stop it.
struct Worker<'a> {
    app: &'a App,
    name: &'a str,
    s: Session,
    halt: Receiver<()>,
}

impl Worker<'_> {
    fn tell(&self, msg: Msg) { self.app.tell(self.name, msg) }

    /// An event for the pages following the job.
    fn event(&self, event: Event) { self.tell(Msg::Event(event)) }

    /// The pages told that the log has grown, and to what length.
    fn logged(&self) { self.tell(Msg::Logged(self.s.count("messages"))) }

    /// Whether the user has stopped the turn: then a reply streaming in stops where it is (and joins
    /// the log so), a call not yet run doesn't run (STOPPED is its result), one running ends (its
    /// tool's run hears so), a memory forming is dropped, and no step comes after.
    fn stopped(&self) -> bool { self.halt.try_recv() == Err(TryRecvError::Disconnected) }

    /// Job `id`; `stop` the hub holds while its turn runs.
    fn run(&self, id: u64, job: Job, stop: Sender<()>) {
        match job {
            Job::Say { text } => {
                self.tell(Msg::Start(id, stop));
                self.event(Event::End(self.turn(text.as_deref()).err()));
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
    /// Messages API continues (a prefill). The pages hear it as it goes (hub's Event); then run tells
    /// them the turn's end, and the memories after it form.
    fn turn(&self, text: Option<&str>) -> Result<(), String> {
        if let Some(text) = text {
            append(&self.s, &obj! {"participant" => "user", "content" => vec![obj! {"type" => "text", "text" => text}]}, &self.app.profile(self.name)?.config);
        }
        self.logged();
        for _ in 0..MAX_STEPS {
            if self.stopped() { return Err(STOPPED.into()) }
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
        if self.stopped() { return Err(STOPPED.into()) } // before the request
        let turns = compiled.map_err(|r| refused(&r, &setup))?;
        if let Some(fold) = costs::fold(s) { self.event(Event::Fold(fold)) }
        let head = sparse(obj! {"model" => setup.settings.model.as_str(), "max_tokens" => reserve, "system" => setup.settings.system.as_deref(), "tools" => setup.definitions.clone()});
        let reply = setup.llm.messages(head, &turns, &mut Reader::new(s, false), |piece| {
            self.event(Event::Piece(piece));
            !self.stopped()
        })?;
        self.spend(&setup, For::Reply);
        append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, cfg);
        self.logged();
        let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
        let run = |call| self.approve(&setup, call).map_or_else(|why| tools::result(call, Err(why.into())), |()| setup.tools.call(call, &|| self.stopped()));
        let results = calls.map(|call| if self.stopped() { tools::result(call, Err(STOPPED.into())) } else { run(call) });
        let results: Vec<Value> = results.inspect(|result| self.event(Event::Result(brief(result.clone())))).collect();
        let called = !results.is_empty();
        if called {
            append(s, &obj! {"participant" => "user", "content" => results}, cfg);
            self.logged();
        }
        if self.stopped() { return Err(STOPPED.into()) }
        if !called { return Ok(false) }
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

    /// Memories form (memory::drive), the pages told while they do; none once the turn is stopped.
    fn form(&self, setup: &Setup) {
        if self.stopped() { return }
        self.event(Event::Forming(true));
        if let Some(error) = drive(&self.s, &mut Summarizer { setup, stopped: &|| self.stopped() }, &setup.settings.config) {
            eprintln!("{}: forming a memory: {error}", self.name);
        }
        self.spend(setup, For::Memories);
        self.event(Event::Forming(false));
    }

    /// What the step's requests used since this was last told (Llm::spent), for `what`, joins the
    /// session's costs log; the pages hear the totals, and a reply's prompt.
    fn spend(&self, setup: &Setup, what: For) {
        let used = setup.llm.spent.take();
        if let Some(spent) = costs::spend(&self.s, what, used) {
            self.event(Event::Spent(spent));
            if what == For::Reply { self.event(Event::Prompt(used.input)) }
        }
    }
}
