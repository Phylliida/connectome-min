//! A session's thread: it does its jobs one at a time, in the order they come (a message or fork told
//! during a turn waits for it, but what the session's partner tells it joins the turn), the only one
//! to write the session's log. A turn's chat message joins the log, then the turn takes steps until a
//! reply calls no tools and the partner has told it nothing more, each reading the settings again:
//! the context compiles and streams from disk into the request, the reply joins the log, and the tools
//! it calls run, each once the user allows it, their results joining the log. The turn's end goes to
//! the partner (companion.rs), then memories form; they form between steps too. The thread tells the
//! hub what the pages following the job see (Msg). The user can stop a turn: it ends where it is
//! (Worker::stopped).

use crate::{api::brief, companion::{self, Heard, How}, costs::{self, For}, hub::{Event, Job, Msg, Routed}, llm::{Llm, RETRIES}, settings::{self, Sandbox, SessionSettings}, tools::{self, bash, edit, mcp, search, Toolbox}, App};
use connectome::{blocks::density_tokens, drive::{append, current, drive}, emit::{select, Fit, Reader, Refusal}, json::{self, sparse, Json, Value}, obj};
use connectome::{memory::{stalled, Host, Request, INTERRUPTED}, store::Session};
use std::{cell::RefCell, collections::VecDeque, panic::{catch_unwind, AssertUnwindSafe}, path::Path, sync::mpsc::{channel, Receiver, Sender, TryRecvError}, time::Duration};

/// A message from `who` of `text`, as the log keeps it.
fn said(who: &str, text: String) -> Value { obj! {"participant" => who, "content" => vec![obj! {"type" => "text", "text" => text}]} }

/// How long a call waits for the user's answer before it's denied.
const APPROVAL_WAIT: Duration = Duration::from_secs(600);

/// A denied call's result.
pub const DENIED: &str = "the user didn't allow this call";

/// A stopped turn's end, and the result of each of its calls that didn't run.
pub const STOPPED: &str = "Stopped by the user.";

/// The result of a call whose step the server stopped in, its reply logged but not its results.
const UNFINISHED: &str = "No result: the server stopped before this call finished (while it ran, or waited for the user to allow it). It may or may not have run.";

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
        let tools = toolbox(app, settings.workspace.as_deref(), settings.sandbox.then_some(all.sandbox), all.search.as_deref()) + companion::tools(app, name, &settings);
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
    ended: Option<(String, f64)>, // how its last reply ended, and the tokens it had
}

/// What forming memories came to: one formed (or a stub), none could, a reply without a memory in it or
/// a refusal (why), or a call that failed (why).
enum Formed {
    Memory,
    Nothing,
    Unused(String),
    Failed(String),
}

impl Host for Summarizer<'_> {
    fn tools(&self) -> Option<&Value> { self.setup.definitions.as_ref() }

    fn complete(&mut self, request: &Request, reader: &mut Reader) -> Result<Value, String> {
        let model = if request.model.is_empty() { &self.setup.settings.model } else { &request.model };
        let none = request.tools.as_ref().map(|_| obj! {"type" => "none"});
        let head = sparse(obj! {"model" => model.as_str(), "max_tokens" => request.max_tokens, "tools" => request.tools.clone(), "tool_choice" => none});
        let reply = self.setup.llm.messages(head, &request.messages, reader, |_| !(self.stopped)(), self.stopped, |_, _| {});
        self.ended = reply.as_ref().ok().map(|r| (r["stopReason"].as_str().unwrap_or("nothing").to_string(), request.max_tokens));
        if (self.stopped)() { Err(INTERRUPTED.into()) } else { reply }
    }
}

/// A session's tools: bash (unless --no-bash) and edit in its `workspace`, bash in `sandbox` if given,
/// both declared but failing without a workspace; web_search, if settings.json names a SearXNG server
/// (`searching`); then the MCP servers'.
pub fn toolbox(app: &App, workspace: Option<&Path>, sandbox: Option<Sandbox>, searching: Option<&str>) -> Toolbox {
    let why = "it has no workspace; the user can set one in the session's settings";
    let (bash, edit) = match workspace {
        Some(dir) => (bash::tool(dir.into(), app.limit, sandbox.clone()), edit::tool(dir.into(), app.limit, sandbox)),
        None => (bash::unavailable(why), edit::unavailable(why)),
    };
    let bash = if app.bash { bash } else { Toolbox::default() };
    let web = searching.map_or_else(Toolbox::default, |base| search::tool(base.into(), app.agent.clone(), app.limit));
    bash + edit + web + mcp::toolbox(&app.mcp)
}

/// Why the context can't compile, as the user can do something about it: at its most folded it's
/// bigger than what the window leaves it, or (rarer) structural repair dropped the newest message.
fn refused(r: &Refusal, setup: &Setup) -> String {
    let s = &setup.settings;
    if r.actual <= r.budget {
        return format!("{} ({:.0} tokens, of {:.0})", r.stage, r.actual, r.budget);
    }
    let ((memories, remembered), (unformed, waiting)) = (r.memories, r.unformed);
    let (memories, unformed) = (if memories == 1 { "1 memory".into() } else { format!("{memories} memories") }, if unformed == 1 { "1 chunk".into() } else { format!("{unformed} chunks") });
    format!("The context doesn't fit: folded as far as it can be, it takes {:.0} tokens, and the window leaves it {:.0} ({:.0}, less {:.0} for the reply and {:.0} for the tools and system prompt). \
        It holds {memories} ({remembered:.0} tokens), {unformed} raw that have no memory yet ({waiting:.0}), {:.0} tokens of messages not yet in a chunk (one closes at config.target_chunk_tokens, now {:.0}, of text), \
        and the {:.0} most recent kept raw (the raw tail, config.recent_window_tokens, now {:.0}). Raise the window, lower the reply reserve, or make chunks or the recent window smaller.",
        r.actual, r.budget, s.window, s.reserve, setup.overhead, r.open, s.config.target_chunk_tokens, r.tail, s.config.recent_window_tokens)
}

/// The tokens a system prompt and tool definitions take.
pub fn overhead(system: Option<&str>, definitions: &Option<Value>) -> f64 {
    density_tokens(system.unwrap_or("")) + definitions.as_ref().map_or(0.0, |d| density_tokens(&json::to_string(d)))
}

/// A session's thread, which does its `jobs` (Worker), and first those a turn took from them but left
/// for after it (`later`). A job that panics (a file edited into something the library can't read)
/// fails, and the next runs.
pub fn work(app: &App, name: &str, jobs: Receiver<(u64, Job)>) {
    let later = RefCell::new(VecDeque::new());
    loop {
        let next = later.borrow_mut().pop_front();
        let Some((id, job)) = next.or_else(|| jobs.recv().ok()) else { return };
        let (stop, halt) = channel();
        let done = catch_unwind(AssertUnwindSafe(|| Worker { app, name, s: app.session(name), halt, jobs: &jobs, later: &later }.run(id, job, stop)));
        if done.is_err() {
            app.tell(name, Msg::Event(Event::Forming(false)));
            app.tell(name, Msg::Event(Event::End(Some("the session's thread failed; the server's output says why".into()))));
        }
        app.tell(name, Msg::Done(id));
    }
}

/// A job in a session, as it reads the session's files again; where it hears that the user has
/// stopped its turn (the hub holds the other end while the turn runs, and lets go of it to stop it);
/// and the session's jobs, from which a turn takes what the partner tells it, the rest left for later.
struct Worker<'a> {
    app: &'a App,
    name: &'a str,
    s: Session,
    halt: Receiver<()>,
    jobs: &'a Receiver<(u64, Job)>,
    later: &'a RefCell<VecDeque<(u64, Job)>>,
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
            Job::Say { text } => self.chat(id, stop, text.map(|text| said("user", text))),
            Job::Hear(Heard { from, text }) => self.chat(id, stop, Some(said(&from, text))),
            Job::Fork { at, name: fork, reply } => {
                let path = self.app.root.join(&fork); // claimed here, when the fork is made
                let made = std::fs::create_dir(&path).map_err(|e| format!("can't fork {} as {fork}: {e}", self.name));
                let _ = reply.send(made.and_then(|()| {
                    Session::fork(&self.s, at, path);
                    let own = self.app.profile(self.name)?; // a fork has no companion of its own
                    settings::write(&self.app.profile_path(&fork), &SessionSettings { companion: None, accompanies: None, auto: false, ..own })
                }));
            }
        }
    }

    /// A turn from message `said` (or none), as the pages hear it: its start, the turn (hub's Event),
    /// its end, which then goes to the session's partner (hand_over); then the memories after it form.
    fn chat(&self, id: u64, stop: Sender<()>, said: Option<Value>) {
        self.tell(Msg::Start(id, stop));
        let start = self.s.count("messages");
        let ended = self.turn(said);
        let auto = self.app.ask_session(self.name, Msg::Ended);
        self.event(Event::End(ended.clone().err()));
        self.hand_over(start, ended, auto);
        if let Ok(setup) = Setup::new(self.app, self.name) { self.form(&setup); }
    }

    /// A chat turn: the log mended if the server stopped in a step (mend), its message joins the log,
    /// then steps run, as many as it takes, until a reply calls no tools and the partner has told the
    /// session nothing more (deliver), or the user stops it. Without a message, the turn continues the
    /// log as it is: after an interruption the context ends with the user's message or the tool
    /// results, and after a reply cut short with that reply, which the Messages API continues (a
    /// prefill).
    fn turn(&self, said: Option<Value>) -> Result<(), String> {
        self.mend()?;
        if let Some(said) = said {
            append(&self.s, &said, &self.app.profile(self.name)?.config);
        }
        self.deliver()?;
        self.logged();
        loop {
            if self.stopped() { return Err(STOPPED.into()) }
            let called = self.step()?;
            if !self.deliver()? && !called { return Ok(()) }
        }
    }

    /// A step the server stopped in, its reply's calls logged but not their results: each call gets a
    /// failed result (UNFINISHED), so the log answers every call before a turn goes on from it.
    fn mend(&self) -> Result<(), String> {
        let n = self.s.count("messages");
        let last = if n > 0 { self.s.read("messages", n - 1) } else { Value::Null };
        let calls = last["content"].arr().iter().filter(|b| last["participant"] == "assistant" && b.kind() == "tool_use");
        let results: Vec<Value> = calls.map(|call| tools::result(call, Err(UNFINISHED.into()))).collect();
        if !results.is_empty() {
            append(&self.s, &obj! {"participant" => "user", "content" => results}, &self.app.profile(self.name)?.config);
        }
        Ok(())
    }

    /// What the session's partner has told it meanwhile joins the log, each as the partner's message;
    /// the session's other jobs wait for the turn to end. Whether the partner told it anything.
    fn deliver(&self) -> Result<bool, String> {
        let mut heard = false;
        while let Ok((id, job)) = self.jobs.try_recv() {
            match job {
                Job::Hear(Heard { from, text }) => {
                    append(&self.s, &said(&from, text), &self.app.profile(self.name)?.config);
                    heard = true;
                }
                job => self.later.borrow_mut().push_back((id, job)),
            }
        }
        if heard { self.logged() }
        Ok(heard)
    }

    /// The turn that started at message `start` and `ended` so, to the session's partner, if it has
    /// one: if the turn ended by itself, what it said and ran, if `auto` (Auto on as it ended) or the
    /// partner is owed it (companion::owed); else Auto goes off for both, and the partner, if owed an
    /// answer, hears why there's none.
    fn hand_over(&self, start: u64, ended: Result<(), String>, auto: bool) {
        let Ok(settings) = self.app.profile(self.name) else { return };
        let Some(partner) = settings.partner() else { return };
        let n = self.s.count("messages");
        let turn: Vec<Value> = if n > start { self.s.span("messages", start, n - 1).collect() } else { vec![] };
        let told = match ended {
            Ok(()) => companion::ended(self.name, &turn),
            Err(why) => {
                if auto {
                    let _ = self.app.ask(|reply| Routed::Auto(self.name.into(), false, reply));
                }
                if !companion::owed(&turn, partner) { return }
                format!("(its turn ended before it could answer: {why})")
            }
        };
        if auto || companion::owed(&turn, partner) {
            self.app.ask_session(partner, |numbered| Msg::Job(Job::Hear(companion::heard(self.name, How::Ended, &told)), numbered));
        }
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
        // what the context may take: the window less what every request carries and the reply, smaller by the
        // lag, so the request and its answer fit even at the library's grace over budget
        let room = (setup.settings.window - setup.overhead - reserve) / lag / (1.0 + cfg.over_budget_grace);
        // past fold_above (a prompt, so less what every request carries, and the lag) it folds, down to fold_to
        let context = |prompt: f64| ((prompt - setup.overhead) / lag).min(room);
        let over = setup.settings.fold_above.map_or(room, context);
        let to = setup.settings.fold_to.map_or(over * (1.0 - cfg.slack), |to| context(to).min(over));
        let fit = Fit { over, to, limit: (room * (1.0 + cfg.over_budget_grace)).floor(), reach: true };
        let mut compiled = select(s, &fit, reported.map_or(last, |u| last.observe(u)), cfg);
        // a context that folds (past fold_above) short of fold_to is refused, which escalates the session, so the
        // memories it waits for form at once, as many as it takes: while one forms, and while memories can make
        // the room (the messages no chunk holds yet and the tail fit without them); if they can't, it goes as it
        // is, if the window takes it
        // (a reply without a memory is tried again, attempt_limit times since the log grew, and then a lift goes
        // past it; a call that fails would fail again)
        let mut failed = None;
        while let Err(r) = &compiled && r.actual > r.budget && r.open + r.tail <= r.budget {
            match self.form(&setup) {
                Formed::Memory => compiled = select(s, &fit, current(s), cfg),
                Formed::Unused(why) => failed = Some(why),
                Formed::Nothing => break,
                Formed::Failed(why) => {
                    failed = Some(why);
                    break;
                }
            }
        }
        if let Err(r) = &compiled && r.actual > r.budget {
            compiled = select(s, &Fit { reach: false, ..fit }, current(s), cfg);
        }
        if self.stopped() { return Err(STOPPED.into()) } // before the request
        let explained = |r: Refusal| refused(&r, &setup) + &failed.or_else(|| stalled(s, cfg)).map_or(String::new(), |why| format!(" A memory it waits for doesn't form: {why}."));
        let turns = compiled.map_err(explained)?;
        if let Some(fold) = costs::fold(s) { self.event(Event::Fold(fold)) }
        let head = sparse(obj! {"model" => setup.settings.model.as_str(), "max_tokens" => reserve, "system" => setup.settings.system.as_deref(), "tools" => setup.definitions.clone()});
        let piece = |piece| {
            self.event(Event::Piece(piece));
            !self.stopped()
        };
        let retrying = |n, why: &str| self.event(Event::Retry { n, of: RETRIES.len(), why: why.into() });
        let reply = setup.llm.messages(head, &turns, &mut Reader::new(s, false), piece, &|| self.stopped(), retrying);
        let reply = reply.map_err(|why| if self.stopped() { STOPPED.into() } else { why })?;
        self.spend(&setup, For::Reply);
        append(s, &obj! {"participant" => "assistant", "content" => reply["content"].clone()}, cfg);
        self.logged();
        let calls = reply["content"].arr().iter().filter(|b| b.kind() == "tool_use");
        let run = |call: &Value| self.approve(&setup, call).map_or_else(|why| tools::result(call, Err(why.into())), |()| {
            self.event(Event::Running { id: call["id"].as_str().unwrap_or("").into(), tool: call["name"].as_str().unwrap_or("").into() });
            setup.tools.call(call, &|| self.stopped())
        });
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
    fn form(&self, setup: &Setup) -> Formed {
        if self.stopped() { return Formed::Nothing }
        self.event(Event::Forming(true));
        let tried = self.s.count("attempts");
        let mut summarizer = Summarizer { setup, stopped: &|| self.stopped(), ended: None };
        let formed = match drive(&self.s, &mut summarizer, &setup.settings.config) {
            Err(why) => Formed::Failed(why),
            Ok(_) if self.s.count("attempts") > tried => Formed::Unused(match summarizer.ended {
                Some((ended, _)) if ended == "refusal" => "the summarizer refused to write it, and memories stop forming until that's mended".into(),
                Some((ended, max)) if ended == "max_tokens" => format!("the summarizer's reply ran out of its {max} tokens, its reasoning included, before it wrote the memory (config.compression_max_tokens)"),
                ended => format!("the summarizer's reply ended ({}) without a memory in it", ended.map_or("nothing".into(), |(ended, _)| ended)),
            }),
            Ok(true) => Formed::Memory,
            Ok(false) => Formed::Nothing,
        };
        if let Formed::Unused(why) | Formed::Failed(why) = &formed {
            eprintln!("{}: forming a memory: {why}", self.name);
        }
        self.spend(setup, For::Memories);
        self.event(Event::Forming(false));
        formed
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
