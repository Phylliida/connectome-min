//! The hub's thread. It owns what changes quickly, so each is told one thing at a time: settings.json
//! and the sessions' settings files, which sessions have a thread (started the first time one is told
//! something), and each session's pages and what they hear (Live).

use crate::{api::Shown, companion::Heard, costs::{Fold, Spent}, llm::Piece, settings::{self, SessionSettings, Settings}, work::{work, DENIED, STOPPED}, App};
use connectome::json::Value;
use serde::Serialize;
use std::{collections::HashMap, sync::mpsc::{channel, Receiver, SendError, Sender}};

/// What the hub is told: a message about a session; a change to settings.json, answered with the
/// settings it leaves; or Auto turned on or off for a session and its partner, answered with whether
/// the back-and-forth needs starting (auto).
pub enum Routed { To(String, Msg), Provide(Value, Sender<Result<Settings, String>>), Auto(String, bool, Sender<Result<bool, String>>) }

/// A job for a session's thread: a turn with a message, or without (None) one that continues the log;
/// what its partner tells it (companion.rs), which joins a turn running between its steps, else starts
/// one; or a fork at message `at` as `name`.
pub enum Job {
    Say { text: Option<String> },
    Hear(Heard),
    Fork { at: u64, name: String, reply: Sender<Result<(), String>> },
}

/// What the hub is told about a session, by requests' threads and by the session's.
pub enum Msg {
    /// A job for the session's thread, which does them in the order they come, numbered from 1 (the
    /// number answered) for as long as the server runs; but what its partner tells it while one of its
    /// calls waits for the answer goes to that call (answered 0).
    Job(Job, Sender<u64>),
    /// From the session's thread: a call waits for what the partner tells it next.
    Await(Sender<Heard>),
    /// The session's companion made (companion::make): the two are paired, Auto on; answered with
    /// whether the back-and-forth needs starting (no turn here has yet to hand over its end).
    Pair(String, Sender<Result<bool, String>>),
    /// From the session's thread: its turn has ended, and is to be handed over; answered with whether
    /// Auto is on now. The hub orders this and Auto's switching, so a turn's end goes to the partner
    /// once: by the hand-over, or by the back-and-forth's start (auto).
    Ended(Sender<bool>),
    /// A page that follows the session from now on, answered once it does: it hears the turn running
    /// so far first.
    Follow(Sender<Event>, Sender<()>),
    /// A page's answer whether call `id` may run: allow, always (its tool, from then on) or deny.
    Approve { id: String, answer: String, reply: Sender<Result<(), String>> },
    /// Whether a job runs.
    Status(Sender<bool>),
    /// A change to the session's settings (SessionSettings::tuned), and whether its memories have merged.
    Tune(Value, bool, Sender<Result<(), String>>),
    /// The user stops the turn running (work's Worker::stopped); answered with whether one ran.
    Stop(Sender<Result<(), String>>),
    /// From the session's thread: job `id`'s turn starts; while it runs the hub holds `stop`, and lets
    /// go of it to stop it.
    Start(u64, Sender<()>),
    /// From the session's thread: the log has grown to this many messages.
    Logged(u64),
    /// From the session's thread: an event for the pages.
    Event(Event),
    /// From the session's thread: a call's question, and where to answer it (Ok: it runs).
    Ask { id: String, tool: String, answer: Sender<Result<(), String>> },
    /// From the session's thread: job `id` is done, its memories formed; no call waits any more.
    Done(u64),
}

/// What a page following a session hears, a JSON object a line, its one field the event's name. Its
/// own thread (api::follow) tells it the log's messages and the session; the hub tells it each turn as
/// it goes: its start, the log's length each time it grows, what the log doesn't hold yet, its end.
#[derive(Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Event {
    /// A message of the log: the newest when the page starts following, then each as it joins.
    Message(Shown),
    /// The session (api's State): once the page has the log's newest messages, and after each job.
    State(Value),
    /// Why the session's files can't be read as they are (a hand edit the library can't read).
    Broken(String),
    /// Job `id`'s turn starts.
    Turn(u64),
    /// The log has grown to this many messages: the reply to come is the nth.
    Logged(u64),
    /// How the context folded for the reply to come (costs::fold).
    Fold(Fold),
    /// What the session's requests have used, after each reply and each round of memories.
    Spent(Spent),
    /// The reply's prompt in tokens, as its provider counted it.
    Prompt(f64),
    /// The call with this id waits for the user's answer whether it may run.
    Approve(String),
    /// A page has answered whether the call with this id may run.
    Answered(String),
    /// A call's result: the tool_result block, brief (api::brief).
    Result(Value),
    /// Memories start or stop forming.
    Forming(bool),
    /// The turn has ended: null once its reply has joined the log, else why it failed. Its memories
    /// form after it.
    End(Option<String>),
    /// Job `id` is done, its memories formed.
    Done(u64),
    /// A piece of the reply streaming in: {"thinking"}, {"text"}, {"calling"} or {"input"}.
    #[serde(untagged)] Piece(Piece),
}

pub fn hub(app: &'static App, inbox: Receiver<Routed>) {
    let mut sessions: HashMap<String, Live> = HashMap::new();
    for routed in inbox {
        match routed {
            Routed::To(name, msg) => sessions.entry(name.clone()).or_insert_with(|| Live::start(app, &name)).handle(app, &name, msg),
            Routed::Provide(v, reply) => drop(reply.send(settings::change(&app.settings_path(), |all: &mut Settings| all.provide(&v)))),
            Routed::Auto(name, on, reply) => drop(reply.send(auto(app, &sessions, &name, on))),
        }
    }
}

/// Auto turned on or off for `name` and its partner; whether the back-and-forth needs starting: Auto
/// on while no turn in either has yet to hand over its end (one that has will hand it over itself).
fn auto(app: &App, sessions: &HashMap<String, Live>, name: &str, on: bool) -> Result<bool, String> {
    let partner = app.profile(name)?.partner().map(str::to_string).ok_or(format!("{name} has no companion"))?;
    let set = |name: &str| settings::change(&app.profile_path(name), |s: &mut SessionSettings| { s.auto = on; Ok(()) });
    set(name)?;
    set(&partner)?;
    let quiet = |name: &str| sessions.get(name).is_none_or(|live| !live.turning());
    Ok(on && quiet(name) && quiet(&partner))
}

impl App {
    /// The answer to what the hub is told, which says where to answer.
    pub fn ask<T>(&self, told: impl FnOnce(Sender<T>) -> Routed) -> T {
        let (reply, answer) = channel();
        self.hub.send(told(reply)).expect("the hub runs");
        answer.recv().expect("the thread told answers")
    }

    /// The answer to what the hub is told about a session.
    pub fn ask_session<T>(&self, name: &str, msg: impl FnOnce(Sender<T>) -> Msg) -> T {
        self.ask(|reply| Routed::To(name.into(), msg(reply)))
    }

    /// Tells the hub something about a session that wants no answer.
    pub fn tell(&self, name: &str, msg: Msg) {
        self.hub.send(Routed::To(name.into(), msg)).expect("the hub runs");
    }
}

/// A session as the hub knows it: where its thread takes jobs and how many it has had, the pages
/// following it, and its turn while one runs (with its memories forming after it).
struct Live {
    jobs: Sender<(u64, Job)>,
    numbered: u64,
    pages: Vec<Sender<Event>>,
    turn: Option<Turn>,
    awaiting: Option<Sender<Heard>>, // a call waiting for what the partner tells it next
}

/// A turn as a page that comes in during it hears it first: its start, then the events since the log
/// last grew (what the log holds, the page reads from it); the calls waiting for an answer; the
/// session's thread's line, until the user stops the turn; and whether it has ended (Msg::Ended).
struct Turn {
    events: Vec<Event>,
    asked: HashMap<String, (String, Sender<Result<(), String>>)>, // by call id: its tool, where to answer
    stop: Option<Sender<()>>,
    ended: bool,
}

impl Live {
    /// A session, its thread started.
    fn start(app: &'static App, name: &str) -> Live {
        let (jobs, queue) = channel();
        let name = name.to_string();
        std::thread::spawn(move || work(app, &name, queue));
        Live { jobs, numbered: 0, pages: vec![], turn: None, awaiting: None }
    }

    fn handle(&mut self, app: &App, name: &str, msg: Msg) {
        match msg {
            Msg::Job(Job::Hear(heard), reply) if self.awaiting.is_some() => match self.awaiting.take().expect("a call waits").send(heard) {
                Ok(()) => drop(reply.send(0)),
                Err(SendError(heard)) => self.handle(app, name, Msg::Job(Job::Hear(heard), reply)), // the call has stopped waiting
            },
            Msg::Await(waiting) => self.awaiting = Some(waiting),
            Msg::Pair(companion, reply) => {
                let paired = settings::change(&app.profile_path(name), |s: &mut SessionSettings| match s.partner() {
                    Some(partner) => Err(format!("{name} is paired with {partner} already")),
                    None => Ok((s.companion, s.auto) = (Some(companion), true)),
                });
                let _ = reply.send(paired.map(|_| !self.turning()));
            }
            Msg::Ended(reply) => {
                if let Some(turn) = &mut self.turn {
                    turn.ended = true;
                }
                let _ = reply.send(app.profile(name).is_ok_and(|s| s.auto));
            }
            Msg::Job(job, reply) => {
                self.numbered += 1;
                let _ = self.jobs.send((self.numbered, job));
                let _ = reply.send(self.numbered);
            }
            Msg::Follow(page, followed) => {
                if self.turn.iter().flat_map(|t| &t.events).all(|e| page.send(e.clone()).is_ok()) {
                    self.pages.push(page);
                }
                let _ = followed.send(());
            }
            Msg::Status(reply) => drop(reply.send(self.turn.is_some())),
            Msg::Tune(v, merged, reply) => {
                let tune = |all: Settings| settings::change(&app.profile_path(name), |old: &mut SessionSettings| old.tuned(&v, merged, &all).map(|new| *old = new));
                let _ = reply.send(app.settings().and_then(tune).map(drop));
            }
            Msg::Start(id, stop) => {
                self.turn = Some(Turn { events: vec![], asked: HashMap::new(), stop: Some(stop), ended: false });
                self.send(Event::Turn(id));
            }
            Msg::Stop(reply) => {
                let Some(turn) = &mut self.turn else { return drop(reply.send(Err("no turn runs".into()))) };
                turn.stop = None;
                let asked: Vec<_> = turn.asked.drain().collect();
                for (id, (_, waiting)) in asked { // the calls waiting for an answer don't run
                    let _ = waiting.send(Err(STOPPED.into()));
                    self.send(Event::Answered(id));
                }
                let _ = reply.send(Ok(()));
            }
            Msg::Done(id) => {
                self.turn = None;
                self.send(Event::Done(id));
            }
            Msg::Approve { id, answer, reply } => match self.turn.as_mut().and_then(|t| t.asked.remove(&id)) {
                Some((tool, waiting)) => {
                    let decided = match answer.as_str() {
                        "always" => settings::change(&app.profile_path(name), |s: &mut SessionSettings| { s.allow.insert(tool); Ok(()) }).map(drop),
                        "allow" => Ok(()),
                        _ => Err(DENIED.into()),
                    };
                    let _ = waiting.send(decided);
                    self.send(Event::Answered(id));
                    let _ = reply.send(Ok(()));
                }
                None => drop(reply.send(Err("no call waits for that answer".into()))),
            },
            Msg::Logged(n) => {
                if let Some(turn) = &mut self.turn {
                    turn.events.truncate(1); // its start
                }
                self.send(Event::Logged(n));
            }
            Msg::Event(event) => self.send(event),
            Msg::Ask { id, tool, answer } => {
                if let Some(turn) = &mut self.turn {
                    turn.asked.insert(id.clone(), (tool, answer));
                    self.send(Event::Approve(id));
                } // without a turn the answer's sender goes, which denies the call
            }
        }
    }

    /// Whether a turn runs that has yet to hand over its end.
    fn turning(&self) -> bool { self.turn.as_ref().is_some_and(|t| !t.ended) }

    /// An event to the pages, kept for those that come later while a turn runs.
    fn send(&mut self, event: Event) {
        self.pages.retain(|page| page.send(event.clone()).is_ok());
        if let Some(turn) = &mut self.turn {
            turn.events.push(event);
        }
    }
}
