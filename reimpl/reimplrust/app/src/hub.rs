//! The hub's thread. It owns what changes quickly, so each is told one thing at a time: settings.json
//! and the sessions' settings files, which sessions have a thread (started the first time one is told
//! something), and what the pages following each session's turn see (Live).

use crate::{settings::{self, SessionSettings, Settings}, work::{work, DENIED}, App};
use connectome::{json::{self, Value}, obj};
use std::{collections::HashMap, sync::mpsc::{channel, Receiver, Sender}};

/// What the hub is told: a message about a session, or a change to settings.json, answered with the
/// settings it leaves.
pub enum Routed {
    To(String, Msg),
    Provide(Value, Sender<Result<Settings, String>>),
}

/// A job for a session's thread: a turn with a message, or without (None) one that continues the log,
/// which `page` follows; or a fork at message `at` as `name`.
pub enum Job {
    Say { text: Option<String>, page: Sender<String> },
    Fork { at: u64, name: String, reply: Sender<Result<(), String>> },
}

/// What the hub is told about a session, by requests' threads and by the session's.
pub enum Msg {
    /// A job for the session's thread, which does them in the order they come.
    Job(Job),
    /// A page that follows the job running; with none, its stream ends at once.
    Follow(Sender<String>),
    /// A page's answer whether call `id` may run: allow, always (its tool, from then on) or deny.
    Approve { id: String, answer: String, reply: Sender<Result<(), String>> },
    /// Whether a job runs.
    Status(Sender<bool>),
    /// A change to the session's settings (SessionSettings::tuned), and whether its memories have merged.
    Tune(Value, bool, Sender<Result<(), String>>),
    /// From the session's thread: a turn starts, which `page` follows.
    Start(Sender<String>),
    /// From the session's thread: the log has grown to this many messages.
    Logged(u64),
    /// From the session's thread: an event for the pages: a piece of a reply, a call's result, whether
    /// memories are forming, the turn's end.
    Event(Value),
    /// From the session's thread: a call's question, and where to answer it (Ok: it runs).
    Ask { id: String, tool: String, answer: Sender<Result<(), String>> },
    /// From the session's thread: the job is done, its memories formed; no call waits any more.
    Done,
}

pub fn hub(app: &'static App, inbox: Receiver<Routed>) {
    let mut sessions: HashMap<String, Live> = HashMap::new();
    for routed in inbox {
        match routed {
            Routed::To(name, msg) => sessions.entry(name.clone()).or_insert_with(|| Live::start(app, &name)).handle(app, &name, msg),
            Routed::Provide(v, reply) => drop(reply.send(settings::change(&app.settings_path(), |all: &mut Settings| all.provide(&v)))),
        }
    }
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

/// A session as the hub knows it: where its thread takes jobs, and what the pages following its job
/// see: whether one runs (a turn, then its memories forming), the events since the log last grew (the
/// first its length; then a reply streaming in, the questions whether the step's calls may run, the
/// answers, the calls' results, the memories forming, the turn's end), the pages, and the calls
/// waiting for an answer. What the log holds it drops, so it holds a step at most.
struct Live {
    jobs: Sender<Job>,
    running: bool,
    events: Vec<String>, // JSON lines
    pages: Vec<Sender<String>>,
    asked: HashMap<String, (String, Sender<Result<(), String>>)>, // by call id: its tool, where to answer
}

impl Live {
    /// A session, its thread started.
    fn start(app: &'static App, name: &str) -> Live {
        let (jobs, queue) = channel();
        let name = name.to_string();
        std::thread::spawn(move || work(app, &name, queue));
        Live { jobs, running: false, events: vec![], pages: vec![], asked: HashMap::new() }
    }

    fn handle(&mut self, app: &App, name: &str, msg: Msg) {
        match msg {
            Msg::Job(job) => drop(self.jobs.send(job)),
            Msg::Follow(page) => if self.running && self.events.iter().all(|line| page.send(line.clone()).is_ok()) { self.pages.push(page) },
            Msg::Approve { id, answer, reply } => {
                let Some((tool, waiting)) = self.asked.remove(&id) else { return drop(reply.send(Err("no call waits for that answer".into()))) };
                let decided = match answer.as_str() {
                    "always" => settings::change(&app.profile_path(name), |s: &mut SessionSettings| { s.allow.insert(tool); Ok(()) }).map(drop),
                    "allow" => Ok(()),
                    _ => Err(DENIED.into()),
                };
                let _ = waiting.send(decided);
                self.send(&obj! {"answered" => id, "answer" => answer});
                let _ = reply.send(Ok(()));
            }
            Msg::Status(reply) => drop(reply.send(self.running)),
            Msg::Tune(v, merged, reply) => {
                let tune = |all: Settings| settings::change(&app.profile_path(name), |old: &mut SessionSettings| old.tuned(&v, merged, &all).map(|new| *old = new));
                let _ = reply.send(app.settings().and_then(tune).map(drop));
            }
            Msg::Start(page) => (self.running, self.events, self.pages) = (true, vec![], vec![page]),
            Msg::Logged(n) => {
                self.events.clear();
                self.send(&obj! {"logged" => n});
            }
            Msg::Event(event) => self.send(&event),
            Msg::Ask { id, tool, answer } => {
                self.send(&obj! {"approve" => id.as_str()});
                self.asked.insert(id, (tool, answer));
            }
            Msg::Done => (self.running, self.pages, self.asked) = (false, vec![], HashMap::new()),
        }
    }

    /// An event to the pages following the job, kept for those that come later.
    fn send(&mut self, event: &Value) {
        let line = format!("{}\n", json::to_string(event));
        self.pages.retain(|page| page.send(line.clone()).is_ok());
        self.events.push(line);
    }
}
