//! The hub's thread. It owns what changes quickly, so each is told one thing at a time: settings.json
//! and the sessions' settings files, which sessions have a thread (started the first time one is told
//! something), and what the pages following each session's turn see (Live).

use crate::{costs::{Fold, Spent}, llm::Piece, settings::{self, SessionSettings, Settings}, work::{work, DENIED}, App};
use connectome::json::{self, Value};
use serde::Serialize;
use std::{collections::HashMap, sync::mpsc::{channel, Receiver, Sender}};

/// What the hub is told: a message about a session, or a change to settings.json, answered with the
/// settings it leaves.
pub enum Routed { To(String, Msg), Provide(Value, Sender<Result<Settings, String>>) }

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
    /// From the session's thread: an event for the pages following its turn.
    Event(Event),
    /// From the session's thread: a call's question, and where to answer it (Ok: it runs).
    Ask { id: String, tool: String, answer: Sender<Result<(), String>> },
    /// From the session's thread: the job is done, its memories formed; no call waits any more.
    Done,
}

/// What the pages following a turn hear, a JSON object a line, its one field the event's name: the
/// log's length each time it grows ({"logged": n}), then what the log doesn't hold yet.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Event {
    /// The log has grown to this many messages; the page reads them from disk.
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
    /// form after it, and the stream ends once they have.
    End(Option<String>),
    /// A piece of the reply streaming in: {"thinking"}, {"text"}, {"calling"} or {"input"}.
    #[serde(untagged)] Piece(Piece),
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

/// A session as the hub knows it: where its thread takes jobs, and its turn, while one runs (with its
/// memories forming after it).
struct Live { jobs: Sender<Job>, turn: Option<Turn> }

/// A turn as the pages following it see it: the events since the log last grew (the first its length),
/// the pages, and the calls waiting for an answer. What the log holds it drops, so it holds a step at
/// most.
struct Turn {
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
        Live { jobs, turn: None }
    }

    fn handle(&mut self, app: &App, name: &str, msg: Msg) {
        match (msg, &mut self.turn) {
            (Msg::Job(job), _) => drop(self.jobs.send(job)),
            (Msg::Status(reply), turn) => drop(reply.send(turn.is_some())),
            (Msg::Tune(v, merged, reply), _) => {
                let tune = |all: Settings| settings::change(&app.profile_path(name), |old: &mut SessionSettings| old.tuned(&v, merged, &all).map(|new| *old = new));
                let _ = reply.send(app.settings().and_then(tune).map(drop));
            }
            (Msg::Start(page), turn) => *turn = Some(Turn { events: vec![], pages: vec![page], asked: HashMap::new() }),
            (Msg::Done, turn) => *turn = None,
            (Msg::Follow(page), Some(turn)) => {
                if turn.events.iter().all(|line| page.send(line.clone()).is_ok()) {
                    turn.pages.push(page);
                }
            }
            (Msg::Approve { id, answer, reply }, Some(turn)) if turn.asked.contains_key(&id) => {
                let (tool, waiting) = turn.asked.remove(&id).expect("asked");
                let decided = match answer.as_str() {
                    "always" => settings::change(&app.profile_path(name), |s: &mut SessionSettings| { s.allow.insert(tool); Ok(()) }).map(drop),
                    "allow" => Ok(()),
                    _ => Err(DENIED.into()),
                };
                let _ = waiting.send(decided);
                turn.send(&Event::Answered(id));
                let _ = reply.send(Ok(()));
            }
            (Msg::Logged(n), Some(turn)) => {
                turn.events.clear();
                turn.send(&Event::Logged(n));
            }
            (Msg::Event(event), Some(turn)) => turn.send(&event),
            (Msg::Ask { id, tool, answer }, Some(turn)) => {
                turn.send(&Event::Approve(id.clone()));
                turn.asked.insert(id, (tool, answer));
            }
            (Msg::Approve { reply, .. }, _) => drop(reply.send(Err("no call waits for that answer".into()))),
            (Msg::Follow(_) | Msg::Logged(_) | Msg::Event(_) | Msg::Ask { .. }, None) => {} // no turn: a page's stream ends at once, a call is denied
        }
    }
}

impl Turn {
    /// An event to the pages following the turn, kept for those that come later.
    fn send(&mut self, event: &Event) {
        let line = format!("{}\n", json::to_string(event));
        self.pages.retain(|page| page.send(line.clone()).is_ok());
        self.events.push(line);
    }
}
