//! The harness's pass (compaction-autobiographical's foldPass, compileFolds and driveTick) over a session
//! on disk. At each step boundary:
//!   1. each new message joins the log, and the chunks that can close do (append);
//!   2. the last step's real prompt size calibrates the price multiplier (observe);
//!   3. the context compiles at the budget less the response reserve; a refusal for size is retried once
//!      at that size if it plus the reserve fits the routed window (compile);
//!   4. memories form until one makes a call or none is due (drive).

use crate::chunking::close_chunks;
use crate::config::Config;
use crate::emit::{select, Refusal};
use crate::estimate::Calibration;
use crate::json::Value;
pub use crate::memory::drive;
use crate::memory::Host;
use crate::obj;
use crate::store::Session;
use crate::turns::Turn;

/// The messages the log gained, the last step's usage (None without a new report), the compile and
/// the memories. Returns the compiled turns, whose blocks an emit::Reader reads as they're written
/// out, or the compile's refusal; and the error a failed call gave, if one did.
#[allow(clippy::too_many_arguments)]
pub fn step(session: &Session, messages: &[Value], usage: Option<f64>, budget: f64, reserve: f64, window: f64, host: &mut dyn Host, cfg: &Config)
    -> (Result<Vec<Turn>, Refusal>, Option<String>) {
    messages.iter().for_each(|message| append(session, message, cfg));
    let calibration = usage.into_iter().fold(current(session), Calibration::observe);
    let compiled = compile(session, budget, reserve, window, calibration, cfg);
    (compiled, drive(session, host, cfg))
}

pub fn append(session: &Session, message: &Value, cfg: &Config) {
    let record = obj! {"participant" => message["participant"].clone(), "content" => message["content"].clone(), "source" => message["id"].clone()};
    session.append("messages", record);
    close_chunks(session, current(session).multiplier, cfg);
}

/// The calibration the last compile left.
pub fn current(session: &Session) -> Calibration {
    let last = session.last("compiles").unwrap_or(obj! {});
    Calibration { multiplier: last["multiplier"].as_f64().unwrap_or(1.0), armed: last["armed"].as_f64().unwrap_or(0.0) }
}

/// compileFolds: maxTokens `budget`, reserveForResponse `reserve`; a refusal for size whose total
/// fits `window` is retried once at that total.
pub fn compile(session: &Session, budget: f64, reserve: f64, window: f64, calibration: Calibration, cfg: &Config) -> Result<Vec<Turn>, Refusal> {
    match crate::emit::select(session, budget - reserve, calibration, cfg) {
        Err(r) if r.actual > r.budget && r.actual + reserve <= window => select(session, (r.actual + reserve) - reserve, calibration, cfg),
        result => result,
    }
}
