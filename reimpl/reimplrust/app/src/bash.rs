//! The bash tool: a command runs in a new shell (bash; cmd on Windows) in the working directory, its
//! output, stdout and stderr together in order, going to a file rather than a pipe, so a process it
//! leaves in the background can't hold the call open. The model gets the output, the middle cut out
//! past a cap, and the exit code. A command still running at its timeout is killed, with every process
//! it started that hasn't left its process group (on Unix).

use crate::tools::{Outcome, Tool, Toolbox};
use connectome::json::Value;
use connectome::obj;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const KEPT: u64 = 12_000; // bytes of output kept from each end
const DEFAULT_TIMEOUT: f64 = 120.0;
const SHELL: [&str; 2] = if cfg!(windows) { ["cmd", "/C"] } else { ["bash", "-c"] };

pub fn tool(dir: PathBuf, limit: Duration) -> Toolbox {
    let description = format!("Run a command in {} in {} and get its output (stdout and stderr together; the middle is cut out past {} characters) \
        and exit code. Each call starts a new shell, so cd and variables don't carry over. Background processes keep running.", SHELL[0], dir.display(), 2 * KEPT);
    let timeout = format!("Seconds before the command is killed (default {DEFAULT_TIMEOUT}, at most {}).", limit.as_secs());
    let properties = obj! {"command" => obj! {"type" => "string"}, "timeout" => obj! {"type" => "number", "description" => timeout}};
    let definition = obj! {"name" => "bash", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["command"]}};
    Toolbox(vec![Arc::new(Tool { definition, run: Box::new(move |input| run(&dir, input, limit)) })])
}

fn run(dir: &Path, input: &Value, limit: Duration) -> Outcome {
    let command = input["command"].as_str().ok_or("bash needs a command")?;
    let timeout = Duration::from_secs_f64(input["timeout"].as_f64().unwrap_or(DEFAULT_TIMEOUT).clamp(0.1, limit.as_secs_f64()));
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!("connectome-bash-{}-{}.out", std::process::id(), CALLS.fetch_add(1, Ordering::Relaxed)));
    let out = File::create(&path).map_err(|e| format!("bash: {e}"))?;
    let mut shell = Command::new(SHELL[0]);
    shell.args([SHELL[1], command]).current_dir(dir).stdin(Stdio::null()).stderr(out.try_clone().map_err(|e| format!("bash: {e}"))?).stdout(out);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut shell, 0);
    let mut child = shell.spawn().map_err(|e| format!("bash: {e}"))?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            done => break done.ok().flatten(),
        }
    };
    if status.is_none() {
        #[cfg(unix)]
        unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) }; // its process group: the command and what it started
        let _ = child.kill();
        let _ = child.wait();
    }
    let output = read(&path);
    let _ = std::fs::remove_file(&path);
    match status.map(|s| s.code()) {
        Some(Some(code)) => Ok(format!("{output}\n[exit code {code}]").into()),
        Some(None) => Ok(format!("{output}\n[killed by a signal]").into()),
        None => Err(format!("{output}\n[killed after {} s]", timeout.as_secs_f64()).into()),
    }
}

/// A file's text, the middle cut out past twice KEPT bytes.
fn read(path: &Path) -> String {
    let Ok(mut file) = File::open(path) else { return String::new() };
    let size = file.metadata().map_or(0, |m| m.len());
    let mut part = |from: u64, len: u64| {
        let mut bytes = vec![];
        let _ = file.seek(SeekFrom::Start(from)).and_then(|_| file.by_ref().take(len).read_to_end(&mut bytes));
        String::from_utf8_lossy(&bytes).into_owned()
    };
    if size <= 2 * KEPT {
        return part(0, size);
    }
    let (head, tail) = (part(0, KEPT), part(size - KEPT, KEPT));
    format!("{head}\n[… {} bytes cut …]\n{tail}", size - 2 * KEPT)
}
