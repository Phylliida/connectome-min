//! The bash tool: a command runs in a new shell (bash; cmd on Windows) in the working directory, its
//! output, stdout and stderr together in order, going to a file rather than a pipe, so a process it
//! leaves in the background can't hold the call open. The model gets the output, the middle cut out
//! past a cap, and the exit code. A command still running at its timeout is killed, with every process
//! it started that hasn't left its process group (on Unix).
//!
//! In a sandbox (Linux's bubblewrap, `bwrap`), the command sees only its workspace, read-write, and
//! the paths the sandbox's settings name that exist, in its own PID namespace, which goes when bwrap is
//! killed, so a timeout still takes everything the command started.

use crate::settings::{expand, Sandbox};
use crate::tools::{Outcome, Tool, Toolbox};
use connectome::json::Value;
use connectome::obj;
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const KEPT: u64 = 12_000; // bytes of output kept from each end
const DEFAULT_TIMEOUT: f64 = 120.0;
const SHELL: [&str; 2] = if cfg!(windows) { ["cmd", "/C"] } else { ["bash", "-c"] };

pub fn tool(dir: PathBuf, limit: Duration, sandbox: Option<Sandbox>) -> Toolbox {
    let mut description = format!("Run a command in {} in {} and get its output (stdout and stderr together; the middle is cut out past {} characters) \
        and exit code. Each call starts a new shell, so cd and variables don't carry over. Background processes keep running.", SHELL[0], dir.display(), 2 * KEPT);
    if let Some(b) = &sandbox {
        let network = if b.network { "" } else { ", and has no network" };
        description += &format!(" It runs in a sandbox: outside its directory it can write only {:?} and read {:?}{network}.", b.rw, b.ro);
    }
    let timeout = format!("Seconds before the command is killed (default {DEFAULT_TIMEOUT}, at most {}).", limit.as_secs());
    let properties = obj! {"command" => obj! {"type" => "string"}, "timeout" => obj! {"type" => "number", "description" => timeout}};
    let definition = obj! {"name" => "bash", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["command"]}};
    Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |input| run(&dir, input, limit, sandbox.as_ref())) }])
}

/// bash where it can't run, declared all the same, so that a session whose history calls it still
/// declares it (as the Messages API requires, and so a call comes back as one, not as text); each
/// call fails at once, saying `why`, without asking.
pub fn unavailable(why: &'static str) -> Toolbox {
    let description = format!("Runs a command in {}, but not in this session: {why}.", SHELL[0]);
    let definition = obj! {"name" => "bash", "description" => description, "input_schema" => obj! {"type" => "object"}};
    Toolbox(vec![Tool { definition, asks: false, run: Box::new(move |_| Err(why.into())) }])
}

fn run(dir: &Path, input: &Value, limit: Duration, sandbox: Option<&Sandbox>) -> Outcome {
    let command = input["command"].as_str().ok_or("bash needs a command")?;
    let timeout = Duration::from_secs_f64(input["timeout"].as_f64().unwrap_or(DEFAULT_TIMEOUT).clamp(0.1, limit.as_secs_f64()));
    // a thread runs one command at a time, so its id makes the file this call's
    let path = std::env::temp_dir().join(format!("connectome-bash-{}-{:?}.out", std::process::id(), std::thread::current().id()));
    let out = File::create(&path).map_err(|e| format!("bash: {e}"))?;
    let mut shell = Command::new(if sandbox.is_some() { "bwrap" } else { SHELL[0] });
    if let Some(b) = sandbox {
        if !cfg!(target_os = "linux") {
            return Err("the sandbox is bubblewrap, which runs only on Linux".into());
        }
        shell.args(bwrap(dir, b)).arg(SHELL[0]);
    }
    shell.args([SHELL[1], command]).current_dir(dir).stdin(Stdio::null()).stderr(out.try_clone().map_err(|e| format!("bash: {e}"))?).stdout(out);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut shell, 0);
    let mut child = shell.spawn().map_err(|e| format!("{}: {e}", shell.get_program().to_string_lossy()))?;
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

/// bubblewrap's arguments up to the command: a new PID namespace and session, /proc and /dev, the
/// sandbox's paths that exist (read-only, then read-write), the workspace, and the network unless the
/// sandbox has none.
fn bwrap(dir: &Path, b: &Sandbox) -> Vec<OsString> {
    let fixed = ["--die-with-parent", "--new-session", "--unshare-pid", "--proc", "/proc", "--dev-bind", "/dev", "/dev"].map(OsString::from);
    let paths = b.ro.iter().map(|p| ("--ro-bind-try", p)).chain(b.rw.iter().map(|p| ("--bind-try", p)));
    let binds = paths.flat_map(|(how, p)| [how.into(), expand(p).into(), expand(p).into()]);
    let workspace = ["--bind".into(), dir.into(), dir.into(), "--chdir".into(), dir.into()];
    fixed.into_iter().chain(binds).chain(workspace).chain((!b.network).then(|| "--unshare-net".into())).chain(["--".into()]).collect()
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
