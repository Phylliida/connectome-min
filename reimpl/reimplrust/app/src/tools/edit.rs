//! The edit tool: a file in the session's workspace changed by replacing text in it, or made. Its paths
//! stay in the workspace: relative to it, or absolute within it, never out of it through `..` or a
//! link. In a sandbox it reads and writes through commands run in bash's (bubblewrap), so however a
//! path changes between its check and its use (another session's command can swap a directory for a
//! link), the kernel keeps it to what a sandboxed command may touch. A call asks first, and the page
//! shows it as a diff of its text, old to new.

use super::{bash, Outcome, Stopped, Tool, Toolbox, POLL};
use crate::settings::Sandbox;
use connectome::{json::Value, obj};
use std::{io::{Read, Write}, path::{Component, Path, PathBuf}, process::{Command, Stdio}, time::{Duration, Instant}};
use wait_timeout::ChildExt;

/// edit in `dir`, in `sandbox` if given (bubblewrap, so Linux only: elsewhere edit is unavailable), a
/// call's commands there killed past `limit`.
pub fn tool(dir: PathBuf, limit: Duration, sandbox: Option<Sandbox>) -> Toolbox {
    if sandbox.is_some() && !cfg!(target_os = "linux") {
        return unavailable("its sandbox is bubblewrap, which runs only on Linux");
    }
    let description = format!("Edit a file in {}: replace text in it, or make a new one. `old` is the exact text to replace, whitespace and all, \
        which must occur once in the file (or each time, with `all`), and `new` what goes in its place; with `old` empty, the file is made, \
        holding `new`, and mustn't exist yet. A path is relative to that directory.", dir.display());
    let text = |what: &str| obj! {"type" => "string", "description" => what};
    let all = obj! {"type" => "boolean", "description" => "Replace the text each time it occurs"};
    let properties = obj! {"path" => text("The file"), "old" => text("The text to replace; empty for a new file"), "new" => text("The text to put in its place"), "all" => all};
    let definition = obj! {"name" => "edit", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["path", "old", "new"]}};
    let files = Files { dir, limit, sandbox };
    Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |input, stopped| edit(&files, input, stopped)) }])
}

/// edit where it can't run (super::unavailable).
pub fn unavailable(why: &'static str) -> Toolbox {
    super::unavailable("edit", "Edits a file", why)
}

fn edit(files: &Files, input: &Value, stopped: Stopped) -> Outcome {
    let (path, old, new) = (input["path"].as_str().ok_or("edit needs a path")?, input["old"].as_str().unwrap_or(""), input["new"].as_str().unwrap_or(""));
    let file = inside(&files.dir, path)?;
    let failed = |e: String| Value::from(format!("{path}: {e}"));
    if old.is_empty() {
        if file.symlink_metadata().is_ok() {
            return Err(format!("{path} exists: give the text to replace in it").into());
        }
        files.write(&file, new, true, stopped).map_err(failed)?;
        return Ok(format!("made {path}, {} lines", new.lines().count()).into());
    }
    let text = files.read(&file, stopped).map_err(failed)?;
    let Some(at) = text.find(old) else { return Err(format!("the text to replace isn't in {path}").into()) };
    let (times, line) = (text.matches(old).count(), text[..at].matches('\n').count() + 1);
    if times > 1 && input["all"] != true {
        return Err(format!("the text to replace is in {path} {times} times: give more of it, around the change, or all").into());
    }
    files.write(&file, &text.replace(old, new), false, stopped).map_err(failed)?;
    Ok(if times == 1 { format!("edited {path} at line {line}") } else { format!("edited {path} in {times} places, the first at line {line}") }.into())
}

/// Where an edit reads and writes: here, or, given a sandbox, through commands run in it.
struct Files {
    dir: PathBuf,
    limit: Duration,
    sandbox: Option<Sandbox>,
}

impl Files {
    /// A file's text.
    fn read(&self, file: &Path, stopped: Stopped) -> Result<String, String> {
        let bytes = match &self.sandbox {
            None => std::fs::read(file).map_err(|e| e.to_string())?,
            Some(b) => self.run(b, r#"exec cat -- "$1""#, file, b"", stopped)?,
        };
        String::from_utf8(bytes).map_err(|_| "not text (UTF-8)".into())
    }

    /// `text` into a file, a `new` one (none there, and its directories made) or one there is.
    fn write(&self, file: &Path, text: &str, new: bool, stopped: Stopped) -> Result<(), String> {
        let Some(b) = &self.sandbox else {
            let made = if new { std::fs::create_dir_all(file.parent().unwrap_or(&self.dir)) } else { Ok(()) };
            return made.and_then(|()| std::fs::write(file, text)).map_err(|e| e.to_string());
        };
        let script = if new { r#"mkdir -p -- "$(dirname -- "$1")" && set -C && exec cat > "$1""# } else { r#"exec cat > "$1""# };
        self.run(b, script, file, text.as_bytes(), stopped).map(drop)
    }

    /// `script` run by sh in sandbox `b` (bash's, in the workspace), `file` its $1 and `input` its
    /// stdin: what it wrote to stdout, or to stderr if it failed. A stop or the time limit kills it.
    fn run(&self, b: &Sandbox, script: &str, file: &Path, input: &[u8], stopped: Stopped) -> Result<Vec<u8>, String> {
        let mut command = Command::new("bwrap");
        command.args(bash::bwrap(&self.dir, b)).args(["sh", "-c", script, "sh"]).arg(file);
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("bwrap: {e}"))?;
        let (mut stdin, mut stdout, mut stderr) = (child.stdin.take().expect("piped"), child.stdout.take().expect("piped"), child.stderr.take().expect("piped"));
        std::thread::scope(|scope| {
            scope.spawn(move || drop(stdin.write_all(input))); // the pipe closes as it ends
            let out = scope.spawn(move || { let mut out = vec![]; drop(stdout.read_to_end(&mut out)); out });
            let err = scope.spawn(move || { let mut err = String::new(); drop(stderr.read_to_string(&mut err)); err });
            let by = Instant::now() + self.limit;
            let status = loop {
                match child.wait_timeout(POLL) {
                    Ok(None) if !stopped() && Instant::now() < by => continue,
                    Ok(Some(status)) => break Some(status),
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break None;
                    }
                }
            };
            let (out, err) = (out.join().expect("its reader"), err.join().expect("its reader"));
            match status {
                Some(status) if status.success() => Ok(out),
                Some(_) => Err(err.trim().to_string()),
                None if stopped() => Err("stopped by the user".into()),
                None => Err(format!("timed out after {} s", self.limit.as_secs())),
            }
        })
    }
}

/// `path` in `dir`, or why it isn't: relative to it or absolute within it, with no `..`, and no link
/// on the way that leads out of it.
fn inside(dir: &Path, path: &str) -> Result<PathBuf, String> {
    let (file, real) = (dir.join(path), |p: &Path| p.canonicalize().map_err(|e| format!("{}: {e}", p.display())));
    let known = file.ancestors().find(|p| p.symlink_metadata().is_ok()).ok_or(format!("{path}: no such directory"))?; // the deepest part there is
    let resolved = real(known)?.join(file.strip_prefix(known).expect("an ancestor"));
    if Path::new(path).components().any(|c| c == Component::ParentDir) || !resolved.starts_with(real(dir)?) {
        return Err(format!("{path} is outside the workspace, {}", dir.display()));
    }
    Ok(file)
}
