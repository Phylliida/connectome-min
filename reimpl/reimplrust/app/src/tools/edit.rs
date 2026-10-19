//! The edit tool: a file in the session's workspace changed by replacing text in it, or made. Its paths
//! stay in the workspace: relative to it, or absolute within it, never out of it through `..` or a
//! link. A call asks first, and the page shows it as a diff of its text, old to new.

use super::{Outcome, Tool, Toolbox};
use connectome::{json::Value, obj};
use std::path::{Component, Path, PathBuf};

/// edit in `dir`.
pub fn tool(dir: PathBuf) -> Toolbox {
    let description = format!("Edit a file in {}: replace text in it, or make a new one. `old` is the exact text to replace, whitespace and all, \
        which must occur once in the file (or each time, with `all`), and `new` what goes in its place; with `old` empty, the file is made, \
        holding `new`, and mustn't exist yet. A path is relative to that directory.", dir.display());
    let text = |what: &str| obj! {"type" => "string", "description" => what};
    let all = obj! {"type" => "boolean", "description" => "Replace the text each time it occurs"};
    let properties = obj! {"path" => text("The file"), "old" => text("The text to replace; empty for a new file"), "new" => text("The text to put in its place"), "all" => all};
    let definition = obj! {"name" => "edit", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["path", "old", "new"]}};
    Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |input, _| edit(&dir, input)) }])
}

/// edit where it can't run (super::unavailable).
pub fn unavailable(why: &'static str) -> Toolbox {
    super::unavailable("edit", "Edits a file", why)
}

fn edit(dir: &Path, input: &Value) -> Outcome {
    let (path, old, new) = (input["path"].as_str().ok_or("edit needs a path")?, input["old"].as_str().unwrap_or(""), input["new"].as_str().unwrap_or(""));
    let file = inside(dir, path)?;
    let failed = |e: std::io::Error| Value::from(format!("{path}: {e}"));
    if old.is_empty() {
        if file.symlink_metadata().is_ok() {
            return Err(format!("{path} exists: give the text to replace in it").into());
        }
        std::fs::create_dir_all(file.parent().unwrap_or(dir)).and_then(|()| std::fs::write(&file, new)).map_err(failed)?;
        return Ok(format!("made {path}, {} lines", new.lines().count()).into());
    }
    let text = std::fs::read_to_string(&file).map_err(failed)?;
    let Some(at) = text.find(old) else { return Err(format!("the text to replace isn't in {path}").into()) };
    let (times, line) = (text.matches(old).count(), text[..at].matches('\n').count() + 1);
    if times > 1 && input["all"] != true {
        return Err(format!("the text to replace is in {path} {times} times: give more of it, around the change, or all").into());
    }
    std::fs::write(&file, text.replace(old, new)).map_err(failed)?;
    Ok(if times == 1 { format!("edited {path} at line {line}") } else { format!("edited {path} in {times} places, the first at line {line}") }.into())
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
