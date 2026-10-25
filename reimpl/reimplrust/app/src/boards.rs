//! Boards: kanban boards as files, which sessions work on with the tools they have. A board is
//! ROOT/.boards/NAME: board.json, its settings (About), and a folder per column, done/ always among
//! them and last, each holding a .md file per task: front matter with its title and owner, then what
//! it's for, then its comments under `## Comments`, each a line `- **who**: text` (the lines after one
//! continue it). A session with a workspace, and not a companion, can be on boards (its settings'
//! `boards`): each one's folder is bound into its sandbox, read-write, its edit tool reaches it, and its
//! system prompt says where it is and how it's laid out (note), so it makes, moves and comments on
//! tasks as files. The page reads the files as they are, and changes them through the hub (apply),
//! which writes a task whole (a new file, renamed into place) or appends a comment.
//!
//! A task's owner, if it's a session on the board, hears as a message when the task is handed to it
//! and when someone else comments on it, the message ending with how boards work and the board's
//! description. The hub keeps what it last saw of each board (Seen), and looks again after each step
//! of a session on it that ran tools, and after each change from the page, putting what changed down to
//! that session, or to the user (changes): so a session that takes a task itself isn't told of it. A
//! change made by hand outside the page is put down to whoever looks next.

use crate::{companion::Heard, settings::{self, SessionSettings}, App};
use connectome::{json::Value, obj};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Write, path::{Path, PathBuf}};

/// The column every board has, last.
pub const DONE: &str = "done";

/// A board's settings (board.json): its columns' order (done last, whatever this says), and its
/// description, which each message its sessions get of it ends with.
#[derive(Default, Serialize, Deserialize)]
pub struct About {
    #[serde(default)] pub columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub description: Option<String>,
}

/// A task as its file has it.
pub struct Task {
    pub file: String,
    pub column: String,
    pub title: String,
    pub owner: Option<String>,
    extra: Vec<String>, // the front matter's other lines, kept as they are
    pub description: String,
    pub comments: Vec<(String, String)>, // who, and what they said
}

impl App {
    fn boards_path(&self) -> PathBuf {
        let path = self.root.join(".boards");
        std::path::absolute(&path).unwrap_or(path)
    }

    /// A board's folder, absolute, as a session's sandbox binds it and its system prompt names it.
    pub fn board_path(&self, name: &str) -> PathBuf { self.boards_path().join(name) }

    pub fn board_exists(&self, name: &str) -> bool { self.board_path(name).is_dir() }

    /// A board's settings, as its file has them (none yet: the defaults).
    pub fn board_about(&self, name: &str) -> About { settings::read(&self.board_path(name).join("board.json")).unwrap_or_default() }
}

/// Whether a session with these settings can be on a board: it has a workspace, and isn't a companion.
fn eligible(s: &SessionSettings) -> bool { s.workspace.is_some() && s.accompanies.is_none() }

/// The boards a session with these settings is on, those there are, if it can be on any.
pub fn of(app: &App, s: &SessionSettings) -> Vec<String> {
    if !eligible(s) { return vec![] }
    s.boards.iter().filter(|b| app.board_exists(b)).cloned().collect()
}

/// The sessions on board `name`.
pub fn members(app: &App, name: &str) -> Vec<String> {
    app.profiles().into_iter().filter(|(_, s)| eligible(s) && s.boards.contains(name)).map(|(member, _)| member).collect()
}

/// A new board, named `name`, unless there's one, with the columns todo, doing and done.
pub fn make(app: &App, name: &str) -> Result<(), String> {
    std::fs::create_dir_all(app.boards_path()).map_err(|e| format!("can't make the boards' directory: {e}"))?;
    let dir = app.board_path(name);
    std::fs::create_dir(&dir).map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => format!("there's a board {name} already"),
        _ => format!("can't make {name}: {e}"),
    })?;
    for column in ["todo", "doing", DONE] {
        std::fs::create_dir(dir.join(column)).map_err(|e| format!("can't make {name}/{column}: {e}"))?;
    }
    settings::write(&dir.join("board.json"), &About { columns: vec!["todo".into(), "doing".into()], description: None })
}

/// A board's columns, in order: those its settings list that are there, then any other folder in it
/// (one a session made), then done.
pub fn columns(app: &App, name: &str) -> Vec<String> {
    let folders = std::fs::read_dir(app.board_path(name)).into_iter().flatten().flatten().filter(|e| e.path().is_dir());
    let there: Vec<String> = folders.filter_map(|e| e.file_name().into_string().ok()).filter(|c| !c.starts_with('.') && c != DONE).sorted().collect();
    let listed = app.board_about(name).columns;
    let known = listed.iter().filter(|c| there.contains(c));
    known.chain(there.iter().filter(|c| !listed.contains(c))).cloned().chain([DONE.to_string()]).unique().collect()
}

/// A board's tasks, column by column, each column's by the number its file's name starts with.
pub fn tasks(app: &App, name: &str) -> Vec<Task> {
    let dir = app.board_path(name);
    let mut tasks = vec![];
    for column in columns(app, name) {
        let files = std::fs::read_dir(dir.join(&column)).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok());
        for file in files.filter(|f| f.ends_with(".md") && !f.starts_with('.')).sorted_by_key(|f| (number(f).unwrap_or(u64::MAX), f.clone())) {
            if let Ok(text) = std::fs::read_to_string(dir.join(&column).join(&file)) {
                tasks.push(parse(&file, &column, &text));
            }
        }
    }
    tasks
}

/// The number a task's file's name starts with, if it does.
fn number(file: &str) -> Option<u64> {
    file.split(|c: char| !c.is_ascii_digit()).next().filter(|n| !n.is_empty())?.parse().ok()
}

/// A task's file `text`, read: front matter between `---` lines (its title and owner, the rest kept),
/// then its description, then its comments, the lines under `## Comments` that start `- **who**:` and
/// those that continue them. Without a title, a task's is its file's name.
pub fn parse(file: &str, column: &str, text: &str) -> Task {
    let text = text.replace("\r\n", "\n");
    let (head, body) = text.strip_prefix("---\n").and_then(|rest| rest.split_once("\n---\n").or_else(|| rest.strip_suffix("\n---").map(|h| (h, "")))).unwrap_or(("", &text));
    let (mut title, mut owner, mut extra) = (None, None, vec![]);
    for line in head.lines() {
        let unquoted = |v: &str| v.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        match line.split_once(':') {
            Some((k, v)) if k.trim() == "title" => title = Some(unquoted(v)).filter(|t| !t.is_empty()),
            Some((k, v)) if k.trim() == "owner" => owner = Some(unquoted(v)).filter(|o| !o.is_empty() && o != "none"),
            _ => extra.push(line.to_string()),
        }
    }
    let lines: Vec<&str> = body.lines().collect();
    let at = lines.iter().position(|l| l.trim().eq_ignore_ascii_case("## comments")).unwrap_or(lines.len());
    let mut comments: Vec<(String, String)> = vec![];
    for line in lines.iter().skip(at + 1) {
        if let Some((who, said)) = line.trim_start().strip_prefix("- **").and_then(|l| l.split_once("**:")) {
            comments.push((who.trim().to_string(), said.trim().to_string()));
        } else if let Some(last) = comments.last_mut() && !line.trim().is_empty() {
            last.1 = format!("{}\n{}", last.1, line.trim());
        }
    }
    let title = title.unwrap_or_else(|| file.trim_end_matches(".md").trim_start_matches(|c: char| c.is_ascii_digit() || c == '-').replace('-', " "));
    Task { file: file.into(), column: column.into(), title, owner, extra, description: lines[..at].join("\n").trim().to_string(), comments }
}

/// A task's file, as parse reads it.
fn render(t: &Task) -> String {
    let head: Vec<String> = [format!("title: {}", t.title), format!("owner: {}", t.owner.as_deref().unwrap_or("none"))].into_iter().chain(t.extra.iter().cloned()).collect();
    let comments = t.comments.iter().map(|(who, said)| format!("- **{who}**: {}\n", said.replace('\n', "\n  "))).join("");
    format!("---\n{}\n---\n{}\n\n## Comments\n{comments}", head.join("\n"), t.description)
}

/// Board `name`'s task in file `file`, wherever it is.
fn find(app: &App, name: &str, file: &str) -> Result<Task, String> {
    tasks(app, name).into_iter().find(|t| t.file == file).ok_or(format!("no task {file} on {name}"))
}

/// A file's new text, written whole: to a hidden file beside it, then renamed into place, so no one
/// reads half of it.
fn write_whole(path: &Path, text: &str) -> Result<(), String> {
    let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let new = path.with_file_name(format!(".{name}.new"));
    std::fs::write(&new, text).and_then(|()| std::fs::rename(&new, path)).map_err(|e| format!("{name}: {e}"))
}

/// One line of text, for a title.
fn one_line(text: &str) -> String { text.split_whitespace().join(" ") }

/// A file name's part from a title: its words, lowercase, letters and digits, joined by -.
fn slug(title: &str) -> String {
    let words = title.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).join("-");
    let cut: String = words.chars().take(40).collect();
    if cut.is_empty() { "task".into() } else { cut.trim_end_matches('-').to_string() }
}

/// A change from the page to board `name`, which `change`'s `op` names: a task made ({column, title,
/// owner?, description?}), changed ({file, title?, owner?, description?, column?}, a null owner none)
/// or deleted ({file}); a comment of the user's ({file, text}); a column added or taken out ({add} or
/// {remove}: an empty one, not done), or the columns put in order ({order}: each but done, once); or
/// the board's description set ({description}, blank: none).
/// Only the hub calls this.
pub fn apply(app: &App, name: &str, change: &Value) -> Result<(), String> {
    if !app.board_exists(name) {
        return Err(format!("no board {name}"));
    }
    let dir = app.board_path(name);
    let text = |k: &str| change[k].as_str().map(str::trim).filter(|s| !s.is_empty());
    let column = |c: &str| columns(app, name).contains(&c.to_string()).then(|| c.to_string()).ok_or(format!("no column {c} on {name}"));
    let task = || find(app, name, text("file").unwrap_or(""));
    match change["op"].as_str().unwrap_or("") {
        "make" => {
            let title = one_line(text("title").ok_or("a task needs a title")?);
            let next = tasks(app, name).iter().filter_map(|t| number(&t.file)).max().map_or(1, |n| n + 1);
            let t = Task { file: format!("{next}-{}.md", slug(&title)), column: column(text("column").unwrap_or("todo"))?, title, owner: text("owner").map(str::to_string),
                extra: vec![], description: text("description").unwrap_or("").to_string(), comments: vec![] };
            write_whole(&dir.join(&t.column).join(&t.file), &render(&t))
        }
        "update" => {
            let mut t = task()?;
            let path = dir.join(&t.column).join(&t.file);
            if let Some(title) = text("title") { t.title = one_line(title) }
            if let Some(owner) = change.get("owner") { t.owner = owner.as_str().map(str::trim).filter(|o| !o.is_empty()).map(str::to_string) }
            if let Some(d) = change.get("description") { t.description = d.as_str().unwrap_or("").trim().to_string() }
            write_whole(&path, &render(&t))?;
            match text("column").filter(|c| *c != t.column) {
                Some(to) => {
                    let to = dir.join(column(to)?);
                    std::fs::create_dir_all(&to).map_err(|e| e.to_string())?;
                    if to.join(&t.file).exists() { return Err(format!("{} has a task {} already", to.display(), t.file)) }
                    std::fs::rename(&path, to.join(&t.file)).map_err(|e| e.to_string())
                }
                None => Ok(()),
            }
        }
        "delete" => {
            let t = task()?;
            std::fs::remove_file(dir.join(&t.column).join(&t.file)).map_err(|e| e.to_string())
        }
        "comment" => {
            let (t, said) = (task()?, text("text").ok_or("say something")?);
            let path = dir.join(&t.column).join(&t.file);
            let was = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let heading = if was.lines().any(|l| l.trim().eq_ignore_ascii_case("## comments")) { "" } else { "\n## Comments\n" };
            let line = format!("{}{heading}- **user**: {}\n", if was.ends_with('\n') || was.is_empty() { "" } else { "\n" }, said.replace('\n', "\n  "));
            std::fs::OpenOptions::new().append(true).open(&path).and_then(|mut f| f.write_all(line.as_bytes())).map_err(|e| e.to_string())
        }
        "column" => {
            let mut about = app.board_about(name);
            if let Some(add) = text("add") {
                if !add.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') || add.starts_with('-') {
                    return Err("a column's name is letters, digits, - and _".into());
                }
                std::fs::create_dir(dir.join(add)).map_err(|e| format!("can't make the column {add}: {e}"))?;
                about.columns.retain(|c| c != add);
                about.columns.push(add.into());
            } else if let Some(order) = change["order"].as_array() {
                let order: Vec<String> = order.iter().filter_map(|c| c.as_str()).map(str::to_string).collect();
                let mut now: Vec<String> = columns(app, name).into_iter().filter(|c| c != DONE).collect();
                let mut given = order.clone();
                now.sort();
                given.sort();
                if given != now {
                    return Err(format!("an order of {name}'s columns lists each but done once: {}", now.join(", ")));
                }
                about.columns = order;
            } else {
                let gone = column(text("remove").ok_or("which column?")?)?;
                if gone == DONE { return Err("done stays".into()) }
                std::fs::remove_dir(dir.join(&gone)).map_err(|_| format!("{gone} has tasks in it: move them out first"))?;
                about.columns.retain(|c| *c != gone);
            }
            settings::write(&dir.join("board.json"), &about)
        }
        "describe" => {
            let about = About { description: text("description").map(str::to_string), ..app.board_about(name) };
            settings::write(&dir.join("board.json"), &about)
        }
        op => Err(format!("no change {op:?}")),
    }
}

/// Session `me` put on board `name` (`on`), or taken off it. Only the hub calls this, as it writes the
/// settings files.
pub fn member(app: &App, me: &str, name: &str, on: bool) -> Result<(), String> {
    let there = app.board_exists(name);
    settings::change(&app.profile_path(me), |s: &mut SessionSettings| {
        if !on {
            s.boards.remove(name);
            return Ok(());
        }
        if !there {
            return Err(format!("no board {name}"));
        }
        if s.workspace.is_none() {
            return Err(format!("{me} has no workspace, which a session needs to work on a board"));
        }
        if s.accompanies.is_some() {
            return Err(format!("{me} is a companion, which can't work on a board"));
        }
        s.boards.insert(name.into());
        Ok(())
    }).map(drop)
}

/// How boards work, for a session's system prompt: where the boards it's on are, and their layout.
pub fn note(app: &App, boards: &[String], me: &str) -> Option<String> {
    if boards.is_empty() { return None }
    let list = boards.iter().map(|b| format!("{b} (at {})", app.board_path(b).display())).join(", ");
    let plural = if boards.len() > 1 { "s" } else { "" };
    Some(format!("You're on the kanban board{plural} {list}, which your bash and edit tools reach. A board has a folder per column, done/ among them, \
        each holding a .md file per task: its title and owner in its front matter, then what it's for, then its comments under ## Comments, each a line \
        \"- **{me}**: …\". Make a task by writing a file, move it with mv, hand it over by changing its owner, and comment by appending a line; a task's \
        owner hears when it's handed to them and when someone else comments on it."))
}

/// What a message about board `name` to session `me` ends with: how the board works, in a sentence,
/// then its description, if it has one.
fn reminder(app: &App, name: &str, me: &str) -> String {
    let about = app.board_about(name).description.map_or(String::new(), |d| format!("\nAbout {name}: {d}"));
    format!("(Board {name} is the folder {}: a folder per column, and a .md file per task, its owner in its front matter and comments appended to it as lines \"- **{me}**: …\".){about}",
        app.board_path(name).display())
}

/// What the hub last saw of a board's tasks, by file name: each one's owner, and how many comments it had.
pub type Seen = HashMap<String, (Option<String>, usize)>;

/// Board `name`'s tasks as Seen.
pub fn seen(app: &App, name: &str) -> Seen {
    tasks(app, name).into_iter().map(|t| (t.file, (t.owner, t.comments.len()))).collect()
}

/// What changed on board `name` since `seen`, which then has it as it is, put down to `by` (a
/// session's name; None, the user): the messages it makes, each with the session it's for. A task
/// handed to a session on the board tells it, unless it took the task itself; a comment tells the
/// task's owner, unless they wrote it.
pub fn changes(app: &App, name: &str, by: Option<&str>, seen: &mut Seen) -> Vec<(String, Heard)> {
    let (now, members) = (tasks(app, name), members(app, name));
    let who = by.unwrap_or("the user");
    let mut told: Vec<(String, String)> = vec![];
    for t in &now {
        let Some(owner) = t.owner.as_ref().filter(|o| members.contains(o)) else { continue };
        let (before, said) = seen.get(&t.file).cloned().unwrap_or((None, 0));
        let at = format!("{}/{}", t.column, t.file);
        if before.as_ref() != Some(owner) && by != Some(owner.as_str()) {
            let new = if seen.contains_key(&t.file) { "" } else { "new " };
            told.push((owner.clone(), format!("On board {name}, {who} handed you the {new}task \"{}\" ({at}).", t.title)));
        }
        for (author, text) in t.comments.iter().skip(said).filter(|(author, _)| author != owner) {
            let author = if author == "user" { "the user" } else { author };
            told.push((owner.clone(), format!("On board {name}, {author} commented on your task \"{}\" ({at}):\n\n{text}", t.title)));
        }
    }
    *seen = now.into_iter().map(|t| (t.file, (t.owner, t.comments.len()))).collect();
    told.into_iter().map(|(to, what)| {
        let text = format!("{what}\n\n{}", reminder(app, name, &to));
        (to, Heard { from: format!("board:{name}"), text })
    }).collect()
}

/// A board as the page shows it: its name, folder and description, its columns with their tasks (each
/// its file, title, owner and comments' count), and the sessions on it.
pub fn view(app: &App, name: &str) -> Value {
    let tasks = tasks(app, name);
    let card = |t: &Task| obj! {"file" => t.file.as_str(), "title" => t.title.as_str(), "owner" => t.owner.as_deref(), "comments" => t.comments.len()};
    let columns: Vec<Value> = columns(app, name).iter().map(|c| obj! {"name" => c.as_str(), "tasks" => tasks.iter().filter(|t| t.column == *c).map(card).collect::<Vec<_>>()}).collect();
    let path = app.board_path(name).to_string_lossy().into_owned();
    obj! {"name" => name, "path" => path, "description" => app.board_about(name).description, "columns" => columns, "members" => members(app, name)}
}

/// A task as the page shows it: its file, column, title, owner, description and comments.
pub fn task(app: &App, name: &str, file: &str) -> Result<Value, String> {
    let t = find(app, name, file)?;
    let comments: Vec<Value> = t.comments.iter().map(|(who, text)| obj! {"who" => who.as_str(), "text" => text.as_str()}).collect();
    Ok(obj! {"file" => t.file.as_str(), "column" => t.column.as_str(), "title" => t.title.as_str(), "owner" => t.owner.as_deref(), "description" => t.description.as_str(), "comments" => comments})
}

/// Every board: its name, its tasks not done, and how many sessions are on it.
pub fn listing(app: &App) -> Value {
    let names = std::fs::read_dir(app.boards_path()).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).sorted();
    Value::Array(names.map(|name| {
        let open = tasks(app, &name).iter().filter(|t| t.column != DONE).count();
        obj! {"name" => name.as_str(), "open" => open, "members" => members(app, &name).len()}
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_file_reads_and_writes_back() {
        let text = "---\ntitle: \"Fix the docs\"\nowner: onion\npriority: high\n---\nThe README says two.\n\nAnd more.\n\n## Comments\n- **bees**: on it\n  still going\n- **user**: thanks!\n";
        let t = parse("12-fix-the-docs.md", "doing", text);
        assert_eq!((t.title.as_str(), t.owner.as_deref(), t.extra.clone(), t.description.as_str()), ("Fix the docs", Some("onion"), vec!["priority: high".to_string()], "The README says two.\n\nAnd more."));
        assert_eq!(t.comments, [("bees".to_string(), "on it\nstill going".to_string()), ("user".to_string(), "thanks!".to_string())]);
        let again = parse(&t.file, &t.column, &render(&t));
        assert_eq!((again.title, again.owner, again.extra, again.description, again.comments), (t.title, t.owner, t.extra, t.description, t.comments));
        let bare = parse("3-loose-ends.md", "todo", "Just a note.\n");
        assert_eq!((bare.title.as_str(), bare.owner, bare.description.as_str(), bare.comments.len()), ("loose ends", None, "Just a note.", 0));
        assert_eq!(parse("x.md", "todo", "---\nowner: none\n---\n").owner, None);
        assert_eq!((number("12-fix.md"), number("fix.md"), slug("Fix: the README's merge docs!"), slug("!!")), (Some(12), None, "fix-the-readme-s-merge-docs".into(), "task".into()));
    }
}
