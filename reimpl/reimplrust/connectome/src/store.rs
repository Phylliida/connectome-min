//! A session on disk: append-only JSONL logs in one directory, read without loading them. Every
//! record has an `id` counting up from 0, so a lookup is a binary search, and the session's state is
//! the logs' tails:
//!
//!   messages.jsonl   the chat log: {id, participant, content, source}
//!   layer0.jsonl     chunks, in log order: {id, first, last, raw} (message ids; raw = uncalibrated price)
//!   layer<n>.jsonl   memories of level n, in order: {id, first, last, text, blocks, tokens, provenance,
//!                    recall, merges?}. Layer 1's entry p summarizes chunk p; layer n+1's entry p
//!                    merges layer n's entries from its `merges`' first to its last, or, lifted,
//!                    for each [layer, first, last] in `merges`, that layer's entries first to last
//!                    (layer n's first, then lower layers'), or, without `merges` (merged by count),
//!                    p*k .. p*k+k-1 (k = merge_threshold, fixed for a session).
//!   compiles.jsonl   one record per compile: the frontier shown, calibration, cache keys, escalation
//!   attempts.jsonl   memory attempts that failed: {id, target, messages, outcome}
//!
//! A fork (Session::fork) shares its parent's history through one message: fork.json names the
//! parent, by its path relative to the fork's, so sessions moved together stay linked, and, per file,
//! how many of the parent's records the fork shares (the messages through that one, and the chunks and
//! memories that end by then). The fork's own records continue those ids.
//!
//! The session is the only writer of its files. A failed read or write of a file panics: the store
//! is the session's memory, and nothing can go on without it.

use crate::{estimate::Priced, json::{self, Json, Value}, obj};
use std::{cell::RefCell, collections::{HashMap, HashSet}, fs::{self, File, OpenOptions}, path::{Component, Path, PathBuf}, rc::Rc};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

pub struct Session {
    pub path: PathBuf,
    parent: Option<Result<Box<Session>, String>>, // a fork's parent, or why it isn't there
    pub fork: Value, // fork.json: {parent, at, shared: {file: records shared}}, or null
    counts: RefCell<HashMap<String, (u64, u64)>>, // file -> (its size, its count) when last counted
    repaired: RefCell<HashSet<String>>,
    /// chunking::priced's memo: what plans read from a message, never the message.
    pub(crate) priced: RefCell<HashMap<u64, Rc<Priced>>>,
}

/// Where fork `path`'s parent is, by its fork.json, or why it isn't there. A fork.json from before
/// parents were named relatively names it by an absolute path, which moving the sessions leaves
/// stale; it's repaired as it's read: the parent is the session of that name beside the fork (the two
/// moved together), or else the one at that path, and fork.json then names it relatively. A folder is
/// a session if it has a log of messages or is a fork; an empty one isn't, as opening a stale path
/// once left behind. No folder is made for a parent that isn't there.
fn parent_of(path: &Path, fork: &mut Value) -> Result<PathBuf, String> {
    let named = PathBuf::from(fork["parent"].as_str().expect("fork.json parent"));
    let session = |dir: &Path| dir.join("messages.jsonl").is_file() || dir.join("fork.json").is_file();
    let missing = |looked: &str| format!("{}: the fork's parent, {}, isn't there{looked}", path.join("fork.json").display(), named.display());
    if named.is_relative() {
        let dir = normal(&path.join(&named));
        return if session(&dir) { Ok(dir) } else { Err(missing("")) };
    }
    let beside = named.file_name().and_then(|name| path.parent().map(|up| up.join(name)));
    let found = [beside, Some(named.clone())].into_iter().flatten().find(|dir| session(dir)).ok_or_else(|| missing(", nor a session of its name beside the fork"))?;
    fork["parent"] = relative(&found, path).to_string_lossy().as_ref().into();
    write_fork(path, fork);
    Ok(normal(&found))
}

/// A fork's fork.json written, whole or not at all.
fn write_fork(path: &Path, fork: &Value) {
    let (meta, new) = (path.join("fork.json"), path.join("fork.json.new"));
    fs::write(&new, json::to_string(fork)).and_then(|()| fs::rename(&new, &meta)).expect("fork.json");
}

/// `to` as a path relative to `from`, both made absolute by name, as Python's os.path.relpath makes it.
fn relative(to: &Path, from: &Path) -> PathBuf {
    let absolute = |p: &Path| normal(&std::path::absolute(p).expect("a path"));
    let (to, from) = (absolute(to), absolute(from));
    let common = to.components().zip(from.components()).take_while(|(a, b)| a == b).count();
    let up = from.components().skip(common).map(|_| Component::ParentDir);
    let path: PathBuf = up.chain(to.components().skip(common)).collect();
    if path.as_os_str().is_empty() { PathBuf::from(".") } else { path }
}

/// A path with its `.` and `..` resolved by name, as Python's os.path.normpath resolves them.
fn normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir if matches!(out.components().next_back(), Some(Component::Normal(_))) => { out.pop(); }
            c => out.push(c),
        }
    }
    out
}

impl Session {
    pub fn open(path: impl AsRef<Path>) -> Session {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path).expect("session directory");
        let meta = path.join("fork.json");
        let mut fork = if meta.exists() { json::parse(&fs::read(&meta).expect("fork.json")).expect("fork.json") } else { Value::Null };
        let parent = (!fork.is_null()).then(|| parent_of(&path, &mut fork).map(|dir| Box::new(Session::open(dir))));
        Session { path, parent, fork, counts: RefCell::default(), repaired: RefCell::default(), priced: RefCell::default() }
    }

    /// A fork's parent. Reading what a fork shares when its parent isn't there panics, saying where it
    /// was looked for.
    fn parent(&self) -> &Session {
        match &self.parent {
            Some(Ok(parent)) => parent,
            Some(Err(why)) => panic!("{why}"),
            None => unreachable!("only a fork shares records"),
        }
    }

    /// A new session at `path` whose history is `parent`'s through message `at`. It starts from the
    /// parent's last frontier, cut to the nodes it shares, and calibration.
    pub fn fork(parent: &Session, at: u64, path: impl AsRef<Path>) -> Session {
        let mut shared = obj! {"messages" => at + 1};
        for layer in (0..).map(|n| format!("layer{n}")).take_while(|layer| parent.count(layer) > 0) {
            let after = bisect(0, parent.count(&layer), |i| parent.read(&layer, i)["last"].as_f64().unwrap() > at as f64);
            shared[&layer] = after.into();
        }
        let path = path.as_ref();
        fs::create_dir_all(path).expect("fork directory");
        write_fork(path, &obj! {"parent" => relative(&parent.path, path).to_string_lossy().as_ref(), "at" => at, "shared" => shared});
        let child = Session::open(path);
        let last = parent.last("compiles").unwrap_or(obj! {});
        let frontier: Vec<Value> = last["frontier"].arr().iter().take_while(|node| node.arr()[1].u64() < child.shared(&format!("layer{}", node.arr()[0].u64()))).cloned().collect();
        let multiplier = last["multiplier"].as_f64().unwrap_or(1.0);
        child.append("compiles", obj! {"frontier" => frontier, "multiplier" => multiplier, "armed" => 0u64, "keys" => Value::Null, "escalated" => false});
        child
    }

    fn shared(&self, name: &str) -> u64 { self.fork["shared"][name].u64() }

    fn file(&self, name: &str) -> PathBuf { self.path.join(format!("{name}.jsonl")) }

    /// The newest record, or None.
    pub fn last(&self, name: &str) -> Option<Value> {
        match last_line(&self.file(name), u64::MAX) {
            Some(line) => Some(json::parse(&line).expect("a record")),
            None => (self.shared(name) > 0).then(|| self.parent().read(name, self.shared(name) - 1)),
        }
    }

    /// The newest record's id + 1, read again only once the file has changed.
    pub fn count(&self, name: &str) -> u64 {
        let size = fs::metadata(self.file(name)).map_or(0, |m| m.len());
        if let Some(&(known, n)) = self.counts.borrow().get(name)
            && known == size
        {
            return n;
        }
        let n = match last_line(&self.file(name), 64) {
            Some(head) => id_of(&head).or_else(|| id_of(&last_line(&self.file(name), u64::MAX)?)).expect("a record id") + 1,
            None => self.shared(name),
        };
        self.counts.borrow_mut().insert(name.to_string(), (size, n));
        n
    }

    /// Record i, from disk: the parent's while the fork shares it.
    pub fn read(&self, name: &str, i: u64) -> Value {
        self.span(name, i, i).next().unwrap_or_else(|| panic!("no record {i} in {name}"))
    }

    /// Records first..=last, in order, read as they're iterated.
    pub fn span(&self, name: &str, first: u64, last: u64) -> Box<dyn Iterator<Item = Value> + '_> {
        let (n, start) = (self.shared(name), first.max(self.shared(name)));
        let inherited = (first < n && first <= last).then(|| self.parent().span(name, first, last.min(n - 1)));
        let own = (start <= last).then(|| self.scan(name, start).take_while(move |r| r.id() <= last));
        Box::new(inherited.into_iter().flatten().chain(own.into_iter().flatten()))
    }

    /// This session's own records from id `i` on, a line at a time.
    fn scan(&self, name: &str, i: u64) -> impl Iterator<Item = Value> + use<> {
        let mut file = BufReader::new(File::open(self.file(name)).unwrap_or_else(|e| panic!("{name}: {e}")));
        let offset = seek(&mut file, i);
        file.seek(SeekFrom::Start(offset)).expect("seek");
        file.split(b'\n').map(|line| line.expect("read")).filter(|line| !line.trim_ascii().is_empty()).map(|line| json::parse(&line).expect("a record"))
    }

    pub fn append(&self, name: &str, record: Value) -> Value {
        let id = self.count(name);
        let Value::Object(fields) = record else { panic!("a record is an object") };
        let record = Value::Object(std::iter::once(("id".to_string(), Value::from(id))).chain(fields.into_iter().filter(|(k, _)| k != "id")).collect());
        let path = self.file(name);
        if self.repaired.borrow_mut().insert(name.to_string()) {
            repair(&path);
        }
        let mut f = OpenOptions::new().create(true).append(true).open(&path).expect("open for append");
        let mut line = json::to_string(&record); // one write: a record is never half there
        line.push('\n');
        f.write_all(line.as_bytes()).expect("append");
        self.counts.borrow_mut().insert(name.to_string(), (f.metadata().expect("metadata").len(), id + 1));
        record
    }
}

/// The offset of the first line whose id is at least `id` (or the end): a binary search over byte
/// offsets, each probe reading the id of the first line starting at or after it.
fn seek(file: &mut BufReader<File>, id: u64) -> u64 {
    let size = file.get_ref().metadata().expect("metadata").len();
    let at = bisect(0, size, |mid| !matches!(id_at(file, mid), Some((_, found)) if found < id));
    id_at(file, at).map_or(at, |(start, _)| start)
}

/// The first of lo..hi where `past` holds (or hi), for a `past` that's false and then true across them.
fn bisect(mut lo: u64, mut hi: u64, mut past: impl FnMut(u64) -> bool) -> u64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if past(mid) { hi = mid } else { lo = mid + 1 }
    }
    lo
}

/// The id of the first record starting at or after `offset`, and where it starts.
fn id_at(file: &mut BufReader<File>, offset: u64) -> Option<(u64, u64)> {
    file.seek(SeekFrom::Start(offset.saturating_sub(1))).expect("seek");
    if offset > 0 {
        file.skip_until(b'\n').expect("read");
    }
    loop {
        let start = file.stream_position().expect("position");
        let mut head = vec![];
        if file.by_ref().take(64).read_until(b'\n', &mut head).expect("read") == 0 {
            return None;
        }
        if head.ends_with(b"\n") && head.trim_ascii().is_empty() {
            continue;
        }
        if id_of(&head).is_none() {
            file.read_until(b'\n', &mut head).expect("read"); // the id doesn't lead the record
        }
        return Some((start, id_of(&head).expect("a record id")));
    }
}

/// A record's id: from its first bytes when it leads them, as this store writes records, else from
/// the whole record.
fn id_of(bytes: &[u8]) -> Option<u64> {
    let lead = bytes.strip_prefix(b"{\"id\":").and_then(|rest| {
        let digits = &rest[..rest.iter().position(|b| !b.is_ascii_digit())?];
        std::str::from_utf8(digits).ok()?.parse().ok()
    });
    lead.or_else(|| json::parse(bytes).ok()?["id"].as_u64())
}

/// The first `most` bytes of the last non-blank line of a file, found from its end.
fn last_line(path: &Path, most: u64) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    let end = rfind(&mut file, size, |b| !b.is_ascii_whitespace())? + 1;
    let start = line_start(&mut file, end);
    Some(read_at(&mut file, start, end.min(start.saturating_add(most))))
}

/// Where the line ending at `end` starts: past the newline before it, or at 0.
fn line_start(file: &mut File, end: u64) -> u64 {
    rfind(file, end, |b| b == b'\n').map_or(0, |nl| nl + 1)
}

/// Where the last byte before `end` that `wanted` holds of is, read backward a block at a time.
fn rfind(file: &mut File, end: u64, wanted: impl Fn(u8) -> bool) -> Option<u64> {
    let (mut pos, mut block) = (end, vec![0; 1 << 16]);
    while pos > 0 {
        let step = pos.min(block.len() as u64) as usize;
        file.seek(SeekFrom::Start(pos - step as u64)).and_then(|_| file.read_exact(&mut block[..step])).expect("read");
        if let Some(k) = block[..step].iter().rposition(|&b| wanted(b)) {
            return Some(pos - step as u64 + k as u64);
        }
        pos -= step as u64;
    }
    None
}

fn read_at(file: &mut File, start: u64, end: u64) -> Vec<u8> {
    let mut bytes = vec![0; (end - start) as usize];
    file.seek(SeekFrom::Start(start)).and_then(|_| file.read_exact(&mut bytes)).expect("read");
    bytes
}

/// Drops a last line a crash cut short, and ends a whole one that lacks its newline (once per
/// session and file, before its first append).
fn repair(path: &Path) {
    let Ok(mut file) = OpenOptions::new().read(true).write(true).open(path) else { return };
    let size = file.metadata().expect("metadata").len();
    if size == 0 { return }
    let end = size - (read_at(&mut file, size - 1, size) == b"\n") as u64;
    let start = line_start(&mut file, end);
    if json::parse(&read_at(&mut file, start, end)).is_err() {
        file.set_len(start).expect("truncate");
    } else if end == size {
        file.seek(SeekFrom::Start(size)).and_then(|_| file.write_all(b"\n")).expect("write");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_relative_and_normal_as_python_makes_them() {
        assert_eq!(relative(Path::new("/a/b/trunk"), Path::new("/a/b/twig")), Path::new("../trunk"));
        assert_eq!(relative(Path::new("/a/b"), Path::new("/a/b/fork")), Path::new(".."));
        assert_eq!(relative(Path::new("/a/x/../b"), Path::new("/a/c/d")), Path::new("../../b"));
        assert_eq!(relative(Path::new("/a"), Path::new("/a")), Path::new("."));
        assert_eq!(normal(Path::new("s/twig/./../trunk")), Path::new("s/trunk"));
        assert_eq!(normal(Path::new("../s/../t")), Path::new("../t"));
    }
}
