# How I want code written

For any Claude working on my code. Read this before writing or restructuring anything, and check your work against the reference code at the end.

## Three rules

1. **Abstract only the minimal amount needed.** Add a type, trait, helper or layer only when the code in front of you needs it now. One caller is no reason for a helper, and one implementation is no reason for a trait.
2. **Keep code that works together next to each other.** Organize by feature, not by role. A reader should understand a feature by reading one place, not five files.
3. **Minimize what a reader must hold in their head.** Code reads top to bottom, and a function makes sense without opening another file.

Fewer lines usually follow, but line count is not the goal. Count the concepts a reader has to learn, not the lines.

## In practice

**Structure**
- A feature lives in one file: a context manager's records, loading, saving and editing together.
- Don't split code into modules named for their role (`segment`, `emit`, `drive`, `turns`).
- When parts form a pipeline, put the stages next to each other, with the one function that wires them together.
- A long procedure can stay one function, with numbered comments marking its steps (`// 1) Emit a distinct probe on each output device`, `// 2) Capture ~3s of aligned input/output data`), rather than being split into helpers that are each called once.
- Define a callback where it's used, as a nested function next to the state it closes over.

**Lifecycle**
- Wire everything in one place, in dependency order, and hand each component what it needs through its constructor. Melodeus's `main` nests `async with` for the audio system, STT, TTS, context and server. No registries or global lookups.
- Every resource has a matching teardown: `__aexit__`, `Drop`, or `finally`.
- Interruptible work runs as a task, and interrupting it means cancelling the task. The cancel handler does the bookkeeping (`tts.py` trims the reply to what was actually heard), and `finally` cleans up.

**Code**
- Plain loops and plain statements. Never hide a state change inside a `filter`, `find` or `map` closure.
- Don't golf: no bools used as integers, no four-way tuple bindings, no chains a reader has to simulate in their head.
- Long names that say what they hold, like `time_still_need_to_sleep` or `num_frames_that_are_behind_current_packet`. Not `p`, `m`, `w`.
- Typed records: dataclasses in Python, serde structs and enums in Rust. Don't pass `dict` or `serde_json::Value` around and read it through string keys. Ids and counts are integers.
- Pick the simple mechanism that is obviously correct: poll every 50 ms, make a channel generously large ("some large constant is fine"), buffer a whole sentence of audio before playing it so it doesn't jitter.
- Two similar blocks, like input and output, are fine written out twice rather than made generic.
- Process a stream as a generator over a generator: `stream_sentences` turns a stream of text chunks into a stream of sentences.
- Use the standard library and small, maintained libraries for mechanics (`bisect`, `mmap`, serde, an HTTP server) rather than writing them by hand. Check the binary-size cost before adding a big one.
- No caches, shortcuts or tuning unless a measurement shows they matter or the domain plainly requires them. An audio callback must not allocate, so it reuses scratch buffers. When you do optimize, say which reason applies.
- No backward-compatibility code. When a format changes, migrate the data once.

**Errors**
- Errors are values: `Result` and `?`, or an exception with a clear message. Don't panic and then catch it with `catch_unwind`.
- An error message says how to fix the problem. When a device isn't found, list the devices that do exist; when a config isn't supported, list the ones that are.

**Config and constants**
- Config is plain records with defaults. Each field gets a comment giving its tradeoff and a value that works: "100 or so is fine", "a larger value here will take longer to connect, but result in more accurate timing alignments".
- Fixed values are named constants at the top of the file, with a comment saying why they're fixed ("these are hardcoded to make aec3 happy").

**Comments**
- Say why, not what. State the fact that makes the code correct ("none can ever arrive too early", so taking the minimum gives a good estimate), and link the source when there is one.
- A worked example with real numbers beats a paragraph; see the mixer's latency comment in `cpal_aec.rs`.
- Don't give every function a doc comment that restates its name.

**Concurrency**
- Don't share state behind `Mutex` or `Arc`. Give each piece of changing state one owning thread, and talk to that thread through channels.
- Control messages are small enums. The owner drains its inbox at natural points in its own loop, such as between chunks or between stream events, and applies each message to its own state.

**Data on disk**
- Append-only JSONL that a person can read and edit by hand. The log is the single source of truth. No index files or other derived files beside it, because those go stale when someone edits the log.
- When a log can grow large (connectome sessions reach 100M tokens), never load it whole. Read only what you need: binary search, or scan backwards from the end.
- Repair a half-written last line before appending.

## Slop I don't want

All of these came from a Rust rewrite I judged "a lot of slop". Each is a habit to catch in yourself:
- Records as untyped `serde_json::Value`, read through string keys, with ids stored as `f64`.
- Many small modules named by role, so a single feature (calibration) spreads across five files.
- Abstractions with no present need: a `Toolbox` that is "a monoid under `+`", with tests of the monoid laws; closure factories; a trait with one implementer.
- Caches and shortcuts nobody measured.
- Panics as error handling, caught by `catch_unwind`.
- Golfed one-liners written to hit a line count.
- A doc comment on every function, paraphrasing the code in odd prose.

One before and after from that code:

```rust
// slop: the state change hides inside the filter closure
let answers = |b: &&Block| b.kind != "tool_result" || uses.contains(&b.tool) && !answered.contains(&b.tool) && { answered.push(b.tool.clone()); true };
let mut content: Vec<Block> = entry.content.iter().filter(answers).cloned().collect();
```

```rust
// plain: a tool result stays only if it answers a call from the turn before, and only once
let mut content = vec![];
for block in &entry.content {
    if block.kind == "tool_result" {
        if !uses.contains(&block.tool) || answered.contains(&block.tool) {
            continue;
        }
        answered.push(block.tool.clone());
    }
    content.push(block.clone());
}
```

## Reference code

I refined these by hand. They are shorter and less buggy than anything I have gotten from Claude. Read them before calling your code clean.
- **`tmp/connectome-min/reimpl/reimpl2/jsonlparse.py`**: a JSONL store in about 100 lines. It's a thin class over plain functions, uses stdlib tools, looks up by any key, and has no caches.
- **[melodeus](https://github.com/Phylliida/melodeus)**, the top-level `.py` files (not `old/`):
  - records and configs as dataclasses, with commented defaults;
  - one feature per file (`context_manager.py` is the whole context);
  - subsystems composed with nested `async with` in `main`;
  - interruption by cancelling a task (`tts.py`).
- **[pure-rust-aec](https://github.com/Phylliida/pure-rust-aec), `src/cpal_aec.rs`**:
  - pipeline stages that each own their state, with ring buffers for the audio;
  - small message enums that the owner drains at each step;
  - comments with worked numbers;
  - error messages that list what is available.

Not examples: pure-rust-aec's `pywrapper.rs` and the `mel-aec` repo. Both were vibecoded.

From `jsonlparse.py`:

```python
def iterate_jsonl_backwards(path):
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return  # can't mmap an empty file
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = len(mm)
            while end > 0:
                nl = mm.rfind(b"\n", 0, end)  # last newline before end, C speed
                line = mm[nl + 1:end]
                if line.strip():
                    yield json.loads(line)
                end = nl  # -1 when no newline remains -> loop ends
```

From `cpal_aec.rs` (abridged): at the start of each step, the mixer applies the messages queued for it to its own state.

```rust
loop {
    match self.output_stream_receiver.try_next() {
        Ok(Some(msg)) => match msg {
            OutputStreamMessage::Add(id, /* ... */) => { self.stream_consumers.insert(id, /* ... */); }
            OutputStreamMessage::Remove(id) => { self.stream_consumers.remove(&id); }
            OutputStreamMessage::InterruptAll() => { self.stream_consumers.clear(); }
            // ...
        },
        Ok(None) => break, // the sender is gone
        Err(_) => break,   // nothing queued right now
    }
}
```

## Working with me

- Give your own judgment and keep it. If you disagree with me, say so once, with your reasons. Change a recommendation only for a reason you can name, and don't flip just to agree with me.
- When I ask to discuss or plan, don't change code until I say so.
- When simplifying, simplify the implementation, not the tests. Keep test runs under a minute.
