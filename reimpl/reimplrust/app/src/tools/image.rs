//! The read_image tool: an image file in the session's workspace, or on one of its boards, given to the
//! model as an image, for sessions whose model takes images (their `images` setting). Its paths and
//! its reading are edit's (edit.rs): they stay in the workspace and the boards' folders, and in a
//! sandbox a command run in bash's reads the file. It takes what both the Messages API and llama.cpp's
//! server take, PNG, JPEG and GIF (not WebP, which llama.cpp can't read) of up to MOST bytes: an image
//! a provider refuses would fail every request after it. A call asks first.

use super::{edit::{inside, Files}, Outcome, Stopped, Tool, Toolbox};
use crate::settings::Sandbox;
use base64::{engine::general_purpose::STANDARD, Engine};
use connectome::{json::Value, obj};
use std::{path::PathBuf, time::Duration};

/// The largest file read_image takes: the Messages API takes images of up to 5 MB in base64.
const MOST: usize = 5 * 1024 * 1024 / 4 * 3;

/// read_image in `dir` and the folders `also` (a session's boards), in `sandbox` if given (bubblewrap,
/// so Linux only: elsewhere read_image is unavailable), a call's command there killed past `limit`.
pub fn tool(dir: PathBuf, also: Vec<PathBuf>, limit: Duration, sandbox: Option<Sandbox>) -> Toolbox {
    if sandbox.is_some() && !cfg!(target_os = "linux") {
        return unavailable("its sandbox is bubblewrap, which runs only on Linux");
    }
    let boards = if also.is_empty() { String::new() } else { format!(" (or, by its absolute path, in {})", also.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")) };
    let description = format!("Look at an image file in {}{boards}: a PNG, JPEG or GIF of up to {} MB, which comes back as an image you see. \
        A relative path is relative to {0}.", dir.display(), MOST as f64 / 1024.0 / 1024.0);
    let properties = obj! {"path" => obj! {"type" => "string", "description" => "The image file"}};
    let definition = obj! {"name" => "read_image", "description" => description, "input_schema" => obj! {"type" => "object", "properties" => properties, "required" => vec!["path"]}};
    let files = Files { dir, also, limit, sandbox };
    Toolbox(vec![Tool { definition, asks: true, run: Box::new(move |input, stopped| read_image(&files, input, stopped)) }])
}

/// read_image where it can't run (super::unavailable).
pub fn unavailable(why: &'static str) -> Toolbox {
    super::unavailable("read_image", "Shows you an image file", why)
}

/// The file's name and the image, as the library logs a tool result's blocks.
fn read_image(files: &Files, input: &Value, stopped: Stopped) -> Outcome {
    let path = input["path"].as_str().ok_or("read_image needs a path")?;
    let bytes = files.read(&inside(&files.dir, &files.also, path)?, stopped).map_err(|e| format!("{path}: {e}"))?;
    let kind = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") { "png" } else if bytes.starts_with(b"\xff\xd8\xff") { "jpeg" } else if bytes.starts_with(b"GIF8") { "gif" } else { "" };
    if kind.is_empty() {
        return Err(format!("{path} isn't a PNG, JPEG or GIF; convert it to one first").into());
    }
    if bytes.len() > MOST {
        return Err(format!("{path} is {:.1} MB, past the {:.2} MB an image may be; make it smaller first", bytes.len() as f64 / 1024.0 / 1024.0, MOST as f64 / 1024.0 / 1024.0).into());
    }
    let image = obj! {"type" => "image", "source" => obj! {"type" => "base64", "mediaType" => format!("image/{kind}"), "data" => STANDARD.encode(&bytes)}};
    Ok(vec![obj! {"type" => "text", "text" => path}, image].into())
}
