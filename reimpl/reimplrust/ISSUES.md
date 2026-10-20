# Known issues

Found by an independent audit (2026-10-19), kept for later. Each was reproduced or traced through
the code; none is fixed yet.

## A hand-edited log the library can't read breaks the page's API, for every session

`GET /api/sessions`, `GET /api/s/NAME`, `/messages`, `/layer`, `/fork`, `/settings`, `/say` and
`/continue` read the log directly, and the library panics on a record it can't read
(`connectome/src/store.rs`: `count`'s "a record id", `read`, `scan`, `last`). `api::handle` runs on
a thread whose panic isn't caught (`main.rs`), so the connection resets. Only the stream
(`api::follow`) and the session's thread catch it. One session whose log has, say, a message
without an `id` makes `GET /api/sessions` fail, so the sidebar loads none of them.

Fix: catch a panic per request in `api::handle` and answer it as an error, and in `sessions` per
session (marking that one broken), the way `follow` does.

## A fork whose parent is deleted can't be read, and opening it makes the parent's directory again

`Session::open` (`connectome/src/store.rs`) creates its directory, and opens its parent's from
`fork.json` the same way, so a deleted parent comes back as an empty directory (and appears in the
sidebar, without settings); the fork's inherited messages are then gone, and reading them panics
(as above). Deleting a companion's directory while it's paired does the same to it when its
partner tells it something.

Fix: open a session without creating its directory (only creating a session or a fork does), and
report a missing parent as the session's problem.

## The bash sandbox shares more of the host than the README says

`tools/bash.rs` passes `--dev-bind /dev /dev`, which binds the host's whole `/dev` (GPU, sound,
FUSE and other devices the user's groups can open), and doesn't unshare IPC or UTS. The command
runs as the user, so root's devices stay closed, but the README says a command sees only its
workspace and the paths `sandbox` lists.

Fix: `--dev /dev` (bubblewrap's minimal `/dev`) and `--unshare-ipc --unshare-uts`, keeping the
sandbox tests green; or say in the README what's shared.

## A window or reply reserve of 0 saves, and only the next turn fails

`SessionSettings::tuned` (`settings.rs`) doesn't bound `window` or `reserve`, and the page sends a
cleared number field as 0 (`ui.html`'s `tune`). The form says "Saved.", and the next turn is
refused with "The context doesn't fit".

Fix: refuse at save a window that doesn't leave room for the reserve and the tools, and a reserve
of 0 or less.
