//! The lazy counter: what the middle of the window shows, as a frontier.
//!
//! The frontier lists, in log order, a node for every closed chunk: (0, j) shows chunk j raw; (n, p)
//! shows layer n's memory p, which covers chunks p*k^(n-1) .. (p+1)*k^(n-1) - 1. Levels fall toward
//! the present: old content coarse, new content fine. A change re-reads everything after it, so a plan
//! keeps the carried frontier and changes it only at the edges of a band, each time where the change
//! starts latest:
//!   over W         fold until the total is at most the target, W * (1 - slack): each time, the
//!                  lowest level's oldest complete group: raw chunk j into memory (1, j), or memories
//!                  (n, p*k) .. (n, p*k+k-1), adjacent in the frontier, into (n+1, p), once it exists
//!   under the low  after a fold, or below W * unfold_below: replace the newest memory with what it
//!   water          merged while the total stays within the target
//! Lower levels sit later in the window, so the lowest level's oldest group is the latest change that
//! keeps levels falling toward the present; a level changes only when the level below it fills, as in
//! a counter.

use crate::config::Config;

pub type Node = (u64, u64); // (layer, id)

/// fixed: the tokens outside the frontier (the open run and the tail).
pub fn plan(frontier: Vec<Node>, cost: impl Fn(Node) -> f64, exists: impl Fn(Node) -> bool, fixed: f64, w: f64, cfg: &Config) -> Vec<Node> {
    let total = |f: &[Node]| fixed + f.iter().map(|&n| cost(n)).fold(0.0, |a, x| a + x);
    let (target, mut f) = (w * (1.0 - cfg.slack), frontier);
    let folded = total(&f) > w;
    while folded && total(&f) > target && let Some(folds) = fold(&f, &exists, cfg.merge_threshold) {
        f = folds;
    }
    if folded || total(&f) < w * cfg.unfold_below {
        while let Some(trial) = unfold(&f, cfg.merge_threshold).filter(|t| total(t) <= target) {
            f = trial;
        }
    }
    f
}

/// F with its lowest level's oldest complete group folded, or None if no group is complete.
pub fn fold(f: &[Node], exists: impl Fn(Node) -> bool, k: u64) -> Option<Vec<Node>> {
    let complete = |i: usize, (level, j): Node| {
        let (width, parent) = if level == 0 { (1, (1, j)) } else { (k, (level + 1, j / k)) };
        let run = f.get(i..i + width as usize)?;
        (j % width == 0 && run.iter().enumerate().all(|(t, &n)| n == (level, j + t as u64)) && exists(parent)).then_some((level, i, width, parent))
    };
    let (_, i, width, parent) = f.iter().enumerate().filter_map(|(i, &n)| complete(i, n)).min_by_key(|&(level, i, ..)| (level, i))?;
    Some([&f[..i], &[parent], &f[i + width as usize..]].concat())
}

/// F with its newest memory replaced by what it merged, or None if it shows none.
pub fn unfold(f: &[Node], k: u64) -> Option<Vec<Node>> {
    let i = f.iter().rposition(|&(level, _)| level > 0)?;
    let (level, p) = f[i];
    let children: Vec<Node> = if level == 1 { vec![(0, p)] } else { (0..k).map(|t| (level - 1, p * k + t)).collect() };
    Some([&f[..i], &children, &f[i + 1..]].concat())
}

/// The chunks a node covers.
pub fn chunks((level, p): Node, k: u64) -> std::ops::Range<u64> {
    if level == 0 { p..p + 1 } else { p * k.pow(level as u32 - 1)..(p + 1) * k.pow(level as u32 - 1) }
}
