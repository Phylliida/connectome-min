//! The lazy counter: what the middle of the window shows, as a frontier.
//!
//! The frontier lists, in log order, a node for every closed chunk: (0, j) shows chunk j raw; (n, p)
//! shows layer n's memory p, which covers the chunks of the nodes it merged (Groups), an L1 its
//! chunk p. Levels fall toward the present: old content coarse, new content fine. A change re-reads
//! everything after it, so a plan keeps the carried frontier and changes it only at the edges of a
//! band, each time where the change starts latest:
//!   over W         fold until the total is at most the target, W * (1 - slack): each time, the
//!                  lowest level's oldest complete group: raw chunk j into memory (1, j), or the
//!                  nodes a memory merged, adjacent in the frontier, into it
//!   under the low  after a fold, or below W * unfold_below: replace the newest memory with what it
//!   water          merged while the total stays within the target
//! Lower levels sit later in the window, so the lowest level's oldest group is the latest change that
//! keeps levels falling toward the present; a level changes only when the level below it fills, as in
//! a counter.

use crate::config::Config;
use std::ops::Range;

pub type Node = (u64, u64); // (layer, id)

/// How memories group: what a memory of layer 2 or above merged, and the memory a group makes. An L1
/// holds its chunk.
pub trait Groups {
    /// The nodes memory (n, p) merged (n >= 2), in log order: entries of layer n-1, or, lifted, a run
    /// of each of several layers below, the highest first.
    fn members(&self, node: Node) -> Vec<Node>;
    /// The memory of layer n+1 whose group starts at (n, j) (n >= 1), if one may: whether it exists
    /// is plan's `exists` to say.
    fn parent(&self, node: Node) -> Option<Node>;
}

/// Groups of k, as the library merges them: layer n+1's memory p merged layer n's p*k .. p*k+k-1.
pub struct Fixed(pub u64);

impl Groups for Fixed {
    fn members(&self, (n, p): Node) -> Vec<Node> { (p * self.0..(p + 1) * self.0).map(|j| (n - 1, j)).collect() }
    fn parent(&self, (n, j): Node) -> Option<Node> { (j % self.0 == 0).then_some((n + 1, j / self.0)) }
}

/// fixed: the tokens outside the frontier (the open run and the tail).
pub fn plan(frontier: Vec<Node>, cost: impl Fn(Node) -> f64, exists: impl Fn(Node) -> bool, fixed: f64, w: f64, groups: &impl Groups, cfg: &Config) -> Vec<Node> {
    let total = |f: &[Node]| fixed + f.iter().map(|&n| cost(n)).fold(0.0, |a, x| a + x);
    let (target, mut f) = (w * (1.0 - cfg.slack), frontier);
    let folded = total(&f) > w;
    while folded && total(&f) > target && let Some(folds) = fold(&f, &exists, groups) {
        f = folds;
    }
    if folded || total(&f) < w * cfg.unfold_below {
        while let Some(trial) = unfold(&f, groups).filter(|t| total(t) <= target) {
            f = trial;
        }
    }
    f
}

/// F with its lowest level's oldest complete group folded, or None if no group is complete.
pub fn fold(f: &[Node], exists: impl Fn(Node) -> bool, groups: &impl Groups) -> Option<Vec<Node>> {
    let complete = |i: usize, node: Node| {
        let (parent, members) = if node.0 == 0 { ((1, node.1), vec![node]) } else { groups.parent(node).map(|p| (p, groups.members(p)))? };
        (f.get(i..i + members.len())? == members.as_slice() && exists(parent)).then_some((node.0, i, members.len(), parent))
    };
    let (_, i, width, parent) = f.iter().enumerate().filter_map(|(i, &n)| complete(i, n)).min_by_key(|&(level, i, ..)| (level, i))?;
    Some([&f[..i], &[parent], &f[i + width..]].concat())
}

/// F with its newest memory replaced by what it merged, or None if it shows none.
pub fn unfold(f: &[Node], groups: &impl Groups) -> Option<Vec<Node>> {
    let i = f.iter().rposition(|&(level, _)| level > 0)?;
    let (level, p) = f[i];
    let children = if level == 1 { vec![(0, p)] } else { groups.members((level, p)) };
    Some([&f[..i], &children, &f[i + 1..]].concat())
}

/// The chunks a node covers: from its first member's first to its last member's last.
pub fn chunks(node: Node, groups: &impl Groups) -> Range<u64> {
    let edge = |mut node: Node, last: bool| {
        while node.0 > 1 {
            let members = groups.members(node);
            node = if last { members[members.len() - 1] } else { members[0] };
        }
        node.1
    };
    edge(node, false)..edge(node, true) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Groups as a table: each merged memory and its members.
    struct Table(Vec<(Node, Vec<Node>)>);

    impl Groups for Table {
        fn members(&self, node: Node) -> Vec<Node> {
            self.0.iter().find(|(m, _)| *m == node).expect("a memory").1.clone()
        }
        fn parent(&self, (n, j): Node) -> Option<Node> {
            self.0.iter().find(|((level, _), members)| *level == n + 1 && members[0] == (n, j)).map(|(m, _)| *m)
        }
    }

    fn run(level: u64, ids: Range<u64>) -> Vec<Node> {
        ids.map(|j| (level, j)).collect()
    }

    #[test]
    fn groups_of_any_size_fold_and_unfold() {
        // L2 0 merged L1 0..3, L2 1 L1 3..5; L3 0 merged L2 0..2: it covers chunks 0..5
        let groups = Table(vec![((2, 0), run(1, 0..3)), ((2, 1), run(1, 3..5)), ((3, 0), run(2, 0..2))]);
        assert_eq!(chunks((3, 0), &groups), 0..5);
        assert_eq!(chunks((2, 1), &groups), 3..5);
        let exists = |(n, p): Node| [6u64, 5, 2, 1][n as usize] > p;   // six chunks, five L1s
        let f = vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (0, 5)];
        let once = fold(&f, exists, &groups).expect("a fold");
        assert_eq!(once, [(2, 0), (1, 3), (1, 4), (0, 5)]);   // the lowest level's oldest complete group: raw 5 has no L1 yet
        let twice = fold(&once, exists, &groups).expect("a fold");
        assert_eq!(twice, [(2, 0), (2, 1), (0, 5)]);
        assert_eq!(fold(&twice, exists, &groups), Some(vec![(3, 0), (0, 5)]));
        assert_eq!(unfold(&[(3, 0), (0, 5)], &groups), Some(twice.clone()));
        assert_eq!(unfold(&twice, &groups), Some(vec![(2, 0), (1, 3), (1, 4), (0, 5)]));
        assert_eq!(fold(&[(1, 1), (1, 2), (1, 3)], exists, &groups), None);   // a group seen only in part doesn't fold
        assert_eq!(chunks((3, 0), &Fixed(3)), 0..9);
    }

    #[test]
    fn a_lifted_group_folds_across_levels() {
        // L2 0 merged L1 0..2; L3 0, lifted, merged L2 0 and L1 2..4: it covers chunks 0..4
        let groups = Table(vec![((2, 0), run(1, 0..2)), ((3, 0), [run(2, 0..1), run(1, 2..4)].concat())]);
        assert_eq!(chunks((3, 0), &groups), 0..4);
        let exists = |(n, p): Node| [6u64, 4, 1, 1][n as usize] > p;   // six chunks, four L1s
        let f = vec![(2, 0), (1, 2), (1, 3), (0, 4), (0, 5)];
        let folded = fold(&f, exists, &groups).expect("a fold");
        assert_eq!(folded, [(3, 0), (0, 4), (0, 5)]);
        assert_eq!(unfold(&folded, &groups), Some(f.clone()));
        assert_eq!(fold(&[(2, 0), (1, 2)], exists, &groups), None);   // its group shown only in part
    }
}
