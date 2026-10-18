//! Generic tools for an append-only sequence: the runs that close among the items between two bounds,
//! and an admission budget for items taken newest first.

/// The runs that close among items start..stop: each joins the current run, which closes once
/// full(run, its total size). A last run that isn't full stays open, for a later call to extend.
pub fn runs(start: u64, stop: u64, size: impl Fn(u64) -> f64, full: impl Fn(&[u64], f64) -> bool) -> Vec<Vec<u64>> {
    let (mut closed, mut run, mut total) = (vec![], vec![], 0.0);
    for i in start..stop {
        run.push(i);
        total += size(i);
        if full(&run, total) {
            closed.push(std::mem::take(&mut run));
            total = 0.0;
        }
    }
    closed
}

/// An admission budget: admit(n) says whether one more item of n bytes fits within `count` items
/// and `size` bytes in all, 0 meaning no limit, and counts it if it does.
pub struct Budget {
    count: usize,
    size: usize,
    kept: usize,
    kept_bytes: usize,
}

impl Budget {
    pub fn new(count: usize, size: usize) -> Budget {
        Budget { count, size, kept: 0, kept_bytes: 0 }
    }

    pub fn admit(&mut self, n: usize) -> bool {
        if 0 < self.count && self.count <= self.kept || 0 < self.size && self.size < self.kept_bytes + n {
            return false;
        }
        self.kept += 1;
        self.kept_bytes += n;
        true
    }
}
