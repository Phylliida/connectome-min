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
pub fn budget(count: usize, size: usize) -> impl FnMut(usize) -> bool {
    let (mut kept, mut kept_bytes) = (0, 0);
    move |n| {
        let fits = !(0 < count && count <= kept || 0 < size && size < kept_bytes + n);
        if fits {
            (kept, kept_bytes) = (kept + 1, kept_bytes + n);
        }
        fits
    }
}
