"""Superseded solver, kept for reference and regression comparison.

`relevance_cut.py` holds the phase-based ideal-cut solver (fold_pass A/B + pack fixpoint) that was
the default until the tiling DP replaced it. The two were validated against each other: byte-identical
ideal cuts and cascade plans on the four documented `connectome_min.py` runs and the demo session,
and 300 fuzzed archives with zero validity violations and zero feasibility regressions (the only
differences were near-target ties, half of which the DP lands closer). Comparison drivers:
/tmp/tiling-work/tiling_diff.py and tiling_fuzz.py.

Nothing in the artifact imports this package.
"""
