"""Superseded solver, kept for reference and regression comparison -- the functional port.

`relevance_cut.py` holds the phase-based ideal-cut solver (fold_pass A/B + pack fixpoint) that was
the default until the tiling DP replaced it. The port keeps the phases as they were; the one change
is that the frontier map is a value each pass returns, instead of a dict both passes mutated.

Nothing in the artifact imports this package.
"""
