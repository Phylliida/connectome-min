"""The inference boundary: everything about calling a model, and nothing about compaction.

Nothing below depends on this package. `connectome_min.py` (the algorithm) imports only `config`;
`minisystem.py` (the system) takes a boundary as an argument to `step()` and never names a class
from here. That is the seam this directory makes explicit: swap the model, keep the memory.

    summarizer.py   request assembly, the terminal-disposition gate, the mock, the HTTP client,
                    the pre-send admission bound, and calibration from observed usage
    prompts/        the prompt text itself, as data
"""
