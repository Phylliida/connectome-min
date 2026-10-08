"""The inference boundary: everything about calling a model, and nothing about compaction --
the functional port of ../../llm. Same seam: nothing below depends on this package;
`connectome_min.py` imports only `config`, and `minisystem.py` takes a boundary as an argument
to `step()` and never names a model constructor from here.

The port's one idea: a model is a CLOSURE, `request -> (Reply, next_model)`, so the mock's
scripted queues ride the return value instead of being popped off a cell, and the Summarizer's
accounting is a frozen record threaded through `complete` rather than fields on an object.

    summarizer.py   request assembly, the terminal-disposition gate, the mock, the HTTP client,
                    the pre-send admission bound, and calibration from observed usage
    prompts/        the prompt text itself, as data
"""
