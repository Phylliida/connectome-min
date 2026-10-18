"""The clean version's settings, at @animalabs/context-manager 0.10.1's defaults where the library has
the same setting (its name in each comment)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # chunks and windows
    target_chunk_tokens: int = 3000                 # targetChunkTokens
    recent_window_tokens: int = 30000               # recentWindowTokens: newest tokens kept raw
    l1_holdback_chunks: int = 1                     # l1HoldbackChunks: newest chunks that wait for a newer one
    max_live_images: int = 6                        # maxLiveImages: images the tail keeps, newest first
    max_live_image_bytes: int = 20 * 1024 * 1024    # maxLiveImageBytes
    # planning (planner.py)
    slack: float = 0.1                              # folding over W stops at the target, W * (1 - slack)
    unfold_below: float = 0.8                       # below W * unfold_below, unfold toward the target
    merge_threshold: int = 6                        # mergeThreshold: memories per merge, 2 or more; fixed for a session
    over_budget_grace: float = 0.02                 # overBudgetGraceRatio
    # rendering
    summary_label: str = 'What do you remember from earlier?'   # summaryContextLabel
    summary_participant: str = 'Claude'             # summaryParticipant (the harness sets 'assistant')
    # memory formation (memory.py)
    compression_model: str = ''                     # compressionModel: the summarizer, whose voice memories are in
    summary_target_tokens: int = 2000               # summaryTargetTokens: the length the instruction asks for
    compression_max_tokens: int | None = None       # compressionMaxTokens: the model's output ceiling, if lower
    min_chunk_chars: int = 200                      # minChunkCharsForLLM: text-only chunks under this get a stub
    recall_budget_tokens: int = 100_000             # compressionRecallBudgetTokens: recall pairs a request replays
    compression_image_bytes: int = 12 * 1024 * 1024   # maxCompressionImageBytes: inline image bytes a request keeps
    attempt_limit: int = 3                          # failed attempts at a memory before waiting for new messages
