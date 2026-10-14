"""The AutobiographicalStrategy options this port reads, at @animalabs/context-manager 0.10.1's
defaults for adaptive resolution with the kv-stable planner (the library's name in each comment).
The harness sets compression_model to the session's model and summary_participant to 'assistant'."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # chunks and windows
    target_chunk_tokens: int = 3000                 # targetChunkTokens
    recent_window_tokens: int = 30000               # recentWindowTokens: newest tokens kept raw, never chunked
    head_window_tokens: int = 0                     # headWindowTokens: oldest tokens kept raw, never chunked
    attachments_ignore_size: bool = True            # attachmentsIgnoreSize: size chunks by text_only
    l1_holdback_chunks: int = 1                     # l1HoldbackChunks: newest closes that wait for a newer close
    image_strip_depth_tokens: int = 30000           # imageStripDepthTokens; 0 turns each image limit off
    max_live_images: int = 6                        # maxLiveImages
    max_live_image_bytes: int = 20 * 1024 * 1024    # maxLiveImageBytes
    # planning
    slack: float = 0.1                              # compressionSlackRatio: target = W * (1 - slack)
    merge_threshold: int = 6                        # mergeThreshold: the pyramid's base, which sets the age bands
    reach_tokens: float | None = None               # kvStableReachTokens, the trust region P; None = W
    quality_gap_ratio: float = 0.35                 # kvStableQualityGapRatio
    over_budget_grace: float = 0.02                 # overBudgetGraceRatio
    # rendering
    summary_label: str = 'What do you remember from earlier?'   # summaryContextLabel
    summary_participant: str = 'Claude'             # summaryParticipant (the harness sets 'assistant')
    # memory formation
    compression_model: str = ''                     # compressionModel: the summarizer, whose voice memories are in
    summary_target_tokens: int = 2000               # summaryTargetTokens: the length the instruction asks for
    compression_max_tokens: int | None = None       # compressionMaxTokens: the model's output ceiling, if lower
    min_chunk_chars: int = 200                      # minChunkCharsForLLM: text-only chunks under this get a stub
    recall_budget_tokens: int = 100_000             # compressionRecallBudgetTokens: recall pairs a request replays
    curve_fallbacks: int = 3                        # compressionRefusalCurveFallbacks: recall variants tried
    context_budget_tokens: int = 200_000            # compressionContextBudgetTokens: a variant's input + output cap
    compression_image_bytes: int = 12 * 1024 * 1024   # maxCompressionImageBytes: inline image bytes a merge request keeps
    merge_attempt_limit: int = 5                    # mergeAttemptLimit: failed attempts before a merge is quarantined
    merge_span_messages: int = 1500                 # mergeMaxSourceSpanMessages: widest L1-L3 merge source, x k per level above
