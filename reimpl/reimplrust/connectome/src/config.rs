//! The settings, at @animalabs/context-manager 0.10.1's defaults where the library has the same
//! setting (its name in each comment). As JSON, an object of the settings it changes, by the Python
//! port's field names (the same as here), and so a session's settings write it.

use smart_default::SmartDefault;

#[derive(Clone, Debug, SmartDefault, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Config {
    // chunks and windows
    #[default(3000.0)] pub target_chunk_tokens: f64,                 // targetChunkTokens
    #[default(30000.0)] pub recent_window_tokens: f64,               // recentWindowTokens: newest tokens kept raw
    #[default(1)] pub l1_holdback_chunks: u64,                       // l1HoldbackChunks: newest chunks that wait for a newer one
    #[default(6)] pub max_live_images: usize,                        // maxLiveImages: images the tail keeps, newest first
    #[default(20 * 1024 * 1024)] pub max_live_image_bytes: usize,    // maxLiveImageBytes
    // planning (planner.rs)
    #[default(0.1)] pub slack: f64,                                  // folding over W stops at the target, W * (1 - slack)
    #[default(0.8)] pub unfold_below: f64,                           // below W * unfold_below, unfold toward the target
    #[default(6)] pub merge_threshold: u64,                          // mergeThreshold: memories per merge by count, 2 or more, and per merge a memory records none of; fixed for a session
    pub merge_by_count: bool,                                        // merge merge_threshold memories at a time, as the library does, not the fewest (min_merge_memories or more) that hold target_chunk_tokens, and none across layers (memory.rs: lifted); on only from a session's start
    #[default(3)] pub min_merge_memories: u64,                       // memories a merge by tokens takes at least, a lifted one's too, 2 or more, but 2 when folding can't reach its target otherwise; may change at any time, as each merge records what it took
    #[default(0.02)] pub over_budget_grace: f64,                     // overBudgetGraceRatio
    // rendering
    #[default("What do you remember from earlier?")] pub summary_label: String, // summaryContextLabel
    #[default("Claude")] pub summary_participant: String,           // summaryParticipant (the harness sets 'assistant')
    // memory formation (memory.rs)
    pub compression_model: String,                                   // compressionModel: the summarizer, whose voice memories are in
    #[default(2000.0)] pub summary_target_tokens: f64,               // summaryTargetTokens: the length the instruction asks for
    pub compression_max_tokens: Option<f64>,                         // compressionMaxTokens: a memory request's max_tokens (its reasoning's too), where the library only lowers its 16000 (or 1.5 * summary_target_tokens) to it
    #[default(200)] pub min_chunk_chars: usize,                      // minChunkCharsForLLM: text-only chunks under this get a stub
    #[default(100_000.0)] pub recall_budget_tokens: f64,             // compressionRecallBudgetTokens: recall pairs a request replays
    #[default(12 * 1024 * 1024)] pub compression_image_bytes: usize, // maxCompressionImageBytes: inline image bytes a request keeps
    #[default(3)] pub attempt_limit: usize,                          // failed attempts at a memory before waiting for new messages
}


