from typing import List
from enum import StrEnum


typedef Participant = str
typedef MessageID = long

FIRST_MESSAGE_ID = 0

class BlockKind(StrEnum):
    TEXT = 'text'
    THINKING = 'thinking'
    REDACTED_THINKING = 'redacted_thinking'
    TOOL_USE = 'tool_use'
    TOOL_RESULT = 'tool_result'
    IMAGE = 'image'
    DOCUMENT = 'document'
    AUDIO = 'audio'
    VIDEO = 'video'

@dataclass(frozen=True)
class Block:
    kind: BlockKind
    tool: Option[str] # tool_use: tool id; tool_result: toolUseId
    text: Option[str] # thinking text, message text, sometimes tool text
    data: Option[Any] # json data, usually for tools
    path: Option[str] # image, document, audio, or video path


@dataclass(frozen=True)
class Message:
    id: MessageID  # counts up from 0
    participant: Participant
    content: List[Block] # content is always at least one entry, empty arrays should turn into single text entry with blank str text


# L1 Merging -> convert raw transcripts into summaries.
# This needs diffrent prompting and handling from L2 and further

def session_messages_json(session_path: Path):
    return JsonlFile(ensure_exists(os.path.join(session_path, MESSAGES_JSON_PATH)), class_factory=idk some message factory)

def session_chunks_json(session_path: Path, level: int):
    return JsonlFile(ensure_exists(os.path.join(session_path, chunks_json_path(level)))

@dataclass(frozen=True)
class Memory:
    id: MessageID # each layer has ids counting up from zero
    children_ids: List[MessageID]
    children_levels: List[int] # 0 is messages, 1 is first memory layer

# gets id of first message not in the tail
# returns None if the tail takes up all messages
def get_first_message_in_tail_id(session_path, tail_tokens):
    tail = 0
    for message in session_messages_json(session_path).backwards():
        tail += estimate_message_tokens(message)
        if tail > tail_tokens: return message.id
    return FIRST_MESSAGE_ID


def form_L1_memory(messages: List[Message])

def form_memories(session_path, tail_tokens)
    # form level 0 memories
    messages = session_messages_json(session_path)
    chunks = session_chunks_json(session_path, level=1)

    first_message_in_tail_id = get_first_message_in_tail_id(session_path)

    most_recent_chunk = chunks.last()
    remaining_messages = next(messages.iter_at(key="id", value=most_recent_chunk.children_ids[-1])) if most_recent_chunk else messages.iter()

    def ready_for_memory(messages: List[Message]):
        token_count = sum(map(estimate_message_tokens, messages)) # could be made slightly more efficient by running sum as we grow accumulated but not worth
        last_is_tool_call = messages[-1].content[-1].kind == BlockKind.TOOL_USE # works because we guarantee one or more in content always
        return token_count >= CHUNK_TOKENS and not last_is_tool_use

    accumulated_messages = []
    for message in remaining_messages:
        if message.id >= first_message_in_tail_id: break # we are in tail, bail (this works because messages count up)
        accumulated_messages.append(message)
        if ready_for_memory(accumulated_messages):
            form_memory(accumulated_messages)
            accumulated_messages = []
    # leftover accumulated messages aren't yet ready for memory, they can wait until later calls'




def update_chunks(session_path):



    def generate_text(messages, max_size):
        for message in messages:
            message_str = message_to_str(message)
            for i in range(0, len(message_str), max_size):
                yield (message.id, i, message_str[i:i+size])

    # skip ahead if there are chunks available
    partial_text = [(
        most_recent_chunk.end_message_id,
        most_recent_chunk.end_message_offset,
        message_to_str(next(remaining_messages))[most_recent_chunk.end_message_offset:]
    )] if most_recent_chunk else []

    # convert (id, start_offset, chars) -> [(id, start_offset, char1), (id, start_offset+1, char2), ...] but in an efficient way that doesn't duplicate stuff'
    def bundle_offsets(messages_with_offsets):
        for message_id, start_offset, message_text in messages_with_offsets:
            yield zip(
                repeat(message_id, len(message_text)),
                range(start_offset, start_offset+len(message_text)),
                message_text)

    for chunk in batched(chain(
                        bundle_offsets(chain(partial_text, generate_text(remaining_messages)))
                        ), chunk_size):
        # only do complete chunks
        if len(chunk) == chunk_size:
            start_message_id, start+message_offset, _ = chunk[0]
            end_message_id, end_message_offset, _ = chunk[-1]
            new_chunk = Chunk(
                start_message_id=start_message_id,
                start_message_offset=start_message_offset,
                end_message_id=end_message_id,
                end_message_offset=end_message_offset+1 # +1 so we can do [:end]
            )
            chunks.append(new_chunk)










@dataclass(frozen=True)
class Config:                             # config.rs: JSON names are the field names; missing ones take these defaults
    # chunks and windows
    target_chunk_tokens: float = 3000.0           # targetChunkTokens: a chunk closes at this many text-only tokens; a merge's goal
    recent_window_tokens: float = 30000.0         # recentWindowTokens: newest tokens kept raw
    l1_holdback_chunks: int = 1                   # l1HoldbackChunks: newer chunks an L1 waits for
    max_live_images: int = 6                      # maxLiveImages: images the tail keeps, newest first
    max_live_image_bytes: int = 20 * 1024 * 1024  # maxLiveImageBytes
    # planning (planner.rs)
    slack: float = 0.1                            # folding over W stops at W * (1 - slack)
    unfold_below: float = 0.8                     # below W * unfold_below, unfold toward the target
    merge_threshold: int = 6                      # mergeThreshold: k, for merges by count; fixed for a session
    merge_by_count: bool = False                  # merge k at a time and never lift; on only from a session's start
    min_merge_memories: int = 3                   # fewest a merge by tokens or a lift takes (escalated: 2, once none is due at this)
    over_budget_grace: float = 0.02               # overBudgetGraceRatio
    # rendering
    summary_label: str = 'What do you remember from earlier?'   # summaryContextLabel
    summary_participant: str = 'Claude'           # summaryParticipant (the harness sets 'assistant')
    # memory formation (memory.rs)
    compression_model: str = ''                   # compressionModel: the summarizer, whose voice memories are in
    summary_target_tokens: float = 2000.0         # summaryTargetTokens: the length the instruction asks for
    compression_max_tokens: float | None = None   # compressionMaxTokens: a request's max_tokens when > 0
    min_chunk_chars: int = 200                    # minChunkCharsForLLM: text-only chunks under this get a stub
    recall_budget_tokens: float = 100_000.0       # compressionRecallBudgetTokens: recall pairs a request replays
    compression_image_bytes: int = 12 * 1024 * 1024   # maxCompressionImageBytes: inline image bytes a request keeps
    attempt_limit: int = 3                        # failed attempts at one memory before it waits for new messages
