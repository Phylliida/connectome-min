import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional
from bisect import bisect_left

UserId = NewType("UserId", str) # UUID string
SessionId = NewType("SessionId", str) # UUID string
MessageId = NewType("MessageId", long) # int counting up from 0 for first message

# --- Users ---

@dataclass
class User:
    id: UserId
    name: str
    display_name: str
    is_human: bool  # is it an LLM or a human
    context_length: Optional[int] = None
    use_reasoning: bool = False
    reasoning_length: Optional[int] = None
    api_path: Optional[str] = None
    provider: str = "anthropic"  # "anthropic" | "openai"


@dataclass
class UserStore:
    users: List[User] = field(default_factory=list)


# --- Messages ---

class MessageBlocksKind(Enum):
    TEXT = auto()
    TOOL_USE = auto()
    TOOL_RESULT = auto()
    ATTACHMENT = auto()
    IMAGE = auto()


@dataclass
class MessageBlocks:
    kind: MessageBlocksKind
    data: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Message:
    id: MessageId
    parent_id: Optional[MessageId]  # previous message
    user_id: UserId
    blocks: List[MessageBlocks]

@dataclass
class Chunk:
    start_message_id: MessageId
    start_message_offset: int
    end_message_id: MessageId
    end_message_offset: int


MESSAGES_JSON_PATH = "messages.jsonl"
SESSION_JSON_PATH = "session.json"
CHUNKS_JSON_PATH = "chunks/chunks{level}.jsonl"
def chunks_json_path(leve):
    return CHUNKS_JSON_PATH.format(level)

@dataclass
class Session:
    session_directory: Path
    parent_session: Optional[Session]
    parent_session_message_id: Optional[Path] # the message we forked from (if we are a fork)
    # messages are append-only

def ensure_exists(path):
    os.path.mkdirs(path, make parents too=True)
    touch file to make it real if it is empty
    return path

def session_messages_json(session_path: Path):
    return JsonlFile(ensure_exists(os.path.join(session_path, MESSAGES_JSON_PATH)), class_factory=idk some message factory)

def session_chunks_json(session_path: Path, level: int):
    return JsonlFile(ensure_exists(os.path.join(session_path, chunks_json_path(level)))

def lookup_message(session_path: Path, message_id: MessageId) -> Message:
    return Message(session_messages_json(session_path).find(key="id", value=message_id))

def get_new_message_id(session_path: Path) -> MessageId:
    last_entry = session_messages_json(session_path, Message).last()
    return MessageId(last_entry['id']) + MessageId(1) if last_entry else MessageId(0)

def add_message(session_path: Path, user_id: UserId, blocks: List[MessageBlocks]):
    id = self.get_new_message_id(session_path)
    message = Message(id=id, user_id=user_id, blocks=blocks)
    session_messages_json(session_path).append(message)

def message_to_str(message: Message):
    # todo: attachments
    return "\n".join([[image] if b.kind == MessageDataKind.IMAGE else (b.data or "") for b in message.data])

MAX_CHUNK_SIZE = somethin

def update_chunks(session_path):
    messages = session_messages_json(session_path)
    chunks = session_chunks_json(session_path, level=0)

    def generate_text(messages, max_size):
        for message in messages:
            message_str = message_to_str(message)
            for i in range(0, len(message_str), max_size):
                yield (message.id, i, i+size, message_str[i:i+size])

    remaining_messages = messages.iter()
    partial_text = []
    most_recent_chunk = chunks.last()
    if most_recent_chunk:
        remaining_messages = messages.iter_at(key="id", value=most_recent_chunk.end_message_id)
        partial_message = message_to_str(next(remaining_messages))[:most_recent_chunk.end_message_offset]
        if partial_message: partial_text.append((most_recent_chunk.end_message_id, most_recent_chunk.end_message_offset, partial_message))


    for message_id, start_offset, end_offset, message_text in itertools.chain(partial_text, generate_text(remaining_messages)):


        def rechunk(sources, size):
            it = chain.from_iterable(sized(p, size) for s in sources for p in s)
            while chunk := "".join(islice(it, size)):
                yield chunk



    for text in generate_text(remaining_messages):


    for remaining_text_piece in itertools.chain([])

        partial_me
        partial_text = [

    frontier_message = messages.first() # todo: handle fork stuff
    frontier_message_offset = 0

    if most_recent_chunk:
        frontier_message = messages.find(key="id", value=most_recent_chunk.end_message_id)
        frontier_message_offset = most_recent_chunk.end_message_offset
        chunk_text = message_to_str(most_recent_chunk)
    for message in messages.iter_at(key="id", value=frontier_message.id): # iterate starting with specified message
        message_str = message_to_str(message)
        if frontier_message_offset == message_str: # we have handled this message entirely, skip to next one
            frontier_message_offset = 0
        else: # we still need to handle this message
            # todo: trim stuff?
            remaining_str = message_str[frontier_message_offset:]
            total_current_chunk_str = "\n".join(strings_in_current_chunk + [remaining_str])
            if len(total_current_chunk_str) < MAX_CHUNK_SIZE: strings_in_current_chunk.append(remaining_str)
            else:


            if "\n".join(strings_in_current_chunk + [remaining_str])
            total_str =




import itertools

MAX_CHUNK_SIZE = 100  # whatever your context budget is


def update_chunks(messages, chunks):
    last = chunks.last()
    start_id, start_off = (last["end_message_id"], last["end_message_offset"]) if last \
        else (messages.first()["id"], 0)
    pieces = message_pieces(
        ((m["id"], message_to_str(m)) for m in messages.iter_at("id", start_id)),
        MAX_CHUNK_SIZE, first_offset=start_off)
    for chunk in full_chunks(pieces, MAX_CHUNK_SIZE):
        chunks.append(chunk_record(chunk))


def message_pieces(messages, max_size, first_offset=0):
    """(message_id, offset, text) slices of <= max_size, in order, from the frontier on."""
    return ((mid, off, text[off:off + max_size])
            for i, (mid, text) in enumerate(messages)
            for off in range(first_offset if i == 0 else 0, len(text), max_size))


def full_chunks(pieces, max_size):
    """Greedily pack pieces into <= max_size chunks; yield only FULL ones.
    The final open chunk is deferred until more pieces arrive."""
    size, gid, prev = 0, 0, None
    def group(piece):
        nonlocal size, gid, prev
        cost = len(piece[2]) + (prev is not None and piece[0] != prev)  # "\n" between messages
        if size and size + cost > max_size:
            gid, size, cost = gid + 1, 0, len(piece[2])  # overflow -> new group, no leading separator
        size += cost
        prev = piece[0]
        return gid
    groups = (list(g) for _, g in itertools.groupby(pieces, group))
    return (closed for closed, _ in itertools.pairwise(groups))  # every group but the last is full


def chunk_record(chunk):
    first, last = chunk[0], chunk[-1]
    return {"start_message_id": first[0], "start_message_offset": first[1],
            "end_message_id": last[0], "end_message_offset": last[1] + len(last[2])}


def chunk_text(chunk):
    """The committed text of a chunk; continuation pieces join directly, messages with "\n"."""
    return chunk[0][2] + "".join(("\n" if b[0] != a[0] else "") + b[2]
                                 for a, b in itertools.pairwise(chunk))


@dataclass
def Config:
    session_directory_path: Path



# we can look at most recent message and walk back until we find the most recent message in that chain that hasn't had a chunk map to it yet
# however, how do we know which chunk maps to it?
# chunks will be added in a different order than message ids (consider a fork that branches)
# we could probably simplify things by just making one file per fork
# if we are guaranteed just a linear chain of messages
# then chunks are also linear
# and we can just look at most recent chunk which gives us most recent message
# that's probably cleaner
# but I am interested in if this is possible
# so simplest thing to do is to
# 1) Store id of all messages in current chain in memory
# 2) walk one chunk at a time (starting from most recent) until we find chunk that starts and ends with one in our history
#     In a file with no branching this is very fast (immediately get most recent chunk)
#     In a file with branching, this could require lots of walking back
#     And then you'd kinda alternate and it would be fast again
#     Wheras, seperate file for each fork is very fast, but memory space grows large quickly
#     Could just store session and message id of parent in that case
#
# 3) that is most recent chunk
# 4) continue iterating forward, adding new chunks in our chain
# 5) eventually we can't add more and we are done


def update_chunks(session: Session)
