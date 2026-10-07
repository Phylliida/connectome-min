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

MAX_CHUNK_SIZE


def update_chunks(session_path):
    chunks =
    chunks_json = session_chunks_json(session_path, level=0)
    most_recent_chunk = Chunk(last_jsonl_entry(chunks_json) # would be nice if this gives None if input is None

    frontier_message = first_jsonl_entry(chunks_json) # todo: handle fork stuff
    frontier_message_offset = 0
    if most_recent_chunk:
        frontier_message = lookup_message(session_path, most_recent_chunk.end_message_id)
        frontier_message_offset most_recent_chunk.end_message_offset

    while True:



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
