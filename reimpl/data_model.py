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

class MessageDataKind(Enum):
    TEXT = auto()
    TOOL_USE = auto()
    TOOL_RESULT = auto()
    ATTACHMENT = auto()
    IMAGE = auto()


@dataclass
class MessageData:
    kind: MessageDataKind
    data: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Message:
    id: MessageId
    parent_id: Optional[MessageId]  # previous message
    user_id: UserId
    data: List[MessageData]


@dataclass
class Chunk:
    start_message_id: MessageId
    start_message_offset: int
    end_message_id: MessageId
    end_message_offset: int


@dataclass
class Session:
    session_id: SessionId
    session_path: Path
    parent_session_id: Optional[SessionId]
    parent_session_message_id: MessageId
    # messages are append-only

    def lookup_message(self, message_id: MessageId) -> Message:
        return Message(binary_search_jsonl(self.path, key="id", value=message_id))

    def get_new_message_id(messages: List[Message]) -> MessageId:
        last_entry = Message(last_jsonl_entry(self.session_path))
        return last_entry.id + MessageId(1) if last_entry else MessageId(0)

    def add_message(self, message: Message):



    messages: List[Message] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)


@dataclass
def Config:
    session_directory_path: Path


def lookup_message(messages: List[Message], message_id: MessageId) -> Message:
    # ids are monotonically increasing, so messages is sorted by id
    index = bisect_left(messages, message_id, key=lambda m: m.id)
    if index < len(messages) and messages[index].id == message_id:
        return messages[index]
    raise ValueError(f"Could not find message with id {message_id}")



def add_message(session: Session, user_id: UserId,
                parent_id: Optional[MessageId],
                data: List[MessageData]) -> Message:
    if parent_id is not None:
        lookup_message(session.messages, parent_id)  # validate it exists
    message = Message(id=get_new_message_id(session.messages),
                      parent_id=parent_id, user_id=user_id, data=data)
    session.messages.append(message)
    update_chunks(session)
    return message

def message_to_str(message: Message):
    # todo: attachments
    return "\n".join([[image] if b.kind == MessageDataKind.IMAGE else (b.data or "") for b in message.data])

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
