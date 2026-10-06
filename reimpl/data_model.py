import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

UserId = NewType("UserId", str)       # UUID string
MessageId = NewType("MessageId", int)


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


def add_user(store: UserStore, name: str, display_name: str,
             is_human: bool = True) -> User:
    user = User(id=UserId(str(uuid.uuid4())), name=name,
                display_name=display_name, is_human=is_human)
    store.users.append(user)
    return user


def edit_user(store: UserStore, user_id: UserId, **changes) -> User:
    for user in store.users:
        if user.id == user_id:
            for key, value in changes.items():
                if not hasattr(user, key):
                    raise ValueError(f"Unknown user field: {key}")
                setattr(user, key, value)
            return user
    raise ValueError(f"Could not find user with id {user_id}")


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
    pass  # TODO: define


@dataclass
class Session:
    # users and messages are append-only
    messages: List[Message] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)


def lookup_message(messages: List[Message], message_id: MessageId) -> Message:
    # TODO: ids are monotonically increasing, so binary search would work
    for message in messages:
        if message.id == message_id:
            return message
    raise ValueError(f"Could not find message with id {message_id}")


def get_new_message_id(messages: List[Message]) -> MessageId:
    # just increment ids
    return MessageId(0 if not messages else messages[-1].id + 1)


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


def update_chunks(session: Session) -> None:
    # TODO: fill in
    pass
