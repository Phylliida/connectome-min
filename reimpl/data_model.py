from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


typedef UserId: str
typedef MessageId: long

@dataclass
class User:
    uuid: UserId
    name: str
    display_name: str
    is_human: bool # is it a llm or a human
    various other settings like context length, whether to use reasoning, reasoning length, api path, anthropic vs openai, etc.

class UserStore:
    users: List[User]


def add_user(store: UserStore, name: str, display_name: str):
    store.users.append(new uuid and fields)

def edit_user(etc. )

## Message

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
    parent_id: Optional[MessageId] # previous message
    user_id: UserId
    id: MessageId
    data: List[Block]

@dataclass
class Session:
    # users and messages are append only
    messages: List[Message]
    chunks: List[Chunk]

def lookup_message(messages: List[Message], message_id: Message):
    # todo: fill in binary search message ids
    if not found:
        raise ValueNotFoundError(f"Could not find message with id {message_id}")
    return message

def get_new_message_id(messages: List[Msg]):
    # just increment ids
    return 0 if len(messages) == 0: messages[-1].id+1

def add_message(session: Session, user_id: UserId, parent_id: Optional[MessageId], data: List[Block]):
    if not parent_id is None: lookup_message(parent_id)
    messages.append(Msg(parent_id=parent_id, id=get_new_message_id(session.messages), user=user, data=data))
    update_chunks(session)



def get_chunks()




## Chunks composed of messages
@dataclass
class Chunk:

