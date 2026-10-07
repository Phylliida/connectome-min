


@dataclass(frozen=True)
class Chunk:
    id: int
    message_ids: Tuple[MessageId, ...]   # tuples, not lists — Chunk is immutable
    tokens: int
    salience: float
    group: Optional[str] = None


@dataclass(frozen=True)
class Buffer:                            # the open tail; never a Chunk
    ids: Tuple[MessageId, ...] = ()
    tokens: int = 0
    salience: float = 1.0

@dataclass(frozen=True)
class Acc:
    chunks: Tuple[Chunk, ...]
    buf: Buffer



