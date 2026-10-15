"""The harness's pass (compaction-autobiographical's foldPass, compileFolds and driveTick) over a session
on disk. At each step boundary:
  1. each new message joins the log, and the chunks that can close do (append);
  2. the last step's real prompt size calibrates the price multiplier (observe);
  3. the context compiles at the budget less the response reserve; a refusal for size is retried once
     at that size if it plus the reserve fits the routed window (compile);
  4. memories form until one makes a call or none is due (drive).
"""
from itertools import count, takewhile

from calls import Host
from chunking import close_chunks
from config import Config
from emit import Refusal, select
from estimate import Calibration
from memory import tick
from store import Session


def step(session: Session, messages: list[dict], usage: float | None, budget: float, reserve: float, window: float,
         host: Host, cfg: Config) -> tuple[list | Refusal, Exception | None]:
    """The messages the log gained, the last step's usage (None without a new report), the compile and
    the memories. Returns the compile's result and the error a failed call raised, if one did."""
    for message in messages:
        append(session, message, cfg)
    calibration = current(session)
    if usage is not None:
        calibration = calibration.observe(usage)
    return compile(session, budget, reserve, window, calibration, cfg), drive(session, host, cfg)


def append(session: Session, message: dict, cfg: Config) -> None:
    session.append('messages', {'participant': message['participant'], 'content': message['content'], 'source': message.get('id')})
    close_chunks(session, current(session).multiplier, cfg)


def current(session: Session) -> Calibration:
    """The calibration the last compile left."""
    last = session.last('compiles') or {}
    return Calibration(last.get('multiplier', 1.0), last.get('armed', 0))


def compile(session: Session, budget: float, reserve: float, window: float, calibration: Calibration,
            cfg: Config) -> list | Refusal:
    """compileFolds: maxTokens `budget`, reserveForResponse `reserve`; a refusal for size whose total
    fits `window` is retried once at that total."""
    result = select(session, budget - reserve, calibration, cfg)
    if isinstance(result, Refusal) and result.actual > result.budget and result.actual + reserve <= window:
        result = select(session, (result.actual + reserve) - reserve, calibration, cfg)
    return result


def drive(session: Session, host: Host, cfg: Config) -> Exception | None:
    """driveTick: memories form until one makes a call or none forms."""
    memories = lambda: sum(takewhile(bool, (session.count(f'layer{n}') for n in count(1))))
    while True:
        before = memories()
        called, error = tick(session, host, cfg)
        if called or memories() == before:
            return error
