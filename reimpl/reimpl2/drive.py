"""The harness's use of the strategy (compaction-autobiographical's foldPass, compileFolds and
driveTick, in its index.ts), over the port. A pass, at each step boundary:

  1. replays what the log gained: each new message joins it and the strategy rebuilds its chunks
     and L1 queue (append);
  2. reports the last step's real prompt size to calibration (observe);
  3. compiles at the budget less the response reserve; a refusal for size, one whose reachable size
     exceeds its budget, is retried once at that size if it plus the reserve fits the routed window
     (compile);
  4. forms memory: ticks until one starts a call or leaves the queues no shorter, a refused compile
     included, since compressing is what lowers the next compile's floor; then releases the chunk
     quarantines a tick wrote while a call was failing retryably, a provider's failure rather than
     the request's (drive).

Not reproduced: what the harness writes to its session log from this state (planFolds,
appendMemory), and concurrency. Here a pass's ticks finish before the next pass begins, where the
harness runs them behind the turn.
"""
from typing import NamedTuple

import emit
import mint
from chunking import close_chunks, enqueue, l1_queue
from config import Config
from estimate import price


class State(NamedTuple):
    """The strategy's state between passes."""
    messages: tuple = ()                  # the log: {'id', 'participant', 'content'}, in order
    priced: tuple = ()                    # estimate.price of each
    chunks: tuple = ()                    # chunking.close_chunks's, in close order
    memory: mint.Memory = mint.Memory()
    session: emit.Session = emit.Session()


def step(state: State, messages: list[dict], usage: float | None, budget: float, reserve: float, window: float,
         host: mint.Host, cfg: Config) -> tuple[State, list | emit.Refusal, Exception | None]:
    """foldPass: the messages the log gained, the last step's usage (None without a new report), the
    compile and the ticks. Returns the compile's result and the error a tick threw, if one did."""
    for message in messages:
        state = append(state, message, cfg)
    if usage is not None:
        state = observe(state, usage)
    state, compiled = compile(state, budget, reserve, window, cfg)
    state, error = drive(state, host, cfg)
    return state, compiled, error


def append(state: State, message: dict, cfg: Config) -> State:
    """ContextManager.addMessage, whose onNewMessage rebuilds."""
    return rebuild(state._replace(messages=state.messages + (message,), priced=state.priced + (price(message),)), cfg)


def observe(state: State, tokens: float) -> State:
    """reportRealInputTokens: the last request's input, cache reads and cache writes."""
    return state._replace(session=state.session._replace(calibration=state.session.calibration.observe(tokens)))


def compile(state: State, budget: float, reserve: float, window: float, cfg: Config) -> tuple[State, list | emit.Refusal]:
    """compileFolds: ContextManager.compile with maxTokens `budget` and reserveForResponse `reserve`,
    retried once at maxTokens actual + reserve after a refusal for size whose total fits `window`.
    A refusal within its budget is structural, and no budget fixes it."""
    state, result = select(state, budget - reserve, cfg)
    if isinstance(result, emit.Refusal) and result.actual > result.budget and result.actual + reserve <= window:
        state, result = select(state, (result.actual + reserve) - reserve, cfg)     # as the library takes it back out
    return state, result


def select(state: State, W: float, cfg: Config) -> tuple[State, list | emit.Refusal]:
    """AutobiographicalStrategy.select: rebuild, compile, then queue what the plan demanded. Chunks
    demanded before are already queued or summarized, so only the new ones join."""
    state = rebuild(state, cfg)
    session, result = emit.select(state.messages, state.priced, state.chunks, state.memory.entries, state.session, W, cfg)
    queue = enqueue(state.memory.queue, session.demanded, mint.summarized(state.memory, state.chunks))
    return state._replace(session=session, memory=state.memory._replace(queue=queue)), result


def drive(state: State, host: mint.Host, cfg: Config) -> tuple[State, Exception | None]:
    """kickTick: driveTick, then releaseRetryableQuarantine. A quarantined chunk returns to the
    queue's front at every rebuild and is consumed without a call, so one tick per pass would spend
    every pass on it; the loop moves past it, still making at most one call."""
    calls, failed = 0, False

    def complete(request: dict) -> dict:
        nonlocal calls, failed
        calls += 1
        try:
            return host.complete(request)
        except Exception as error:
            failed = failed or getattr(error, 'retryable', None) is True
            raise

    memory, error, length = state.memory, None, lambda m: len(m.queue) + len(m.merges)
    while error is None:
        started, before = calls, length(memory)
        memory, error = mint.tick(state.messages, state.priced, state.chunks, memory, state.session.calibration.multiplier,
                                  host._replace(complete=complete), cfg)
        if calls != started or length(memory) >= before:
            break
    if failed:
        memory = memory._replace(quarantine=memory.quarantine & state.memory.quarantine)
    return state._replace(memory=memory), error


def rebuild(state: State, cfg: Config) -> State:
    """rebuildChunks: the chunks that can close do, and the L1 queue is cut afresh."""
    chunks = state.chunks + tuple(close_chunks(state.priced, state.chunks, state.session.calibration.multiplier, cfg))
    return queued(state._replace(chunks=chunks), cfg)


def queued(state: State, cfg: Config) -> State:
    queue = l1_queue(state.chunks, mint.summarized(state.memory, state.chunks), state.session.demanded, cfg)
    return state._replace(memory=state.memory._replace(queue=queue))
