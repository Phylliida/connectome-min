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
import emit
import mint
from calls import Host
from chunking import rebuild
from config import Config
from estimate import price
from state import State


def step(state: State, messages: list[dict], usage: float | None, budget: float, reserve: float, window: float,
         host: Host, cfg: Config) -> tuple[State, list | emit.Refusal, Exception | None]:
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
    return state._replace(calibration=state.calibration.observe(tokens))


def compile(state: State, budget: float, reserve: float, window: float, cfg: Config) -> tuple[State, list | emit.Refusal]:
    """compileFolds: ContextManager.compile with maxTokens `budget` and reserveForResponse `reserve`,
    retried once at maxTokens actual + reserve after a refusal for size whose total fits `window`.
    A refusal within its budget is structural, and no budget fixes it."""
    state, result = emit.select(state, budget - reserve, cfg)
    if isinstance(result, emit.Refusal) and result.actual > result.budget and result.actual + reserve <= window:
        state, result = emit.select(state, (result.actual + reserve) - reserve, cfg)     # as the library takes it back out
    return state, result


def drive(state: State, host: Host, cfg: Config) -> tuple[State, Exception | None]:
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

    ticked, error, length = state, None, lambda s: len(s.queue) + len(s.merges)
    while error is None:
        started, before = calls, length(ticked)
        ticked, error = mint.tick(ticked, host._replace(complete=complete), cfg)
        if calls != started or length(ticked) >= before:
            break
    if failed:
        ticked = ticked._replace(quarantine=ticked.quarantine & state.quarantine)
    return ticked, error
