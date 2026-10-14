"""Merges in @animalabs/context-manager 0.10.1 (executeMerge and the merge queue), reproduced exactly
in default mode. A merge consolidates a run of unmerged summaries of one level into one a level up,
with one call per tick; its failures so far shape what the next attempt asks (merge_request). After
every memory each level queues at most one run (enqueue_merges); a run that reaches
merge_attempt_limit is quarantined until another merge takes one of its members (sweep)."""
from itertools import chain, pairwise

from calls import Failure, Reply, call, written
from config import Config
from js import js_round, js_trim
from prompts import (MERGE_INSTRUCTION, NO_TOOLS, PLAIN_PROSE, Context, ask, leaves, pairs, raw, say,
                      without_tools)
from state import Entry, Merge, State
from turns import cap_images

SERVER_STREAK = 12                        # server errors in a row that count as one failed merge attempt


def merge(ctx: Context, state: State) -> tuple[State, Exception | None]:
    """executeMerge for the merge queue's head, one call, and what its result does to the merge, with
    tick's accounting of an error: one marked not retryable is a failed attempt, and so is each server
    error from the SERVER_STREAK-th on; any other escapes the tick, the merge still queued."""
    job = state.merges[0]
    counted = lambda: state._replace(merges=(job._replace(streak=job.streak + 1),) + state.merges[1:])
    match result := call(ctx.host, merge_request(ctx, state, job)):
        case Failure(retryable=False, kind=str()):
            return reject(state, 'provider_error', None, ctx.cfg), None
        case Failure(retryable=True, kind='server') if job.streak + 1 >= SERVER_STREAK:
            return reject(counted(), 'provider_error', None, ctx.cfg), None
        case Failure(retryable=True, kind='server'):
            return counted(), result.error
        case Failure():
            return state, result.error
        case Reply(verdict=('valid', _)) if js_trim(result.text):
            return merged(ctx, state, job, result), None
        case Reply(verdict=('valid', _)):
            return reject(state, 'unusable_empty', 'end_turn', ctx.cfg), None
        case Reply(verdict=(outcome, stop)):
            return reject(state, outcome, stop, ctx.cfg), None


def merged(ctx: Context, state: State, job: Merge, reply: Reply) -> State:
    """The merge's memory stored, its sources marked merged into it, and the merges it completes queued."""
    nodes = {s.id: s for s in state.entries}
    entry = written(state, job.level, job.sources, (nodes[job.sources[0]].span[0], nodes[job.sources[-1]].span[1]), reply, ctx.cfg)
    entries = tuple(s._replace(parent=entry.id) if s.id in job.sources else s for s in state.entries) + (entry,)
    return enqueue_merges(ctx, state._replace(entries=entries, merges=state.merges[1:]))


def merge_request(ctx: Context, state: State, job: Merge) -> dict:
    """executeMerge's request, shaped by the merge's failures so far. It recalls the summaries before
    the run whose sources aren't the run's messages (above L1 they are summaries, so only position
    leaves one out), on a budget halved for each failure, four times at most, to no less than 8000. It
    shows the run one level deeper (an L1's messages raw, a higher summary's children as recall
    pairs) or, once a reply was refused, the run itself; adds the no-tools line after a tool_use and
    the prose line after an empty reply; caps its images; and drops the tools once two attempts have
    failed and the latest stop reason was tool_use."""
    cfg, nodes, tries = ctx.cfg, {s.id: s for s in state.entries}, len(job.failures)
    sources = [nodes[i] for i in job.sources]
    under, start = {m for s in sources for m in leaves(s, nodes)}, ctx.position[sources[0].span[0]]
    prior = [s for s in state.entries if s.parent is None and under.isdisjoint(s.sources) and ctx.position[s.span[0]] < start]
    level, who = sources[0].level, cfg.summary_participant
    target = (pairs(sources, who) if job.refused else raw(state.messages[ctx.position[m]] for s in sources for m in s.sources) if level == 1
              else pairs((nodes[c] for s in sources for c in s.sources), who))
    shown = level if job.refused else level - 1
    instruction = MERGE_INSTRUCTION.format(level=job.level, target=cfg.summary_target_tokens, seen=f'the L{shown} memories above'
                                           if shown else 'the slices of recent experience above (raw conversation)') \
        + NO_TOOLS * (job.stop == 'tool_use') + PLAIN_PROSE * (job.outcome == 'unusable_empty')
    request, _ = ask(ctx, state, prior, max(8000, js_round(cfg.recall_budget_tokens * 0.5 ** min(tries, 4))), start,
                     target + [say(instruction)])
    request = {**request, 'messages': cap_images(request['messages'], cfg.compression_image_bytes)}
    return without_tools(request) if job.stop == 'tool_use' and tries >= 2 else request


def reject(state: State, outcome: str, stop: str | None, cfg: Config) -> State:
    """recordMergeRejection for the queue's head: record the failure; at merge_attempt_limit, quarantine
    the run."""
    job = (old := state.merges[0])._replace(failures=old.failures + ((outcome, stop),))
    if len(job.failures) >= cfg.merge_attempt_limit:          # a limit under 1 acts as 1
        return state._replace(merges=state.merges[1:], merge_quarantine=state.merge_quarantine | {job.sources})
    return state._replace(merges=(job,) + state.merges[1:])


def sweep(ctx: Context, state: State) -> State:
    """sweepPaidOffMergeQuarantine, before every tick: a quarantined run that another merge took a
    member of can never run again, so it is dropped and the merges are requeued."""
    unmerged = {s.id for s in state.entries if s.parent is None}
    stale = {run for run in state.merge_quarantine if not unmerged.issuperset(run)}
    return enqueue_merges(ctx, state._replace(merge_quarantine=state.merge_quarantine - stale)) if stale else state


def enqueue_merges(ctx: Context, state: State) -> State:
    """checkMergeThresholdRecursive: for each level, a merge over the run that contiguousMergeCandidates
    picks among its unmerged, unqueued summaries, unless that run is quarantined."""
    seq, queued = {m: k for k, m in enumerate(chain.from_iterable(state.chunks))}, {s for m in state.merges for s in m.sources}
    runs = (candidates([s for s in state.entries if s.level == level and s.parent is None and s.id not in queued], seq, ctx.cfg)
            for level in range(1, max((s.level for s in state.entries), default=0) + 1))
    return state._replace(merges=state.merges + tuple(
        Merge(level, run) for level, run in enumerate(runs, 2) if run and run not in state.merge_quarantine))


def candidates(unmerged: list[Entry], seq: dict, cfg: Config) -> tuple | None:
    """contiguousMergeCandidates: once merge_threshold summaries are unmerged, split them, ordered by
    chunk position, into runs that leave no gap, leaving out any whose span is too wide for its level.
    The first run that has merge_threshold members gives its first ones; a run with a newer run after
    it needs only two.

    The library orders each span's ends and extends a run to the furthest end so far. A summary covers
    a contiguous window of chunk order (a merge takes adjacent ones), and unmerged windows are
    disjoint, so neither ever acts."""
    k = cfg.merge_threshold
    if len(unmerged) < k:
        return None
    windows = sorted((seq[s.span[0]], seq[s.span[1]], s.id) for s in unmerged
                     if seq[s.span[1]] - seq[s.span[0]] <= cfg.merge_span_messages * k ** max(0, s.level - 3))
    gaps = [i for i, (a, b) in enumerate(pairwise(windows), 1) if b[0] != a[1] + 1]
    runs = [[sid for *_, sid in windows[i:j]] for i, j in pairwise([0, *gaps, len(windows)])]
    return next((tuple(run[:k]) for i, run in enumerate(runs) if len(run) >= k or i < len(runs) - 1 and len(run) >= 2), None)
