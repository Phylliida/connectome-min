"""The demo that drives the system: a synthetic session, a scripted model, and the loop.

    python3 demo.py            # the whole story: failures, retries, terminal debt, a window cut

`minisystem.py` is the system itself (log, projections, work derivation, guards, the request
projection, the failure policy). What lives here is a HOST: the session loop, the synthetic
messages, the scripted outcomes, and the printed narrative. A real host writes its own version of
`run`/`tick`; nothing in the system depends on this file.
"""
import connectome_min as cm
from config import DEMO, GRACE
from minisystem import ACTIONS, MAX_ATTEMPTS, Refused, archive, assert_coverage, attempts, clear_debt, cleared, demanded_chunks, emit, fails, frontier, merge_exclusions, messages, overlap_blocked, policy, protected_ids, protections, salience, spans, stalled, step, streak, strip_images, work
from llm.summarizer import MockModel, Summarizer


def run(log, model, window, reach, turns, start=0, fork=None, label=""):
    """Session loop: at most one compression call per MESSAGE (autoTickOnNewMessage), and only while the
    last plan was over target -- LAZY production: production mints an L1 at every chunk close, but a
    memory nobody will fold this turn is a model call spent early. The display is re-planned after each
    turn and the wall enforced after every plan. An ESCALATED plan also produces L1 requests for the
    uncovered foldable runs, which the NEXT turn derives past the holdback -- recorded before the wall check,
    because the solve produces whether or not the emission that follows is refused."""
    last_stamp, last, demand, pressure = None, None, (), False
    for n in range(turns):
        t = start + n
        for i in range(4):
            emit(log, "msg", t * 4 + i + 1, content(t, i))
            tick(log, model, t if fork == t and i == 0 else None, label, demand, pressure)
        a = archive(log)
        plan = cm.plan_controlled_frontier(a, frontier(log), window, reach or window,
                                           protected_ids(log, a))
        assert_coverage(a, plan["F"])                      # fatal, before anything is emitted
        demand = demanded_chunks(plan["produced"], a)
        pressure = plan["tokens"] > window - int(window * cm.SLACK)
        if frontier(log) != plan["F"]:
            emit(log, "frontier", dict(plan["F"]))
        wall = window * (1 + GRACE)
        if plan["tokens"] > wall:
            raise Refused(f"turn {t}: {plan['tokens']} tokens > wall {wall:.0f} "
                          f"(W={window} + {GRACE:.0%}) -- production raises OverBudgetError here",
                          plan, demand)
        stamp = (plan["branch"], plan["tokens"] > window)
        if stamp != last_stamp:                            # print only when the plan changes
            last_stamp = stamp
            flag = " OVER W (inside grace)" if plan["tokens"] > window else ""
            print(f"  {label}turn {t:>2}  plan {plan['branch']:<12} tokens={plan['tokens']:>5} "
                  f"dP={plan['perturbation']:>5}{flag}")
        last = plan
    return last


def tick(log, model, fork_here, label, demanded=(), pressure=True):
    """One tick is a DRAIN, like production's driveSpeculativeDrain (:4105 -> :4120-4166): one attempt per
    derivable unit, L1 before merge, the work set re-derived between attempts -- so a merge enqueued by a
    compress mints in the same tick, as production's does. And one tick is one ROUND in the distributed-
    systems sense: a unit gets at most ONE attempt per round, so a retry waits a round rather than
    tight-looping to terminal, and the `attempted` set is the round's fuel bound. `pressure` is this
    host's lazy-production gate: False mints no new L1s (demand still opens), and then the drain has
    nothing to drain -- which is the point of the gate."""
    attempted = set()
    forked = False
    while True:
        item = next((i for kind in ("L1", "merge")
                     for i in work(log, demanded, pressure)
                     if i[1] == kind and i[0] not in attempted), None)
        if item is None:
            return
        attempted.add(item[0])
        v = len(log)
        if fork_here is not None and item[1] == "L1" and not forked:
            forked = True                             # fault injection, one fork per turn: it lands
            emit(log, "msg", 900_000 + fork_here, [("text", "a forked branch lands mid-flight")])
        outcome = step(log, item, model, v)           # while the first call of the drain is out
        print(f"  {label}        {outcome:<15} {item[0][:30]:<30} "
              f"attempts={attempts(log, item[0])} streak={streak(log, item[0])}")


def content(t, i):
    """Synthetic messages with real shape, so salience, image stripping, the message cap and sharding bite.
    Tool traffic lands on EVEN turns only: a chunk's salience is its cheapest message's, and a tool block
    in every turn floored every chunk at 0.2, which left the salience term dead in the shipped run (AUDIT D7)."""
    if i == 1 and t % 2 == 0:
        return [("text", f"running m{t} " + "x " * 120),           # re-derivable chatter
                ("tool_use", '{"cmd": "grep", "pattern": "connectome", "path": "ref/"}')]
    if i == 2 and t % 2 == 0:
        return [("tool_result", "match: " + "y " * 200)]
    if i == 1:
        return [("text", f"running m{t} " + "x " * 100)]
    if i == 2:
        return [("text", f"steady m{t} " + "y " * 140)]
    if i == 3:
        return [("text", "see https://example.com/a/b\n```\nfold_me()\n```\n" + "z " * 55)]
    if t == 6:
        return [("text", "attachment: " + "doc " * 700)]           # > 2x target -> shards
    return [("image",), ("text", f"look at this m{t} " + "w " * 120)]


def demo():
    print("=== failure policy: one table, one bounded rule ===")
    for r in ACTIONS:
        entry = []
        emit(entry, "fail", "T", r)
        print(f"  {r:<15} -> {policy(entry, 'T', r):<10} "
              f"(after {MAX_ATTEMPTS} non-transient failures: terminal)")
    log = []
    # Scripted outcomes are keyed by the chunk they land on, so the keys move with the compressible
    # zone: the head and the tail chunks are never compressed, so the demo's first L1 is c3 and c3,
    # c4, c5, c6 are the only chunks that ever reach the boundary. README.md `## minisystem.py`.
    model = Summarizer(MockModel({"c3": ["refusal"] * 2,                  # a stale discard eats one,
                                        # one more refusal, then the source-only shape passes
                                  "c4": ["tool_call"] + ["truncated"] * 4,  # bounded -> terminal
                                  "c5": ["provider_error"] * 2,         # transient: no burn
                                  "c6": ["overflow"]}))                 # one reshape, then success
    emit(log, "protect", 9, 12, 0, 0, "operator pin")
    print(f"\n=== phase 1: W={DEMO['window']}, lazy production: calls only under pressure ===")
    try:
        end = run(log, model, window=DEMO["window"], reach=DEMO["reach"], turns=DEMO["turns"],
                  fork=DEMO["fork_turn"])
        print(f"  phase ends at {end['tokens']} tokens: W={DEMO['window']} and the enforced wall "
              f"is {DEMO['window'] * (1 + GRACE):.0f} (W + {GRACE:.0%} grace) -- and the "
              f"demand to mint arrived only under pressure, one turn before the fold used it")
    except Refused as refusal:
        print(f"  REFUSED: {refusal}")
    print(f"\n=== phase 2: the operator cuts the window to {DEMO['cut_window']} ===")
    demand = ()                                            # chunk ids the escalated plan produced
    try:
        run(log, model, window=DEMO["cut_window"], reach=DEMO["cut_window"],
            turns=DEMO["extra_turns"], start=DEMO["turns"])
    except Refused as refusal:
        print(f"  REFUSED: {refusal}")
        demand = tuple(sorted(refusal.demand))
        requested = [f"L1:{cid}" for cid in demand]
        blocked = [t for t in requested if stalled(log, t)]
        print(f"  demand path: the escalated plan produced {refusal.plan['produced']} -- "
              f"{requested or 'nothing'} would skip the holdback"
              + (f", blocked by terminal debt: {blocked}" if blocked else ""))

    print("\n=== derived state (nothing here is a store) ===")
    a = archive(log)
    stuck = sorted({t for t, _r in fails(log) if stalled(log, t)})
    raw = sum(c.tokens for c in a["chunks"] if f"L1:{c.id}" in stuck)
    zone = cm.raw_zone(a, protected_ids(log, a))
    print(f"  stalled work (terminal debt): {stuck}")
    print(f"  fold floor from it: {raw} tokens no plan can fold away until the debt is cleared")
    print(f"  compressible zone: {len(a['chunks']) - len(zone)} of {len(a['chunks'])} chunks "
          f"({len(zone)} protected by head/tail/pin: {sorted(zone, key=lambda c: int(c[1:]))})")
    print(f"  overlap-blocked chunks (a live L1 already owns their messages): "
          f"{overlap_blocked(log, a) or 'none -- no re-derivation moved a chunk boundary here'}")
    print(f"  merge candidates excluded (resolved span / wide for its level): "
          f"{merge_exclusions(log, a) or 'none -- every candidate spans 150 msgs or fewer'}")
    print(f"  failed attempts by reason: " + ", ".join(
        f"{r}x{sum(1 for _t, x in fails(log) if x == r)}" for r in
        sorted({r for _t, r in fails(log)})))
    per_msg = [round(salience(blocks), 2) for _seq, blocks in strip_images(messages(log))[:8]]
    print(f"  message salience (composition): {per_msg}")
    print(f"  -> chunk salience is its cheapest message: "
          f"{ {c.id: round(c.salience, 2) for c in a['chunks'][:4]} }")
    shards = [cid for cid, _f, _l, _t, _x, _s, group in spans(log) if group]
    print(f"  shards minted from the one long message: {shards}")
    print(f"  compression calls spent: {model.calls} (lazy: only while the last plan was over target)")
    u = model.usage()
    print(f"  model usage: {u['input_tokens']} in / {u['output_tokens']} out tokens "
          f"({type(model.model).__name__} bills the request's own byte estimate; HttpModel bills "
          f"what the provider reports)")

    print("\n=== restart: rebuild everything from the log alone ===")
    print(f"  messages={len(messages(log))} chunks={len(a['chunks'])} mints={len(a['summaries'])} "
          f"fails={len(fails(log))} protections={len(protections(log))} events={len(log)}")
    outstanding = work(log)
    print(f"  work still outstanding: {len(outstanding)} "
          f"unit{'s' if len(outstanding) != 1 else ''}; terminal debt survives: {stuck}")
    F = frontier(log)
    profile = {k: sum(1 for c in a["chunks"] if F.get(c.id, 0) == k) for k in set(F.values()) | {0}}
    pin = {cid: F.get(cid, 0) for cid in sorted(protected_ids(log, a))}
    print("  frontier (L0 = raw): " + " ".join(f"L{k}:{v}" for k, v in sorted(profile.items())))
    print(f"  protections honored (one predicate for pin/document/lock/level-bound): {pin}")

    print("\n=== clearing terminal debt: one event, and the clear that fires by itself ===")
    print(f"  automatic half: a unit that mints later is already cleared, no event needed "
          f"(`cleared(log, 'L1:c3')`={cleared(log, 'L1:c3')} -- :3441 clearQuarantineForCompressedChunk)")
    for target in stuck:
        print(f"  explicit half: {target} attempts={attempts(log, target)} "
              f"stalled={stalled(log, target)}; derivable work for it: "
              f"{[i[0] for i in work(log, demand) if i[0] == target]}")
        clear_debt(log, target, "operator: the request shape changed")
        print(f"                 clear_debt({target}) -> attempts={attempts(log, target)} "
              f"stalled={stalled(log, target)}, derivable again: "
              f"{[i[0] for i in work(log, demand) if i[0] == target]}")


if __name__ == "__main__":
    demo()
