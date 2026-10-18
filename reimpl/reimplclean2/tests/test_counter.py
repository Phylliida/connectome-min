"""Property test of the lazy counter (planner.py) on generated frontiers: no oracle, its own guarantees.

    python3 tests/test_counter.py [sessions] [seed]

Chunks close turn by turn; L1s and merges of k appear with a lag; prices are random and the window
moves now and then. Each plan must
  - be a valid frontier: every closed chunk covered once, in order, by nodes that exist;
  - change nothing when the carried total is in [W * unfold_below, W];
  - after folding, end at or under the target, unless no complete group is left to fold;
  - after an unfold pass, stop only where unfolding the newest memory would pass the target;
  - keep levels falling toward the present.
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from config import Config  # noqa: E402
from planner import chunks, fold, plan, unfold  # noqa: E402


def check(F, carried, cost, exists, fixed, W, closed, cfg) -> str | None:
    total = lambda nodes: fixed + sum(map(cost, nodes))
    if [c for node in F for c in chunks(node, cfg.merge_threshold)] != list(range(closed)) or not all(map(exists, F)):
        return f'invalid frontier {F}'
    if W * cfg.unfold_below <= total(carried) <= W and F != carried:
        return 'changed inside the band'
    folded = total(carried) > W
    if folded and total(F) > W * (1 - cfg.slack) and fold(F, exists, cfg.merge_threshold):
        return 'stopped folding above the target with a group left'
    if (folded or total(carried) < W * cfg.unfold_below) and (trial := unfold(F, cfg.merge_threshold)) \
            and total(trial) <= W * (1 - cfg.slack):
        return 'stopped unfolding with room for the newest memory'
    if any(b[0] > a[0] for a, b in zip(F, F[1:])):
        return f'levels rise toward the present: {F}'
    return None


def session(r) -> str | None:
    k = r.choice([2, 3, 6])
    cfg = Config(merge_threshold=k, slack=r.choice([0.1, 0.0, 0.3]), unfold_below=r.choice([0.8, 0.5, 0.9]))
    W, have, prices, frontier = r.choice([2000, 10000, 40000]), [0], {}, []   # have[n]: memories at layer n (have[0]: chunks)
    cost = lambda node: prices.setdefault(node, r.randint(800, 3000) if node[0] == 0 else r.randint(50, 600))
    exists = lambda node: node[0] < len(have) and node[1] < have[node[0]]
    for _ in range(r.randint(5, 200)):
        have[0] += r.choice([0, 1, 1, 2])
        frontier = frontier + [(0, j) for j in range(sum(1 for n in frontier for _ in chunks(n, k)), have[0])]
        for n in range(len(have)):                            # each layer catches up with the one below, with a lag
            allowed = have[0] if n == 0 else have[n] // k
            if n + 1 == len(have) and allowed:
                have.append(0)
            if n + 1 < len(have):
                have[n + 1] = max(have[n + 1], allowed - r.choice([0, 0, 1, 2]))
        if r.random() < 0.05:
            W = max(500, round(W * r.uniform(0.5, 2)))
        fixed = r.randint(0, W // 4)
        result = plan(frontier, cost, exists, fixed, W, cfg)
        if message := check(result, frontier, cost, exists, fixed, W, have[0], cfg):
            return message
        frontier = result
    return None


def main(count=500, seed=0):
    for i in range(int(count)):
        if message := session(random.Random(f'{seed}/{i}')):
            print(f'session {i}: {message}')
            return 1
    print(f'{count} sessions: every plan holds the counter\'s guarantees')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
