"""Differential test of fold.py against the real library's kv-stable picker (oracle_fold.mjs).

    python3 test_fold.py [sessions] [seed]

Each session grows a log turn by turn: chunks close outside a tail window whose size wanders (so
it later covers folded messages again), L1s arrive late, merges of k form, and the window, reach
and quality gap vary. Every compile's inputs go to both planners; levels, tokens, perturbation,
override, escalation, produced ranges and recall prices must match. The carried frontier is
committed as the library commits it: only for messages between the windows, only if the plan fits.
"""
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from fold import Config, Inputs, Leaf, Summary, fits, plan, recall_tokens

HERE = Path(__file__).resolve().parent


def session(r):
    cfg = Config(slack=r.choice([0.1, 0.1, 0.0, 0.3]), merge_threshold=r.choice([2, 3, 6, 6]),
                 reach_tokens=r.choice([None, None, 0, 30, 200, 1500]),
                 quality_gap_ratio=r.choice([0.35, 0.35, 0.0, 1.0, 20.0]))
    W = r.choice([150, 600, 1500, 4000])
    head, tail_budget = r.choice([0, 0, 1, 4]), r.choice([0, 50, 200, 600, 1500])
    tokens, salience, text = [], [], {}       # per message; text per summary id
    level_of, sources, parent = {}, {}, {}     # per summary id
    pending, chunked, levels, turns = [], head, {}, []

    def mint(level, srcs):
        sid = f'L{level}-{len(level_of)}'
        level_of[sid], sources[sid] = level, tuple(srcs)
        text[sid] = r.choice(['', 'é', '😀']) + 'w' * r.choice([r.randint(0, 40), r.randint(40, 1200)])
        return sid

    for _ in range(r.randint(3, 40)):
        for _ in range(r.randint(1, 5)):
            tokens.append(r.choice([0, r.randint(1, 60), r.randint(60, 400)]))
            salience.append(r.choice([1, 1, 0.2, 0.5, 0.7333333333333334, r.random(), 1.5, -0.25]))
        n = len(tokens)
        if r.random() < 0.3:
            tail_budget = r.choice([0, 50, 200, 600, 1500])
        if r.random() < 0.2:                   # near the session's size, so plans meet the wall
            W = max(1, round(sum(tokens) * r.uniform(0.3, 1.1)))
        elif r.random() < 0.1:
            W = r.choice([150, 600, 1500, 4000])
        start, total = 0, 0
        for i in reversed(range(n)):
            total += tokens[i]
            if total > tail_budget:
                start = i + 1
                break
        lo = min(head, n)
        hi = max(start, lo)
        while chunked + (size := r.randint(1, 5)) <= hi:
            pending.append([f'm{i}' for i in range(chunked, chunked + size)])
            chunked += size
        while pending and r.random() < 0.7:
            mint(1, pending.pop(0))
        for level in range(1, 9):
            orphans = [s for s in level_of if level_of[s] == level and s not in parent]
            while len(orphans) >= cfg.merge_threshold and r.random() < 0.8:
                sid = mint(level + 1, orphans[:cfg.merge_threshold])
                for child in orphans[:cfg.merge_threshold]:
                    parent[child] = sid
                orphans = orphans[cfg.merge_threshold:]

        summaries = [Summary(s, level_of[s], recall_tokens(text[s]), sources[s], parent.get(s)) for s in level_of]
        l1 = {m: s.id for s in summaries if s.level == 1 for m in s.sources}
        leaves = [Leaf(f'm{i}', i, tokens[i], levels.get(f'm{i}', 0), l1.get(f'm{i}'), salience[i]) for i in range(lo, hi)]
        inputs = Inputs(leaves, summaries, sum(tokens[:lo]), sum(tokens[hi:]), lo + n - hi, n - 1)
        if not fits(inputs.head_tokens + inputs.tail_tokens, W, cfg):
            continue                           # refused before the picker runs
        result = plan(inputs, W, cfg)
        turns.append((library_inputs(inputs, range(lo), range(hi, n), tokens, text, W, cfg), result))
        if fits(result.tokens, W, cfg):
            levels |= result.levels
        if result.produced:                    # demand lets held-back chunks through
            while pending and r.random() < 0.8:
                mint(1, pending.pop(0))
    return turns


def library_inputs(inputs, head, tail, tokens, text, W, cfg):
    """The PickerInputs selectAdaptive builds: the windows' messages ride along as pinned chunks."""
    window = lambda i: {'id': f'm{i}', 'sequence': i, 'rawTokens': tokens[i], 'currentResolution': 0,
                        'lockedByAgent': False, 'pinned': True}
    middle = [{'id': m.id, 'sequence': m.seq, 'rawTokens': m.tokens, 'currentResolution': m.level,
               'lockedByAgent': False, 'pinned': False, 'salience': m.salience,
               **({'l1Id': m.l1} if m.l1 else {})} for m in inputs.leaves]
    summaries = [{'id': s.id, 'level': s.level, 'tokens': s.tokens, 'recall': s.tokens, 'content': text[s.id],
                  'sourceLevel': s.level - 1, 'sourceIds': list(s.sources),
                  'sourceRange': {'first': s.sources[0], 'last': s.sources[-1]},
                  **({'parentId': s.parent} if s.parent else {})} for s in inputs.summaries]
    options = {'mergeThreshold': cfg.merge_threshold, 'qualityGapRatio': cfg.quality_gap_ratio,
               **({'reachTokens': cfg.reach_tokens} if cfg.reach_tokens is not None else {})}
    return {'chunks': middle + [window(i) for i in head] + [window(i) for i in tail],
            'head': [f'm{i}' for i in head], 'tail': [f'm{i}' for i in tail],
            'headTokens': inputs.head_tokens, 'tailTokens': inputs.tail_tokens, 'summaries': summaries,
            'budget': {'totalBudget': W, 'targetBudget': W * (1 - cfg.slack), 'slack': cfg.slack},
            'options': options}


def expected(result, recorded):
    override = result.branch if result.branch in ('infeasible', 'quality-gap') else None
    return {'levels': result.levels, 'tokens': result.tokens, 'planTokens': result.tokens,
            'perturbation': result.perturbation, 'override': override, 'escalated': result.escalated,
            'exhausted': result.escalated, 'produced': [list(p) for p in result.produced],
            'recall': [s['recall'] for s in recorded['summaries']]}


def main(count=200, seed=0):
    rng = random.Random(seed)
    sessions = [session(rng) for _ in range(count)]
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([[recorded for recorded, _ in turns] for turns in sessions], f)
    oracle = subprocess.run(['node', str(HERE / 'oracle_fold.mjs'), f.name], capture_output=True, text=True, check=True)
    Path(f.name).unlink()
    branches = {}
    for index, (line, turns) in enumerate(zip(oracle.stdout.splitlines(), sessions, strict=True)):
        for step, (library, (recorded, result)) in enumerate(zip(json.loads(line), turns, strict=True)):
            mine = expected(result, recorded)
            for key in mine:
                if mine[key] != library[key]:
                    (HERE / 'failing_session.json').write_text(json.dumps([[recorded]]))
                    print(f'session {index} turn {step} ({result.branch}): {key}\n  library {library[key]!r}\n  port    {mine[key]!r}')
                    return 1
            branches[result.branch] = branches.get(result.branch, 0) + 1
    print(f'{count} sessions, {sum(branches.values())} compiles: identical; branches {dict(sorted(branches.items()))}')
    return 0


if __name__ == '__main__':
    sys.exit(main(*map(int, sys.argv[1:])))
