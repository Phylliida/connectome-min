"""Mutation test of the two differential tests: each mutant breaks one rule of the port, and the
test that covers that rule must then fail.

    python3 test_mutants.py [scenarios] [sessions] [name filter]

Each mutant is applied to a temporary copy of the port and its tests, which then run against the
real library (see CONTEXT_MANAGER in oracle.mjs). A mutant reported MISSED is a rule no generated
case exercises; BROKEN means its pattern no longer matches the source. Some mutants need a rare
exact boundary (the 2% budget grace, an image depth equal to its limit), hence the default 400
scenarios; a full run takes a few minutes.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ['estimate.py', 'chunking.py', 'fold.py', 'test_faithful.py', 'test_fold.py', 'oracle.mjs', 'oracle_fold.mjs']
ENTRY = os.environ.get('CONTEXT_MANAGER') or str((HERE / '../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/'
                                                          'node_modules/@animalabs/context-manager/dist/src/index.js').resolve())

# name: (test, file, [(old, new), ...]); each old must occur exactly once
MUTANTS = {
    # estimate.py, covered by test_faithful.py
    'banker rounding':           ('faithful', 'estimate.py', [('return (whole := math.floor(x)) + (x - whole >= 0.5)', 'return round(x)')]),
    'code-point lengths':        ('faithful', 'estimate.py', [("return len(s.encode('utf-16-le', 'surrogatepass')) // 2", 'return len(s)')]),
    'python number repr':        ('faithful', 'estimate.py', [('return number_len(value)', 'return len(repr(value))')]),
    'escapes not counted':       ('faithful', 'estimate.py', [('js_len(json.dumps(value, ensure_ascii=False))', 'js_len(value) + 2')]),
    'lone surrogates raw':       ('faithful', 'estimate.py', [('+ 5 * len(_LONE_SURROGATE', '+ 0 * len(_LONE_SURROGATE')]),
    'exponent sign dropped':     ('faithful', 'estimate.py', [('k + (k > 1) + 2 + len', 'k + (k > 1) + 1 + len')]),
    'small-number cutoff':       ('faithful', 'estimate.py', [('k + 2 - n if -6 < n <= 0', 'k + 2 - n if -7 < n <= 0')]),
    'dense share >=':            ('faithful', 'estimate.py', [('/ n > 0.12', '/ n >= 0.12')]),
    'probe in code points':      ('faithful', 'estimate.py', [("units[:4000].decode('utf-16-le', 'ignore')", 'text[:2000]')]),
    'bool counts as stamp':      ('faithful', 'estimate.py', [('type(stamped) in (int, float)', 'isinstance(stamped, (int, float))')]),
    'python sum of stamps':      ('faithful', 'estimate.py', [('reduce(add, map(raw_tokens, blocks), 0)', 'sum(map(raw_tokens, blocks))')]),
    'python strip':              ('faithful', 'estimate.py', [('line.strip(_JS_SPACE)', 'line.strip()')]),
    'salience floor 0':          ('faithful', 'estimate.py', [('max(0.2, 1 - 0.8 *', 'max(0.0, 1 - 0.8 *')]),
    'fences not external':       ('faithful', 'estimate.py', [("external += sum(js_len(fence) for fence in text.split('```')[1::2])", 'pass')]),
    'calibration alpha':         ('faithful', 'estimate.py', [('m + 0.2 *', 'm + 0.25 *')]),
    # chunking.py, covered by test_faithful.py
    'min 3 messages':            ('faithful', 'chunking.py', [('len(run) >= 4', 'len(run) >= 3')]),
    'no tool_use deferral':      ('faithful', 'chunking.py', [(" and 'tool_use' not in message.kinds", '')]),
    'no gap close':              ('faithful', 'chunking.py', [('if run and (full or run[-1] != i - 1):', 'if run and full:')]),
    'full last run dropped':     ('faithful', 'chunking.py', [('closed + [run] * full', 'closed')]),
    'holdback off by one':       ('faithful', 'chunking.py', [('- cfg.l1_holdback_chunks', '- cfg.l1_holdback_chunks - 1')]),
    'placeholder priced 0':      ('faithful', 'chunking.py', [('max(0, raw - IMAGE_PLACEHOLDER)', 'max(0, raw)')]),
    'strip depth >=':            ('faithful', 'chunking.py', [('0 < cfg.image_strip_depth_tokens < depth', '0 < cfg.image_strip_depth_tokens <= depth')]),
    'depth counts stripped':     ('faithful', 'chunking.py', [('depth += full', 'depth += out[-1]')]),
    'crossing >=':               ('faithful', 'chunking.py', [('if total > budget', 'if total >= budget')]),
    'no tail retreat':           ('faithful', 'chunking.py', [("return start - (0 < start < len(messages) and 'tool_result' in messages[start].kinds)", 'return start')]),
    'no head retreat':           ('faithful', 'chunking.py', [("end - (end > 0 and 'tool_use' in messages[end - 1].kinds)", 'end')]),
    'demand open end':           ('faithful', 'chunking.py', [('elif last is None: last = len(chunks) - 1', 'elif last is None: last = first')]),
    # fold.py's glue to the log, covered by test_faithful.py's capture of selectAdaptive
    'zone excludes head':        ('faithful', 'fold.py', [('lo + len(messages) - hi', 'len(messages) - hi')]),
    'tail may precede head':     ('faithful', 'fold.py', [('hi = max(chunking.recent_start(messages, multiplier, cfg), lo)', 'hi = chunking.recent_start(messages, multiplier, cfg)')]),
    'no grace in fits':          ('faithful', 'fold.py', [('math.floor(W * (1 + cfg.over_budget_grace))', 'W')]),
    'planner sees text-only':    ('faithful', 'fold.py', [('Leaf(m.id, i, priced[i]', 'Leaf(m.id, i, m.text_only')]),
    # fold.py's planner, covered by test_fold.py
    'no carried projection':     ('fold', 'fold.py', [('carried = project({leaf.id: leaf.level for leaf in leaves})', 'carried = {leaf.id: leaf.level for leaf in leaves}')]),
    'no suffix projection':      ('fold', 'fold.py', [('F = project({m.id: ideal[m.id]', 'F = ({m.id: ideal[m.id]')]),
    'shallowest-first':          ('fold', 'fold.py', [('sorted(leaves, key=lambda m: -F[m.id])', 'sorted(leaves, key=lambda m: F[m.id])')]),
    'projection ignores group':  ('fold', 'fold.py', [('and all(F[m] == level for m in members(node.id))', '')]),
    'projection on snapshot':    ('fold', 'fold.py', [('changed = False\n            for leaf in sorted(', 'changed, F0 = False, dict(F)\n            for leaf in sorted('),
                                                      ('and all(F[m] == level for m in members(node.id))', 'and all(F0[m] == level for m in members(node.id))')]),
    'phase B whenever > target': ('fold', 'fold.py', [('((True, target), (False, W))', '((True, target), (False, target))')]),
    'bands off by one':          ('fold', 'fold.py', [('math.floor(math.log(ratio) / math.log(k)) + 1', 'math.floor(math.log(ratio) / math.log(k))')]),
    'phase A ignores bands':     ('fold', 'fold.py', [('(not within_bands or min(cap[m] for m in members(node.id)) >= level)', 'True')]),
    'pack accepts ties':         ('fold', 'fold.py', [('abs(t - target) < abs(tokens - target)', 'abs(t - target) <= abs(tokens - target)')]),
    'pack ignores W':            ('fold', 'fold.py', [('if (t := size(layout(trial))) <= W and', 'if (t := size(layout(trial))) <= 10**18 and')]),
    'pack skipped':              ('fold', 'fold.py', [('    if tokens <= W:     # phase C', '    if False:     # phase C')]),
    'pack oldest-first':         ('fold', 'fold.py', [('for m in reversed(leaves) if ideal[m.id] == level', 'for m in leaves if ideal[m.id] == level')]),
    'pack unfolds two levels':   ('fold', 'fold.py', [('dict.fromkeys(members(node.id), level - 1)', 'dict.fromkeys(members(node.id), max(0, level - 2))')]),
    'pack past the target':      ('fold', 'fold.py', [('                if tokens < target:\n', '                if True:\n')]),
    'hold ignores gap':          ('fold', 'fold.py', [('and loss(carried) - ideal_loss <= gap:', ':')]),
    'priority by age only':      ('fold', 'fold.py', [('key=lambda m: (salience[m.id], m.seq)', 'key=lambda m: m.seq')]),
    'loss counts all depth':     ('fold', 'fold.py', [('(F[m.id] - cap[m.id]) for m in leaves', 'F[m.id] for m in leaves')]),
    'suffix linear scan':        ('fold', 'fold.py', [('bisect_left(range(len(changes)), True, key=lambda i: suffix(i)[2] <= P)',
                                                       'next((i for i in range(len(changes)) if suffix(i)[2] <= P), len(changes))')]),
    'kv_cost off by one unit':   ('fold', 'fold.py', [('zip(before, after))):])', 'zip(before, after))) + 1:])')]),
    'no salience clamp':         ('fold', 'fold.py', [('min(1, max(0, leaf.salience)) if math.isfinite(leaf.salience) else 1', 'leaf.salience')]),
    'escalated at target':       ('fold', 'fold.py', [('escalated = tokens > W', 'escalated = tokens > target')]),
    'demand when not escalated': ('fold', 'fold.py', [('if uncovered and escalated)]', 'if uncovered)]')]),
    'suffix ignores gap':        ('fold', 'fold.py', [('if (partial := size(units)) <= W and loss(F) - ideal_loss <= gap:', 'if (partial := size(units)) <= W:')]),
    'quality-gap as infeasible': ('fold', 'fold.py', [("'infeasible' if partial > W else 'quality-gap'", "'infeasible'")]),
    'recall label 8 tokens':     ('fold', 'fold.py', [('return 9 + math.ceil', 'return 8 + math.ceil')]),
}


def run(name, test, file, edits, counts):
    with tempfile.TemporaryDirectory() as copy:
        for f in FILES:
            shutil.copy(HERE / f, copy)
        path = Path(copy, file)
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                return 'BROKEN', name, f'pattern matches {text.count(old)} times: {old[:50]!r}'
            text = text.replace(old, new)
        path.write_text(text)
        result = subprocess.run([sys.executable, f'test_{test}.py', str(counts[test]), '0'], cwd=copy,
                                capture_output=True, text=True, env={**os.environ, 'CONTEXT_MANAGER': ENTRY})
        first = (result.stdout + result.stderr).strip().splitlines()[:1]
        return 'MISSED' if result.returncode == 0 else 'caught', name, first[0][:90] if first else ''


def main(scenarios=400, sessions=300, only=''):
    counts = {'faithful': int(scenarios), 'fold': int(sessions)}
    chosen = [(name, *spec) for name, spec in MUTANTS.items() if only in name]
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        results = list(pool.map(lambda m: run(*m, counts), chosen))
    for verdict, name, detail in results:
        print(f'{verdict}  {name:27} {detail}')
    failed = sum(verdict != 'caught' for verdict, _, _ in results)
    print(f'{len(results)} mutants, {len(results) - failed} caught')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
