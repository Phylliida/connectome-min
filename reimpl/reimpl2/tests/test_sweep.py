"""Every simplifying edit of the port, each run through the differential tests the way test_mutants
runs its mutants: an edit no test fails on is code the tests don't need, or a rule they don't reach.

    python3 tests/test_sweep.py [scenarios] [sessions]

The edits: drop an and/or operand; take one branch of an if or a conditional expression; drop a
comprehension filter or a match case; unwrap sorted/list/tuple/set/frozenset/dict/reversed/abs/int/
float/bool/str or either side of a two-argument max/min; drop a slice; drop the first or last link
of a comparison chain; drop a try's handlers; replace a statement with pass. Each runs the tests
that reach the edited module, the one written for it first, until one fails. The edits in ACCEPTED
survive on purpose; any other survivor fails the sweep.
"""
import ast
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_mutants  # noqa: E402

TESTS = {                                 # the tests that reach each module, the one written for it first
    'estimate.py': ['faithful', 'l1', 'compile', 'fold', 'drive'], 'chunking.py': ['faithful', 'compile', 'l1', 'drive'],
    'fold.py': ['fold', 'compile', 'faithful', 'drive'], 'emit.py': ['compile', 'l1', 'drive'],
    'state.py': ['l1', 'compile', 'drive'], 'mint.py': ['l1', 'drive'], 'drive.py': ['drive'],
}
WRAPPERS = {'sorted', 'list', 'tuple', 'set', 'frozenset', 'dict', 'reversed', 'abs', 'int', 'float', 'bool', 'str'}
STATEMENTS = (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.If, ast.For, ast.With, ast.Try)
ACCEPTED = {                              # 'module: edit: source' -> why it survives
    "fold.py: drop operand 1: (now := size(layout(ideal))) <= W or now >= rendered":
        "the library's stop for a round that didn't shrink the render; no generated session reaches it (none in 100,000)",
    "fold.py: delete statement: rendered = now": "as above: without it no round counts as shrinking",
}


def edits(node):
    """(description, replacement) for each simplification of `node`; a list replaces a statement."""
    match node:
        case ast.BoolOp(op, values):
            for i in range(len(values)):
                rest = values[:i] + values[i + 1:]
                yield f'drop operand {i}', rest[0] if len(rest) == 1 else ast.BoolOp(op, rest)
        case ast.IfExp(_, body, orelse):
            yield 'take the if branch', body
            yield 'take the else branch', orelse
        case ast.If(_, body, orelse):
            yield 'if always', body
            yield 'if never', orelse or [ast.Pass()]
        case ast.Try(body=body):
            yield 'no handlers', body
        case ast.Call(ast.Name(name), [arg], _) if name in WRAPPERS:
            yield f'unwrap {name}', arg
        case ast.Call(ast.Name('max' | 'min' as name), [a, b], []):
            yield f'{name} is its first', a
            yield f'{name} is its second', b
        case ast.Subscript(value, ast.Slice()):
            yield 'drop slice', value
        case ast.Compare(left, ops, comparators) if len(ops) >= 2:
            yield 'drop first comparison', ast.Compare(comparators[0], ops[1:], comparators[1:])
            yield 'drop last comparison', ast.Compare(left, ops[:-1], comparators[:-1])
        case ast.comprehension(target, iter_, ifs, is_async):
            for i in range(len(ifs)):
                yield f'drop filter {i}', ast.comprehension(target, iter_, ifs[:i] + ifs[i + 1:], is_async)
        case ast.Match(subject, cases) if len(cases) >= 2:
            for i in range(len(cases)):
                yield f'drop case {i}', ast.Match(subject, cases[:i] + cases[i + 1:])
    if isinstance(node, STATEMENTS) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
        yield 'delete statement', [ast.Pass()]


def mutants(module: str) -> list[tuple[str, str]]:
    """(label, edited source) for each edit of `module`."""
    src = (HERE.parent / module).read_text()
    out = []
    for index, node in enumerate(ast.walk(ast.parse(src))):
        for which, (what, _) in enumerate(edits(node)):
            tree = ast.parse(src)
            target = list(ast.walk(tree))[index]
            new = list(edits(target))[which][1]
            for parent in ast.walk(tree):
                for field, value in ast.iter_fields(parent):
                    if value is target:
                        setattr(parent, field, new[0] if isinstance(new, list) else new)
                    elif isinstance(value, list) and any(v is target for v in value):
                        i = next(i for i, v in enumerate(value) if v is target)
                        value[i:i + 1] = new if isinstance(new, list) else [new]
            out.append((f'{module}: {what}: {ast.unparse(node)}', ast.unparse(ast.fix_missing_locations(tree))))
    return out


def survives(label: str, module: str, edited: str, counts: dict, env: dict) -> bool:
    original = (HERE.parent / module).read_text()
    for test in TESTS[module]:
        verdict, _, _ = test_mutants.run(label, test, module, [(original, edited)], counts, env)
        if verdict != 'MISSED':
            return False
    return True


def main(scenarios=400, sessions=300):
    counts = {'faithful': int(scenarios), 'fold': int(sessions), 'compile': int(sessions), 'l1': 5 * int(sessions), 'drive': int(sessions)}
    cache = os.environ.get('ORACLE_CACHE') or str(Path(tempfile.gettempdir(), 'connectome-oracle'))
    os.makedirs(cache, exist_ok=True)
    env = {**os.environ, 'CONTEXT_MANAGER': test_mutants.ENTRY, 'ORACLE_CACHE': cache}
    jobs = [(label, module, edited) for module in TESTS for label, edited in mutants(module)]
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        verdicts = list(pool.map(lambda j: survives(*j, counts, env), jobs))
    survivors = [label for (label, _, _), alive in zip(jobs, verdicts) if alive]
    for label in survivors:
        print(f'{"accepted" if label in ACCEPTED else "SURVIVED"}  {label[:160]}')
    unexpected = [label for label in survivors if label not in ACCEPTED]
    print(f'{len(jobs)} edits, {len(jobs) - len(survivors)} caught, {len(survivors) - len(unexpected)} accepted, {len(unexpected)} unexpected')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
