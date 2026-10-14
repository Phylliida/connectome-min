"""Mutation test of the differential tests: each mutant breaks one rule of the port, and the test
that covers that rule must then fail.

    python3 tests/test_mutants.py [scenarios] [sessions] [name filter]

Each mutant is applied to a temporary copy of the port and its tests, which then run against the
real library (see CONTEXT_MANAGER in the oracles). First the unmutated port runs each test, in
SHARDS parallel shards, and must pass; those runs fill the oracle cache (library.py), kept in
ORACLE_CACHE or the system temp directory between suite runs, so a mutant that leaves a case's
payload unchanged reads the library's output instead of starting node. A test stops at its first
difference, so a caught mutant costs only the cases before it. A mutant reported MISSED is a rule
no generated case exercises; BROKEN means its pattern no longer matches the source. Some mutants
need a rare exact boundary (the 2% budget grace, an image depth equal to its limit), hence the
default 400 scenarios, and test_l1 runs five times `sessions`.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = ['config.py', 'js.py', 'blocks.py', 'estimate.py', 'chunking.py', 'cut.py', 'planner.py', 'state.py', 'turns.py',
        'emit.py', 'calls.py', 'prompts.py', 'merges.py', 'mint.py', 'drive.py']
TESTS = ['test_faithful.py', 'test_fold.py', 'test_compile.py', 'test_l1.py', 'test_drive.py', 'library.py', 'oracle.mjs', 'oracle_fold.mjs',
         'oracle_compile.mjs', 'oracle_l1.mjs', 'fixtures_l1.json', 'fixtures_drive.json']
ENTRY = os.environ.get('CONTEXT_MANAGER') or str((HERE / '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/'
                                                          'node_modules/@animalabs/context-manager/dist/src/index.js').resolve())

TIMEOUT = 600                             # seconds a mutant's test may take; the unmutated port's take under 120
SHARDS = 16                               # parallel runs of each test on the unmutated port

# name: (test, file, [(old, new), ...]); each old must occur exactly once
MUTANTS = {
    # estimate.py, covered by test_faithful.py
    'banker rounding':           ('faithful', 'js.py', [('return (whole := math.floor(x)) + (x - whole >= 0.5)', 'return round(x)')]),
    'code-point lengths':        ('faithful', 'js.py', [("return len(s.encode('utf-16-le', 'surrogatepass')) // 2", 'return len(s)')]),
    'python number repr':        ('faithful', 'js.py', [('case int() | float(): return js_number(value)', 'case int() | float(): return repr(value)')]),
    'escapes not counted':       ('faithful', 'js.py', [('json.dumps(value, ensure_ascii=False))', 'f\'"{value}"\')')]),
    'lone surrogates raw':       ('faithful', 'js.py', [('lambda c: f', 'lambda c: c[0] or f')]),
    'exponent sign dropped':     ('faithful', 'js.py', [("'-+'[n > 1]", "'-'[:n <= 1]")]),
    'small-number cutoff':       ('faithful', 'js.py', [('if -6 < n <= 0', 'if -7 < n <= 0')]),
    'index keys in place':       ('l1', 'js.py', [('for k in sorted(value, key=index)', 'for k in value')]),
    'dense share >=':            ('faithful', 'blocks.py', [('/ n > 0.12', '/ n >= 0.12')]),
    'probe in code points':      ('faithful', 'blocks.py', [("units[:4000].decode('utf-16-le', 'ignore')", 'text[:2000]')]),
    'bool counts as stamp':      ('faithful', 'blocks.py', [('type(stamped) in (int, float)', 'isinstance(stamped, (int, float))')]),
    'python sum of stamps':      ('faithful', 'blocks.py', [('reduce(add, map(raw_tokens, blocks), 0)', 'sum(map(raw_tokens, blocks))')]),
    'python strip':              ('faithful', 'js.py', [('    return s.strip(JS_SPACE)', '    return s.strip()')]),
    'salience floor 0':          ('faithful', 'blocks.py', [('max(0.2, 1 - 0.8 *', 'max(0.0, 1 - 0.8 *')]),
    'fences not external':       ('faithful', 'blocks.py', [("return js_len(text), sum(map(js_len, text.split('```')[1::2])) + sum(map(js_len, links))", 'return js_len(text), sum(map(js_len, links))')]),
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
    'no tail retreat':           ('faithful', 'chunking.py', [("tail - (0 < tail < len(messages) and 'tool_result' in messages[tail].kinds), prices", 'tail, prices')]),
    'no head retreat':           ('faithful', 'chunking.py', [("end - (end > 0 and 'tool_use' in messages[end - 1].kinds)", 'end')]),
    'demand in index order':     ('faithful', 'chunking.py', [('dict.fromkeys([first, last, *range(min(first, last), max(first, last) + 1)])', 'dict.fromkeys(range(min(first, last), max(first, last) + 1))')]),
    'demand from the first end': ('faithful', 'chunking.py', [('*range(min(first, last), max(first, last) + 1)', '*range(first, max(first, last) + 1)')]),
    'demand queues twice':       ('faithful', 'chunking.py', [(' and c not in queue)]', ')]')]),
    'demand queues summarized':  ('faithful', 'chunking.py', [('if c not in summarized and c not in queue)]', 'if c not in queue)]')]),
    'demand open end':           ('faithful', 'chunking.py', [('elif last is None: last = len(chunks) - 1', 'elif last is None: last = first')]),
    # planner.py's glue to the log, covered by test_faithful.py's capture of selectAdaptive
    'one ideal round':           ('fold', 'planner.py', [('    for _ in range(3):\n        tokens = total(ideal)', '    for _ in range(1):\n        tokens = total(ideal)')]),
    'rounds by the ledger':      ('fold', 'planner.py', [('if (now := size(view.layout(ideal))) <= W or', 'if (now := total(ideal)) <= W or')]),
    'zone excludes head':        ('fold', 'planner.py', [('max(1, len(view.prices) - len(view.leaves))', 'max(1, len(view.prices) - len(view.leaves) - view.head)')]),
    'no grace in fits':          ('faithful', 'planner.py', [('math.floor(W * (1 + cfg.over_budget_grace))', 'W')]),
    'planner sees text-only':    ('faithful', 'planner.py', [('Leaf(m.id, i, prices[i]', 'Leaf(m.id, i, m.text_only')]),
    # planner.py and cut.py, covered by test_fold.py
    'no carried projection':     ('fold', 'planner.py', [('carried = view.project({leaf.id: leaf.level for leaf in leaves})', 'carried = {leaf.id: leaf.level for leaf in leaves}')]),
    'no suffix projection':      ('fold', 'planner.py', [('F = view.project({m.id: ideal[m.id]', 'F = ({m.id: ideal[m.id]')]),
    'shallowest-first':          ('fold', 'cut.py', [('sorted(self.leaves, key=lambda m: -cut[m.id])', 'sorted(self.leaves, key=lambda m: cut[m.id])')]),
    'projection ignores group':  ('fold', 'cut.py', [('and all(cut[m] == level for m in self.members(node.id))', '')]),
    'projection on snapshot':    ('fold', 'cut.py', [('changed = False\n            for leaf in sorted(', 'changed, cut0 = False, dict(cut)\n            for leaf in sorted('), ('and all(cut[m] == level for m in self.members(node.id))', 'and all(cut0[m] == level for m in self.members(node.id))')]),
    'phase B whenever > target': ('fold', 'planner.py', [('((True, target), (False, W))', '((True, target), (False, target))')]),
    'bands off by one':          ('fold', 'planner.py', [('math.floor(math.log(ratio) / math.log(k)) + 1', 'math.floor(math.log(ratio) / math.log(k))')]),
    'phase A ignores bands':     ('fold', 'planner.py', [('(not within_bands or min(cap[m] for m in members(node.id)) >= level)', 'True')]),
    'pack accepts ties':         ('fold', 'planner.py', [('abs(t - target) < abs(tokens - target)', 'abs(t - target) <= abs(tokens - target)')]),
    'pack ignores W':            ('fold', 'planner.py', [('if (t := total(trial)) <= W and', 'if (t := total(trial)) <= 10**18 and')]),
    'pack skipped':              ('fold', 'planner.py', [('        for level in range(view.top, 0, -1):', '        for level in []:')]),
    'pack oldest-first':         ('fold', 'planner.py', [('for m in reversed(leaves) if ideal[m.id] == level', 'for m in leaves if ideal[m.id] == level')]),
    'pack unfolds two levels':   ('fold', 'planner.py', [('dict.fromkeys(members(node.id), level - 1)', 'dict.fromkeys(members(node.id), max(0, level - 2))')]),
    'pack past the target':      ('fold', 'planner.py', [('            if tokens < target:\n', '            if True:\n')]),
    'hold ignores gap':          ('fold', 'planner.py', [('and loss(view, carried, cap) - ideal_loss <= gap:', ':')]),
    'priority by age only':      ('fold', 'planner.py', [('key=lambda m: (m.salience, m.seq)', 'key=lambda m: m.seq')]),
    'loss counts all depth':     ('fold', 'planner.py', [('(F[m.id] - cap[m.id]) for m in view.leaves', 'F[m.id] for m in view.leaves')]),
    'suffix linear scan':        ('fold', 'planner.py', [('bisect_left(range(len(changes)), True, key=lambda i: suffix(i)[2] <= P)', 'next((i for i in range(len(changes)) if suffix(i)[2] <= P), len(changes))')]),
    'kv_cost off by one unit':   ('fold', 'cut.py', [('zip(before, after))):])', 'zip(before, after))) + 1:])')]),
    'escalated at target':       ('fold', 'planner.py', [('escalated = size(view.layout(ideal)) > W', 'escalated = size(view.layout(ideal)) > target')]),
    'demand when not escalated': ('fold', 'planner.py', [('if uncovered and escalated)]', 'if uncovered)]')]),
    'suffix ignores gap':        ('fold', 'planner.py', [('if (partial := size(units)) <= W and loss(view, F, cap) - ideal_loss <= gap:', 'if (partial := size(units)) <= W:')]),
    'quality-gap as infeasible': ('fold', 'planner.py', [("'infeasible' if partial > W else 'quality-gap'", "'infeasible'")]),
    'layout keeps negative tail': ('fold', 'cut.py', [("+ [('tail', tail)] * (tail > 0)", "+ [('tail', tail)]")]),
    'layout keeps negative head': ('fold', 'cut.py', [("return [('head', head)] * (head > 0) +", "return [('head', head)] +")]),
    'phases steer by layout':    ('fold', 'planner.py', [('        tokens = total(ideal)\n        for within_bands', '        tokens = size(view.layout(ideal))\n        for within_bands'), ('                        tokens = total(ideal)', '                        tokens = size(view.layout(ideal))'), ('if (t := total(trial)) <= W', 'if (t := size(view.layout(trial))) <= W')]),
    'escalation by the ledger':  ('fold', 'planner.py', [('escalated = size(view.layout(ideal)) > W', 'escalated = view.total(ideal) > W')]),
    'reported total from layout': ('fold', 'planner.py', [('Plan(F, view.total(F), pert', 'Plan(F, size(view.layout(F)), pert')]),
    'recall label unpriced':     ('fold', 'emit.py', [('return math.ceil(js_len(cfg.summary_label) / 4) + ', 'return ')]),
    # emit.py, covered by test_compile.py
    'answer keeps reasoning':    ('compile', 'emit.py', [("stripped = [b for b in memory.blocks or () if b['type'] not in REASONING]", 'stripped = list(memory.blocks or ())'), ("any(js_trim(b['text']) for b in stripped)", "any(b['type'] == 'text' and js_trim(b['text']) for b in stripped)"), ('recall_tokens(answers[s.id], cfg)', "recall_tokens([b for b in answers[s.id] if b['type'] == 'text'], cfg)")]),
    'answer ignores blocks':     ('compile', 'emit.py', [("stripped = [b for b in memory.blocks or () if b['type'] not in REASONING]", 'stripped = []')]),
    'blank prose renders':       ('compile', 'emit.py', [("return stripped if any(js_trim(b['text']) for b in stripped) else", 'return stripped if stripped else')]),
    'answer python trim':        ('compile', 'js.py', [('    return s.strip(JS_SPACE)', '    return s.strip()')]),
    'pair priced at content':    ('compile', 'emit.py', [('recall_tokens(answers[s.id], cfg)', "recall_tokens([{'type': 'text', 'text': s.content}], cfg)")]),
    'pair priced at tokens':     ('compile', 'emit.py', [('recall_tokens(answers[s.id], cfg)', 'max(recall_tokens(answers[s.id], cfg), recall_tokens([], cfg) + s.tokens if s.blocks else 0)')]),
    'pair priced as one block':  ('compile', 'emit.py', [("sum(math.ceil(js_len(b['text']) / 4) for b in answer)", "math.ceil(sum(js_len(b['text']) for b in answer) / 4)")]),
    'tail evicted oldest first': ('compile', 'emit.py', [('chunking.crossing(reversed(view.prices[tail_start:]), limit - total)', 'chunking.crossing(view.prices[tail_start:], limit - total)')]),
    'empty compile forgets keys': ('compile', 'emit.py', [('    if not entries:\n        return set(), previous\n', '')]),
    'empty turns split away':    ('compile', 'turns.py', [(" or not has(entry, 'tool_result')", '')]),
    'results searched to the end': ('compile', 'turns.py', [('entries[i + 1:i + 1 + LOOKAHEAD]', 'entries[i + 1:]')]),
    'null input priced as null': ('faithful', 'blocks.py', [("json_len({} if block.get('input') is None else block['input'])", "json_len(block['input'])")]),
    'big integers exact':        ('faithful', 'js.py', [('Decimal(repr(x := float(x)))', 'Decimal(repr(x))')]),
    'no trailing trim':          ('compile', 'emit.py', [("lambda e: has(e, 'tool_use') and not has(e, 'tool_result'), reversed(entries)))]", 'lambda e: False, reversed(entries)))]')]),
    'no tool pairing':           ('compile', 'emit.py', [('entries = strip_images(pair_tools(entries), ', 'entries = strip_images(entries, ')]),
    'duplicate results kept':    ('compile', 'turns.py', [("and b['toolUseId'] not in seen and not seen.add(b['toolUseId'])", '')]),
    'no result relocation':      ('compile', 'turns.py', [("for later in entries[i + 1:i + 1 + LOOKAHEAD] for b in later['content']", "for later in [] for b in later['content']")]),
    'results appended':          ('compile', 'turns.py', [("entry['content'] = results + entry['content']", "entry['content'] = entry['content'] + results")]),
    'stub entry after':          ('compile', 'turns.py', [("                entries.insert(i, {'participant': 'user', 'content': results, 'source': None})\n                continue", "                entries.insert(i + 1, {'participant': 'user', 'content': results, 'source': None})\n                i += 1\n                continue")]),
    'no final stubs':            ('compile', 'turns.py', [('    if uses:\n        entries.append(', '    if False:\n        entries.append(')]),
    'images unstripped':         ('compile', 'emit.py', [('entries = strip_images(pair_tools(entries), priced, multiplier, cfg)', 'entries = pair_tools(entries)')]),
    'images stripped oldest-first': ('compile', 'emit.py', [('key=lambda i: -pos(i))', 'key=lambda i: pos(i))')]),
    'image depth on post-strip': ('compile', 'emit.py', [('deep = chunking.crossing(reversed([store_tokens(m, multiplier) for m in priced])', 'deep = chunking.crossing(reversed(chunking.post_strip_tokens(priced, multiplier, cfg))')]),
    'no newest check':           ('compile', 'emit.py', [("if messages and not any(e['source'] == messages[-1]['id'] for e in entries):", 'if False:')]),
    'no tail orphan check':      ('compile', 'emit.py', [(" or tail and has(tail[0], 'tool_result') and not has(tail[0], 'tool_use')", '')]),
    'no stable-prefix marker':   ('compile', 'emit.py', [('({stable} if stable > last_head else set())', 'set()')]),
    'stable prefix ignores history': ('compile', 'emit.py', [('    if previous is None:\n        stable', '    if True:\n        stable')]),
    'no head marker':            ('compile', 'emit.py', [('| {last_head} - {-1} |', '|')]),
    'bundled turns unsplit':     ('compile', 'turns.py', [("user = entry['participant'].lower() == 'user' or not has(entry, 'tool_result')", 'user = True')]),
    'case-sensitive user':       ('compile', 'turns.py', [("entry['participant'].lower() == 'user'", "entry['participant'] == 'user'")]),
    'answer as Claude':          ('compile', 'emit.py', [("{'participant': cfg.summary_participant,", "{'participant': 'Claude',")]),
    'calibration not armed':     ('compile', 'emit.py', [('calibration=state.calibration.arm(plan.tokens)', 'calibration=state.calibration')]),
    'levels not committed':      ('compile', 'emit.py', [('levels=state.levels | plan.levels', 'levels=state.levels')]),
    'emission checks the total':  ('compile', 'emit.py', [('        if total + cost > room:', '        if False:')]),
    'no tail eviction':          ('compile', 'emit.py', [('evicted = chunking.crossing(reversed(view.prices[tail_start:]), limit - total) is not None', 'evicted = False')]),
    'demand dropped':            ('compile', 'emit.py', [('wanted = [c for first, last in plan.produced for c in chunking.demand(state.chunks, first, last)]', 'wanted = []')]),
    # mint.py, covered by test_l1.py
    'stub with tool blocks':     ('l1', 'mint.py', [("if all(b['type'] == 'text' for b in blocks) and sum(", 'if sum(')]),
    'stub counts untrimmed':     ('l1', 'mint.py', [("sum(js_len(js_trim(b['text'])) for b in blocks)", "sum(js_len(b['text']) for b in blocks)")]),
    'stub at the threshold':     ('l1', 'mint.py', [('for b in blocks) < cfg.min_chunk_chars', 'for b in blocks) <= cfg.min_chunk_chars')]),
    'merged recalled':           ('l1', 'mint.py', [('[s for s in state.entries if s.parent is None], cfg.recall_budget_tokens', 'list(state.entries), cfg.recall_budget_tokens')]),
    'recall in creation order':  ('l1', 'prompts.py', [('prior = sorted(prior, key=lambda s: ctx.position[s.span[0]])', 'prior = list(prior)')]),
    'recall cap breaks':         ('l1', 'prompts.py', [('else acc\n    return reduce(fit', 'else (acc[0], math.inf)\n    return reduce(fit')]),
    'recall cap no overhead':    ('l1', 'prompts.py', [('if acc[1] + s.tokens + 50 <= budget', 'if acc[1] + s.tokens <= budget')]),
    'covered rendered raw':      ('l1', 'prompts.py', [("for m in messages if m['id'] not in covered]", 'for m in messages]')]),
    'raw reasoning kept':        ('l1', 'prompts.py', [("'content': strip_reasoning(m['content'])}", "'content': m['content']}")]),
    'head uncalibrated':         ('l1', 'mint.py', [('chunking.head_end(state.priced, state.calibration.multiplier, cfg), host, cfg)', 'chunking.head_end(state.priced, 1.0, cfg), host, cfg)')]),
    'answer as Claude in mint':  ('l1', 'prompts.py', [('+ pairs(kept, cfg.summary_participant) +', "+ pairs(kept, 'Claude') +")]),
    'answer without reasoning':  ('l1', 'prompts.py', [('return list(s.blocks) if s.blocks else', 'return [b for b in s.blocks if b["type"] == "text"] if s.blocks else')]),
    'mixed turns unsplit':       ('l1', 'prompts.py', [('collapse([part for t in seen + task for part in split(t, False)])', 'collapse(seen + task)')]),
    'turns not joined':          ('l1', 'prompts.py', [('turns = drop_unpaired(collapse(', 'turns = drop_unpaired(list(')]),
    'joined without rule':       ('l1', 'turns.py', [('for b in [SEPARATOR] * (k > 0) + t[', 'for b in [] + t[')]),
    'unpaired tools kept':       ('l1', 'prompts.py', [('turns = drop_unpaired(collapse(', 'turns = (lambda t: t)(collapse(')]),
    'emptied turn dropped':      ('l1', 'turns.py', [("or [OMITTED] * bool(t['content'])}", '}')]),
    'defer ignores model':       ('l1', 'mint.py', [('if not ctx.host.tools and (_TOOLS_LESS.search(cfg.compression_model)', 'if not ctx.host.tools and (False')]),
    'defer ignores tool blocks': ('l1', 'mint.py', [("or any(b['type'] in TOOL_BLOCKS for t in request['messages'] for b in t['content'])):", "or any(False for t in request['messages'] for b in t['content'])):")]),
    'empty tool list declares':  ('l1', 'mint.py', [('if not ctx.host.tools and (', 'if ctx.host.tools is None and (')]),
    'blank text sent':           ('l1', 'prompts.py', [("for b in t['content'] if b['type'] != 'text' or js_trim(b['text'])])]", "for b in t['content']])]")]),
    'cache_control kept':        ('l1', 'prompts.py', [("{k: v for k, v in b.items() if k != 'cache_control'} for b in t['content']", "b for b in t['content']")]),
    'seams when capped':         ('l1', 'prompts.py', [('marks = set() if len(kept) < len(prior) else seams(', 'marks = seams(')]),
    'no head seam':              ('l1', 'prompts.py', [('return {found[0][0] - 1, found[-1][0] + 1,', 'return {found[-1][0] + 1,')]),
    'no deep seam':              ('l1', 'prompts.py', [(', max((i + 1 for i, level in found if level >= 2), default=-1)} - {-1}', '} - {-1}')]),
    'deep seam at L1':           ('l1', 'prompts.py', [('for i, level in found if level >= 2)', 'for i, level in found if level >= 1)')]),
    'header prefix match':       ('l1', 'prompts.py', [('(m := _HEADER.fullmatch(text))', '(m := _HEADER.match(text))')]),
    'ttl without seams':         ('l1', 'prompts.py', [("**({'cacheTtl': '1h'} if marks else {})", "**{'cacheTtl': '1h'}")]),
    'max tokens 1.4x':           ('l1', 'prompts.py', [('js_round(cfg.summary_target_tokens * 1.5)', 'js_round(cfg.summary_target_tokens * 1.4)')]),
    'max tokens uncapped':       ('l1', 'prompts.py', [('cap = c if (c := cfg.compression_max_tokens or 0) > 0 else math.inf', 'cap = math.inf')]),
    'empty system sent':         ('l1', 'prompts.py', [("**({'system': ctx.host.system} if ctx.host.system else {})", "**({'system': ctx.host.system} if ctx.host.system is not None else {})")]),
    'empty tools omitted':       ('l1', 'prompts.py', [("**({'tools': ctx.host.tools} if ctx.host.tools is not None else {})", "**({'tools': ctx.host.tools} if ctx.host.tools else {})")]),
    'no shape cap':              ('l1', 'mint.py', [('or sum(q[0] == ids for q in state.quarantine) >= SHAPE_CAP', 'or False')]),
    'shape cap 4':               ('l1', 'mint.py', [('SHAPE_CAP = 3 ', 'SHAPE_CAP = 4 ')]),
    'quarantine by chunk':       ('l1', 'mint.py', [('if family in state.quarantine or', 'if any(q[0] == ids for q in state.quarantine) or')]),
    'quarantine survives mint':  ('l1', 'mint.py', [('quarantine=frozenset(q for q in state.quarantine if q[0] != entry.sources))', 'quarantine=state.quarantine)')]),
    'no tool_use retry':         ('l1', 'mint.py', [("            case 'first', Reply(stop='tool_use'):\n", "            case 'first', Reply(stop='never'):\n")]),
    'no toolless retry':         ('l1', 'mint.py', [("            case 'no tools', Reply(stop='tool_use'):\n", "            case 'no tools', Reply(stop='never'):\n")]),
    'toolless keeps tools':      ('l1', 'mint.py', [('call(ctx.host, without_tools(lined))', 'call(ctx.host, lined)')]),
    'retry without the line':    ('l1', 'mint.py', [('call(ctx.host, lined := with_line(request, NO_TOOLS))', 'call(ctx.host, lined := request)')]),
    'no variants':               ('l1', 'mint.py', [('options = options or admitted(request, variants(ctx, state, request, kept), input_tokens(first.response), ctx.cfg)', 'options = iter(())')]),
    'variants oldest first':     ('l1', 'prompts.py', [('key=lambda s: -ctx.position[s.span[1]])', 'key=lambda s: ctx.position[s.span[1]])')]),
    'variants unchecked pairing': ('l1', 'prompts.py', [('        if well_paired(turns):', '        if True:')]),
    'variants unchecked order':  ('l1', 'prompts.py', [('if len(at) == 1 and all(ctx.position[a] < ctx.position[b] for a, b in pairwise(leaves(parent, nodes))):', 'if len(at) == 1:')]),
    'pair found by header':      ('l1', 'prompts.py', [("if header(wire[i]) == parent.id and wire[i + 1]['content'] == answer(parent)]", 'if header(wire[i]) == parent.id]')]),
    'children as Claude':        ('l1', 'prompts.py', [("pairs((nodes[c] for c in parent.sources), wire[at[0] + 1]['participant'])", "pairs((nodes[c] for c in parent.sources), 'Claude')")]),
    'admission by bytes always': ('l1', 'prompts.py', [('(bound(v) if provider is None else', '(bound(v) if True else')]),
    'admission counts rejected': ('l1', 'prompts.py', [('islice(filter(fits, options), cfg.curve_fallbacks)', 'filter(fits, islice(options, cfg.curve_fallbacks))')]),
    'no output reserve':         ('l1', 'prompts.py', [("+ request['config']['maxTokens'] <= cfg.context_budget_tokens", '<= cfg.context_budget_tokens')]),
    'no growth floor':           ('l1', 'prompts.py', [('provider + max(1, bound(v) - bound(request))', 'provider + (bound(v) - bound(request))')]),
    'bound in characters':       ('l1', 'prompts.py', [('len(js_json(r).encode())', 'len(js_json(r))')]),
    'cache writes uncounted':    ('l1', 'calls.py', [("n += (read('cacheCreationTokens') or 0) + (read('cacheReadTokens') or 0)", "n += (read('cacheReadTokens') or 0)")]),
    'cache reads uncounted':     ('l1', 'calls.py', [("n += (read('cacheCreationTokens') or 0) + (read('cacheReadTokens') or 0)", 'n += 0')]),
    'provider case-sensitive':   ('l1', 'calls.py', [("provider.lower() in ('anthropic', 'bedrock')", "provider in ('anthropic', 'bedrock')")]),
    'details ignored':           ('l1', 'calls.py', [("usage = details['usage'] if isinstance(details.get('usage'), dict) else obj(response.get('usage'))", "usage = obj(response.get('usage'))")]),
    'zero input counted':        ('l1', 'calls.py', [('    return n or None\n', '    return n\n')]),
    'no carrier retry':          ('l1', 'calls.py', [('    if isinstance(result, Failure) and carrier_rejected(result.error, request):', '    if False:')]),
    'carrier retry needs status': ('l1', 'calls.py', [("rejected = getattr(error, 'status', None) == 400 or", "rejected = getattr(error, 'status', None) == 400 or False and")]),
    'carrier retry unconditional': ('l1', 'calls.py', [("\n            and any(b['type'] in REASONING for t in request['messages'] for b in t['content']))", ')')]),
    'variant errors escape':     ('l1', 'mint.py', [("            case _, Failure() if rung != 'variant':", '            case _, Failure():')]),
    'variants judged by stop':   ('l1', 'calls.py', [('if stop not in STOPS or not isinstance(content, list) or not all(map(well_formed, content)):', 'if stop not in STOPS or not isinstance(content, list):'), ("'\\n'.join(b['text'] for b in content if b['type'] == 'text')", "'\\n'.join(b.get('text') or '' for b in content if b['type'] == 'text')")]),
    'malformed thinking ok':     ('l1', 'blocks.py', [("return isinstance(block.get('signature', ''), str)", 'return True')]),
    'no prose retry':            ('l1', 'mint.py', [("                rung, result = 'prose', call(ctx.host, with_line(request, PLAIN_PROSE))", '                return None')]),
    'prose retry of the winner': ('l1', 'mint.py', [('call(ctx.host, with_line(request, PLAIN_PROSE))', 'call(ctx.host, with_line(result.sent, PLAIN_PROSE))')]),
    'preamble kept':             ('l1', 'calls.py', [('    if m := _WRAPPED.match(text):', '    if m := None:')]),
    'unclosed thinking kept':    ('l1', 'calls.py', [("return '' if _OPENS.match(text) else text", 'return text')]),
    'preamble case-sensitive':   ('l1', 'calls.py', [("<thinking>.*?</thinking>[{JS_SPACE}]*', re.I | re.A | re.S)", "<thinking>.*?</thinking>[{JS_SPACE}]*', re.A | re.S)")]),
    'python whitespace':         ('l1', 'js.py', [('    return s.strip(JS_SPACE)', '    return s.strip()')]),
    'tokens chars/4':            ('l1', 'calls.py', [('else math.ceil(js_len(text) / 3)', 'else math.ceil(js_len(text) / 4)')]),
    'zero output tokens kept':   ('l1', 'calls.py', [('out if out and out > 0 else', 'out if out is not None else')]),
    'reasoning never kept':      ('l1', 'calls.py', [("blocks=tuple(b for b in content if b['type'] in REASONING + ('text',)) if any(b['type'] in REASONING for b in content) else None,", 'blocks=None,')]),
    'provenance of the request': ('l1', 'mint.py', [('written(state, 1, ids, (ids[0], ids[-1]), reply, cfg)', 'written(state, 1, ids, (ids[0], ids[-1]), reply._replace(sent=request), cfg)')]),
    'reply error keeps the chunk': ('l1', 'mint.py', [('    except Exception as error:            # reading a malformed reply; the library\'s tick catches that too\n        return state, error', '    except Exception as error:\n        return state._replace(queue=(index, *state.queue)), error')]),
    'failure read as a reply':   ('l1', 'mint.py', [('        case Failure(error=error):\n            return state, error\n', '')]),
    'error keeps the chunk':     ('l1', 'mint.py', [('        case Failure(error=error):\n            return state, error', '        case Failure(error=error):\n            return state._replace(queue=(index, *state.queue)), error')]),
    # mint.py's merges, covered by test_l1.py
    'L4 span unscaled':          ('l1', 'merges.py', [('* k ** max(0, s.level - 3))', '* k ** 0)')]),
    'rate limits end a streak':  ('l1', 'merges.py', [("Failure(retryable=True, kind='server') if job.streak + 1 >= SERVER_STREAK", 'Failure() if job.streak + 1 >= SERVER_STREAK')]),
    'server kind alone streaks': ('l1', 'merges.py', [("case Failure(retryable=True, kind='server') if", "case Failure(kind='server') if"), ("case Failure(retryable=True, kind='server'):", "case Failure(kind='server'):")]),
    'untyped errors counted':    ('l1', 'merges.py', [('case Failure(retryable=False, kind=str()):', 'case Failure(retryable=False):')]),
    'variants ignore adjacency': ('l1', 'prompts.py', [(" and (b['type'] != 'tool_result' or any(c.get('id') == b['toolUseId'] for c in turns[i - 1]['content']))", '')]),
    'text tool results malformed': ('l1', 'blocks.py', [("        case {'type': 'tool_result', 'toolUseId': str(), 'content': str()}: return True\n", '')]),
    'covered by any summary':    ('l1', 'prompts.py', [('cfg, covered = ctx.cfg, {m for s in state.entries if s.level == 1 for m in s.sources}', 'cfg, covered = ctx.cfg, {m for s in state.entries for m in s.sources}')]),
    'summarized by merges':      ('l1', 'state.py', [('l1s = {s.sources for s in self.entries if s.level == 1}', 'l1s = {s.sources for s in self.entries}')]),
    'negative output tokens kept': ('l1', 'calls.py', [('out if out and out > 0 else', 'out if out else')]),
    'reply blocks unfiltered':   ('l1', 'calls.py', [("blocks=tuple(b for b in content if b['type'] in REASONING + ('text',))", 'blocks=tuple(content)')]),
    'index keys unbounded':      ('l1', 'js.py', [('_INDEX.fullmatch(key) and int(key) < 2**32 - 1', '_INDEX.fullmatch(key)')]),
    'merge recall not halved':   ('l1', 'merges.py', [('cfg.recall_budget_tokens * 0.5 ** min(tries, 4)', 'cfg.recall_budget_tokens')]),
    'merge recall floor 0':      ('l1', 'merges.py', [('request, _ = ask(ctx, state, prior, max(8000, ', 'request, _ = ask(ctx, state, prior, max(0, ')]),
    'merge recalls later':       ('l1', 'merges.py', [(' and ctx.position[s.span[0]] < start]', ']')]),
    'merge recalls its run':     ('l1', 'merges.py', [('if s.parent is None and under.isdisjoint(s.sources) and ctx.position', 'if s.parent is None and ctx.position')]),
    'merge target filtered':     ('l1', 'merges.py', [('raw(state.messages[ctx.position[m]] for s in sources for m in s.sources)', 'raw((state.messages[ctx.position[m]] for s in sources for m in s.sources), {m for s in state.entries if s.level == 1 for m in s.sources})')]),
    'no refusal fallback':       ('l1', 'merges.py', [('target = (pairs(sources, who) if job.refused else', 'target = (pairs(sources, who) if False else'), ('shown = level if job.refused else', 'shown = level if False else')]),
    'fallback forgets refusals': ('l1', 'merges.py', [('target = (pairs(sources, who) if job.refused else', "target = (pairs(sources, who) if job.stop == 'refusal' else"), ('shown = level if job.refused else', "shown = level if job.stop == 'refusal' else")]),
    'refusal not sticky':        ('l1', 'state.py', [("return any(stop == 'refusal' for _, stop in self.failures)", "return bool(self.failures) and self.failures[-1][1] == 'refusal'")]),
    'fallback says deeper':      ('l1', 'merges.py', [('shown = level if job.refused else level - 1', 'shown = level - 1')]),
    'no merge no-tools line':    ('l1', 'merges.py', [("NO_TOOLS * (job.stop == 'tool_use')", 'NO_TOOLS * False')]),
    'no merge prose line':       ('l1', 'merges.py', [("PLAIN_PROSE * (job.outcome == 'unusable_empty')", 'PLAIN_PROSE * False')]),
    'merge toolless early':      ('l1', 'merges.py', [("job.stop == 'tool_use' and tries >= 2 else request", "job.stop == 'tool_use' and tries >= 1 else request")]),
    'merge never toolless':      ('l1', 'merges.py', [("return without_tools(request) if job.stop == 'tool_use' and tries >= 2 else request", 'return request')]),
    'merge images uncapped':     ('l1', 'merges.py', [("'messages': cap_images(request['messages'], cfg.compression_image_bytes)}", "'messages': request['messages']}")]),
    'images capped oldest first': ('l1', 'turns.py', [('for t in turns[::-1]][::-1]', 'for t in turns]')]),
    'nested images uncapped':    ('l1', 'turns.py', [("if isinstance(b.get('content'), list) else b", 'if False else b')]),
    'image cap exclusive':       ('l1', 'turns.py', [('if fits := kept + image_bytes(b) <= cap:', 'if fits := kept + image_bytes(b) < cap:')]),
    'blank merge no stop':       ('l1', 'merges.py', [("return reject(state, 'unusable_empty', 'end_turn', ctx.cfg), None", "return reject(state, 'unusable_empty', None, ctx.cfg), None")]),
    'merge reply judged by stop': ('l1', 'merges.py', [("case Reply(verdict=('valid', _)) if js_trim(result.text):", "case Reply(verdict=('valid' | 'provider_error', _)) if js_trim(result.text):"), ("case Reply(verdict=('valid', _)):", "case Reply(verdict=('valid' | 'provider_error', _)):")]),
    'merges ignore quarantine':  ('l1', 'merges.py', [('if run and run not in state.merge_quarantine))', 'if run))')]),
    'queued sources requeued':   ('l1', 'merges.py', [('s.parent is None and s.id not in queued]', 's.parent is None]')]),
    'no total threshold':        ('l1', 'merges.py', [('    if len(unmerged) < k:\n        return None\n', '')]),
    'no span limit':             ('l1', 'merges.py', [('if seq[s.span[1]] - seq[s.span[0]] <= cfg.merge_span_messages * k ** max(0, s.level - 3))', 'if True)')]),
    'runs ignore gaps':          ('l1', 'merges.py', [('if b[0] != a[1] + 1]', 'if False]')]),
    'interior runs wait':        ('l1', 'merges.py', [(' or i < len(runs) - 1 and len(run) >= 2), None)', '), None)')]),
    'newest run merges early':   ('l1', 'merges.py', [('or i < len(runs) - 1 and len(run) >= 2', 'or len(run) >= 2')]),
    'attempt limit off by one':  ('l1', 'merges.py', [('if len(job.failures) >= cfg.merge_attempt_limit:', 'if len(job.failures) > cfg.merge_attempt_limit:')]),
    'provider errors uncounted': ('l1', 'merges.py', [("        case Failure(retryable=False, kind=str()):\n            return reject(state, 'provider_error', None, ctx.cfg), None\n", '')]),
    'no server streak':          ('l1', 'merges.py', [("        case Failure(retryable=True, kind='server') if job.streak + 1 >= SERVER_STREAK:\n            return reject(counted(), 'provider_error', None, ctx.cfg), None\n        case Failure(retryable=True, kind='server'):\n            return counted(), result.error\n", '')]),
    'server streak 13':          ('l1', 'merges.py', [('SERVER_STREAK = 12 ', 'SERVER_STREAK = 13 ')]),
    'rate limits streak':        ('l1', 'merges.py', [("case Failure(retryable=True, kind='server') if", "case Failure(retryable=True, kind='server' | 'rate_limit') if"), ("case Failure(retryable=True, kind='server'):", "case Failure(retryable=True, kind='server' | 'rate_limit'):")]),
    'stop reason overwritten':   ('l1', 'state.py', [('return next((stop for _, stop in reversed(self.failures) if stop is not None), None)', 'return self.failures[-1][1] if self.failures else None')]),
    'merge dropped on error':    ('l1', 'merges.py', [('        case Failure():\n            return state, result.error', '        case Failure():\n            return state._replace(merges=state.merges[1:]), result.error')]),
    'sources stay unparented':   ('l1', 'merges.py', [('entries = tuple(s._replace(parent=entry.id) if s.id in job.sources else s', 'entries = tuple(s')]),
    'no merge cascade':          ('l1', 'merges.py', [('    return enqueue_merges(ctx, state._replace(entries=entries, merges=state.merges[1:]))', '    return state._replace(entries=entries, merges=state.merges[1:])')]),
    'L1s queue no merges':       ('l1', 'mint.py', [('    return enqueue_merges(ctx, state._replace(entries=state.entries + (entry,),', '    return (lambda c, m: m)(ctx, state._replace(entries=state.entries + (entry,),')]),
    'merge reasoning dropped':   ('l1', 'merges.py', [('nodes[job.sources[-1]].span[1]), reply, ctx.cfg)', 'nodes[job.sources[-1]].span[1]), reply, ctx.cfg)._replace(blocks=None)')]),
    'merge head uncalibrated':   ('l1', 'mint.py', [('return merge(ctx, state) if state.merges', 'return merge(ctx._replace(head=chunking.head_end(state.priced, 1.0, cfg)), state) if state.merges')]),
    'recall halved past four':   ('l1', 'merges.py', [('0.5 ** min(tries, 4)', '0.5 ** tries')]),
    'zero image cap applies':    ('l1', 'turns.py', [('return turns if cap <= 0 else', 'return')]),
    'no quarantine sweep':       ('l1', 'mint.py', [('    state = sweep(ctx, state)\n', '')]),
    'sweep without requeue':     ('l1', 'merges.py', [('return enqueue_merges(ctx, state._replace(merge_quarantine=state.merge_quarantine - stale)) if stale', 'return state._replace(merge_quarantine=state.merge_quarantine - stale) if stale')]),
    'merge provenance of the request': ('l1', 'merges.py', [('return merged(ctx, state, job, result), None', 'return merged(ctx, state, job, result._replace(sent=merge_request(ctx, state, job))), None')]),
    'answer as Claude in merge': ('l1', 'merges.py', [('level, who = sources[0].level, cfg.summary_participant', "level, who = sources[0].level, 'Claude'")]),
    # drive.py, covered by test_drive.py
    'failure flag resets':       ('drive', 'drive.py', [("failed = failed or getattr(error, 'retryable', None) is True", "failed = getattr(error, 'retryable', None) is True")]),
    'no compile retry':          ('drive', 'drive.py', [('    if isinstance(result, emit.Refusal) and result.actual > result.budget and result.actual + reserve <= window:', '    if False:')]),
    'retry ignores window':      ('drive', 'drive.py', [(' and result.actual + reserve <= window:', ':')]),
    'retry structural refusals': ('drive', 'drive.py', [('result.actual > result.budget and ', '')]),
    'retry at the budget':       ('drive', 'drive.py', [('select(state, (result.actual + reserve) - reserve, cfg)', 'select(state, result.budget, cfg)')]),
    'reserve kept':              ('drive', 'drive.py', [('state, result = emit.select(state, budget - reserve, cfg)', 'state, result = emit.select(state, budget, cfg)')]),
    'one tick per pass':         ('drive', 'drive.py', [('        if calls != started or length(ticked) >= before:\n            break', '        break')]),
    'ticks past a call':         ('drive', 'drive.py', [('if calls != started or length(ticked) >= before:', 'if length(ticked) >= before:')]),
    'no quarantine release':     ('drive', 'drive.py', [('    if failed:\n', '    if False:\n')]),
    'release on any failure':    ('drive', 'drive.py', [("failed = failed or getattr(error, 'retryable', None) is True", 'failed = True')]),
    'release clears all':        ('drive', 'drive.py', [('quarantine=ticked.quarantine & state.quarantine', 'quarantine=frozenset()')]),
    'queue ignores demand':      ('compile', 'chunking.py', [('state.summarized, state.demanded, cfg)', 'state.summarized, {}, cfg)')]),
    'compile without rebuild':   ('drive', 'emit.py', [('    state = chunking.rebuild(state, cfg)\n', '')]),
    'queue kept after compile':  ('compile', 'emit.py', [('queue=chunking.enqueue(state.queue, wanted, state.summarized))', 'queue=state.queue)')]),
    'ticks before compile':      ('drive', 'drive.py', [('    state, compiled = compile(state, budget, reserve, window, cfg)\n    state, error = drive(state, host, cfg)',
                                                         '    state, error = drive(state, host, cfg)\n    state, compiled = compile(state, budget, reserve, window, cfg)')]),
    'usage ignored':             ('drive', 'drive.py', [('    if usage is not None:\n        state = observe(state, usage)', '    if False:\n        state = observe(state, usage)')]),
    'appends without rebuild':   ('drive', 'drive.py', [('    return rebuild(state._replace(messages=', '    return (lambda s, c: s)(state._replace(messages=')]),
}


def run(name, test, file, edits, counts, env, shard='0/1'):
    """(verdict, name, the test's first output line): `test` on a copy of the port with `edits` applied
    to `file` is caught if it fails, MISSED if it passes, HUNG if it runs past TIMEOUT."""
    with tempfile.TemporaryDirectory() as copy:
        os.mkdir(Path(copy, 'tests'))
        for f in PORT:
            shutil.copy(HERE.parent / f, copy)
        for f in TESTS:
            shutil.copy(HERE / f, Path(copy, 'tests'))
        path = Path(copy, file)
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                return 'BROKEN', name, f'pattern matches {text.count(old)} times: {old[:50]!r}'
            text = text.replace(old, new)
        path.write_text(text)
        try:
            result = subprocess.run([sys.executable, f'tests/test_{test}.py', str(counts[test]), '0', shard], cwd=copy,
                                    capture_output=True, text=True, env=env, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return 'HUNG', name, f'still running after {TIMEOUT} s'
        first = (result.stdout + result.stderr).strip().splitlines()[:1]
        return 'MISSED' if result.returncode == 0 else 'caught', name, first[0][:90] if first else ''


def main(scenarios=400, sessions=300, only=''):
    counts = {'faithful': int(scenarios), 'fold': int(sessions), 'compile': int(sessions), 'l1': 5 * int(sessions), 'drive': int(sessions)}
    chosen = [(name, *spec) for name, spec in MUTANTS.items() if only in name]
    tests = sorted({test for _, test, *_ in chosen})
    cache = os.environ.get('ORACLE_CACHE') or str(Path(tempfile.gettempdir(), 'connectome-oracle'))
    os.makedirs(cache, exist_ok=True)
    env = {**os.environ, 'CONTEXT_MANAGER': ENTRY, 'ORACLE_CACHE': cache}
    start = time.monotonic()
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        # Unless the unmutated port passes, no verdict means anything; its runs also fill the cache.
        shards = [(test, f'{k}/{SHARDS}') for test in tests for k in range(SHARDS)]
        baseline = list(pool.map(lambda t: run('unmutated', t[0], 'config.py', [], counts, env, t[1]), shards))
        if failing := [f'{test} {shard}: {detail}' for (test, shard), (verdict, _, detail) in zip(shards, baseline) if verdict != 'MISSED']:
            print('the unmutated port fails', *failing, sep='\n  ')
            return 1
        checked = time.monotonic()
        results = list(pool.map(lambda m: run(*m, counts, env), chosen))
    for verdict, name, detail in results:
        print(f'{verdict}  {name:27} {detail}')
    failed = sum(verdict != 'caught' for verdict, _, _ in results)
    print(f'{len(results)} mutants, {len(results) - failed} caught; '
          f'{checked - start:.0f} s for the unmutated port, {time.monotonic() - checked:.0f} s for the mutants')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
