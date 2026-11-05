#!/usr/bin/env python3
"""dsml_compare.py -- the DSML gate's TWO-SAMPLE read: A (reference, e.g. the final) vs B (e.g. the MK).

  dsml_compare.py A B [--labels final,mk] [--metrics raw,parsed,differ] [--allow-unequal] [--allow-missing]
                      [--json PATH] [--md PATH]
  dsml_compare.py --self-test

A and B are each one arm's `dsml/analyze.py grade` output, as any of:
  - a phase's selfparse.json (grade writes <phase>/selfparse.json),
  - a saved grade stdout (`analyze.py grade ... > <phase>.grade.txt`: its `PASS <phase> <case> <mode> sends=...
    defective raw=K/N parsed=K/N differ=K/N` row lines),
  - a directory holding either: every selfparse.json in it or one level down (one phase per subdirectory),
    else its *.grade.txt files (e.g. raw/s4-mk-b1/dsml/).
Only the REPEATS rows (T1_todo_nested and T1_todo_nested_strictfalse, one per stream / nonstream mode) carry
counts; they are matched by (case, mode). Per row and metric -- `raw` (defective calls in the model's own DSML),
`parsed` (defective returned arguments), `differ` (sends whose raw and parsed sides differ); K counts DEFECTS, so
lower is better -- it prints B - A with Newcombe's hybrid-score 95% CI, the two-sided Fisher exact p and the
one-sided Fisher p that B is worse, through quality/compare.py's own compare_counts.

analyze.py's own bar (TODO_BAR) is a one-arm CATASTROPHE gate against a fixed null: a 2 % defect passes it, and
final 1/470 vs MK 10/470 passes it while the two-sample read is +1.91 pp [+0.54, +3.67], Fisher p 0.011. This is
the regression read the user's bar asks for: both arms at equal n, prod sampling (T 1.0, top_p 1.0), a two-sample
test, never a fixed bar alone. Run it beside the grade, not instead of it.

The gate: a metric is WORSE when its CI excludes 0 in the bad direction or the one-sided Fisher p < 0.05 (no
multiplicity correction: 4 rows x 3 metrics, raw and parsed nearly the same count, so a false WORSE on a healthy
pair is possible; rerun both arms to settle one). Each row's arms must have sent the same number of requests
(`sends`; a send can carry more than one call, so the calls n may still differ, noted); unequal sends are refused
unless --allow-unequal, and a row present in only one arm is refused unless --allow-missing.
A metric with no calls in one arm and calls in the other (a zero-call collapse) is WORSE, and so is a run where
no comparison was made at all (every metric 0 calls in both arms): neither reads as a pass. A refusal still
prints whatever the compared rows found WORSE.
Exit 0 when nothing is WORSE, 2 when anything is, 3 on a refusal (unequal sends, a missing row; it outranks a
WORSE), 1 on bad input.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'quality'))
import compare  # noqa: E402
import qlib  # noqa: E402

METRICS = ('raw', 'parsed', 'differ')
EXIT_WORSE, EXIT_REFUSED = 2, 3
# print_todo_row's header line: "<verdict> <phase> <case> <mode> sends=%d with-call=%d defective raw=%d/%d
# parsed=%d/%d differ=%d/%d (...)"
ROW_RE = re.compile(r'^(\S+) +(\S+) +(\S+) +(\S+) +sends=(\d+) with-call=(\d+) defective raw=(\d+)/(\d+) '
                    r'parsed=(\d+)/(\d+) differ=(\d+)/(\d+)')


def rows_from_json(path):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for r in d.get('rows') or []:
        t = r.get('todo')
        if not t:
            continue        # a per-request case row: graded PASS/FAIL by analyze.py, no counts to compare
        rows.append({'case': r['case'], 'mode': r['mode'], 'phase': r.get('phase'),
                     'verdict': r.get('verdict') or ('PASS' if r.get('ok') else 'FAIL'), 'sends': r['sends'],
                     'raw': (t['raw']['defective'], t['raw']['calls']),
                     'parsed': (t['parsed']['defective'], t['parsed']['calls']),
                     'differ': (t['differ'], t['with_call']), 'src': path})
    return rows


def rows_from_txt(path):
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        m = ROW_RE.match(line)
        if m:
            g = m.groups()
            rows.append({'case': g[2], 'mode': g[3], 'phase': g[1], 'verdict': g[0], 'sends': int(g[4]),
                         'raw': (int(g[6]), int(g[7])), 'parsed': (int(g[8]), int(g[9])),
                         'differ': (int(g[10]), int(g[11])), 'src': path})
    return rows


def load_arm(path):
    """{(case, mode): row} from a selfparse.json, a grade stdout, or a directory of either. Raises ValueError."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'selfparse.json')) +
                       glob.glob(os.path.join(path, '*', 'selfparse.json')))
        if not files:
            files = sorted(glob.glob(os.path.join(path, '*.grade.txt')))
        if not files:
            raise ValueError(f'{path}: no selfparse.json (in it or one level down) and no *.grade.txt')
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ValueError(f'{path}: no such file or directory')
    arm = {}
    for f in files:
        rows = rows_from_json(f) if f.endswith('.json') else rows_from_txt(f)
        for r in rows:
            key = (r['case'], r['mode'])
            if key in arm:
                raise ValueError(f'{path}: {key[0]} {key[1]} graded twice ({arm[key]["src"]} phase '
                                 f'{arm[key]["phase"]}, {f} phase {r["phase"]}): name one phase')
            arm[key] = r
    if not arm:
        raise ValueError(f'{path}: no T1_todo_nested count rows in {", ".join(files)}')
    return arm


def compare_arms(arm_a, arm_b, metrics=METRICS, la='A', lb='B', allow_unequal=False, allow_missing=False):
    """{'rows': [...], 'refused': [...], 'notes': [...], 'gate': 'WORSE'|'pass'|'refused', 'worse': [...]}"""
    out = {'rows': [], 'refused': [], 'notes': [], 'worse': []}
    for key in sorted(set(arm_a) - set(arm_b)):
        out['refused' if not allow_missing else 'notes'].append(f'{key[0]} {key[1]}: only in {la}')
    for key in sorted(set(arm_b) - set(arm_a)):
        out['refused' if not allow_missing else 'notes'].append(f'{key[0]} {key[1]}: only in {lb}')
    for key in sorted(set(arm_a) & set(arm_b)):
        ra, rb = arm_a[key], arm_b[key]
        if ra['sends'] != rb['sends']:
            msg = f'{key[0]} {key[1]}: sends {la}={ra["sends"]} {lb}={rb["sends"]}'
            if not allow_unequal:
                out['refused'].append(msg + ' (unequal n)')
                continue
            out['notes'].append(msg + ' (unequal n allowed)')
        row = {'case': key[0], 'mode': key[1], 'sends': (ra['sends'], rb['sends']),
               'verdict': (ra['verdict'], rb['verdict']), 'metrics': {}}
        for name in metrics:
            (k1, n1), (k2, n2) = ra[name], rb[name]
            if not n1 and not n2:
                out['notes'].append(f'{key[0]} {key[1]} {name}: no calls in either arm ({la} {k1}/{n1}, {lb} {k2}/{n2})')
                continue
            if not n1 or not n2:
                # a zero-call collapse: one arm sent the case and never ended in a call -- no two-sample read is
                # possible, and it must not pass as one
                out['worse'].append(f'{key[0]} {key[1]} {name}: zero-call collapse, no calls in '
                                    f'{lb if not n2 else la} ({la} {k1}/{n1}, {lb} {k2}/{n2}; sends '
                                    f'{ra["sends"]}/{rb["sends"]})')
                continue
            if n1 != n2 and ra['sends'] == rb['sends']:
                out['notes'].append(f'{key[0]} {key[1]} {name}: same sends, n differs ({la} {n1}, {lb} {n2}: '
                                    f'sends with more than one call)')
            m = compare.compare_counts(k1, n1, k2, n2, 0.0, True, la, lb)
            row['metrics'][name] = m
            if m['gate'] == 'WORSE':
                out['worse'].append(f'{key[0]} {key[1]} {name}: ' + '; '.join(m['gate_why']))
        out['rows'].append(row)
    if not out['refused'] and not out['worse'] and not any(r['metrics'] for r in out['rows']):
        out['worse'].append(f'no comparison made: no metric of {",".join(metrics)} has calls in either arm of any '
                            f'row')
    out['gate'] = 'refused' if out['refused'] else ('WORSE' if out['worse'] else 'pass')
    return out


def render(res, la, lb, src_a, src_b):
    def c(p):
        return f"{p['k']}/{p['n']} ({100 * p['p']:.2f}%)"
    out = [f'## DSML two-sample gate: {la} (reference) vs {lb}', '',
           f'- {la}: {src_a}', f'- {lb}: {src_b}',
           '- K counts defects (lower is better); WORSE = Newcombe 95% CI excludes 0 upward, or one-sided Fisher '
           f'p (B worse) < {compare.GATE_ALPHA:g}', '',
           f'| case | mode | metric | {la} | {lb} | {lb} - {la} (pp) [Newcombe 95%] | Fisher p | Fisher p ({lb} worse) '
           f'| grade {la}/{lb} | gate |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in res['rows']:
        for name, m in r['metrics'].items():
            out.append(f"| {r['case']} | {r['mode']} | {name} | {c(m['a'])} | {c(m['b'])} | {100 * m['diff']:+.2f} "
                       f"[{100 * m['lo']:+.2f}, {100 * m['hi']:+.2f}] | {m['fisher_p']:.4g} | "
                       f"{m['fisher_p_worse']:.4g} | {r['verdict'][0]}/{r['verdict'][1]} | {m['gate']} |")
    out.append('')
    for n in res['notes']:
        out.append(f'- note: {n}')
    for n in res['refused']:
        out.append(f'- REFUSED: {n}')
    if res['gate'] == 'refused':
        out.append(f'DSML-COMPARE REFUSED: {len(res["refused"])} row(s) not comparable as asked (equal sends in both '
                   f'arms, every row in both; --allow-unequal / --allow-missing waive it)')
        if res['worse']:
            out.append(f'DSML-COMPARE WORSE (in the rows compared): {lb} is worse than {la} on ' +
                       '; '.join(res['worse']))
    elif res['gate'] == 'WORSE':
        out.append(f'DSML-COMPARE WORSE: {lb} is worse than {la} on ' + '; '.join(res['worse']))
    else:
        n = sum(len(r['metrics']) for r in res['rows'])
        out.append(f'DSML-COMPARE PASS: {n} comparison(s) over {len(res["rows"])} row(s), nothing WORSE')
    return '\n'.join(out)


def main(argv=None):
    ap = compare.ArgParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                           epilog=__doc__)
    ap.add_argument('a', nargs='?', help='reference arm (the final): selfparse.json, grade stdout, or a dir')
    ap.add_argument('b', nargs='?', help='arm under test (the MK)')
    ap.add_argument('--labels', default='A,B')
    ap.add_argument('--metrics', default=','.join(METRICS), help='any of raw,parsed,differ')
    ap.add_argument('--allow-unequal', action='store_true', help='compare rows whose arms sent different counts')
    ap.add_argument('--allow-missing', action='store_true', help='a row in only one arm is a note, not a refusal')
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    ap.add_argument('--self-test', action='store_true', help='run the offline unit tests and exit')
    a = ap.parse_args(argv)
    if a.self_test:
        r = unittest.main(argv=[sys.argv[0], '-v'], module=__name__, exit=False).result
        sys.exit(0 if r.wasSuccessful() else 1)
    if not a.a or not a.b:
        ap.error('need A and B (or --self-test)')
    metrics = [x for x in a.metrics.split(',') if x]
    bad = [x for x in metrics if x not in METRICS]
    if bad or not metrics:
        ap.error(f'--metrics: unknown {bad or "(none)"}; any of {",".join(METRICS)}')
    la, lb = (a.labels.split(',') + ['B'])[:2]
    try:
        arm_a, arm_b = load_arm(a.a), load_arm(a.b)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        sys.exit(f'dsml_compare.py: {type(e).__name__}: {e}')
    res = compare_arms(arm_a, arm_b, metrics, la, lb, a.allow_unequal, a.allow_missing)
    text = render(res, la, lb, a.a, a.b)
    print(text)
    if a.md:
        with open(a.md, 'w') as f:
            f.write(text + '\n')
    if a.json:
        qlib.write_json(a.json, {'labels': [la, lb], 'a': a.a, 'b': a.b, 'metrics': metrics, **res})
    sys.exit({'pass': 0, 'WORSE': EXIT_WORSE, 'refused': EXIT_REFUSED}[res['gate']])


# ----------------------------------------------------------------------------------------------- self-test
S4_DSML = os.path.join(HERE, '..', 'raw', 's4-mk-b1', 'dsml')


def synth_selfparse(path, counts, phase='dsml-X', sends=470, extra_rows=()):
    """A selfparse.json with one REPEATS row per (case, mode): counts = {(case, mode): (raw, parsed, differ)} as
    defect counts out of `sends` calls (or (k, n) tuples)."""
    def kn(v):
        return v if isinstance(v, tuple) else (v, sends)
    rows = list(extra_rows)
    for (case, mode), (raw, parsed, differ) in counts.items():
        (rk, rn), (pk, pn), (dk, dn) = kn(raw), kn(parsed), kn(differ)
        rows.append({'mode': mode, 'case': case, 'phase': phase, 'ok': True, 'verdict': 'PASS', 'sends': sends,
                     'why': [], 'info': [],
                     'todo': {'raw': {'calls': rn, 'defective': rk}, 'parsed': {'calls': pn, 'defective': pk},
                              'with_call': dn, 'differ': dk}})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'rows': rows, 'missing': []}, f)
    return path


def synth_grade_txt(path, arm):
    """The grade stdout's row lines (print_todo_row's format, tail abbreviated) for the rows of `arm`."""
    with open(path, 'w') as f:
        f.write('EXPECT-UNSET --expect-unset pass (given; no tree checked)\n')
        for r in arm.values():
            f.write('%s %-8s %-22s %-9s sends=%d with-call=%d defective raw=%d/%d parsed=%d/%d differ=%d/%d '
                    '(CATASTROPHE gate: ...)\n' % (r['verdict'], r['phase'], r['case'], r['mode'], r['sends'],
                                                  r['differ'][1], *r['raw'], *r['parsed'], *r['differ']))
            f.write('     raw: %d/%d call(s) defective (...)\n' % r['raw'])
        f.write('DSML-SELFPARSE PASS: ...\n')
    return path


FOUR = [('T1_todo_nested', 'stream'), ('T1_todo_nested', 'nonstream'),
        ('T1_todo_nested_strictfalse', 'stream'), ('T1_todo_nested_strictfalse', 'nonstream')]


class ComparePy(unittest.TestCase):
    """quality/compare.py's gate: equal n, one-sided Fisher, exit codes."""

    def run_cmp(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, 'quality', 'compare.py'), *args],
                              capture_output=True, text=True)

    def test_fisher_worse(self):
        # Fisher's tea table [[3, 1], [1, 3]]: one-sided p = 17/70; B ahead is p 1 - 1/70 the other way round
        self.assertAlmostEqual(compare.fisher_worse(3, 4, 1, 4), 17 / 70, places=12)
        self.assertAlmostEqual(compare.fisher_worse(1, 4, 3, 4), 69 / 70, places=12)
        # failures counted: B with MORE failures is the bad tail
        self.assertAlmostEqual(compare.fisher_worse(1, 4, 3, 4, lower_better=True), 17 / 70, places=12)
        self.assertEqual(compare.fisher_worse(5, 470, 5, 470, True), compare.fisher_worse(465, 470, 465, 470))
        self.assertAlmostEqual(compare.fisher_worse(0, 10, 0, 10, True), 1.0)
        # big n stays exact (comb(2638, 1319) overflows a float)
        p = compare.fisher_worse(1250, 1319, 1200, 1319)
        self.assertTrue(0 < p < 1e-4, p)

    def test_counts_gate(self):
        m = compare.compare_counts(1, 470, 10, 470, 0.03, lower_better=True)
        self.assertEqual(m['gate'], 'WORSE')
        self.assertAlmostEqual(100 * m['diff'], 1.91, places=2)
        self.assertAlmostEqual(100 * m['lo'], 0.54, places=2)
        self.assertAlmostEqual(100 * m['hi'], 3.67, places=2)
        self.assertAlmostEqual(m['fisher_p'], 0.0113, places=4)
        self.assertLess(m['fisher_p_worse'], 0.01)
        self.assertEqual(compare.compare_counts(5, 470, 5, 470, 0.03, True)['gate'], 'pass')
        # the better direction never gates
        self.assertEqual(compare.compare_counts(10, 470, 1, 470, 0.03, True)['gate'], 'pass')
        # one-sided Fisher alone (the CI still reaches past 0) gates: 3/470 vs 10/470, p 0.045
        m = compare.compare_counts(3, 470, 10, 470, 0.0, True)
        self.assertNotEqual(m['class'], 'worse')
        self.assertEqual(m['gate'], 'WORSE')
        self.assertEqual(len(m['gate_why']), 1)
        self.assertIn('one-sided Fisher', m['gate_why'][0])

    def test_counts_exit(self):
        r = self.run_cmp('counts', '1', '470', '10', '470', '--lower-better', '--name', 'dsml', '--labels', 'final,mk')
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('mk WORSE than final', r.stdout)     # the report text stands
        self.assertIn('+1.91 [+0.54, +3.67]', r.stdout)
        self.assertIn('GATE: WORSE', r.stdout)
        r = self.run_cmp('counts', '5', '470', '5', '470', '--lower-better')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('GATE: PASS', r.stdout)
        r = self.run_cmp('counts', '400', '470', '400', '470')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_counts_unequal(self):
        r = self.run_cmp('counts', '5', '470', '5', '472', '--lower-better')
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn('REFUSED', r.stderr)
        self.assertEqual(r.stdout, '')
        r = self.run_cmp('counts', '5', '470', '5', '472', '--lower-better', '--allow-unequal')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('unequal n allowed', r.stdout)
        r = self.run_cmp('counts', '1', '470', '10', '472', '--lower-better', '--allow-unequal')
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_usage_error_is_not_worse(self):
        r = self.run_cmp('counts', '1', '470', '10')
        self.assertEqual(r.returncode, 1, r.stderr)

    def summary(self, d, cell, k, n, ans_k, ans_n, extra=None):
        os.makedirs(d, exist_ok=True)
        s = {'suite': 'gsm8k', 'cell': cell, 'primary': 'exact', 'n_items': ans_n,
             'metrics': {'answered': qlib.prop(ans_k, ans_n), 'exact': qlib.prop(k, n), **(extra or {})}}
        with open(os.path.join(d, f'gsm8k-{cell}.summary.json'), 'w') as f:
            json.dump(s, f)

    def test_dirs_gate(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = os.path.join(t, 'a'), os.path.join(t, 'b')
            self.summary(a, 't0', 450, 500, 500, 500)
            self.summary(b, 't0', 449, 500, 500, 500)
            r = self.run_cmp('dirs', a, b, '--json', os.path.join(t, 'c.json'))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(t, 'c.json')) as f:
                self.assertEqual(json.load(f)['gate'], 'pass')
            # B lost 30 requests: same 500 items sent, exact n 470 -- compared (noted), answered gates WORSE
            self.summary(b, 't0', 420, 470, 470, 500)
            r = self.run_cmp('dirs', a, b)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn('metric n differs', r.stdout)
            self.assertIn('answered:', r.stdout)
            # B ran 400 items: refused
            self.summary(b, 't0', 360, 400, 400, 400)
            r = self.run_cmp('dirs', a, b)
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            self.assertEqual(self.run_cmp('dirs', a, b, '--allow-unequal').returncode, 0)
            # a secondary metric worse warns but gates only under --gate-all
            self.summary(a, 't0', 450, 500, 500, 500, {'truncated': qlib.prop(2, 500)})
            self.summary(b, 't0', 450, 500, 500, 500, {'truncated': qlib.prop(40, 500)})
            r = self.run_cmp('dirs', a, b)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('WARNING truncated', r.stdout)
            self.assertEqual(self.run_cmp('dirs', a, b, '--gate-all').returncode, 2)

    def test_dirs_unsupported_pair_not_refused(self):
        # a pair marked 'nothing to compare' (supported false in both, no metrics, n_items missing on one side) is
        # noted beside a comparable pair, never refused for unequal n
        with tempfile.TemporaryDirectory() as t:
            a, b = os.path.join(t, 'a'), os.path.join(t, 'b')
            self.summary(a, 't0', 450, 500, 500, 500)
            self.summary(b, 't0', 450, 500, 500, 500)
            for d, s in ((a, {'n_items': 0}), (b, {})):
                with open(os.path.join(d, 'json-schema.summary.json'), 'w') as f:
                    json.dump({'suite': 'json', 'cell': 'schema', 'primary': 'valid', 'supported': False,
                               'metrics': {}, **s}, f)
            r = self.run_cmp('dirs', a, b)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('nothing to compare', r.stdout)
            self.assertNotIn('REFUSED', r.stderr)


class DsmlCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='dsml-compare-')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_dc(self, *args):
        return subprocess.run([sys.executable, os.path.abspath(__file__), *args], capture_output=True, text=True)

    def arm(self, name, counts, **kw):
        return synth_selfparse(os.path.join(self.tmp, name, 'dsml-X', 'selfparse.json'), counts, **kw)

    def test_equal_counts_pass(self):
        c = {k: (5, 5, 0) for k in FOUR}
        a, b = self.arm('final', c), self.arm('mk', c)
        r = self.run_dc(a, b, '--labels', 'final,mk')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('DSML-COMPARE PASS: 12 comparison(s) over 4 row(s)', r.stdout)

    def test_one_row_worse(self):
        a = self.arm('final', {k: (1, 1, 0) for k in FOUR})
        b = self.arm('mk', {**{k: (1, 1, 0) for k in FOUR}, FOUR[2]: (10, 10, 0)})
        r = self.run_dc(a, b, '--labels', 'final,mk', '--json', os.path.join(self.tmp, 'c.json'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('DSML-COMPARE WORSE', r.stdout)
        self.assertIn('T1_todo_nested_strictfalse stream raw', r.stdout)
        self.assertIn('+1.91 [+0.54, +3.67]', r.stdout)
        with open(os.path.join(self.tmp, 'c.json')) as f:
            j = json.load(f)
        self.assertEqual(j['gate'], 'WORSE')
        self.assertEqual(len(j['worse']), 2)          # raw and parsed; differ stays 0/470 vs 0/470
        # the same pair the other way round is MK better: passes
        self.assertEqual(self.run_dc(b, a).returncode, 0)

    def test_differ_only_worse(self):
        a = self.arm('final', {k: (3, 3, 0) for k in FOUR})
        b = self.arm('mk', {**{k: (3, 3, 0) for k in FOUR}, FOUR[1]: (3, 3, 6)})
        r = self.run_dc(a, b)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('T1_todo_nested nonstream differ', r.stdout)
        self.assertEqual(self.run_dc(a, b, '--metrics', 'raw,parsed').returncode, 0)

    def test_unequal_sends_refused(self):
        a = self.arm('final', {k: (5, 5, 0) for k in FOUR})
        b = self.arm('mk', {k: (5, 5, 0) for k in FOUR}, sends=369)
        r = self.run_dc(a, b)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn('DSML-COMPARE REFUSED: 4 row(s)', r.stdout)
        r = self.run_dc(a, b, '--allow-unequal')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('unequal n allowed', r.stdout)

    def test_unequal_calls_compared(self):
        # S4's strictfalse stream: 470 sends carrying 472 calls -- same sends, compared, noted
        a = self.arm('final', {k: (5, 5, 0) for k in FOUR})
        b = self.arm('mk', {**{k: (5, 5, 0) for k in FOUR}, FOUR[2]: ((7, 472), (7, 472), 0)})
        r = self.run_dc(a, b)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('same sends, n differs', r.stdout)

    def test_zero_call_collapse_worse(self):
        a = self.arm('final', {k: (5, 5, 0) for k in FOUR})
        b = self.arm('mk', {**{k: (5, 5, 0) for k in FOUR}, FOUR[1]: ((0, 0), (0, 0), (0, 0))})
        r = self.run_dc(a, b, '--labels', 'final,mk', '--json', os.path.join(self.tmp, 'c.json'),
                        '--md', os.path.join(self.tmp, 'c.md'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('T1_todo_nested nonstream raw: zero-call collapse, no calls in mk', r.stdout)
        with open(os.path.join(self.tmp, 'c.json')) as f:
            j = json.load(f)
        self.assertEqual((j['gate'], len(j['worse'])), ('WORSE', 3))
        with open(os.path.join(self.tmp, 'c.md')) as f:
            self.assertEqual(f.read(), r.stdout)
        # the reference collapsed: no two-sample read either
        self.assertEqual(self.run_dc(b, a).returncode, 2)

    def test_no_comparison_is_not_pass(self):
        # 0/0 everywhere in both arms: nothing compared, which is not a PASS
        z = {k: ((0, 0), (0, 0), (0, 0)) for k in FOUR}
        r = self.run_dc(self.arm('final', z), self.arm('mk', z))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('no comparison made', r.stdout)
        self.assertNotIn('PASS', r.stdout)

    def test_refused_still_lists_worse(self):
        a = self.arm('final', {k: (1, 1, 0) for k in FOUR})
        b = self.arm('mk', {**{k: (1, 1, 0) for k in FOUR[:3]}, FOUR[2]: (10, 10, 0)})
        r = self.run_dc(a, b)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn('DSML-COMPARE REFUSED: 1 row(s)', r.stdout)
        self.assertIn('DSML-COMPARE WORSE (in the rows compared)', r.stdout)
        self.assertIn('T1_todo_nested_strictfalse stream raw', r.stdout)

    def test_missing_row_refused(self):
        a = self.arm('final', {k: (5, 5, 0) for k in FOUR})
        b = self.arm('mk', {k: (5, 5, 0) for k in FOUR[:3]})
        r = self.run_dc(a, b)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn('only in A', r.stdout)
        self.assertEqual(self.run_dc(a, b, '--allow-missing').returncode, 0)

    def test_inputs_agree(self):
        # selfparse.json, its phase dir, the out dir above it, and the grade stdout all read the same counts;
        # a non-REPEATS row (no counts) is skipped
        c = {FOUR[0]: (8, 8, 0), FOUR[1]: (6, 6, 0), FOUR[2]: ((7, 472), (7, 472), 0), FOUR[3]: (3, 3, 0)}
        p = self.arm('mk', c, extra_rows=[{'mode': 'stream', 'case': 'T2_write_file', 'ok': True, 'sends': 1}])
        js = load_arm(p)
        self.assertEqual(load_arm(os.path.dirname(p)), js)
        self.assertEqual(load_arm(os.path.join(self.tmp, 'mk')), js)
        txt = load_arm(synth_grade_txt(os.path.join(self.tmp, 'mk.grade.txt'), js))
        strip = lambda arm: {k: {f: v for f, v in r.items() if f != 'src'} for k, r in arm.items()}  # noqa: E731
        self.assertEqual(strip(txt), strip(js))
        self.assertEqual(js[FOUR[2]]['raw'], (7, 472))
        self.assertEqual(len(js), 4)

    def test_duplicate_phase_refused(self):
        c = {k: (5, 5, 0) for k in FOUR}
        self.arm('mk', c)
        synth_selfparse(os.path.join(self.tmp, 'mk', 'dsml-Y', 'selfparse.json'), c, phase='dsml-Y')
        with self.assertRaises(ValueError):
            load_arm(os.path.join(self.tmp, 'mk'))
        r = self.run_dc(os.path.join(self.tmp, 'mk'), os.path.join(self.tmp, 'mk', 'dsml-Y'))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('graded twice', r.stderr)

    @unittest.skipUnless(os.path.isdir(S4_DSML), 'raw/s4-mk-b1/dsml not in this tree')
    def test_s4_grade_files(self):
        # S4's MKL1 grade, as recorded: the stdout and the selfparse.json agree, and an arm against itself passes
        txt = load_arm(os.path.join(S4_DSML, 'dsml-MKL1.grade.txt'))
        js = load_arm(os.path.join(S4_DSML, 'dsml-MKL1', 'selfparse.json'))
        self.assertEqual(load_arm(S4_DSML), js)
        want = {FOUR[0]: (8, 470), FOUR[1]: (6, 470), FOUR[2]: (7, 472), FOUR[3]: (3, 470)}
        for key, kn in want.items():
            self.assertEqual(txt[key]['raw'], kn)
            self.assertEqual(js[key]['raw'], kn)
            self.assertEqual(js[key]['sends'], 470)
        r = self.run_dc(os.path.join(S4_DSML, 'dsml-MKL1.grade.txt'), S4_DSML, '--labels', 'txt,json')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == '__main__':
    main()
