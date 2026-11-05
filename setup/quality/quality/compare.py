#!/usr/bin/env python3
"""compare.py -- two-sample comparison of quality results: A (reference, e.g. the final) vs B (e.g. the MK).

  compare.py dirs  A_DIR B_DIR            every <suite>-<cell>.summary.json present in both dirs, matched by
                                          (suite, cell)
  compare.py files A.summary.json B.summary.json
                                          one explicit pair (suite/cell may differ: idle vs under load)
  compare.py counts K1 N1 K2 N2           two raw proportions (e.g. the DSML gate's pass counts)

Options: --labels A,B (default A,B); --margin M (non-inferiority margin as a proportion, default 0.03 = 3 pp;
0 turns it off); --lower-better (counts mode: K counts failures); --json PATH and --md PATH also write the
result as JSON / Markdown; --allow-unequal (compare runs of different size); --gate-all (every metric gates).

For every proportion metric present in both (k/n from the summaries, `answered` included):
  diff      p_B - p_A with Newcombe's hybrid-score 95% CI (1998, method 10)
  p-values  Fisher exact (two-sided) and the pooled two-proportion z test
  verdict   'B better' / 'B WORSE' when the CI excludes 0; otherwise 'non-inferior' when the CI's bad end
            stays within the margin, 'inconclusive' when it reaches past it ('no detectable difference' with
            --margin 0). Metrics where lower is better (`truncated`) are read the other way round.
When both sides have their per-item jsonl next to the summary, the primary metric is also compared PAIRED on
the shared item ids: both/A-only/B-only counts, exact McNemar p, and a paired bootstrap CI of the difference.
Paired is the sharper read when the two runs scored the same items (same seeds, same subset).
The suite verdict is its primary metric's, plus a warning line for any other metric whose CI excludes 0 in the
bad direction (a lost-requests `answered` drop included).

THE GATE. Both arms must be the same size: counts mode needs N1 == N2, files/dirs need the same item count
(`answered`'s n: every item sent, errors included) in each pair; anything else is refused with exit 3 unless
--allow-unequal (a metric's own n may still differ by the items an arm lost to errors: noted, and `answered`
scores the loss; a pair with no metric compared, e.g. supported/served false, is noted, never refused). A metric is WORSE when its Newcombe 95% CI excludes 0 in the bad direction OR the one-sided
Fisher exact p (B worse than A) is < 0.05. A pair gates on its primary metric and on `answered`; --gate-all
makes every metric gate (the sub-metrics are many, so expect false alarms). Exit 0 when nothing gates WORSE,
2 when anything does, 3 on an unequal-n refusal, 1 on bad input (usage errors included).
"""
import argparse
import glob
import json
import os
import math
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

LOWER_BETTER = {'truncated'}
GATE_ALPHA = 0.05
EXIT_WORSE, EXIT_UNEQUAL = 2, 3


class ArgParser(argparse.ArgumentParser):
    """Usage errors exit 1, so exit 2 only ever means a WORSE verdict."""

    def error(self, message):
        self.print_usage(sys.stderr)
        sys.exit(f'{self.prog}: error: {message}')


def fisher_worse(k1, n1, k2, n2, lower_better=False):
    """One-sided Fisher exact p that B is worse than A: the hypergeometric tail of B's count at fixed margins,
    P(X <= k2) when K counts successes, P(X >= k2) when K counts failures (lower_better). Exact integers."""
    c, n = k1 + k2, n1 + n2
    lo, hi = max(0, c - n1), min(n2, c)
    xs = range(k2, hi + 1) if lower_better else range(lo, k2 + 1)
    num = sum(math.comb(n2, x) * math.comb(n1, c - x) for x in xs)
    return float(Fraction(num, math.comb(n, c)))


def gate_why(m, alpha=GATE_ALPHA):
    """Why metric comparison m gates WORSE ([] when it does not)."""
    why = []
    if m['class'] == 'worse':
        why.append(f"Newcombe 95% CI [{100 * m['lo']:+.2f}, {100 * m['hi']:+.2f}] pp excludes 0")
    if m['fisher_p_worse'] < alpha:
        why.append(f"one-sided Fisher p {m['fisher_p_worse']:.4g} < {alpha:g}")
    return why


def verdict(d, lo, hi, margin, lower_better=False, la='A', lb='B'):
    # Work in "goodness" units: positive = B better.
    g, glo, ghi = (-d, -hi, -lo) if lower_better else (d, lo, hi)
    pp = 100 * g
    if glo > 0:
        return f'{lb} better than {la} ({pp:+.2f} pp, CI excludes 0)', 'better'
    if ghi < 0:
        return f'{lb} WORSE than {la} ({pp:+.2f} pp, CI excludes 0)', 'worse'
    if margin and margin > 0:
        if glo >= -margin:
            return (f'{lb} non-inferior to {la} within {100 * margin:.1f} pp (CI bad end {100 * glo:+.2f} pp)',
                    'noninferior')
        return (f'inconclusive: CI bad end {100 * glo:+.2f} pp reaches past the {100 * margin:.1f} pp margin '
                f'(more items needed to call it)', 'inconclusive')
    return f'no detectable difference (CI [{100 * glo:+.2f}, {100 * ghi:+.2f}] pp includes 0)', 'nodiff'


def compare_counts(k1, n1, k2, n2, margin, lower_better=False, la='A', lb='B'):
    d, lo, hi = qlib.newcombe_diff(k1, n1, k2, n2)
    p_f = qlib.fisher_exact(k1, n1 - k1, k2, n2 - k2)
    z, p_z = qlib.two_prop_z(k1, n1, k2, n2)
    text, cls = verdict(d, lo, hi, margin, lower_better, la, lb)
    m = {'a': qlib.prop(k1, n1), 'b': qlib.prop(k2, n2), 'diff': d, 'lo': lo, 'hi': hi, 'fisher_p': p_f, 'z': z,
         'z_p': p_z, 'verdict': text, 'class': cls, 'lower_better': lower_better,
         'fisher_p_worse': fisher_worse(k1, n1, k2, n2, lower_better)}
    m['gate_why'] = gate_why(m)
    m['gate'] = 'WORSE' if m['gate_why'] else 'pass'
    return m


def design_n(s):
    """A summary's run size: every item sent, errors included."""
    ans = (s.get('metrics') or {}).get('answered') or {}
    return ans.get('n') if ans.get('n') is not None else s.get('n_items')


def apply_gate(res, gate_all=False):
    """res['gate'] = 'WORSE' | 'pass' and res['gate_why'] = ['<metric>: <why>', ...] from the gating metrics: the
    primary and `answered` (every metric with gate_all)."""
    why = []
    for name, m in res['metrics'].items():
        if gate_all or name in (res.get('primary'), 'answered'):
            why += [f'{name}: {w}' for w in m['gate_why']]
    res['gate_why'] = why
    res['gate'] = 'WORSE' if why else 'pass'
    return res


def load_jsonl(path):
    recs = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[str(r.get('id'))] = r
    return recs


def paired(ra, rb, metric, reps=5000, seed=1):
    ids = sorted(set(ra) & set(rb))
    pairs = []
    for i in ids:
        a, b = ra[i], rb[i]
        if a.get('ok') is None or b.get('ok') is None:
            continue
        va, vb = (a.get('metrics') or {}).get(metric), (b.get('metrics') or {}).get(metric)
        if va is None or vb is None:
            continue
        pairs.append((bool(va), bool(vb)))
    if not pairs:
        return None
    both = sum(1 for x, y in pairs if x and y)
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    n = len(pairs)
    d = (b_only - a_only) / n
    rnd = random.Random(seed)
    ds = []
    for _ in range(reps):
        s = 0
        for _ in range(n):
            x, y = pairs[rnd.randrange(n)]
            s += (1 if y else 0) - (1 if x else 0)
        ds.append(s / n)
    ds.sort()
    return {'n_pairs': n, 'shared_ids': len(ids), 'both': both, 'a_only': a_only, 'b_only': b_only,
            'neither': n - both - a_only - b_only, 'diff': d, 'lo': ds[int(0.025 * reps)],
            'hi': ds[min(reps - 1, int(0.975 * reps))], 'mcnemar_p': qlib.mcnemar_exact(a_only, b_only)}


def compare_summaries(pa, pb, margin, la, lb):
    sa, sb = json.load(open(pa)), json.load(open(pb))
    res = {'a_file': pa, 'b_file': pb, 'suite': sa.get('suite'), 'cell_a': sa.get('cell'), 'cell_b': sb.get('cell'),
           'primary': sa.get('primary'), 'metrics': {}, 'notes': [], 'n_a': design_n(sa), 'n_b': design_n(sb)}
    for key in ('supported', 'served'):
        if sa.get(key) is False or sb.get(key) is False:
            res['notes'].append(f'{key}: {la}={sa.get(key)} {lb}={sb.get(key)}; nothing to compare'
                                if sa.get(key) is False and sb.get(key) is False else
                                f'{key}: {la}={sa.get(key)} {lb}={sb.get(key)} -- the two arms differ in capability')
    if sa.get('condition') != sb.get('condition'):
        diffs = sorted(k for k in set(sa.get('condition') or {}) | set(sb.get('condition') or {})
                       if (sa.get('condition') or {}).get(k) != (sb.get('condition') or {}).get(k))
        if diffs:
            res['notes'].append('conditions differ in: ' + ', '.join(diffs))
    for name, ma in (sa.get('metrics') or {}).items():
        mb = (sb.get('metrics') or {}).get(name)
        if not mb or not ma.get('n') or not mb.get('n'):
            continue
        res['metrics'][name] = compare_counts(ma['k'], ma['n'], mb['k'], mb['n'], margin,
                                              name.split('@')[0] in LOWER_BETTER, la, lb)
    if res['n_a'] == res['n_b']:
        uneq = [f"{n} {m['a']['n']}/{m['b']['n']}" for n, m in res['metrics'].items() if m['a']['n'] != m['b']['n']]
        if uneq:
            res['notes'].append(f'same {res["n_a"]} items, but metric n differs ({la}/{lb}: items lost to errors '
                                f'or not scored): ' + ', '.join(uneq))
    prim = sa.get('primary')
    ja = load_jsonl(pa.replace('.summary.json', '.jsonl'))
    jb = load_jsonl(pb.replace('.summary.json', '.jsonl'))
    if prim and ja and jb:
        pr = paired(ja, jb, prim)
        if pr:
            res['paired'] = pr
    res['means'] = {k: {'a': (sa.get('means') or {}).get(k), 'b': (sb.get('means') or {}).get(k)}
                    for k in (sa.get('means') or {}) if k in (sb.get('means') or {})}
    pm = res['metrics'].get(prim)
    if pm:
        res['verdict'] = pm['verdict']
        res['class'] = pm['class']
        warn = [f'{n}: {m["verdict"]}' for n, m in res['metrics'].items() if n != prim and m['class'] == 'worse']
        if warn:
            res['warnings'] = warn
    else:
        res['verdict'] = 'no comparable primary metric'
        res['class'] = 'none'
    return res


def fmt_counts(m):
    return f"{m['k']}/{m['n']} ({100 * m['p']:.2f}%)"


def render(res, la, lb):
    out = []
    out.append(f"### {res['suite']}: {la} `{res['cell_a']}` vs {lb} `{res['cell_b']}` (primary `{res['primary']}`)")
    out.append('')
    out.append(f'**Verdict:** {res["verdict"]}')
    out.append(f'**Gate:** {res["gate"]}' + (' -- ' + '; '.join(res['gate_why']) if res['gate_why'] else ''))
    for w in res.get('warnings', []):
        out.append(f'- WARNING {w}')
    for n in res.get('notes', []):
        out.append(f'- note: {n}')
    pr = res.get('paired')
    if pr:
        out.append(f"- paired on {pr['n_pairs']} shared items: both right {pr['both']}, only {la} {pr['a_only']}, "
                   f"only {lb} {pr['b_only']}, neither {pr['neither']}; diff {100 * pr['diff']:+.2f} pp "
                   f"[{100 * pr['lo']:+.2f}, {100 * pr['hi']:+.2f}] (paired bootstrap), McNemar exact p = "
                   f"{pr['mcnemar_p']:.4g}")
    out.append('')
    out.append(f'| metric | {la} | {lb} | {lb} - {la} (pp) [Newcombe 95%] | Fisher p | z p | verdict |')
    out.append('|---|---|---|---|---|---|---|')
    for name, m in res['metrics'].items():
        out.append(f"| {name}{'*' if name == res['primary'] else ''} | {fmt_counts(m['a'])} | {fmt_counts(m['b'])} | "
                   f"{100 * m['diff']:+.2f} [{100 * m['lo']:+.2f}, {100 * m['hi']:+.2f}] | {m['fisher_p']:.4g} | "
                   f"{m['z_p']:.4g} | {m['class']} |")
    if res.get('means'):
        out.append('')
        out.append(f'| mean | {la} | {lb} |')
        out.append('|---|---|---|')
        for k, v in res['means'].items():
            def f(x):
                return f"{x['mean']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]" if x and x.get('n') else 'n/a'
            out.append(f'| {k} | {f(v["a"])} | {f(v["b"])} |')
    out.append('')
    return '\n'.join(out)


def main():
    ap = ArgParser(description=__doc__.split('\n')[0],
                   formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('mode', choices=['dirs', 'files', 'counts'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--labels', default='A,B')
    ap.add_argument('--margin', type=float, default=0.03)
    ap.add_argument('--lower-better', action='store_true', help='counts mode: K counts failures')
    ap.add_argument('--name', default='counts', help='counts mode: what is counted (printed)')
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    ap.add_argument('--allow-unequal', action='store_true',
                    help='compare arms of different size (refused with exit 3 otherwise)')
    ap.add_argument('--gate-all', action='store_true', help='every metric gates, not just the primary and answered')
    a = ap.parse_args()
    la, lb = (a.labels.split(',') + ['B'])[:2]
    results = []
    if a.mode == 'counts':
        if len(a.args) != 4:
            ap.error('counts needs K1 N1 K2 N2')
        k1, n1, k2, n2 = (int(x) for x in a.args)
        if not (0 <= k1 <= n1 and 0 <= k2 <= n2 and n1 > 0 and n2 > 0):
            ap.error('need 0 <= K <= N and N > 0')
        m = compare_counts(k1, n1, k2, n2, a.margin, a.lower_better, la, lb)
        res = {'suite': a.name, 'cell_a': la, 'cell_b': lb, 'primary': a.name, 'metrics': {a.name: m},
               'verdict': m['verdict'], 'class': m['class'], 'notes': [], 'n_a': n1, 'n_b': n2}
        results.append(res)
    elif a.mode == 'files':
        if len(a.args) != 2:
            ap.error('files needs A.summary.json B.summary.json')
        results.append(compare_summaries(a.args[0], a.args[1], a.margin, la, lb))
    else:
        if len(a.args) != 2:
            ap.error('dirs needs A_DIR B_DIR')
        da, db = a.args
        idx_b = {}
        for p in glob.glob(os.path.join(db, '*.summary.json')):
            s = json.load(open(p))
            idx_b[(s.get('suite'), s.get('cell'))] = p
        seen = set()
        for p in sorted(glob.glob(os.path.join(da, '*.summary.json'))):
            s = json.load(open(p))
            key = (s.get('suite'), s.get('cell'))
            if key in idx_b:
                results.append(compare_summaries(p, idx_b[key], a.margin, la, lb))
                seen.add(key)
            else:
                print(f'only in {la}: {os.path.basename(p)}')
        for key, p in sorted(idx_b.items(), key=lambda kv: str(kv[0])):
            if key not in seen:
                print(f'only in {lb}: {os.path.basename(p)}')
        if not results:
            sys.exit('no (suite, cell) present in both dirs')
    # a pair with no metric compared (supported/served false: 'nothing to compare') has no n to hold equal; its
    # design_n can fall back to a missing n_items, so it is noted, never refused
    uneq = [r for r in results if r['n_a'] != r['n_b'] and r['metrics']]
    if uneq and not a.allow_unequal:
        for r in uneq:
            print(f"REFUSED {r['suite']} {r['cell_a']} vs {r['cell_b']}: {la} n={r['n_a']}, {lb} n={r['n_b']}",
                  file=sys.stderr)
        print('unequal n: compare both arms at equal n (rerun the short one), or pass --allow-unequal',
              file=sys.stderr)
        sys.exit(EXIT_UNEQUAL)
    for r in uneq:
        r['notes'].append(f"unequal n allowed (--allow-unequal): {la} n={r['n_a']}, {lb} n={r['n_b']}")
    for r in results:
        apply_gate(r, a.gate_all)
    md = [f'## Quality comparison: {la} (reference) vs {lb}', '',
          f'Non-inferiority margin: {100 * a.margin:.1f} pp' if a.margin else 'No margin (CI vs 0 only)', '']
    md += [render(r, la, lb) for r in results]
    md.append('| suite | verdict |')
    md.append('|---|---|')
    for r in results:
        md.append(f"| {r['suite']} `{r['cell_a']}`{'' if r['cell_a'] == r['cell_b'] else ' vs `' + str(r['cell_b']) + '`'}"
                  f" | {r['verdict']} |")
    worse = [r for r in results if r['gate'] == 'WORSE']
    md.append('')
    md.append(f'GATE: WORSE ({len(worse)} of {len(results)}: ' + '; '.join(
        f"{r['suite']} `{r['cell_a']}`: " + '; '.join(r['gate_why']) for r in worse) + ')' if worse else
        f'GATE: PASS ({len(results)} compared; nothing WORSE by Newcombe 95% CI or one-sided Fisher p < '
        f'{GATE_ALPHA:g})')
    text = '\n'.join(md)
    print(text)
    if a.md:
        with open(a.md, 'w') as f:
            f.write(text + '\n')
    if a.json:
        qlib.write_json(a.json, {'labels': [la, lb], 'margin': a.margin, 'gate': 'WORSE' if worse else 'pass',
                                 'results': results})
    if worse:
        sys.exit(EXIT_WORSE)


if __name__ == '__main__':
    main()
