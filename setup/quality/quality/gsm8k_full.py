#!/usr/bin/env python3
"""gsm8k_full.py -- GSM8K, the whole test set (1,319), exact match, Wilson 95% CI, in up to two cells:

  t0    temperature 0 (greedy)                       cell name `t0`
  prod  prod sampling T 1.0 / top_p 1.0 (no seed)    cell name `prod`

Both at reasoning_effort high (the serving effort) and max_tokens 8192 (the deployment gate's 4096 truncates
a few long traces; truncation is still counted as wrong and reported as `truncated`). The prompt and the
scoring are gsm8k.py's: the question plus 'Solve step by step, then give the final numeric answer
on its own line as "#### <answer>"', scored on the number of the last '#### n' line, else the last number in
the content. `exact` is the primary metric; `format` (a '#### n' line exists) and `truncated`
(finish_reason == length) ride along. A failed request is an error (ok=None), not a wrong answer.

    python3 gsm8k_full.py --out OUT [--cells t0,prod] [--n 1319] [-c 16]

Writes OUT/gsm8k-<cell>.jsonl (one line per problem: id = the problem's index in the upstream test.jsonl)
and OUT/gsm8k-<cell>.summary.json. --n N < 1319 takes the first N of a seed-7 shuffle (gsm8k.py's order), so
`--n 500` is the same 500 problems everywhere (under_load.py uses it). Expected wall at 16 streams on the
final: ~3-5 min per cell (100 problems took 14 s at C16, mean 360 completion tokens).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

DATASET_URL = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
SUFFIX = '\n\nSolve step by step, then give the final numeric answer on its own line as "#### <answer>".'
CELLS = {
    't0': {'temperature': 0.0, 'top_p': 1.0},
    'prod': {'temperature': qlib.PROD_TEMPERATURE, 'top_p': qlib.PROD_TOP_P},
}
MEAN_TOKENS_EST = 450


def load_items(n=None, dataset=None):
    path = dataset or os.path.join(qlib.DATA_DIR, 'gsm8k_test.jsonl')
    raw = qlib.load_or_download(path, DATASET_URL)
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    items = []
    for i, r in enumerate(rows):
        gold = qlib.parse_number(r['answer'].split('####')[-1])
        items.append({'id': f'gsm8k-{i:04d}', 'question': r['question'], 'gold': gold})
    if n is not None and n < len(items):
        order = list(range(len(items)))
        random.Random(7).shuffle(order)
        keep = set(order[:n])
        items = [it for i, it in enumerate(items) if i in keep]
    return items


def make_scorer(conn, cell, effort, max_tokens, seed=None, label_extra=None):
    samp = CELLS[cell]

    def score(it):
        r = qlib.chat(conn, [{'role': 'user', 'content': it['question'] + SUFFIX}], max_tokens=max_tokens,
                      temperature=samp['temperature'], top_p=samp['top_p'], effort=effort, seed=seed)
        rec = {'id': it['id'], 'gold': it['gold']}
        if r.error:
            rec.update(ok=None, error=r.error, wall_s=round(r.wall, 3))
            return rec
        pred, strict = qlib.extract_answer(r.content)
        ok = qlib.num_eq(pred, it['gold'])
        rec.update(ok=ok, pred=pred, metrics={'exact': ok, 'format': strict is not None,
                                              'truncated': r.finish_reason == 'length'},
                   content_tail=r.content[-300:], **r.stats())
        if label_extra:
            rec.update(label_extra)
        return rec
    return score


def summarize(out, cell, recs, wall, a, conn, suite='gsm8k', extra=None):
    samp = CELLS[cell.split('-')[0]]
    cond = {'temperature': samp['temperature'], 'top_p': samp['top_p'], 'reasoning_effort': a.effort,
            'max_tokens': a.max_tokens, 'seed': a.seed, 'concurrency': a.concurrency, 'n': len(recs)}
    metrics = {k: qlib.metric_from_records(recs, k) for k in ('exact', 'format', 'truncated')}
    means = {'completion_tokens': qlib.mean_ci([r.get('completion_tokens') for r in recs if r.get('ok') is not None]),
             'wall_s': qlib.mean_ci([r.get('wall_s') for r in recs if r.get('ok') is not None])}
    return qlib.write_summary(out, suite, cell, recs, primary='exact', metrics=metrics, means=means, condition=cond,
                              wall_s=wall, tag=a.tag, conn=conn, extra=extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap)
    ap.add_argument('--cells', default='t0,prod', help='comma list of t0,prod (default both)')
    ap.add_argument('--n', type=int, default=None, help='problems (default all 1,319)')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=None, help='request seed (default none sent)')
    ap.add_argument('--dataset', default=None, help='local test.jsonl (default data/gsm8k_test.jsonl, downloaded once)')
    a = ap.parse_args()
    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    for c in cells:
        if c not in CELLS:
            ap.error(f'unknown cell {c!r} (have {sorted(CELLS)})')
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    items = load_items(a.n, a.dataset)
    est = qlib.est_decode_s(len(items), MEAN_TOKENS_EST, a.concurrency)
    print(f'gsm8k: {len(items)} problems x {len(cells)} cell(s) {cells}; expected wall ~{qlib.fmt_s(est * len(cells))} '
          f'total at {a.concurrency} streams on the final -> {out}', flush=True)
    for cell in cells:
        recs, wall = qlib.run_items(items, make_scorer(conn, cell, a.effort, a.max_tokens, a.seed),
                                    concurrency=a.concurrency, out_jsonl=os.path.join(out, f'gsm8k-{cell}.jsonl'),
                                    label=f'gsm8k-{cell}', est_s=est)
        summarize(out, cell, recs, wall, a, conn)


if __name__ == '__main__':
    main()
