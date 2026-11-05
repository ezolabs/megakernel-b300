#!/usr/bin/env python3
"""needles.py -- multi-key retrieval in synthetic haystacks at 100k / 250k / 500k / 900k prompt tokens.

Each item is ONE prompt: a fixed instruction, a <document> of number-free filler prose (seeded per item) with
50 key-value records planted in it ("The access code for key K-3F9A0C is 4829173."), and a question asking for
the codes of 3 of those keys. The 3 asked records sit together at the item's DEPTH (fraction of the document:
depth-0.01, depth, depth+0.01); the other 47 are spread uniformly as distractors. Every key and code is unique.
Scored exact per key (`key`, n = 3 x items) and per item (`all3`, primary), overall, per length (`all3@250k`)
and per depth (`key@d0.50`).

Sizing: with the `tokenizers` package and the served tokenizer.json (--tokenizer, default the L3a repack's)
the prompt is built to the target within ~0.5%; without it, from a chars/token ratio calibrated on the
engine's own usage.prompt_tokens after the first answer. Each prompt opens with a per-run nonce line so no
item shares a cached prefix with an earlier run (--no-nonce drops it). Lengths over the served context
(read from /v1/models max_model_len and /get_server_info context_length) minus max_tokens are skipped.

Time: the suite MEASURES the first item of each length alone, then sizes what is left to --budget-min
(default 55): if the projection (this length's measured per-item time, longer lengths scaled by
(L'/L)^1.1) overruns, it drops whole repetitions from the back, never a depth. Prior estimate on the final:
cold prefill ~31k tok/s (979,973 tokens in 31.4 s) -> ~6/11/19/32 s per item at 100k/250k/500k/900k plus
thinking; the default 4 lengths x 5 depths x 4 reps = 80 items ~ 20-30 min at concurrency 2.

    python3 needles.py --out OUT [--lengths 100000,250000,500000,900000] [--depths 0.05,0.25,0.5,0.75,0.95]
                       [--reps 4] [-c 2] [--budget-min 55] [--dry-run]

--dry-run builds one prompt per length, prints its measured token count and build time, sends nothing.
Writes OUT/needles-<cell>.jsonl and OUT/needles-<cell>.summary.json (cell = t0 unless --temperature).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

DEFAULT_TOKENIZER = '<MODEL_PATH>/tokenizer.json'
N_RECORDS = 50
N_ASK = 3
_SUBJ = ['the archivist', 'a gardener', 'the old ferryman', 'a weaver', 'the night porter', 'a cartographer',
         'the beekeeper', 'a lighthouse keeper', 'the choir', 'a potter', 'the river pilot', 'a glassblower',
         'the miller', 'a shepherd', 'the bookbinder', 'a clockmaker', 'the innkeeper', 'a fisherwoman']
_VERB = ['described', 'remembered', 'painted', 'mended', 'catalogued', 'sang about', 'walked past', 'sketched',
         'listened to', 'folded away', 'polished', 'wrote about', 'admired', 'carried', 'forgot', 'repaired']
_OBJ = ['a quiet harbour at dusk', 'the faded tapestry', 'an orchard after rain', 'the cedar staircase',
        'a lantern of green glass', 'the salt marsh', 'an unfinished letter', 'the clock tower bells',
        'a field of heather', 'the copper kettle', 'a ribbon of fog', 'the cathedral windows', 'the stone bridge',
        'a basket of plums', 'the frozen pond', 'an old sea chart']
_TAIL = ['while the tide turned', 'before the market opened', 'under a pale sky', 'as the wind rose',
         'in the hush of winter', 'beside the reed beds', 'long after supper', 'without a word',
         'with careful hands', 'as the lamps were lit', 'in the morning haze', 'among the willows',
         'during the long rain', 'at the edge of town', 'when the bells stopped', 'near the old mill']
HEAD = ('You will read a long document. Somewhere in it are records of the form "The access code for key '
        'K-XXXXXX is NNNNNNN." Read carefully; you will be asked for some of the codes at the end.\n\n<document>\n')
QUESTION = ('</document>\n\nWhat are the access codes for keys {keys}? Answer with exactly one line per key, '
            'in the form "K-XXXXXX: NNNNNNN", and nothing else.')
PAIR_RE = re.compile(r'K-([0-9A-F]{6})\b[^\S\n]*[^\w\n]{0,6}[^\S\n]*(?:IS[^\S\n]+|CODE[^\S\n]*[:=]?[^\S\n]*)?'
                     r'[*`_"\']{0,3}(\d{7})\b')


class Sizer:
    """Token counts: the served tokenizer when available, else chars / ratio (calibrated from usage)."""

    def __init__(self, path):
        self.tok = None
        self.ratio = 4.75      # chars per token for this prose (measured on the L3a tokenizer) until calibrated
        self.calibrated = False
        self.cache = {}
        try:
            from tokenizers import Tokenizer   # noqa
            if path and os.path.exists(path):
                self.tok = Tokenizer.from_file(path)
        except Exception:
            self.tok = None
        self.kind = 'tokenizer' if self.tok else 'chars-ratio'

    def count(self, text):
        if self.tok:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        return int(len(text) / self.ratio)

    def piece(self, s):
        """Tokens of one sentence as it appears mid-document (with its leading space)."""
        if not self.tok:
            return len(s) / self.ratio
        v = self.cache.get(s)
        if v is None:
            v = len(self.tok.encode(s, add_special_tokens=False).ids)
            self.cache[s] = v
        return v

    def calibrate(self, chars, prompt_tokens):
        if self.tok or not prompt_tokens:
            return
        r = chars / float(prompt_tokens)
        self.ratio = r if not self.calibrated else 0.5 * (self.ratio + r)
        self.calibrated = True


def make_item(length, depth, rep, run_seed):
    seed = (length * 1000003 + int(round(depth * 1000)) * 7919 + rep * 104729 + run_seed) & 0x7FFFFFFF
    rnd = random.Random(seed)
    keys, vals = set(), set()
    recs = []
    while len(recs) < N_RECORDS:
        k = '%06X' % rnd.randrange(16 ** 6)
        v = str(rnd.randrange(1000000, 10000000))
        if k in keys or v in vals:
            continue
        keys.add(k)
        vals.add(v)
        recs.append((k, v))
    ask = recs[:N_ASK]
    d = [max(0.0, min(1.0, depth - 0.01)), depth, max(0.0, min(1.0, depth + 0.01))]
    pos = d + [rnd.random() for _ in range(N_RECORDS - N_ASK)]
    order = list(range(N_ASK))
    rnd.shuffle(order)
    return {'id': f'needle-{length // 1000:04d}k-d{int(round(depth * 100)):02d}-r{rep}', 'length': length,
            'depth': depth, 'rep': rep, 'seed': seed, 'records': recs, 'positions': pos,
            'ask': [ask[i] for i in order]}


def build_prompt(it, sizer, nonce):
    """The full user message for item `it`, sized to it['length'] tokens. Returns (text, est_tokens)."""
    rnd = random.Random(it['seed'] ^ 0x5DEECE66)
    head = (f'Session {nonce}.\n' if nonce else '') + HEAD
    q = QUESTION.format(keys=', '.join('K-' + k for k, _ in it['ask']))
    rec_txt = [f' The access code for key K-{k} is {v}.' for k, v in it['records']]
    overhead = sizer.count(head) + sizer.count(q) + sum(sizer.count(r) for r in rec_txt) + 8
    target = max(1000, it['length'] - overhead)
    sents, cum, tot = [], [], 0.0
    while tot < target:
        s = f' {rnd.choice(_SUBJ)} {rnd.choice(_VERB)} {rnd.choice(_OBJ)} {rnd.choice(_TAIL)}.'
        s = ' ' + s[1].upper() + s[2:]
        sents.append(s)
        tot += sizer.piece(s)
        cum.append(tot)
    # Insert each record at the sentence boundary nearest its token fraction.
    inserts = {}
    for (rt, p) in zip(rec_txt, it['positions']):
        want = p * tot
        lo, hi = 0, len(cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < want:
                lo = mid + 1
            else:
                hi = mid
        inserts.setdefault(lo, []).append(rt)
    parts = [head]
    for i, s in enumerate(sents):
        for rt in inserts.get(i, ()):
            parts.append(rt)
        parts.append(s)
        if i % 8 == 7:
            parts.append('\n')
    for rt in inserts.get(len(sents), ()):
        parts.append(rt)
    parts.append('\n')
    parts.append(q)
    text = ''.join(parts)
    return text, int(tot + overhead)


def score_content(content, ask):
    txt = qlib.norm_text(content).upper()
    got = {}
    for k, v in PAIR_RE.findall(txt):
        got.setdefault(k, v)
    per = []
    for k, v in ask:
        per.append(got.get(k) == v or (k not in got and v in txt))
    return per, got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap, concurrency=2)
    ap.add_argument('--lengths', default='100000,250000,500000,900000')
    ap.add_argument('--depths', default='0.05,0.25,0.5,0.75,0.95')
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--budget-min', type=float, default=55.0, help='wall budget the plan is sized to (default 55)')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--seed', type=int, default=20261029, help='haystack seed (default fixed: same items every run)')
    ap.add_argument('--no-nonce', action='store_true', help='no per-run nonce line (items may hit a warm cache)')
    ap.add_argument('--dry-run', action='store_true', help='build one prompt per length, print its size, send nothing')
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(',') if x.strip()]
    depths = [float(x) for x in a.depths.split(',') if x.strip()]
    sizer = Sizer(a.tokenizer)
    nonce = None if a.no_nonce else uuid.uuid4().hex[:16]

    if a.dry_run:
        print(f'needles dry run: sizer={sizer.kind}')
        for L in lengths:
            it = make_item(L, depths[len(depths) // 2], 0, a.seed)
            t0 = time.time()
            text, est = build_prompt(it, sizer, nonce)
            tb = time.time() - t0
            t1 = time.time()
            real = sizer.count(text) if sizer.tok else None
            print(f'  {L:>8}: est {est} tokens, measured {real if real is not None else "n/a"}, {len(text)} chars, '
                  f'build {tb:.2f}s, count {time.time() - t1:.2f}s; ask {it["ask"]}')
            # sanity: the asked records are in the text
            assert all(f'K-{k} is {v}.' in text for k, v in it['ask'])
        return

    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    cell = 't0' if a.temperature == 0 else f'T{a.temperature:g}'
    info = qlib.server_info(conn)
    ctx = info.get('context_length')
    usable = [L for L in lengths if not ctx or L + a.max_tokens + 512 <= ctx]
    skipped = [L for L in lengths if L not in usable]
    print(f'needles: served context {ctx}; lengths {usable}' + (f' (skipped over context: {skipped})' if skipped else '')
          + f'; sizer={sizer.kind}; nonce={nonce}', flush=True)
    prior = {L: L / qlib.EST_PREFILL_TPS + 4.0 for L in usable}
    n_per = len(depths) * a.reps
    est = sum(prior[L] * n_per for L in usable) / max(1.0, min(a.concurrency, 2) * 0.75)
    print(f'needles: plan {len(usable)} lengths x {len(depths)} depths x {a.reps} reps = {n_per * len(usable)} items; '
          f'expected wall ~{qlib.fmt_s(est)} on the final (prior), budget {a.budget_min:.0f} min', flush=True)

    budget = a.budget_min * 60.0
    t_start = time.time()
    all_recs = []
    plan_log = []
    ratio_eff = 1.0      # measured effective per-item time in a batch / the lone first item's time
    lock_cal = threading.Lock()

    def scorer(it):
        text, est_tok = build_prompt(it, sizer, nonce)
        r = qlib.chat(conn, [{'role': 'user', 'content': text}], max_tokens=a.max_tokens, temperature=a.temperature,
                      top_p=a.top_p, effort=a.effort)
        rec = {'id': it['id'], 'length': it['length'], 'depth': it['depth'], 'rep': it['rep'], 'est_tokens': est_tok,
               'chars': len(text), 'ask': it['ask']}
        if r.error:
            rec.update(ok=None, error=r.error, wall_s=round(r.wall, 3))
            return rec
        with lock_cal:
            sizer.calibrate(len(text), r.usage.get('prompt_tokens'))
        per, got = score_content(r.content, it['ask'])
        rec.update(ok=all(per), per_key=per, parsed={k: v for k, v in list(got.items())[:6]},
                   metrics={'all3': all(per)}, content_head=r.content[:300], **r.stats())
        return rec

    for li, L in enumerate(usable):
        items = [make_item(L, d, rep, a.seed) for rep in range(a.reps) for d in depths]   # rep-major
        first, rest = items[0], items[1:]
        recs1, w1 = qlib.run_items([first], scorer, concurrency=1,
                                   out_jsonl=os.path.join(out, f'.needles-{L}-first.jsonl'), label=f'needles-{L}-first')
        t_item = max(1.0, recs1[0].get('wall_s') or w1)
        elapsed = time.time() - t_start
        left = budget - elapsed
        need_rest_here = len(rest) * t_item * ratio_eff
        need_longer = sum(len(depths) * a.reps * t_item * ratio_eff * (L2 / L) ** 1.1 for L2 in usable[li + 1:])
        keep = len(rest)
        if need_rest_here + need_longer > left:
            scale = max(0.0, left) / (need_rest_here + need_longer)
            keep = max(len(depths) - 1, int(math.floor(len(rest) * scale)))
            keep = min(keep, len(rest))
        if keep < len(rest):
            print(f'needles-{L}: first item {t_item:.1f}s; budget left {qlib.fmt_s(left)} -> running {keep + 1} of '
                  f'{len(items)} items here (whole reps dropped from the back)', flush=True)
        run = rest[:keep]
        est_here = len(run) * t_item * ratio_eff
        recs2, w2 = qlib.run_items(run, scorer, concurrency=a.concurrency,
                                   out_jsonl=os.path.join(out, f'.needles-{L}-rest.jsonl'), label=f'needles-{L}',
                                   est_s=est_here) if run else ([], 0.0)
        if run and w2 > 0:
            ratio_eff = max(0.2, (w2 / len(run)) / t_item)
        plan_log.append({'length': L, 'planned': len(items), 'ran': 1 + len(run), 'first_item_s': round(t_item, 2),
                         'batch_wall_s': round(w2, 1), 'effective_per_item_s': round(w2 / len(run), 2) if run else None})
        for f in (f'.needles-{L}-first.jsonl', f'.needles-{L}-rest.jsonl'):
            p = os.path.join(out, f)
            if os.path.exists(p):
                os.remove(p)
        all_recs += recs1 + recs2
    wall = time.time() - t_start
    all_recs.sort(key=lambda r: qlib.natural_key(r['id']))
    with open(os.path.join(out, f'needles-{cell}.jsonl'), 'w') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    ans = [r for r in all_recs if r.get('ok') is not None]

    def keyprop(rs):
        ks = [x for r in rs for x in (r.get('per_key') or [])]
        return qlib.prop(sum(ks), len(ks))
    metrics = {'all3': qlib.metric_from_records(all_recs, 'all3'), 'key': keyprop(ans)}
    for L in usable:
        tag = f'{L // 1000}k'
        metrics[f'all3@{tag}'] = qlib.metric_from_records(all_recs, 'all3', where=lambda r, L=L: r['length'] == L)
        metrics[f'key@{tag}'] = keyprop([r for r in ans if r['length'] == L])
    for d in depths:
        metrics[f'key@d{d:.2f}'] = keyprop([r for r in ans if abs(r['depth'] - d) < 1e-9])
    means = {'key_fraction_per_item': qlib.mean_ci([sum(r['per_key']) / len(r['per_key']) for r in ans]),
             'prompt_tokens': qlib.mean_ci([r.get('prompt_tokens') for r in ans]),
             'completion_tokens': qlib.mean_ci([r.get('completion_tokens') for r in ans])}
    for L in usable:
        means[f'wall_s@{L // 1000}k'] = qlib.mean_ci([r.get('wall_s') for r in ans if r['length'] == L])
    cond = {'temperature': a.temperature, 'top_p': a.top_p, 'reasoning_effort': a.effort, 'max_tokens': a.max_tokens,
            'lengths': usable, 'skipped_lengths': skipped, 'depths': depths, 'reps': a.reps, 'records': N_RECORDS,
            'asked': N_ASK, 'seed': a.seed, 'nonce': nonce, 'sizer': sizer.kind,
            'chars_per_token': round(sizer.ratio, 3) if not sizer.tok else None, 'concurrency': a.concurrency,
            'budget_min': a.budget_min}
    qlib.write_summary(out, 'needles', cell, all_recs, primary='all3', metrics=metrics, means=means, condition=cond,
                       wall_s=wall, tag=a.tag, conn=conn, extra={'plan': plan_log, 'server_info': info})


if __name__ == '__main__':
    main()
