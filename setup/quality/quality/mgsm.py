#!/usr/bin/env python3
"""mgsm.py -- non-English math: MGSM (the 250 GSM8K problems human-translated), exact match, Wilson 95% CI.

Default 6 languages x 20 problems = 120 items (zh, es, fr, de, ja, ru; --per-lang 250 runs every problem,
1,500 items). The question goes in its own language with a native instruction asking for step-by-step work
and a final '#### <number>' line; scoring is the gsm8k rule (the '#### n' number, else the last number),
after NFKC (full-width digits) and with locale thousands separators read ('1.234', '1 234', '1,234').
Metrics: `exact` (primary), `exact@<lang>`, `format`, and `script@<lang>` for zh/ja/ru -- the share of
replies whose letters are mostly in that language's script, a cheap reply-language heuristic (not a language
classifier). Data: google-research/url-nlp mgsm/mgsm_<lang>.tsv (public GitHub), downloaded once to data/.
Sampling defaults to T 0; --cells t0,prod runs prod sampling too. Effort high, max_tokens 8192.

    python3 mgsm.py --out OUT [--langs zh,es,fr,de,ja,ru] [--per-lang 20] [--cells t0] [-c 16]

Writes OUT/mgsm-<cell>.jsonl and OUT/mgsm-<cell>.summary.json. Expected wall on the final: ~1-2 min per cell.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

URL = 'https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_{lang}.tsv'
INSTR = {
    'zh': '请逐步解答，最后单独一行以“#### <答案>”的格式给出最终的数字答案。',
    'ja': '段階的に解き、最後に最終的な数値の答えを「#### <答え>」の形式で独立した行に書いてください。',
    'es': 'Resuelve el problema paso a paso y escribe la respuesta numérica final en una línea aparte como "#### <respuesta>".',
    'fr': 'Résous le problème étape par étape, puis donne la réponse numérique finale sur une ligne à part sous la forme « #### <réponse> ».',
    'de': 'Löse die Aufgabe Schritt für Schritt und gib die endgültige numerische Antwort in einer eigenen Zeile als „#### <Antwort>“ an.',
    'ru': 'Решите задачу пошагово и запишите окончательный числовой ответ отдельной строкой в виде «#### <ответ>».',
    'bn': 'ধাপে ধাপে সমাধান করুন এবং চূড়ান্ত সংখ্যাসূচক উত্তরটি আলাদা লাইনে "#### <উত্তর>" আকারে লিখুন।',
    'sw': 'Tatua hatua kwa hatua, kisha andika jibu la mwisho la nambari kwenye mstari wake kama "#### <jibu>".',
    'th': 'แก้โจทย์ทีละขั้นตอน แล้วเขียนคำตอบที่เป็นตัวเลขสุดท้ายในบรรทัดของตัวเองในรูปแบบ "#### <คำตอบ>"',
    'te': 'దశలవారీగా పరిష్కరించి, చివరి సంఖ్యా సమాధానాన్ని ప్రత్యేక పంక్తిలో "#### <సమాధానం>" రూపంలో రాయండి.',
    'en': 'Solve step by step, then give the final numeric answer on its own line as "#### <answer>".',
}
SCRIPTS = {'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
           'ja': [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
           'ru': [(0x0400, 0x04FF)]}
CELLS = {'t0': (0.0, 1.0), 'prod': (qlib.PROD_TEMPERATURE, qlib.PROD_TOP_P)}
MEAN_TOKENS_EST = 600


def load_items(langs, per_lang, seed=7):
    items = []
    for lang in langs:
        raw = qlib.load_or_download(os.path.join(qlib.DATA_DIR, f'mgsm_{lang}.tsv'), URL.format(lang=lang))
        rows = []
        for i, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            q, ans = line.rsplit('\t', 1)
            rows.append({'id': f'mgsm-{lang}-{i:03d}', 'lang': lang, 'question': q, 'gold': qlib.parse_number(ans)})
        order = list(range(len(rows)))
        random.Random(f'{seed}-{lang}').shuffle(order)
        keep = sorted(order[:per_lang])
        items += [rows[i] for i in keep]
    return items


def script_share(text, lang):
    rngs = SCRIPTS[lang]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if any(a <= ord(ch) <= b for a, b in rngs)) / len(letters)


def make_scorer(conn, cell, a):
    t, p = CELLS[cell]

    def score(it):
        r = qlib.chat(conn, [{'role': 'user', 'content': it['question'] + '\n\n' + INSTR[it['lang']]}],
                      max_tokens=a.max_tokens, temperature=t, top_p=p, effort=a.effort)
        rec = {'id': it['id'], 'lang': it['lang'], 'gold': it['gold']}
        if r.error:
            rec.update(ok=None, error=r.error, wall_s=round(r.wall, 3))
            return rec
        pred, strict = qlib.extract_answer(r.content)
        ok = qlib.num_eq(pred, it['gold'])
        m = {'exact': ok, 'format': strict is not None}
        if it['lang'] in SCRIPTS:
            body = '\n'.join(l for l in r.content.splitlines() if '####' not in l)
            m['script'] = script_share(body, it['lang']) >= 0.5
        rec.update(ok=ok, pred=pred, metrics=m, content_tail=r.content[-300:], **r.stats())
        return rec
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap)
    ap.add_argument('--langs', default='zh,es,fr,de,ja,ru')
    ap.add_argument('--per-lang', type=int, default=20)
    ap.add_argument('--cells', default='t0')
    ap.add_argument('--max-tokens', type=int, default=8192)
    a = ap.parse_args()
    langs = [x.strip() for x in a.langs.split(',') if x.strip()]
    for lang in langs:
        if lang not in INSTR:
            ap.error(f'no instruction for {lang!r} (have {sorted(INSTR)})')
    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    items = load_items(langs, a.per_lang)
    est = qlib.est_decode_s(len(items), MEAN_TOKENS_EST, a.concurrency)
    for cell in cells:
        if cell not in CELLS:
            ap.error(f'unknown cell {cell!r}')
        recs, wall = qlib.run_items(items, make_scorer(conn, cell, a), concurrency=a.concurrency,
                                    out_jsonl=os.path.join(out, f'mgsm-{cell}.jsonl'), label=f'mgsm-{cell}', est_s=est)
        metrics = {'exact': qlib.metric_from_records(recs, 'exact'), 'format': qlib.metric_from_records(recs, 'format')}
        for lang in langs:
            metrics[f'exact@{lang}'] = qlib.metric_from_records(recs, 'exact', where=lambda r, L=lang: r['lang'] == L)
            if lang in SCRIPTS:
                metrics[f'script@{lang}'] = qlib.metric_from_records(recs, 'script',
                                                                     where=lambda r, L=lang: r['lang'] == L)
        t, p = CELLS[cell]
        cond = {'temperature': t, 'top_p': p, 'reasoning_effort': a.effort, 'max_tokens': a.max_tokens,
                'langs': langs, 'per_lang': a.per_lang, 'concurrency': a.concurrency}
        means = {'completion_tokens': qlib.mean_ci([r.get('completion_tokens') for r in recs if r.get('ok') is not None])}
        qlib.write_summary(out, 'mgsm', cell, recs, primary='exact', metrics=metrics, means=means, condition=cond,
                           wall_s=wall, tag=a.tag, conn=conn)


if __name__ == '__main__':
    main()
