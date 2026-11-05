#!/usr/bin/env python3
"""decode_distribution.py: pool the think-ramp per-rep rows of every config X' boot on the record.

Reads the `think-<arm>.jsonl` files the bench (`box-c/qa/speed-20261026/bench.py think-ramp`) wrote for each
boot of fork `prod` with the megakernel (config X', trees 4943d6ed90 and 9a9ede7ae0) on box-a and box-b, and prints,
per concurrency, the distribution of the per-rep decode rate `dec_med` (completion tokens / (last delta - first
delta), median over the rep's streams; at C1 one request = one rep). Prompts: synthetic reasoning prompts, 300
completion tokens, thinking on, temperature 1.0, idle box (off pool or row held out), keep-alive connection.

Output: a markdown table on stdout and `decode-distribution.csv` beside this script's parent `data/` folder.
Usage (from the repo root): python3 record/data/scripts/decode_distribution.py
"""
import csv
import json
import os
import statistics as st

QA = "box-c/qa"
# (box, tree, label, path) — every config X' think-ramp file on the record as of 2026-10-31.
RUNS = [
    ("box-a", "4943d6ed90", "MKP1 (#84 gate)", "prod-merge-20261030/raw/gate/arm/think-MKP1.jsonl"),
    ("box-a", "4943d6ed90", "box-a #178", "box-b-bringup-20261030/raw/speed/box-a/think-box-a.jsonl"),
    ("box-b", "4943d6ed90", "box-b #178", "box-b-bringup-20261030/raw/speed/box-b/think-box-b.jsonl"),
    ("box-a", "4943d6ed90", "CTL1", "sglang-takes-20261031/raw/speed/CTL1/think-CTL1.jsonl"),
    ("box-a", "4943d6ed90", "CTL2", "sglang-takes-20261031/raw/speed/CTL2/think-CTL2.jsonl"),
    ("box-a", "4943d6ed90", "K3", "sglang-takes-20261031/raw/roll-box-a/speed/K3/think-K3.jsonl"),
    ("box-a", "9a9ede7ae0", "CANDB1", "sglang-takes-20261031/raw/speed/CANDB1/think-CANDB1.jsonl"),
    ("box-a", "9a9ede7ae0", "CANDB2", "sglang-takes-20261031/raw/speed/CANDB2/think-CANDB2.jsonl"),
    ("box-a", "9a9ede7ae0", "C3", "sglang-takes-20261031/raw/roll-box-a/speed/C3/think-C3.jsonl"),
    ("box-a", "9a9ede7ae0", "C4 (serving)", "sglang-takes-20261031/raw/roll-box-a/speed/C4/think-C4.jsonl"),
    ("box-a", "9a9ede7ae0", "A1 (C1 settle)", "sglang-takes-20261031/c1-settle/speed/A1/think-A1.jsonl"),
    ("box-a", "9a9ede7ae0", "A2 (C1 settle)", "sglang-takes-20261031/c1-settle/speed/A2/think-A2.jsonl"),
    ("box-a", "4943d6ed90", "B1 (C1 settle)", "sglang-takes-20261031/c1-settle/speed/B1/think-B1.jsonl"),
    ("box-a", "4943d6ed90", "B2 (C1 settle)", "sglang-takes-20261031/c1-settle/speed/B2/think-B2.jsonl"),
]


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    by_c, ttft_c, boots_c, errs = {}, {}, {}, 0
    for box, tree, label, rel in RUNS:
        for line in open(os.path.join(QA, rel)):
            if not line.strip() or line.startswith("#"):
                continue
            r = json.loads(line)
            if "C" not in r:
                continue
            errs += r.get("errs", 0)
            if r.get("n_ok", 0) <= 0:
                continue
            c = r["C"]
            by_c.setdefault(c, []).append(r["dec_med"])
            ttft_c.setdefault(c, []).append(1000 * r["ttft_med"])
            boots_c.setdefault(c, {}).setdefault(label, []).append(r["dec_med"])
    rows = []
    for c in sorted(by_c):
        xs, ts = by_c[c], ttft_c[c]
        boot_means = [st.mean(v) for v in boots_c[c].values()]
        rows.append({
            "concurrency": c, "boots": len(boot_means), "reps": len(xs),
            "mean": round(st.mean(xs), 1), "p10": round(pct(xs, 10), 1), "p50": round(pct(xs, 50), 1),
            "p90": round(pct(xs, 90), 1), "max": round(max(xs), 1),
            "boot_mean_min": round(min(boot_means), 1), "boot_mean_max": round(max(boot_means), 1),
            "ttft_p50_ms": round(pct(ts, 50), 1), "ttft_p90_ms": round(pct(ts, 90), 1),
        })
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decode-distribution.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"files {len(RUNS)}, request errors {errs}")
    # C1 split by tree and by run family, so no one run's boots dominate a pooled figure unseen.
    c1 = {}
    for box, tree, label, rel in RUNS:
        fam = "c1-settle (32 reps/boot)" if "settle" in label else "bench178 (8 reps/boot)"
        vals = [json.loads(l)["dec_med"] for l in open(os.path.join(QA, rel))
                if l.strip() and not l.startswith("#") and '"C": 1,' in l.replace('"C":1,', '"C": 1,')
                and json.loads(l).get("n_ok", 0) > 0]
        c1.setdefault((tree, fam), []).append((label, vals))
    split_rows = []
    print("\nC1 by tree and run family")
    print("| tree | run family | boots | reps | pooled mean | p50 | p90 | boot means (min–max) | mean of boot means |")
    print("|---|---|---|---|---|---|---|---|---|")
    groups = sorted(c1.items()) + [((t, "all"), [b for (tt, _), bs in c1.items() if tt == t for b in bs])
                                   for t in sorted({t for t, _ in c1})]
    for (tree, fam), boots in groups:
        xs = [v for _, vals in boots for v in vals]
        bm = [st.mean(vals) for _, vals in boots]
        r = {"tree": tree, "family": fam, "boots": len(boots), "reps": len(xs), "mean": round(st.mean(xs), 1),
             "p50": round(pct(xs, 50), 1), "p90": round(pct(xs, 90), 1), "boot_min": round(min(bm), 1),
             "boot_max": round(max(bm), 1), "mean_of_boot_means": round(st.mean(bm), 1)}
        split_rows.append(r)
        print(f"| {tree} | {fam} | {r['boots']} | {r['reps']} | {r['mean']} | {r['p50']} | {r['p90']} | "
              f"{r['boot_min']}–{r['boot_max']} | {r['mean_of_boot_means']} |")
    out2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decode-c1-by-tree.csv")
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        w.writeheader()
        w.writerows(split_rows)
    print()
    print("| streams | boots | reps | mean | p10 | p50 | p90 | max | boot means | TTFT p50 / p90 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['boots']} | {r['reps']} | {r['mean']} | {r['p10']} | {r['p50']} | "
              f"{r['p90']} | {r['max']} | {r['boot_mean_min']}–{r['boot_mean_max']} | "
              f"{r['ttft_p50_ms']} / {r['ttft_p90_ms']} |")


if __name__ == "__main__":
    main()
