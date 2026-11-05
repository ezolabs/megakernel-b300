#!/usr/bin/env python3
"""analyze.py (#405): the C1 settle's paired analysis over the four interleaved boots A1 B1 A2 B2.

A = prod 9a9ede7ae0 (candidate), B = 4943d6ed90 (control). Input: speed/<L>/think-<L>.jsonl from bench-c1.sh
(C1, each row one request, decode tok/s = dec_med). Prints per-boot mean / median / sd / n, then:
- per pair (A1-B1, A2-B2): candidate minus control, % of control, Welch 95 % CI over requests;
- over the two pairs: the mean paired difference with a t(1) 95 % CI (two pairs, 1 df: wide by design) and the
  pooled difference over requests (all A requests vs all B requests, Welch 95 % CI);
- boot as the unit: a two-sample t on the four boot means (2 df);
- a request-level linear model dec ~ tree + pair (pair as a block), 95 % CI on the tree term.
Verdict (the brief): 'cost confirmed' only if both pairs show A >= 2 % slower AND the CI over requests excludes 0;
otherwise 'no cost confirmed'.
Usage: python3 analyze.py [DIR]   (DIR defaults to the script's own speed/)
"""
import json
import math
import os
import statistics as st
import sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 15: 2.131, 20: 2.086,
        30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df):
    # conservative: the critical value of the largest tabulated df at or below df
    df = max(1, int(df))
    return T975[max(k for k in T975 if k <= df)]


def load(d, tag):
    p = os.path.join(d, tag, f"think-{tag}.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip() and not l.startswith("#")]
    ok = [r for r in rows if r["C"] == 1 and r["n_ok"] > 0]
    errs = sum(r["errs"] for r in rows)
    return [r["dec_med"] for r in ok], [1000 * r["ttft_med"] for r in ok], errs


def welch(a, b):
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    d = ma - mb
    h = t975(df) * se
    return d, d - h, d + h, df


def blocked(groups):
    """dec ~ tree + pair: OLS with pair fixed effects; returns tree coef (A - B), its 95 % CI, residual df."""
    # within each pair, centre on the pair mean of its two boot means' grand mean: equivalent to OLS with dummies
    xs, ys, pairs = [], [], []
    for pi, (a, b) in enumerate(groups):
        for v in a:
            xs.append(1.0); ys.append(v); pairs.append(pi)
        for v in b:
            xs.append(0.0); ys.append(v); pairs.append(pi)
    # demean x and y within pair
    xd, yd = [], []
    for pi in set(pairs):
        idx = [i for i, p in enumerate(pairs) if p == pi]
        mx = st.mean(xs[i] for i in idx)
        my = st.mean(ys[i] for i in idx)
        for i in idx:
            xd.append(xs[i] - mx); yd.append(ys[i] - my)
    sxx = sum(x * x for x in xd)
    beta = sum(x * y for x, y in zip(xd, yd)) / sxx
    res = [y - beta * x for x, y in zip(xd, yd)]
    df = len(ys) - 1 - len(set(pairs))
    s2 = sum(r * r for r in res) / df
    se = math.sqrt(s2 / sxx)
    h = t975(df) * se
    return beta, beta - h, beta + h, df


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed")
    tags = ["A1", "B1", "A2", "B2"]
    data = {t: load(d, t) for t in tags}
    print("boot  tree        n   errs  mean   median  sd     min    max    TTFT C1 ms (mean)")
    for t in tags:
        v, tt, e = data[t]
        tree = "9a9ede7ae0" if t[0] == "A" else "4943d6ed90"
        print(f"{t:4}  {tree}  {len(v):2}  {e:4}  {st.mean(v):6.1f} {st.median(v):6.1f}  {st.stdev(v):5.1f}  "
              f"{min(v):6.1f} {max(v):6.1f}  {st.mean(tt):5.1f}")
    print()
    pairs = [("A1", "B1"), ("A2", "B2")]
    pct = []
    for a, b in pairs:
        va, vb = data[a][0], data[b][0]
        dd, lo, hi, df = welch(va, vb)
        p = 100 * dd / st.mean(vb)
        pct.append(p)
        print(f"pair {a}-{b}: {dd:+6.1f} tok/s ({p:+.2f} %), Welch 95 % CI over requests [{lo:+.1f}, {hi:+.1f}] "
              f"tok/s ([{100 * lo / st.mean(vb):+.2f}, {100 * hi / st.mean(vb):+.2f}] %), df {df:.0f}")
    diffs = [st.mean(data[a][0]) - st.mean(data[b][0]) for a, b in pairs]
    md = st.mean(diffs)
    sd = st.stdev(diffs)
    h = t975(1) * sd / math.sqrt(2)
    ctl = st.mean(data["B1"][0] + data["B2"][0])
    print(f"over the two pairs (pair = unit, t(1)): mean {md:+.1f} tok/s ({100 * md / ctl:+.2f} %), 95 % CI "
          f"[{md - h:+.1f}, {md + h:+.1f}] ([{100 * (md - h) / ctl:+.2f}, {100 * (md + h) / ctl:+.2f}] %)")
    allA = data["A1"][0] + data["A2"][0]
    allB = data["B1"][0] + data["B2"][0]
    dd, lo, hi, df = welch(allA, allB)
    print(f"over requests, pooled (A {len(allA)} vs B {len(allB)}): {dd:+.1f} tok/s ({100 * dd / ctl:+.2f} %), Welch 95 % "
          f"CI [{lo:+.1f}, {hi:+.1f}] ([{100 * lo / ctl:+.2f}, {100 * hi / ctl:+.2f}] %), df {df:.0f}")
    beta, lo2, hi2, df2 = blocked([(data[a][0], data[b][0]) for a, b in pairs])
    print(f"over requests, pair-blocked OLS (dec ~ tree + pair): {beta:+.1f} tok/s ({100 * beta / ctl:+.2f} %), 95 % CI "
          f"[{lo2:+.1f}, {hi2:+.1f}] ([{100 * lo2 / ctl:+.2f}, {100 * hi2 / ctl:+.2f}] %), df {df2}")
    bmA = [st.mean(data["A1"][0]), st.mean(data["A2"][0])]
    bmB = [st.mean(data["B1"][0]), st.mean(data["B2"][0])]
    dd3, lo3, hi3, df3 = welch(bmA, bmB)
    print(f"boot = unit (4 boot means, Welch): {dd3:+.1f} tok/s, 95 % CI [{lo3:+.1f}, {hi3:+.1f}], df {df3:.1f}")
    both = all(p <= -2.0 for p in pct)
    excl = hi2 < 0 and hi < 0
    print()
    print(f"both pairs A >= 2 % slower: {both} ({pct[0]:+.2f} %, {pct[1]:+.2f} %); CI over requests excludes 0 on the "
          f"slow side: {excl}")
    print("VERDICT:", "cost confirmed" if (both and excl) else "no cost confirmed")


if __name__ == "__main__":
    main()
