#!/usr/bin/env python3
"""Loaded serving hours of config X' per box and tree, from prod sbr-vmsingle raw samples.

Inputs: a directory holding, per box (box-a, box-b), the JSON answers of two instant queries over range vectors
(raw samples, no step interpolation), evaluated at the read time END:
  run-<box>.json     sglang:num_running_reqs{backend="<box>"}[<H>h]
  health-<box>.json  sbr_backend_health_state{job="sbr-prod",backend="<box>"}[<H>h]   (one series per router pod)
plus end.txt (END as unix seconds). The raw answers carry router pod addresses and door host names, so they stay
out of this folder; the per-sample table this script writes carries none.

A sample counts as in the pool when every router pod with a health sample in the 60 s before it reads 0
(healthy); held (3) or DOWN (4) on any pod excludes it, so out-of-rotation benches and restarts never count.
A loaded sample is an in-pool sample with num_running_reqs > 0; each sample stands for 15 s (the scrape
interval; gaps and staleness markers add nothing, so loaded hours are a floor). Bound with 0 events in T loaded hours: 2.996 / T.

Usage: python3 xprime_loaded_hours.py <dir-with-json> <out.csv>   (prints the summary table)
"""
import bisect
import csv
import datetime as dt
import json
import os
import sys


def U(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def F(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Which tree each box served when (records: box-c/qa/prod-merge-20261030, box-c/qa/box-b-bringup-20261030,
# box-c/qa/sglang-takes-20261031/ROLL-box-a.md, box-c/qa/box-b-roll-20261031/NOTES.md). None = the read time.
WINDOWS = [
    ("box-a", "4943d6ed90", "2026-10-30T12:51:00Z", "2026-10-31T17:42:04Z"),
    ("box-b", "4943d6ed90", "2026-10-30T17:56:00Z", "2026-10-31T17:49:32Z"),
    ("box-a", "9a9ede7ae0", "2026-10-31T17:42:04Z", None),
    ("box-b", "9a9ede7ae0", "2026-10-31T18:06:10Z", None),
]
STEP = 15.0
BAND = lambda v: "0" if v == 0 else "1" if v == 1 else "2-4" if v <= 4 else ">=5"


def main(src, out):
    end = int(open(os.path.join(src, "end.txt")).read())
    rows, summary = [], []
    for box in ("box-a", "box-b"):
        run = json.load(open(os.path.join(src, f"run-{box}.json")))["data"]["result"][0]["values"]
        health = [([float(t) for t, _ in s["values"]], [v for _, v in s["values"]])
                  for s in json.load(open(os.path.join(src, f"health-{box}.json")))["data"]["result"]]
        for b, tree, a, z in WINDOWS:
            if b != box:
                continue
            a, z = U(a), (U(z) if z else end)
            bands = {"0": 0, "1": 0, "2-4": 0, ">=5": 0}
            peak, last = 0, None
            for t, v in run:
                t, v = float(t), float(v)
                if v != v or not (a <= t < z):  # NaN = vmsingle staleness marker (series ended), not a sample
                    continue
                cur = []
                for ts, vs in health:
                    i = bisect.bisect_right(ts, t) - 1
                    if i >= 0 and t - ts[i] <= 60:
                        cur.append(vs[i])
                in_pool = bool(cur) and all(c == "0" for c in cur)
                rows.append((F(t), box, tree, int(v), int(in_pool)))
                if in_pool:
                    bands[BAND(v)] += 1
                    peak = max(peak, v)
                    last = t
            h = {k: n * STEP / 3600 for k, n in bands.items()}
            loaded = h["1"] + h["2-4"] + h[">=5"]
            summary.append((box, tree, F(a), F(min(z, last or z)), sum(h.values()), h, loaded, int(peak)))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_utc", "box", "tree", "num_running_reqs", "in_pool"])
        w.writerows(rows)
    print("box  tree        window (UTC)                                 in-pool h  0 / 1 / 2-4 / >=5 running h       loaded h  peak")
    for box, tree, a, z, tot, h, loaded, peak in summary:
        print(f"{box} {tree}  {a} -> {z}  {tot:7.3f}  {h['0']:.3f} / {h['1']:.3f} / {h['2-4']:.3f} / {h['>=5']:.3f}  {loaded:8.3f}  {peak}")
    for tree in ("4943d6ed90", "9a9ede7ae0"):
        T = sum(s[6] for s in summary if s[1] == tree)
        print(f"{tree}: T = {T:.3f} loaded h, 0 hangs -> 95% bound {2.996 / T:.3f} per loaded h")
    T = sum(s[6] for s in summary)
    print(f"both X' trees counted as one: T = {T:.3f} loaded h -> {2.996 / T:.3f} per loaded h")
    print(f"read time {F(end)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
