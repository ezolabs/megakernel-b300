#!/usr/bin/env python3
"""greedycmp.py X.jsonl REF1.jsonl [REF2 ...] : greedy.py outputs (sequential, T0, 256 tokens), byte-equal
(sha) count of X against each REF, and the median common-prefix length in characters of the texts."""
import json, os, sys
def load(p): return {r["i"]: r for r in map(json.loads, open(p))}
def cp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
x = load(sys.argv[1])
for ref in sys.argv[2:]:
    r = load(ref); eq = [i for i in sorted(x) if i in r and r[i]["sha"] == x[i]["sha"]]
    cps = sorted(cp(x[i]["text"], r[i]["text"]) for i in sorted(x) if i in r)
    print("%s vs %s: %d/%d byte-equal; differ at %s; common prefix chars median %d min %d" % (
        os.path.basename(sys.argv[1]), os.path.basename(ref), len(eq), len(x),
        [i for i in sorted(x) if i not in eq], cps[len(cps) // 2], cps[0]))
