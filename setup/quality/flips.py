#!/usr/bin/env python3
"""flips.py REF.jsonl X.jsonl [Y.jsonl ...] : teacher-forced top-1 flip sets against REF (tf.py raw).
A flip is a scored position (prompt i, position j) whose top-1 token id differs from REF's.
Prints each file's short (i 0-15) / long (i 16-31) flip count, then for every pair of the
files after REF the overlap of their flip sets (shared, only-first, only-second), with the
shared flips' top-1 agreement between the two (same wrong token or not)."""
import gzip, json, sys


def load(p):
    op = gzip.open if p.endswith(".gz") else open
    return [json.loads(l) for l in op(p, "rt")]


def flips(ref, x):
    out = {}
    for i, (a, b) in enumerate(zip(ref, x)):
        if a["n"] != b["n"]:
            raise SystemExit("prompt %d: n %d vs %d" % (i, a["n"], b["n"]))
        for j, (p, q) in enumerate(zip(a["top"], b["top"])):
            if p and q and p[0][1] != q[0][1]:
                out[(i, j)] = q[0][1]
    return out


ref = load(sys.argv[1])
names = sys.argv[2:]
F = {n: flips(ref, load(n)) for n in names}
part = {"short": lambda k: k[0] < 16, "long": lambda k: k[0] >= 16}
for n in names:
    print("%s: short %d flips, long %d flips" % (n, sum(1 for k in F[n] if k[0] < 16), sum(1 for k in F[n] if k[0] >= 16)))
for ai in range(len(names)):
    for bi in range(ai + 1, len(names)):
        a, b = names[ai], names[bi]
        for pn, pf in part.items():
            sa = {k for k in F[a] if pf(k)}
            sb = {k for k in F[b] if pf(k)}
            sh = sa & sb
            same = sum(1 for k in sh if F[a][k] == F[b][k])
            print("%s | %s vs %s: shared %d (same top-1 at %d), only-first %d, only-second %d" % (
                pn, a, b, len(sh), same, len(sa - sb), len(sb - sa)))
