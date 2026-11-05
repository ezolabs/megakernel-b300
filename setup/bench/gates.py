#!/usr/bin/env python3
"""Adoption gate: does an arm regress against its base beyond the A/A noise band on both boots, or miss on quality?

The rule (the user, 2026-10-27 ~22:35Z): an arm FAILs a metric only on a REGRESSION beyond the A/A band (the wrong
direction, outside the band) or on a quality miss. A delta inside the band, or a bootstrap interval that includes zero,
is "no proven gain": NEUTRAL, never FAIL. The gate reports which metrics show a proven gain and which are neutral.
Both boots (the user, 2026-10-27: "adopt when the arm beats the A/A noise band on both boots"): a regression counts
only when it replicates, the same sub-gate worse beyond the band on EVERY boot of the arm; worse on some boots only is
UNCONFIRMED, reported and never a FAIL, because one boot against a band from few A/A boots is worse beyond it by
chance far too often (test_gates.py NullSim). A quality miss still FAILs on any boot. The band is K_SIGMA x sigma-hat
x sqrt(1 + 1/nb) (below), sigma-hat from the base runs' within-boot SE and the real boot effect (BOOT_EXCESS),
K_SIGMA calibrated so a null arm passes >= 95% of the time on two base boots.

  gates.py                      window 2's record: TP4 (2 copies, boots b1 + b2) against window 1's TP8
  gates.py --arm b1=RUN[,RUN..] [--arm b2=...] --base RUN[,RUN..] [--base-aa RUNS]...
           [--arm-aa RUNS]... [--copies N] [--base-copies M] [--levels 2,4,8,16] [--metrics decode,...]
           [--bootstrap N] [--allow-one-base]

A RUN is one engine's measurements:
  DIR       a battery.sh raw dir: DIR/ramp-c<C>.jsonl, DIR/coldfill.jsonl, DIR/resume.jsonl
  DIR@TAG   a bench.py flat dir (window 1): DIR/think-TAG.jsonl (rows carry C), DIR/coldfill-TAG.jsonl,
            DIR/extend-TAG.jsonl

--arm LABEL=RUNS   one boot of the arm (repeat per boot; two or more for a verdict). Each boot is judged against the
                   base, then a regression must replicate on every boot. No arm run may also be a --base or --base-aa
                   run: a boot inside the base it is judged by is no A/A check (bad input).
--base [LABEL=]RUNS  the base; repeat for more groups, all pooled into one base. Its nb runs (boots, or copies)
                   set the band for a metric at a level: K_SIGMA x sigma-hat x sqrt(1 + 1/nb), sigma-hat being one
                   boot mean's relative noise, the runs' pooled within-boot SE (rep sd / sqrt(n)) x sqrt(1 +
                   BOOT_EXCESS[metric]^2), and sqrt(1 + 1/nb) the base mean's own noise. Each row prints sigma-hat. The
                   runs' between-boot sd is not used (one df on two boots; it set bands from 3% to 55% on box-a).
                   Fewer than two base runs is bad input (a null arm on two boots FAILs ~10% of the time against one,
                   test_gates.py NullSim) unless --allow-one-base is given, and then the header warns. A metric (at a
                   level) where no base run has two or more reps has no within-boot SE, so no band: bad input.
--copies N         the arm's runs in a boot are N copies serving side by side (window 2's TP4 pair: 2). Box total L
                   reads the arm at C = L/N; the aggregate is N x the per-copy mean; fill/dip/gap/resume are judged
                   per copy. --base-copies M does the same for the base (default 1).
--base-aa RUNS     optional, a model check: two or more runs of the BASE config (two boots or two copies, e.g. A0
                   vs A0b), read at the base's C. Not in the band. Where the sd of the group's run means passes
                   AA_Z_WARN x the group's own sigma-hat, a warning names the sub-gate: this box's boot effect is
                   larger than BOOT_EXCESS there, so its band is too narrow and the sub-gate is read by hand.
--arm-aa RUNS      the same model check on runs of the arm config, read at the arm's C.
--levels           box totals for decode / aggregate / ttft (default 2,4,8,16 at --copies 2, else 1,2,4,8,16).
--metrics          any of decode,aggregate,ttft,fill,dip,gap,resume,quality (default all). Only the files they need
                   are read.
--bootstrap N      bootstrap resamples for the CI (default 2000, at least 100).

Every metric sub-gate, per arm boot (and per copy for the fill-time metrics), gets one of four verdicts. With
delta = (arm - base) / base, "better" is delta > band for decode and aggregate (higher is better) and delta < -band
for ttft, fill, dip, gap and resume (lower is better); "worse" is the mirror:
  GAIN     better beyond the band AND the bootstrap 95% CI of arm - base excludes zero: a proven gain.
  NEUTRAL  no proven gain and no regression: inside the band, or better beyond it with a CI that includes zero.
  FAIL     a regression: worse beyond the band (no CI needed to fail) on EVERY boot of the arm (same metric, level
           and copy), or an arm outlier worse than the base median (below) on EVERY boot. Each kind replicates only
           with itself: an outlier FAIL inside the band on one boot does not confirm a beyond-band delta on another.
  UNCONFIRMED  either regression on some boots only, or on the only boot: reported, not a FAIL. A metric holding
           one reads UNCONFIRMED in the per-metric summary, never GAIN or neutral.
quality (per arm run, so per copy): every gsm8k file of the run must say PASS (gsm8k.py's >= 95% with no failed
  request): battery dirs read gsm8k*.txt (gsm8kT1.txt, gsm8kP16.txt), flat dirs gsm8k*-TAG.txt (gsm8k-TAG.txt,
  gsm8kT1-TAG.txt, ...); a run with none is bad input. A needle file (needle.jsonl / needle-TAG.jsonl), when present,
  must hit on every row; one with no rows, or a row with no boolean 'hit', is bad input, as is a gsm8k file with no
  score line. Any miss FAILs. Only arm runs are read for quality.

Outliers: a rep whose value is more than OUTLIER_FACTOR (2x) above or below its own run's median.
  base side (--base runs, and the --base-aa / --arm-aa model-check runs): dropped before pooling or banding, so a
           first-use stall stays out of the base mean and the band; each drop is printed.
  arm side: never silently dropped, and never excused by the base's own outlier rate (the user, 2026-10-27 ~22:35Z:
           "an outlier rep fails only if it's worse than the base median"). Each is printed and set against the base
           median (the pooled --base reps after their drops, at the aggregate's box scale): one WORSE than the base
           median stays in the arm's mean and FAILs the sub-gate ("arm outlier worse than base median"); one that
           still beats the base median is reported, not failed, and is dropped from the arm's mean so a lucky rep
           cannot make a gain. Each row also shows the arm's outlier count against the base's, for context only.

Exit: 0 no confirmed regression and no quality miss on two or more boots (the summary names the metrics with a
      proven gain, the sub-gates with a gain on every boot, the neutral ones and the unconfirmed ones), 1 a confirmed
      regression or any quality miss, 2 bad input (a missing file, a file with no rows for what is asked, a bad
      argument, an arm run that is also a base run, one base run without --allow-one-base, a sub-gate whose base
      runs have no two reps), 3 one arm boot and no quality miss: no verdict until a second boot.
"""
import argparse
import functools
import json
import math
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W1 = os.path.normpath(os.path.join(HERE, "..", "..", "window1-20261027", "raw"))
W2 = os.path.normpath(os.path.join(HERE, "..", "raw"))
TP8 = ["BV1", "BV2", "final"]
OUTLIER_FACTOR = 2.0
# The band is K_SIGMA x sigma-hat x sqrt(1 + 1/nb). sigma-hat is one boot mean's relative noise: the within-boot SE
# (rep sd / sqrt(n)) pooled over the nb base runs, widened by the real boot effect, sqrt(1 + BOOT_EXCESS[metric]^2);
# sqrt(1 + 1/nb) adds the base mean's own noise. The base runs' between-boot sd is not used: on two boots it has one df,
# so the band it set swung from 3% to 55% on box-a's A0/A0b. K_SIGMA is calibrated (test_gates.py NullSim,
# null_fail_rate) so a null arm on two boots against TWO base boots overall-PASSes >= 95% of the time over 19 sub-gates
# (2.5: ~97%; 2.25: ~94.9%).
K_SIGMA = 2.5
# Real A/A boots' between-boot sd beyond the within-boot SE, as a multiple of that SE: the rms over B0/B0b, A0/A0b,
# STK1/STK2, BV1/BV2/final and window 2's copies a and b over b1/b2 (30 level sub-gates per metric, 6 fill-time ones).
# Resume leaves out A0/A0b, whose A0 ran 5 reps against A0b's 20 (8.6 SE apart; 0 without it); the A/A model check
# prints such a pair. sigma-hat's widening and null_fail_rate's model.
BOOT_EXCESS = {"decode": 0.25, "aggregate": 0.25, "ttft": 1.2, "fill": 2.0, "dip": 0.7, "gap": 1.3}
# An A/A group whose run means sit more than this many sigma-hat apart (their sd) is printed as a model-check warning.
AA_Z_WARN = 3.0
LEVEL_METRICS = ("decode", "aggregate", "ttft")
FILL_METRICS = ("fill", "dip", "gap", "resume")
ALL_METRICS = LEVEL_METRICS + FILL_METRICS + ("quality",)
HIGHER = {"decode", "aggregate"}
UNIT = {"decode": "tok/s", "aggregate": "tok/s", "ttft": "s", "fill": "s", "dip": "", "gap": "s", "resume": "s"}


class BadInput(Exception):
    pass


def jl(p):
    if not os.path.isfile(p):
        raise BadInput(f"missing file: {p}")
    out = []
    with open(p) as f:
        for n, l in enumerate(f, 1):
            l = l.strip()
            if l.startswith("{"):
                try:
                    out.append(json.loads(l))
                except ValueError as e:
                    raise BadInput(f"{p}:{n}: not JSON ({e})")
    return out


def _dip(x):
    bg = [b for b in x.get("bg") or [] if not b.get("ended_before_fill_end")]
    pre = [b["pre"] for b in bg if b.get("pre") is not None]
    dur = [b["during"] for b in bg if b.get("during") is not None]
    if not pre or not dur or st.mean(pre) <= 0:
        return None
    return 1 - st.mean(dur) / st.mean(pre)


def _gap(x):
    g = [b["max_gap_during_s"] for b in x.get("bg") or [] if b.get("max_gap_during_s") is not None]
    return max(g) if g else None


class Run:
    def __init__(self, spec):
        self.spec = spec
        d, tag = spec, None
        if "@" in spec and not os.path.isdir(spec):
            d, tag = spec.rsplit("@", 1)
        if not os.path.isdir(d):
            raise BadInput(f"missing run dir: {d} (from {spec!r})")
        self.dir, self.tag = d, tag
        self.label = tag if tag else os.path.basename(os.path.normpath(d))
        self._cache = {}

    def key(self):
        return os.path.realpath(self.dir), self.tag

    def path(self, kind, C=None):
        if self.tag:
            return os.path.join(self.dir, {"ramp": f"think-{self.tag}.jsonl", "coldfill": f"coldfill-{self.tag}.jsonl",
                                           "resume": f"extend-{self.tag}.jsonl"}[kind])
        return os.path.join(self.dir, {"ramp": f"ramp-c{C}.jsonl", "coldfill": "coldfill.jsonl",
                                       "resume": "resume.jsonl"}[kind])

    def rows(self, kind, C=None):
        p = self.path(kind, C)
        if p not in self._cache:
            self._cache[p] = jl(p)
        return p, self._cache[p]

    def values(self, metric, C=None):
        """The run's per-rep values for one metric (at C for the level metrics), before the outlier policy."""
        if metric in LEVEL_METRICS:
            p, rows = self.rows("ramp", C)
            field = {"decode": "dec_med", "aggregate": "agg_tok_s", "ttft": "ttft_med"}[metric]
            v = [x[field] for x in rows if x.get("C") == C and x.get(field) is not None]
            what = f"{field} at C={C}"
        elif metric == "resume":
            p, rows = self.rows("resume")
            v = [x["ext_ttft"] for x in rows if x.get("ext_ttft") is not None]
            what = "ext_ttft"
        else:
            p, rows = self.rows("coldfill")
            fill = [x for x in rows if x.get("fill_ttft_s") is not None]
            f = {"fill": lambda x: x["fill_ttft_s"], "dip": _dip, "gap": _gap}[metric]
            v = [f(x) for x in fill]
            v = [y for y in v if y is not None]
            what = {"fill": "fill_ttft_s", "dip": "bg pre/during", "gap": "bg max_gap_during_s"}[metric]
        if not v:
            raise BadInput(f"no {what} rows in {p}")
        return v

    def quality(self):
        """[(ok, text)] for each gsm8k file of the run (at least one) and its needle file when present."""
        if self.tag:
            pat = re.compile(rf"gsm8k([A-Za-z0-9]*)-{re.escape(self.tag)}\.txt$")
            needle = os.path.join(self.dir, f"needle-{self.tag}.jsonl")
        else:
            pat = re.compile(r"gsm8k([A-Za-z0-9]*)\.txt$")
            needle = os.path.join(self.dir, "needle.jsonl")
        files = sorted(f for f in os.listdir(self.dir) if pat.match(f))
        if not files:
            want = f"gsm8k*-{self.tag}.txt" if self.tag else "gsm8k*.txt"
            raise BadInput(f"missing file: no {want} in {self.dir} (quality needs the run's gsm8k)")
        out = []
        for f in files:
            p = os.path.join(self.dir, f)
            with open(p) as fh:
                lines = [l for l in fh if "GSM8K sampled:" in l]
            m = re.search(r"GSM8K sampled: (\d+)/(\d+).*\b(PASS|FAIL)\b", lines[-1]) if lines else None
            if not m:
                raise BadInput(f"no 'GSM8K sampled: N/M ... PASS|FAIL' line in {p}")
            out.append((m.group(3) == "PASS", f"gsm8k{pat.match(f).group(1)} {m.group(1)}/{m.group(2)} {m.group(3)}"))
        if os.path.isfile(needle):
            rows = jl(needle)
            if not rows:
                raise BadInput(f"no needle rows in {needle} (a needle file present must be scored)")
            for n, x in enumerate(rows, 1):
                if not isinstance(x.get("hit"), bool):
                    raise BadInput(f"{needle}: row {n} has no boolean 'hit' (needle.py's schema)")
            hits = sum(1 for x in rows if x["hit"])
            out.append((hits == len(rows), f"needle {hits}/{len(rows)}"))
        return out


def drop_outliers(v):
    """The one outlier policy: drop a rep more than OUTLIER_FACTOR x off its own run's median, either way."""
    med = st.median(v)
    keep, gone = [], []
    for i, x in enumerate(v):
        if med > 0 and (x > OUTLIER_FACTOR * med or x < med / OUTLIER_FACTOR):
            gone.append((i, x))
        else:
            keep.append(x)
    return keep, gone, med


class Sampler:
    def __init__(self, nboot, seed=1):
        self.nboot = nboot
        self.rnd = random.Random(seed)
        self.dropped = []
        self._clean = {}
        self.arm_out = []
        self._split = {}

    def split(self, run, metric, C=None):
        """(all reps, reps kept by the outlier rule, [(index, value) of each outlier], run median)."""
        k = (run.spec, metric, C)
        if k not in self._split:
            v = run.values(metric, C)
            keep, gone, med = drop_outliers(v)
            self._split[k] = (v, keep, gone, med)
        return self._split[k]

    @staticmethod
    def _desc(run, metric, C, i, x, med):
        return (f"{run.label} {metric}{f' C={C}' if C is not None else ''}: rep {i + 1} = {x:g} "
                f"(run median {med:g}, > {OUTLIER_FACTOR:g}x off)")

    def clean(self, run, metric, C=None):
        """Base-side reps: outliers dropped, each drop printed once."""
        k = (run.spec, metric, C)
        v, keep, gone, med = self.split(run, metric, C)
        if k not in self._clean:
            self._clean[k] = keep
            self.dropped += [self._desc(run, metric, C, i, x, med) + " dropped" for i, x in gone]
        return keep

    def arm(self, run, metric, C=None):
        """Arm-side reps: (all reps, kept reps, outlier count); each outlier printed once."""
        k = ("arm", run.spec, metric, C)
        v, keep, gone, med = self.split(run, metric, C)
        if k not in self._clean:
            self._clean[k] = keep
            self.arm_out += [self._desc(run, metric, C, i, x, med) for i, x in gone]
        return v, keep, len(gone)

    def ci(self, a, b, sa=1.0, sb=1.0):
        """Bootstrap 95% CI of sa*mean(a) - sb*mean(b)."""
        n = self.nboot
        vals = []
        for _ in range(n):
            ra = [a[self.rnd.randrange(len(a))] for _ in a]
            rb = [b[self.rnd.randrange(len(b))] for _ in b]
            vals.append(sa * st.mean(ra) - sb * st.mean(rb))
        vals.sort()
        return vals[int(.025 * n)], vals[min(n - 1, int(.975 * n))]


def parse_group(s, flag, need_label=False):
    label, _, runs = s.rpartition("=") if "=" in s else ("", "", s)
    if need_label and not label:
        raise BadInput(f"{flag} {s!r}: expected LABEL=RUN[,RUN...]")
    specs = [r for r in runs.split(",") if r]
    if not specs:
        raise BadInput(f"{flag} {s!r}: no runs")
    return label, [Run(r) for r in specs]


def window2_defaults():
    r = lambda c, b: os.path.join(W2, f"raw-{c}-{b}")
    t = lambda tag: f"{W1}@{tag}"
    return dict(arm=[f"b1={r('a', 'b1')},{r('b', 'b1')}", f"b2={r('a', 'b2')},{r('b', 'b2')}"],
                base=[",".join(t(x) for x in TP8)],
                base_aa=[",".join(t(x) for x in TP8)], copies=2)


def evaluate(a):
    """Every sub-gate row: dicts with metric, level, unit, arm, base, delta, band, ci, verdict, why."""
    if a.bootstrap < 100:
        raise BadInput(f"--bootstrap {a.bootstrap}: need at least 100 resamples")
    S = Sampler(a.bootstrap)
    metrics = a.metrics
    arms = [parse_group(s, "--arm", need_label=True) for s in a.arm]
    if len({l for l, _ in arms}) != len(arms):
        raise BadInput("--arm labels must be distinct")
    base_runs = [r for s in a.base for r in parse_group(s, "--base")[1]]
    aa = [("base", parse_group(s, "--base-aa")[1]) for s in a.base_aa] + \
         [("arm", parse_group(s, "--arm-aa")[1]) for s in a.arm_aa]
    for side, g in aa:
        if len(g) < 2:
            raise BadInput(f"--{side}-aa needs two or more runs, got {[r.spec for r in g]}")
    # an arm boot inside the base or the band is no A/A check: its delta to a pair it belongs to is about the band
    # itself by construction (gate-AA's B0b against B0+B0b), so it sits on the band's edge and flips on rounding
    base_side = {r.key() for r in base_runs} | {r.key() for side, g in aa if side == "base" for r in g}
    for l, runs in arms:
        for r in runs:
            if r.key() in base_side:
                raise BadInput(f"--arm {l}: {r.spec} is also a --base or --base-aa run; an A/A check judges a "
                               f"held-out base boot against a base and band built from OTHER base boots")
    if len(base_runs) < 2 and not a.allow_one_base:
        raise BadInput(f"--base: {len(base_runs)} base run(s); the band needs two or more (against one, a null arm on "
                       f"two boots FAILs ~10% of the time), or pass --allow-one-base to accept that")
    N, M = a.copies, a.base_copies
    if N < 1 or M < 1:
        raise BadInput("--copies and --base-copies must be >= 1")
    if N > 1:
        for l, runs in arms:
            if len(runs) != N:
                raise BadInput(f"--arm {l}: --copies {N} needs exactly {N} runs per boot, got {len(runs)}")
    levels = a.levels or ([L for L in (1, 2, 4, 8, 16) if L % N == 0 and L % M == 0])
    for L in levels:
        if L % N or L % M:
            raise BadInput(f"level {L} is not divisible by --copies {N} / --base-copies {M}")

    def cfor(side, L):
        return None if L is None else L // (N if side == "arm" else M)

    # Read everything first, so any missing file is bad input before a verdict is printed.
    qual = []
    if "quality" in metrics:
        for label, runs in arms:
            for r in runs:
                qual.append(dict(unit=f"{label}/{r.label}" if len(runs) > 1 else label, checks=r.quality()))
    plan = [(m, L) for m in metrics if m != "quality" for L in (levels if m in LEVEL_METRICS else [None])]
    for m, L in plan:
        for _, runs in arms:
            for r in runs:
                S.arm(r, m, cfor("arm", L))
        for r in base_runs:
            S.clean(r, m, cfor("base", L))
        for side, g in aa:
            for r in g:
                S.clean(r, m, cfor(side, L))

    rows, checks = [], []
    n_base = len(base_runs)
    for m, L in plan:
        sigma = sigma_hat([S.clean(r, m, cfor("base", L)) for r in base_runs], m)
        if not sigma:
            raise BadInput(f"{m}{'' if L is None else f' at box total {L}'}: no --base run has two or more reps "
                           f"(or their reps are identical, or their mean is 0), so there is no within-boot SE and no "
                           f"band; add reps")
        band = K_SIGMA * sigma * math.sqrt(1 + 1 / n_base)
        band_by = f"{K_SIGMA:g} x sigma-hat x sqrt(1 + 1/{n_base})"
        # the model check: an A/A group's runs further apart than the model's boot noise allows means this box's boot
        # effect is larger than BOOT_EXCESS says, so the band is too narrow there; it is printed, not fed to the band
        for side, g in aa:
            vs = [S.clean(r, m, cfor(side, L)) for r in g]
            means = [st.mean(v) for v in vs]
            gs = sigma_hat(vs, m)
            if not gs:
                checks.append(f"{'+'.join(r.label for r in g)} {m}{'' if L is None else f' C{L}'}: no run has two "
                              f"or more reps, so no model check there: read it by hand")
                continue
            z = st.stdev(means) / abs(st.mean(means)) / gs
            if z > AA_Z_WARN:
                checks.append(f"{'+'.join(r.label for r in g)} {m}{'' if L is None else f' C{L}'}: run means "
                              f"{z:.1f} sigma-hat apart (sd), beyond the model's boot effect (BOOT_EXCESS "
                              f"{BOOT_EXCESS.get(m, 0.0):g}): the band is too narrow there, read it by hand")
        bv = [x for r in base_runs for x in S.clean(r, m, cfor("base", L))]
        b_all = sum(len(S.split(r, m, cfor("base", L))[0]) for r in base_runs)
        b_rate = (b_all - len(bv)) / b_all
        bscale = M if m == "aggregate" else 1
        base_val = bscale * st.mean(bv)
        base_med = bscale * st.median(bv)
        units = []
        for label, runs in arms:
            if N > 1 and m in FILL_METRICS:
                units += [(f"{label}/{r.label}", label, i, [r]) for i, r in enumerate(runs)]
            else:
                units.append((label, label, None, runs))
        for ulabel, boot, copy, runs in units:
            per = [S.arm(r, m, cfor("arm", L)) for r in runs]
            a_all, a_keep = [x for v, _, _ in per for x in v], [x for _, k, _ in per for x in k]
            n_out = sum(n for _, _, n in per)
            a_rate = n_out / len(a_all)
            ascale = N if m == "aggregate" else 1
            # an arm outlier rep is worse than the base median when, at the box scale, it is on the wrong side of it
            worse = (lambda x: ascale * x < base_med) if m in HIGHER else (lambda x: ascale * x > base_med)
            outs = [x for r in runs for _, _, gone, _ in [S.split(r, m, cfor("arm", L))] for _, x in gone]
            bad_out = [x for x in outs if worse(x)]
            # kept reps, plus only the outliers worse than the base median (a better one never makes a gain)
            av = a_keep + bad_out
            arm_val = ascale * st.mean(av)
            delta = (arm_val - base_val) / abs(base_val) if base_val else float("inf")
            lo, hi = S.ci(av, bv, ascale, bscale)
            better, worse_d = (delta > band, delta < -band) if m in HIGHER else (delta < -band, delta > band)
            proven = lo > 0 if m in HIGHER else hi < 0
            if worse_d:
                verdict, why = "FAIL", "worse beyond band"
            elif better and proven:
                verdict, why = "GAIN", "better beyond band"
            elif better:
                verdict, why = "NEUTRAL", "better beyond band but CI includes 0: no proven gain"
            else:
                verdict, why = "NEUTRAL", "inside band"
            kinds = {"band"} if worse_d else set()
            if n_out:
                rate = f"{n_out}/{len(a_all)} arm vs {b_all - len(bv)}/{b_all} base"
                nb = len(outs) - len(bad_out)
                if bad_out:
                    verdict = "FAIL"
                    kinds.add("outlier")
                    why += (f"; arm outliers {rate}, {len(bad_out)} worse than base median "
                            f"{fmt(m, base_med)}, kept in the mean")
                if nb:
                    why += (f"; {'' if bad_out else f'arm outliers {rate}, '}{nb} better than base median "
                            f"{fmt(m, base_med)}, reported and dropped")
            rows.append(dict(metric=m, level=L, unit=ulabel, boot=boot, copy=copy, arm=arm_val, base=base_val,
                             n_arm=len(av), n_base=len(bv), delta=delta, band=band, band_by=band_by, ci=(lo, hi),
                             sigma=sigma,
                             verdict=verdict, why=why, arm_outliers=n_out, arm_rate=a_rate, base_rate=b_rate,
                             kinds=kinds))
    # The both-boots rule (the user, 2026-10-27): a regression counts only when it replicates, the same sub-gate
    # (metric, level, copy) worse on EVERY boot of the arm. Worse on some boots only is UNCONFIRMED: reported, never
    # a FAIL.
    # Each kind replicates only with itself: a delta worse beyond the band on every boot, or an arm outlier worse than
    # the base median on every boot. An outlier FAIL inside the band on one boot confirms no beyond-band regression on
    # another, nor the reverse.
    boots = [l for l, _ in arms]
    said = {"band": "worse", "outlier": "a worse arm outlier"}
    for r in rows:
        if r["verdict"] != "FAIL":
            continue
        same = [x for x in rows if (x["metric"], x["level"], x["copy"]) == (r["metric"], r["level"], r["copy"])]
        bad = {k: [x["boot"] for x in same if k in x["kinds"]] for k in r["kinds"]}
        hit = [k for k in ("band", "outlier") if k in bad and len(boots) > 1 and len(bad[k]) == len(boots)]
        if hit:
            r["why"] += "; " + " and ".join(said[k] for k in hit) + " on every boot: confirmed"
            continue
        r["verdict"] = "UNCONFIRMED"
        if len(boots) == 1:
            r["why"] += "; one arm boot: a second boot confirms or clears it"
        else:
            for k in ("band", "outlier"):
                if k in bad:
                    ok = [b for b in boots if b not in bad[k]]
                    r["why"] += f"; {said[k]} on {'+'.join(bad[k])} only, not on {'+'.join(ok)}"
            r["why"] += ": not a FAIL"
    for q in qual:
        q["verdict"] = "PASS" if all(ok for ok, _ in q["checks"]) else "FAIL"
    groups = [(side, "+".join(r.label for r in g)) for side, g in aa]
    return rows, qual, S.dropped, S.arm_out, groups, checks, boots, n_base


def sigma_hat(runs, metric):
    """One boot mean's relative noise from base runs' rep values: the pooled within-boot SE (rep sd / sqrt(n), averaged
    as a variance over the runs, stable on many df) times sqrt(1 + BOOT_EXCESS[metric]^2), the real boot effect beyond
    it. Relative to the mean of the run means. The runs' between-boot sd is not used: on two boots it has one df.
    0.0 when no run has two reps or the mean is 0: no estimate (bad input for the band, no model check for an A/A
    group)."""
    runs = [v for v in runs if v]
    means = [st.mean(v) for v in runs]
    mu = abs(st.mean(means)) if means else 0.0
    se2 = [st.variance(v) / len(v) for v in runs if len(v) > 1]
    if not mu or not se2:
        return 0.0
    return math.sqrt(st.mean(se2) * (1 + BOOT_EXCESS.get(metric, 0.0) ** 2)) / mu


def null_fail_rate(n_base, n_boots, metrics, trials=4000, seed=0, k=None, reps=8, excess=None):
    """The chance a NULL arm (the base config again) FAILs this gate: per sub-gate (one per entry of metrics), every run
    (n_base base runs, n_boots arm boots) a boot mean of unit within-boot SE plus a boot effect of excess[metric] x that
    SE (default BOOT_EXCESS, the gate's own model), its reps' SE estimated on reps - 1 df; the band is the gate's,
    k x sqrt(1 + BOOT_EXCESS^2) x pooled SE x sqrt(1 + 1/n_base), and the sub-gate FAILs when every arm boot is worse
    than the base mean by more than it. An estimate, not data; excess other than BOOT_EXCESS probes a wrong model."""
    ex = tuple(sorted((excess or BOOT_EXCESS).items()))
    return _null_fail_rate(n_base, n_boots, tuple(metrics), trials, seed, K_SIGMA if k is None else k, reps, ex)


@functools.lru_cache(maxsize=None)
def _null_fail_rate(n_base, n_boots, metrics, trials, seed, k, reps, excess):
    true_ex = dict(excess)
    rnd = random.Random(seed)
    se2 = lambda: rnd.gammavariate((reps - 1) / 2, 2) / (reps - 1)
    fails = 0
    for _ in range(trials):
        for m in metrics:
            ex = true_ex.get(m, 0.0)
            boot = lambda: rnd.gauss(0, ex) + rnd.gauss(0, 1)
            mu = sum(boot() for _ in range(n_base)) / n_base
            sig = math.sqrt(sum(se2() for _ in range(n_base)) / n_base * (1 + BOOT_EXCESS.get(m, 0.0) ** 2))
            band = k * sig * math.sqrt(1 + 1 / n_base)
            if all(boot() - mu > band for _ in range(n_boots)):
                fails += 1
                break
    return fails / trials


def fmt(m, v):
    if m == "dip":
        return f"{v * 100:.1f}%"
    if UNIT[m] == "s":
        return f"{v * 1000:.1f}ms" if m in ("gap", "resume", "ttft") else f"{v:.3f}s"
    return f"{v:.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--base", action="append", default=[])
    p.add_argument("--arm-aa", action="append", default=[])
    p.add_argument("--base-aa", action="append", default=[])
    p.add_argument("--copies", type=int, default=None)
    p.add_argument("--base-copies", type=int, default=1)
    p.add_argument("--levels", default=None)
    p.add_argument("--metrics", default=",".join(ALL_METRICS))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--allow-one-base", action="store_true")
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        if not (a.arm or a.base or a.arm_aa or a.base_aa):
            d = window2_defaults()
            a.arm, a.base, a.base_aa = d["arm"], d["base"], d["base_aa"]
            a.copies = d["copies"] if a.copies is None else a.copies
            print("window 2 defaults: arm = TP4 2 copies, boots b1 + b2 (raw/raw-<a|b>-<b1|b2>); "
                  "base = window 1 TP8 BV1 + BV2 + final; A/A model check = BV1/BV2/final")
        elif not a.arm or not a.base:
            raise BadInput("--arm and --base are both required once any input is given")
        a.copies = 1 if a.copies is None else a.copies
        a.metrics = [m for m in a.metrics.split(",") if m]
        bad = [m for m in a.metrics if m not in ALL_METRICS]
        if bad or not a.metrics:
            raise BadInput(f"--metrics: unknown {bad}; choose from {','.join(ALL_METRICS)}")
        if a.levels is not None:
            try:
                a.levels = [int(x) for x in a.levels.split(",") if x]
            except ValueError:
                raise BadInput(f"--levels {a.levels!r}: expected comma-separated integers")
        rows, qual, dropped, arm_out, groups, checks, boots, n_base = evaluate(a)
    except BadInput as e:
        print(f"BAD INPUT: {e}", file=sys.stderr)
        print("OVERALL: BAD INPUT (exit 2)")
        return 2
    print(f"band: {K_SIGMA:g} x sigma-hat x sqrt(1 + 1/{n_base}) over {n_base} base run(s); sigma-hat = one boot "
          f"mean's noise, the base runs' pooled within-boot SE (rep sd / sqrt(n)) x sqrt(1 + BOOT_EXCESS^2), "
          f"BOOT_EXCESS {', '.join(f'{m} {x:g}' for m, x in BOOT_EXCESS.items())} (else 0)")
    if n_base < 2:
        print("  WARNING: ONE base run (--allow-one-base): the base mean carries one boot's effect, and a null arm on "
              "two boots FAILs ~10% of the time (test_gates.py NullSim); a second base boot brings it under 5%")
    print(f"A/A model check (printed, not in the band; a group's run means more than {AA_Z_WARN:g} sigma-hat apart):")
    for side, g in groups:
        print(f"  {side}-aa: {g}")
    if not groups:
        print("  no --base-aa group given")
    for c in checks:
        print(f"  WARNING: {c}")
    if groups and not checks:
        print("  every A/A group within the model")
    print(f"outlier policy: a rep > {OUTLIER_FACTOR:g}x above or below its run's median; base side dropped; arm side "
          f"FAILed if worse than the base median, else reported and dropped (whatever the base's rate)")
    for d in dropped:
        print(f"  dropped (base side, band): {d}")
    if not dropped:
        print("  dropped (base side, band): none")
    for d in arm_out:
        print(f"  arm outlier: {d}")
    if not arm_out:
        print("  arm outlier: none")
    print(f"{'metric':<9} {'box':>4} {'unit':<16} {'arm':>10} {'base':>10} {'delta':>7} {'band':>8} {'sigma^':>6} "
          f"{'95% CI arm-base':>22}  verdict")
    for r in rows:
        lo, hi = r["ci"]
        ci = f"[{fmt(r['metric'], lo) if r['metric'] != 'dip' else f'{lo * 100:+.1f}'}, " \
             f"{fmt(r['metric'], hi) if r['metric'] != 'dip' else f'{hi * 100:+.1f}'}]"
        print(f"{r['metric']:<9} {r['level'] if r['level'] is not None else '-':>4} {r['unit']:<16} "
              f"{fmt(r['metric'], r['arm']):>10} {fmt(r['metric'], r['base']):>10} {r['delta'] * 100:+6.1f}% "
              f"{r['band'] * 100:7.1f}% {r['sigma'] * 100:5.2f}% {ci:>22}  {r['verdict']} ({r['why']}; "
              f"band from {r['band_by']})")
    for q in qual:
        print(f"{'quality':<9} {'-':>4} {q['unit']:<16} {', '.join(t for _, t in q['checks'])}  {q['verdict']}")
    gain, neutral, unconf = [], [], []
    for m in a.metrics:
        if m == "quality":
            nf = sum(q["verdict"] == "FAIL" for q in qual)
            print(f"gate {m:<9} {'FAIL' if nf else 'PASS'} ({len(qual) - nf}/{len(qual)} arm runs pass)")
            continue
        mr = [r for r in rows if r["metric"] == m]
        n = {v: sum(r["verdict"] == v for r in mr) for v in ("GAIN", "NEUTRAL", "FAIL", "UNCONFIRMED")}
        # a metric holding an unconfirmed regression reads UNCONFIRMED, never GAIN or neutral, whatever else it holds
        g = "FAIL" if n["FAIL"] else "UNCONFIRMED" if n["UNCONFIRMED"] else "GAIN" if n["GAIN"] else "NEUTRAL"
        if g == "GAIN":
            gain.append(f"{m} ({n['GAIN']}/{len(mr)})")
        elif g == "NEUTRAL":
            neutral.append(m)
        elif g == "UNCONFIRMED":
            unconf.append(f"{m} ({n['UNCONFIRMED']}/{len(mr)}" + (f", gain {n['GAIN']}" if n["GAIN"] else "") + ")")
        print(f"gate {m:<9} {g:<11} (gain {n['GAIN']}, neutral {n['NEUTRAL']}, regression {n['FAIL']}, "
              f"unconfirmed {n['UNCONFIRMED']} of {len(mr)} sub-gates)")
    print(f"proven gain: {', '.join(gain) or 'none'}")
    sub = {}
    for r in rows:
        sub.setdefault((r["metric"], r["level"], r["copy"]), []).append(r)
    both = [k for k, rs in sub.items() if len(boots) > 1 and len(rs) == len(boots)
            and all(r["verdict"] == "GAIN" for r in rs)]
    print(f"gain on every boot: {len(both)} of {len(sub)} sub-gates"
          + (f" ({', '.join(m + ('' if L is None else f' C{L}') + ('' if c is None else f' copy {c + 1}') for m, L, c in both)})"
             if both else ""))
    print(f"neutral (no proven gain, no regression): {', '.join(neutral) or 'none'}")
    print(f"unconfirmed (a regression not on every boot or on the only one, not a FAIL): {', '.join(unconf) or 'none'}")
    if rows:
        nb = max(len(boots), 2)
        print(f"null check: a null arm (the base config again) on {nb} boots would FAIL this design's {len(sub)} "
              f"sub-gates ~{null_fail_rate(n_base, nb, [k[0] for k in sub]) * 100:.0f}% of the time ({n_base} base "
              f"run(s); Gaussian model with the real A/A boot effects, BOOT_EXCESS)")
    regs = sum(r["verdict"] == "FAIL" for r in rows)
    unc = sum(r["verdict"] == "UNCONFIRMED" for r in rows)
    misses = sum(q["verdict"] == "FAIL" for q in qual)
    tally = (f"{regs} regression(s) in {len(rows)} metric sub-gates, {unc} unconfirmed (worse on some boots only), "
             f"{misses} quality miss(es) in {len(qual)} arm runs")
    if regs or misses:
        print(f"OVERALL: FAIL ({tally}) -> not adopted (exit 1)")
        return 1
    if len(boots) < 2:
        print(f"OVERALL: ONE BOOT ({tally}) -> no verdict: a regression must replicate on both boots and a gain "
              f"must beat the band on both, so run a second arm boot (exit 3)")
        return 3
    print(f"OVERALL: PASS ({tally}; proven gain on {sum(r['verdict'] == 'GAIN' for r in rows)} of {len(rows)}, "
          f"on every boot {len(both)} of {len(sub)}) (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
