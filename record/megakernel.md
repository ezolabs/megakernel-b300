# The megakernel on 4× B300: speed, numerics, cost (box-a, 2026-10-28 to 10-30)

Everything here comes from the dated record `box-c/qa/box-a-megakernel-20261029/README.md` (MK below = that
folder) unless a row names another file. All load was synthetic and off every router pool. Model L3a
(`repack-1017-l3a`) throughout. Machine-readable rows: `data/measurements.csv` (area `megakernel`).

## What was tested

NVIDIA's `sgl-nvidia-mega` megakernel plugin (megakernel API v2, plugin main `54b4289`) plus our B300 patch
(148 resident blocks instead of 152, to match the B300's 148 SMs, and `Feature.KV_OFFLOAD` so HiCache works),
on our SGLang fork. The patch is `fleet-history/box-a/sgl-nvidia-mega-b300-spec.patch`; the
applied diff is `MK/raw/build-latest/plugin-applied.diff`. Upstream later merged a similar 148-SM gate
(sgl-nvidia-mega PR #35, `b61745e`, 10-30) behind an opt-in env; `KV_OFFLOAD` remains ours.

| arm | tree | what it is |
|---|---|---|
| final (FB, FREF) | `819756d4` | box-a's pre-megakernel served stack: EP4, `stack-b300-box-a.args` + `env-box-a-w1.txt` |
| EP1old | `819756d4` | 10-28 MK3 minus `--enable-nvidia-megakernel` (EP1, HiCache direct, MK env) |
| ENVdelta | `819756d4` | the final (EP4) + the MK env deltas only |
| EP1only | `819756d4` | the final with only `--ep-size 4` → `1` |
| latest-F (LF1) | `box-a-mk-latest` | the final's args on the new merged tree, EP4, no MK |
| MK-latest (MKL1 = S4, MKL2 = S7) | `box-a-mk-latest` (`41bb61ec53`+fix, then `9ec65dacaf`) | megakernel on, plugin pin `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample` |
| MK-syncsample (S6B) | `box-a-mk-latest` | MK-latest with a patched plugin that broadcasts the sampled DSpark accept |
| MK-syncall | `box-a-mk-latest` | built, never booted |

Sessions S1–S7 each have a `MK/raw/s<N>-*/NOTES.md`. Args/env files for every arm: `MK/args/`. The FB arm's own
raw (10-28 19:49Z boot, the baseline every "vs FB" gate reads: per-rep think-ramp and cold-fill rows, acceptance,
gsm8k, greedy and teacher-forced outputs, boot record) is in `data/raw/fb/`; it was measured before the MK record
existed and lives outside `box-c/qa/`.

## Speed: decode tok/s per stream, MK vs the final (S4, S7)

Source: `MK/raw/s4-mk-b1/gates-MKL1-vs-FB.txt`, `MK/raw/s7-mk-b2/gates-MKL1+MKL2-vs-FB.txt`,
`MK/raw/s7-mk-b2/gates-MKL2-vs-MKL1.txt`, `MK/raw/s4-mk-b1/gates-MKL1-vs-MK3a.txt`. Copies in `data/raw/megakernel/`.

| streams | MKL1 (S4) | MKL2 (S7) | final FB | Δ MKL1 | Δ MKL2 | MKL1 vs old plugin MK3a |
|---|---|---|---|---|---|---|
| 1 | 885.8 | 849.0 | 638.4 | **+38.8%** gain | **+33.0%** gain | +2.1% neutral |
| 2 | 687.9 | 688.5 | 530.6 | **+29.6%** gain | **+29.8%** gain | −1.9% neutral |
| 4 | 556.6 | 557.5 | 450.0 | **+23.7%** gain | **+23.9%** gain | −1.0% neutral |
| 8 | 392.0 | 402.4 | 350.9 | +11.7% gain | +14.7% gain | −1.0% neutral |
| 16 | 297.9 | 296.7 | 294.6 | +1.1% neutral | +0.7% neutral | −0.0% neutral |

- `gates.py` over both boots: **OVERALL PASS, 0 regressions in 38 metric sub-gates**; decode and aggregate at
  C1/C2/C4/C8 a gain on every boot; the A/A model check holds.
- S7 vs S4 (MK A/A): all 19 sub-gates NEUTRAL; the largest move, decode C1 −4.2%, is inside its 5.9% band.
- Aggregate tok/s follows decode. TTFT neutral at 1/2/4/16 streams on both boots. S4's TTFT at 8 streams read
  +12.2% (one rep 0.209 s vs a run median 0.0985 s) and did not replicate (S7 +3.5%, neutral).
- Cold fill, decode dip, gap and resume neutral on both boots: fill 1.419 s / 1.450 s vs the final's 1.468 s;
  dip 77.0% / 75.8% vs 73.2%.
- S5's shorter `speed.sh` (8 reps at C1/C2/C4, probe on) sat inside MKL1's band: S5P −4.5% / −0.2% / −2.3%,
  S6BL −3.0% / +6.1% / −1.7%, all NEUTRAL (`MK/raw/s5-soak/NOTES.md`, `MK/raw/s6b-syncsample/NOTES.md` §2).
- The new tree **without** the megakernel (S3, LF1): all 19 sub-gates NEUTRAL vs the final; MK-latest is
  +30.5 / +21.8 / +21.4 / +10.1% faster than LF1 at C1/2/4/8 (stated in the source as LF1 −30.5 / −21.8 / −21.4 /
  −10.1% vs MK) — the gain is the megakernel's, not the merge's (`MK/raw/s3-latest-nomk/NOTES.md`).
- TP-consistency probe cost: S5P (probe every 10 steps) is 0.7% / 1.0% / +0.2% per step from plain MK (S6BL)
  at C1/C2/C4; serving probes every 100 steps (NEUTRAL, under ~1%).
- MK-syncsample: C1 +2.3%, C4 −0.5% vs MK-latest with the probe on (S5P); C2 −3.3% over two reads (mostly the
  accept draw); every sub-gate NEUTRAL; a consistent ~0.5–2% per decode step beyond the probe's.

### Speculative decoding accept length (tokens per verify step)

Unmatched content (the arm's own prompts): MKL1 3.31–3.41 per stream count vs the final's 3.24–3.42
(`MK/raw/s4-mk-b1/arm/raw/acc-MKL1.txt`); S7 3.28–3.38.

Matched content (`acc_drive.py`, same cells; final S2 4 reps, MKL1 4, MKL2 4, S6B 3), source
`MK/raw/s7-mk-b2/acc-compare.txt`, CI from `MK/raw/s7-mk-b2/acc-vs-final.txt`:

| cell | final | MKL1 | MKL2 | MKL1+MKL2 vs final (Welch t) | S6B | all three MK boots |
|---|---|---|---|---|---|---|
| c1 | 3.150 | 3.059 | 3.151 | 3.106, −1.4%, t −0.7, not significant | 3.008, −4.5% | −2.2%, t −1.2 |
| c4 identical | 3.163 | 3.090 | 3.075 | 3.083, **−2.5%, t −2.9** (95% CI −0.144 to −0.017) | 3.075, −2.8% | −2.6%, t −3.4 |
| c4 mixed | 2.911 | 2.912 | 2.912 | 2.912, +0.0%, t +0.1, equal | 2.885, −0.9% | −0.2%, t −0.6 |

MK A/A (MKL2 − MKL1): c1 +0.092, c4-identical −0.014, c4-mixed −0.000, all CIs across 0. The per-stream speed
gains above are measured **after** this small accept-length cost.

## Numerics (drift)

### Teacher-forced top-1 vs the final (`tfcmp.py`, 32 prompts; short = 0–15, long = 16–31)

| run | vs | short top-1 | short KL | long top-1 | long KL | raw |
|---|---|---|---|---|---|---|
| EP1old | FB | 94.353% | 4.00e-02 | 94.461% | 2.89e-02 | `MK/raw/s1-ep1-old/tfcmp-FB-vs-EP1old.txt` |
| EP1old | MK3a | 100.000% | 0 | 95.196% | 2.62e-02 | `MK/raw/s1-ep1-old/tfcmp-MK3a-vs-EP1old.txt` |
| EP1only | FB | 94.353% | 4.00e-02 | 95.025% | 2.91e-02 | `MK/raw/s1-ep1-old/tfcmp-FB-vs-EP1only.txt` |
| EP1only | MK3a | 100.000% | 0 | 95.098% | 2.44e-02 | `MK/raw/s1-ep1-old/tfcmp-MK3a-vs-EP1only.txt` |
| ENVdelta | FB | 100.000% | 0 | 94.706% | 2.42e-02 | `MK/raw/s1-ep1-old/tfcmp-FB-vs-ENVdelta.txt` |
| ENVdelta | MK3a | 94.353% | 3.95e-02 | 94.240% | 3.11e-02 | `MK/raw/s1-ep1-old/tfcmp-MK3a-vs-ENVdelta.txt` |
| MKL1 | FB | 94.298% | 4.03e-02 | — | — | `MK/raw/s4-mk-b1/tfcmp-FB-vs-MKL1.txt` |
| MKL2 | FB | 94.353% | 4.00e-02 | — | — | `MK/raw/s7-mk-b2/tfcmp-FB-vs-MKL2.txt` |
| MKL2 | MKL1 | 99.722% | 2.46e-03 | 95.882% | — | `MK/raw/s7-mk-b2/tfcmp-MKL1-vs-MKL2.txt` |
| LF1 (new tree, no MK) | FB, FB2 | 100.000% | 0 | 95.343% (FB2) | — | `MK/raw/s3-latest-nomk/NOTES.md` |

**Finding:** the short-prompt drift (≈5.65% of positions, 203 flips) is entirely `--ep-size 1`, which the
plugin requires, not the megakernel: a plain EP1 engine with no megakernel reproduces the same 203 flips with
the same wrong token at each. MKL1 has 205 short flips, 200 of them EP1old's; MKL2 has exactly the 203.
The long half (94.0–95.2% in every row, the final's two boots included) is the A/A floor and carries no
signal. Judged equivalent to box-c's accepted TP8 → TP4 move (94.83% short,
`box-c/qa/window2-tp4/README.md`): the gap is ~17 flips of ~3,600 positions against a Poisson spread of ~14.

### NVIDIA's KL test (`MK/harness/nvidia_kl.py`)

Method: T0, ignore_eos, one batched `/generate`, prefill re-score, k3 KL on the sampled token, mean of
per-prompt means. Short = 8 LongBench-v2 prompts × 256 tokens, bound 0.03; long = 4 windows of 18k–27k
tokens × 128 tokens, bound 0.06; 3 repeats each.

| boot | short avg_kl (mean of 3) | long avg_kl (mean of 3) | raw |
|---|---|---|---|
| final | 0.004824 PASS | 0.010901 PASS | `MK/raw/s1-ep1-old/kl-FINAL.json` |
| EP1old | 0.005289 PASS | 0.009847 PASS | `MK/raw/s1-ep1-old/kl-EP1old.json` |
| ENVdelta | 0.004811 PASS | 0.010696 PASS | `MK/raw/s1-ep1-old/kl-ENVdelta.json` |
| EP1only | 0.004297 PASS | 0.011841 PASS | `MK/raw/s1-ep1-old/kl-EP1only.json` |
| LF1 (new tree, no MK) | 0.00526 PASS | 0.01004 PASS | `MK/raw/s3-latest-nomk/kl-LF1.json` |
| MKL1 | 0.0042 PASS | 0.0087 PASS | `MK/raw/s4-mk-b1/kl-MKL1.json` |
| MKL2 (reps 0.0045/0.0059/0.0045; 0.0103/0.0102/0.0125) | 0.0050 PASS | 0.0110 PASS | `MK/raw/s7-mk-b2/kl-MKL2.json` |
| S6B (syncsample) | 0.0054 PASS | 0.0113 PASS | `MK/raw/s6b-syncsample/NOTES.md` |

Every boot in this table is under NVIDIA's bounds by ≥ 5.0× on its mean of 3 (closest: EP1only long 0.011841,
5.1×); single repeats come closer (EP1only long rep 0.015910, 3.8×; MKL2 long rep 0.012515, 4.8×). Across every
NVIDIA KL run on the record, including the config X' gates (`quality.md` §3), every boot is under the bounds and the
worst is MKP1's long 0.020538 (2.9× under; one run that overlapped its prewarm, so not idle); the worst idle long is
CANDB2's 0.016239 (3.7×) and the worst short CTL2's 0.007618 (3.9×). Over both MK boots the MK decode path's
per-repeat KL straddles the final's (short 0.0040–0.0059 vs 0.0047–0.0049; long 0.0073–0.0125 vs 0.0103–0.0117):
wider spread, both ends, no shift up.

### Greedy byte-equality (32 fixed prompts, T0, 256 tokens)

| run | vs final (FB) | vs MK | other |
|---|---|---|---|
| EP1old | 1/32 | MK3a 0, MK3b 0, MK2a 0, MK2b 1, MK2bR 0 | self re-run 32/32 |
| ENVdelta | 32/32 | MK3a 1 | EP1old 1/32 |
| EP1only | 1/32 | MK3a 0 | EP1old 32/32, ENVdelta 1/32 |
| LF1 (new tree, no MK) | 32/32 | — | — |
| MK-latest (3 clean runs, S4) | — | 32/32 to each other and to MK3a | — |

Under load (S4's load spec): final 3/32 at 8 streams and 4/32 at 16; MK 2/32 and 5/32 — both diverge almost
totally under batching, the MK no less stable (`MK/raw/s2-final-ref/greedycmp-FREF-load.txt`).

## KV cost

| build | KV pool (tokens) | source |
|---|---|---|
| final (EP4) | 66,653,184 | `MK/raw/s1-ep1-old/NOTES.md` |
| ENVdelta | 66,831,616 | same |
| LF1 (new tree, EP4, no MK) | 66,550,784 | `MK/raw/s3-latest-nomk/NOTES.md` |
| EP1old | 51,389,952 | `MK/raw/s1-ep1-old/NOTES.md` |
| EP1only | 51,211,520 | same |
| 10-28 MK (old plugin) | 49,956,352 | same |
| MK-latest (S4, S7) | 49,879,552 (**−25.2%** vs the final) | `MK/README.md` "Cost" |

The cost is the EP1 weight share plus ~1.4–1.5 M tokens inferred to be the plugin's own memory (not measured
directly). 49.9 M tokens of KV is still far beyond any single request (the served context is 1 M).

## Hang / rank-desync soaks (S5, S6b)

95% one-sided upper bound with 0 events in T loaded hours = 2.996 / T per hour.

| build | soaks | loaded h | requests | hangs / probe mismatches | 95% upper bound | raw |
|---|---|---|---|---|---|---|
| old tree 10-28 MK3 (old plugin) | 30-min off-pool soak + a live window | 1.08 (0.58 prod) | — | 0 / no probe | 2.77 per h | `box-c/qa/box-a-tuning-20261026/README.md` |
| MK-latest `9ec65dacaf`, probe every 10 | 2 × 1 h synthmix (seeds 51, 52) | **1.445** | **13,410** | **0 / 0** (852,300 checks per rank) | **2.07 per h** | `MK/raw/s5-soak/NOTES.md`, `soak-stats.txt` |
| MK-syncsample, probe every 10 | 1 × 1 h synthmix (seed 61) | 0.729 | 6,963 | 0 / 0 (405,980 checks per rank) | 4.11 per h | `MK/raw/s6b-syncsample/NOTES.md` §5 |

Over both builds: 0 mismatches in ~1.37 M probe checks per rank. S5's first soak alone made 170,992 checks.
The router's known-bad test (one TP rank `kill -STOP` for 9.9 s, 3 synthetic streams on the dev row): the
row went DOWN on every dev pod within ≤ 5.6 s and the in-flight streams paused ~10 s and finished normally
(`MK/raw/box-a-return/NOTES.md`).

## Quality vs the final

See `quality.md` §1 for the full two-sample table (gsm8k, JSON, agent sessions, mgsm, needles, DSML, images,
L3a sign-off 21/21).

## Caveats

- **Boots, not reps, are the unit.** Two plain MK boots (S4, S7; S7 is S6b's end-state boot, not a fresh
  reboot) against one final boot (FB, 10-28). The single-boot C1 figure moves between boots (885.8 vs 849.0
  here; 841–907 tok/s across later boots, see `speed.md`).
- **Gains shrink with concurrency.** The megakernel is a per-step latency win; at C16 it is neutral (+0.7–1.1%).
  At high concurrency the EP1 constraint and the −25% KV pool are the price.
- **The hang rate is bounded, not shown low.** 0 hangs in 1.445 loaded hours bounds the rate at ≤ 2.07/h (95%);
  excluding a 1-per-6 h rate needs ~18 loaded hang-free hours (the shipped config X' has since served 15.43
  loaded hang-free production hours on `4943d6ed90` and 8.76 on `9a9ede7ae0`: `stability.md`). All soaks ran
  T 1.0 / top_p 1.0 / no top_k; none exercised top_p < 1 (the path of upstream sglang #38565, which config X' carries).
- **Accept length** is 2.5% lower at c4-identical (≈3 SE over reps; the boot is the true unit).
- **The teacher-forced drift is real** (94.3% short top-1 vs EP4) and is the EP1 config's cost, judged
  equivalent to box-c's accepted TP8 → TP4 move; it does not show up as a quality loss (`quality.md`).
- **Images** (80/80) have no final twin; absolute score only.
