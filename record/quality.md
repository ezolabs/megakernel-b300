# Quality on 4× B300: does the fast stack still give the same answers?

Model throughout: L3a (`repack-1017-l3a`, our repack of `DeepSeek-V4.1-Flash-1017`), served as
`DeepSeek-V4.1-Flash`. All prompts are public benchmark items or synthetic; none came from production.
Machine-readable rows: `data/measurements.csv` (area `quality`). Statistics conventions: `methodology.md`.

## 1. Megakernel vs the pre-megakernel final, box-a (2026-10-28 to 10-30)

Two-sample, same patched suite and seeds, the final of record (tree `819756d4`, EP4, no megakernel) vs MK-latest
(tree `box-a-mk-latest`, megakernel, EP1). Source: `box-c/qa/box-a-megakernel-20261029/README.md` "Quality";
raw `box-c/qa/box-a-megakernel-20261029/raw/s2-final-ref/quality-FREF/` and `raw/s4-mk-b1/quality-MKL1/`;
comparison `raw/s2-final-ref/quality-compare-final-vs-mk.md` (`harness/quality/compare.py`: exit 0, GATE PASS,
9 pairs, nothing WORSE).

| suite | final | MK-latest | MK − final, pp [Newcombe 95%] | paired McNemar p | verdict |
|---|---|---|---|---|---|
| gsm8k full (1,319), T0 | 1279/1319 | 1282/1319 | +0.23 [−1.08, +1.54] | 0.549 | non-inferior |
| gsm8k full, prod sampling | 1288/1319 | 1280/1319 | −0.61 [−1.87, +0.64] | 0.039 (one draw, see below) | non-inferior |
| gsm8k under 16 streams, T0 | 485/500 | 485/500 | +0.00 [−2.23, +2.23] | 1 | non-inferior |
| gsm8k under 16 streams, prod | 483/500 | 485/500 | +0.40 [−1.88, +2.70] | 0.688 | non-inferior |
| JSON object | 99/100 | 100/100 | +1.00 [−2.79, +5.45] | 1 | non-inferior |
| JSON schema, strict-fitting | 80/80 | 80/80 | +0.00 [−4.58, +4.58] | — | inconclusive (n) |
| agent sessions, prod | 90/100 | 95/100 | +5.00 [−2.63, +12.96] | 0.125 | non-inferior |
| mgsm, T0 | 114/120 | 112/120 | −1.67 [−8.19, +4.70] | 0.5 | inconclusive (n) |
| needles, T0 | 80/80 | 79/80 | −1.25 [−6.75, +3.45] | 1 | inconclusive (n) |

- **gsm8k prod p 0.039 is one sampled draw.** A second final draw scored 1283/1319; S7 drew the MK twice more:
  **1286/1319 (97.50%) and 1283/1319 (97.27%)**. Every pairing of the final's two draws with S7's two MK draws sits
  at McNemar p ≥ 0.18 (the final's own draw-to-draw p is 0.180). Of the ten pairings among the five draws, the
  single p < 0.05 is S4's draw vs the final's r1 (`raw/s7-mk-b2/compare/`).
- **JSON config family:** with xgrammar's `uniqueItems` dropped from the schema (a harness fix; both trees refuse
  `uniqueItems`), MK scores config 20/20, schema 100/100, object 100/100 — the final's result item for item.
- **DSML tool-call format** (T1.0 / top_p 1.0, 470 sends per case and mode, `strict: false`, the arm prod uses):
  final 5/470 defective streamed, 1/470 non-streamed; MK 7/472 and 3/470; `dsml_compare.py` exit 0, nothing WORSE
  (+0.42 pp [−1.18, +2.08] stream, +0.43 pp [−0.64, +1.66] non-stream).
- **Images** (`images.py`, S7): exact **80/80** (95% CI 95.4–100%; count, shape, colour, position 20/20 each);
  no-image controls 3/20 (chance). No final twin; absolute score only.
- **L3a sign-off slice** (S7, concurrency 16, bands measured at 4): **PASS 21/21 rows**. gsm8k 0.967, math500 0.890,
  HumanEval+ 0.957, GPQA-diamond 0.899 against bars 0.957 / 0.827 / 0.909 / 0.865; truncation 0; jbb_harmful
  refusal 0.020 (≤ 0.075); falsereject refusal 0.013 (≤ 0.025); DSML 1,916 sends, 40/40 rows PASS
  (`raw/s7-mk-b2/l3a/`).
- The new tree without the megakernel (S3): gsm8k 98/100 and 293/300 T0; MK-latest 295/300 on the same ids.

## 2. L3a regression re-run on today's serving stack, box-b (2026-10-31)

Question asked: whether our serving settings had degraded the model since the repack. Source:
`box-c/qa/l3a-regression-20261031/NOTES.md`; raw rows, summaries and comparisons in
`box-c/qa/l3a-regression-20261031/raw/`. Target box-b, config X' (fork `prod` `4943d6ed90`, megakernel), in the live
pool the whole time at concurrency 4 under a load gate (142 load reads, box-b at 0–8 running requests).
T 1.0, top_p 1.0, effort high, max_tokens 131,072.

**Verdict: no detectable degradation.** A drop of ~7–15 points on the small rows would be missed (per-row lower
bounds below).

### 2a. Sign-off slice, two reps pooled vs the box-c (8× B200) reference's two reps (unpaired, Newcombe 95%)

Source `raw/compare-slice-pooled.md`. Pooling two seeds of the same items narrows intervals more than the data
support; single-rep intervals in the source are the honest ones.

| cell | n per rep | box-b s0 / s1 | box-b pooled | box-c ref pooled | Δ pp [95%] | read |
|---|---|---|---|---|---|---|
| math500_w4 accuracy | 100 | 0.890 / 0.910 | 0.900 (180/200) | 0.890 (178/200) | +1.0 [−5.1, +7.1] | SAME |
| gsm8k accuracy | 300 | 0.977 / 0.980 | 0.978 (587/600) | 0.975 (585/600) | +0.3 [−1.5, +2.2] | SAME |
| humaneval_plus accuracy | 163 | 0.951 / 0.951 | 0.951 (310/326) | 0.945 (308/326) | +0.6 [−2.9, +4.2] | SAME |
| jbb_harmful refusal | 100 | 0.030 / 0.020 | 0.025 (5/200) | 0.030 (6/200) | −0.5 [−4.2, +3.1] | SAME |
| jbb_harmful caveat | 100 | 0.190 / 0.150 | 0.170 (34/200) | 0.145 (29/200) | +2.5 [−4.7, +9.7] | SAME |
| falsereject_p300 refusal | 300 | 0.010 / 0.010 | 0.010 (6/600) | 0.010 (6/600) | +0.0 [−1.3, +1.3] | SAME |
| falsereject_p300 caveat | 300 | 0.010 / 0.000 | 0.005 (3/600) | 0.005 (3/600) | +0.0 [−1.0, +1.0] | SAME |
| gpqa_diamond accuracy | 198 | 0.924 / 0.904 | 0.914 (362/396) | 0.907 (359/396) | +0.8 [−3.3, +4.8] | SAME |

Paired by item against box-a's megakernel S7 run (McNemar): every cell SAME in both reps (largest: gpqa +2.5 pp,
8 gained / 3 lost, p 0.23). DSML self-parse PASS: 1,916/1,916 sends, defective raw 13/1,881 (0.69%) vs S7's
6/1,881 (Fisher p 0.11) and the gate's pooled healthy rate 0.73%. Mean reasoning tokens matched the reference
(gsm8k 287 vs 291/293; gpqa 6,315 vs 6,776/6,267) — no "thinks longer/shorter" drift.

### 2b. The weights campaign's rows the harness scores alone (unpaired vs the campaign's L3a figure)

Source `raw/compare-t1.md`. "Then" = the campaign's pushed point on a campaign box at c16; today = the served
repack on config X' at c4.

| row | box-b today | L3a then | STOCK then | Δ vs then pp [95%] | read |
|---|---|---|---|---|---|
| math500_w4@high | 0.8900 (89/100) | 0.9200 | 0.9100 | −3.00 [−11.57, +5.46] | SAME |
| gpqa_diamond@max | 0.9343 (185/198) | 0.9394 | 0.9394 | −0.51 [−5.55, +4.51] | SAME |
| mmlu_pro@max (12,028 scored) | 0.6296 (7573/12028) | 0.6321 | 0.6395 | −0.25 [−1.46, +0.97] | SAME |
| ifbench@high (prompt-level loose) | 0.7433 (223/300) | 0.7433 | 0.7233 | +0.00 [−6.97, +6.97] | SAME |

MMLU-Pro is one `/v1/completions` logprob pass per item: it tests prefill only, not the megakernel's decode or
the sampler.

### 2c. Refusal behaviour on the campaign's registered local judge, unpaired

Source `raw/compare-stage2.md`.

| row | box-b today | L3a then | Δ [95%] | read |
|---|---|---|---|---|
| stage-2 redirect, jbb_harmful@high | 16/100 = 16.0% [10.1, 24.4] | 20.0% | −4.0 pp [−14.7, +6.7] | SAME |
| refusal, falsereject_p300@max | 15/285 = 5.3% [3.2, 8.5] | 5.8% (STOCK 48.2%) | −0.5 pp [−4.4, +3.3] | SAME |
| refusal removal, jbb_harmful@high | 82.8% [73.1, 89.0] | 84.9% | then inside today's CI | SAME (consistency check) |
| refusal removal, jbb_harmful@max | 83.9% [74.3, 89.9] | 81.7% | then inside today's CI | SAME (consistency check) |

### 2d. MultiChallenge (a hosted model judge over the OpenAI API, the campaign's judge), unpaired

Source `raw/release-gate/compare.md`. 266/266 graded.

| read | box-b today | campaign L3a | Δ [95%] | read |
|---|---|---|---|---|
| macro | 0.6767 [0.6165, 0.7360] | 0.6619 [0.6024, 0.7209] | +1.48 pp [−6.95, +9.88] | SAME |
| micro | 185/266 = 0.6955 | 183/266 = 0.6880 | +0.8 pp [−7.1, +8.6] | SAME |

Every axis SAME; the gate bar against the campaign's STOCK still passes (+2.96 pp, unpaired [−5.53, +11.47]).
Arena-Hard v2 was generated (250/250 at @high and @max, 0 at the cap) but not judged: it is head-to-head
against STOCK answers, none of which survive, and the user chose not to pay for a STOCK run.

## 3. Per-change numerics gates (box-a and box-b, 2026-10-30 to 10-31)

Every engine change on the B300s passes the same battery before serving: greedy byte-equality (32 prompts, T0,
256 tokens), teacher-forced top-1 and KL (`tfcmp.py`, 32 prompts), NVIDIA's KL test (bounds 0.03 short / 0.06 long),
DSML at T1 (470 per mode), gsm8k 100 × 16 streams at T1, a 10-request smoke.

| event | box / tree | greedy | tf short top-1 / KL | tf long top-1 | NVIDIA KL short / long | other | source |
|---|---|---|---|---|---|---|---|
| config X' merge (#84), MKP1b | box-a `4943d6ed90` | 32/32 vs MKL2, MKL1, MK3a | 100% / 0 vs MKL2 | 95.343% | 0.004604 / 0.009151 (MKP1, the first boot, x1: 0.004563 / **0.020538**, the run overlapped its prewarm) | DSML 4/470 both modes, compare nothing WORSE | `box-c/qa/prod-merge-20261030/raw/gate/NOTES.md` |
| box-b bring-up | box-b `4943d6ed90` | 32/32 vs box-a MKP1b | 100% / 0 | 95.294% | 0.005950 / 0.010813 | gsm8k 98/100; smoke 10/10; 1 M-token probe PASS; DSML PASS | `box-c/qa/box-b-bringup-20261030/NOTES.md` |
| upstream takes, CTL2 (control) | box-a `4943d6ed90` | 27/32 vs MKP1, 32/32 vs MKP1b | 100% / 0 | 95.098% | 0.007618 / 0.011460 | smoke 10/10 | `box-c/qa/sglang-takes-20261031/NOTES.md` |
| takes CAND1 (with route_warp) | box-a | 3/32 / 1/32 | 94.937% / 2.98e-02 | 94.387% | 0.004332 / 0.008866 | DSML PASS, gsm8k 98/100 → route_warp DROPPED for moving numerics | same |
| takes CANDB (shipped, `9a9ede7ae0`) | box-a | 27/32 / 32/32 (both boots) | 100% / 0 | 95.098% | 0.004215 / 0.011155; 0.003361 / 0.016239 | DSML PASS, gsm8k 96/100 | same |

NVIDIA KL over every run on the record (this table, box-b's gate, and `megakernel.md`'s eight study boots; k3 mean of
per-prompt means, x1 per gate boot, mean of 3 in the study): every boot is under the bounds. Margins (bound /
value), worst first: long MKP1 0.020538 → 2.9× (one run, overlapping prewarm, so not idle), CANDB2 0.016239 → 3.7×
(idle), CTL2 0.011460 → 5.2×; short CTL2 0.007618 → 3.9×, box-b 0.005950 → 5.0×. Raw: `kl-*.json` under
`box-c/qa/prod-merge-20261030/raw/gate/box/`, `box-c/qa/sglang-takes-20261031/raw/gate/box/`,
`box-c/qa/box-b-bringup-20261030/raw/gate/` and `box-c/qa/box-a-megakernel-20261029/raw/`.

A/A reference: two boots of the same tree (MKP1 vs MKP1b) give 27/32 and 99.611% / 1.57e-03 — so 27/32 vs MKP1
is the A/A floor, not a change.

## Caveats

- Small n. Most rows have n = 100–300; a 3–7 pp move is the smallest they can see. MMLU-Pro (12,028) is the only
  sub-point row, and it does not exercise decode.
- The box-c reference and the campaign figures are unpaired (their per-item rows are not kept); box-a S7 is paired.
- Different concurrency: the reference at c4, S7 at c16, the campaign at c16/c32, today's runs at c4.
- Shared box: the box-b re-run shared the box with live traffic (≤ 4–5 traffic requests on top of ours).
- Judges: refusal rows on a local judge (the campaign's registered one); MultiChallenge on a hosted model judge.
