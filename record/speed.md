# Speed on 4× B300: decode, TTFT, prefill, cold fill, live traffic

All figures are DeepSeek-V4.1-Flash (L3a repack) on one 4× B300 box at TP4, with DSpark speculative decoding
(block 5), unless a row says otherwise. Bench definitions: `methodology.md`. Machine-readable rows:
`data/measurements.csv` (area `speed`); per-request raw rows: `data/raw/`.

**Read across tables with care.** Three different benches appear here (the `think` tool of 10-26/27, the
megakernel harness's speed gate of 10-28/29, and `bench178` from 10-30 on). Numbers from different benches are
not comparable, even at the same concurrency; each table names its bench.

## 1. The serving stack today (config X'), the distribution over every boot

Bench: `bench178` think-ramp (synthetic reasoning prompts, 300 completion tokens, thinking on, T 1.0 / top_p 1.0,
idle box, keep-alive). 14 boot-files on box-a and box-b, trees `4943d6ed90` and `9a9ede7ae0` (fork `prod`, megakernel on),
2026-10-30 to 10-31. 0 request errors. Computed by `data/scripts/decode_distribution.py` → `data/decode-distribution.csv`
and `data/decode-c1-by-tree.csv`.

**Pooling, stated.** The rows below pool requests across boots and both trees. At C1, 128 of the 208 requests come
from the four 32-rep boots of one run (the C1 settle, #405); the other 80 come from ten 8-rep boots. The per-boot
and per-tree view, which weights every boot once, is the table after this one; the headline uses it.

| streams | boots | reps | mean tok/s per stream | p10 | p50 | p90 | max | range of boot means | TTFT p50 / p90 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 14 | 208 | **868.8** | 784.6 | 865.2 | 954.0 | 1063.8 | 840.9–906.7 | 39.0 / 46.0 |
| 2 | 10 | 80 | 694.3 | 656.3 | 692.6 | 739.8 | 803.3 | 682.2–712.6 | 58.0 / 67.1 |
| 4 | 10 | 80 | 554.4 | 524.1 | 553.0 | 584.4 | 613.0 | 545.0–561.9 | 81.5 / 93.0 |
| 8 | 9 | 72 | 396.2 | 383.6 | 395.9 | 410.0 | 422.1 | 390.8–402.7 | 99.0 / 106.0 |

### C1 by tree and by run (each boot weighted once in "mean of boot means")

| tree | run | boots | requests | pooled mean | p50 | p90 | boot means | mean of boot means |
|---|---|---|---|---|---|---|---|---|
| `4943d6ed90` | bench178 (8 reps per boot) | 6 | 48 | 878.5 | 879.3 | 971.1 | 846.5–906.7 | 878.5 |
| `4943d6ed90` | C1 settle (32 reps per boot) | 2 | 64 | 866.7 | 862.1 | 942.9 | 855.4–878.1 | 866.7 |
| `9a9ede7ae0` | bench178 | 4 | 32 | 860.6 | 861.1 | 947.2 | 840.9–880.1 | 860.6 |
| `9a9ede7ae0` | C1 settle | 2 | 64 | 867.7 | 863.9 | 965.3 | 867.0–868.5 | 867.7 |
| `4943d6ed90` | all | 8 | 112 | 871.8 | 866.5 | 953.4 | 846.5–906.7 | **875.6** |
| `9a9ede7ae0` | all | 6 | 96 | 865.4 | 863.9 | 953.1 | 840.9–880.1 | **863.0** |
| both | all | 14 | 208 | 868.8 | 865.2 | 954.0 | 840.9–906.7 | **870.2** |

The `9a9ede7ae0` bench178 boots include CANDB1/CANDB2, the candidate patch series on `4943d6ed90` that became
`9a9ede7ae0` (which adds only a CI-only commit). The 12.6 tok/s gap between the trees' boot means is inside the C1
settle's interval (pooled +0.12% [−2.43, +2.66]) and the boot-level spread; it is not a measured difference.

Aggregate (all streams) tok/s at 8 streams: 2,804–2,900 per boot (`box-c/qa/sglang-takes-20261031/NOTES.md`).
At C2..C8 a "rep" is the median over that rep's streams, so p10/p90 there are spreads of rep medians, not of
single requests.

### Per-boot table (bench178), source `box-c/qa/sglang-takes-20261031/NOTES.md`, `box-c/qa/box-b-bringup-20261030/SPEED.md`, `box-c/qa/prod-merge-20261030/raw/gate/NOTES.md`

| boot | box | tree | decode C1 / C2 / C4 / C8 | agg C8 | TTFT C1 / C8 ms | cold fill 60k s | longest gap ms |
|---|---|---|---|---|---|---|---|
| MKP1 (#84 gate, 10-30) | box-a | `4943d6ed90` | 904.4 / 689.8 / 561.9 / — | — | 42.9 / — | 0.989 | 142 |
| box-a #178 (10-30) | box-a | `4943d6ed90` | 846.5 / 706.2 / 545.0 / 395.5 | 2,837 | 42.1 / 98.6 | 1.043 | 152 |
| box-b #178 (10-30) | box-b | `4943d6ed90` | 860.1 / 687.1 / 554.6 / 393.6 | 2,871 | **26.8** / 73.5 | **0.869** | 86 |
| CTL1 (10-31) | box-a | `4943d6ed90` | **907** / 694 / 559 / 395 | 2,849 | 40.0 / 99.4 | 1.018 | 139 |
| CAND1 (all five takes) | box-a | candidate | 900 / 700 / 564 / 392 | 2,804 | 45.0 / 100.2 | 1.004 | 140 |
| CANDB1 | box-a | `9a9ede7ae0` | 841 / 713 / 551 / 394 | 2,805 | 52.9 / 95.5 | 0.990 | 140 |
| CANDB2 | box-a | `9a9ede7ae0` | 848 / 683 / 556 / 391 | 2,851 | 41.6 / 93.4 | 1.007 | 148 |
| CTL2 | box-a | `4943d6ed90` | 895 / 682 / 552 / 399 | 2,810 | 41.8 / 103.3 | 1.021 | 145 |
| C3 | box-a | `9a9ede7ae0` | 873 / 704 / 550 / 400 | 2,880 | 38.8 / 109.4 | 0.998 | 147 |
| K3 | box-a | `4943d6ed90` | 858 / 692 / 556 / 403 | **2,900** | 40.6 / 105.4 | 1.020 | 143 |
| C4 (serving boot) | box-a | `9a9ede7ae0` | 880 / 693 / 559 / 395 | 2,866 | 41.0 / 100.8 | 1.007 | 147 |

Gates: CANDB1/CANDB2/CAND1 vs CTL1+CTL2+#178 — OVERALL PASS, 0 regressions in 45 sub-gates; C3+C4 vs
CTL1+CTL2+K3 — OVERALL PASS, 0 regressions, 0 unconfirmed in 30 sub-gates (`data/raw/sglang-takes/`).

### C1 settle: is the newest tree slower at one stream? (#405, box-a, 2026-10-31)

32 reps at C1 (seed 4243) on four interleaved boots A1 B1 A2 B2; A = `9a9ede7ae0`, B = `4943d6ed90`.
Source `box-c/qa/sglang-takes-20261031/c1-settle/analysis.txt`.

| boot | mean (sd) | median | min–max | TTFT ms | accept length |
|---|---|---|---|---|---|
| A1 | 867.0 (72.3) | 864.1 | 737.8–1,063.8 | 39.3 | 3.340 |
| B1 | 878.1 (55.1) | 866.7 | 791.2–1,015.9 | 38.9 | 3.387 |
| A2 | 868.5 (66.6) | 863.5 | 773.8–1,001.4 | 43.7 | 3.368 |
| B2 | 855.4 (56.2) | 850.8 | 771.3–949.5 | 40.9 | 3.310 |

Pairs: −1.26% [−4.95, +2.44] and +1.53% [−2.07, +5.13]; pooled +0.12% [−2.43, +2.66]. **No cost confirmed.**
The boot as the unit has a ±17.6% interval: two boots per tree cannot see a few-percent move.
GPU clocks held 2,032 / 3,996 MHz (SM / memory), no throttle.

## 2. box-b vs box-a: two boxes of the same SKU (#178, 2026-10-30 22:01–22:14Z)

Same tree (`4943d6ed90`), same args, same bench, one boot per box. Source `box-c/qa/box-b-bringup-20261030/SPEED.md`;
raw `data/raw/box-b-vs-box-a/`.

| metric | box-b | box-a | box-b vs box-a | band | verdict |
|---|---|---|---|---|---|
| decode C1 / C2 / C4 / C8 | 860.1 / 687.1 / 554.6 / 393.6 | 846.5 / 706.2 / 545.0 / 395.5 | −1.8 to +0.2% | ±4.6–7.4% | same |
| aggregate C2 / C4 / C8 | 1,316 / 2,048 / 2,871 | 1,318 / 2,014 / 2,837 | +1.2 to +1.9% | ±4.7–5.7% | same |
| TTFT C1 / C2 / C4 / C8 (ms) | 26.8 / 38.9 / 52.4 / 73.5 | 42.1 / 79.2 / 81.6 / 98.6 | −25.5 to −38.1% | ±7.9–31.8% | **box-b faster** |
| cold fill, 60k-token prompt | 0.869 s (69,494 tok/s) | 1.043 s (57,919 tok/s) | −14.5% | ±6.7% | **box-b faster** |
| decode dip during fill | 69.0% | 66.1% | +3.7% | ±6.9% | same |
| longest stream gap during fill | 86 ms | 152 ms | −41.6% | ±8.3% | **box-b better** |
| MTP accept C1 / C2 / C4 / C8 | 3.47 / 3.27 / 3.34 / 3.29 | 3.32 / 3.34 / 3.35 / 3.32 | — | — | — |

Decode is a dead heat; box-b wins every host-side (prefill/TTFT) metric. The two boxes differ in host CPU (box-b
AMD EPYC 9575F; box-a Intel Xeon 6776P), GPU board variant (B300 SXM6 PC vs AC) and driver (580.178.04 vs
580.173.02), so the TTFT/prefill gap is attributed to the host but not isolated. box-a showed non-zero
power-violation in 6 of 1,436 `dmon` samples, box-b none.

### Live traffic, both boxes serving (21:26:28–22:01:00Z, 34.5 min)

Source `box-c/qa/box-b-bringup-20261030/SPEED.md` "Live", raw `data/raw/box-b-vs-box-a/raw__speed__live__*.txt`.
Real production traffic; only aggregate metrics were read, no request content. Load-confounded (box-a carried more).

| metric | box-a | box-b |
|---|---|---|
| running requests mean / max | 0.84 / 5 | 0.25 / 3 |
| generated tok/s; requests | 584; 1,127 | 231; 522 |
| ITL p50 (→ tok/s) | 1.151 ms (869) | 1.087 ms (920) |
| ITL p90 (→ tok/s) | 2.678 ms (373) | 1.956 ms (511) |
| TPOT p50 / p90 | 1.093 / 1.967 ms | 1.011 / 1.820 ms |
| router decode tok/s p50 / p90 | 855 / 1,728 | 1,055 / 1,860 |
| router TTFT p50 / p90 | 439 / 680 ms | 306 / 1,119 ms |
| learned prefill rate (1–8 jobs, per router pod) | 70.6k / 73.9k / 74.7k tok/s | 84.6k / 84.9k / 88.7k tok/s |
| MTP accept mean | 4.00 | 4.05 |

Live accept length (≈4.0) is higher than the synthetic bench's (≈3.3): real traffic is more predictable for the
drafter. Router decode p90 above 1,700 tok/s per stream is the router's own per-request rate over the visible
stream (it includes the speculative bursts), not a sustained per-stream figure.

## 3. Cold fill and long context (config X')

Source `box-c/qa/prod-merge-20261030/raw/gate/arm/coldfill-summary.txt` (box-a, MKP1, 10-30), `data/raw/prod-merge-x-prime/`.

| prompt | n | TTFT | prefill tok/s | decode dip of 4 background streams | longest gap |
|---|---|---|---|---|---|
| 60k tokens | 8 | 0.982 s | 61,114 | 66.9% | 141 ms |
| 70,051 tokens | 3 | 1.12 s | 62,565 | 62.6% | — |
| 100,208 tokens | 3 | 1.599 s | 62,666 | 64.1% | 95 ms |
| 979,971 tokens (1 M probe, box-b bring-up) | 1 | 15.4 s | ~63,600 (derived) | — | — |

Config X → X' cut the 60k cold fill from 1.435 s to 0.989 s (−31.0%, a GAIN in the gate) and raised prefill from
41.3k–42.2k to 61.1k tok/s; the bundle adds decoder-side SWA bounded replay, selective HiCache write-through,
parallel host-cache allocation (boot READY 795 → 705 s; host-cache alloc 117 → 25 s) and exact decoder trim.
The 1 M-token probe is `box-c/qa/box-b-bringup-20261030/raw/gate/probe-1m.log` (copy `data/raw/box-b-gate/`).

Earlier long-context points on box-a (pre-megakernel, other trees): 1 M cold prompt 979,980 tokens in 66.87 s at
first boot (10-26); 979,973 tokens in 31.36 s with 4/4 needles found on the 10-27 final; host-cache resume of a
400k / 900k conversation in 0.97 / 1.84 s against cold fills of 9.48 / 25.58 s (MK3a, 10-28)
(`box-c/qa/box-a-bringup-20261026/README.md`, `box-c/qa/box-a-ship1-20261027/README.md`,
`box-c/qa/box-a-tuning-20261026/README.md`).

## 4. How box-a got here: the speed history (2026-10-26 → 10-31)

Each step is the record's own bench; steps on different benches are not comparable as numbers — read the
step's own "before → after".

| date | step | bench | before → after | source |
|---|---|---|---|---|
| 10-26 | first boot (TP4/EP4, FlashMLA, fp8 KV, MXFP4 MoE, DSpark block 5, 1 M context) | think-ramp (`speed-20261026/bench.py`) | C1/2/4/8/16 per stream 439 / 416 / 352 / 299 / 274; agg 3,222 at C16; cold 60k 3.90 s | `box-c/qa/box-a-bringup-20261026/README.md` |
| 10-26/27 | tuning round: chunked prefill 8192, prefill-decode interleave 4, Engram tables in host memory | same | cold 60k TTFT 3.55 → 2.20 s (+38% [+26, +51]); KV pool 25.4 M → 55.4 M tokens (×2.18); background decode during a cold fill 44 → 66 tok/s | `box-c/qa/box-a-tuning-20261026/README.md` |
| 10-27 | ship 1 "final": block-verify, tails 2048, breakable prefill CUDA graph, TRT-LLM all-reduce fusion | `think` tool | C1 443 → 516 tok/s; TTFT −66 to −74% at C1–C16; agg 3,990 at C16; KV 66.7 M | `box-c/qa/box-a-ship1-20261027/README.md` |
| 10-28/29 | fast-core pinning (scheduler threads on the 16 turbo-capable host cores of 128) | think-ramp | C1 step 7.06–7.15 → 5.40–5.49 ms; C1 514 → 614–639 (+20–24%) | `box-c/SPEED-REGISTER.md` SR-051 |
| 10-28/29 | megakernel, old plugin (MK2/MK3) | MK speed gate | C1 632 → 866–882; C2 546 → 675–703; C4 448 → 550–561; C8 348 → 391–402; C16 296 → 296–310 | `fleet-history/box-a/README.md` |
| 10-29/30 | megakernel, plugin `54b4289` + B300 patch (config X) | MK speed gate | C1 638.4 → 885.8 / 849.0 (+38.8 / +33.0%) | `megakernel.md` |
| 10-30 | config X' (prod merge #84) | bench178 | C1 904.4 (neutral); cold 60k 1.435 → 0.989 s (−31%) | `box-c/qa/prod-merge-20261030/raw/gate/NOTES.md` |
| 10-31 | upstream takes (`9a9ede7ae0`) | bench178 | neutral on every sub-gate | `box-c/qa/sglang-takes-20261031/NOTES.md` |

### Cost per output token (10-28/29 measurement, box-a)

From the MK2/MK3 speed gate's aggregate tok/s (`fleet-history/box-a/README.md`), at full
utilisation of one box at the stated concurrency (an idealisation: real traffic does not fill every slot).
Cost per output token is the reciprocal of aggregate tok/s at a fixed hourly cost, so the table gives
each width's change as a percentage of the final stack's cost:

| streams | aggregate tok/s, final → MK | cost per output token, final → MK |
|---|---|---|
| 1 | 632 → 874 | 100% → 72% (−28%) |
| 2 | 1,008 → 1,294 | 100% → 78% (−22%) |
| 4 | 1,624 → 1,994 | 100% → 81% (−18.5%) |
| 8 | 2,508 → 2,813 | 100% → 89% (−11%) |
| 16 | 4,104 → 4,227 | 100% → 97% (−3%) |

Both boxes were rented; the record keeps no hourly price.

## Caveats

- **The boot is the unit of variance.** The single-boot mean at C1 ranges 841–907 tok/s across 14 config X'
  boots of the same code: any one-boot comparison at C1 has a ±7–18% band. The gates use bands built from A/A boots.
- **Synthetic prompts, short inputs, 300 output tokens.** The think-ramp prompts are short reasoning questions;
  long-input decode and long-output decode are not covered by this bench. Cold-fill tables cover long inputs.
- **Idle box.** Every bench row was taken on a box out of the pool or idle (`--max-live 0` gate); the live table
  is the only loaded read and is confounded by unequal load.
- **Speculative decoding inflates per-stream tok/s** by the accepted draft length (≈3.3 synthetic, ≈4.0 live);
  this is real user-visible speed, but compare only against other speculative figures.
- **Cost figures** assume a fully used box at fixed concurrency; they are a ceiling on efficiency, not a bill.
