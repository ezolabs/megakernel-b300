# Headline: DeepSeek-V4.1-Flash on 4× B300, our best numbers

One box = 4× NVIDIA B300 (148 SMs each, rented, x86 host). Model: our L3a repack of DeepSeek-V4.1-Flash, TP4,
DSpark speculative decoding, 1 M-token context. Serving config X' (SGLang fork + NVIDIA's megakernel with our B300
patch), boxes box-a and box-b, 2026-10-26 → 11-01 (serving hours read 11-01 04:41Z). Every row links to the file
with its method and raw data; the machine-readable copy is `data/measurements.csv` (row ids in the last column).

## Lead: NVIDIA's megakernel was built for GB300 — we run it on a B300 it did not support

| | NVIDIA, 4× GB300 (published) | us, 4× B300 (patched; unsupported as shipped) | read |
|---|---|---|---|
| hardware | Grace-Blackwell GB300, 152 SMs/GPU (the plugin's hard floor) | B300 HGX, **148 SMs/GPU**, x86 host — refused by the shipped plugin | `b300-vs-gb300.md` §2 |
| megakernel gain, 1–8 concurrent | +12–38% per-user decode (8K/1K, simulated acceptance 3.51) | **+11.7–38.8%** per-stream decode (C1–C8, real acceptance) | same shape; conditions differ |
| single-request decode | 928.4 tok/s P90 (AgentX replay); 945.96 tok/s p90 (PR screen) | **870 tok/s per-boot mean** (14 boots, 841–907); p90 954 / max 1,064 over 208 pooled requests (128 of them from one 4-boot run) | workloads differ: not a ranking |
| quality evidence | none published ("not an accuracy result") | full battery, nothing worse than the non-megakernel stack; L3a sign-off 21/21 | ours only |
| stability evidence | none published | 0 hangs, 0 rank desyncs: 13,410 soak + 7,408 production requests (10-30). Shipped config X', 0 hangs in its production serving: **≤ 0.19 hangs per loaded h** (95%) on `4943d6ed90` (15.43 loaded h, box-a + box-b, 10-30 → 10-31) and ≤ 0.34 on `9a9ede7ae0`, serving now (8.76 loaded h). Config X (the soak build): ≤ 1.43 (soak + serving, 2.095 h), ≤ 2.07 on the soak alone | ours only |
| B300 support upstream | added after us: opt-in `sm148` (PR #35, 10-30 20:12Z), speed measured at one request | our patch from 10-28; measured C1–C16, max-running 64 | `b300-vs-gb300.md` §2 |

Full table with every condition labelled: `b300-vs-gb300.md` §3.

## Speed

| what | best figure | conditions | source / row |
|---|---|---|---|
| decode per stream, 1 stream | **870.2 tok/s, the mean of 14 boot means** (each boot weighted once; boot means 841–907). Per tree: `4943d6ed90` 875.6 (8 boots), `9a9ede7ae0` 863.0 (6 boots, two of them the candidate patch series with the same runtime code; `speed.md`); the 12.6 tok/s gap is inside the boot-to-boot spread. Pooled over all 208 requests: p50 865.2, p90 954.0, max 1,063.8 — 128 of the 208 come from the four 32-rep boots of one run (C1 settle), the other 80 from ten 8-rep boots | synthetic reasoning prompts, 300 tokens, idle box | `speed.md` §1 · S01–S03, S07 |
| decode per stream, 2 / 4 / 8 streams | 694.3 / 554.4 / 396.2 tok/s | same | `speed.md` §1 · S04–S06 |
| aggregate decode, 8 streams | 2,900 tok/s (2,804–2,900 over boots) | same | `speed.md` §1 · S08 |
| megakernel speed-up vs our tuned non-megakernel stack | **+38.8% / +29.6% / +23.7% / +11.7% / +1.1%** at 1 / 2 / 4 / 8 / 16 streams | same tree ± megakernel family, 2 boots vs 1; 0 regressions in 38 sub-gates | `megakernel.md` · M01–M07 |
| TTFT, 1 stream, warm | **26.8 ms** (box-b); 39 ms p50 over all boots | short prompt, idle | `speed.md` §2 · S09 |
| cold prefill | **69,494 tok/s** (60k-token prompt in 0.869 s, box-b); 62,666 tok/s at 100k | one cold prompt, 4 streams decoding meanwhile | `speed.md` §2–3 · S10, S11, S14 |
| 1 M-token prompt | **979,971 tokens in 15.4 s** | cold, needles found | `speed.md` §3 · S15 |
| live production decode (ITL p50) | 869 tok/s (box-a), 920 tok/s (box-b); live acceptance ≈ 4.0 tokens/step | real traffic, 34.5 min, both boxes serving | `speed.md` §2 · S16, S17 |
| first boot → today, 1 stream | 439 → 870 tok/s (×1.98; today = mean of 14 boot means), different benches along the way | box-a, 10-26 → 10-31 | `speed.md` §4 |
| cost at 1 stream | **−28% per output token** with the megakernel | box-a, full utilisation | `speed.md` §4 |

## Stability

| what | figure | source / row |
|---|---|---|
| hang soak, megakernel | 0 hangs, 0 TP-rank mismatches in 13,410 requests / 1.445 loaded h (852,300 probe checks per rank) | `stability.md` · T01, T02 |
| production on the megakernel | 7,408 requests on 10-30 with 0 hangs; one engine process served 13 h 05 min without a restart | `stability.md` · T03–T05 |
| hang-rate bound, shipped config X' | **≤ 0.194 per loaded hour** (95%) = 2.996 / 15.43 h with 0 hangs on `4943d6ed90` (production pool, box-a 6.25 h + box-b 9.18 h loaded, 10-30 12:51Z → 10-31 17:49Z); ≤ 0.342 = 2.996 / 8.76 h on `9a9ede7ae0` (box-a 3.29 h + box-b 5.47 h, 10-31 17:42Z → 11-01 04:41Z). Trees are not pooled; counted as one config, 24.19 h would give ≤ 0.124 | `stability.md` "The hang-rate bound, derived" · T10–T12 |
| hang-rate bound, config X (the soak build) | ≤ 1.43 per loaded hour (95%) = 2.996 / 2.095 h with 0 hangs, the 2.095 h being config X's S5 soaks (1.445 h) + its 10-30 production serving (0.650 h loaded); S5 alone: ≤ 2.07 per loaded hour (2.996 / 1.445 h) | `stability.md` "The hang-rate bound, derived" · T01, T06 |
| fresh rented box → serving | 72 min (box-b) | `stability.md` · T08 |
| stop → start → back in rotation | 14 min 8 s – 14 min 22 s, no hand step, 0 pages (3 cycles) | `stability.md` · T07 |

## Fleet operations (two boxes, one router pool)

| what | figure | source |
|---|---|---|
| console stop → out of every router pod | 11–14 s, no hand step (4 runs) | `fleet-ops.md` §2 |
| red/blue roll of a new tree, model online | box-a out 62 min (3 boots + benches), then box-b out 16 min 38 s; the other box carried all traffic | `fleet-ops.md` §4 |
| load split after the sibling decode switch | synthetic 12-fork fan-outs split 4–6 / 6–8 instead of 12 / 0; ~490 tok/s per stream at 5–7 streams on each box | `fleet-ops.md` §3 |

## Quality

| what | figure | source / row |
|---|---|---|
| megakernel vs non-megakernel stack | nothing worse in 9 suite pairs (gsm8k full 1282 vs 1279 of 1,319; agent sessions 95 vs 90 of 100; JSON 100/100) | `quality.md` §1 · Q01, Q02 |
| L3a sign-off on the megakernel | 21/21 rows PASS; images 80/80 | `quality.md` §1 · Q03, Q04 |
| today's serving stack vs the 8× B200 reference (box-c) | every cell SAME: GPQA-diamond 0.914 vs 0.907, gsm8k 0.978 vs 0.975, math500 0.900 vs 0.890, HumanEval+ 0.951 vs 0.945 | `quality.md` §2 · Q05–Q08 |
| vs the weights campaign's own figures | MMLU-Pro 0.630 vs 0.632 (12,028 items); MultiChallenge 0.677 vs 0.662; false refusals 5.3% vs 5.8% (stock 48.2%) | `quality.md` §2 · Q09–Q11 |
| NVIDIA's own KL acceptance test | every boot under NVIDIA's bounds (0.03 short / 0.06 long): megakernel boots 0.0034–0.0076 short / 0.0087–0.0205 long. Worst **0.0205 long (2.9× under)**, MKP1's single run overlapping its prewarm; worst idle long 0.0162 (3.7×, CANDB2); worst short 0.0076 (3.9×, CTL2) | `quality.md` §3, `megakernel.md` · M11, Q14–Q17 |

## The caveats that matter most

1. **The boot is the unit.** Single-boot C1 means range 841–907 tok/s on identical code; trust distributions and
   gated comparisons, not one boot.
2. **Bench shape.** Speed rows use short synthetic prompts and 300-token replies; long-context rows are separate.
   NVIDIA's figures use 8K-input or agentic traces and a simulated acceptance length.
3. **Costs of the megakernel:** KV pool −25.2% (still 49.9 M tokens), acceptance −2.5% at one cell, teacher-forced
   top-1 94.3% vs the EP4 stack — traced to the required EP1 layout, not the kernel, and invisible in every accuracy suite.
4. **Hang rate is bounded, not shown to be zero** (shipped config X': ≤ 0.19 per loaded h over 15.43 h on `4943d6ed90`, ≤ 0.34 over 8.76 h on `9a9ede7ae0`; config X: ≤ 1.43 over 2.095 h). In the pool, running requests peaked at 13–14 (some of it may be our own synthetic fan-outs through the router); 87% (`4943d6ed90`) and 97% (`9a9ede7ae0`) of the loaded hours were at 1–4 running.
5. **The B300 patch is unofficial**; upstream's `sm148` opt-in arrived later and is not yet adopted.
6. **Quality sets are small.** Most eval rows (100–300 items) would miss a drop of about 7–15 points; MMLU-Pro
   (12,028 items) is the tight one but scores prefill only, not generation (`quality.md`).
