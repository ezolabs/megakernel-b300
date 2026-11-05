# Methodology and caveats, per measurement

One section per kind of measurement in this folder: what was run, how the number is computed, what it can and
cannot show. The record files each table cites carry the exact commands.

## Common setup

- **Hardware.** One box = 4× NVIDIA B300 SXM6 (148 SMs, sm_103, 275,040 MiB each), NV18 between every pair,
  rented. box-a: 2× Intel Xeon 6776P, driver 580.173.02, board "AC". box-b: AMD EPYC 9575F,
  driver 580.178.04, board "PC". Both 1,100 W limit, 2,032 MHz SM
  clock (`box-c/qa/box-b-bringup-20261030/NOTES.md`, `fleet-history/box-a/README.md`).
- **Model.** L3a = `repack-1017-l3a`, our repack of `DeepSeek-V4.1-Flash-1017` (≈ 510 GB on disk),
  served as `DeepSeek-V4.1-Flash`. Quality comparisons are against the same weights on other
  stacks unless a row names the stock model.
- **Engine.** our SGLang fork, TP4, FlashMLA attention, fp8_e4m3 KV cache, MXFP4 MoE,
  DSpark speculative decoding block 5, 1,048,576-token context, HiCache host KV offload, Engram tables in host
  memory. Config names: **final** (tree `819756d4`, EP4, no megakernel, 10-27 to 10-29), **config X** (tree
  `9ec65dacaf`, megakernel, 10-30 07:24–12:01Z), **config X'** (fork `prod` `4943d6ed90`, then `9a9ede7ae0` from
  10-31; megakernel; the serving config). Args/env files: `args/stack-b300-box-a-mk2.args`,
  `env-box-a-mk2.txt`; per-arm files in `box-c/qa/box-a-megakernel-20261029/args/`.
- **Load isolation.** Every bench ran with the box out of the router pool (or its row held) and an idle gate
  (`--max-live 0`: wait for 0 running requests). The live-traffic reads are the only loaded measurements.
- **No production data used.** Bench prompts are synthetic or public benchmark items; live reads use aggregate
  engine/router metrics only (no prompts or completions were copied).

## Decode tok/s per stream (think-ramp)

- Bench: `box-c/qa/speed-20261026/bench.py think-ramp` (the `bench178` wrapper:
  `data/raw/box-b-vs-box-a/raw__speed__bench178.sh`). Prompts: four synthetic reasoning templates (relative-motion
  puzzle, Pell equation, prime palindromes, truck loading) with random numbers; thinking on; T 1.0 / top_p 1.0;
  300 completion tokens (finish = length); C streams started together, 8 reps per C (32 at C1 in the settle run);
  seed 4242 + C.
- Metric: per request, `completion tokens / (time of last delta − time of first delta)` — TTFT excluded. A rep's
  value is the median over its C streams; a cell is the mean over reps. Aggregate = all streams' tokens / wall time.
- Caveats: short prompts (tens of tokens) and short outputs; DSpark acceptance on these prompts ≈ 3.3, live ≈ 4.0;
  one boot is one sample of a boot-level random effect (C1 boot means 841–907 across 14 boots of the same code).

## TTFT

- Client-side time from request send to first streamed delta, same think-ramp run, idle box, warm (no cold prefix).
- Caveat: dominated by host-side work at these prompt sizes; it moved between hosts (box-b's 26.8 ms vs box-a's 42.1 ms
  at C1 on identical code) more than between code versions.

## Cold fill / prefill

- Bench: `bench.py coldfill`: 4 background streams decoding, then one never-seen prompt of N tokens (60k × 8 reps,
  seed 777; 70k and 100k × 3); TTFT of the big prompt, prefill tok/s = N / TTFT, the background streams' decode dip
  (percent of their pre-fill rate kept during the fill) and the longest gap between their deltas.
- 1 M probe: one 979,971-token prompt with needles; checks the answer and the time (`probe-1m.log`).
- Caveat: one prompt at a time; batched-prefill throughput is not measured.

## The gate (`gates.py`) and its bands

- `gates.py` compares an arm's per-rep cells against a base of one or more boots. Each metric gets a band from
  the base's A/A spread (boot-to-boot and rep-to-rep); a move outside the band is GAIN or REGRESSION, inside
  NEUTRAL; a move outside the band on one boot but not confirmed is UNCONFIRMED. OVERALL PASS = no regression.
- Typical bands: decode ±4.6–8.3%, TTFT ±8–32%, cold fill ±6.7%.
- Caveat: with one boot per arm the band understates boot-level variance (`gates.py` exits 3 in that case and the
  record says so).

## Speculative acceptance length

- Unmatched: the engine's own `accept len` log over the arm's run. Matched: `acc_drive.py` replays identical
  content (c1, c4-identical, c4-mixed cells) on each build; Welch t over reps, 95% CI; the boot is the true unit.

## Greedy byte-equality

- 32 fixed prompts, T 0, 256 tokens, compared byte-for-byte against a reference boot; "n/32 equal".
- Caveat: DeepSeek-V4.1 decode is not bitwise stable across batch composition (SGLang cookbook,
  https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1, read 2026-10-31); under load every build
  diverges almost totally (3–5/32), so this is an idle-only test. Two boots of the same tree can give 27/32.

## Teacher-forced top-1 / KL (`tfcmp.py`)

- 32 prompts; the reference's generated sequence is fed to the candidate; per position, does the candidate's argmax
  match the reference's token, and the KL between their next-token distributions. Short = prompts 0–15, long = 16–31.
- Caveat: the long half reads 94–95.5% between any two builds including A/A — it is the noise floor and carries no
  signal; the short half is the signal (100% = identical numerics).

## NVIDIA's KL test (`box-c/qa/box-a-megakernel-20261029/harness/nvidia_kl.py`)

- NVIDIA's acceptance test for the megakernel: T 0, ignore_eos, one batched `/generate`, the output re-scored by a
  prefill pass, k3 KL estimator on the sampled token, mean of per-prompt means. Short: 8 LongBench-v2 prompts ×
  256 tokens, bound 0.03. Long: 4 windows of 18k–27k tokens × 128 tokens, bound 0.06. 3 repeats.

## DSML tool-call format

- 470 sends per case and mode (streamed and non-streamed) at T 1.0 / top_p 1.0, concurrency 16; a send is defective
  if the tool call fails to parse. `dsml_compare.py` compares defect rates with Fisher's exact test.

## Accuracy suites

- gsm8k (full 1,319 or subsets), mgsm, needles, JSON (object / schema / config), agent sessions, images, the L3a
  sign-off slice (gsm8k, math500, HumanEval+, GPQA-diamond, jbb_harmful, falsereject), MMLU-Pro, IFBench,
  MultiChallenge. Two-sample differences with Newcombe 95% intervals; paired items with McNemar's exact test;
  "non-inferior" means the interval's lower end is above −3 pp (the record's margin), "inconclusive (n)" means the
  interval is too wide to decide, "SAME" means the interval contains 0.
- Judges: refusal and stage-2 rows on a local a local judge judge; MultiChallenge on a model judge via the OpenAI API.

## Hang soaks and the hang-rate bound

- `synthmix`: mixed widths and prompt shapes for 1 h per seed, TP probe every 10 steps, hangwatch on.
- Bound: 0 events in T loaded hours → the 95% one-sided upper bound on a Poisson rate is −ln(0.05) / T ≈ 2.996 / T.
  Builds are not pooled. "Loaded" hours = hours with ≥ 1 running request (from 15 s samples; a floor).

## Live-traffic reads

- Engine: `sglang:*` histograms (ITL, TPOT) and running-request gauges over a fixed window; router: its own
  per-request decode/TTFT and its learned prefill-rate line per pod. Only aggregates; the two boxes received
  unequal load, so the comparison is confounded.

## Statistics conventions

- "±x%" on a gate = the gate's band; "[a, b]" = a 95% interval; "p" = two-sided unless stated.
- Percent changes are (new − old) / old.
- Every number here is copied from a dated record or computed by a script in `data/scripts/`; where the copy and
  the source disagree, the source wins.
