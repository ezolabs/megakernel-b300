# 4× B300 performance record (box-a + box-b)

The complete measured record of serving DeepSeek-V4.1-Flash (our L3a repack, served as
`DeepSeek-V4.1-Flash`) on 4× NVIDIA B300 boxes, 2026-10-26 → 2026-10-31: speed, stability and quality,
with the raw data, compiled for a research write-up. Start with `HEADLINE.md`.

## Index

| section | what it covers | its source files (repo-relative) |
|---|---|---|
| `HEADLINE.md` | the best numbers on one page, each tied to its method, source and a `data/measurements.csv` row | the sections below |
| `b300-vs-gb300.md` | **lead story**: NVIDIA wrote the megakernel for GB300 (152 SMs); we run it on B300 (148 SMs), which the shipped plugin refused; our patch, safety layer, upstream's later `sm148`, a labelled side-by-side with NVIDIA's published figures | `fleet-history/box-a/sgl-nvidia-mega-b300-spec.patch`, `fleet-history/box-a/RETRY-MEGAKERNEL.md`, `box-c/qa/box-a-megakernel-20261029/`, `box-c/qa/sglang-sweep-20261031/NOTES.md`, `box-c/SGLANG-UPDATES.md`; public URLs (read 2026-10-31) |
| `speed.md` | decode, TTFT, cold fill, 1 M-token prefill; distribution over 14 boots; box-b vs box-a; live traffic; speed history; $/token | `box-c/qa/box-b-bringup-20261030/SPEED.md`, `box-c/qa/sglang-takes-20261031/`, `box-c/qa/prod-merge-20261030/raw/gate/`, `box-c/qa/box-a-bringup-20261026/`, `box-c/qa/box-a-tuning-20261026/`, `box-c/qa/box-a-ship1-20261027/`, `box-c/08-measurements.md` §22, §25, §26 |
| `megakernel.md` | the megakernel study: speed by concurrency, acceptance, numerics (teacher-forced, NVIDIA KL, greedy), KV cost, soaks | `box-c/qa/box-a-megakernel-20261029/README.md` and its `raw/` |
| `stability.md` | hang soaks, TP-rank probe, production serving hours, restart and roll timings | `fleet-history/box-a/RETRY-MEGAKERNEL.md`, `box-c/qa/box-a-megakernel-20261029/raw/s5-soak/`, `box-c/qa/box-a-outage-20261031/NOTES.md`, `box-c/qa/box-b-bringup-20261030/NOTES.md` |
| `quality.md` | megakernel vs the non-megakernel stack; today's stack vs the 8× B200 reference and the weights campaign; per-change numerics gates | `box-c/qa/box-a-megakernel-20261029/raw/s2-final-ref/`, `raw/s7-mk-b2/`, `box-c/qa/l3a-regression-20261031/`, `box-c/qa/prod-merge-20261030/raw/gate/`, `box-c/qa/sglang-takes-20261031/NOTES.md` |
| `fork-and-config.md` | the software stack and its flags; the fork's merges and picks and what each bought; every B300 / megakernel row of the speed register, adopted and rejected | `args/stack-b300-box-a-mk2.args`, `args/env-box-a-mk2.txt`, `args/README.md`, `box-c/SPEED-REGISTER.md`, `box-c/qa/prod-merge-20261030/README.md`, `box-c/qa/sglang-takes-20261031/` |
| `fleet-ops.md` | the two-box fleet: autopool stop/start and rejoin times, smoke cycles, the load split and the sibling decode switch, red/blue rolls | `box-c/qa/autopool-20261030/` (DESIGN, DRILL, SMOKE), `box-c/qa/box-a-maintenance-20261031/` (drain-first stop, the provider maintenance notice; no rejoin timing in it), `box-c/qa/box-b-bringup-20261030/raw/router/lb-audit/AUDIT.md`, `box-c/qa/sglang-takes-20261031/ROLL-box-a.md`, `box-c/qa/box-b-roll-20261031/NOTES.md`; the serving router's `docs/research/sibling-switch-decode-20261031.md`, `sibling-ownership-balance-20261030.md`, `autopool-golive-20261030.md` |
| `methodology.md` | how each kind of number was measured, and what it cannot show | the benches and harnesses named in it |
| `data/` | `measurements.csv`, the decode distribution and its script, raw copies, the sanitising scan (`data/README.md`) | copies from the folders above |

## The software stack (2026-10-31)

| layer | what |
|---|---|
| engine | our SGLang fork branch `prod`: `4943d6ed90` from 10-30 12:51Z, then `9a9ede7ae0` (box-a from 10-31 17:42Z, box-b from 18:06Z) |
| megakernel | NVIDIA `sgl-nvidia-mega` plugin main `54b4289` (megakernel API v2) + our B300 patch: `resident_blocks` 152 → 148, `Feature.KV_OFFLOAD` (`fleet-history/box-a/sgl-nvidia-mega-b300-spec.patch`) |
| config of record | config X': `args/stack-b300-box-a-mk2.args` + `args/env-box-a-mk2.txt`, both boxes |
| key flags | TP4 / EP1, `--enable-nvidia-megakernel`, DSpark block 5 with block verification, 1,048,576-token context, chunked prefill 8192, prefill-decode interval 4, SWA prefix tails 2048, max running 64 (decode graphs to 64), breakable prefill CUDA graph, TRT-LLM all-reduce fusion, HiCache (ratio 2, selective write-through, direct I/O), decoder-side SWA bounded replay, KV events and load feed on 127.0.0.1 |
| key env | `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample`, Engram host table `per_rank`, TP probe every 100 steps with assert, host-cache parallel alloc 32, exact decoder trim, thinking on at `high` by default |
| safety | TP-consistency probe, `hangwatch.py --service --on-hang kill`, `watchdog.sh`, router liveness |

Full flag and env tables, with the reason for each: `fork-and-config.md` §1.

## The boxes

| | box-a | box-b |
|---|---|---|
| where | rented | rented |
| GPUs | 4× B300 SXM6 AC, 148 SMs, 275,040 MiB, NV18 | 4× B300 SXM6 PC, same |
| host | 2× Intel Xeon 6776P (256 threads), ~3 TiB RAM | AMD EPYC 9575F |
| driver | 580.173.02 | 580.178.04 |
| role | test box; serving since 10-29 (megakernel since 10-30) | serving since 10-30 17:56Z |

Sources: `fleet-history/box-a/README.md`, `fleet-history/box-b/README.md`,
`box-c/qa/box-b-bringup-20261030/NOTES.md`, `box-c/qa/autopool-20261030/DESIGN.md`.

## Timeline

| date (UTC) | event | record |
|---|---|---|
| 10-26 | box-a first boot on B300 (TP4/EP4, 1 M context) and tuning round | `box-c/qa/box-a-bringup-20261026/`, `box-c/qa/box-a-tuning-20261026/` |
| 10-27 | ship 1: box-a's tuned non-megakernel "final" stack | `box-c/qa/box-a-ship1-20261027/` |
| 10-28/29 | first megakernel trials on the old plugin with our B300 patch (MK1–MK3) | `box-c/qa/box-a-tuning-20261026/` "Megakernel", `fleet-history/box-a/README.md` |
| 10-29 → 10-30 | megakernel retry on plugin `54b4289`: sessions S1–S7, soaks, quality | `box-c/qa/box-a-megakernel-20261029/` |
| 10-30 07:24Z | box-a serves production on the megakernel (config X) | same |
| 10-30 12:51Z | config X' (fork `prod` merge) serves | `box-c/qa/prod-merge-20261030/` |
| 10-30 16:44 → 17:56Z | box-b rented, gated and serving in 72 min | `box-c/qa/box-b-bringup-20261030/` |
| 10-30/31 | stop → start smoke cycles under autopool | `box-c/qa/autopool-20261030/SMOKE.md` |
| 10-31 | upstream takes (`9a9ede7ae0`), C1 settle, box-b roll, L3a regression re-run on box-b | `box-c/qa/sglang-takes-20261031/`, `box-c/qa/box-b-roll-20261031/`, `box-c/qa/l3a-regression-20261031/` |

## Found and left out

A search of `box-c/`, `fleet-history/`, `qa/` and `research/` for `B300|box-a|box-b|megakernel` (markdown files,
2026-10-31) turned up these records with B300-related matches that the sections above do not use, one line each:

| record | what it holds | why left out |
|---|---|---|
| `box-c/archive/box-g.md`, `box-c/archive/box-g-deepseek-v41-benchmark-2026-09-11.md`, `box-c/archive/box-g-LAUNCH-SPEC.md` | box-g, an earlier 4× B300 box (09-11): DeepSeek-V4.1-Flash on SGLang `dev-dsv41`, greedy C1 502–616 tok/s buffered across 4 boots (`box-g.md` lines 111–112) | a different box, a September engine build and stack, a different bench (buffered greedy); the brief covers box-a + box-b. A historical baseline if the article wants one |
| `box-c/archive/8xB300-box-f.md`, `8xB300-box-f-LAUNCH-SPEC.md`, `box-f.md`, `box-f-FORK.md` | box-f, an 8× B300 box | different hardware (8 GPUs) and an older stack |
| `box-c/qa/decode-gap-1326-20261029/` | live per-request decode through the router, 10-29: box-a p50 722 [597–908] tok/s, n 583, beside box-c copy a (SR-177) | pre-megakernel stack, mixed with box-c; superseded by the config X' live read in `speed.md` §2 |
| `box-c/qa/cache-eval-20261026/` | router cache-hit reads with box-a in the pool (e.g. box-a 86/95, 10-27) | a router cache-affinity study, not box performance; pre-megakernel |
| `box-c/qa/agent-speed-20261026/` | one-page summary of the 10-26/27 box-c + box-a agent-speed push | a summary of records already used directly (`box-a-tuning`, `box-a-ship1`) |
| `box-c/qa/tree-roll-20261026/` | box-a tree roll; same tree measured C1–C8 +10–15% over `9d908424` | covered by SR-023 (`fork-and-config.md` §2); pre-megakernel |
| `box-c/qa/fork-20261026/`, `box-c/qa/upstream-sweep-20261028.md`, `box-c/qa/sglang-sweep-20261031b/` | fork merges and upstream sweeps naming box-a/box-b | composition records with no new B300 measurement; sweep b's verdicts are covered through `sglang-takes-20261031`; `20261031b` is a HOLD/SKIP list with no runs |
| `box-c/qa/window1-20261027/`, `box-c/qa/window2-tp4/`, `box-c/qa/r1-ladder-20261024/` | box-c (8× B200) tuning windows that mention box-a in the pool or as a probe | box-c's measurements; only the LMSYS/brief citation from window 1 is used (`b300-vs-gb300.md`) |
| `box-c/qa/ix-20261026/` | an audit note that the 4× B300 C1 figures in box-c docs came from box-g | a provenance check, no new data |
| `box-c/qa/speed-register-evidence-20261029/` (SR-xxx folders) | evidence behind speed-register rows | used through the register rows in `fork-and-config.md` §3; the SR-004 overnight report's cost figures and `DESIGN.md`'s agree |
| `box-c/qa/box-c-shutdown-20261030/`, `box-c/qa/box-c-rebuild-check-20261030/`, `box-c/qa/no-reroute-20261031/`, `box-c/qa/router-releases-20261027.md`, `box-c/qa/router-releases-20261029/`, `box-c/qa/final-bands-20261030/` | box-c's shutdown and rebuild check, the no-reroute order, router releases, box-c live bands — each mentions box-a/box-b in passing | operations of box-c or the router; no B300 measurement (the no-reroute order is cited for the fleet setup) |
| `box-c/01-hardware.md` … `box-c/10-updating.md`, `box-c/README.md`, `CHANGELOG.md`, `NOTES.md`, `ISSUES.md`, `TODO.md`, `REBUILD.md`, `README.md`, `caddy-h2/README.md` | the deployment's living docs | state and runbooks; the dated records they point to are cited instead (`box-c/08-measurements.md` §22, §25, §26 match the records used) |
| `fleet-history/box-c/`, `fleet-history/README.md` | box-c's box sheet | the B200 fallback box |
| `fleet-history/glm-5.3-flash/` | GLM-5.3 boxes; B300 availability on other clouds | another model and fleet |
| `qa/16/` | rent-vs-buy arithmetic for 8× B300 | a cost projection, not a measurement |
| `research/box-c-hardware/`, `research/rtx-6000-pro-build/`, `research/decision-models/` | hardware purchase research; third-party 4× GB300 latency quotes; a training-hardware note | not our measurements; the GB300 figures this record uses come from NVIDIA's own article and PR |
| `box-c/qa/dp-attn-1209-20261030/` (raw only) | box-c DP-attention work whose stage log lists box-a env files | box-c only |

### Not included: kept out of the tree at the source

- `box-c/qa/box-a-maintenance-20261031/`: the raw the provider instance JSON (it carries tokens and the onstart script) stays
  out of the tree by the record's own note; only its derived state lines are committed.
- the serving router's `docs/research/sibling-switch-decode-20261031.md`: the synthetic fan-out's text, its
  throwaway key and the per-run raw reads were the release agent's scratch and are not in that repo; only the counts
  and timings quoted in `fleet-ops.md` are.
- No figure in this folder comes from a scratch-only or uncommitted source. Every number traces to a committed file
  in this repo or in the serving router (whose four cited research files have no uncommitted changes).

## Citation rules used here

- Every figure names the dated record it came from (paths relative to the repo root); the record wins over this copy.
- NVIDIA and upstream figures are quoted from public URLs, read 2026-10-31, and never compared with ours without
  stating which conditions differ.
- Nothing in this folder contains production prompts, completions, credentials or internal addresses.
