# Fork, merges and configuration science on 4× B300

What code and which flags the B300s run, how each piece got there, and every B300 / megakernel configuration
experiment in the speed register — adopted and rejected — with its numbers. Machine-readable rows for the headline
items: `data/measurements.csv`. Statistics: `methodology.md`.

## 1. The software stack of record (2026-10-31)

| layer | what | source |
|---|---|---|
| engine | our SGLang fork, branch `prod`: `4943d6ed90` (10-30 12:51Z → 10-31), then `9a9ede7ae0` (box-a from 10-31 17:42Z, box-b from 18:06Z) | `box-c/qa/prod-merge-20261030/README.md`, `box-c/qa/sglang-takes-20261031/ROLL-box-a.md`, `box-c/qa/box-b-roll-20261031/NOTES.md` |
| megakernel plugin | NVIDIA `sgl-nvidia-mega`, plugin main `54b4289` (megakernel API v2) + our B300 patch (`resident_blocks` 152 → 148, `Feature.KV_OFFLOAD`); not on the fork's `prod` — installed beside it on each box | `fleet-history/box-a/sgl-nvidia-mega-b300-spec.patch`, `b300-vs-gb300.md` §2 |
| args | `args/stack-b300-box-a-mk2.args` (both boxes) | the file; its role, rollback pair (`stack-b300-box-a-mk.args` + `env-box-a-mk.txt`, config X on `9ec65dacaf`) and the composition rule (tuning flags only) in `args/README.md` |
| env | `args/env-box-a-mk2.txt` (both boxes) | the file |
| model | `repack-1017-l3a` (L3a), served as `DeepSeek-V4.1-Flash` | `boxes/box-a.env`, `box-b.env` |

### Key flags (`stack-b300-box-a-mk2.args`, tuning flags only; model path and ports come from the box file)

| flag | value | why |
|---|---|---|
| `--tp-size` / `--ep-size` | 4 / **1** | the megakernel requires EP1 (KV −25.2%, `megakernel.md`) |
| `--enable-nvidia-megakernel` | on | the plugin |
| `--context-length` | 1048576 | 1 M-token context |
| `--mem-fraction-static` | 0.75 | SR-052 |
| `--speculative-algorithm DSPARK`, `--speculative-dspark-block-size 5`, `--speculative-use-block-verification` | on | MTP drafts; block 6 rejected (SR-084); block verify SR-054 |
| `--chunked-prefill-size` | 8192 | SR-020 |
| `--prefill-decode-interval` | 4 | SR-021 |
| `--swa-prefix-tails` | 2048 | SR-055 |
| `--max-running-requests` / `--cuda-graph-max-bs-decode` | 64 / 64 | 96 rejected (SR-131) |
| `--cuda-graph-backend-prefill breakable`, `--cuda-graph-max-bs-prefill 2048`, `--cuda-graph-prefill-max-context 131072` | on | SR-052 (breakable prefill graph) |
| `--flashinfer-allreduce-fusion-backend` | trtllm | SR-052 (fusion off rejected, SR-092) |
| `--enable-hierarchical-cache`, `--hicache-ratio 2`, `--hicache-write-policy write_through_selective`, `--hicache-io-backend direct`, `--hicache-mem-layout page_first_direct`, `--hicache-host-memory-mode cache` | on | host KV offload for long-context resume; needs the patch's `KV_OFFLOAD` |
| `--enable-decoder-swa-bounded-replay` | on (config X') | cold fill −31% with the exact decoder trim |
| `--short-prefill-*` (threshold 3072, chunk 1024, max 4096, lookahead 8, swap limit 16384) | on | short prompts jump ahead of long fills |
| `--enable-priority-scheduling`, `--disable-priority-preemption`, `--max-queued-requests 512` | on | router priority lanes |
| `--kv-events-config` (zmq on `127.0.0.1`), `--load-publish-endpoint` (`127.0.0.1`) | on | router cache and load awareness; loopback only |

Attention backend (FlashMLA), KV dtype and MoE kernels are not set in the args file: they are the engine's
B300 defaults with the plugin's requirements (TP4, EP1, fp8_e4m3 KV, FlashMLA, DSpark block 5; plugin README).

### Key env (`env-box-a-mk2.txt`)

| variable | value | why |
|---|---|---|
| `SGLANG_SPEC_TP_SYNC` | `rng,-dspark-accept-sample` | the plugin's sampler pin: keeps ranks' draws identical |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | 1, `per_rank` | Engram tables in host memory (SR-022); per-rank gets huge pages on a host whose shared-memory THP is off (SR-116) |
| `SGLANG_MK_TP_PROBE`, `_EVERY`, `_ASSERT` | 1, 100, 1 | TP-consistency probe (SR-182) |
| `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS` | 32 | boot 795 → 705 s (host-cache alloc 117 → 25 s) |
| `SGLANG_DSV4_EXACT_DECODER_TRIM`, `SGLANG_MM_CACHED_REFS` | 1, 1 | config X' bundle |
| `SGLANG_DEFAULT_THINKING`, `SGLANG_DSV41_REASONING_EFFORT` | 1, `high` | thinking on at high by default (served behaviour, not speed) |
| `SGLANG_SWA_BRANCH_REBUILD` | true | needed by the merged tree to boot |

## 2. The fork line: merges and picks, and what each bought

| when | tree | what | what it bought (measured) | source |
|---|---|---|---|---|
| 10-26 | `9d908424` → `fd3545a4` | first B300 boot; DSpark draft-graph glue fix (the draft graph failed to capture and ran eager) | C1 439 → 498 (+13.5%) | SR-023 |
| 10-27 | `819756d4` | the "final" non-megakernel stack (block verification, tails 2048, breakable prefill CUDA graph, TRT-LLM all-reduce fusion, mem 0.75) | C1 443 → 516 (+16%), TTFT C1 0.244 → 0.063 s, KV 55.4 M → 66.7 M; 5,008-request load proof, 0 errors; greedy 28/32 vs A0 (not output-identical) | SR-052, `box-c/qa/box-a-ship1-20261027/README.md` |
| 10-27/28 | `819756d4` → `5f117c89` | upstream #46700 SWA tombstone clamp (cross-request KV corruption fix) | correctness: freed 16 vs 8 slots before the fix; 96.5% cache hit on a co-serve hour after | SR-024 |
| 10-28 | `d03143934671` (`box-a-latest`) | upstream catch-up, not gated | host alloc ~57 s vs ~4 min; C1/C2/C4 638 / 552 / 440 vs 624 / 526 / 450 | SR-150 |
| 10-29 | `41bb61ec53`+fix → `9ec65dacaf` (`box-a-mk-latest`) | upstream `3a3a8611a4` catch-up (megakernel API v2) + TP-consistency probe (parts 0048/0048b) + tool-call refusal kept (0049) + `swa_branch_rebuild` import fix (0050) | the megakernel boots (the v1 tree silently ran without it); the new tree alone is NEUTRAL vs the final on all 19 sub-gates — the speed gain is the megakernel's | `box-c/qa/box-a-megakernel-20261029/README.md`, `megakernel.md` |
| 10-30 | `0ab213b25a` → `4943d6ed90` (`prod`) | `prod` merges `box-a-mk-latest`; picks #46645 (grammar masks stop tokens), #38565 (TP-deterministic top-p renorm), #47149 (DSV4 kernel grids) | config X': cold fill 60k 1.435 → 0.989 s (−31%), prefill 41–42k → 61k tok/s, READY 795 → 705 s; decode NEUTRAL; numerics identical (greedy 32/32, top-1 100% vs config X) | `box-c/qa/prod-merge-20261030/README.md`, `raw/gate/NOTES.md` |
| 10-31 | `9a9ede7ae0` (`prod`) | sweep b picks: #43110 (HiCache sidecars over every transfer), #47521 (abort retracted requests that recompute cannot rebuild), #46720 (CUDA-graph attention metadata kept alive), #47627 (inert on X') | correctness; speed NEUTRAL on every sub-gate (45 and 30 sub-gates); numerics byte-identical to control | SR-185, `box-c/qa/sglang-takes-20261031/NOTES.md` |
| 10-31 | — (dropped) | upstream #47641 `route_warp` one-warp MoE router | **rejected**: speed NEUTRAL but numerics moved (greedy 3/32 and 1/32, top-1 94.9%); its fast path ignores V4.1's `sqrtsoftplus_log1p` scoring | SR-184 |

## 3. The configuration science: every B300 / megakernel row in the speed register

Source: `box-c/SPEED-REGISTER.md` (rows quoted by id; evidence folders named in each row). "box-a #152 (x)" is the
10-26/27 tuning round's knob x (`box-c/qa/box-a-tuning-20261026/README.md`).

### Adopted

| id | change | measured | status |
|---|---|---|---|
| SR-019 | megakernel (plugin + B300 patch + HiCache `KV_OFFLOAD`) | see `megakernel.md`: C1 +33–39%, C8 +12–15%, C16 neutral; KV −25.2%; 0 hangs in 13,410 soak requests | live, config X' |
| SR-020 | `--chunked-prefill-size 8192` (was 4096) | box-a #152 (e) alone: cold TTFT −43% [−44.5, −42.3]; background streams during the fill +24–28% | live |
| SR-021 | `--prefill-decode-interval 4` | box-a #152 (h): background streams during a fill +42% | live |
| SR-022 | Engram host table | box-a #152 (c): KV 25.38 M → 55.36 M tokens (×2.18); C16 −2.3% [−3.9, −0.8] | live |
| SR-023 | DSpark draft-graph glue fix | C1 439 → 498 (+13.5%) | live |
| SR-024 | upstream #46700 SWA tombstone clamp | correctness (stale SWA slot could serve another request's KV) | live |
| SR-051 | scheduler threads pinned to the host's 16 turbo cores (of 128) | C1 step 7.06–7.15 → 5.40–5.49 ms; C1 514 → 614–639 (+20–24%) | retired with the pre-megakernel stack |
| SR-052 | box-a "stack" (block verify + tails 2048 + breakable prefill + all-reduce fusion, mem 0.75) | C1 443 → 516 (+16%), C2 418 → 466, C4 404 → 444, C8 335 → 352, C16 283 → 290; TTFT C1 0.244 → 0.063 s; KV 55.4 M → 66.7 M | superseded by SR-019 |
| SR-054 | DSpark block verification | box-a (b): C8 gain over 3 boots (box-c: C8 +3.6/+5.3%, accept +1–4%) | live |
| SR-055 | `--swa-prefix-tails 2048` | box-a #152 (g): SWA capacity +44% (36.55 M), speed unchanged | live |
| SR-101 | the gate (`gates.py`) rebuilt on box-a's A/A spreads (up to 9.51% at C1); K = 2.5 boot band | null passes 97.7% | live |
| SR-185 | upstream correctness picks (#43110, #47521, #46720, #47627) | speed NEUTRAL, numerics identical (§2) | live on `9a9ede7ae0` |

### Rejected or retired

| id | change | measured | why not |
|---|---|---|---|
| SR-083 | EP1 without the megakernel (#152 a) | KV −61%, C16 −9% | no gain to pay for it (EP1 returned only as the megakernel's requirement) |
| SR-084 | DSpark draft block 6 (#152 d) | C1 / C4 −7 / −8% (accept 3.32 → 3.42) | longer drafts cost more than the extra accepted tokens |
| SR-091 | megakernel without HiCache (MK1, 10-27) | C1 581 (+15%); TTFT 3–6× higher; resume +71%; aggregate −6 to −12.5% at C4–C16; greedy 1/32 | losing the cache made TTFT and aggregate worse |
| SR-092 | all-reduce fusion off | decode +2.6 / +2.0% at C1, inside the band; output-identical | no proven gain |
| SR-131 | 96 concurrent requests (#152 f) | no change; KV −2.6% | — |
| SR-149 | all-reduce multicast | push kernels 2–3× faster at decode; multicast wins only 1–2k-token prefill chunks by ~5 µs | not worth a fork step |
| SR-184 | upstream #47641 `route_warp` | speed NEUTRAL; greedy 3/32, top-1 94.9% vs A/A 99.6% | moves V4.1's numerics |
| SR-148 | FlashInfer SM103 MLA auto-select | — | retired (B300 only) |
| SR-146 | router's stale box-a `kv_capacity` (55.36 M vs 66.65 M) | — | moot: every router pod routes on the engine-reported capacity |

Block verification and tails 2048 were HOLD in the 10-26/27 tuning round and adopted in ship 1 (SR-052).
Never tried on B300: SR-116 (Engram `per_rank` vs `shared` with THP off; the served config uses `per_rank`), SR-124 (megakernel KV events
in the router's indexable design).

### Candidates for box-c that came out of the B300 work

SR-181 (hangwatch: a forced 45 s rank stop wrote HANG 12–13 s after the stop; 0 false stalls in 1.445 loaded h),
SR-182 (TP probe: under ~1% per decode step, 0 mismatches in 852,300 checks per rank), SR-183 (two-sample gates:
the megakernel's first DSML read, 10/470 vs 1/470, passed the old fixed bar but was WORSE two-sample,
+1.91 pp [+0.54, +3.67], Fisher p 0.011 — the harness fix that followed brought it to nothing WORSE, `quality.md` §1).

## Caveats

- Register rows are one-line summaries of their evidence; the evidence folder each names is the record.
- Many tuning-round knobs were measured on one or two boots against a band later found too narrow at C1
  (SR-101's A/A spread up to 9.51%); small C1 effects from that round are uncertain.
- SR-051's pinning applied to the pre-megakernel stack on box-a's Intel host; it was not re-measured with the
  megakernel or on box-b's AMD host.
- The flag tables summarise; `stack-b300-box-a-mk2.args` and `env-box-a-mk2.txt` are the authority.
