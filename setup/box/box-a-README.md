# box-a: retirement record

> **box-a serves beside box-b and is the standing test box** (the standing order of 2026-10-31: box-a stays
> online for testing changes and upstream merges). Tests and upstream merges run here out of rotation; a
> validated build serves on box-a first, then box-b is rebooted onto it; never both out; a stop → start smoke
> after each session, and it stays online. Before that it was stopped 2026-10-31 09:45:40Z to 13:55Z by the user's choice, to save money
> (record: [`../../box-c/qa/box-a-maintenance-20261031/`](../../box-c/qa/box-a-maintenance-20261031/README.md)),
> then started for sweep b's tests ([`../../box-c/qa/sglang-takes-20261031/`](../../box-c/qa/sglang-takes-20261031/NOTES.md)),
> and rolled onto fork `prod` `9a9ede7ae0` on 10-31
> ([`ROLL-box-a.md`](../../box-c/qa/sglang-takes-20261031/ROLL-box-a.md)). How it is run: `../../README.md`
> "Standing orders" and the update-sglang workflow's steps 5 and 7.
> Contract end 2026-11-27T01:20:44Z.

> **box-a is NOT retired now.** It was started again for the megakernel retry and, on that job's verdict
> (SAFE as a monitored canary beside box-c), has served config X since **2026-10-30 07:24Z** (#39): rows
> `box-a` in `box-c-prod` (prod config v254) and `box-a-dev` in `box-c-dev` (dev v224), on
> `tools/args/stack-b300-box-a-mk.args` + `env-box-a-mk.txt` (the NVIDIA megakernel, our SGLang fork
> `box-a-mk-latest` @ `9ec65dacaf`), with the TP probe and `hangwatch.py` on. The return, every command and
> its rollback: [`../../box-c/qa/box-a-megakernel-20261029/raw/box-a-return/NOTES.md`](../../box-c/qa/box-a-megakernel-20261029/raw/box-a-return/NOTES.md);
> how it is run: `box-c/07-operations.md` "box-a: the megakernel canary in the pools". What follows is the
> 10-26 → 10-29 service and its retirement, kept as written.

box-a was the second serving box for L3a (`DeepSeek-V4.1-Flash`) beside box-c: one TP4 SGLang
engine on 4× B300, a second path in the router's `box-c-prod` pool. It ran from 2026-10-26 to 2026-10-29.
On 2026-10-29 the box was retired for the time being and one retry was named: a megakernel run in about a
week, if upstream fixed the megakernel's numeric drift. That retry is
[`RETRY-MEGAKERNEL.md`](RETRY-MEGAKERNEL.md). The user set the stop order: archive, out of the pools,
the fault drill on box-c, then `hostctl stop` (see "Stopping box-a" below), and all four are done: the
instance was stopped at 2026-10-29 05:27:26Z. It is stopped, **never destroyed**: a stopped provider
instance keeps its disk, so the model, venvs and trees on it survive for the retry, until its
contract ends (2026-11-27T01:20:44Z).

This file is the whole record in one place. The dated detail lives in the run records it links. All times
are UTC.

## Identity

| what | value |
| --- | --- |
| the provider | instance **<INSTANCE_ID>** (offer <OFFER_ID>, machine <MACHINE_ID>, host <HOST_ID>), label `box-a-dsv41-b300`, created 2026-10-26 |
| GPUs | 4× NVIDIA B300 SXM6 AC (sm_103, cc 10.3), 275,040 MiB (~269 GB) each, 148 SMs, NV18 between every pair; driver 580.173.02, CUDA 13.0 |
| host | 2× Xeon 6776P (256 threads, 2 NUMA nodes, every GPU on NUMA0), 3023 GiB RAM, 1,500 GB disk, everything under `/`; Ubuntu 24.04.4, kernel 7.0.0-29-generic, container hostname `<CONTAINER_HOSTNAME>` |
| ssh | `root@<BOX_IP> -p <SSH_PORT>` (the direct mapping of container port 22). The provider's listed `ssh_port` never answered |
| front door | `https://box-a.<DOMAIN>:<APP_PORT_EXT>` (DNS record deleted 10-29 03:06:51Z; container 8001 → external <APP_PORT_EXT>), Caddy with a Let's Encrypt cert behind the router allowlist; engine on loopback `:8002` |
| box file | [`../../boxes/box-a.env`](../../boxes/box-a.env) (since 10-30 07:24Z: config X; the final's values are its rollback comments) |
| router rows | prod `box-a` in pool `box-c-prod` (sealed KV-event tunnel `box-a-kv`), dev `box-a-dev` in `box-c-dev`; direct router row priority 5 (below box-c's 10). Out of both pools 10-29 (v214) → 10-30 07:22-07:24Z, back in both since (dev v224, prod v254; kv 49,879,552) |
| state | **serving** (router pools) since 2026-10-30 07:24Z; was stopped 2026-10-29 05:27:26Z (`actual_status` `exited` at 05:29:09Z) until started for the megakernel retry; contract end 2026-11-27T01:20:44Z |
| standing order | never `hostctl start`, `hostctl stop` or destroy <INSTANCE_ID> without the user's word |

## Dates

| date | what |
| --- | --- |
| 2026-10-26 | rented; the L3a repack copied box to box from box-c (510 GB, 56 files, sha256-verified), tree and venv built, first boot, glue-graph fix ([bring-up record](../../box-c/qa/box-a-bringup-20261026/README.md)) |
| 10-26 → 10-27 | overnight tuning round on the idle box ([tuning record](../../box-c/qa/box-a-tuning-20261026/README.md)) |
| 10-27, early | joined the router: `box-a` in `box-c-prod` (config v162 → v166), `box-a-dev` in `box-c-dev` (v131 → v135) |
| 10-27 → 10-28 | ship 1: the stacked args become box-a's final ([ship-1 record](../../box-c/qa/box-a-ship1-20261027/README.md)) |
| 10-28 12:50:14Z | the final moves onto fork `prod` `819756d4` (upstream #46700's SWA tombstone clamp) |
| 10-28 | n-gram copy and graph-accept arms (#694 and others), CPU turbo-core pins (#696) |
| 10-28 19:39Z → 10-29 01:07Z | megakernel MK2 and MK3 (#621), then the final restored |
| 10-29 02:05–02:35Z | `box-a-latest` window (#869); stopped part-way by the retirement and rolled back to the final |
| 10-29 ~02:30–02:50Z | archive taken (#978) |
| 10-29 | **retired** by the user |
| 10-29 03:00:31Z / 03:02:08Z | out of the pools: `box-a-dev` left `box-c-dev` (dev config v214), then `box-a` left `box-c-prod` (prod v214); DNS record deleted 03:06:51Z |
| 10-29 05:17–05:25Z | the fault drill on box-c (#866), after router release 14: 15 s held, 0 client errors |
| 10-29 05:27:26Z | `hostctl stop instance <INSTANCE_ID>`; `actual_status` `exited` at 05:29:09Z |
| 10-29 → 10-30 | started again for the megakernel retry ([record](../../box-c/qa/box-a-megakernel-20261029/README.md)); verdict SAFE as a monitored canary |
| 10-30 07:22Z / 07:24Z | back in the pools on config X: `box-a-dev` into `box-c-dev` (dev v224), then `box-a` into `box-c-prod` (prod v254) (#39, [return record](../../box-c/qa/box-a-megakernel-20261029/raw/box-a-return/NOTES.md)) |

## Role

box-a was a second path for `box-c-prod`. Cache affinity, not speed, decided where traffic landed, and the
share it took swung widely:

- **10-27 co-serve hour:** box-c-a 1,063 requests, box-a 270, box-c-b 54.
- **10-28 06:00–10:24Z:** box-c-a 3,497, box-a 3,394, box-c-b 1,442.
- **10-28, the hour after the `#46700` boot (13:03:28–14:03:28Z):** box-a took 220 requests, all 200,
  with a 96.5% cache hit; box-c-a took 310. A fresh box-a is priced lower on small prefills until its cache
  warms (39.6k vs ~49k tok/s).
- **10-29 00:38Z, attempts per minute:** box-a 2.8, box-c-a 12.0, box-c-b 12.2.

It was never in the router's speed group (`box-c-a` + `box-c-b`). It covered box-c's engine restarts, so a box-c
roll could cut and roll one copy while box-a and the other copy served. The case
for keeping it was speed per GPU-hour, hangs and a fault drill. The user decided to retire it before
the drill ran.

## Final config (box-a's last served state)

| piece | value |
| --- | --- |
| tree | our SGLang fork `prod` = `box-a-ship1-46700` = **`819756d4632e9a1f07234e97c11137a591a79193`**: `box-a-ship1` `ace628a0` (prod 452f2438 + box-e-img-order 35e6d667 + box-e-swa-interior-evict 9e295882, left off) + upstream #46700's SWA tombstone clamp. Checked out at `/workspace/box-e-up`, editable in `/opt/sglang-box-e` |
| args | [`stack-b300-box-a.args`](../../args/stack-b300-box-a.args): TP4 / EP4, mem-fraction 0.75, DSpark block 5 with block verification, decode graphs to bs 64, max-running 64, context 1,048,576, chunked prefill 8192, prefill-decode interval 4, `--swa-prefix-tails 2048`, priority scheduling, the short-prefill lane, KV events (zmq 5557/5558) and load publish, HiCache (`ratio 2`, `write_through`, `kernel`, `page_first`, host-memory `cache`), the breakable prefill CUDA graph (bs 2048, context 131,072), the trtllm all-reduce fusion |
| env | [`env-box-a-w1.txt`](../../args/env-box-a-w1.txt): `env-box-e.txt` + `SGLANG_DSV41_ENGRAM_HOST_TABLE=1` with `LAYOUT=shared` (tuning knob c) + `TRITON_PRINT_AUTOTUNING=1` |
| CPU | `SCHED_CPU_PIN="16 17 32 33"`: TP0–3 schedulers pinned to node-0 turbo cores (4,600 MHz busy) |
| front door | stock Caddy v2.11.7, sha256 `22e29521…5aba` (#869's h2-window swap was rolled back, see below) |
| model | `<MODEL_PATH>` (476 GB, L3a; `config.json` sha256 `8be45ce0…1879`) |
| memory | weights 73.70 GB per GPU, draft 2.02 GB, KV 123.32 GB → a KV pool of **66,653,184 tokens** |
| measured | decode per stream C1/C2/C4/C8/C16 **632 / 546 / 448 / 348 / 296** tok/s (the #621 gate's base); gsm8k 98/100 on three boots; L3a sign-off PASS (live, 10-28 01:25–02:08Z: gsm8k .977, MATH .880, HumanEval+ .951, GPQA .919, harmful refused .030, DSML 36/36); true-cold 1M probe 979,973 tokens in 31.36 s, 4/4 needles |
| ways back | `ace628a0` on the same args and env; then `v1-b300-box-a.args` + `env-box-a.txt` on `prod-20261026` `fd3545a46b53` (the tuning round's config) |

The final does not carry fork #901 (the `todo_write` strict default). The DSML gate's `T1_todo_nested`
case fails on it: 361/362 stream and 362/363 non-stream calls defective. Prod is covered by the router's
`strict: false` (release 12, live since 2026-10-29 00:31:42Z). #869 was going to give box-a a tree with
#901 and was rolled back when box-a retired.

## Every arm

A verdict is the gate's, against the base each run names. "A/A band" is the paired noise band of two boots
of the same config (`box-c/qa/window2-tp4/harness/gates.py`).

### Bring-up (2026-10-26; [record](../../box-c/qa/box-a-bringup-20261026/README.md))

| arm | numbers | verdict |
| --- | --- | --- |
| first boot, `9d908424` + `v1-b300-box-a.args` + `env-box-a.txt` | READY 27 min 24 s (cold FlashInfer autotune ~10 min); KV pool 25,377,024; per stream C1/C2/C4/C8/C16 439 / 416 / 352 / 299 / 274, aggregate C16 3,222 tok/s; gsm8k 98; 1M probe PASS; 1,800 s soak 0 hangs; **DSML 4/36 FAIL** (the box-e parse bug) | not served |
| `fd3545a4` (fork fix: DSpark draft glue graph + DSML parse) | glue graph captured on every rank; C1 439 → 498 (+13.5%), C2 → 468, C4 → 418, C8 → 333, C16 274 → 273; DSML 36/36; L3a slice seed 0 PASS (gsm8k .973, math500 .900, HE+ .951, gpqa .889) | **adopted** (the bring-up tree) |

Against box-c's 8×B200 baseline of that day, box-a won at C8/C16 throughput and warm resume. It lost at
C1–C4 decode, cold prefill and the cold-fill crater.

### Tuning round (#152, 10-26 → 10-27; [record](../../box-c/qa/box-a-tuning-20261026/README.md))

| knob | numbers | verdict |
| --- | --- | --- |
| (a) EP1 | KV pool −61% (9.93M), C16 −9% | rejected |
| (b) block verification | C8 gain on 3 boots, mixed elsewhere | held; adopted later in the stack |
| (c) Engram host table, shared layout | KV pool 25.38M → 55.36M (×2.18), C16 −2.3% [−3.9, −0.8], greedy 32/32, gsm8k 98, L3a PASS | **adopted** |
| (d) DSpark block 6 | C1 −7%, C4 −8% (accept 3.32 → 3.42, slower steps) | rejected |
| (e) chunked prefill 8192 | cold TTFT −43%, crater +27% | **adopted** |
| (f) 96 concurrent | no change at C ≤ 16 (every CI crosses 0), KV −2.6% | rejected |
| (g) `--swa-prefix-tails 2048` | speed and resume unchanged, KV +44% (36.55M) | held; adopted later in the stack |
| (h) prefill-decode interval 4 | crater +42% / +46% | **adopted** |
| FIN (c + e + h) vs A0 | cold 60k TTFT 3.55 → 2.20 s, crater 44 → 66 tok/s, KV 25,377,024 → 55,363,328 | **served** from 10-27 ~05:00Z |

### Ship 1: the stack (#385, 10-27 → 10-28; [record](../../box-c/qa/box-a-ship1-20261027/README.md))

Per stream decode tok/s / TTFT s (aggregate at C8/C16):

| | A0 | A0b | STK1 | STK2 | FINAL (`ace628a0`) | MK1 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 443 / .244 | 478 / .220 | 503 / .061 | 513 / .060 | 516 / .063 | 581 / .206 |
| C2 | 418 / .296 | 418 / .330 | 502 / .098 | 476 / .089 | 466 / .092 | 490 / .268 |
| C4 | 404 / .465 | 383 / .350 | 434 / .118 | 446 / .126 | 444 / .132 | 495 / .420 |
| C8 agg | 1,995 | 1,960 | 2,409 | 2,375 | 2,472 | 2,272 |
| C16 agg | 3,400 | 3,410 | 3,960 | 4,000 | 3,990 | 3,576 |
| warm 700-token extend on ~70k | .457 | .373 | .243 | .235 | .241 | |
| gsm8k | 98 | 99 | 98 | 98 | 98 | 98 |

- **The stack** is `v1-b300-box-a` + block verification + tails 2048 + the breakable prefill graph + the
  trtllm fusion, with `env-box-a-w1.txt`. The KV pool went from 55.4M to 66.7M tokens. The user ruled at
  22:35Z: "Ship the stack as box-a's final tonight, not A0". Under the corrected rule (gates.py
  `20466654`) it passes with 0 regressions. Resume is about 40% faster. Greedy is 28/32 against A0. In
  the load proof (00:40–01:01Z) it served 5,008 requests with 0 errors and a max gap of 1.54 s.
  Verdict: **adopted, the final**.
- **#46700 tree move** (10-28): `ace628a0`'s tombstone probe frees 16 pages where 8 are expected, and
  `819756d4` frees 8. READY 12:50:14Z, prewarm 824 requests with 0 errors, smoke 10/10, back in the
  pool 13:03:28Z. Verdict: **adopted**.
- **CPU pinning** (#385/#636): pinning every thread of each rank to one core gave a C1 step of
  6.00–6.36 ms and 508–538 tok/s, against 6.51–6.96 ms and 405–505 tok/s unpinned. It won all three
  alternations. Verdict: **adopted**.
- **Turbo cores** (#696): only 16 of the 128 physical cores reach 4,600 MHz busy. Cores 16, 17, 32 and
  33 are node-0 turbo cores. In the A/B (A1/B1/A2/B2), the old pins 2/4/6/8 gave a C1 step of
  7.06–7.15 ms and the turbo pins 5.40–5.49 ms. C1 decode went from 514 to 614–639 tok/s. With turbo
  pins every context bin is 5.35–5.44 ms, inside box-c copy b's 5.3–5.9 ms, so the C1 gap to box-c came
  from the pin choice, not the host. The drafted "ask the provider for turbo" was withdrawn and never sent.
  Verdict: **adopted**.
- **trtllm fusion off** (#666): 1 gain out of 38, C1 +2.0 / +2.6% inside the band. Verdict: **no proven
  gain**; the final keeps the fusion.
- **All-reduce multicast** (#368): at decode sizes the push kernels are 2–3× faster. Multicast wins only
  on 1–2k prefill chunks, by about 5 µs, and changes the numbers. Verdict: **not worth a fork step**.
- **SWA interior-evict** (`9e295882`): merged into every box-a tree since `ace628a0` and switched off
  (`SGLANG_SWA_EVICT_INTERIOR_FIRST` unset). The A/B (#619) never ran. Verdict: **never armed**.

### n-gram copy drafts (#694, #830, #836, #862): NO-GO

The fork chain was `15dfd47e` → `1da773bf` (the copy) → `271960e4` (defer the build until the pool
exists) → `4d5dc2f2` (`req.kv.req_pool_idx`) → **`28097294`** (non-vocab ids). It took three boot fixes
before it could boot on box-a's line at all. It is kept on fork branch `box-a-picks-ngram-copy`.

| arm | numbers | verdict |
| --- | --- | --- |
| N1: MIN_N 12, HISTORY 8192, max_bs 1 | gsm8k 97; real-prompt replay geomean **0.975** (CI 0.942–1.009), 18/40 faster, aggregate 762.1 → 753.2 tok/s; C1 step 5.38–5.44 → 6.17–6.22 ms (+15%); C1 decode 646 → 580 (−10.2%); the copy fired on ~1% of lookups | rejected |
| N2: HISTORY 262,144 | replay geomean 0.984 (CI 0.949–1.022); step 7.63 ms (+42%); C1 −29.4% | rejected |
| offline and wide verify (#830/#836/#862) | the copy would save 1.4–1.9% of steps; on real outputs a buildable selector gives +5…+8% per code-edit turn but ≤ 1–4% fleet-wide; as shipped (W6, sync lookup, block 5) C1 −10.7…−11.9% | **final NO-GO** on both boxes; reopen only if sglang #39173 (compact/ragged verify) lands |

### DSpark sampling-accept graph (#694): never set

`SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH` worked like this. G1 ran the probe commit `1b8cbaf3`, now fork
branch `archive/box-a-g694-probe-20261029`. Every sampled step went through the graph. The bs-1 accept
length collapsed to 1.00–1.10 (vs 3.23) and decode fell from ~700 to ~190 tok/s. A CUDA illegal memory
access followed 1 s after the first bs-2 step. G2 on a clean `15dfd47e` reproduced the collapse:
accept 1.084, 194.5 tok/s. Verdict: **rejected; never set this flag**. The root cause was not pinned.
One lead: the captured region holds three in-graph TP broadcasts under `SGLANG_SPEC_TP_SYNC=all`.

### Megakernel (`sgl-nvidia-mega`, #301 and #621): not adopted

The plugin was upstream `sgl-project/sgl-nvidia-mega` at `39595061519a` (#26), carrying one local patch,
[`sgl-nvidia-mega-b300-spec.patch`](sgl-nvidia-mega-b300-spec.patch). The patch lowers
`resident_blocks` from 152 to 148 (GB300's SM floor down to B300's 148 SMs; unofficial) and adds
`Feature.KV_OFFLOAD` to the spec's features (needed by MK3). The plugin ran in its own venv
`/opt/sglang-mega` (SGLang `fd3545a4` editable, plugin editable). MK2/MK3 overlaid the final tree with
`PYTHONPATH=/workspace/box-e-up/python`. The plugin forces EP1, per-rank Engram tables and
`SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample`.

| arm | what booted | C1 / C2 / C4 / C8 / C16 per stream | verdict |
| --- | --- | --- | --- |
| final (base) | the stack, EP4 | 632 / 546 / 448 / 348 / 296 | — |
| MK1 (#301, 10-27) | not like-for-like: no kv-events/load-publish, no hicache, no BV, no breakable graph, no trtllm, tails 4096, EP1, on `:8012` | 581 / 490 / 495 per stream; C8 / C16 aggregate 2,272 / 3,576 (the stack 516 / 466 / 444; 2,472 / 3,990) | 10 regressions (aggregate −6…−12.5% at C4–C16, TTFT 3–6×, resume +71%); not shipped |
| MK2a (#621) | final − hicache, EP 4 → 1, + `--enable-nvidia-megakernel` | 866 / 698 / 558 / 402 / 302 | `gates-MK2` PASS over both boots: decode +38.5 / +29.0 / +24.4 / +13.9 / +2.7%, aggregate +38.5 / +31.2 / +23.7 / +12.5 / +4.0% |
| MK2bR (second boot, rerun) | same | 865 / 703 / 560 / 394 / 310 | (in `gates-MK2`) |
| MK3a | final, EP1, hicache io `direct` + `page_first_direct`, + MK (needs the KV_OFFLOAD patch) | 881 / 694 / 561 / 391 / 296 | `gates-MK3` PASS over both boots |
| MK3b | same | 882 / 675 / 550 / 393 / 298 | (in `gates-MK3`) |

- **Speed and cost.** Decode per stream against the final:

  | | C1 | C2 | C4 | C8 | C16 |
  | --- | --- | --- | --- | --- | --- |
  | change | +37…+39% | +22…+29% | +22…+25% | +12…+14% | +1…+3% (neutral) |
  | aggregate tok/s, final → MK | 632 → 874 | 1,008 → 1,294 | 1,624 → 1,994 | 2,508 → 2,813 | 4,104 → 4,227 |
  | cost per output token, final → MK | −28% | −22% | −18.5% | −11% | −3% |

  Cold 71k fill 1.46 → 1.42–1.44 s. Accept length unchanged.
- **Memory.** EP1 puts the whole expert set on every GPU. Weights go from 73.70 to 99.95 GB per GPU,
  plus a 1.70 GB megakernel workspace. That is a net +25.95 GiB/GPU after the draft's change. The KV pool
  falls from 66,653,184 to **49,956,352 tokens (−25.05%)**; MK1's was 38,666,496. Idle resume from host
  (MK3a): 400k 0.972 s, 900k 1.839 s; cold 400k 9.48 s, 900k 25.58 s.
- **Quality held.** gsm8k 98/98 (MK2a), 99/97 (MK3a), 98/98 (MK3b). L3a sign-off on MK2a PASS 17/17
  with DSML 36/36. Tool-call padding probe 0/1,200. On DSML `T1_todo_nested`, MK3b fails exactly like
  the final (361/362, 362/363: the tree's defect, not the megakernel's), with fewer padded keys
  (157/181 vs 6,347/6,322).
- **Numerics changed: the open drift.** Teacher-forced top-1 agreement against the final is short
  **94.35%** (KL 4.0e-2) and long 94.78% (KL 3.0e-2). The final against a second boot of itself is
  short 100% / KL 0 and long 94.93% / KL 2.28e-2. Greedy byte-equal outputs against the final:
  **1/32**; MK2a against MK1: 28/32. Under live load MK3b's greedy matched its own idle reference
  **0/32**: its numerics depend on the batch. The cause is not found. The suspects are the attention
  chain's swap-AB path for 1–24 rows (the chain's `SWAPAB_MAX_ROWS=24`; sglang #44632 raised SGLang's own
  cap to 40) and the plugin's other numeric slots. Upstream plugin #27 (FlashMLA schedules-ahead) is an
  optimisation, not this fix. Upstream main `54b42892` speaks megakernel API v2; SGLang `fd3545a4` and
  `819756d4` speak v1 and would silently run without it. See the
  [retry runbook](RETRY-MEGAKERNEL.md).
- **Hangs.** There were 0 hangs in 0.58 loaded prod hours (95% upper bound 5.2/h) and 0 in 1.08 h
  counting the 30-minute mixed off-pool soak (1,502 requests, 0 errors, max gap 0.163 s; bound 2.8/h).
  Neither is enough to show the no-hangs condition. The MK3b live soak (00:26:52–00:44:18Z, 40 prod
  requests, all 200, TTFT p50 1,046 ms) was stopped early on the DSML gate's FAIL, which the final
  shares. The user's 1-hour synthetic 1–16-stream run (#953) was cancelled by the retirement.
- **box-c fit.** A box-c copy under the megakernel does not fit at mem-fraction 0.70. It keeps ~0.87–1.25M
  device tokens where 2.15–3.25M are needed (`<HOME>/workspace/inference/.tmp/box-c-mk-fit/FIT.md`).
  It needs about 0.80.
- **Verdict: not adopted.** Of the provider-ticket conditions, the speed condition (≥ 20% at 2–4 streams)
  was met. No-hangs and box-c-fit were not.

The design note on making the megakernel indexable is
[`../../box-c/qa/box-a-ship1-20261027/megakernel-kv.md`](../../box-c/qa/box-a-ship1-20261027/megakernel-kv.md).
The full #621 tables are in the [tuning record](../../box-c/qa/box-a-tuning-20261026/README.md)
"Megakernel" and [`agent-speed`](../../box-c/qa/agent-speed-20261026/README.md) §3.5.

### `box-a-latest` (#869, 10-29): not adopted (retired mid-window)

Fork branch `box-a-latest` `d03143934671` = `prod-tool-strict-default` `faabdcae` (#901) + upstream #45673,
#43978, #46835 (parallel host-pool allocation) + min_p validation `219c0feb`. Tests passed on the build host. It was
the Caddy h2-window swap (16 MiB connection window) plus this tree, with `PAR_ALLOC=32`:

- The cut was 02:05:07Z. READY came at 02:17:05Z, 473 s into the boot, and host-pool allocation took
  ~57 s against ~4 min on the final.
- Smoke was 10/10 and gsm8k 98. Greedy was 32/32 against both the final and its fresh boot. The short
  eq-set was 16/16 byte-equal and the long one 0/16, which is normal run to run.
- Accept was C1 3.423 and C2 3.371. Decode at C1/C2/C4 was 638 / 552 / 440 against 624 / 526 / 450.

The retirement came at 02:2xZ, before the DSML gate. Caddy rolled back to stock at 02:21:55Z. The final
relaunched at 02:23:12Z and was READY at 02:33:32Z. box-a was back in the pool at 02:34:24Z, health 0 on
all three prod pods at 02:35:21Z. Verdict: **not adopted, not gated**; the branch stays on the fork.

## Cost (#910, read 2026-10-28 ~23:5xZ)

The two boxes cost the same per GB of GPU memory-hour; box-c, the older 8×B200 box, costs more per hour
overall. Hourly prices, bandwidth and per-day totals are not reproduced in this repository.

- box-a's whole life ran from creation (~15:00Z 10-26) to the stop at 05:27:26Z 10-29: about 62.5 h.
- A stopped instance still bills its disk. The rental contract ends 2026-11-27T01:20:44Z. The provider's
  `Instances FAQ` says that at the rental end date the contract expires and the instance is stopped; its data
  stays until it is destroyed, but the provider says to assume the instance is lost then, and only the host can
  extend the contract. So the retry starts before that date or rebuilds from the archive
  ([`RETRY-MEGAKERNEL.md`](RETRY-MEGAKERNEL.md) §4). A start needs the machine's four GPUs to be
  free again.
- Source: `<HOME>/workspace/inference/.tmp/host-cost/instance-<INSTANCE_ID>.json`.

## Outages and lessons

| when | what | cause | lesson / fix |
| --- | --- | --- | --- |
| 10-26 bring-up | the watchdog relaunched Caddy three times during the 27-min cold boot | its 180 s boot grace | boot grace + ceiling (#167); `WATCHDOG_BOOT_GRACE=2700` in the box file |
| 10-26 bring-up | the DSpark draft glue graph failed to capture and fell back to eager on every rank | `seq_lens.tolist()` under capture | fork `fd3545a4` |
| 10-27 | the 1M-probe gate ran ~2.5 min after the pool publish instead of before | order | gate before publish |
| 10-28 ~21:12Z | **MK2b window lapse**: box-a, on an untested megakernel boot, served 205 prod requests 21:17:55–21:35:16Z (all 200) before the re-hold at 21:37:16Z; MK2b's battery was contaminated and rerun as MK2bR | the hold's re-arms went to a router pod that had been replaced, so the window ran out on the live pods | `window-keep.sh` re-arms every pod hourly; confirm a window on every pod (`max by (backend, instance)`), never one pod's log |
| 10-29 00:24–00:45Z | the live-soak watchdog (`mkwatch.sh`) ran on the Mac: a harness reaper killed one launch, and it needed `caffeinate` and a double fork to survive | the watchdog sat on the operator's machine, not the box | a soak's hang watchdog runs **on the box** (the user's #953 ask) |
| 10-28/29 | the prod router pods rolled several times during holds | release work beside the soak | windows live in the router DB; read every window again after any pod roll |
| standing | box-a's router `kv_capacity` read 55,363,328 while the engine had 66.65M | left alone to keep the pool diff minimal | moot once the rows are removed |

## Archive (#978)

The working copy is `<HOME>/workspace/backups/box-a/` on the Mac. It was taken 2026-10-29 ~02:30–02:50Z,
read-only, while box-a served the final.

- **Size:** 10,764 files, 409,789,896 bytes. `shasum -a 256 -c SHA256SUMS` passes.
- **`MANIFEST.md`:** a "What a retry / rebuild needs" table, the NOT-copied list with sizes, and one row
  per file.
- **Layout:**
  - `box/<absolute box-a path>` holds the files. Secret values are redacted in `*.REDACTED` copies.
  - `meta/` holds the git states, `fork-refs.txt`, the engine argv and environ, the pip freezes of all
    three venvs, `venvs.txt` and system / `nvidia-smi`.
  - `_work/` holds the scripts that gathered it.
- **Not copied:**
  - the model, 476 GB. It is on box-c and `<ARCHIVE_VOLUME>`, and its config and index sha256 are in
    `meta/model.txt`.
  - the venvs (19–21 GB each), uv caches and driver caches.
  - git checkouts that equal fork commits, and the Caddy binaries. The h2-window binary is on the Mac at
    `<HOME>/workspace/backups/caddy-h2/`.
  - secrets: ssh keys, `box-a-kv-key`, `.host_api_key` and `/etc/sglang-api-key.yaml`.
- **Unredacted key on the box:** box-a's own logs on the box (`/workspace/sglang-serve.log`,
  `/workspace/mega/mega-serve.log`, `/workspace/box-a-mk2/qual.sh`) carry the engine API key in
  plaintext. Never copy those files off unredacted.
- **Fork commits:** every box-a commit is on our SGLang fork. The one box-only commit, `1b8cbaf3`
  (the G1 probe), was pushed to the new branch `archive/box-a-g694-probe-20261029`.
- **What to cold-store:** the curated subset for `<ARCHIVE_VOLUME>/AI/` is the user's
  call. box-a never served anything box-c did not, so the bank rule does not apply.

## Where the raw lives

| what | where |
| --- | --- |
| in this repo | [bring-up](../../box-c/qa/box-a-bringup-20261026/README.md), [tuning + megakernel](../../box-c/qa/box-a-tuning-20261026/README.md), [ship 1](../../box-c/qa/box-a-ship1-20261027/README.md) (its `raw/`), [agent-speed §3.5, §4, §7](../../box-c/qa/agent-speed-20261026/README.md), [`../../box-c/CHANGELOG.md`](../../box-c/CHANGELOG.md) |
| on the Mac (git-ignored scratch) | `<HOME>/workspace/inference/.tmp/`: `box-a-mk2/` (#621: NOTES.md, raw/, gates-MK*.txt, l3a-MK2a-s0/, and the Mac-side `mkwatch.sh`, `begin-hold.sh`, `end-hold.sh`, `health-read.sh`, also archived at `<HOME>/workspace/backups/box-a/mac-side/` with their own `SHA256SUMS`), `box-a-ship1/`, `box-a-fusion-off/`, `box-a-turbo-map/`, `box-a-arms-ngram-accept/` (NOTES-694.md), `ngram-copy/`, `ngram-offline/`, `ngram-wide-verify/`, `fork-ngram-fixes/`, `box-a-window-869/`, `box-c-mk-fit/`, `host-cost/`; the overnight summary `overnight-report.md` |
| the box archive | `<HOME>/workspace/backups/box-a/` (above) |
| L3a runs | the release gate's run output for box-a (git-ignored) in the checkout that ran them; the MK2a run is `<HOME>/workspace/inference/.tmp/box-a-mk2/l3a-MK2a-s0/` |

## Stopping box-a

The order is the user's. It is not tied to box-c's reset and has no deadline.

1. **Archive:** done (#978, above).
2. **Out of the pools:** done 2026-10-29.
   - `box-a-dev` left `box-c-dev` in dev config v214 at 03:00:31Z.
   - `box-a` left `box-c-prod` in prod v214 at 03:02:08Z.
   - box-a's DNS record was deleted at 03:06:51Z.
   - box-a's silences `0d1d1e5a` / `a66cefa8` run to 2026-10-30T12:00Z.
   - The KV-event tunnel `deploy/box-a-kv` is scaled to 0.

   box-c's copies `box-c-a` and `box-c-b` are now the whole pool, and the restart recipes in
   [`../../box-c/07-operations.md`](../../box-c/07-operations.md) read that way. Nothing routes to box-a
   any more.
3. **The fault drill on box-c:** done 2026-10-29 05:17–05:25Z (#866), after router release 14. box-c's
   Caddy and both `box-c-kv` tunnels were stopped 15.0 s with `box-c-a` held out: the router held every
   request (7 served, 0 expired), no client saw an error, the slowest held request took 17.7 s
   against the 20 s hold ([`../../box-c/CHANGELOG.md`](../../box-c/CHANGELOG.md), the #866 drill entry).
4. **`hostctl stop instance <INSTANCE_ID>`:** done 2026-10-29 05:27:26Z, the user's order's last step;
   `actual_status` read `exited` at 05:29:09Z. The instance is never destroyed. Its 1,500 GB disk
   keeps billing its disk, and its contract ends 2026-11-27T01:20:44Z. A restart
   needs machine <MACHINE_ID>'s four B300s still free ([`RETRY-MEGAKERNEL.md`](RETRY-MEGAKERNEL.md) §1,
   step 1).

The box file stays as the record of what box-a served, and it is what the retry stages from. The
retry's start re-creates the DNS record with `dns.sh`.
