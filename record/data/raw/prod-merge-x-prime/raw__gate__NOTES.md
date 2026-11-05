# box-a on prod's tip with config X' — gate and return (#84, 2026-10-30)

box-a (4×B300) left the router at 12:00:46Z. In that time it moved from config X on `box-a-mk-latest`
`9ec65dacaf` to **config X'** on our SGLang fork **prod's tip `4943d6ed90`**, passed a light gate against S7,
and returned to both pools at 12:51:19Z, about 51 min out. Both router configs now name box-a for cached image
refs.

`4943d6ed90` is `0ab213b25a` (box-a-mk-latest merged into prod `460674361992`) plus the sbr test fixes, two
redaction gaps and the upstream picks #46645 (xgrammar stop tokens), #38565 (deterministic top-p/top-k renorm for
TP>1) and #47149 / `4201b8fa99` (amax8_varlen and mega-MoE pre-dispatch grid-stride for 1024-thread SMs).

Config X' is config X plus box-c's four wins, in new files. The old pair is kept as the way back.

| File | Change from config X |
|---|---|
| `tools/args/stack-b300-box-a-mk2.args` | `--hicache-write-policy write_through_selective` (was `write_through`); `--enable-decoder-swa-bounded-replay` (was `--no-enable-decoder-swa-bounded-replay`) |
| `tools/args/env-box-a-mk2.txt` | adds `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32`, `SGLANG_DSV4_EXACT_DECODER_TRIM=1`, `SGLANG_MM_CACHED_REFS=1`, `SGLANG_MM_CACHED_REFS_EMBEDDING_CACHE=0`; keeps the TP probe |
| `env-box-a-mk2-gate.txt` (this dir) | the gate boot's env: `env-box-a-mk2.txt` without the three `SGLANG_MK_TP_PROBE*` lines |

The box file `tools/boxes/box-a.env` now reads `ENV_SEED=env-box-a-mk2.txt`, `FORK_BRANCH=prod` and
`FORK_TREE=4943d6ed906ad0f761c4845ce85ad664e7f1b7fb`. The venv `/opt/sglang-mk29`, the plugin
`/workspace/mk29/sgl-nvidia-mega` (`54b4289` + B300 patch) and `CACHE_DIR` are unchanged.

**Way back:** `restart.sh` with `--tree 9ec65dacaf` + `args/stack-b300-box-a-mk.args` + `args/env-box-a-mk.txt`, and
the box file's three lines back (FORK_BRANCH=box-a-mk-latest, FORK_TREE=9ec65dacaf52…, ENV_SEED=env-box-a-mk.txt).
On the router: dev → publish v224's document, prod → publish v254's document (the cached-image path only).

## Timeline (UTC)

| Time | Event |
|---|---|
| 12:00:46 | Maintenance windows opened (3600 s, `--hold-until-recheck`) on prod `box-a` and dev `box-a-dev`; all 5 pods read `draining` by 12:00:58 |
| 12:00:5x | Alertmanager silences: prod `e1b95096` (`backend=~"box-a"`) and dev `7ed42bca` (`box-a-dev`), 3 h |
| 12:01:07 | hangwatch stopped (`touch …/watch/serving/stop`); last line `probe_ok=282568 probe_mismatch=0` |
| 12:01:44 | Own `watchdog.hold` written; engine pgid 308820 sent TERM; GPUs at 0 MiB by 12:02:28 (the hold is gone since: see "Watchdog hold" below) |
| 12:02 | `/workspace/mk29/sglang`: `git fetch` prod (FETCH_HEAD `4943d6ed906a…`), `git switch --detach 4943d6ed90`; `git status --short` empty. Rollback `9ec65dacaf` is an ancestor (60 files differ, no pyproject or sgl-kernel change) |
| 12:02–12:03 | GPU proofs (below) |
| 12:03:49 | Gate boot `prodtip-gate` (`BOX_HOLD_OVERRIDE=1`, mk2 args + gate env); READY 12:15:39 (705 s) |
| 12:15:45–12:34:20 | Gate (below) |
| 12:34:32 | Serving boot `prodtip-serve` (mk2 args + `env-box-a-mk2.txt`, probe on); READY 12:46:21 (703 s); prewarm 75.0 s, 824 req, 0 errors |
| 12:34:45 | dev router v226 published (cached refs for `box-a-dev`) |
| 12:46:43 | hangwatch restarted (`--service --on-hang kill`, pid 357157) |
| 12:47:5x | dev `backend recheck box-a-dev` → `window_ended`; both dev pods healthy 12:48:00 |
| 12:48–12:50 | Synthetic multi-turn image chat through the dev router (below) |
| 12:51:10 | prod router v256 published (adds `box-a` to the cached-image backends) |
| 12:51:19 | prod `backend recheck box-a` → `window_ended`; runtime healthy on all 3 prod pods 12:51:23; vmsingle `max by (backend, instance)(sbr_backend_health_state{backend="box-a"})` 0 on <router-pod-ip>, .1.190, .1.238 at 12:52:01 |
| 12:52 | Both #84 silences expired. The older billing silence `f5d9aac7` (#1324) is not ours and still stands |

## GPU proofs (#91's, engine down, 4943d6ed90, `/opt/sglang-mk29` python)

Logs: `box/test_renorm.log`, `box/test_mega_moe_pre_dispatch.log`.

| Test | Result |
|---|---|
| `test/registered/kernels/ops/sampling/test_renorm.py` (GPU 0) | 53 passed, 3 skipped, EXIT 0, 16.7 s (`-q` without `-rs`, so the skip reasons were not printed; they are explained below) |
| `test/registered/kernels/ops/moe/test_mega_moe_pre_dispatch.py` (GPU 1) | 6 passed, EXIT 0, 16.7 s |

**The 3 renorm skips** are the test's own parameter guard, not a missing GPU. The file at `4943d6ed90` (160 lines)
has 56 cases: `test_top_k_renorm_probs` 27 (batch 1/99/989 × vocab 111/32000/128256 × k 10/100/500),
`test_top_p_renorm_probs` 27, and the two `_is_deterministic` cases. Line 56 is
`pytest.skip("k should be less than vocab_size")`, which fires for vocab 111 with k 500 at each of the 3 batch sizes:
3 skips, 53 + 3 = 56. The only other skip is `skipif(is_hip())` on the two deterministic cases, which ran (CUDA).

**GPU 0 only is the test's design.** Every case runs on one device: #38565 makes the renorm kernel return the same
output for the same input, which is what keeps the TP ranks in agreement. No test in the file starts more than one
rank. Rank agreement on TP4 is checked live by the serving boot's TP probe instead: `MK_TP_PROBE ok` on every rank and
hangwatch `probe_mismatch=0` (Watch below, and `box/watchdog-state.txt`).

## Really on (both boots)

Boot sections, copied from the box `/workspace/mk29/gate84/` with the engine API key redacted (`'api_key':
'<redacted>'` on line 22 of each): `box/boot-section.log` (gate) and `box/serve-boot-section.log` (serving).

- **Megakernel:** the target runs 27/27 decode graphs (bs 1…64); the DSpark draft runs 8/27 (bs 1–8), as on S7.
  `max_total_num_tokens=49,879,552` (unchanged), `swa_tokens` 877,568.
- **Errors:** 0 Tracebacks and no refusal. No `no dense fp4` indexer fallback or any fallback line. FlashInfer
  AllReduce Fusion enabled (`trtllm`) on all 4 TP, with no fusion init error.
- **Trim on:** `SGLANG_DSV4_EXACT_DECODER_TRIM` raises the decoder tail's max rows 2048 → 2816 (log line on every TP).
  The args carry `--enable-decoder-swa-bounded-replay` and `write_through_selective` (READY line,
  `restart-*.log`).
- **Host-cache allocation:** with 32 threads it took 25 s (TP0's first `Allocating … host memory` → `Attached hybrid
  pool`), against 117 s on the previous boot. The gate ran 12:15:03 → 12:15:28 and serving 12:45:39 → 12:46:04;
  the previous boot ran 07:16:01 → 07:17:58. READY took 705 / 703 s, against 795 s.
- **Limits and listeners:** `Max core file size` is 0 on all 7 serve, scheduler and detokenizer pids (347961,
  348410, 348411, 348476, 348542, 348608, 348609; #84's live check counted 8 and was not saved). Ports 5557/5558/5559
  listen on 127.0.0.1 only, with nothing on 0.0.0.0 in 5557–5569. The live checks were not saved, so both were
  read again read-only at 13:58Z: `box/limits-core.txt` (its last pid, 361277 `bash`, is the capture's own ssh shell
  matched by `pgrep -f sglang`) and `box/netstat-listen.txt`. In `netstat -ltnp`, 0.0.0.0 has only sshd :22. The
  engine listens on 127.0.0.1:8002, and TP0 (348411) listens on 127.0.0.1:5557–5559 and :<port>. The schedulers
  listen on `::1:<port>`, and TP0 also on `:::<port>` (IPv6 any). Caddy is on :80/:8001 and the remaining listeners
  are on the container address <container-ip>.
- **Serving env:** the scheduler environ carries all four new keys and `SGLANG_MK_TP_PROBE=1 / EVERY=100 /
  ASSERT=1`. The first probe line was `MK_TP_PROBE ok it=1000` at 12:46:08 on every rank.

## Gate (label MKP1, probe off, S7's commands; box `/workspace/mk29/gate84/`, script `box/gate.sh`)

Run 1 started 10 s after READY. `restart.sh`'s own prewarm ran 12:15:50–12:17:40 (`/workspace/prewarm-prodtip-gate.log`),
so run 1's greedy, tf and KL overlapped it. `flush_cache` answered `Flush cache failed.` because requests were in
flight. Those three steps were run again idle, as MKP1b, after DSML.

That in-flight cause is shown by MKP1b: the same `flush_cache` call with the same auth header answered `Cache
flushed.` before its greedy and tf (`box/gate-greedy2.log`, `gate-tf2.log`; KL flushes in neither run, per
`box/gate.sh`). sglang always
appends "Please check backend logs … (When there are running or waiting requests, the operation will not be
performed.)" to its answer, so that line is no failure.

**Correction: the arm's warm-up step overlapped the last 19 s of the prewarm. Its measured legs did not.** The
arm began at 12:17:21 (KL rc in `box/gate.log`), before the prewarm ended at 12:17:40. Its first step is
`bench.py`'s unmeasured warm-up (`arm/warm-MKP1.txt`). The warm think ramp at 12:17:21 decoded at 104 tok/s, against
~900 idle (84–115 tok/s at C1/C4/C16, 12:17:21–12:17:32). At 12:17:37 the warm cold fill logged `live #running-req 14 > 0; waiting 20 s`: bench.py's own idle guard
waited out the prewarm. The first measured row, think C1 rep 0, is 12:18:27 (`arm/think-MKP1.jsonl`), 47 s after the
prewarm ended. No measured arm row overlaps it.

| Gate | Result | Reference (S7 = MKL2) | Verdict |
|---|---|---|---|
| greedy 32, idle (MKP1b, 12:32:58) | 32/32 byte-equal vs MKL2, MKL1 and MK3a; 1/32 vs FB | S7: 32/32 vs MKL1 | **PASS** |
| (run 1, overlapping prewarm) | 27/32 vs MKL2/MKL1/MK3a; differ [0,1,2,15,16], coherent near-tie flips | — | not idle |
| tf short, idle (MKP1b) | vs MKL2: top-1 100.000 %, KL 0, the same vs MKL2b and MK3a; vs FB 94.353 %, KL 4.00e-02 (same as S7's MKL2 vs FB) | S7 MKL2 vs MKL2b: 100 % / 0 | **PASS (identical)** |
| tf long, idle (MKP1b) | vs MKL2 95.343 %, KL 2.43e-02; vs MKL2b 95.098 % | S7 within-boot long ~95 % (decode-nondeterministic arm) | in range |
| (run 1) | short 99.611 %, KL 1.63e-03; long 95.466 % | — | not idle |
| KL 1 rep, idle (MKP1b, 12:34:04) | short 0.004604 (bound 0.03), long 0.009151 (bound 0.06) | S7 short ~0.0045, long ~0.010 | **PASS** |
| (run 1) | short 0.004563, long 0.020538 (per prompt .0045 .0300 .0067 .0409) | — | PASS, not idle |
| DSML T1 strict-unset + strict:false, 470 / arm / mode @16 (12:26:30–12:32:41) | 1880/1880 recorded. T1 stream 4/470, nonstream 4/470; strict:false stream 3/471, nonstream 3/470; differ 0. `DSML-SELFPARSE PASS` | — | **PASS** |
| DSML two-sample vs S7 (`dsml_compare.py`, `--allow-missing`) | 12 comparisons, nothing WORSE (largest +0.43 pp, Fisher p ≥ 0.31) | S7 l3a/dsml (@16) | **PASS** |

**1880 vs 471.** 1880 counts sends: 2 cases × 2 modes × 470. The defect denominators count tool calls. One send,
strict:false stream `s160`, emitted two `todo_write` invokes in one reply. Its nonstream twin emitted one
(`box/dsml-strictfalse-s160/stream.raw.txt` lines 16 and 19; `nonstream.raw.txt` line 26). So that cell is 3/471,
from 470 sends. A scan of all 470 strict:false sends per mode for more than one `invoke name=` finds only `s160
stream`. `dsml_compare.py` checks sends, not calls: the `n differs … sends with more than one call` notes in
`dsml-compare-S7-vs-MKP1.txt` are this.

**Concurrency.** S7 ran @16 too, not @8: S7's
`box-c/qa/box-a-megakernel-20261029/raw/s7-mk-b2/l3a/dsml/probe.log` line 1 reads `repeat={… 470 …} concurrency=16`.
MKP1's `repeat.json` reads `concurrency: 16`. Both arms ran the same load.

**`--allow-missing` waived no cell.** `dsml_compare.py` reads only count rows (`rows_from_json`, lines 59–66, drops a
row without `todo` counts). S7's selfparse has 40 rows; the 36 single-shot ones (C1, M1–M3, P1, P2, S1, S2 and
N1–N5 auto/required, × 2 modes) are graded PASS by `analyze.py` and never enter the compare. Both arms then hold the
same 4 T1 rows. With the flag, a row in only one arm prints `… only in <label>` (lines 118–122). The output has no
such line, and its footer reads `12 comparison(s) over 4 row(s)`. The flag only kept the run from refusing (exit 3)
had a row been missing. None was.

The arm (12:17:21–12:26:13) ran `bench.py` warm, then think C1/C2/C4 ×8 (seed 4242+C), then coldfill ×8 at offset 46.
`gates.py --arm p1=…@MKP1 --base s7…@MKL2,s4…@MKL1 --levels 1,2,4 --metrics decode,aggregate,ttft,fill,dip,gap` →
`gates-MKP1-vs-MKL1+MKL2.txt`, one boot (exit 3).

**Exit 3 means one arm boot, so no verdict. It is not a failure.**
`box-c/qa/window2-tp4/harness/gates.py` lines 651–658: exit 1 only on a regression that replicates or on a quality
miss, and there is neither here. With fewer than 2 arm boots it prints `OVERALL: ONE BOOT … run a second arm
boot` and returns 3, because the rule (its docstring, the user 2026-10-27) needs a regression to replicate and a gain
to beat the band on both boots. Here that is `0 regression(s) in 12 metric sub-gates, 1 unconfirmed`. A second arm
boot was not run: box-a went back to prod on this boot.

| Metric | C1 | C2 | C4 | Verdict |
|---|---|---|---|---|
| decode tok/s | 904.4 (+4.3 %) | 689.8 (+0.2 %) | 561.9 (+0.9 %) | NEUTRAL |
| aggregate tok/s | 904.4 | 1266.0 (−0.5 %) | 2017.5 (+0.8 %) | NEUTRAL |
| TTFT | 42.9 ms (0 %) | 67.4 ms (+10.5 %) | 87.5 ms (+3.9 %) | C2 UNCONFIRMED: one 134 ms outlier rep, inside the band (below) |
| DSpark accept (`acc-MKP1.txt`) | 3.532 | 3.306 | 3.344 | — |
| cold fill (60k) | 0.989 s vs 1.435 s (−31.0 %) | | | **GAIN** |
| dip | 67.0 % vs 76.4 % | | | **GAIN** |
| gap | 142 ms vs 152 ms | | | NEUTRAL |

**The C2 TTFT outlier** is `arm/think-MKP1.jsonl` line 16: C 2, rep 5, 12:18:58Z, `ttft_med` 0.134 s. The other 7
C2 reps read 0.040–0.066 s, so the C2 median is 0.061 s. Without rep 5 the mean is 57.9 ms, against the base's 61.0 ms
(−5 %). gates.py line 15 marks it `inside band` (interval [−7.5, 27.1] ms; arm outliers 1/8 vs base 0/16, kept in
the mean). A rerun was out of bounds here: box-a is serving prod, so no bench load. A second arm boot would confirm or
clear it.

Larger cold fills are descriptive: there is no S7 run at these sizes. Each ran ×3 at `--fill-tokens` 56400 and 80600
(`arm/coldfill70-MKP1.jsonl`, `coldfill100-…`, `arm/coldfill-summary.txt` via `coldfill_summary.py`).

| Fill | TTFT | Prefill tok/s | Background dip | Max gap |
|---|---|---|---|---|
| 60,000 tokens (std, ×8) | 0.982 s | 61.1k | 66.9 % | 0.141 s |
| 70,051 (×3) | 1.120 s | 62.6k | 62.6 % | 0.135 s |
| 100,208 (×3) | 1.599 s | 62.7k | 64.1 % | 0.095 s |
| MKL2 60k (S7) | 1.456 s | 41.3k | 76.1 % | 0.150 s |
| MKL1 60k (S4) | 1.423 s | 42.2k | 77.7 % | 0.155 s |

(medians per `coldfill_summary.py`; gates.py's fill/dip above are means.)

**Max gap at 100k (0.095 s) under 60k (0.141 s).** The 100k reps read 0.095, 0.092 and 0.124 s. The 60k reps read
0.141–0.146 s (×8) and the 70k reps 0.134–0.138 s (`arm/coldfill*-MKP1.jsonl`, max `max_gap_during_s` over each
row's background streams). A shorter gap at a bigger fill is not a contradiction. With `chunked_prefill_size=8192`
(both boot sections) the fill is prefilled in 8192-token chunks interleaved with decode steps. The longest background
decode stall is one chunk step, which does not grow with the fill length. Why the 100k steps came out shorter was not
investigated: there are only 3 reps and no reference at this size. The value is descriptive and in no gate.

**FAIL rule.** Idle greedy, tf, KL and DSML all held, so no trim-off or selective-off arm was booted. Run 1's 5/32
greedy flips came from the prewarm overlap: the idle rerun on the same boot is 32/32.

## Router: cached image refs for box-a

Both edits went through `config draft --from-live`, then `--write` of the document with the one change, then
`config diff`, then `publish`. Each diff shows only the cached-image path (`router/diff-*.json`). `box-c-a`, `box-c-b`,
`box-c-a-dev` and `box-c-b-dev` are untouched.

| Env | Before | After | Diff |
|---|---|---|---|
| dev | v224 (no `cached_image_bytes`) | **v226** `{omit: true, backends: [box-a-dev]}`, published 12:34:45Z | `/admission/cached_image_bytes` |
| prod | v254 (`[box-c-a, box-c-b]`) | **v256** `[box-c-a, box-c-b, box-a]`, published 12:51:10Z, hash `1c2a42c5` | `/admission/cached_image_bytes/backends/2` = `box-a` |

**No roll needed.** Every pod logged the admission line ("… the cached-image omission … follow the live config
version", `omit_cached_image_bytes: true`) for the new version within 1 s: dev at 12:34:46 on both pods, prod at
12:51:11–12 on router-s13 `jrnp9` and router-s8 `xdr7m` / `zsh56`.

The dev router was rolled by someone else at 12:36:36Z: rs `7b99bf55f5`, image `sha256:22adab17…`, revision 160.
The new pods `5bww4` and `bcqrm` booted on v226.

**Synthetic test** (`mmrefs_dev_chat.py`; generated noise PNGs of 224 KB; no production data). The model is `box-c`
on the dev path with the dev paid key, buffered so `x-sbr-served-by` names the copy.

| Try | Setup | Outcome |
|---|---|---|
| 1 | Lone asks | Never placed on `box-a-dev`; `max_tokens` 200 gave empty answers |
| 2 | Burst of 12 | Turns 2–4 left `box-a-dev`: the prompt was under one 256-token page |
| 3 | Load balancer | All turns on `box-a-dev`, but no omission: the per-pod ledger was split across pods |
| 4 | Pod pinned (`kubectl port-forward pod/router-7b99bf55f5-5bww4 43171:8790`) | No omission: the image ended past the cached pages |
| 5 | Image first, then a 4k-token text, pod pinned | **PASS** (`mmrefs-dev-chat.jsonl`) |

Try 5, conversation `6c094de0`:
- Turns 1–4 all returned 200 from `box-a-dev`, with cached tokens 0 / 4864 / 4864 / 5120.
- Router pod `5bww4`: `sbr_egress_cached_images_omitted_total{backend="box-a-dev"}` 2, `…_bytes_saved_total` 599,320,
  and no `sbr_egress_mm_bytes_required_total` series.
- Engine: `sglang:mm_cached_refs_omitted_images_total` 2, `mm_cached_refs_served_total{source="prefix"}` 2.
- The serve log since the boot has 0 `mm_bytes_required` and 0 400/409 lines.
- Raw captures: `box/dev-metrics-1250.txt` holds the router-pod and engine scrape lines from 12:50:43–48Z. They
  were piped to the terminal and not saved, so the file is transcribed verbatim from #84's tool output. The pod
  `bcqrm` line is empty there: no omission on the other pod. The engine counters were re-read at 13:58:20Z, still
  2 and 2 (`box/engine-mm-metrics-1358.txt`), so no image traffic had reached box-a since the dev test.

## Real-traffic watch (15 min after the prod recheck)

Readout at 13:04:12Z (`watch.sh 15m` → `watch-1304.txt`; vmsingle via `kubectl port-forward svc/sbr-vmsingle`):

| Item | Value |
|---|---|
| Health state, `max by (backend, instance)` | `box-a` 0 on all 3 prod pods, the same for `box-c-a` and `box-c-b` |
| Engine | hangwatch `probe_ok=19620 probe_mismatch=0 health_fail=0 episodes=0`; 0 Tracebacks, 0 probe mismatch lines, 0 `mm_bytes_required` in the serve log since boot |
| Router errors | `mid_stream_aborts` 0 on every backend; `box-a` retries none |
| Cached refs, `box-c-a` / `box-c-b` | omitted 652 / 668, `mm_bytes_required` 0 / 0 |
| **Prod attempts on `box-a`** | **0** in the 13 min after the recheck: `box-c-a` 276, `box-c-b` 348 over 15 min. Still 0 over 3 min at 13:04:54 |

Before the window, `box-a` carried 401 attempts in the hour to 11:59Z (53 in its last 10 min). `runtime` on router-s13
at 13:05 reads `box-a` `healthy`, `placement_health healthy`, load fresh (`age_ms` 970, `source both`) with nothing
running. Its `cache_hit_rate` and `gen_throughput` are 0 and `accept_length` is 2.56, against `box-c-a`'s
0.836 / 567.6 / 3.78. Its only notes are about `or-standby`.

No `box-a` log line other than the breaker and recheck lines appears on any prod pod. `sbr_index_snapshot_refusals_total`
did not increase in 20 min, and the cold-start gate is off (`sbr_cold_start_gate_entries_total` 0).

The likeliest reading is cold-cache placement at light load: the restart emptied box-a's prefix cache, the copies
holding the shared prefixes win every placement, and box-a is picked only once they queue. The dev router behaved
the same way: 16/16 lone asks went to `box-c-a-dev` / `box-c-b-dev`, and only a burst of 12 put 2–4 on `box-a-dev`. box-c
has a precedent: SR-013's freshly restarted copy b took 15 requests against copy a's 269 in its first 11 min
(`qa/speed-register-evidence-20261029/SR-013/box-c-mmrefs-1113__notes.md`). At #39's return, by contrast, every
router pod was rolled and so started cold for all copies. This is not proven. The real-traffic leg of the watch
(probe, errors and refs under prod load on box-a) is therefore not yet measured.

### Watch under prod load (13:50Z)

Prod traffic reached box-a later. The manager's `watch.sh 45m` at 13:50 counted `box-a` 131 attempts (`box-c-a` 994,
`box-c-b` 929). Mine, `watch.sh 15m` at 13:50:36Z (`watch-1350.txt`):

| Item | Value |
|---|---|
| Attempts, 15 min | `box-a` 102, `box-c-a` 315, `box-c-b` 175 |
| Retries | none |
| `mid_stream_aborts` | 0 on every backend |
| Health state | 0 for `box-a`, `box-c-a` and `box-c-b` on all 3 instances (<router-pod-ip>, .1.165, .1.86; new pod IPs, not those read at 12:52) |
| hangwatch | `episodes=0 probe_ok=71716 probe_mismatch=0 health_fail=0` |
| Serve log since boot | 0 Tracebacks, 0 probe-mismatch lines, 0 `mm_bytes_required` |
| Engine counters | 1143 requests (948 non-stream, 195 stream), 273,799 generated tokens |
| Cached images omitted | `box-c-a` 249, `box-c-b` 414; `box-a` none, so no prod image traffic reached box-a in the window. The engine's `mm_cached_refs_omitted_images_total` is still 2, from the dev test |

Verdict: box-a served real traffic on config X' without a probe mismatch, a hang, an abort or an error. The prod image
path on `box-a` is unexercised; the dev test above is its proof. Not a fail.

## Watchdog hold: gone

The `watchdog.hold` written at 12:01:44 no longer exists, and the watchdog and hangwatch are both running. At 13:57Z
the root checked:
- `find /workspace -maxdepth 3 -name '*hold*'` finds nothing.
- `watchdog.sh` (pid 1073) and `hangwatch.py --service --on-hang kill` (pid 357157) are running.
- `watch.log` at 13:56:43Z reads `episodes=0 probe_ok=75944 probe_mismatch=0 health_fail=0`.

A read-only capture at 13:58:39Z shows the same (`box/watchdog-state.txt`):
- The find is still empty.
- pid 1073 `bash /workspace/tools/watchdog.sh` has been up 96,141 s, and pid 357157 matches `hangwatch.pid`.
- `watch.log` at 13:57:43Z reads `episodes=0 probe_ok=76604 probe_mismatch=0 health_fail=0`.
- `box-a-watchdog.log` logged `engine up` for both boots, at 12:17:56Z and 12:47:50Z.
