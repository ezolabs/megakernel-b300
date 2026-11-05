# DeepSeek-V4.1-Flash serve lines and rung ladder

## Composition rule (fixed, shared with relaunch.sh)

Every `.args` file here carries **only the tuning flags**, on **one line**, after its comment
header — never `--model-path`, `--served-model-name`, `--host`, `--port` or `--api-key`.
`relaunch.sh` appends those five from the box file (a box is a topology + identity, an args
file is a tuning line; crossing them would mean one file per box × per rung instead of one
rung ladder shared by every box). A `.args` file that is missing this separation is wrong,
not merely inconsistent — `check.sh` in this directory enforces it.

An `env-*.txt` file is the matching environment-variable pairing for one or more `.args`
files (`SGLANG_...=...` lines); it is not itself an args file and carries no flags.

**Never add** to a config-of-record line or a rung (every top-level `*.args` below; the reasons are
`../../04-engine.md` "Never set"): `--attention-backend` / `--dsv4-attn-backend` /
`--moe-runner-backend` / `--fp8-gemm-backend` — they auto-resolve to `dsv4` / `flashmla` (at
attention TP > 1) / `flashinfer_mxfp4` / `flashinfer_cutedsl`, and an override moves the attention
kernels or drops the FP8/ue8m0 blocks onto a slow Triton fallback; `--enable-dp-attention` — not a
ban (5f117c89 accepts DSpark with DP attention given `--enable-dp-lm-head` and no CP) but an open
speed A/B; `--enable-mixed-chunk` — speed only (its ~4 vs ~23 tok/s is arithmetic, never measured);
`--kv-cache-dtype` — never lossier than the checkpoint's KV (a packed FP4 main KV, sglang#38902,
would be lossless: gated if it lands for SM100); `--enable-decoder-swa-bounded-replay`. A backend
override, DP attention or mixed chunk is measured in an A/B arm (next paragraph) and reaches a
config of record only through the gate. The bounded-replay lines are the trim lines, box-c's
`v11-b200-box-e-tp4.args` (what serves) and `v10-b200-box-e-tp4.args` (its rollback), and box-a's
`stack-b300-box-a-mk2.args` (their bullets below): bounded replay
there is paired with the exact decoder trim (`SGLANG_DSV4_EXACT_DECODER_TRIM=1` in `env-box-e-w4.txt`,
what box-c serves, and in `env-box-e-w3.txt` / `env-box-e-w2.txt`, its rollbacks; in `env-box-a-mk2.txt`,
what box-a serves) on a tree carrying the trim, which makes it output-exact (CHANGELOG #732, #802;
`../../qa/prod-merge-20261030/raw/gate/NOTES.md` for box-a), and `check.sh` exempts those three files
alone (`FORBIDDEN_EXEMPT`), each pinned to an env that carries the trim (v11 through its pin to v10). **Every other line on an upstream tree sets
`--no-enable-decoder-swa-bounded-replay` by name:** upstream defaults bounded replay ON
(our SGLang fork 5f117c89 `python/sglang/srt/arg_groups/fields/exec_.py` :141–142), so leaving
the flag out serves it on without the trim. `check.sh` refuses every line that lacks it except a
trim line and the unmarked 264ea0f9-era records its `LEGACY_UNMARKED` list names, whose tree
defaults the flag off and rejects the `--no-` form. That exemption covers those named files only:
the list is frozen (a new line is never added to it, and a name whose file is gone FAILs), and a
record that gains a `# TREE:` marker leaves it, so booting one on an upstream tree means appending
the flag and the marker first ("The 8×B200 mirrors"). A new line carries
`--no-enable-decoder-swa-bounded-replay` with its tree's `# TREE:` marker, or it is a trim line.

**The one gated exception is an A/B arm.** A file under `arms/` whose header (the comment block
before its flag line) carries a line `# A/B ARM: <reason>` may carry a flag `check.sh`'s
`ARM_ALLOWED_FLAGS` lists, nothing else from the list above. That list holds the speed levers an arm
has been written for — today `--dsv4-attn-backend` (SR-070) and `--attn-dp-size` (SR-169 / #1209;
the DP width's other spellings `--attention-data-parallel-size`, `--dp-size` and
`--data-parallel-size` stay forbidden everywhere); a backend override or mixed chunk joins it in the
change that writes its arm — and never `--kv-cache-dtype`. Each arm
is pinned in `check.sh`'s `ARM_PINS` (`arm|base|added`, optionally `|flag|from|to` triples): its flag
line is a top-level base file's flag line with the declared flags appended and nothing else but
the declared in-place value changes, which only a value the arm's flags make unbootable earns, so it differs from the line it is
measured against by the backend alone, and it inherits that base's own exemption (v10's bounded
replay) with the line. An arm with no header, no pin, or any other forbidden flag FAILs, and so does
a pin whose arm is gone; every top-level `*.args` — every config of record and every rung — still
refuses the flag, and an `# A/B ARM:` header up there FAILs too. An arm is booted by path for its
A/B only (`restart.sh --engine a|b <tag> args/arms/<arm>.args`, in that copy's router
cut); no box file, rollback or rung names it, and a winning backend reaches a config of record only
by changing this paragraph and `FORBIDDEN_FLAGS` in its own change. Today:
`arms/v10-trtllm-attn.args` (SR-070, the TRT-LLM decode-attention A/B) = `v10-b200-box-e-tp4.args` +
`--dsv4-attn-backend trtllm`, the flag and value as the served trees have them (our SGLang fork
5f117c89, bdc3f67f and 9b262265, `python/sglang/srt/arg_groups/fields/exec_.py` `dsv4_attn_backend`, choices
`auto|flashmla|flashmla_mega|trtllm`; `deepseek_v4_hook.py`'s startup checks, which v10's line
meets, are listed in the arm's header). `arms/v10-dp-attn.args` (SR-169 / #1209, the DP-attention
A/B) = `v10-b200-box-e-tp4.args` + `--attn-dp-size 4 --enable-dp-lm-head`, with v10's short-prefill
block halved in place (threshold 1536, chunk 512, max tokens 2048, swap limit 8192): the fork divides
the 8192 chunk per rank (2048, `parallel_hook.py`) before `validate_short_prefill_protection`, which
refuses v10's 4096 / 16384 against it. Its ranks publish KV events per rank, so `relaunch.sh <id>`
reads the width from the line and moves the copy onto its DP area (`lib.sh` `engine_kv_ranks`: a
5600–5611, b 5612–5623; events, replay, load four each), off both copies' record ports; the router's
tunnels see none of it, so the arm copy runs out of the pool. Never with
`SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1` (sglang #46676 unmerged; `relaunch.sh` refuses the
pair). `check-import.sh` reads only its `--args-dir`, so an arm is gated with
`--args-dir args/arms`.

**Arm records.** An arm that ran and was not adopted (rejected or inconclusive) is kept under
`arms/records/`, with the env files it booted, so the A/B can be re-run byte for byte: each file opens
with a record header (what ran, where, the sha256 of the bytes that ran, the verdict and its register
row), then those bytes. `check.sh` holds a record's `.args` to the arm rules above, pinned in
`ARM_PINS` as `records/<file>`, counts records apart from the live arms, and puts every record file,
the `env-*.txt` ones included, through the deploy gate (`relaunch.sh --rest-gate`). Like an arm, a
record is never a config of record, a rung, a seed or a rollback. Today: box-c #1320's (on `prod`
a84f5742 through the `/workspace/box-c-prod-a847` overlay) `records/v10-m1-nvls.args` +
`records/env-a847-nvls.txt` (NCCL NVLS, `--enable-nccl-nvls`) and `records/v10-m2-symm.args` +
`records/env-a847-mc.txt` (NCCL symmetric-memory all-reduce, `--enable-symm-mem`, with
`TORCH_SYMM_MEM_DISABLE_MULTICAST` removed), both inconclusive at ±6% (SR-042); and
`records/env-a847-megamhc.txt` (`SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1` with v10's line), rejected
(SR-165); and box-c #1346's `records/env-4606-m3.txt` (on `prod` 46067436 through the `/workspace/box-c-prod-4606`
overlay, with v10's line: `TORCH_SYMM_MEM_DISABLE_MULTICAST` removed, so custom all-reduce v2's multicast
switch is on), NO (SR-042, rejected as env-only arms).

## Duty-cycle arithmetic

The R1 rungs below exist because a cold, multi-hundred-k-token prefill starves every
concurrent decode stream for the duration of its chunking: with `--prefill-decode-interval N`,
decode gets one step every `N` prefill chunks, so a background stream's _per-stream_ tok/s
during a cold prefill is approximately

```
tok/s_during ≈ 3.65 · N / (c + N · 0.02 s)
```

where `c` = chunk-size / prefill-tok/s (the wall time one prefill chunk costs). `c` is
measured per box, at its own fresh-prompt prefill rate and again at its own
large-context prefill rate — it is not a constant this ladder can assume in advance. Raising
`N` trades decode smoothness (fewer, longer stalls) for background throughput; `N=0` (no
interleaving, `n0.args`) is the zero point every R1 candidate is measured against.

## Why the storm rungs exist

The R1s and R8 rungs below exist because the duty-cycle arithmetic above describes a single
cold prefill, and production traffic does not arrive one prefill at a time: the observed
shape is several 500k-to-1M-token cold prompts landing within seconds of each other on top
of roughly 40 already-warm streams. This box prefills one 4096-token chunk in 0.406 s
(13,986 tok/s) and saturates near 36M uncached tokens/hour at that rate, so four concurrent
500k-token cold prompts are already most of an hour's uncached budget arriving in one burst.
Prefill-first scheduling re-queues every one of those chunks ahead of the running decode
batch on every scheduling pass, so for as long as any of the storm's cold prompts still has
a pending chunk, every warm stream decodes at whatever floor the base line's
`--prefill-decode-interval` leaves it — a compounding the single-cold-prefill freeze probe
that picked N=2 (§4a, `08-measurements.md`) never exercised, because it never ran more than
one cold prefill at a time. The storm probe (`tools/freeze-probe.py --cold-streams 4
--prompt-tokens 500000 --streams 20`) retests the ladder's existing N=4 candidate and the
prefill-batching and prefill-delayer knobs against that compounding directly.

## The ladder

| File                                    | Rung           | The one change (vs the base line, `v3.args`)                                                                     | What it measures                                                                                                                    | Accept rule                                                                                                                                                                  |
| --------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v3.args` (+ `env-v3.txt`)              | 4-GPU base     | none — the line itself: `--tp-size 4 --ep-size 4`, chunk `4096`, `--prefill-decode-interval 2`                   | —                                                                                                                                   | base line, not a rung                                                                                                                                                        |
| `n0.args`                               | N=0 reference  | `--prefill-decode-interval 0`                                                                                    | freeze probe (C20, cold 250k-token prefill): `fresh_ttft_s` and per-stream `tok/s`-during, taken as the ladder's zero point         | none — reference value only                                                                                                                                                  |
| `r1-4096x4.args`                        | R1             | `--prefill-decode-interval 4` (chunk stays 4096)                                                                 | freeze probe at C20 during a cold 250k prefill → `fresh_ttft_s`, `bg_maxgap_med`/`max`, `bg_ttft_during_med`/`max`, tok/s-during    | highest `N` with `fresh_ttft <= 1.30×` the N=0 value **and** `bg_maxgap_med <= 1.5×(chunk/prefill_tok_s + 0.02 s)`; tie among candidates → prefer 4096                       |
| `r1-2048x4.args`                        | R1             | `--chunked-prefill-size 2048`, `--prefill-decode-interval 4`                                                     | same freeze probe as above                                                                                                          | same rule, evaluated against this candidate's own `chunk/prefill_tok_s`                                                                                                      |
| `r1-2048x2.args`                        | R1             | `--chunked-prefill-size 2048` (interval stays 2)                                                                 | same freeze probe as above                                                                                                          | same rule                                                                                                                                                                    |
| `r2-tok2.args`                          | R2             | `--tokenizer-worker-num 2`                                                                                       | **BLOCKED** — see below                                                                                                             | —                                                                                                                                                                            |
| `r2-tok4.args`                          | R2             | `--tokenizer-worker-num 4`                                                                                       | **BLOCKED** — see below                                                                                                             | —                                                                                                                                                                            |
| `lpm.args`                              | R-LPM          | `--schedule-policy lpm`                                                                                          | two concurrent freeze probes                                                                                                        | `bg_ttft_during_med` drops vs the base line **and** `fresh_ttft` max stays `<= 2×` the single-cold-probe value                                                               |
| `r3-mfs1.args`                          | R3             | `--min-free-slots-delay 1` (default for DSPARK is 4)                                                             | C64 ramp + soak                                                                                                                     | TTFT p90 and tok/s at C64 do not regress vs the base line, retractions 0                                                                                                     |
| `r3-mfs8.args`                          | R3             | `--min-free-slots-delay 8` (default for DSPARK is 4)                                                             | C64 ramp + soak                                                                                                                     | same rule                                                                                                                                                                    |
| `adm128.args`                           | R-ADM          | `--cuda-graph-max-bs-decode 128 --max-running-requests 128` (base line: 64/64)                                   | greedy/sampled C1 throughput, C64 ramp                                                                                              | greedy/sampled C1 within the base line's measured spread, retractions 0, C64 TTFT p90 holds                                                                                  |
| `moetp.args`                            | R-MOETP        | `--ep-size 1` (base line: 4)                                                                                     | C1 throughput, C20/C64 ramp                                                                                                         | C1 gains beyond spread **and** C20/C64 hold; forfeits R5 (EPLB)                                                                                                              |
| `r4-cp8192.args`                        | R4             | `--chunked-prefill-size 8192` (our rung value), `--mem-fraction-static 0.8` (the SGLang cookbook cell's)         | 1M-token probe                                                                                                                      | passes with 0 first-try allocator failures and tok/s@20 holds vs the base line; on pass, rerun R1's freeze probe at chunk 8192                                               |
| `r5-eplb.args`                          | R5             | `--enable-eplb`                                                                                                  | expert-load skew under sustained mixed traffic, tok/s at bs 8 and 20 over a soak                                                    | tok/s gain at bs 8 and 20 over the soak, run only if R-MOETP was declined                                                                                                    |
| `r6-compact.args` (+ `env-compact.txt`) | R6             | `--speculative-dspark-sps-table-path /workspace/tools/sps.json`, paired with `SGLANG_RAGGED_VERIFY_MODE=compact` | —                                                                                                                                   | **BLOCKED** — see below                                                                                                                                                      |
| `r7-hicache.args`                       | R7             | `--enable-hierarchical-cache --hicache-ratio 2`                                                                  | canary boot + smoke, then a multi-day prefix-persistence soak                                                                       | canary passes cleanly (smoke, GSM8K, one soak) before it is left running longer-term; two-day prefix-persistence read confirms cache entries survive                         |
| `r1s-int4.args`                         | R1s            | `--prefill-decode-interval 4` (same tuning line as `r1-4096x4.args`, chunk stays 4096)                           | storm probe: `freeze-probe.py --cold-streams 4 --prompt-tokens 500000 --streams 20` plus `ramp.py --context cached --levels 1,8,20` | `bg_tok_s_during_med` improves vs the base line while `fresh_ttft_sum_s` stays `<= 1.3x` the base line's and the C20 cached ramp does not regress; ties prefer the base line |
| `r1s-mpt8192.args`                      | R1s            | `--max-prefill-tokens 8192` (default 16384; interval stays 2)                                                    | same storm probe as above                                                                                                           | same rule                                                                                                                                                                    |
| `r8-delayer.args`                       | R8             | `--enable-prefill-delayer --prefill-delayer-max-delay-passes 4 --prefill-delayer-token-usage-low-watermark 0.05` | same storm probe as above                                                                                                           | same rule                                                                                                                                                                    |

**R2 is BLOCKED.** Measured on the box: booting `v3.args` plus `--tokenizer-worker-num 2`
starts the schedulers, then the HTTP server dies at FastAPI lifespan startup with
`sglang/srt/entrypoints/http_server.py:235 init_multi_tokenizer:
AssertionError: API key is not supported in multi-tokenizer mode`, the process exits, and
`watchdog.sh` relaunches the same args — a crash loop, ~6 min per cycle. `relaunch.sh` always
appends `--api-key` (the bearer is the direct door's only lock: Caddy is bearer-only and the
engine listens on loopback), so `--tokenizer-worker-num` above `1` cannot run here as SGLang
stands. `r2-tok2.args` and `r2-tok4.args` stay in the tree as the record of this rung; each
now starts with a `# BLOCKED:` line carrying the assertion text, and `check.sh` skips their shape
and forbidden-flag checks (counted, not checked) while still refusing any other file that sets
`--tokenizer-worker-num` to anything but `1`; the live-line check reads them like every other rung,
which exempts no blocked file. Their 8×B200 mirrors, `r2-tok2-b200.args` and
`r2-tok4-b200.args`, are the same rung on the other box and are blocked identically: each
carries the same `# BLOCKED:` first line, its tuning line swapped to `--tp-size 8 --ep-size 8`,
and neither is a working rung — do not boot either, and do not read a mirror as unblocking R2.

**R6 is BLOCKED as well, on a different kind of failure.** `r6-compact.args` and its 8×B200
mirror `r6-compact-b200.args` are the DSpark compact ragged-verify rung (paired with
`env-compact.txt`, `SGLANG_RAGGED_VERIFY_MODE=compact`), and the fault is upstream:
sglang issue #39173 miscounts the ragged-verify path itself, so this is an upstream-bug
block, **not** the crash-at-lifespan block R2's is — no boot of this pairing dies at
FastAPI lifespan startup, which is exactly why the block cannot be discovered by trying it.
Do not boot this pairing, on either box, before #39173 is fixed upstream. Both files now
carry their own `# BLOCKED:` first line naming #39173, so `check.sh` skips their shape and
forbidden-flag checks (counted, not checked) exactly as it does the R2 files, rather than
validating them as bootable rungs — the live-line check still reads them, as it reads every rung.

**A rung whose server dies at lifespan is a functional reject, not a measurement to
interpret** — restore the previous args file immediately: `cutover.sh <tag>-back <previous
args> - --tree <ref>`.

**R1's winner is carried forward by hand.** Whichever `--chunked-prefill-size` /
`--prefill-decode-interval` pair R1 picks (tie → 4096) becomes the pair every later rung
(R-LPM, R3, R-ADM, R-MOETP, R5, R6, R7) should actually boot with — every one of those files sits
on the base line's own `4096`/`2` today, because R1 has not run yet. Before booting any rung past
R1, the operator edits that rung's `--chunked-prefill-size` and `--prefill-decode-interval` values
to match R1's accepted pair; on a `-b200` file that is the same two-value edit against
`v4-b200.args`' line, and if R1's winner is promoted into `v4-b200.args` the `-b200` rungs are
re-based on it instead of each carrying it by hand. `r4-cp8192.args` is the one
exception: it always tests `8192`/`0.8` (`0.8` is the SGLang cookbook cell's mem fraction;
`8192` is this ladder's own value — the cookbook cell sets no `--chunked-prefill-size`; with
`--prefill-decode-interval` left at the base line's `2`), regardless of what R1 picked. R2 and
R6 are excluded from this list — both are blocked and do not boot at all. The same hand-carry
applies to the `-b200` mirrors, and against R1's own 8×B200 winner: the pair R1 accepts on 8×B200 is
not necessarily the pair it accepts on 4 GPUs (the chunk is the same size, the box's prefill rate is
not), so carry each box's winner into that box's rungs. R1 on 8×B200 accepted no candidate
(`../../08-measurements.md` §19), so the `-b200` rungs keep the served 4096×2 pair
(`--chunked-prefill-size 4096 --prefill-decode-interval 2`). The served line itself reads 1.47–1.62×
its N=0 TTFT there, above the `1.30×` bound; the R1 row's rule above is unchanged, and whether it
holds for 8×B200 is open (§19).

R-MOETP and R5 are mutually exclusive in practice: R-MOETP moves MoE traffic onto the
tensor-parallel group (`--ep-size 1`), so once it is accepted there is no expert-parallel
load left for EPLB to balance. Run R5 only if R-MOETP was declined.

## The 8×B200 mirrors

`--tp-size` / `--ep-size` live **in the args files**, not in the box file (the composition rule
above: a box is a topology + identity, an args file is a tuning line), so a rung does not move
to another box by editing the box file — it needs its own args file on that box. Every file in
the ladder above therefore has an 8×B200 twin in this directory, named with the suffix
**`X-b200.args`** — except the three storm rungs (`r1s-int4.args`, `r1s-mpt8192.args`,
`r8-delayer.args`), which are `--tp-size 4 --ep-size 4` lines with no `-b200` file. The 8×B200
ladder is maintained against the live line (`v4-b200.args`, which box-c serves plus one box-d flag)
rather than as a copy of the 4-GPU file: each
`-b200` rung's tuning line is `v4-b200.args`' own flag line plus that rung's own delta, so it
carries the live flag sets and every rung promoted into the live line too. The 4-GPU `X.args` file
stays the record of the rung as the retired 4-GPU box ran it (`v3.args` is kept only as the template
for provisioning a future 4-GPU box); do not read one off the other, and never "restore" a `-b200`
file to its 4-GPU twin's flags.

- `v4-b200.args` is the **8×B200 base line**: the line the rungs are measured against, the one
  `check.sh` and this README call **the live line**, and the flag line box-c serves minus one flag. box-c
  does not serve this file as it stands: it serves `v5-b200-box-d.args`, which is this flag line plus
  `--no-enable-decoder-swa-bounded-replay`, on the `box-d-up` tree (its bullet below). Paired with
  `env-v3.txt`, this file is the **old box-c line's** flag set, the official-settings line on tree
  `264ea0f9774b` (the fork's tag `archive/box-c-orig-thinkfix-20261020`; its branch is deleted, and no
  box runs that tree: every box file, the campaign boxes `gpu1018`/`gpu1018b` included, pins `prod`).
  box-c never returns to it: the fork's box-c line is retired and rollback is forward-only on
  `prod` (a swap's fallback is the model serving before it, on the same tree). Its flags are the
  ladder's accepted tuning set — `v3.args`'s own flags with `--tp-size 8 --ep-size 8` in place of
  its `4/4`, plus the R-TAILS / R-PRIO / R-MEM rungs promoted into it — **and** the live flag sets
  no rung measured: `--kv-events-config` / `--load-publish-endpoint` / `--metric-label-denylist`,
  the five `--short-prefill-*` and the HiCache L2 set (`--enable-hierarchical-cache` with the five
  `--hicache-*`). It carries no `--enable-strict-thinking` (its header says what dropping it gave
  up); the thinking default is `env-v3.txt`'s last two lines, `SGLANG_DEFAULT_THINKING=1` and
  `SGLANG_DSV41_REASONING_EFFORT=high`, so the engine's default is DeepSeek's own (thinking ON at
  `high` for a request that names no effort, `reasoning_effort: none` still switching it off), and
  neither file boots this line without the other — on a tree other than `box-d-up`, `stage.sh`'s seed
  and `restart.sh … default` both keep it once the box's staged copy is this file: `default`
  reseeds from `$TOOLS_REMOTE/args/env-v3.txt` there, which only `stage.sh` refreshes, so until
  `stage.sh` has run after an `env-v3.txt` change a `default` restart reseeds the old copy (a box
  staged before the thinking-default lines were added reseeds the strict env and drops the
  default) — run `stage.sh` first, or name the env file (`07-operations.md`'s restart recipe says
  the same). On the `box-d-up` tree box-c serves, the seed is `env-box-d-official.txt` instead (the
  `env-box-d-official.txt` bullet below). The strict-thinking line it replaced (this line plus
  `--enable-strict-thinking`, `env-v3.txt` without its last two lines, on tree `99a36a44`) is not
  kept in this directory and has no way back: rollback is forward-only on `prod`.
  No older line is kept here as a fallback: a change's fallback is what served before it — the
  previous args file and tree, which the change's own rollback anchor names — never an older
  last-known-good line. `v4-b200.args` is not a rung; it is the anchor every `-b200` rung's accept
  rule is measured against, exactly as `v3.args` is for the 4-GPU rungs.
- `v4-b200-official.args` + `env-v3-official.txt` are the **official-settings pair under the names
  the bundle staged and the `e2e-official-thinkfix` cutover installed**, kept as named anchors:
  dated records (`08-measurements.md`, the release gate's `1018-*` reports,
  `CHANGELOG.md`) cite them by these names, and `env-box-d-official.txt` is built on
  `env-v3-official.txt`. Both are now copies of the live pair: `v4-b200-official.args`' flag line is
  `v4-b200.args`' flag line exactly, and `env-v3-official.txt` is `env-v3.txt` byte for byte (16
  lines, md5 `7d92bc7580d52d3e5f538851da0093a6`). Neither is a rung: `v4-b200-official.args` is not
  a `-b200` mirror and is not measured against the live line. A change to the live pair is made in
  `v4-b200.args` / `env-v3.txt` and copied into these two in the same change, and `check.sh` fails
  until it is: the flag line must equal `v4-b200.args`' exactly (same flags, order, values, quoting;
  the headers may differ) and the env must equal `env-v3.txt` byte for byte (`## Test`), and a
  missing copy is a `FAIL` too — retiring the pair is a change to `check.sh` and this bullet. The
  same comparison by hand:
  `diff <(grep -v '^#' v4-b200.args) <(grep -v '^#' v4-b200-official.args) && cmp env-v3.txt env-v3-official.txt`;
  and a change to `env-v3-official.txt` is re-based into `env-box-d-official.txt` too (next bullet).
  Handing either name to `cutover.sh` / `restart.sh` / `bringup.sh` installs the same line.
  `../default-thinking-probe.py` is the pair's behavioural gate on the box (exit 0 = a no-effort
  request thinks at `high` with its reasoning in `reasoning_content`, `none` switches it off; exit 3 =
  the strict-thinking line, a leak or a broken off switch).
- `env-box-d-official.txt` is **box-d's serve env and the env box-c serves**, with `v5-b200-box-d.args` on
  the upstream-based `box-d-up` tree (the old `264ea0f9` line never reads it):
  `env-v3-official.txt` byte for byte plus exactly two lines, each under its own one-line comment.
  `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`: upstream defaults the radix tree core to `rust`,
  and the fork's `SGLANG_SWA_BRANCH_REBUILD=true` refuses to start without the Python TreeCore (the
  old line already defaults to `python`). `SGLANG_FORWARD_UNKNOWN_TOOLS=1`: after upstream #44956
  the DSv4.1 detector consumes a streamed call to an undeclared tool inside the engine, so nothing
  reaches the router, unless this is set; set, stream and non-stream both forward it as a structured
  call, which is box-c's streamed behaviour. It pairs with `v5-b200-box-d.args` (next bullet), which is
  `v4-b200-official.args`' flag line plus box-d's one args delta. `env-v3-official.txt` is never
  edited for box-d (it is box-c's official env, `env-v3.txt`'s copy): a change to it is re-based into
  this file in the same change, and `check.sh` fails until it is. **It is the `box-d-up` tree's
  seed.** `restart.sh`'s and `cutover.sh`'s `default` env argument, and `stage.sh`'s first-boot
  seed, write the box's `sglang-env.txt` from the box's staged copy of the env of the box file's
  tree, by `stage_env_seed_name` (one text in all three): `$TOOLS_REMOTE/args/env-box-d-official.txt`
  when `SGLANG_TREE`'s last component is `box-d-up`, `$TOOLS_REMOTE/args/env-v3.txt` otherwise.
  `env-v3.txt` is this file without `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` and
  `SGLANG_FORWARD_UNKNOWN_TOOLS=1`: on `box-d-up` the first is required
  (`SGLANG_SWA_BRANCH_REBUILD=true` refuses upstream's `rust` tree core), and without the second an
  undeclared-tool call is consumed inside the engine again, so `env-v3.txt` is never named on a
  box-d box. Only `stage.sh` refreshes the staged copy, so after a change here run `stage.sh` before a
  `default` restart, or name this file. No `default` restart has yet read this file back on box-c
  (`../../ISSUES.md`, the `default` env row), so the box-c recipes restart with `-` (keep the box's
  `sglang-env.txt`, which is this file) or name it (`tools/args/env-box-d-official.txt`).
- `v5-b200-box-d.args` is **box-d's serve line and the line box-c serves**, on the same `box-d-up` tree, and
  pairs with `env-box-d-official.txt` (`../../plans/box-d-cutover.md` §1 "args", prework P4). box-c runs it
  at `T_FINAL` `7b0a2e8d93` (the box file's `FORK_TREE`, checked out in
  `SGLANG_TREE=/workspace/box-d-up`, run from `VENV=/opt/sglang-up`): the box-d window (plan §6 steps
  2–11) installed the pair with `restart.sh box-d-7b0a2e8d93 tools/args/v5-b200-box-d.args
  tools/args/env-box-d-official.txt --tree 7b0a2e8d933dfab54e54772be95137e08fe0fe45`, so the box's
  `sglang-args.txt` / `sglang-env.txt` are these two files (`../../CHANGELOG.md`'s box-d window entry
  is the record). There is no way back to the pre-box-d pair: rollback is forward-only on `prod`, a
  swap's fallback being the model serving before it on this same tree and line
  (`../../07-operations.md` "Restart and cutover recipes"). Its flag line is `v4-b200.args`' flag
  line, string for string, with exactly one flag appended: `--no-enable-decoder-swa-bounded-replay`.
  Upstream turns decoder SWA bounded replay ON by default for DSv4, and without the exact decoder
  trim (box-c's later trim tree, `v10-b200-box-e-tp4.args`) that path is not numerically equivalent, so
  box-d switches it off by name. That is the same OFF state the `264ea0f9` line has by
  default (`enable_decoder_swa_bounded_replay=False` in both boot logs,
  the release gate's `1019-box-d-cold-ttft.md`), so this is not the forbidden flag above. This
  is the screen's N0 line: the kit's `N0-newbase.args` carries this exact flag line (that file, with
  its header, was md5 `ab02a787…`; the release gate's `1018-opt-screen.md`), and no screen
  knob is added. `check.sh` pins it (`## Test`): a live-line change is copied here in the same
  change, with the delta kept last, and any other difference is a `FAIL`. A missing file is a `FAIL`
  too, so retiring the line after the cutover is a change to `check.sh` and this bullet. It is not a
  `-b200` rung: the name does not end in `-b200.args`, so the mirror rule never reads it, and it is
  not measured against the live line. **The import gate:** the `--no-` form parses only on
  `box-d-up`. On the `264ea0f9` tree the flag is a plain `store_true`, and `sglang serve` rejects
  `--no-enable-decoder-swa-bounded-replay` as an unrecognized flag. So the file carries a header
  line `# TREE: box-d-up`. A default `check-import.sh` run on box-c reads the box file's
  `SGLANG_TREE=/workspace/box-d-up` and `VENV=/opt/sglang-up`, so its tree is `box-d-up` and it ships
  and parses this file with every other unblocked file: the default run gates the served line. A run against the
  old line's tree (`--repo /sgl-workspace/sglang --python /opt/sglang/bin/python3`, tree `sglang`)
  skips it: not shipped, not parsed, the skip named on stderr and counted in the OK line. The
  marker's grammar is `# TREE: <name>[,<name>...]` (names `[A-Za-z0-9._-]+`, comma- or
  space-separated, several lines a union; the run's tree is `--tree NAME`, else `--repo`'s last path
  component), defined in `check-import.sh`'s "TREE-PINNED FILES" header paragraph. `--all-trees`
  ships it anyway, and then the `264ea0f9` failure above is the expected result on that tree. The
  gate of record for this file is the default run, `BOX=box-c bash args/check-import.sh`,
  which is the window preflight's (plan §6 step 2) `--python /opt/sglang-up/bin/python3 --repo
  /workspace/box-d-up` with the box file supplying both. **The two lines are not folded into one
  file.** `v4-b200.args` stays the base this file is built from, and `check.sh` still pins this file
  to it, so a change to the flags box-c serves is made in `v4-b200.args` and copied here, delta last,
  in the same change. Folding them (this file becoming the live line, or `v4-b200.args` taking the
  delta) is a change to `check.sh`, this bullet and the `-official` bullet together.
- `v7-b200-box-e.args` + `env-box-e-w1.txt` are **box-c's TP8 line: its config of record from window 1,
  and the 2×TP4 line's rollback** (one engine, `restart.sh --single`; the `v8-b200-box-e-tp4.args`
  bullet below), on the `box-e-up` tree at `fd3545a4` (the box file's `FORK_TREE` until the copy tuning moved it to
  35e6d667b7a5, the v9 bullet below): the flag line of the window's shipped
  `P.args` and its shipped `final.env` (below the file's header), as the user ruled them the config of
  record. The record is `../../qa/window1-20261027/README.md`. `v7-b200-box-e.args`' flag line is
  `v6-b200-box-e.args`' with three values changed in place (`--chunked-prefill-size 4096 -> 8192`,
  `--prefill-decode-interval 2 -> 4`, `--hicache-ratio 2 -> 1.45`) and
  `--speculative-use-block-verification` appended, then `--enforce-disable-flashinfer-allreduce-fusion`
  (not a window-1 change: pinned since v7 became the TP8 rollback, because the tree auto-enables
  FlashInfer all-reduce fusion at TP8 once multicast binds, and every boot of this line ran with it
  off on box-c's multicast fault); `env-box-e-w1.txt`'s KEY=VALUE lines are
  `env-box-e.txt`'s then the engram host table pair (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`,
  `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared`) and the log-only `TRITON_PRINT_AUTOTUNING=1`, then
  `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` (not a window-1 change: appended so every box-c rollback keeps
  today's collective path after a provider reset that fixes multicast, and inherited by w2 and w3 through
  their pins on this file; a no-op while box-c's `cuMulticastBindMem` fails). `env-box-e-w1.txt` is 50
  lines: 22 KEY=VALUE lines under a 16-line comment header, the last one under its own 6-line comment;
  whole-file md5 `729a05b62734e51b5a68a8c6b4be1ba8` (`md5 -q env-box-e-w1.txt` / `md5sum
  env-box-e-w1.txt`; no secret). `../boxes/box-c.env`'s `ENV_SEED` made it the env seed over the tree name's
  `env-box-e.txt`, as box-a's box file does with `env-box-a.txt`; the key names `env-box-e-w2.txt` since
  the trim shipped and `env-box-e-w3.txt` since the #750 roll, and a rollback to this env sets
  `ENV_SEED=env-box-e-w1.txt` again. **The way back** is
  `v6-b200-box-e.args` + `env-box-e.txt` on the same tree, with the `ENV_SEED` line deleted in the same
  change. `check.sh`'s window-1 pin (`BOX_C_W1_ARGS_SUB`, `BOX_C_W1_ARGS_ADD`, `BOX_C_TP8_ARGS_ADD`,
  `BOX_C_W1_ENV_ADD`, then `BOX_C_MC_ENV_ADD`) holds the pair to the box-e pair (and the 2×TP4 pin drops the TP8 fusion flag
  again, `BOX_C_TP4_ARGS_SUB`'s `flag||` entry, so v8 and v9 stand as they booted): a box-e change not copied here, or an undeclared edit, is a `FAIL`. The args
  file carries `# TREE: box-e-up`, as v6 does.
- `v11-b200-box-e-tp4.args` + `env-box-e-w5.txt` are **what box-c serves, as 2×TP4**, on both copies since #797,
  on our SGLang fork `prod` 9a9ede7ae018 (4943d6ed90 + four Python-only picks) through the launch overlay
  `/workspace/box-c-prod-9a9ede7a` (installed env = `env-box-e-w5.txt` + its PYTHONPATH line,
  `../../qa/box-c-return-20261104/env-9a9e-w5.txt`; gates and boots in that dir's `NOTES.md`). `env-box-e-w5.txt` is
  `env-box-e-w4.txt` with `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS` 32 → 0 (on the 32-thread THP path box-c's boots
  went silent for up to 1012 s, about 17 min, at a stretch, under memory pressure that points at THP
  reclaim and compaction; the mechanism is partly inferred, `../../qa/box-c-return-20261104/NOTES.md`) and
  `SGLANG_SPEC_TP_SYNC=all` added (box-c's value pinned against upstream's default moving to `init`); its header
  says why. `check.sh` pins it to w4's derived line list with
  that one value swapped plus the sync line (`BOX_C_W5_ALLOC_SUB`, `BOX_C_W5_ENV_PLUS`), so a changed value, a dropped
  line or an extra one FAILs. The first way back, per copy, is v11 +
  `env-box-e-w4.txt` on 4943d6ed90, as the rest of this bullet describes.
- `v11-b200-box-e-tp4.args` + `env-box-e-w4.txt` were **what box-c served, as 2×TP4**, on both copies from the
  #96 roll until #797 (`../../CHANGELOG.md`, its #96 entry), on our SGLang fork `prod` 4943d6ed906a (now the box
  file's first `ROLLBACK_TREES` entry: 46067436 + the merge of upstream 3a3a8611 with box-a's megakernel-job parts + five sbr fixes + three
  upstream picks; the megakernel stays off, v11 passes no `--enable-nvidia-megakernel`). v11 is v10's flag line
  with one value changed: `--load-publish-endpoint tcp://127.0.0.1:5559` (was `tcp://*:5559`), so the load feed
  binds loopback like the KV events (relaunch.sh keeps the host and moves copy b to `127.0.0.1:5569`; the router
  reads both through its box-c-kv pod's ssh tunnel to the box's loopback). `check.sh` pins it to v10 by that one
  substitution (`BOX_C_V11_ARGS_SUB`) and exempts it from the bounded-replay ban as v10 is. The copies import the
  tree through the #96 launch overlay `/workspace/box-c-prod-4943d6ed` (installed env = `env-box-e-w4.txt` +
  `PYTHONPATH=/workspace/box-c-prod-4943d6ed/python`, `../../qa/prod-merge-20261030/raw/box-c-roll/env-4943.txt`),
  not config of record; the per-copy gates are in `../../qa/prod-merge-20261030/raw/box-c-roll/NOTES.md`.
  **The first way back**, per copy, is `v10-b200-box-e-tp4.args` + `env-box-e-w4.txt` on `prod`'s previous head
  46067436 through `/workspace/box-c-prod-4606` (the copy's `sglang-args/env-<id>.txt.pre-<tag>` pair, then
  `restart.sh --engine a|b <tag>-back - -`; `FORK_TREE=460674361992…` in the same change); the v10 bullet next
  lists the ways back behind it.
- `v10-b200-box-e-tp4.args` + `env-box-e-w3.txt` were **what box-c served, as 2×TP4**, on both copies
  (until the #96 roll; v10 + `env-box-e-w4.txt` on 46067436 is now the first per-copy rollback, above)
  (copy `a` and `b` as in the v8 bullet below) since the #750 roll
  (`../../CHANGELOG.md`, its #750 entry, has the timeline), on our SGLang fork `box-c-trim-latest`
  5f117c895756 (the box file's `FORK_BRANCH` / `FORK_TREE`), checked out **plain** at
  `/workspace/box-e-up` — no overlay. That tree is 601f5614 (#901's strict default on the trim line,
  from 070154af) + four cherry-picks (`-x`): 2c79f4a1 (upstream #45673, from 3454fd57), 6d60bdd5
  (upstream #43978, from bd73037c), bed13faa (upstream #46835, parallel host-pool allocation, from
  cc0b4344) and 5f117c89 (min_p validation, from 219c0feb); 35e6d667..5f117c89 is Python, tests and
  docs only, so the venv's built parts serve it without a rebuild. `env-box-e-w3.txt` (37 lines: 24
  KEY=VALUE lines under a 7-line comment header, w1's multicast switch under its own 6-line comment;
  whole-file md5 `63ff01231b781216845908ebfd6e90a1`, by
  `md5 -q env-box-e-w3.txt` / `md5sum env-box-e-w3.txt`; no secret) is `env-box-e-w2.txt`'s KEY=VALUE lines
  with `PYTHONPATH=/workspace/box-e-trim/python` removed (the trim is in `box-e-up`'s own checkout now)
  and `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32` appended (#46835's private prefaulted host-pool
  allocation; 0 or unset is the shared prefaulted mmap; boot's host-pool allocation 12 s / 8 s on
  copies b / a against 41 s before), keeping `SGLANG_DSV4_EXACT_DECODER_TRIM=1`; after the roll it
  gained w1's `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` among w1's lines (below; a no-op while box-c's
  `cuMulticastBindMem` fails). The roll left the args file unchanged. Gate per copy: smoke 10/10, short greedy 16/16 byte-equal (logprob diff 0), gsm8k 98/100,
  the DSML probe's T1 gate passing on 5f117c89 where the old line failed it (CHANGELOG). `check.sh`
  pins it as `env-box-e-w1.txt`'s KEY=VALUE lines then exactly `BOX_C_W3_ENV_ADD`, which it derives from
  w2's pin (`BOX_C_TRIM_ENV_ADD` minus the overlay's `PYTHONPATH` line, plus `BOX_C_W3_ENV_PLUS`'s
  `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32`), so the two pins cannot drift apart; the
  multicast switch comes with w1's lines.
  **All-reduce fusion is off on this line whether or not box-c's GPUs bind multicast**: the tree's
  FlashInfer fusion auto-enable skips DeepSeek-V4.1 at TP4 on one Blackwell node
  (`prefer_custom_dsv41` in our SGLang fork `python/sglang/srt/arg_groups/overrides.py`; the
  window-2 TP4 boots in `../../qa/window2-tp4/raw/` log `flashinfer_allreduce_fusion_backend` None
  and no FlashInfer workspace init, where the TP8 boots tried mnnvl and fell back on the multicast
  fault). The args file still appends `--enforce-disable-flashinfer-allreduce-fusion`
  (`check.sh`'s `BOX_C_TRIM_ARGS_ADD`): a no-op on 5f117c89 and 070154af, it keeps fusion off if a
  tree bump drops `prefer_custom_dsv41`. What a multicast fix would change here is the custom
  all-reduce's multicast band and the multimem logits all-gather, both keyed on torch symmetric
  memory's `multicast_ptr`; the env's `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` holds both on today's
  path (`/opt/sglang-box-e`'s torch 2.14.1+cu130 carries the string in `libtorch_cuda.so`), and
  `../../07-operations.md` "the provider GPU/host reset" has the read. Fusion or multicast on is #997's
  gated A/B: an explicit `--flashinfer-allreduce-fusion-backend` with the enforce flag dropped (it
  wins over the backend flag), and the env line dropped.
  **The env of record is `env-box-e-w4.txt`** (not `env-box-e-w3.txt`): w3's KEY=VALUE lines then
  `SGLANG_MM_CACHED_REFS=1` and `SGLANG_MM_CACHED_REFS_EMBEDDING_CACHE=0` (`check.sh`'s
  `BOX_C_W4_ENV_PLUS`; the pin FAILs a w4 that carries any `PYTHONPATH`), on the fork's `prod` at the box
  file's `FORK_TREE` (46067436 since #1346, a84f5742 from #1320, 9b262265 before it from #1161; all carry `box-c-trim-latest-mm-refs`
  bdc3f67fcb47 = `box-c-trim-latest` 5f117c89 + the four `mm_cached_refs` patches): the engine serves a
  repeated image from its cached multimodal refs, and the router omits those images' bytes for both
  copies. The copies run a launch overlay over `box-e-up` at 5f117c89, not config of record (both copies
  46067436 through #1346's `/workspace/box-c-prod-4606`; installed env `2a8e925b` = `env-box-e-w4.txt` +
  `PYTHONPATH=/workspace/box-c-prod-4606/python`): each copy's next restart is `restart.sh --engine a|b <tag>
  v10-b200-box-e-tp4.args env-box-e-w4.txt --tree <the box file's FORK_TREE>`, never a tree-less one.
  **The first way back**, per copy, is the same pair with `--tree a84f5742e5c79b4e1b51f3a042ad7a836e8e1166`
  (`prod`'s previous head; `FORK_TREE=a84f5742…` in the same change). **The second** is the same pair with
  `--tree 9b2622650cea3c94a09ef55465743801d84c2409` (`FORK_TREE=9b262265…` in the same change). **The third** is the same pair with
  `--tree bdc3f67fcb472b0319bb443f95ccb39bf7580aee` (`FORK_BRANCH=box-c-trim-latest-mm-refs`,
  `FORK_TREE=bdc3f67f…` in the same change), the cached-refs line before `prod`'s move. **The fourth** is `env-box-e-w3.txt` on `box-c-trim-latest` 5f117c89, the copy first
  taken off the router's image-bytes omission (`restart.sh --engine a|b <tag> v10-b200-box-e-tp4.args
  env-box-e-w3.txt --tree 5f117c895756ec7a8a95d159d4196a7abb576f5f`, with the box file's
  `FORK_BRANCH=box-c-trim-latest`, `FORK_TREE=5f117c89…` and `ENV_SEED=env-box-e-w3.txt` in the same change).
  **The fifth way back** is the line w3 replaced: `v10-b200-box-e-tp4.args` +
  `env-box-e-w2.txt` on `box-c-trim-46700` 070154aff514 through the `/workspace/box-e-trim` overlay
  (`restart.sh --engine a|b <tag> v10-b200-box-e-tp4.args env-box-e-w2.txt`, no `--tree`: the overlay's
  `PYTHONPATH` wins over `box-e-up`'s 5f117c89; the box file's `FORK_BRANCH=box-c-trim-46700`,
  `FORK_TREE=070154aff514…` and `ENV_SEED=env-box-e-w2.txt` in the same change; first
  `git -C /workspace/box-e-trim rev-parse HEAD` must print 070154af… with a clean status). Then the v9
  rollback below. The Caddy swap the roll made has its own, box-wide rollback
  (`../caddy-h2/README.md`, the box file's `CADDY_EXPECT_SHA256` comment).
- `v10-b200-box-e-tp4.args` + `env-box-e-w2.txt` were **what box-c served, as 2×TP4**, on both copies
  from the trim's ship until the #750 roll, and are now its fifth per-copy rollback (above), on
  our SGLang fork `box-c-trim-46700` 070154aff514
  (fd3545a4 → the exact decoder trim commits (… a0a5a184ab)
  → three fork-CI commits → d2cfddf9, the DSV4.1 image-order fix picked onto them (so the tree holds
  35e6d667b7a5's code) → upstream #46700 3953ff82 → the plain late-layer tail at graph replay 070154af;
  not on `prod`). It
  is #732's arm XT3: `v9-b200-box-e-tp4.args`' flag line with one boolean flipped in
  place, `--no-enable-decoder-swa-bounded-replay` → `--enable-decoder-swa-bounded-replay`, plus
  the later `--enforce-disable-flashinfer-allreduce-fusion` appended (a no-op on 070154af and
  5f117c89, the v10 + w3 bullet above), nothing else. `env-box-e-w2.txt` (52 lines: 24 KEY=VALUE lines under a 15-line comment header;
  whole-file md5 `0206bd62af0e8d5c4f4713b6fe87fe8c`, by `md5 -q env-box-e-w2.txt`; no secret) is
  `env-box-e-w1.txt`'s KEY=VALUE lines (w1's multicast switch, appended after the trim shipped,
  included) then two appended:
  `PYTHONPATH=/workspace/box-e-trim/python` (the trim overlay: the copies imported sglang from box-c's
  `/workspace/box-e-trim` checkout, which must hold 070154af, because `/workspace/box-e-up` then sat at
  35e6d667; a rollback to this env keeps importing the overlay whatever `box-e-up` holds) and
  `SGLANG_DSV4_EXACT_DECODER_TRIM=1` (what makes bounded replay exact; inert without it, since the
  fork's `exact_trim_enabled` turns the trim on only with bounded replay on, so this env on v9, v8 or
  v7 — a re-create's first onstart boot is v7 on it — runs untrimmed). Gate on copy b, both copies'
  eq: short greedy 16/16 byte-equal vs SD and SD2, short TF exact, needle 9/9, gsm8k 98/98/97, images
  15/15, gates.py 0/19 regressions; fill 1.45 → 1.03 s, partial re-prefill 0.94–1.24 → 0.65–0.78 s,
  cold ~100k 2.14 → 1.41 s; non-logprob capped trim hits 0.95% / 1.37% (a / b), under the 5% bar
  (`../../CHANGELOG.md` is the record). This is box-c's one file `check.sh` lets carry bounded replay
  (`FORBIDDEN_EXEMPT`; box-a's is `stack-b300-box-a-mk2.args`); its trim pin (`BOX_C_TRIM_ARGS_SUB`, `BOX_C_TRIM_ARGS_ADD`, `BOX_C_TRIM_ENV_ADD`) holds the args to v9's
  with only that flip and that flag and the env to `env-box-e-w1.txt`'s with only those two lines, and the 2×TP4
  property asserts run on it as on v8 and v9. `TREE: box-e-up`. **The second way back**, per copy, is
  `v9-b200-box-e-tp4.args` + `env-box-e-w1.txt` on the fork's old `prod` 819756d4632e (`archive/prod-819756d4-20261029`) (= 35e6d667 + the SWA
  interior-evict path 9e295882, off because `SGLANG_SWA_EVICT_INTERIOR_FIRST` is unset in
  `env-box-e-w1.txt`, + upstream #46700 e7c55f69, which has no env gate and always runs), so a rollback keeps the KV-corruption fix
  (`restart.sh --engine a|b <tag> v9-b200-box-e-tp4.args env-box-e-w1.txt --tree
  819756d4632e9a1f07234e97c11137a591a79193`, the box file's `FORK_BRANCH=prod`,
  `FORK_TREE=819756d4632e…` and `ENV_SEED=env-box-e-w1.txt` in the same change; first
  `git -C /workspace/box-e-up cat-file -t 819756d4632e9a1f07234e97c11137a591a79193` must print
  `commit`, else fetch fork `prod` or the `prod-819756d4.bundle` into that checkout). 819756d4 is the
  tree box-a serves and has not been booted on box-c: only if it fails to boot there is the same pair
  on 35e6d667b7a5 (`box-e-img-order`, no #46700) the fallback. Then v8 on fd3545a4; the TP8 rollback
  stays `v7-b200-box-e.args` + `env-box-e-w1.txt`.
- `v9-b200-box-e-tp4.args` + `env-box-e-w1.txt` were **what box-c served as 2×TP4** from the copy tuning
  until the trim shipped, on both copies
  (copy `a` and `b` as in the v8 bullet below: same GPUs, ports, doors, rows and installed names), then
  on `box-e-up` at our SGLang fork `box-e-img-order` 35e6d667b7a5 (fd3545a4 + the DSV4.1 image-order
  fix). They are now v10's sixth per-copy rollback (after v10 + `env-box-e-w2.txt`), on the fork's `prod` 819756d4632e (that tree + #46700's clamp,
  always on + the SWA interior-evict path, env-off; the v10 bullet above has the command and the box-file flips), with
  35e6d667b7a5 only the fallback if 819756d4 fails to boot on box-c. It is the copy tuning's config **SD** (#387,
  `../../qa/window2-tp4/tuning-387/README.md`): `v8-b200-box-e-tp4.args`' flag line with
  `--hicache-write-policy write_through -> write_through_selective` and three appended, the breakable
  prefill CUDA graph (`--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2048
  --cuda-graph-prefill-max-context 131072`, box-a's stack flags). TTFT C1..C16 146/181/272/308/327 →
  40/60/70/90/111 ms, warm 70k extend 0.26 → 0.15 s, resume 267 → 152 ms, box total +10–17% at
  C4–C16; gates.py exit 0, 0/38 regressions. Not output-identical to v8 on short prompts
  (teacher-forced short top-1 96.28%; long TF in band; gsm8k 98; L3a slice PASS). The env is
  unchanged (`env-box-e-w1.txt`, byte-equal to both copies' live env). The kv-events ports, the absent
  `--nccl-port` and the `TREE: box-e-up` pin follow v8's convention. **The way back**, per copy, is
  `v8-b200-box-e-tp4.args` + `env-box-e-w1.txt` on tree fd3545a4 (`restart.sh --engine a|b <tag>
  v8-b200-box-e-tp4.args env-box-e-w1.txt --tree fd3545a46b53df3b01a4ed1b01a460d29f442385`); the TP8
  rollback stays `v7-b200-box-e.args` + `env-box-e-w1.txt`. `check.sh`'s tuning pin
  (`BOX_C_T387_ARGS_SUB`, `BOX_C_T387_ARGS_ADD`) holds the line to v8's with only those deltas, and the
  2×TP4 property asserts below run on v9 as on v8.
- `v8-b200-box-e-tp4.args` + `env-box-e-w1.txt` were **what box-c served as 2×TP4** from window 2 until
  the copy tuning, and are v9's per-copy rollback, on both copies:
  - copy `a`: GPUs 0-3, engine `:8002`, `--nccl-port 8102`, kv events / replay / load publish
    5557-5559, door `box-ca.<DOMAIN>`, router row `box-c-a`, installed as
    `sglang-args-a.txt` / `sglang-env-a.txt`;
  - copy `b`: GPUs 4-7, engine `:8012`, `--nccl-port 8112`, 5567-5569, door
    `box-cb.<DOMAIN>`, router row `box-c-b`, installed as `sglang-args-b.txt` /
    `sglang-env-b.txt`.

  It is the one line both of box-c's TP4 engines boot, one copy per four GPUs, under
  `../boxes/box-c.env`'s `ENGINES="a:0,1,2,3:8002:5557 b:4,5,6,7:8012:5567"`
  (`../README.md` "Engines (2×TP4)"). Its flag line is `v7-b200-box-e.args`' with three values
  changed in place and nothing added or dropped: `--tp-size 8 -> 4`,
  `--ep-size 8 -> 4` and `--hicache-ratio 1.45 -> 2.0` (a host-RAM fit for two copies — 191.35 GiB
  of host cache per copy per ratio unit against the 1532.9 GiB gate, 80% of the container's
  `memory.max` — not a speed knob). It pairs with `env-box-e-w1.txt` (the box file's `ENV_SEED`) on
  the same `box-e-up` tree, venv and cache as v7, and carries `# TREE: box-e-up`. One file serves both
  copies: its `--kv-events-config` names copy a's ports, 5557/5558, exactly, and it carries no
  `--nccl-port`; `relaunch.sh` maps each copy's kv ports (`kv_base`, `+1`), its `--nccl-port` (engine
  port + 100: 8102 / 8112) and its `CUDA_VISIBLE_DEVICES` from the box file's `ENGINES` list.
  **The way back** is `v7-b200-box-e.args` + `env-box-e-w1.txt` as one TP8 engine (`restart.sh
  --single`). `check.sh`'s 2×TP4 pin (`BOX_C_TP4_ARGS_SUB`) holds the line to v7's with only those three
  substitutions, and asserts by name `--tp-size 4`, `--ep-size 4`, `--hicache-ratio 2.0`, the
  5557/5558 `--kv-events-config`, `--speculative-use-block-verification` and no `--nccl-port`: a v7
  change not copied here, an undeclared edit or a missing file is a `FAIL`. The name does not end
  `-b200.args` on purpose: the live-line mirror rule claims every `*-b200.args`, and this file is not
  a rung and not a mirror.
- `stack-b300-box-a-mk2.args` + `env-box-a-mk3.txt` are **what box-a serves** since #959 (box-a back on
  our SGLang fork `prod` `163591d9fc`; the box file's `ENV_SEED=env-box-a-mk3.txt`). The env is
  `env-box-a-mk2.txt` with ONE line appended, `PYTHONPATH=/workspace/site-mk3`: the box's copy of
  `site-mk3/sitecustomize.py` (this directory), first on every engine interpreter's path. It calls
  `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`, so `py-spy dump --nonblocking` attaches to a live rank with no
  restart, and registers a `gc.callbacks` timer that logs only gen2 collections over 0.1 s (UTC time, duration,
  counts, pid) to the serve log and `/workspace/gclog/<pid>.log`: content-free, no SIGUSR2 handler, and the
  engine's imports unchanged (the dir holds only `sitecustomize.py`, which then runs the system one it
  shadows). It is the live check `../../qa/eviction-stall-20261105/NOTES.md` names: a slow eviction pass with
  no gen2 collection beside it is the case for `prod-dleaf-count-20261105` (`../../SPEED-REGISTER.md`).
  **The way back** is `env-box-a-mk2.txt` (the same engine without the site dir). The mk2 pin below does not
  read this file.
- `stack-b300-box-a-mk2.args` + `env-box-a-mk2.txt` are **what box-b serves** (its box file's
  `ENV_SEED=env-box-a-mk2.txt`, box-a's seed byte for byte) and box-a's way back from mk3; they were what box-a
  served from 2026-10-30 12:51Z (#84; `../../qa/prod-merge-20261030/raw/gate/NOTES.md`): config X' — the
  megakernel line on our SGLang fork `prod` 4943d6ed90 (the merge of `box-a-mk-latest` into prod + follow-up patches +
  upstream picks #46645, #38565, #47149) in `/opt/sglang-mk29`, with box-c's wins (the box file's
  `FORK_BRANCH=prod`, `FORK_TREE=4943d6ed90…`, `ENV_SEED=env-box-a-mk2.txt`; venv, plugin and cache
  unchanged). The flag line is `stack-b300-box-a-mk.args`' with two values changed in place:
  `--hicache-write-policy write_through` → `write_through_selective` (box-c's) and
  `--no-enable-decoder-swa-bounded-replay` → `--enable-decoder-swa-bounded-replay` (the path the exact
  trim rides). The env is `env-box-a-mk.txt`'s with four lines inserted before the TP probe block:
  `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32` (host-pool alloc 117 → 25 s),
  `SGLANG_DSV4_EXACT_DECODER_TRIM=1`, `SGLANG_MM_CACHED_REFS=1` and
  `SGLANG_MM_CACHED_REFS_EMBEDDING_CACHE=0` (the engine side of the router's cached image refs); the
  probe stays on. Gate vs S7: idle greedy 32/32, tf short 100%, KL 0.0046 / 0.0092, DSML 1,880 sends
  recorded (defects 4/470, 4/470, 3/471, 3/470 across T1 stream/nonstream and strict:false stream/nonstream;
  the two-sample compare vs S7 found nothing worse), cold fill 60k 0.989 vs 1.435 s. It is a trim line,
  so `check.sh`'s `FORBIDDEN_EXEMPT` lets it carry
  bounded replay, and its box-a mk2 pin holds the args to the mk line with only box-c's two substitutions
  (`BOX_C_T387_ARGS_SUB`, `BOX_C_TRIM_ARGS_SUB`) and the env to `env-box-a-mk.txt`'s with exactly
  `BOX_A_MK2_ENV_ADD` (the trim among them) before the probe lines. `TREE: sglang`. **The way back** is
  the mk pair below on 9ec65dacaf (`restart.sh --tree 9ec65dacaf` + `stack-b300-box-a-mk.args` +
  `env-box-a-mk.txt`, and the box file's `FORK_BRANCH=box-a-mk-latest`, `FORK_TREE=9ec65dacaf52…`,
  `ENV_SEED=env-box-a-mk.txt` in the same change).
- `stack-b300-box-a-mk.args` + `env-box-a-mk.txt` are **config X, box-a's rollback**; they were what box-a
  served from its return to the router pools (#39; `qa/box-a-megakernel-20261029/raw/box-a-return/NOTES.md`)
  until config X': config X of the megakernel retry,
  on our SGLang fork `box-a-mk-latest` 9ec65dacaf52 in `/opt/sglang-mk29` (the box file's `VENV`,
  `SGLANG_TREE`, `FORK_BRANCH` / `FORK_TREE`; `ENV_SEED=env-box-a-mk.txt`). The flag line is
  `qa/box-a-megakernel-20261029/args/mk-latest.args`' byte for byte: `stack-b300-box-a.args`' with
  `--ep-size 4` → 1, `--load-publish-endpoint` onto loopback, `--hicache-io-backend direct
  --hicache-mem-layout page_first_direct`, and `--enable-nvidia-megakernel` appended (KV pool 49.9M).
  The env is `mk-latest.env` (`env-box-a-w1.txt` with the engram layout `per_rank`,
  `SGLANG_SPEC_TP_SYNC`, PATH/XDG_CACHE_HOME onto the mk29 venv/cache) + the TP-consistency probe
  (`SGLANG_MK_TP_PROBE=1`, `_EVERY=100`, `_ASSERT=1`). `TREE: sglang` (the basename of
  `/workspace/mk29/sglang`); no `check.sh` pin of its own beyond the shape, forbidden-flag and deploy-gate
  rules (it is the base the box-a mk2 pin holds config X' to). Behind it, the way back is the pair below on
  819756d4.
- `stack-b300-box-a.args` + `env-box-a-w1.txt` are **box-a's final** and the megakernel pair's rollback; they
  were what box-a served from its ship-1 run (#385,
  `qa/box-a-ship1-20261027/README.md`; the user's call, 22:35Z: ship the stack, not A0), last on the
  fork's `prod` 819756d4632e (= `box-a-ship1` ace628a0e9d2 — prod 452f2438 + box-e-img-order +
  box-e-swa-interior-evict, the latter's env off — + upstream #46700's SWA tombstone clamp; the box
  file's rollback comments), with `box-a-ship1` ace628a0 on the same args/env as its way
  back. The flag line is `v2-b300-box-a.args`' with
  `--swa-prefix-tails` 4096 → 2048 (box-c copy P's; the KV pool went 55.4M → 66.7M tokens) and four
  appended: the breakable prefill CUDA graph (`--cuda-graph-backend-prefill breakable
  --cuda-graph-max-bs-prefill 2048 --cuda-graph-prefill-max-context 131072`, 3.36 GB graph pool; TTFT
  −66 to −74% at C1–C16) and `--flashinfer-allreduce-fusion-backend trtllm`. gates.py (fixed rule):
  exit 0, 0 regressions; gsm8k 98/100 on three boots; greedy NOT output-identical to A0 (28/32,
  near-tie splits). The env is byte-identical to the FINAL boot's. `TREE: box-e-up`; no `check.sh` pin
  beyond the shape and forbidden-flag rules. Behind ace628a0, the way back is `v1-b300-box-a.args` +
  `env-box-a.txt` on fd3545a4.
- `v2-b300-box-a.args` + `env-box-a-w1.txt` were **box-a's candidate after window 1**; v2 never served
  (the ship-1 run booted `stack-b300-box-a.args` on top of it instead): the box-a pair plus
  only the window's exact or sampled-gated winners that apply to a TP4 box and that box-a lacks —
  `--speculative-use-block-verification` appended (its sampled gate passed on two box-c boots, and it ran
  at TP4 on box-c's GPUs 0-3; unmeasured on B300) and the log-only `TRITON_PRINT_AUTOTUNING=1`. Chunk
  8192, interval 4 and the engram host table are already box-a's; `--hicache-ratio 1.45` is box-c's
  host-RAM fit; exact decoder trim, bounded replay and engram peer shards were not adopted. `check.sh`
  pins the pair to the box-a pair (`BOX_A_W1_ARGS_ADD`, `BOX_A_W1_ENV_ADD`). A box-a window boots it and
  re-runs the sampled gate on the box before its box file points at it.
- **The files here have to stay equal to the line box-c serves.** (That is `v11-b200-box-e-tp4.args` +
  `env-box-e-w4.txt` on both copies on `prod` 4943d6ed906a, each installed as
  `sglang-args-<id>.txt` / `sglang-env-<id>.txt`, with the per-copy rollbacks in the box file's
  `ROLLBACK_TREES` order: `v10-b200-box-e-tp4.args` + `env-box-e-w4.txt` on 46067436, the same pair on a84f5742,
  9b262265 and bdc3f67f, `env-box-e-w3.txt` on
  5f117c89, `env-box-e-w2.txt` over the 070154af overlay, then `v9-b200-box-e-tp4.args` +
  `env-box-e-w1.txt` on `prod` 819756d4 (35e6d667 only if 819756d4 fails to boot),
  `v8-b200-box-e-tp4.args` on fd3545a4 behind it, and
  `v7-b200-box-e.args` as its TP8 rollback — the bullets above; the rest
  of this bullet is the box-d-era text.) box-c served `v5-b200-box-d.args` +
  `env-box-d-official.txt`, and `check.sh` pins those to `v4-b200.args` plus the one box-d flag and to
  `env-v3.txt` (through `env-v3-official.txt`) plus the two box-d lines, so the served line is still
  recorded through `v4-b200.args` and `env-v3.txt`. A cutover or bring-up stages the args file it is
  handed to the box as `sglang-args.txt`, so a live flag missing here is silently dropped by the next
  cutover, and a flag the box no longer serves is silently put back. The same holds for the env, which
  `stage.sh` seeds and `restart.sh … default` reseeds from the box's staged copy of the env of its
  tree — `env-box-d-official.txt` on box-c's `box-d-up` (the `env-box-d-official.txt` bullet), so a
  change to the served env that is not re-based into that file is dropped by the next `default`
  restart. What box-c serves lands in this directory in the same batch: its
  flags in `v4-b200.args` and copied into `v5-b200-box-d.args`, its env in `env-v3.txt` and copied into
  `env-v3-official.txt` and `env-box-d-official.txt`, with every `-b200` rung mirror re-based in the
  same change (`check.sh` compares the mirrors to `v4-b200.args`) and the `-official` and box-d
  copies above kept equal (`check.sh` compares them too); a candidate the box is not serving belongs
  in a rung file, never here. A `-b200` rung booted on box-c as it stands runs on `box-d-up` without
  `--no-enable-decoder-swa-bounded-replay`, so upstream's bounded-replay default is ON for it and it
  is one flag off the served line: append that flag to the rung's line before booting it there (an
  added flag is the rung's own, which `check.sh` does not compare; the `--no-` form parses only on
  `box-d-up`, so the rung then needs the `# TREE: box-d-up` marker too, as `v5-b200-box-d.args`
  carries).
- Every `-b200` rung file is **on the live line**: its tuning line is `v4-b200.args`' own flag line
  plus that rung's delta and nothing else, so booting one, or a `cutover.sh` to it, cannot drop a
  `v4-b200.args` flag. It does drop the box-d flag: `--no-enable-decoder-swa-bounded-replay`, the
  one flag box-c serves beyond `v4-b200.args` (`v5-b200-box-d.args`), is on no rung's line but the R1
  ladder's four — `n0-b200.args` and the three `r1-*-b200.args` carry it appended last, under the
  `# TREE: box-d-up` marker, so each is box-c's served line plus its own delta and nothing else — and any
  other rung booted on box-c as it stands serves with bounded replay ON unless that flag is appended
  first (the bullet above). The rule that keeps the rungs on `v4-b200.args`' line: any change to the live line
  — a flag added, a rung promoted — is followed by re-basing every `-b200` rung file in the same
  change, each rung's own delta preserved exactly. A rung whose delta was promoted is left with no
  delta: `r-tails-b200.args`, `r-prio-b200.args`, `r-mem070-b200.args` and `r7-hicache-b200.args`
  are `v4-b200.args` verbatim, kept as those rungs' record. The failure this rule exists to prevent
  is the pre-rebase state, where a rung still held a superseded base line's flags —
  `r-tails-b200.args` carried `--mem-fraction-static 0.75` and none of R-PRIO, and a cutover to it
  served without the HiCache L2, the short-prefill budget and `--kv-events-*`.
- **`check.sh` asserts that rule** (`## Test`), and **the loop below is its by-hand diagnostic**:
  run it in this directory, in bash, after any live-line change, to see every differing flag at once
  and the number of mirrors the run read. `OFF-LINE <file> missing: …` names live flags the file
  does not set at all — always the failure above, since a boot or a `cutover.sh` to it would serve
  without them. `differs <file>: …` lists live flags that file carries at another value — the value
  shown is the live line's, `grep '^--' <file>` prints the file's own line — which is legitimate only
  where that rung's ladder row declares the change (R-ADM's `128` pair, R-MOETP's `--ep-size 1`,
  `n0-b200.args`' `--prefill-decode-interval 0`, the three R1 files, R4's chunk `8192`); anywhere
  else it is the same drift — a live flag added, promoted or re-valued, and a rung not re-based with
  it — and `check.sh` fails on it: the declared deltas are the script's fixed `DECLARED_DELTAS`
  table, and a live flag a rung sets at a value outside it is a `FAIL` naming the file.
  `check.sh` compares a flag's whole value run, so it also catches an edit inside the one live value
  with two tokens, `--metric-label-denylist priority model_name`, which the `pairs` comparison above
  would not show. Only `v4-b200.args` itself is skipped — every other `-b200` file is a rung; the
  three blocked R2/R6 mirrors are **not** exempt — a blocked file is still a rung and stays on the
  live line, and `check.sh` reads them there too. Against the pre-rebase tree the loop printed
  `OFF-LINE` for all 18 rung mirrors; on the line as it stands, none. Its closing line counts the
  mirrors it read — `0 rung mirrors` means it was run outside this directory, not that the rule
  holds.

  ```bash
  tune()  { grep -vE '^[[:space:]]*(#|$)' "$1" | head -n 1; }
  names() { tune "$1" | grep -oE -- '--[a-z0-9-]+' | sort -u; }
  pairs() { awk '{for(i=1;i<=NF;i++) if($i ~ /^--/){n=$i;v=""; if(i<NF && $(i+1) !~ /^--/){v=" "$(i+1);i++} print n v}}' < <(tune "$1") | sort -u; }
  live=$(names v4-b200.args); livep=$(pairs v4-b200.args); seen=0
  for f in $(command ls -1 | grep -E -- '-b200[.]args$'); do
    case $f in v4-b200.args) continue;; esac   # the live line itself
    seen=$((seen + 1))
    off=$(comm -23 <(echo "$live") <(names "$f") | tr '\n' ' ')
    dif=$(comm -23 <(echo "$livep") <(pairs "$f") | tr '\n' ' ')
    [ -n "$off" ] && echo "OFF-LINE $f missing: $off"
    [ -n "$dif" ] && echo "differs $f: $dif"
  done
  echo "$seen rung mirrors checked against v4-b200.args"
  ```
- **The 2048 rungs carry a halved short-prefill block.** box-d-up's
  `validate_short_prefill_protection` (`python/sglang/srt/arg_groups/validation_hook.py`) ties the
  live line's short-prefill block to `--chunked-prefill-size`: max tokens ≤ chunk, short chunk <
  max tokens, threshold ≤ max tokens − short chunk, threshold < swap limit ≤ 4 × chunk. The live
  block (threshold 3072, short chunk 1024, max tokens 4096, lookahead 8, swap limit 16384) is built
  for chunk 4096 and is refused at startup at chunk 2048 (`ValueError: short_prefill_max_tokens
  cannot exceed chunked_prefill_size`). So `r1-2048x4-b200.args` and `r1-2048x2-b200.args` carry
  that block at the live line's ratios, halved: **1536 / 512 / 2048 / 8192**, lookahead 8
  unchanged — the line both booted with in the R1 ladder window, and four more `DECLARED_DELTAS`
  entries each in `check.sh`. The cost is part of what those rungs measure: **a prompt of 1,536 to
  3,072 tokens is short-prefill protected on the live line and not on a 2048 rung.** The 4-GPU
  `r1-2048x{4,2}.args` are `v3.args`-based and carry no short-prefill block, so they need no change.
- Only the referents move in a mirror: its base line and its accept rule read against
  `v4-b200.args` instead of `v3.args`, and its N=0 point is `n0-b200.args` instead of `n0.args`.
  Its `env-*.txt` pairing, where the twin has one (`env-v3.txt` for the base lines,
  `env-compact.txt` for R6), is unchanged.
- The duty-cycle arithmetic above and every accept rule stated in seconds or tok/s are per-box:
  an R1 mirror's `c` / `prefill_tok_s` are the 8×B200 box's own measurements, never the 4-GPU
  numbers carried over.

| 4-GPU file        | 8×B200 mirror          | Note                                                                                                    |
| ----------------- | ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `v3.args`         | `v4-b200.args`         | the live line (box-c serves it + one flag as `v5-b200-box-d.args`) — `v3.args` for 8×B200; not a rung       |
| `n0.args`         | `n0-b200.args`         | N=0 reference for the 8×B200 ladder                                                                     |
| `r1-4096x4.args`  | `r1-4096x4-b200.args`  | R1 on 8×B200; tie still prefers 4096                                                                    |
| `r1-2048x4.args`  | `r1-2048x4-b200.args`  | R1 on 8×B200; short-prefill block halved for chunk 2048 (below)                                         |
| `r1-2048x2.args`  | `r1-2048x2-b200.args`  | R1 on 8×B200; short-prefill block halved for chunk 2048 (below)                                         |
| `r2-tok2.args`    | `r2-tok2-b200.args`    | **BLOCKED** — same first line, same reason                                                              |
| `r2-tok4.args`    | `r2-tok4-b200.args`    | **BLOCKED** — same first line, same reason                                                              |
| `lpm.args`        | `lpm-b200.args`        | **BLOCKED** on box-d-up — `# BLOCKED:` first line: `check_server_args` refuses `lpm` with priority on    |
| `r3-mfs1.args`    | `r3-mfs1-b200.args`    | R3 on 8×B200                                                                                            |
| `r3-mfs8.args`    | `r3-mfs8-b200.args`    | R3 on 8×B200                                                                                            |
| `adm128.args`     | `adm128-b200.args`     | R-ADM on 8×B200                                                                                         |
| `moetp.args`      | `moetp-b200.args`      | **keeps `--ep-size 1`** — see below                                                                     |
| `r4-cp8192.args`  | `r4-cp8192-b200.args`  | R4 on 8×B200; 8192 at the live line's `0.70`, not the cookbook cell's `0.8` — **REJECTED**, see below   |
| `r5-eplb.args`    | `r5-eplb-b200.args`    | R5 on 8×B200; still only if R-MOETP was declined                                                        |
| `r6-compact.args` | `r6-compact-b200.args` | **BLOCKED** — same `# BLOCKED:` first line, same upstream reason (sglang#39173); counted, not checked   |
| `r7-hicache.args` | `r7-hicache-b200.args` | R7 on 8×B200; its L2 set is live in `v4-b200.args`, so this file is `v4-b200.args` verbatim             |
| — (no 4-GPU twin) | `r-tails-b200.args`    | R-TAILS — `--swa-prefix-tails 2048`; **ACCEPTED**, promoted into `v4-b200.args` (now verbatim)          |
| — (no 4-GPU twin) | `r-prio-b200.args`     | R-PRIO — priority scheduling + bounded queue; **ACCEPTED**, promoted into `v4-b200.args` (now verbatim) |
| — (no 4-GPU twin) | `r-mem070-b200.args`   | R-MEM — `--mem-fraction-static 0.70`; **ACCEPTED**, promoted into `v4-b200.args` (now verbatim)         |

**R-TAILS, R-PRIO and R-MEM are 8×B200-only rungs; none has a 4-GPU twin** — the SWA prefix-cache
capacity behaviour and the storm-scale memory pressure they tune only bite at this box's scale of
concurrent long conversations and concurrent cold prefills.
`r-tails-b200.args` was, at its measurement, `v4-b200.args` plus `--swa-prefix-tails 2048`
(default 256 = 4 ×
`max_running_requests`; the DSV4 pool configurator's `_resolve_swa_prefix_tails`,
`python/sglang/srt/model_executor/pool_configurator.py`, sizes the SWA KV pool in "cap" mode to
hold room for only this many radix-cached prefix tails). Measures: `tails-probe.py` (40
conversations × 4 turns, ≥95% hit on turns 2–4, 0 zero-cached), `probe-1m.py` 4/4, and the boot
log's `DSV4 SWA sizing … prefix_tails=2048` with `full_token >= 25,000,000`. Accept → promote
into `v4-b200.args`; reject → try 1024. **Result: ACCEPTED** (READY in 124 s;
`full_token=31,994,112 swa_tokens=869,376 prefix_tails=2048`): 1, 8 and 16 conversations × 275k
tokens all warm (hit ratio ≈ 0.9997), 40 conversations × 25k tokens all warm; but 24
conversations × 275k left only 19 warm (5 zero-cached), and 40 conversations × 275k came back
0/40 warm — an LRU-cyclic thrash once the working set exceeds the tail budget. 2048 tails raises
the resting line's roughly 4-long-conversation capacity to roughly 19–24, it does not remove the
ceiling; see `ISSUES.md`, "cache-capacity knee on the 2048 line", and `08-measurements.md` §10.
Now promoted into `v4-b200.args` as the 8×B200 resting line.

`r-prio-b200.args` was, at its measurement, `v4-b200.args` (carrying forward R-TAILS's winning
`--swa-prefix-tails`)
plus `--enable-priority-scheduling --disable-priority-preemption --default-priority-value 1
--max-queued-requests 512` — the router sends `priority: 1` on a light or cache-warm turn and `-1` on
a heavy cache-cold one (`06-priority.md`),
so the scheduler's fcfs sort orders the waiting queue by that field ahead of arrival order,
behind a bounded 512-request queue that fails fast (503) instead of holding a caller through the
whole 300 s wait. Measures: `smoke.py`, `freeze-probe.py --cold-streams 4` with light turns
interleaved, `ramp.py --context cached --levels 1,8,20`. Accept: light-turn TTFT p50 < 5 s while
four 300k-token cold prefills run, and cached-ramp tok/s stays within 10% of the resting line.
**Result: ACCEPTED, with a caveat.** `freeze-probe.py --cold-streams 4 --prompt-tokens 300000`
(4 cold + 20 warm, run on the final line described under R-MEM below): warm-stream TTFT during
the storm dropped from 55.2–56.2 s (no priority field) to 14.9–15.8 s (the router's 1/-1 spread) — a
3.7× ordering gain — while cached-ramp tok/s held within noise of the resting line
(`08-measurements.md` §11c). The 5 s accept target is **not met**: `--disable-priority-preemption`
never retracts a chunked prefill already in flight, so a light turn arriving mid-storm still
waits out whatever chunks are already queued ahead of it; this is an ordering rung, not a
preemption one. See `ISSUES.md`, "light-turn TTFT under a cold storm". Now promoted into
`v4-b200.args`; the router's priority hint armed.

`r-mem070-b200.args` was, at its measurement, `v4-b200.args` (carrying forward R-TAILS and R-PRIO)
with
`--mem-fraction-static 0.70` in place of 0.75. What it fixes: at 0.75 the 8×B200 line idles at
178–182 GB of 183 GB used per rank — effectively zero headroom, evidenced by 408 `Triton kernel …
device-loaded after serving started` lines at 0.00–0.13 GiB free and by two real crashes at that
margin (a 4-cold-stream prefill storm's index-k gather asking 7.95 GiB against ~5 GiB free, and a
needle-probe `CUBLAS_STATUS_EXECUTION_FAILED` right after a first-use kernel load) — see
`NOTES.md` 2026-09-17 for both incidents and the index-k gather fix
(`tools/index-k-gather-test.py`). Measures: `smoke.py`, the boot log's `available_gpu_mem` and
`max_total_num_tokens`, `freeze-probe.py --cold-streams 4` (4×300k cold + 20 warm) with the
engine alive and 0 OutOfMemory lines after, `ramp.py --context cached --levels 1,8,20` within
10% of the resting line. **Result: ACCEPTED** (READY in 132 s; `available_gpu_mem=43.53 GB`,
`max_total_num_tokens` 32,335,360 -> 26,330,880, -18.6%; idle GPU settles at ~173.4 GB used per
rank, ~10 GB free; smoke 10/10, needle cold 5.74–5.92 s; ramp within noise —
`08-measurements.md` §11b/§11c). Now promoted into `v4-b200.args`.

`r4-cp8192-b200.args` was the final line as it then stood (R-TAILS+R-MEM+R-PRIO) with
`--chunked-prefill-size 8192`
in place of 4096 — kept at R-MEM's `--mem-fraction-static 0.70` rather than the cookbook cell's
`0.8` (that cell sets no chunk size; `8192` is this ladder's own value), so this rung only
tests the chunk size against the already-accepted headroom, never trades the headroom away for
it. Measures: `smoke.py`, the storm probe
(`freeze-probe.py --cold-streams 4 --prompt-tokens 300000 --streams 20 --bg-priority 1
--cold-priority -1`) with the engine alive and 0 `OutOfMemory` lines after it, `ramp.py --context
short --levels 1,8,20`, `ramp.py --context cached --levels 1,8,20`, `probe-1m.py` 4/4 with
`memory allocation failed with OOM` lines unchanged. Accept: storm `prefill_tok_s >= 1.25x` the
4096 line's (29,041 with the priority fields), cached ramp within 10%, 1M probe clean. **Result:
REJECTED.** READY in 133 s, `max_total_num_tokens` 26,213,632, smoke 10/10; `ramp.py` held (short
346/200/143 per-stream, agg 346/1,162/1,842; cached 293/152/134, agg 293/1,084/2,276 — both within
noise of the 4096 line) — but the storm probe flips TRIAGE from OK to NOT-PREFILL-FIRST:
`bg_maxgap_med` 0.078 s -> 1.533 s (max 3.749 s), `bg_tok_s_during_med` only 1.81 tok/s,
warm-turn TTFT during the storm 11.3 s med / 21.1 s max, cold TTFT 22.0/28.4/39.1/49.8 s,
`prefill_tok_s` 13,606 (worse than the 4096 line's 29,041, not the required 1.25x gain),
`storm_prefill_tok_s` 21,489. `probe-1m.py` still passed 4/4 but logged 112 `memory allocation
failed with OOM` allocator retries the 4096 line does not. The bigger chunk starves decode under
a cold storm and repeats sglang#38956's OOM shape; `r4-cp16384-b200.args` was never run — no file
exists for it, a bigger chunk again was not worth measuring once 8192 itself regressed the storm.
Rolled back to `v4-b200.args` (`cutover.sh r4-back`); `08-measurements.md` §12.

**`moetp-b200.args` is the one rung whose `--ep-size` is not the 4→8 swap.** It keeps
`--ep-size 1` with `--tp-size 8`, rather than taking `--ep-size 8`: `--ep-size 1` _is_ R-MOETP
(the rung moves MoE all-to-all onto the tensor-parallel group), so a file that set it to 8
would cancel the rung's own change and would measure nothing. Every other `-b200` file takes its
`--tp-size 8 --ep-size 8` from `v4-b200.args`; this is the only one that overrides the pair.

## Test

`check.sh` in this directory asserts that every `*.args` file is exactly one non-comment, non-blank
line, and that none of them contains a flag from the forbidden list ("Never add"): the backend
overrides and the other config-of-record exclusions (`--attention-backend`, `--dsv4-attn-backend`,
`--moe-runner-backend`, `--fp8-gemm-backend`, `--enable-dp-attention`, `--kv-cache-dtype`,
`--enable-mixed-chunk`, `--enable-decoder-swa-bounded-replay`, which `FORBIDDEN_EXEMPT` allows in
the trim lines `v11-b200-box-e-tp4.args`, `v10-b200-box-e-tp4.args` and `stack-b300-box-a-mk2.args` alone), that every other line sets `--no-enable-decoder-swa-bounded-replay`
unless it is an unmarked 264ea0f9-era record that `LEGACY_UNMARKED` names (upstream's default is ON;
"Never add" above; a listed name with no file FAILs), the fixed serve-identity flags that belong to
the box file, not here (`--model-path`, `--served-model-name`, `--host`, `--port`, `--api-key`), and
`--tokenizer-worker-num` set to anything other than `1` (the SGLang multi-tokenizer / `--api-key`
assertion above). The `-b200` mirrors are checked exactly like every other file, and count in the
same totals. The `arms/*.args` A/B arms ("Never add" above) are checked by the same rules in a
section of their own, `--dsv4-attn-backend` allowed there alone under an `# A/B ARM: <reason>`
header, each held to its `ARM_PINS` base plus declared flags and to the bounded-replay rule through
its base; `tests/args-check.test.sh` covers both refusals, the allowed arm, the pin and the
bounded-replay rule.

It also asserts **the live-line rule** under "The 8×B200 mirrors": every `*-b200.args` file but the
anchor `v4-b200.args` itself is a rung and must set every flag `v4-b200.args` set — a missing one is
`FAIL: <file> is off the live line: …`, and a boot or a `cutover.sh` to that file serves without it
— and must set it at the live value unless that rung's ladder row declares the override, which is
the script's fixed `DECLARED_DELTAS` table (`file|flag|value`, one entry per override the R-ADM,
R-MOETP, N=0, R1 and R4 rows state, the 2048 rungs' halved short-prefill block among them); a value
outside that table is `FAIL: <file> sets <flag> …`, the live line having moved under a file nobody
re-based. A rung may still add a flag the live line does not set — that is its own delta and is not
compared. A new rung with a value override therefore carries its own `DECLARED_DELTAS` entry in the
same change, and so does an override the README tells an operator to carry by hand (R1's winner
written into a later rung's file before the live line takes it): until the live line takes the
value, the check fails and names it, which is the point — an override no row declares is the drift
the rule exists to stop. A `v4-b200.args` that is missing, or a directory with no `*-b200.args` rung
mirror beside it, is a `FAIL` rather than a vacuous pass, and the closing line reports what was
read: `OK: <checked> checked + <blocked> blocked, one tuning-flag line each, no forbidden flags,
<mirrors> rung mirrors on the live line (<deltas> declared deltas), <envs> env files exportable, the
-official pair equal to the live pair, the box-d line = the live line +
--no-enable-decoder-swa-bounded-replay`.

The env files are checked next. Every `env-*.txt` line must be `KEY=VALUE`, blank or a `#` comment,
because `relaunch.sh` `export`s each other line of the staged `sglang-env.txt` as it stands. Any
other shape is `FAIL: <file> has lines relaunch.sh cannot export`. `env-box-d-official.txt` must
start with `env-v3-official.txt` byte for byte, and its non-comment lines after that must be exactly
the script's `BOX_D_ENV_DELTA` (the two lines in the bullet above), in order. A drifted prefix or an
extra, missing or re-valued line is a `FAIL`. A new box-d-only variable goes into `BOX_D_ENV_DELTA` in
the same change that adds it to the file.

The `-official` pair is checked after that. `v4-b200-official.args`' tuning line must be
`v4-b200.args`' tuning line exactly — a string compare, so a flag added, dropped or re-valued, an
edit inside `--metric-label-denylist`'s two-token value, and a reorder are all `FAIL:
v4-b200-official.args' flag line is not v4-b200.args' flag line (…)`, the parenthesis naming each
side's differing `--flag value` run (or "same flags and values in another order or spacing"); the
header comments are free to differ. `env-v3-official.txt` must be `env-v3.txt` byte for byte, else
`FAIL: env-v3-official.txt is not env-v3.txt byte for byte (…)` with the first differing lines.
Either copy missing is a `FAIL` rather than a skip (official-pair bullet above).

The box-d line is checked last. `v5-b200-box-d.args`' tuning line must be `v4-b200.args`' tuning line
followed by one space and the script's `BOX_D_ARGS_DELTA` (`--no-enable-decoder-swa-bounded-replay`).
This is a string compare. A live-line change not copied into it, a second added flag, a dropped
delta, or a reorder or spacing change are all `FAIL: v5-b200-box-d.args' flag line is not
v4-b200.args' flag line + " --no-enable-decoder-swa-bounded-replay" (…)`. The parenthesis names each
side's differing `--flag value` run beyond the delta, says the delta is missing, or reads "same
flags and values, but not in the live line's order and spacing". A missing file is `FAIL:
v5-b200-box-d.args, the box-d serve line …, is missing` (the box-d bullet above).

A file whose first line is `# BLOCKED: ...` (the two R2 files, their `-b200` mirrors, R6's
`r6-compact.args` / `r6-compact-b200.args`, and `lpm-b200.args`) is skipped by the shape and
forbidden-flag checks but still counted and printed as `BLOCKED: <file>`, so the final line reads
`OK: <checked> checked + <blocked> blocked, ...` rather than folding blocked rungs into the plain
count. It is **not** exempt from the live-line check: a blocked file is still a rung. Run it with
`bash args/check.sh` (or `./check.sh` from inside this directory).

### The fork-side import gate

`check.sh` asserts the files' shape and the live-line rule and never imports sglang and never parses
a line, so the other half is invisible to it: a branch whose `python/sglang/srt/server_args.py`
cannot import at all — the unmatched `from sglang.srt.arg_groups.overrides import …` that made every
commit after the publish rebase die `ImportError` before `sglang serve` parsed a single flag, and
that pinned a box back to the pre-publish `99a36a44` — passes every file-shape check, because the
files were never wrong. That half is the fork repo's own gate,
`scripts/box-c/check_server_args_import.py` on our SGLang fork `box-c`: it imports
`sglang.srt.server_args` from a checkout and parses every `*.args` file with the parser `sglang
serve` itself builds. The outage entry is in `../CHANGELOG.md`; the gate's own README sits beside it
in the fork.

`check-import.sh` in this directory is this repo's one command for it. The args files are this
repo's and the fork clone is not, so it ships this directory's `*.args` files to a target, runs the
gate there against a sglang checkout on that target, and cleans the staging directory up:

```bash
BOX=<box> bash args/check-import.sh              # THE mode: on the box itself -- $SGLANG_TREE, $VENV/bin/python3, GPUs hidden
BOX=<box> bash args/check-import.sh --where fork # same gate on $FORK_HOST:$FORK_CLONE instead
BOX=<box> bash args/check-import.sh --self-test  # the gate's own fixtures: broken import, undeclared flag, empty dir
BOX=<box> bash args/check-import.sh --dry-run    # print every remote command, touch nothing
```

**The first line — `--where box`, the default and the mode of record — is the invocation that
counts**: the line a box is about to serve, parsed by the tree that box serves, which is the check a
cutover has to pass. `--where fork` is the identical gate on the fork clone, for a checkout no box
holds; `--self-test` runs the gate's own fixtures in place of these `*.args` files, the one-command
proof that the gate bites. `--dry-run` is not a fourth mode: it prints the command the selected mode
would run and touches nothing.

The gate that runs by default is this directory's tracked `check_server_args_import.py` — a carried
copy of the fork's at `71426a93` (its header names what differs), shipped to the target alone (its
self-test fixtures are embedded) so a branch that never had the gate (box-d-up, any upstream tree) is
still gated. Its boot closure is tree-aware: an `a|b` entry — by default the one slot
`sglang.srt.layers.communicator|sglang.srt.layers.layer_boundary`, the module upstream renamed — has
every alternative imported, and an alternative counts as ABSENT only when its import raises
`ModuleNotFoundError` naming that module itself (or a package above it); a module that exists but
imports a missing dependency is BROKEN, not absent. The slot passes when at least one alternative is
present and every present one imports. It FAILs when none is present (naming each one's
`ModuleNotFoundError`), and it FAILs when any present alternative dies on import **even when the
other name imports cleanly** — a broken module is never papered over by its rename. A single-name
entry is the fork gate's rule unchanged: absent or broken, it FAILs.

Flags: `--where box|fork`, `--args-dir DIR`, `--repo PATH` (the checkout to test), `--gate PATH`
(a gate script on the target, run in place), `--tree-gate` (the branch's own
`<repo>/scripts/box-c/check_server_args_import.py`), `--gate-src PATH` (ship a local gate and its
directory instead), `--python PATH` (the interpreter whose env can import sglang),
`--boot-closure LIST` (passed to the gate as `--boot-closure=LIST`; refused with `--self-test`),
`--no-cuda` (run the gate, and every probe interpreter it spawns, with `CUDA_VISIBLE_DEVICES=''` —
**the default under `--where box`**, so a serving box's GPUs are never touched by the gate; under
`--where fork` there is no prefix unless this flag asks), `--cuda` (the opt-out: leave the target's
`CUDA_VISIBLE_DEVICES` as it stands even on a box; refused together with `--no-cuda`), `--raw` (ship
the files byte-identical), `--no-hooks` (skip the validation-hook phase below), `--self-test`,
`--dry-run`. `--gate`, `--tree-gate` and `--gate-src` are
mutually exclusive; with none of them the tracked copy runs. `BOX` is required, as it is for every
script here, and under `--where fork` the box is only a config carrier — every box file names the
same fork host. The exit status is the gate's own: 0 only when the import resolved inside `--repo`
and every args line parsed with no unrecognized flag; then, the gate green, 3 when a validation hook
refused a line and 4 when the hook phase could not run; 1 is the script's own refusal (a flag, a
missing `FORK_HOST`, a box file naming no `SGLANG_TREE` or `VENV` under `--where box`, a target that
never answered, an args dir that is not one).

Two facts decide whether a run can go green. The target must be able to import sglang — a box's
`$VENV` can, while `root@<FORK_HOST>`'s plain `python3` carries neither torch nor sglang, so a `--where fork`
run there needs `--python` pointing at an interpreter that does. And `--repo` must be the tree you
mean: the gate tests the checkout it is given, and `--gate`/`--repo` may name different trees (a
pinned rollback worktree under test, the branch's own gate script, or a local `--gate-src` copy). A
`ModuleNotFoundError` in the gate's import line is the INTERPRETER, not the tree.

What is shipped per file is the line a box would SERVE, not the raw file: the file's single tuning
line with the box file's identity flags appended (`--model-path`, `--served-model-name`, `--host`,
`--port`, `--api-key`), which is the composition rule above as `relaunch.sh` applies it. The gate
parses a file line alone, and a raw tuning line stops at argparse's required `--model-path` before
any tuning flag is read — every file then reads as one indistinguishable "required: --model-path"
FAIL. A file whose shape is anything but exactly one tuning line is shipped byte-identical instead,
so the gate's own one-tuning-line rule still bites, and `--raw` ships everything verbatim.

**Parsing is not booting: the validation-hook phase.** A line can import and parse and still die at
startup: the 2048 rungs did, on box-d-up's `validate_short_prefill_protection`, while `check.sh` and the
gate both passed them. So once the gate passes, `check-import.sh` runs a second command on the same
target, with the same interpreter and the same GPU mode (hidden by default under `--where box`): its
embedded validation-hook phase, shipped to the staging dir's `hooks/`. For every staged line it builds
`ServerArgs` the way `sglang serve` does, runs the boot's own resolution as far as that machine goes
(GPUs hidden, a line naming no `--device` is resolved as `cuda` when `nvidia-smi` lists a GPU, the one
machine fact the phase supplies; on box-c every line then resolves fully), then every hook of the tree's
`sglang/srt/arg_groups/validation_hook.py` — each `validate_*`/`check_*` function taking exactly
`(server_args)`, `check_server_args` included, the Torch-MPS ones excluded — on that record. A hook's
`ValueError` or `AssertionError` fails the run (exit 3, the line and the hook's message named); a
resolution stopped by the machine is named and never fails a line; a hook that needs a field only a
full resolution fills is a note; a tree with no `validation_hook.py` is `HOOKS N/A`; a phase that
cannot import the tree exits 4. `--no-hooks` turns it off (the OK line says so). Run over this
directory against box-d-up, the phase passes the halved 2048 rungs and refuses the pre-fix block on its
max-tokens rule and the swap-limit-16384 variant on its swap rule, the two refusals the R1 window hit.
It would also refuse one tracked line: **`lpm-b200.args` does not boot on box-d-up** —
`check_server_args` asserts `To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'.
'lpm' is not supported.`, and that rung is the live line (priority scheduling on) plus
`--schedule-policy lpm`. That file now starts with a `# BLOCKED:` line naming the assertion, and
`check-import.sh` skips a `# BLOCKED:` file exactly as `check.sh` does (the test is the same: the
file's very first line). A blocked file is not shipped, not parsed and not run through the hooks; it
is printed on stdout as `BLOCKED: <file> -- <its BLOCKED reason> -- not shipped, not parsed, no
hooks run` and counted in the OK line (`(<k> BLOCKED, not judged: <files>)`). `--all-trees` does not
ship it; to judge a re-based rung again, delete its BLOCKED line. So a default `check-import.sh` run
on box-c judges every unblocked tracked line, and every one of them passes every hook.
