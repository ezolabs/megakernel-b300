# box-a megakernel retry: runbook

The user's retry, named at box-a's retirement on 2026-10-29: in about a week, run the `sgl-nvidia-mega`
megakernel on box-a again, **if upstream has fixed its numeric drift**. This file is the whole procedure.
The record it rests on is [`README.md`](README.md) "Megakernel". The local plugin patch it needs is
[`sgl-nvidia-mega-b300-spec.patch`](sgl-nvidia-mega-b300-spec.patch). The repo copy is the durable one;
the same bytes are in the archive at `<HOME>/workspace/backups/box-a/meta/`.

## Result of the 10-29 retry (2026-10-30)

**SAFE to serve on box-a as config X, as a monitored canary beside box-c; the hang rate is bounded, not shown low.**
Record and every number: [`box-c/qa/box-a-megakernel-20261029/README.md`](../../box-c/qa/box-a-megakernel-20261029/README.md).
- Config X: `box-a-mk-latest` @ `9ec65dacaf` (pushed, not on prod), plugin `54b4289` + the B300 patch,
  [`args/mk-latest.{args,env}`](../../box-c/qa/box-a-megakernel-20261029/args/) on `/opt/sglang-mk29`, serving with
  `SGLANG_MK_TP_PROBE=1 SGLANG_MK_TP_PROBE_EVERY=100 SGLANG_MK_TP_PROBE_ASSERT=1` and `hangwatch.py --service
  --on-hang kill`.
- Speed: +38.8 / +29.6 / +23.7 / +11.7 / ~0% decode per stream at C1/2/4/8/16 (S4), +33.0 / +29.8 / +23.9 /
  +14.7 / ~0% on the second boot (S7); `gates.py` over both boots OVERALL PASS, 0 regressions in 38 sub-gates.
- Drift: all of it is `--ep-size 1` (S1); the MK decode path adds none on NVIDIA's KL; the merged tree without the
  megakernel is bit-identical to the final (S3). Teacher-forced short 94.30%, judged equivalent to box-c's accepted
  TP8 → TP4 94.83%.
- Quality against the final (S2, S7): nothing WORSE. DSML two-sample nothing WORSE; greedy under load no less
  stable; MK gsm8k prod 97.50 / 97.27% on two more draws, every pair with the final's two draws McNemar p ≥ 0.18;
  JSON 100/100 = the final; images served (80/80). The L3a sign-off (gate 3) PASSES 21/21 rows on S7's boot.
- Costs: KV pool −25.2% (66,653,184 → 49,879,552); accept length 2.5% lower at c4-identical over the two plain-MK
  boots (S4, S7; ~3 SE), c1 −1.4% and not significant, c4-mixed equal.
- Hangs: 0, and 0 probe mismatches, in 1.445 loaded hours on `9ec65dacaf` (≤ 2.07 per loaded hour). The ~18 h that
  would exclude the port's ~1 per 6 h accrue in serving. The bound covers T 1.0 / top_p 1.0 with no top_k only:
  no soak ran top_p < 1 or top_k > 0, the path where upstream sglang #38565 reproduces TP ranks diverging in
  DSpark's sampled accept and hanging NCCL. X' carries #38565 (deterministic renorm at TP > 1); X does not.
  Prod reaches box-a with top_k always dropped and top_p < 1 only on thinking requests whose client sets top_p
  (the router clamps it to [0.95, 1.0]); how often is not measured.
- Nothing owed: the router known-bad test (#39) passed on the dev pool 10-30 07:29-07:34Z
  ([`README.md`](../../box-c/qa/box-a-megakernel-20261029/README.md) "Hangs"). The megakernel stays off box-c (SR-144,
  gate 7); box-c runs the same `prod` tree without it since 10-30 (#96).
- Since 10-30 12:51Z box-a serves **config X'**: the fork's `prod` `4943d6ed90` (`box-a-mk-latest` merged in, plus
  upstream #46645, #38565, #47149) with [`stack-b300-box-a-mk2.args` + `env-box-a-mk2.txt`](../../args/);
  config X is its rollback. Record: [`box-c/qa/prod-merge-20261030/raw/gate/NOTES.md`](../../box-c/qa/prod-merge-20261030/raw/gate/NOTES.md).
- Serving hours in the router pools, 10-30. Hours by load band come from vmsingle
  `sglang:num_running_reqs{backend="box-a"}` (15 s samples). Hangwatch keeps no bands, and its per-minute peak puts
  the loaded time higher, so the loaded hours here are a floor.

  | | X (`9ec65dacaf`) | X' (`4943d6ed90`) |
  |---|---|---|
  | window | 07:24:22–12:01:07Z | 12:51:00–14:48:44Z |
  | wall in pool / loaded | 4.61 h / 0.650 h | 1.96 h / 0.342 h |
  | hours at 0 / 1 / 2–4 / ≥5 running | 3.96 / 0.49 / 0.16 / 0 | 1.62 / 0.27 / 0.08 / 0 |
  | peak running | 4 | 3 |
  | chat 200s | 5,462 | 1,946 (+2 × 409, cached-image refs, retried to 200) |
  | TP probe ok / mismatch | 283,044 / 0 | 114,256 / 0 |
  | hang episodes / kills | 0 / 0 | 0 / 0 |

  At 0 hangs the 95% bound is 3/T per loaded hour, and the builds are not pooled. `9ec65dacaf` (S5's 1.445 h plus
  X's 0.650 h) gives ≤ 1.43; `4943d6ed90` (the table's 0.342 loaded h) gives ≤ 8.77. Prod load peaked at 4 running, so serving
  adds evidence at low concurrency only. The 12:01–12:46Z gate boot ran with no hangwatch and counts toward
  neither bound.
- The 409s are the cached-image-refs check (`mm_bytes_required`), not the megakernel. The router left out an image's
  bytes, believing the engine held the conversation at least 1,417 tokens deeper than the engine actually matched
  (15,872 of an image ending at 15,881; matches are floored to 256-token pages, `scheduler.py:3590-3613`). The engine
  refused, and the router resent the conversation whole and got 200 (the serving router `sbr-backend/src/sglang.rs:3782-3832`,
  `:4322-4351`). box-a logged 4 by 15:34Z; box-c's copy a logged ~12 on `4606` before its roll, so the class predates
  `4943d6ed90`. About 1 an hour per copy, and no client saw one (#113).
- Upstream `sgl-nvidia-mega` PR #35 merged 10-30 as `b61745e`. It lets 148-SM B300 boards pass the SM gate
  when `sm148` is listed in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`, with no max-running cap. The open draft required
  `max-running-requests=1`; the merged form dropped that.
  - Our B300 patch already lowers the same gate, on evidence to C16 / max-running 64, so X and X' keep 64.
  - `b61745e` does not add our `Feature.KV_OFFLOAD`.
  - `SLOTS=sm148` alone replaces the 39 shipped slots and runs stock code.
  - Verdict: SKIP now, fold it in at the next plugin bump
    ([`sglang-sweep-20261031/NOTES.md`](../../box-c/qa/sglang-sweep-20261031/NOTES.md)). Revisit if upstream finds a
    wide-batch fault on 148 SMs ([`README.md`](../../box-c/qa/box-a-megakernel-20261029/README.md) "Plugin").

The rest of this file is the procedure as run, with the 10-29 bars.

## 0. Is it time? (no box needed)

The megakernel was fast and kept quality. It failed on numerics, and the cause is not found:

- **Short prompts.** Teacher-forced top-1 agreement with the final was **94.35%** (KL 4.0e-2). The final
  against a second boot of itself scores 100% / KL 0.
- **Long prompts.** 94.78% (KL 3.0e-2), against the final's own 94.93% / KL 2.28e-2.
- **Greedy outputs.** Byte-equal with the final on **1/32**.
- **Batch dependence.** Under live load, its greedy output matched its own idle reference on 0/32. The
  final's are batch-invariant.

The suspects are the attention chain's swap-AB path for 1–24 rows (the chain's `SWAPAB_MAX_ROWS=24`;
sglang #44632 raised SGLang's own cap to 40) and the plugin's other numeric slots.

**Superseded in part, 2026-10-29.** The short teacher-forced 94.35% is `--ep-size 1` alone: a plain EP1
engine with no megakernel scores 100% / KL 0 against MK3a and has the same 203 flips (S1,
[`box-c/qa/box-a-megakernel-20261029/README.md`](../../box-c/qa/box-a-megakernel-20261029/README.md)
"Drift"). What the plugin moves is decode: MK greedy is 0–1/32 byte-equal with plain EP1. The
batch-dependence row compared MK loaded with MK idle and was never run on the final at the same load.
The bars below replace these.

Retry only when one of these is true:

1. **The plugin.** `sgl-project/sgl-nvidia-mega` has a commit after `54b428921749` that names the
   DeepSeek-V4.1 numerics, the swap-AB / small-row attention path, or batch invariance. Read
   `git log 39595061519a..origin/main` in a clone. As of 10-29 the commits after box-a's `3959506` were
   `9a21cb0` (#20, Qwen3.5 + API v2), `4f1cefc` (#29, Engram host tables), `fe3f6fd` (#28,
   MoeDeferredFinalize import) and `54b4289` (#27, FlashMLA schedules-ahead: an optimisation, not this
   fix). As of 10-31 the only commit since then is `b61745e` (#35, the `sm148` SM-gate opt-in). It is not
   such a fix: its kernel arithmetic is unchanged.
2. **SGLang.** Upstream bumps `SGL_NVIDIA_MEGA_COMMIT` in `docker/Dockerfile` past such a commit, and
   the PR shows an accuracy eval. sglang #47145 (merged 2026-10-28 as `8d11638c`, pin → `54b42892`)
   showed 6,904 tok/s/GPU on 4×GB300 and **no accuracy eval**.

Otherwise the retry would only reproduce the 10-28 result. Tell the user that and stop.

**The API version gate.** Plugin `54b42892` and later speak **megakernel API v2**. SGLang's
`MEGAKERNEL_API_VERSION` (`python/sglang/kernels/plugin_kernels/nvidia/spec.py:16`) is **1** on both
box-a trees, `fd3545a46b53` and `819756d4632e`. Upstream moved to 2 in `8f1b7691ca` (#45266, 2026-10-27).
A v1 SGLang refuses a v2 plugin with "uses megakernel API version 2; this SGLang requires version 1"
**and boots without the megakernel**: no error, just the final's speed. So a retry on a fixed plugin
needs an SGLang tree at or past `8f1b7691ca` (path B below). The old pair (path A) is the control.

## 1. Bring box-a back (the user's word first)

box-a is stopped, never destroyed, and its disk is intact: `hostctl stop instance <INSTANCE_ID>` at
2026-10-29 05:27:26Z, `actual_status` `exited` at 05:29:09Z ([`README.md`](README.md) "Stopping
box-a"). The stopped disk (1,500 GB) still bills its storage. The instance's contract ends
2026-11-27T01:20:44Z. The provider's Instances FAQ says the instance is stopped at that date and its data stays until it is destroyed, but to assume the
instance is lost then; only the host can extend the contract. So start the retry before then, or
rebuild from the archive (§4). A start also needs machine <MACHINE_ID>'s four B300s free (step 1 below).
These are on the box, with the
[KV_OFFLOAD + resident-blocks patch](sgl-nvidia-mega-b300-spec.patch) already applied in the plugin
checkout:

- the model, `<MODEL_PATH>`
- the final venv `/opt/sglang-box-e` and the MK venv `/opt/sglang-mega`
- the trees `/workspace/box-e-up` (`819756d4`) and `/workspace/mega/sglang` (`fd3545a4`)
- the plugin checkout `/workspace/mega/sgl-nvidia-mega-b300`
- the harnesses `/workspace/box-a-mk2/` and `/workspace/mega/harness/`

1. **Start it, with the user's word, never on your own:**
   ```bash
   hostctl start instance <INSTANCE_ID>
   hostctl show instance <INSTANCE_ID> --raw   # public_ipaddr, ports["22/tcp"], ports["8001/tcp"]
   ```
   A start needs machine <MACHINE_ID>'s four B300s to be free. If they are rented, it waits, and that is the
   user's call. Never `hostctl destroy`; never create a replacement without the user.
2. **The box file.** If the IP or port mappings changed, edit
   [`boxes/box-a.env`](../../boxes/box-a.env): `BOX_IP`,
   `SSH_PORT` (the direct `22/tcp` mapping; the provider's `ssh_port` never answered), `APP_PORT_EXT` (the
   `8001/tcp` mapping). box-a's A record was deleted at the pool exit (2026-10-29 03:06:51Z), so
   `dns.sh` always runs, moved IP or not. Then, from the repo root:
   ```bash
   (cd setup/box && BOX=box-a bash dns.sh)      # re-create the A record -> BOX_IP
   (cd setup/box && BOX=box-a bash tls.sh)      # door cert + Caddyfile, if the cert lapsed
   (cd setup/box && BOX=box-a bash stage.sh)    # the tool set to /workspace/tools
   ```
3. **The onstart boots the final** (`sglang-args.txt` / `sglang-env.txt` = `stack-b300-box-a.args` +
   `env-box-a-w1.txt`, ~10 min) under the watchdog. Wait for READY. box-a is in no router pool (its rows
   were removed on 2026-10-29: dev v214 03:00:31Z, prod v214 03:02:08Z), so nothing reaches it. **Keep it out of every pool for the whole retry.**
   Anything that does join a pool goes through the window-keep / every-pod confirm in
   [`box-c/07-operations.md`](../../box-c/07-operations.md). The 10-28 MK2b lapse came from a window
   re-armed on a replaced pod.
4. **Check the disk survived.**
   ```bash
   cd /workspace/mega/sgl-nvidia-mega-b300 && git rev-parse HEAD && git diff --stat
   ```
   This should print `3959506…` and `spec.py | 4 ++--`. Also check
   `/opt/sglang-mega/bin/python -c 'import sgl_nvidia_mega, sglang; print(sglang.__file__)'`. If the disk
   is gone, rebuild from §4 first.

## 2. Path A: the control (box-a's old pair, reproduces 10-28)

The control uses plugin `3959506` + the patch in `/opt/sglang-mega` (SGLang `fd3545a4` editable), run
over the final tree through `PYTHONPATH=/workspace/box-e-up/python`. The args and env are the archived
#621 files. Run from the repo root on the Mac:

```bash
A=<HOME>/workspace/backups/box-a/box/workspace/box-a-mk2
(cd setup/box && BOX=box-a bash restart.sh box-a-mkr-MK3a "$A/mk3-args.txt" "$A/mk3-env.txt")
```

| arm | args | env |
| --- | --- | --- |
| MK2 | `mk2-args.txt` = the final with `--ep-size 1`, no HiCache flags, `--enable-nvidia-megakernel` | `mk2-env.txt` |
| **MK3** (the candidate; like-for-like) | `mk3-args.txt` = the final with `--ep-size 1`, HiCache io `direct` + layout `page_first_direct`, `--enable-nvidia-megakernel`; needs the patch's `Feature.KV_OFFLOAD` | `mk3-env.txt` |

The MK env, for MK2 and MK3 alike, is `env-box-a-w1.txt` with these changes:

- `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`, because the plugin refuses shared.
- `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample`, the plugin's requirement.
- `PATH=/opt/sglang-mega/bin:…` first, so `relaunch.sh` runs the MK venv.
- `PYTHONPATH=/workspace/box-e-up/python`.
- `XDG_CACHE_HOME=/workspace/cache-mega`.

The plugin forces EP1 itself and refuses EP4.

A megakernel boot waits ~690–800 s, longer than the final's ~600–690 s, because of the plugin's JIT and
autotune. The MK1 shape (#301, launched by `/workspace/mega/launch-mega.sh` on `:8012`) is not
like-for-like. Do not rerun it.

## 3. Path B: the real retry (a fixed plugin on an API-v2 SGLang)

1. **The plugin.** Pick the plugin commit `P` (§0). In a new checkout on the box:
   ```bash
   git clone https://github.com/sgl-project/sgl-nvidia-mega.git /workspace/mega2/sgl-nvidia-mega
   cd /workspace/mega2/sgl-nvidia-mega && git checkout P
   git apply --check /workspace/tools/sgl-nvidia-mega-b300-spec.patch
   ```
   Copy the patch over first with `scp -P <SSH_PORT> fleet-history/box-a/sgl-nvidia-mega-b300-spec.patch
   root@<BOX_IP>:/workspace/tools/`. On 10-29 the patch applied cleanly on `54b42892`.

   Check whether `P` already allows 148 SMs (`resident_blocks`) and lists `Feature.KV_OFFLOAD`. Apply
   only what it still lacks: the 148 for B300 is unofficial, the KV_OFFLOAD line is needed only for MK3.
   Record exactly what was applied.
2. **The SGLang tree.** It must be at or past `8f1b7691ca` and should be the commit `S` whose
   `docker/Dockerfile` pins `SGL_NVIDIA_MEGA_COMMIT` to `P`, or later. Build it as a **new fork branch**
   (`box-a-mk-<YYYYMMDD>`) on s6, per [`box-c/10-updating.md`](../../box-c/10-updating.md) "Syncing the fork to
   latest upstream" and "Cherry-picking one more upstream PR": `S` plus whatever of box-a's final it
   lacks. Test each with `git merge-base --is-ancestor <c> S`:
   - #46700's tombstone clamp
   - `35e6d667` (box-e image order)
   - `prod-20261026`'s own fixes in `fd3545a4`: the DSpark draft glue graph and the DSML parse
   - `faabdcae` (#901, the `todo_write` strict default), so the DSML gate can pass
   - `9e295882` (interior-evict) can stay out; it was always off

   **Never** force-push `prod`. This is a new branch, and `prod` stays `819756d4`.
3. **The venv.** Check out the branch at `/workspace/mega2/sglang`. Build a new venv `/opt/sglang-mega2`.
   On 10-28, `/opt/sglang-mega` was a `cp -a /opt/sglang-box-e` followed by these two installs (the
   editable build takes ~10 min):
   ```bash
   /opt/sglang/bin/uv pip install --python /opt/sglang-mega/bin/python3 --no-deps --no-build-isolation -e /workspace/mega/sglang/python
   /opt/sglang/bin/uv pip install --python /opt/sglang-mega/bin/python3 -e /workspace/mega/sgl-nvidia-mega-b300
   ```
   Do the same with the `mega2` paths. If `S`'s `python/pyproject.toml` pins differ from
   `<HOME>/workspace/backups/box-a/meta/pip-freeze-sglang-box-e.txt` (torch, flashinfer, sgl-kernel, cutlass
   DSL), build fresh instead. Use `/root/box-e-build/box-a-build.sh`
   (archive `box/root/box-e-build/`) with `VENV=/opt/sglang-mega2 SGL=/workspace/mega2/sglang` and S's
   own freeze. Then confirm:
   ```bash
   /opt/sglang-mega2/bin/python -c 'from sglang.kernels.plugin_kernels.nvidia import spec; print(spec.MEGAKERNEL_API_VERSION)'
   ```
   It must print **2**. Also check `import sgl_nvidia_mega` resolves to `/workspace/mega2/sgl-nvidia-mega`.
4. **Args and env.** Copy `mk3-args.txt` and `mk3-env.txt` from the archive to new files (`mkr-args.txt`,
   `mkr-env.txt`). In the env:
   - `PATH=/opt/sglang-mega2/bin:…`
   - **drop** `PYTHONPATH=/workspace/box-e-up/python`; that overlay is the v1 tree and would silently turn
     the megakernel off
   - `XDG_CACHE_HOME=/workspace/cache-mega2`

   Check every flag still parses on `S`. Upstream may have renamed or moved one; a refused flag fails the
   boot. Then:
   ```bash
   (cd setup/box && BOX=box-a bash restart.sh box-a-mkr-B1 <mkr-args> <mkr-env>)
   ```
5. **A second boot** of the same config (`-B2`), for the A/A band. If the base is to be re-measured on
   the new tree, also boot the final's args on the new tree without `--enable-nvidia-megakernel`
   (`-F`). That separates "new tree" from "megakernel".

## 4. Rebuild from the archive (only if box-a's disk is gone)

Follow `<HOME>/workspace/backups/box-a/MANIFEST.md` "What a retry / rebuild needs".

- **Box and base.** Create a 4×B300 box like box-a, at the user's word and through
  [`agent-speed`](../../box-c/qa/agent-speed-20261026/README.md) §7: the provider image as in the box file, 1,500 GB
  disk, `XDG_CACHE_HOME=/workspace/cache-box-a`.
- **Model.** Copy the L3a repack box to box from box-c, sha256-checked against `meta/model.txt`. Never via
  a public host.
- **Final tree and venv.** Clone the fork at `819756d4` to `/workspace/box-e-up` with the deploy key.
  Build `/opt/sglang-box-e` with `box/root/box-e-build/box-a-build.sh` to
  `meta/pip-freeze-sglang-box-e.txt`.
- **MK venv.** Check out fork `prod-20261026` = `fd3545a46b53` to `/workspace/mega/sglang`. Then
  `cp -a /opt/sglang-box-e /opt/sglang-mega` and run the two installs in §3.3. The plugin goes to
  `/workspace/mega/sgl-nvidia-mega-b300` at `39595061519a`, plus
  [`sgl-nvidia-mega-b300-spec.patch`](sgl-nvidia-mega-b300-spec.patch). The archive also holds that
  checkout whole, with `.git`.
- **Check the freeze.** Check against `meta/pip-freeze-sglang-mega.txt`.
- **Tools and harnesses.** Restore `/workspace/tools/` with `stage.sh`, and `/workspace/box-a-mk2/` and
  `/workspace/mega/harness/` from `box/workspace/`. Use the `.REDACTED` copies, and put the engine key
  back from `<HOME>/env/box-a.env`.
- **Front door.** `dns.sh`, `tls.sh`, the router tunnel key `box-a-kv` (re-issued; never from the
  archive). Stock Caddy v2.11.7 for the door.

## The bars (2026-10-29; replace gate 4's numbers, gate 6's bar and §6's list)

The 10-29 retry built path B. Fork branch `box-a-mk-latest` (our SGLang fork, pushed, not on prod)
has tip `9ec65dacaf`, with ancestry `9ec65dacaf` → `a5c32f373c` → `41bb61ec53` → `fbf4d51dc5` → the merge
`aee986068c`. `9ec65dacaf` commits the `swa_branch_rebuild` import fix that S4's engine ran uncommitted,
and `a5c32f373c` adds the probe extension (env-gated, off by default), so the tip reproduces the code S4
measured. On box-a: `/workspace/mk29/sglang`, the venv's editable tree, at `9ec65dacaf`, clean since 10-30 (it
held `41bb61ec53` + that fix, uncommitted, when S3 and S4 booted it;
[`raw/build-latest/tree-ff.md`](../../box-c/qa/box-a-megakernel-20261029/raw/build-latest/tree-ff.md)), and
`/workspace/mk29/sglang-next`, the same commit, no longer needed. Plugin `/workspace/mk29/sgl-nvidia-mega` (`54b42892` +
the B300 patch); the sync arms' plugin copies are `/workspace/mk29/sgl-nvidia-mega-syncall` and
`/workspace/mk29/sgl-nvidia-mega-syncsample`. Venvs `/opt/sglang-mk29` (mk-latest),
`/opt/sglang-mk29s` (syncall), `/opt/sglang-mk29r` (syncsample); cache `/workspace/cache-mk29`. The
arms' args and env are
[`box-c/qa/box-a-megakernel-20261029/args/`](../../box-c/qa/box-a-megakernel-20261029/args/), and the build
record is [`merge-latest.md`](../../box-c/qa/box-a-megakernel-20261029/merge-latest.md). Every arm is
judged against its own tree's controls: latest-F (the final's args, EP4) and latest-EP1 (the same with
EP1, no megakernel). As run: latest-F was booted (S3) and is bit-identical to the final, so latest-EP1 was not
booted and S1's old-tree EP1 boots stand in for it. The 10-29 results are recorded in
[`box-c/qa/box-a-megakernel-20261029/README.md`](../../box-c/qa/box-a-megakernel-20261029/README.md).

1. **NVIDIA's KL test** on each arm's own server
   ([`harness/nvidia_kl.py`](../../box-c/qa/box-a-megakernel-20261029/harness/nvidia_kl.py): decode vs
   prefill re-score, k3, 3 repeats). The bars are `KL_BOUND` 0.03 short and `LONG_KL_BOUND` 0.06 long. The
   old final scored 0.0048 / 0.0109. This bounds one server against itself. It does not compare two
   servers.
2. **Teacher-forced drift** (`tf.py`/`tfcmp.py`, short half), judged against the EP1 config's own cost:
   - MK vs latest-EP1 within latest-EP1's own A/A. On the old trees this was 100% / KL 0.
   - MK vs latest-F no worse than latest-EP1 vs latest-F. On the old trees that was 94.35% / KL 4.0e-2.
   - That EP1 cost sits 0.5 pt under box-c's accepted TP8 → TP4 move (94.83% short,
     [`box-c/qa/window2-tp4/README.md`](../../box-c/qa/window2-tp4/README.md)). Accepting it is the user's call.
     As run: the root judged the two equivalent (~17 flips of ~3,600, inside Poisson noise) and accepted it.
   - The long half is noise at the A/A floor (94.0–95.2% on every boot) and is reported, not judged.
3. **Greedy under load.** Run `greedy.py` (32 prompts, T0) at idle and under the same `synthmix.py` load,
   on MK and on the final (latest-F) alike. MK's loaded-vs-idle byte-equal count must not be below the
   final's at the same load. Idle MK against latest-EP1 is reported beside it, not judged.
4. **DSML, two-sample.** MK and the final at T1.0 / top_p 1.0, equal n per arm. Pass when the difference
   in defective-call rate has a Newcombe 95% CI that includes 0 or favours MK (Fisher's exact test as the
   check). The megakernel must not be worse than its own tree's final.
5. **Hangs, a Poisson bound.** Count loaded hours only, not idle uptime. With 0 hangs in T loaded hours
   the one-sided 95% upper bound is −ln 0.05 / T ≈ 3.0/T per hour. Excluding the public port's ~1 hang per 6 h takes
   T ≥ 18 loaded hang-free hours (2.996 × 6). Any shorter soak states its bound; the length is the
   user's call. A soak's load must carry sampled cells beside top_p 1.0: top_p < 1 (0.95, the router's
   thinking floor, and 0.9) and top_k > 0. DSpark's sampled accept takes a separate whole-vocab renorm path
   there (upstream sglang #38565), and a bound measured at top_p 1.0 says nothing about it. The 10-29 soaks
   (S5, S6b) ran top_p 1.0 only.

Superseded: gate 4's "short top-1 ≥ 99% vs the final", which every EP1 boot in S1 missed (94.35%, the
EP1-vs-EP4 reduction-order cost, plugin or not), and "greedy
≥ 28/32 vs the final", the same EP1 cost in decode. Gate 4's "greedy idle vs 16-stream load 32/32" was
never measured on the final at the same load. Gate 6's "0 hangs in 1 h" bounds the rate only to ~3/h.

## 5. The gates (every MK arm, in this order)

All of these run with box-a out of every pool. Each must pass before the next. The base is the final, or
the `-F` boot on a new tree.

1. **The megakernel is really on.** All three must hold.
   - The serve log has no `megakernel API version` refusal and no other plugin refusal line.
   - `max_total_num_tokens` reads **49,956,352** (EP1's fingerprint at mem-fraction 0.75). The final
     reads 66,653,184; a silent fallback with EP4 would read that, and a fallback with EP1 would still
     be slow.
   - The decode graphs report the megakernel for bs 1…64 (27/27 on 10-28), and C1 decode is ≥ ~850 tok/s.
2. **Speed.** On the box:
   ```bash
   cd /workspace/box-a-mk2 && ./arm.sh <L> <rep> std,pl
   ```
   Do two boots. On the Mac:
   ```bash
   cd box-c/qa/window2-tp4/harness && python3 gates.py --arm b1=$R/raw@<L1> --arm b2=$R/raw@<L2> --base final=$R/raw@<F> --copies 1 --base-copies 1
   ```
   Here `$R` is the fetched copy of the box's `/workspace/box-a-mk2/`. Each boot is judged against the base,
   and a regression must replicate on both boots.
   The bar is ≥ 20% decode per stream at 2–4 streams, the provider-ticket condition 1, plus no gate
   regression.
3. **Quality.**
   - On the box, `./qual.sh <L>`: gsm8k ≥ 97 on P16 and the teacher-forced top-5 dump on `eq-Ta`.
   - L3a sign-off on the Mac:
     ```bash
     <L3A_SLICE_SCRIPT> --concurrency 16 --tree <served sha> --expect-unset pass box-a 0
     ```
     Use `--expect-unset fail` on a tree without #901. It must PASS all 17 checks.
4. **Drift: the reason for the retry.** *Superseded 2026-10-29 by "The bars" 1–3: S1 showed these
   steps measured EP1 against EP4, not the plugin. They are the old bars, kept for path A.*
   - On the box, `python3 /workspace/mega/harness/tfcmp.py <final-tf.jsonl> <mk-tf.jsonl>`. Short
     top-1 must be ≥ 99% and KL within the final's own A/A, which was short 100% / 0 and long 94.93% /
     2.28e-2. In 10-28 terms: from 94.35% to ~100% on short.
   - Greedy (`greedy.py`, 32 fixed prompts, T0) against the final: from 1/32 to ≥ 28/32, the level the
     final scores against its own neighbours.
   - Greedy at idle against greedy under a 16-stream load (`spot.sh`'s check): 32/32. Anything less
     means the numerics still depend on the batch.
5. **DSML.** Copy the repo's current `dsml/probe.py` to the box; the copy #621 used in
   `/workspace/box-a-mk2/` is older. Run it on the box, which sends 470 per mode, both arms:
   ```bash
   PROBE_MODEL=DeepSeek-V4.1-Flash python3 -I probe.py <out> T1_todo_nested,T1_todo_nested_strictfalse
   ```
   Fetch `<out>` and grade it on the Mac against the served tree's own detector:
   ```bash
   python3 -I dsml/analyze.py grade <out> --detector-tree <served sha> --expect-unset pass
   ```
   Use `--expect-unset pass` on a tree that carries #901, and `--expect-unset fail` on path A (`819756d4`
   has no #901 and fails exactly like the final: 361/362, 362/363). The `strict: false` arm must pass
   either way. The megakernel must match its own tree's final, never be worse.
6. **Hangs: one synthetic hour, the watchdog on the box.**
   - Start `mkprobe.py` on box-a under `setsid nohup`. It writes `live/HANG` and evidence (py-spy per
     rank, metrics, log tail, `nvidia-smi`), then exits on a stall (30 s with no tokens while requests
     run, or `/health` failing twice).
   - Start `mixsoak.py <out> 3600 1,2,4,8,16`: 1–16 streams, mixed lengths, for 1 h.
   - Nothing for this gate runs on the Mac. The 10-28 Mac-side scripts (`mkwatch.sh`, `begin-hold.sh`,
     `end-hold.sh`, `health-read.sh`; archived at `<HOME>/workspace/backups/box-a/mac-side/` with
     `SHA256SUMS`) are a reference for what to port to the box, not a tool to rerun. `mkwatch.sh`
     needed `caffeinate` and a double fork, and a harness reaper killed it once. A HANG ends the arm;
     restart the final with `restart.sh … final-args.txt final-env.txt`.
   - *Superseded 2026-10-29 by "The bars" 5.* The old bar was 0 hangs. With 0 in 1 h the 95% upper
     bound is ~3/h. The port's known rate was about one hang per 6 h, so a longer run, or the user's
     earlier 12-h live soak, is the user's call.
7. **box-c fit (only if the megakernel is meant for box-c).** A box-c copy under the megakernel did not fit at
   mem-fraction 0.70 (`<HOME>/workspace/inference/.tmp/box-c-mk-fit/FIT.md`): ~0.87–1.25M device
   tokens kept against 2.15–3.25M needed. It needs ~0.80 or a smaller EP1 weight share. Re-read on the
   new plugin, which may have changed the workspace.

Record every arm in a new `box-c/qa/box-a-mk-retry-<YYYYMMDD>/README.md` with the same tables as the
[tuning record](../../box-c/qa/box-a-tuning-20261026/README.md) "Megakernel".

## 6. What flips the verdict

The 10-28 verdict was **not adopted**: speed met, quality held, numerics drifted, hangs not shown, box-c
fit failed. The retry adopts the megakernel on box-a when all of these hold:

- gates 1–3 and 5 pass
- "The bars" 1–4 pass (KL, teacher-forced drift against the EP1 cost, greedy under load against the
  final at the same load, DSML two-sample)
- the hang bound of "The bars" 5, at the soak length the user sets

The 10-28 list (top-1 and greedy inside the final's A/A band, batch-invariant greedy, 0 hangs in the
synthetic hour) is superseded: S1 showed that its drift bars measured the EP1 config, not the plugin.

It is then a candidate for the router. Rejoining box-a to a pool, or keeping the instance at all, is
the user's decision, as is anything for box-c: its fit and its own A/B on box-c's copies.

As run (2026-10-30): gates 1–3 and bars 1–4 pass (gate 2 on both boots, S4 and S7; gate 3's L3a sign-off 21/21 on
S7), and bar 5 stands at ≤ 2.07 hangs per loaded hour on 1.445 h. The root's verdict is SAFE as a monitored canary,
not the full adoption above. The user ruled on 10-29 that box-a returns serving the MK build if SAFE, else the
final of record; the return is #39's step.

If the drift is unchanged, record the arm and the plugin commit, then put box-a back to stopped. The
stop order is in [`README.md`](README.md) "Stopping box-a".

## 7. After the retry

As run: box-a stays up and serves config X (above); steps 1 and 5 apply only if the canary is withdrawn. Steps 3
and 4 stand.

1. Restore the final:
   ```bash
   restart.sh box-a-mkr-FINAL <HOME>/workspace/backups/box-a/box/workspace/box-a-mk2/final-args.txt <HOME>/workspace/backups/box-a/box/workspace/box-a-mk2/final-env.txt
   ```
2. Stop every box watcher: `touch /workspace/box-a-mk2/live/probe.stop`.
3. Never copy the box's logs off unredacted. `sglang-serve.log`, `mega-serve.log` and `qual.sh` carry
   the engine key.
4. Archive new raw beside `<HOME>/workspace/backups/box-a/`.
5. Stop the instance again on the user's word.
