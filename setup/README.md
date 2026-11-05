# Serving DeepSeek-V4.1-Flash on 4× B300 with NVIDIA's `sgl-nvidia-mega` megakernel

The setup kit for **config X'**: rebuild the served stack from this repository alone, except the model weights (the
L3a repack, §8). The measured record of that stack is `../record/`; the published write-up is `../article/`.

**The stack.** SGLang (the fork, delta in `sglang-fork/`) at `9a9ede7ae0`, with NVIDIA's
`sgl-nvidia-mega` megakernel plugin at `54b4289` plus a two-line B300 patch (`patches/sgl-nvidia-mega-b300-spec.patch`:
`resident_blocks` 152 → 148, `Feature.KV_OFFLOAD`), booted TP4 / EP1 with the flags in
`config/stack-b300-box-a-mk2.args` and the env in `config/env-box-a-mk2.txt`. It ran on rented 4× B300 SXM6 hosts
named box-a and box-b (2026-10-26 → 11-05).

## 0. Prerequisites

| | of record |
|---|---|
| GPU | 4× NVIDIA B300 SXM6 (sm_103 / compute capability 10.3), 148 SMs, 275,040 MiB (~269 GB) each, NV18 NVLink between every pair |
| host | box-a: 2× Intel Xeon 6776P (256 threads), ~3 TiB RAM. box-b: AMD EPYC 9575F |
| driver / CUDA | box-a: 580.173.02 / CUDA 13.0. box-b: 580.178.04 |
| container image | `lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` (the digest `CONTAINER_IMAGE` pins in both box files) |
| Python / venv | the megakernel venv's full freeze: `config/pip-freeze-sglang-mk29.txt` (torch, Triton and FlashInfer are the versions that freeze names; FlashInfer is 0.7.2 with `flashinfer-jit-cache==0.7.2+cu130`) |
| model | `<MODEL_PATH>`: the L3a repack of `deepseek-ai/DeepSeek-V4.1-Flash-1017`, 510 GB, served as `DeepSeek-V4.1-Flash`. **Not in this repository** (§8) |
| disk | 1,500 GB (`DISK_GB=1500`): the repack, plus a re-create's ~510 GB stock download the repack is hardlinked from, plus caches. The provider's disk cannot be resized after create |

Two boot-time settings the record fixes: `ulimit -c 0` (no core files) and one cache dir per tree
(`XDG_CACHE_HOME=/workspace/cache-mk29`); see `config/box-a.env` for the full box sheet.

## 1. Build the engine

`sglang-fork/fork-vs-upstream-3a3a8611a4.diff` is `git diff 3a3a8611a45f5a81fd07ffa6b0ed3ba33e7a4db9 9a9ede7ae0`
from our SGLang fork (449 files, 64,558 insertions, 2.7 MB; 152 commits listed in `sglang-fork/fork-commits.txt`).
`3a3a8611a4` is the upstream `sgl-project/sglang` commit the fork's `prod` tip still sits on:
`git merge-base 9a9ede7ae0 upstream/main` = `3a3a8611a4`.

```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang
git checkout 3a3a8611a45f5a81fd07ffa6b0ed3ba33e7a4db9
git apply /path/to/setup/sglang-fork/fork-vs-upstream-3a3a8611a4.diff
```

Check out the result where the box expects it (`SGLANG_TREE=/workspace/mk29/sglang` is the box file's value; any
path works if you write it down), then build a venv over it, editable, beside the plugin:

```bash
/opt/sglang/bin/uv pip install --python /opt/sglang-mega/bin/python3 --no-deps --no-build-isolation -e /workspace/mega/sglang/python
```

These are the commands of record (`box/RETRY-MEGAKERNEL.md` §3.3): `/opt/sglang` is
the box's base venv, `/opt/sglang-mega` the megakernel one made with `cp -a /opt/sglang-box-e /opt/sglang-mega`
before the two installs. If your `python/pyproject.toml` pins differ from `config/pip-freeze-sglang-mk29.txt`
(torch, flashinfer, sgl-kernel, cutlass DSL), build the venv fresh from that freeze instead.

## 2. Build the plugin, with the B300 patch

`sgl-project/sgl-nvidia-mega` is public. The patch applies to its `main` at
**`54b428921749fef89fcbd043000f839ac9cc4638`** (`54b4289`, megakernel API v2).

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git sgl-nvidia-mega
cd sgl-nvidia-mega
git checkout 54b428921749fef89fcbd043000f839ac9cc4638
git apply --check /path/to/setup/patches/sgl-nvidia-mega-b300-spec.patch   # exit 0
git apply        /path/to/setup/patches/sgl-nvidia-mega-b300-spec.patch
```

The `git apply --check` result, run for this repository on 2026-11-05, is
`sglang-fork/plugin-apply-check.txt`: **clean, exit 0**. The patch is two lines in
`sgl_nvidia_mega/deepseek_v41/spec.py`: `resident_blocks=152` → `148` (the SM gate is GB300's 152; B300 boards
have 148) and `features=frozenset({... Feature.KV_OFFLOAD})` (the HiCache KV-offload path). Install it editable into
the same venv:

```bash
/opt/sglang/bin/uv pip install --python /opt/sglang-mega/bin/python3 -e /workspace/mega/sgl-nvidia-mega-b300
```

Upstream's own `sm148` opt-in (`b61745e`, PR #35, merged 2026-10-30) lowers the same gate via
`SGLANG_NVIDIA_MEGAKERNEL_SLOTS=sm148`, but it **replaces** the 39 shipped slots instead of adding to them and does
not add `Feature.KV_OFFLOAD`. The record skipped it for that reason (`../record/b300-vs-gb300.md` §2,
`../record/fork-and-config.md` §2). Use the patch.

Also in `patches/`: `plugin-applied.diff` (the checkout's full applied delta), the two TP-sync arms'
plugin patches (`plugin-spec-tp-sync-all.patch`, `plugin-spec-tp-sync-rng.patch`), `merge-audit/probe-commit.patch`
(the fork commit that adds the TP-consistency probe), `json-schema-400.diff`, `fix-swa-commit.md`,
`build-45207-fix.diff`, box-b's `sgl-nvidia-mega-worktree.diff`, and box-b's installed `fork-picks.txt`.

## 3. The config of record

| file | what |
|---|---|
| `config/stack-b300-box-a-mk2.args` | **config X'** boot line (of record) |
| `config/env-box-a-mk2.txt` | **config X'** env (of record) |
| `config/stack-b300-box-a-mk.args` + `config/env-box-a-mk.txt` | config X (the rollback pair for X') |
| `config/arms/` | the 10-29 arms (`mk-latest`, `mk-syncall`, `mk-syncsample`, `latest-F`, `latest-ep1`) |
| `config/box-b-installed/` | what box-b had installed: `sglang-args.txt`, `sglang-env.txt`, `serve-model.env` |
| `config/box-a.env`, `config/box-b.env` | the box sheets (models, trees, paths, what each key is for) |
| `config/args-README.md` | the args-file composition rule |
| `config/pip-freeze-sglang-mk29.txt` | the venv freeze |

**Every flag and env var, with the reason for each, is tabulated in `../record/fork-and-config.md` §1** ("Key
flags" / "Key env"). That table is the authority, and it names which register row (SR-xxx) each value came from.
Two things the table does not carry: **the megakernel requires `--ep-size 1`** (it forces it
and refuses EP4), and the model path and ports come from the box sheet, not the args file (the args file is
"tuning flags only, per args/README.md's composition rule").

Config X' differs from X in exactly two flag values and four env lines (both files' headers say so):
`--hicache-write-policy write_through` → `write_through_selective`, `--no-enable-decoder-swa-bounded-replay` →
`--enable-decoder-swa-bounded-replay`; `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32`,
`SGLANG_DSV4_EXACT_DECODER_TRIM=1`, `SGLANG_MM_CACHED_REFS=1`, `SGLANG_MM_CACHED_REFS_EMBEDDING_CACHE=0`.

## 4. Boot it

`box/` is the tool set the boxes ran, plus box-b's installed render. The
boot path is: `boxenv.py` reads the box sheet → `build-onstart.sh <args-file>` renders `box/onstart.sh.tmpl` →
`stage.sh` puts the tool set on the box → the onstart writes `/etc/sglang-api-key.yaml`, `sglang-args.txt`,
`sglang-env.txt`, starts `relaunch.sh` (which starts `box/watchdog.sh`), and the engine comes up.

On an existing box:

```bash
(cd setup/box && BOX=box-a bash stage.sh)    # the tool set to /workspace/tools
```

and an engine restart with an explicit args/env pair:

```bash
(cd setup/box && BOX=box-a bash restart.sh box-a-mkr-B1 <mkr-args> <mkr-env>)
```

On a **fresh box**, `box/bringup.sh` is the orchestrator: preflight → offer select → `build-onstart.sh` →
create → ssh → `stage.sh` → download/graft gate → engine boot gates → smoke. Its own usage:

```bash
bash bringup.sh <args-file> [--offer <offer-id> | --instance <instance-id>]
                [--engine-args <per-copy args-file>] [--dry-run]
```

`stage.sh` also seeds `sglang-env.txt` from the box sheet's `ENV_SEED`; set `ENV_SEED=env-box-a-mk2.txt` to boot
config X'. A megakernel boot waits ~690–800 s (the plugin's JIT and autotune; the final took ~600–690 s), so
`WATCHDOG_BOOT_GRACE=2700` in the box sheet is the figure to keep.

The flags `SERVE_BUILD`/`MODEL_DIR`/`STOCK_DIR`/`ONSTART_MODEL_SOURCE` in the box sheet say how the onstart gets
the weights: `ONSTART_MODEL_SOURCE=copied` (box-a/box-b) gates the engine on `MODEL_DIR`'s 48 shards plus a
`MODEL_READY_MARKER` written by the copy's own `sha256sum -c`; the stock path downloads the public checkpoint
first. Both are in `box/onstart.sh.tmpl`.

## 5. The safety layer

Three things watch the engine; all three are part of the record, not optional dressing:

1. **The TP-consistency probe** (in the fork; `patches/merge-audit/probe-commit.patch`). On in serving via
   `config/env-box-a-mk2.txt`:
   `SGLANG_MK_TP_PROBE=1 SGLANG_MK_TP_PROBE_EVERY=100 SGLANG_MK_TP_PROBE_ASSERT=1`. Every 100th scheduler
   iteration every TP rank compares iteration, forward count, running-batch seq-len sum, KV allocator
   available size, a crc32 of the sorted running rids, and the speculative accept/verify counters; a mismatch logs
   `MK_TP_PROBE MISMATCH` and, with `_ASSERT=1`, raises: the scheduler dies and the watchdog relaunches it.
   Off unless `SGLANG_MK_TP_PROBE=1`. Its selftest is `safety/mk_tp_probe_selftest.py`.
2. **`safety/hangwatch.py`**: the stall watcher, started per boot by the box sheet's `HANGWATCH_CMD`:

   ```bash
   python3 /workspace/mk29/harness/hangwatch.py --service --on-hang kill --out /workspace/mk29/watch/serving --dump-signal none
   ```

   `--service` keeps it running across episodes; `--on-hang kill` ends the engine process group on a stall so the
   watchdog relaunches a clean engine; `--dump-signal none` is required on this stack because the ranks' USR2
   catcher is LLVM/Triton's `SA_RESETHAND` handler (the default dump signal would kill the rank). It writes
   `<out>/HANG` then captures `reason.json`, metrics, `/health`, the process tree, `py-spy dump` per rank,
   `nvidia-smi`, the serve-log tail (key redacted) into `<out>/hang-<UTC>/`.
3. **`box/watchdog.sh`**: the daemon that relaunches a dead or unresponsive engine, and Caddy independently,
   without relaunching an engine that is merely busy (it checks the serve log's recency and `/metrics`'
   `num_running_reqs`/`num_queue_reqs` first). Its run record is `box/installed/watchdog.run` (pid, start time,
   and the md5s of the box file and `lib.sh` it started with).

## 6. Bench: reproduce the headline

The headline decode figures are `bench.py think-ramp`: four synthetic reasoning prompts, thinking on, T 1.0 /
top_p 1.0, 300 completion tokens, C streams started together, 8 reps per C (32 at C1 in the settle run), seed
4242 + C; per request `completion tokens / (last delta − first delta)`, TTFT excluded (`../record/methodology.md`).
Run it with the box out of every router pool and `--max-live 0` (wait for 0 running requests before each rep).

```bash
# C1, C2, C4, C8 at 8 reps each + 8 x 60k cold fills; cwd /workspace/mk29/speed178
L=<label> bash raw__speed__bench178.sh
# C1 at n=32 (the settle run); cwd /workspace/mk29/c1settle
L=<label> bash c1-settle__bench-c1.sh
```

Both wrappers call `bench.py` (`bench/bench.py`, the `box-c/qa/speed-20261026/bench.py` of record) and `acc.py`.
`bench178.sh` covers C1–C8 and the cold fill; the **C16** cell of the headline table is the same
`think-ramp` at `--levels 16` (seed 4242 + 16), which is what the 10-29 arms ran. The 60k cold fill is
`bench.py coldfill` (4 background streams decoding, one never-seen 60k prompt; TTFT of the big prompt, prefill
tok/s = N / TTFT, the background streams' decode retention and longest delta gap).

The gate over two boots against a base, as run (`box/RETRY-MEGAKERNEL.md` §5.2, `gates.py`'s own
usage):

```bash
python3 bench/gates.py --arm b1=$R/raw@<L1> --arm b2=$R/raw@<L2> --base final=$R/raw@<F> \
    --copies 1 --base-copies 1
```

`$R` is your fetched copy of the box's bench output. A regression counts only when it replicates beyond the A/A
band on **both** boots; `OVERALL PASS` = no regression. `bench/decode_distribution.py` pools every C1 rep's
`dec_med` over the boot files (the distribution behind the record's decode figure).

## 7. Quality and KL

| check | tool | how |
|---|---|---|
| NVIDIA's KL (decode vs prefill re-score) | `quality/nvidia_kl.py` | `python3 quality/nvidia_kl.py --url http://127.0.0.1:8002 --mode both --repeats 3 --out kl.json`; bars `KL_BOUND` 0.03 short, `LONG_KL_BOUND` 0.06 long; exit 1 on a miss. Bounds one server against itself, not two servers against each other |
| teacher-forced drift | `quality/flips.py` + the `tf.py`/`tfcmp.py` comparison | top-1 agreement and mean KL between two arms' teacher-forced dumps; short = prompts 0–15, long = 16–31. **`tf.py`/`tfcmp.py` themselves are not in this repository**; they were 10-28's `/workspace/mega/harness/`, and the comparison outputs they produced are in `../record/data/raw/megakernel/raw__s1-ep1-old__tfcmp-*.txt` |
| greedy under load / idle | `quality/greedycmp.py` | 32 fixed prompts, T0, byte-equality between two arms; run idle and under `synthmix.py` load at the same settings |
| DSML two-sample | `quality/dsml/probe.py` → `quality/dsml/analyze.py grade` → `quality/dsml_compare.py` | both arms at equal n, T 1.0 / top_p 1.0, one phase each of `T1_todo_nested` and `T1_todo_nested_strictfalse`; `dsml_compare.py A B --labels final,mk` prints B − A with Newcombe's 95% CI and Fisher p, exit 0 when nothing is WORSE |
| synthetic mixed load (soaks, quality under load) | `quality/synthmix.py` | `setsid nohup python3 harness/synthmix.py soak --out runs/soak-<tag> --duration 3600 --streams 1,2,4,8,16 --temperature 1.0 --top-p 1.0 …`; agent/code/qa/long/json/intl mix, per-request checks in `requests.jsonl` and `summary.json` |
| the quality harness itself | `quality/quality/` | the 10-29 harness's quality suite (`run_all.sh`, `test_quality.py`, `gsm8k_full.py`, `needles.py`, `under_load.py`, …), with `quality/test_harness.py` |
| L3a sign-off (the release gate) | not in this repository | `<L3A_SLICE_SCRIPT> --concurrency 16 --tree <served sha> --expect-unset pass box-a 0`; it must PASS all 17 checks (21/21 rows on the 10-29 boot) |

## 8. What still depends on things not in this repository

- **The model weights.** `<MODEL_PATH>`: the L3a repack (510 GB) built from the stock
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint plus our graft. The checkpoint is public (MIT, 1,048,576
  `max_position_embeddings`, 384 routed experts); the repack is not. `MODEL_DIR`'s path is what the args and box
  sheet expect; point it at your own weights and the stack runs, but the numbers are the repack's.
- **The fork.** `sglang-fork/` carries the delta and the commit list, not the git history. Apply the diff
  (§1); the fork's own branch names and deploy key are not needed.
- **Provider-specific bring-up.** `box/bringup.sh` drives the provider's CLI (`hostctl`, from `$HOSTCTL`
  on `PATH`, authenticated), an ssh key, and an `ENV_FILE` holding `VLLM_API_KEY` and `HF_TOKEN`. The values are
  not here, by design. A host on a different provider needs the same shape: the container image, the disk, the
  onstart, and the tool set staged at `TOOLS_REMOTE`.
- **The door.** `box/tls.sh` renders the Caddyfile and serves the door's certificate files for a hostname you
  control; the certificate itself is issued outside this kit. The door is not needed by the engine itself.
- **The router.** The KV events (`zmq` on 127.0.0.1:5557/5558) and the load feed (127.0.0.1:5559) are loopback
  only; nothing in this repository consumes them. `../record/fleet-ops.md` describes the router side.
- **The graft / stock-download path.** `bringup.sh`'s graft step needs `GRAFT_SRC`, `GRAFT_SHA256` and an
  `apply_graft.py` from our grafting kit (`APPLY_GRAFT_PY`); a box that serves `MODEL_DIR` as copied leaves
  `GRAFT_SRC` empty and the step is skipped.

## 9. Layout

```
setup/
  README.md                      this file
  MANIFEST.tsv                   every copied file: dest, source repo, source sha, source path, scrubbed, note
  patches/                       the B300 plugin patch + the other plugin/fork patches and fix notes
  config/                        the args/env of record (X', X), the arms, the box sheets, the venv freeze
  box/                           the tool set: bringup, build-onstart, stage, restart, relaunch, watchdog,
                                 tls, the onstart template, the probes, box-b's installed render, the
                                 runbook and the two box sheets
  box/boxes/, box/args/          the box sheets and the args/env files again, where lib.sh and stage.sh read
                                 them ($TOOLS_DIR/boxes/$BOX.env, $TOOLS_DIR/args/$ENV_SEED)
  box/installed/                 box-b's installed /root/onstart.sh and watchdog.run
  safety/                        README (the TP-probe material), hangwatch.py, the probe's selftest
  bench/                         bench.py, acc.py, the two wrappers, gates.py, decode_distribution.py
  quality/                       the harness (nvidia_kl, dsml_compare, synthmix, test_harness, quality/*),
                                 greedycmp.py, flips.py, and the DSML probe/analyzer it needs
  sglang-fork/                   the fork's delta vs upstream, its commits, the plugin apply-check
```

`box/` is **not** the whole internal tool directory. It holds every script `bringup.sh` sources or calls:
`lib.sh`, `offer-search.sh`, `build-onstart.sh`, `stage.sh`, `restart.sh`, `relaunch.sh`, `watchdog.sh`,
`smoke.py`, `tls.sh`, plus the probes the brief names, the onstart template, `freeze-gc-timer.sh`, and the box
sheets/args where `lib.sh` and `stage.sh` read them. Left
out, because a bring-up does not run them: `tests/`, `caddy-h2/`, `systemd/`, `fixtures/`, `gatekit/`, and the
tools that serve other flows (`fork-sync.sh`, `mirror-and-branch.sh`, `image-build.sh`, `cutover.sh`,
`boxout-gate.sh`, `failover-watch.sh`, `graft-apply.sh`, `probe-1m.py`, `soak.py`, `ramp.py`, `gsm8k.py`,
`loganal.py`, `seed-autotune.sh`, `trim-ab.py`, and their kin). The graft step's `graft-apply.sh` and the
grafting kit's `apply_graft.py` are **not** here: the step is skipped when the box sheet's `GRAFT_SRC` is
empty, and both belong to the grafting repo. One caveat: both sheet files name
`ENV_SEED=env-box-a-mk3.txt` (box-a, a 2026-11-04 change) or `env-box-a-mk2.txt` (box-b). The mk3 seed is not in this
repository; to boot config X' set `ENV_SEED=env-box-a-mk2.txt` in the box sheet (or `ENV_SEED=env-box-a-mk.txt` for
X). `stage.sh` refuses a bare file name under `args/` that is not there, so this matters.

Every copied file is scrubbed of identifiers: IPs, ssh ports, hostnames, the provider instance/offer/machine ids and
operator paths are `<BOX_IP>`, `<SSH_PORT>`, `<APP_PORT_EXT>`, `<DOMAIN>`, `<INSTANCE_ID>`, `<OFFER_ID>`,
`<MACHINE_ID>`, `<HOST_ID>`, `<HOME>`-style placeholders. Every substituted value sits inside a
quoted string, so the box sheets still source cleanly; where a placeholder replaced a value the script compares
against (`BOX_C_INSTANCE` in `bringup.sh`), that guard is inert: box-c is not part of this setup. `MANIFEST.tsv`'s
`scrubbed` column says which files changed; its `note` column names the placeholders each file carries and how many
lines changed against the source. Box names (box-a, box-b), the GPU model and host specs, and every
measurement are kept verbatim.
