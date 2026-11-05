# Safety layer: the TP-consistency probe and the hang watcher

Two independent watchers kept a bad engine from serving a wrong or silent stream. Both are part of the record
(`../record/stability.md` §"the safety layer"), not optional dressing.

## 1. The TP-consistency probe (`SGLANG_MK_TP_PROBE*`)

The probe is **in the engine tree**, not in this directory: its commit is
`../patches/merge-audit/probe-commit.patch` (`fbf4d51dc5`, "[box-a-MK] scheduler: env-gated TP-consistency probe
(SGLANG_MK_TP_PROBE)"), and it is part of the fork delta in `../sglang-fork/`. It is off unless
`SGLANG_MK_TP_PROBE=1`; the env of record (`../config/env-box-a-mk2.txt`) turns it on:

| variable | value | what it does |
|---|---|---|
| `SGLANG_MK_TP_PROBE` | `1` | enable the probe |
| `SGLANG_MK_TP_PROBE_EVERY` | `100` | one check every N scheduler iterations (default 100) |
| `SGLANG_MK_TP_PROBE_ASSERT` | `1` | a `MK_TP_PROBE MISMATCH` raises instead of only logging — the scheduler dies and the watchdog relaunches it |

Every `EVERY` iterations, every TP rank snapshots `(iteration, forward_ct, sum of the running batch's seq lens,
KV allocator available_size, crc32 of the sorted running rids, cumulative accepted spec tokens, cumulative verify
count)` before the request receive. TP rank 0's snapshot rides that iteration's existing request broadcast as one
extra list element (no new collective); every rank strips it and compares. A mismatch logs `MK_TP_PROBE MISMATCH`
at ERROR (and raises with `_ASSERT=1`); one `MK_TP_PROBE ok` line is logged every 10 checks.

Cost, measured: under ~1% per decode step (NEUTRAL) — S5P (probe every 10) sat 0.7% / 1.0% / +0.2% per step from
plain MK-latest at C1/C2/C4 (`../record/fork-and-config.md` §3, SR-182), and serving probes 10× less often.

`mk_tp_probe_selftest.py` is the probe's own offline selftest (the one the 10-30 merge gate ran); it needs no
engine:

```bash
python3 mk_tp_probe_selftest.py
```

## 2. The hang watcher (`hangwatch.py`)

Runs ON THE BOX, Python 3 standard library only. The box sheet's `HANGWATCH_CMD` starts one per boot:

```bash
python3 /workspace/mk29/harness/hangwatch.py --service --on-hang kill \
    --out /workspace/mk29/watch/serving --dump-signal none
```

- Every `--interval` s (default 3) it reads `/metrics` (`num_running_reqs`, `num_queue_reqs` and the progress
  counters) and every `--health-every` s GETs `/health`.
- A **stall** is: no progress for `--stall-s` (60) s while requests are in flight (`no_token`); a non-empty queue
  with nothing admitted (`queue_wedge`); `--health-fails` (2) consecutive `/health` failures (`health`); or
  `/metrics` unreachable for `--stall-s` s (`metrics_down`). Progress is any change of
  `realtime_tokens_total`, `generation_tokens_total`, `prompt_tokens_total`, or of the count of `Prefill batch` /
  `Decode batch` lines in the serve log.
- On a stall it writes `<out>/HANG` (one JSON line per episode) and captures `reason.json`, `metrics.txt`,
  `health.txt`, `procs.txt`, one `pyspy-<role>-<pid>.txt` per engine process (`py-spy dump --pid`, plus a
  `--native` dump for each scheduler rank), `nvidia-smi.txt`, `serve-tail.txt` (last 300 log lines, the engine key
  redacted) and `ps.txt` into `<out>/hang-<UTC>/`.
- `--service` keeps it running across episodes; `--on-hang kill` ends the stalled engine's process group so the
  watchdog relaunches a clean engine; `--dump-signal none` is **required on this stack** — the ranks' USR2 catcher
  is LLVM's (Triton's) `SA_RESETHAND` handler, so the default dump signal kills the rank.
- Ending it: `touch <out>/stop`.
- It logs `RELAUNCH`-style events only for its own kills; it never signals the engine otherwise.

What the watcher saw on the two boxes — box-a zero episodes through 2026-10-31 01:38:46Z, box-b's `HANG` and
episode captures — is tabulated in `../record/stability.md` and `../record/fleet-ops.md`; the raw captures are in
`../record/data/raw/`.
