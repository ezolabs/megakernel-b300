# Data for the 4× B300 performance record

| path | what |
|---|---|
| `measurements.csv` | Every headline measurement as one row: id, date, box, config, tree, area (speed / megakernel / stability / quality), measurement, condition, value, unit, comparator and its value, change, verdict, n, source file. Row ids are the ones `HEADLINE.md` cites. |
| `decode-distribution.csv` | Per-concurrency distribution of decode tok/s per stream over every config X' boot on the record (14 boot-files, 2026-10-30/31). Written by `scripts/decode_distribution.py`. |
| `decode-c1-by-tree.csv` | C1 decode split by tree and by run (bench178 vs the C1 settle), with each boot's mean weighted once. Written by the same script. |
| `scripts/decode_distribution.py` | Recomputes `decode-distribution.csv` from the per-rep `think-*.jsonl` files under `box-c/qa/` (run from the repo root; standard library only). |
| `scripts/sanitize_scan.sh` | The sanitising scan (below); exit 0 = clean. |
| `scripts/xprime_loaded_hours.py` | Loaded production hours of config X' per box and tree (in pool on every router pod, ≥ 1 running request, 15 s samples) from prod sbr-vmsingle raw samples; writes `raw/x-prime-serving/samples.csv` and prints `summary.txt` (`stability.md`, rows T10–T12). |
| `raw/` | Copies of the raw files the tables rest on, so the folder stands alone. File names keep the source path with `/` → `__`, under one folder per record (table below). The full raw sets (boot logs, per-request records, per-item quality rows) stay in the source folders. |

## Raw folders and where they came from

| folder | source record (under `box-c/qa/` unless named otherwise) | contents |
|---|---|---|
| `raw/megakernel/` | `box-a-megakernel-20261029/raw/` | `gates.py` outputs (MK vs final, MK A/A, soak arm), matched acceptance comparison, teacher-forced compares, NVIDIA KL JSONs (final, EP1 controls, MKL1, MKL2), greedy compares idle and under load, soak statistics, per-rep think-ramp rows for MKL1 / MKL2 / LF1 |
| `raw/quality-mk-vs-final/` | `box-a-megakernel-20261029/raw/s2-final-ref/` | quality and DSML two-sample comparisons, final vs megakernel |
| `raw/prod-merge-x-prime/` | `prod-merge-20261030/raw/gate/` | config X' gate notes, cold-fill summary (60k / 70k / 100k), gates vs config X, per-rep rows of MKP1 |
| `raw/box-b-vs-box-a/` | `box-b-bringup-20261030/raw/speed/` | `bench178.sh`, both boxes' per-rep rows, gates, think and cold-fill summaries, `dmon` power summary, live-traffic aggregate reads |
| `raw/box-b-gate/` | `box-b-bringup-20261030/raw/gate/` | box-b's numerics gate: greedy, teacher-forced, NVIDIA KL, DSML compare, 1 M-token probe log, smoke |
| `raw/sglang-takes/` | `sglang-takes-20261031/` | upstream-takes and roll gates, roll summary, C1 settle (script, analysis, per-request rows of A1/A2/B1/B2), per-rep rows of every arm |
| `raw/l3a-regression/` | `l3a-regression-20261031/raw/` | quality comparisons: sign-off slice (per rep and pooled), campaign rows, refusal judge rows, MultiChallenge |
| `raw/fb/` | the Mac backup of box-a's `/workspace/box-a-mk2/raw/` (`~/workspace/backups/box-a/box/workspace/box-a-mk2/raw/`, every `*-FB*` file but the empty `arm-FB.out`) | the final pre-megakernel base arm FB (10-28 19:49Z boot, tree `819756d4632e`, EP4): boot record (args, env, load log), per-rep think-ramp, cold-fill, resume and extend rows with their per-request records, acceptance by concurrency, context-step timing, gsm8k (98/100; 97/100 with a 16k prefix), greedy outputs on the 32 fixed prompts, teacher-forced logprobs (two runs, token ids only). The think-ramp rows' C1/C2/C4/C8/C16 means (638.4 / 530.6 / 450.0 / 350.9 / 294.6 tok/s, 8 reps each) are the "final FB" column of `megakernel.md`'s speed table and rows M01–M06 |
| `raw/x-prime-serving/` | prod sbr-vmsingle, read 2026-11-01 04:41:44Z (no record folder) | per-sample `num_running_reqs` and in-pool flag for box-a and box-b from 10-30 12:51Z, tagged with the tree serving; `summary.txt` = the script's output |

## Sanitising

- No prompts or completions from production traffic are included; live reads are aggregate metrics only. The FB
  copies were checked for it: their bench rows hold counts, timings and seeds only (prompts come from `bench.py`'s
  seeded generator, the same commands and seeds as `raw/box-b-vs-box-a/raw__speed__bench178.sh`), the greedy file
  holds the model's outputs on the 32 fixed synthetic prompts (arithmetic, Fibonacci, primes, haiku, translation
  templates; the same set as `box-c/qa/box-a-megakernel-20261029/raw/s1-ep1-old/greedy-EP1old.jsonl`), the
  teacher-forced files hold token ids and logprobs only, and every request went to the on-box loopback port. The provider
  instance, offer, machine and host ids, box IPs, ssh and door ports, door hostnames, router keys and account ids
  appear nowhere in this folder; the boxes are named box-a and box-b.
- Replaced in the raw copies: router pod IPs → `<router-pod-ip>` / `<router-pod>`, LAN hosts → `<lan-host>`, the
  container IP → `<container-ip>`, in-container listener ports and a local port-forward → `<port>` / `<local-port>`,
  and one instance id in a header line (files: `raw/box-b-gate/raw__gate__probe-1m.log`,
  `raw/box-b-vs-box-a/raw__speed__live__learner-2126-2201.txt`, `raw/box-b-vs-box-a/raw__speed__live__live-2126-2201.txt`,
  `raw/prod-merge-x-prime/raw__gate__NOTES.md`). `127.0.0.1` (loopback, in bench headers) is kept.

### The scan

`scripts/sanitize_scan.sh` reads every file under `record/` (`.gz` files decompressed, none
present) for the patterns below. It reads the box identifiers (the provider instance, IP, ssh/app/door ports, door
hostname of box-a and box-b) from `setup/box/boxes/*.env` when it runs, so the script itself
carries none of the values. It was checked on a scratch file holding a planted instance id, IP:port and machine id,
in both plain and gzip form: every one was reported and the script exited 1.

Command (from the repo root), run 2026-11-01 after the last edit to this folder:

```
bash record/data/scripts/sanitize_scan.sh
```

Output (exit 0):

```
files scanned: 123 (gz: 0)
box identifiers read from setup/box/boxes/*.env: 2
box ids (instance, IP, ports, host) 0
the provider ids written as 'instance N' 0
IPv4 other than loopback/any       0
door hostnames                     0
router keys / tokens               0
engine API-key var / kube config   0
router key / account ids           0
router pod names                   0
ssh logins                         0
```
