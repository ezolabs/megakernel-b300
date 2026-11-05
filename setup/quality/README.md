# Megakernel job harness: synthmix.py + hangwatch.py

Both scripts run ON box-a (`ssh -p <SSH_PORT> root@<BOX_IP>`) against the engine on 127.0.0.1:8002, Python 3 stdlib
only (checked to parse on 3.9). The box copies are `/workspace/mk29/harness/synthmix.py` and
`/workspace/mk29/harness/hangwatch.py`; this directory is the source of truth (re-`scp` after an edit).
The engine key comes from `$SGLANG_API_KEY` / `$VLLM_API_KEY` / `$API_KEY`, else `VLLM_API_KEY=` in
`/etc/environment`; neither script writes it anywhere (hangwatch redacts it from every file it captures).

## synthmix.py: synthetic agent-shaped mixed load

Every prompt is generated from `--seed` (default 29): no corpus, no production data, so its outputs may be committed.
Kinds (`--mix`, default `agent=35,code=20,qa=20,long=10,json=10,intl=5`):

| kind | what | check recorded |
|---|---|---|
| agent | coding-agent session: generated system prompt, 6 tools in OpenAI `tools` format, history of prior assistant tool calls + tool results; each stream walks its own session turn by turn (growing shared prefix, 4-14 steps, then a new session) | `tool_parse_ok` (known tool name + JSON-object arguments), `n_tool_calls`, `tool_names` |
| code | refactor / bug-hunt / test-writing ask over a generated 60-400 line Python module | `bug_found` (names the planted off-by-one function) |
| qa | short question; half are `x * y` | `answer_ok` |
| long | generated report of `--long-min..--long-max` tokens (default 8k-120k, log-uniform) with planted unit codes; 70% ask for one | `answer_ok` (needle) |
| json | `response_format: json_object` extraction | `json_ok`, `json_year_ok` |
| intl | fr, de, es, ja, zh, ru, pt; some arithmetic | `answer_ok` |

Every request streams (`stream_options.include_usage`), T `--temperature` (1.0) / `--top-p` (1.0). Thinking:
`--thinking-off` (0.3) of requests send `reasoning_effort: "none"`, the rest `reasoning_effort: --effort-on`
(`high`, box-c's official default). `max_tokens` is drawn per kind (128-2048, capped by
`--max-tokens-cap` 2048), larger when thinking is on, so an hour stays sane: thinking off / on, agent 512-1536 /
1024-2048, code 1024-2048 / 1536-2048, qa and intl 128-512 / 512-1024, long 256-768 / 768-1536, json 256-512 /
768-1024. The thinking switch is the top-level `reasoning_effort` field, as box-c uses it (`box-c/04-engine.md`
"Thinking and reasoning effort": a top-level effort word turns thinking on, `"none"` turns it off and wins over
`chat_template_kwargs`); `chat_template_kwargs` is not sent. Long documents are generated at ~4.3 chars/token, so the
120k-token top end is ~130k real tokens, far inside box-a's `--context-length 1048576`. Every request also checks for a leaked
`</think>` / DSML marker in content (`bad_marker`) and empty output (`empty`). A request is never cancelled; a failed
one makes its stream pause `--error-backoff-s` (2 s).

```bash
cd /workspace/mk29 && mkdir -p runs
# 1 h soak: phases cycle 1,2,4,8,16 streams (240 s each); half the phase boundaries drain to ZERO in flight and stay
# idle 5-90 s (the idle-desync case)
setsid nohup python3 harness/synthmix.py soak --out runs/soak-<tag> --duration 3600 --streams 1,2,4,8,16 \
    --temperature 1.0 --top-p 1.0 > runs/soak-<tag>.out 2>&1 < /dev/null &
# constant background load (quality-under-load / greedy-under-load: run your probe while this runs)
setsid nohup python3 harness/synthmix.py load --out runs/load8-<tag> --streams 8 --duration 1800 \
    > runs/load8-<tag>.out 2>&1 < /dev/null &
# self-contained greedy-under-load: C8 load, after 60 s warm-up 2 rounds of the 32 greedy.py prompts
# (T0, 256 tok, sequential, sha = sha256(reasoning NUL content)[:16] exactly as box-a-mk2/greedy.py), then stop
python3 harness/synthmix.py load --out runs/gul8-<tag> --streams 8 --duration 3000 --greedy-rounds 2 \
    --warmup-s 60 --stop-after-greedy
touch runs/<dir>/stop                         # ends any run early (in-flight requests finish)
python3 harness/synthmix.py sample --n 20     # print generated bodies (abbreviated), sends nothing
```

Other knobs: `--phase-s`, `--idle-prob` (0.5), `--idle-min/--idle-max` (5/90 s), `--think-max` (2 s pause between
a stream's requests, uniform), `--long-min/--long-max`, `--request-seed` (per-request sampling seed),
`--read-timeout` (900 s), `--hang-gap-s` (30 s), `--save-text` (keep output text), `--url`, `--model`
(default `DeepSeek-V4.1-Flash`).

Outputs in `--out`:
- `requests.jsonl` one row per request: `phase, C, stream, n, kind, thinking, max_tokens, ttft, dur, max_gap, hang,
  prompt_tok, cached_tok, out_tok, chunks, tps` (decode tok/s after the first chunk), `finish, err, http,
  out_sha` + the checks above.
- `phases.jsonl` per phase: wall, n, errs, hangs, out_tok, `agg_tok_s`, TTFT p50/p99, `tps_p50`; idle rows carry
  `idle_s`.
- `summary.json`: totals and the same block `by_C`, `by_kind`, `by_thinking` (n, errs, hangs, max_gap, empty,
  bad_marker, finish reasons, tokens, TTFT p50/p99/max, decode tok/s p10/p50/p90, tool_parse_ok, answer_ok,
  json_ok), `per_stream_tps` (p50 per stream id), `error_kinds`, `harness_errors`, phases.
- `config.json`, `started` (first request sent), `done`; `greedy-r<k>.jsonl` in greedy-under-load runs (with
  `load_inflight`, the in-flight load requests when each probe was sent).

Scoring and exit status (hardened before the soak):
- `answer_ok` is an exact whole-number match: `2401` no longer passes `24010` (or `12401`, `2401.5`, `v2401`);
  digit groups `7,203` / `7.203` / `7 203` read as `7203`; a number inside CJK text (`答案是7203。`) counts.
  `bug_found` matches the planted function name as a whole identifier.
- Every request that carried a check is scored, so errors, empty and truncated outputs stay in the denominator:
  each row has `answer_outcome` / `json_outcome` / `bug_outcome` = `ok`, `wrong`, `empty`, `truncated` (finish not
  `stop`) or `error`; `summary.json` blocks carry `answer_ok` / `json_ok` / `bug_found` as ok / all sent plus
  `answer_outcomes` / `json_outcomes` / `bug_outcomes` (the five counts). Earlier runs dropped non-`stop` rows from
  these rates, so their rates are not comparable with these.
- `empty` vs `truncated`: a request with no content and no tool call is `empty` (it counts against the gate) whatever
  its finish reason (`stop`, `abort`, none at all), EXCEPT a max_tokens cut inside the reasoning: finish `length`
  with reasoning streamed (`reasoning_content`, read into the row's `reasoning_chars`), which is not empty (thinking
  on, the budget spent thinking; its checks score `truncated`). A `length` finish with no reasoning either is empty.
- Exit 1 (stderr `GATE FAIL: ...`) when (errors + empty outputs + harness errors) / requests sent exceeds
  `--max-error-rate` (default 0.02) or nothing was sent; `summary.json` `gate` records the numbers. A 0 exit is
  therefore evidence of a clean run.
- `quality/qlib.py` `extract_answer(content, *, fallback=True)`: the `#### <n>` answer line wins; the last-number
  fallback for a reply without one is now an explicit keyword. `fallback=False` scores the answer line alone. The
  default stays True because `gsm8k_full.py` / `mgsm.py` score `exact` by gsm8k.py's rule (their `format` metric is
  the answer-line rate).

## hangwatch.py: stall watcher with per-rank py-spy

```bash
mkdir -p /workspace/mk29/watch
setsid nohup python3 /workspace/mk29/harness/hangwatch.py --out /workspace/mk29/watch/<tag> \
    > /workspace/mk29/watch/<tag>.nohup 2>&1 < /dev/null &
touch /workspace/mk29/watch/<tag>/stop        # ends it within one poll
```

Polls `/metrics` every `--interval` (3 s, timeout 5 s) and `/health` every `--health-every` (default: the same
interval, `--health-timeout` 5 s, so a hung `/health` cannot hold the loop). The first good `/metrics` logs which metric
names matched (`metrics matched: ...` in `watch.log`, `metric_names` in `state.json`); if the running-requests metric
or every progress counter is absent it logs ONE `WARNING ... no_token/queue_wedge stall detection is INERT` (a missing
queue metric alone makes only `queue_wedge` inert; `state.json` carries `metric_warning`).

PROGRESS is any change of `sglang:realtime_tokens_total` (all `mode`s; the scheduler bumps it at every prefill chunk
and every decode step), `sglang:generation_tokens_total` or `sglang:prompt_tokens_total` (these two move only when a
request FINISHES), or of the count of `Prefill batch` / `Decode batch` lines in the serve log (one per prefill chunk,
one per decode-log interval). IN FLIGHT is `num_running_reqs + num_queue_reqs`; a zero reading counts as idle (resets
the stall clock) only after holding `--idle-hold-s` 35 s. Series of tp_rank != 0 are ignored. A STALL is `no_token`
(running > 0 and no progress for `--stall-s`, default 60 s), `queue_wedge` (running 0, queue > 0, no progress for
`--stall-s`), `health` (two consecutive failures: non-200, error, or > `--health-timeout`; NOT while that poll's
progress moved with work in flight -- `/health` is a real generation queued behind user work, so that is a busy
engine, logged once as `busy, not a stall`, watchdog.sh's own rule) or `metrics_down` (`/metrics` unreachable
`--stall-s`). Why: the old rule watched generation tokens alone with a 30 s limit and counted
running == 0 as progress, so a healthy cold 900k prefill (~30-60 s at `EST_PREFILL_TPS` 31000, no request finishing)
was a false `no_token`, and a queue-only wedge never fired. Checked on box-a's build (read-only, no engine request):
`realtime_tokens_total` and `prompt_tokens_total` are in its `metrics_collector.py`, it runs `--enable-metrics
--chunked-prefill-size 8192`, and its serve log carries `Prefill batch` / `Decode batch` lines. Keep `--stall-s`
above the longest single prefill batch; 60 s is the soak default.
On a stall it appends one JSON line to `<out>/HANG` FIRST (poll that file), then captures `<out>/hang-<UTC>/`:
`reason.json`, `metrics.txt`, `health.txt`, `procs.txt` (sglang process tree with roles `tp0..tp3`, `scheduler`,
`detokenizer`, `launcher`, found via `/proc`: everything named sglang plus its descendants),
`pyspy-<role>-<pid>.txt` for every one of them plus `-native` dumps of each scheduler/TP rank, `nvidia-smi.txt`,
`serve-tail.txt` (last 300 lines, key redacted), `ps.txt`. Apart from the dump signal below and the opt-in
`--on-hang kill` it never kills, signals or restarts the engine and keeps watching: the next capture comes after
progress resumes (or `--recapture-s` 600 s of the same stall), at most `--max-captures` (5; 50 with `--service`).

BOOT. An engine is BOOTING while it has not answered `/health` 200 since this watcher killed it, or while its
top-level serve process is younger than `--boot-grace-s` (2700 s: box-a boots in 10-27 min; watchdog.sh's own boot
ceiling) and has not answered `/health` 200 yet. While booting there is no HANG and no kill (logged once as `engine
booting (...)`), every stall clock is held at zero, and it ends at the first `/health` 200 (`engine up`) or at the
grace (`boot grace over ... without /health 200`). The serve process is found with `ps -ww -A`: args matching
`--serve-pattern` (default `sglang serve\b|sglang\.launch_server\b`, watchdog.sh's liveness is `sglang serv[e]`),
not a zombie, not hangwatch, and, when its args carry `--port`, that port = `--serve-port` (default: `--url`'s port,
8002); a match whose parent also matches (a forked rank still showing the launcher's argv) is not top-level. This
applies at `--on-hang none` too: a watchdog relaunch that boots is not a `metrics_down` HANG.

`--on-hang kill` (default `none`: never kill). Why: if box-a serves the megakernel in the router, a hang outside a
forward stays stuck -- the router ejects the wedged backend (~2 s) but watchdog.sh never relaunches an engine whose
`/metrics` shows running + queue > 0 (its progress guard reads that as busy), and sglang's `--watchdog-timeout` (300 s)
fires only inside a forward; watchdog.sh DOES relaunch at once when no `sglang serve` process exists. So: when a stall
is still on at a poll AFTER its capture finished (the evidence is on disk and the stall outlived it), or
`--on-hang-after` (120) s into an episode that took no capture (capture cap reached), hangwatch SIGTERMs the
top-level serve process's process group (relaunch.sh execs `sglang serve` under `setsid`, so it leads one; if it does
not, or the group is hangwatch's own, it signals the process and its descendants instead), SIGKILLs whatever is left
after `--on-hang-term-s` (20), logs `ON-HANG KILL pid=<pid> pgid=<pgid> at <UTC>: <reason> episode <n> ...` and its
`done: term|killed|kill_failed` in `watch.log`, appends one JSON line per decision to `<out>/KILLS` (also
`state.json` `kills`), then keeps watching while the engine boots again. Limits: one kill per episode; at most
`--on-hang-max` (3) per `--on-hang-window-s` (3600), then `ON-HANG KILL suppressed ...` and KILLS outcome
`rate_limited` (only logged; a hang that keeps coming back needs a human). The window counts the kills already in
`<out>/KILLS` (its kill outcomes `term` / `killed` / `kill_failed` younger than the window, read at start: `rate limit
seeded from KILLS: <n> kills ...` in `watch.log`), so a restarted `--service` watcher on the same `--out` keeps the
limit; no kill while `--hold-file`
(`$TOOLS_REMOTE/watchdog.hold`, i.e. `/workspace/tools/watchdog.hold`) is younger than `--hold-max-s` (900, watchdog.sh's
`HOLD_MAX`): a deliberate restart holds the engine and the watchdog relaunches nothing then (outcome `held`); no kill
with zero or several matching serve processes (`no_target` / `ambiguous`), nor when `ps` fails (`ps_failed`). A
refusal is not final for the episode: `held` / `no_target` / `ambiguous` / `ps_failed` are tried again every
`--on-hang-retry-s` (60) while the stall lasts (a hold goes stale, a relaunch race settles), and `rate_limited` once
the window has room again (the KILLS line carries `retry_in_s`); after a kill the episode tries no more. The router
(or a direct row) is never touched: the router ejects and re-admits the backend by its own health checks.

`--service`: the long-running mode beside watchdog.sh. One instance per `--out` (`<out>/hangwatch.pid`; a second
exits 1 while the first lives; the file is removed when the watcher stops), a stale `stop` file is removed at start,
`watch.log` rotates to `watch.log.1` at half
of `--log-max-mb` (100 MB for both), an exception in the loop is logged and the loop re-entered with a fresh stall
clock instead of ending the watcher, `--max-captures` defaults to 50 (each capture is a few MB). There is no
`--duration` in any mode: `touch <out>/stop` ends it.

Serving recipe (box-a serving the megakernel through the router, watchdog.sh running). In the engine's env (the box
file / serve env, before the relaunch that applies it): `SGLANG_MK_TP_PROBE=1 SGLANG_MK_TP_PROBE_EVERY=100` (the
TP-consistency probe every 100th step, counted by hangwatch from the serve log). Then, on the box:

```bash
mkdir -p /workspace/mk29/watch
setsid nohup python3 /workspace/mk29/harness/hangwatch.py --service --on-hang kill \
    --out /workspace/mk29/watch/serving > /workspace/mk29/watch/serving.nohup 2>&1 < /dev/null &
tail -f /workspace/mk29/watch/serving/watch.log      # ON-HANG KILL / engine booting / engine up lines
cat /workspace/mk29/watch/serving/KILLS               # one JSON line per kill decision
touch /workspace/mk29/watch/serving/stop             # ends it within one poll (and before a deliberate restart
                                                     # if you do not want it to watch the boot)
```

The defaults find box-a's engine (checked 2026-10-30 against the live box, read-only): `ps -ww -A -o pid,pgid,args |
grep 'sglang serv[e]'` shows one process, `284772 284772 /opt/sglang-mk29/bin/python3 /opt/sglang-mk29/bin/sglang
serve --tp-size 4 ...` carrying `--port 8002 --host 127.0.0.1` (relaunch.sh's `--port "$ENGINE_PORT"`, the box file's
`ENGINE_PORT=8002`). Fed that box's whole `ps -ww -A -o pid=,ppid=,pgid=,stat=,etime=,args=` output (23 rows),
hangwatch's own `ps_table` + `serve_targets` with the default `--serve-pattern` and `--serve-port` 8002 (from `--url`)
select exactly pid 284772 (ppid 1, pgid 284772: it leads its group, 7 members, so a kill signals the group; no other
row matches the pattern). A pid differs after each relaunch; the match does not.

In-process stack fallback (`--dump-signal`, default `none`, i.e. off; `USR2` only where a faulthandler site file registers it first, since our engines' own USR2 catcher is LLVM's SA_RESETHAND handler and the rank dies; `--dump-wait`, default 2 s). box-a's container has no
`CAP_SYS_PTRACE` and yama `ptrace_scope` is 1, so py-spy gets `Permission denied` on every engine pid unless the
engine opts in. `raw/s5-soak/site/sitecustomize.py`, first on the engine's `PYTHONPATH` (`raw/s5-soak/s5-probe.env`),
does both halves at every engine interpreter's start: `prctl(PR_SET_PTRACER, ANY)` so py-spy may attach, and
`faulthandler.register(SIGUSR2, all_threads=True)` so `kill -USR2` writes every thread's Python stack to the rank's
stderr (the serve log), even while it sits in a C call (NCCL, a CUDA sync); then it runs the system sitecustomize it
shadows. In a capture, each scheduler/TP rank whose `pyspy-<role>-<pid>.txt` shows no `Thread` stack (or every rank,
without py-spy) gets the signal, ONLY if `/proc/<pid>/status` `SigCgt` shows it caught (the default action of
`SIGUSR2` is to terminate), one rank at a time: hangwatch waits `--dump-wait` s and writes the dump's own lines
(`Thread 0x` / `Current thread 0x` headers, `File "` lines) from the serve-log bytes appended since the signal to
`faulthandler-<role>-<pid>.txt` (key redacted); a rank that does not catch it gets a `not sent: ...` file instead.
`--dump-signal` takes a name (`USR2`, `SIGUSR2`), a number, or `none` (off); it is checked at start, so a typo (or an
uncatchable `KILL` / `STOP`, or a negative `--dump-wait`) exits 2 before the watcher runs, never mid-capture.
Raise `--dump-wait` if a rank's dump is cut short (a slow log writer).

It tails the serve log (`--serve-log`, default `$SERVE_LOG` or `/workspace/sglang-serve.log`) incrementally and counts
`MK_TP_PROBE MISMATCH` / `MK_TP_PROBE ok` lines from its start position (`--scan-from-start` counts the existing log
too); the first 200 MISMATCH lines go to `mk_tp_probe_mismatch.txt` (redacted). `state.json` (counters, last metrics,
episodes, probe counts) is rewritten every poll; `watch.log` has a status line every 60 s.

py-spy: `--pyspy PATH`, else `/opt/pyspy/bin/py-spy`, else one on PATH; if none, it creates the separate venv
`/opt/pyspy` (`python3 -m venv /opt/pyspy && /opt/pyspy/bin/pip install py-spy`) at start, never inside an engine
venv (`--no-install` skips it; captures then carry `pyspy-MISSING.txt`).

## Offline test (Mac): test_harness.py

`python3 test_harness.py -v` (~120 s; 28 tests, run to OK on the Mac's `/usr/bin/python3` 3.9.6 and Homebrew 3.15.2)
runs both scripts against an in-process stdlib fake OpenAI-compatible engine on a free 127.0.0.1 port (SSE chat completions with reasoning, content, streamed tool calls
with every 9th malformed, usage, an HTTP 500 every 37th streaming request; `/metrics` with a tp_rank="1" decoy and
`/health`, switchable to stall / 503 / hung health / missing metrics / all-down). It checks: `sample` is byte-deterministic and
covers every kind; long docs land within 0.95-1.1x of their token target and hold every planted fact; a 1,4,16 soak
with idle gaps records every phase, only injected errors, catches malformed tool calls, scores JSON and arithmetic
right with every check's outcomes summing to all sent (the injected 500s land in `error`), and its exit gate passes at
`--max-error-rate 0.1` and fails at 0; `load --greedy-rounds 2 --stop-after-greedy` writes 2x32 greedy rows and stops
early (exit 1 iff any request failed at `--max-error-rate 0`); exact-number scoring (`2401` vs `24010`, digit groups,
CJK), the five outcomes, the gate, and qlib's `extract_answer(..., fallback=False)`; hangwatch catches a `no_token`
and a hung-`/health` stall, recovers between them, writes a 300-line `serve-tail.txt` that is the LAST 300 lines of a
5 MiB log with the key redacted, counts 2 probe ok / 1 MISMATCH, leaks the key into no file, warns exactly once when
`/metrics` lacks the running/progress series; a `prefill` mode (running > 0, generation flat for 6x `--stall-s`, with
only the realtime prefill counter, then only `prompt_tokens_total` jumps, then only serve-log `Prefill batch` lines
moving) raises no stall until everything goes flat, then `no_token`; a `queued` mode (queue 2, running 0, nothing
moving) raises `queue_wedge` after idle (nothing in flight) raised none. In-process units: `sig_num` (names, numbers,
`none`, rejects `USR3` / `SIG_DFL` / `0` / `999`), `main` exits 2 on a bad `--dump-signal` / `KILL` / negative
`--dump-wait` before creating `--out`, `catches` reads a fake `/proc/<pid>/status` `SigCgt`, `fault_dumps` signals
only scheduler/TP ranks without a py-spy stack that catch the signal, in rank order, keeps only that rank's dump lines
(redacted), and does nothing at `--dump-signal none`; `scan_log` counts real-format `MK_TP_PROBE MISMATCH` lines (from
its start position, a half-written line once complete, all with `--scan-from-start`) and keeps them redacted;
`--on-hang kill` against sleepers standing in for `sglang serve` (own session like relaunch.sh's `setsid`, a `sleep`
child in the group, argv carrying a marker and `--port`): `serve_targets` picks only the top-level process on the
right port (never a forked child, another port, hangwatch, a zombie) and `etime_s` parses ps; `on_hang_kill` TERMs the
group (the child goes too), SIGKILLs a TERM-ignoring sleeper after `--on-hang-term-s`, refuses a third kill at
`--on-hang-max 2` (`rate_limited`, the process lives), a fresh hold file (`held`; a stale one kills) and another port
(`no_target`), and leaves a retry time (none after a kill, `--on-hang-retry-s` after `held` / `no_target` /
`ps_failed`, the window's next opening after `rate_limited`); a `KILLS` from an earlier watcher seeds the rate limit
with only its kill outcomes inside the window (torn and foreign lines skipped); `--service` rotates `watch.log` under
the cap and a second instance on a live `hangwatch.pid` exits 1.
End to end, `--service --on-hang kill --boot-grace-s 4 --on-hang-max 2` against the fake gone `down` (no `/metrics`,
no `/health`): the stale `stop` is removed, a second `--service` exits 1, the `health` HANG's `evidence done` precedes
`ON-HANG KILL pid=<sleeper> pgid=<sleeper>`, a relaunched sleeper raises no HANG inside the grace and is killed
after it, a third is only `rate_limited`, and a clean stop removes `hangwatch.pid`; with a fresh hold file a `down`
stall's kill is refused (`held`) and retried every `--on-hang-retry-s 1`, and once the hold goes stale the same
episode kills; at `--on-hang none` a young serve process with no `/health` yet raises no
HANG for 4 s (`engine booting`), `engine up` at its first 200, then a `health` HANG when it dies, and nothing is
killed;
`score`'s empty-vs-truncated rule; the SSE reader fills `reasoning` from `reasoning_content`. No request goes to box-a.
