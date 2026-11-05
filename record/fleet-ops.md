# Fleet operations on two 4× B300 boxes: autopool, rejoin times, load split, red/blue rolls

How box-a and box-b run as a pair behind the router (the serving router, "sbr"): stopping and starting a box from the
provider console with no hand step, how long a box takes to come back, how traffic splits between the two, and how a
new engine tree is rolled across both without taking the model offline. Machine-readable rows: `data/measurements.csv`
(area `stability`, rows T07, T08). Router-side sources are in the serving router (cited as
its `docs/research/<file>`).

All production reads here are aggregate counts and timings; no prompt or completion was read or copied.

## 1. The setup

- The two boxes form one speed group (`b300-4x` = [box-a, box-b]) in the router pool `box-c-prod`; the router places
  each turn by cache affinity and a load/latency score. Automatic rerouting to any outside provider is off: with no
  own box serving, requests fail (the serving router's `docs/research/2026-10-31-no-reroute.md`).
- **Autopool** (#137, live 10-30): the ask was that a provider stop take the box out of the router pool and a start
  put it back by itself, since typically only one of the two boxes needs to serve and the second is backup capacity.
  The controller watches the provider's instance state; on a stop it opens a held maintenance window for the box on every
  router pod (park), and on a start it waits for the engine's health through its door and the warm-up prewarm, restarts
  the KV-event tunnel, and rechecks the box back in on every pod (`box-c/qa/autopool-20261030/DESIGN.md`;
  the serving router's `docs/research/autopool-golive-20261030.md`).
- What survives a provider stop: the whole container filesystem (model, venv, onstart script); processes do not. Each
  start runs `/root/onstart.sh`, which brings up the door, engine, watchdog and hangwatch by itself (`DESIGN.md` §1.1).

## 2. Stop → start → back in rotation (real `hostctl stop` / `hostctl start`, no hand step)

Sources: `box-c/qa/autopool-20261030/DRILL.md` (drill, 10-30) and `box-c/qa/autopool-20261030/SMOKE.md` (cycles 1–3,
10-30/31); per-10 s timelines in `box-c/qa/autopool-20261030/raw/`. "Controller" times are autopool's log; the metric
(`sbr_backend_health_state` per pod) lags 20–60 s.

| run | box | stop → out on every pod | streams cut at stop | the provider `running` after start | engine READY after start | prewarm | start → back in on every pod | pages |
|---|---|---|---|---|---|---|---|---|
| drill (#143), 10-30 20:55Z | box-b | 14 s | 0 (box-a took all traffic) | +22 s | 11.9 min (682 s after launch) | 60 s, 824 requests, 0 errors | **14.1 min** | none in prod (one dev ticket) |
| cycle 1 (#181), 10-30 23:03Z | box-b | 11 s | 4 of 7 running (container died ~80 s after the stop) | +34 s | 11 min 55 s (688 s) | 60 s, 824 requests, 0 errors | **14 min 8 s** | 1 page (SbrMidStreamAbortSpike) → fixed by #191 |
| cycle 2 (#206), 10-31 00:52Z | box-b | 13 s | 1 (container died ~11 s after the stop) | +36 s | 11 min 56 s (688 s) | 60 s, 824 requests, 0 errors | **14 min 8 s** (all 5 router pods redeployed mid-cycle) | none |
| cycle 3 (#220), 10-31 03:02Z | box-a | 13 s | 7 of 8 (synthetic proof streams; no user stream) | — | — | — | **14 min 22 s** | none (inhibited by the park alert, #191) |

Drain-first stop (#259, box-a, 10-31; `box-c/qa/box-a-maintenance-20261031/stop/` and `README.md`): a 300 s hand window
on every pod, 63 s fully drained (0 in flight, 0 running), `hostctl stop` at 09:45:40Z, autopool parked the row 9 s
later, the provider `exited` at 09:46:32Z; `mid_stream_aborts` +0 and nothing paged. box-a then stayed stopped by the user's
choice (to save money) until the 10-31 sessions restarted it. The same record holds the provider's 24 h GPU-maintenance notice
for box-a (05:38Z), which the provider never executed: the provider read `running` on every 15–30 s sample from 05:21Z to the user's
stop (`box-c/qa/box-a-maintenance-20261031/MAINT-TEST.md`). It has no start → rejoin timing.

After each return, real traffic resumed with 0 errors: 6, 43 and 115 requests (cycles 1–3, the last in its first
9 min). IP and ports came back unchanged every time. The drill's park lasted 29.5 min in all (held 20:56:12 → recheck
21:25:42), of which ~12 min was the engine loading and ~1 min the prewarm.

Other stop → start smokes (box-a, after test sessions, no autopool hand step): 14 min 28 s (10-31 18:12:54Z start →
serving 18:27:22Z, `box-c/qa/box-b-roll-20261031/NOTES.md`); 14.6 min (16:07:02 → 16:21:37Z) and 13 min 56 s
(20:51:32 → 21:05:28Z) (`box-c/qa/sglang-takes-20261031/NOTES.md`). A fresh rent of box-b reached production in 72 min
(`box-c/qa/box-b-bringup-20261030/NOTES.md`).

## 3. Load split between the two boxes

### The problem: cache affinity keeps conversations where they started

Read-only audit #188, 10-30 (`box-c/qa/box-b-bringup-20261030/raw/router/lb-audit/AUDIT.md`; summarised in
the serving router's `docs/research/sibling-ownership-balance-20261030.md`). One heavy user's agent
traffic, clean windows with both boxes in rotation:

| read | box-a | box-b |
|---|---|---|
| turns | 2,939 (59.6%) | 1,993 |
| prompt tokens | 628.9 M (69.0%) | 282.1 M |
| turns placed by cache affinity | 2,850 | 1,855 |
| free (non-affinity) placements | 89 | 138 (60.8%) |
| conversations born with both up, turns | 1,408 | 1,711 |

By segment box-a took 68.5%, 72.4% and 32.5%; during each out span (drill, bench, smoke cycle) box-a took 100%.
Each ~30 min out span re-homes the active conversations to the other box, and affinity keeps them there after the
return (the returning box is also cold: KV used 15,104 vs 354,304 tokens).

10-31 ~01:3xZ incident (#212): twelve forked agent conversations sharing one long cached prefix all landed on box-a
(GPU 95% vs box-b 0%; 740 vs 212 requests in 30 min on the user's key), each streaming at 221–455 tok/s
(the serving router's `docs/research/sibling-switch-decode-20261031.md`).

### The fix: the sibling decode switch (#212, released #219, 10-31)

The router now prices a turn's **own decode** on each box at the box's live concurrency (from box-a's measured curve:
1.181 / 1.416 / 1.835 / 2.525 ms per token at C1 / C2 / C4 / C8 — `speed.md` §2) against the one-time re-prefill of
the prefix it would strand (box-b's ~69k tok/s line: ~1.5 s per 100k tokens). Worked example from the source: a
2,048-token turn takes 2.4 s alone and 5.2 s beside seven others — 2.8 s more on every turn — against 1.5–2.2 s once.
The count-only balance became speed- and load-aware (25 ms tie band).

Synthetic fan-out through the prod router (52k-token fresh prefix, parent turn cached on one box, then forks):

| run | forks | max_tokens | parent cached on | forks box-a / box-b |
|---|---|---|---|---|
| A | 10, 1 s apart | 2,048 | box-b | 4 / 6 |
| B | 12, 0.3 s apart | 8,192 | box-b | 5 / 7 |
| C1 | 8 | 8,192 | box-b | 5 / 3 |
| C2 (sent while C1 decoded) | 8 | 8,192 | box-b | 3 / 5 |
| D (after box-a's rejoin) | 12 | 8,192 | box-b | 6 / 6 |
| E (distinct user per fork) | 10 | 8,192 | box-a | 4 / 6 |

No run left a box idle (offline tests with the switch off: 12 / 0). In B, box-a's 5 forks averaged 16.9 s of decode and
box-b's 7 averaged 16.7 s: about **490 tok/s per stream at 5–7 concurrent streams on each box**. A fork that
re-prefilled the 52k prefix on the other box paid 0.73–0.90 s of first-token time once; every later fork hit 51,968
cached tokens.

The user's own traffic, the hour after the release (02:57:59–04:01:47Z): 917 box-a / 1,246 box-b (42.4% box-a; box-a
was out 03:02–03:18 for smoke cycle 3). From box-a's rejoin: 851 / 651 (56.7% box-a); free placements 30 / 28;
first-token p50 264 ms on box-a, 187 ms on box-b. Before the release the same key ran 70.5% box-a (474 / 36 in one
15-min bucket). 1,444 of the 1,502 turns after the rejoin were affinity-forced.

Live off-switch drill (dev, 10-31 07:25–07:38Z): with the switch off, 0 moves in 45 decisions; on, 3 of 17 and 2
of 13 turns moved; it follows the published config within a publish, no restart.

### Live per-box behaviour (both serving, 10-30 21:26–22:01Z)

See `speed.md` §2: box-b ITL p50 1.087 ms vs box-a 1.151 ms; router TTFT p50 306 vs 439 ms; learned prefill line
~18% higher on box-b.

## 4. Red/blue rolls: a new tree across both boxes with the model online

The user's order (10-31): "when its validated on box-a, you red/blue it by having box-a serving and then reboot box-b
with the latest so they are both in-sync". Each step is a hard cut and roll: the box's router row is held out on
every pod (`sbr admin backend maintenance … --hold-until-recheck`), the engine restarted on the new tree, gated, and
rechecked back in; the other box carries all traffic meanwhile.

| step | box | out (hold → recheck) | what ran in the window | first real traffic after | source |
|---|---|---|---|---|---|
| red (#341), 10-31 | box-a → `9a9ede7ae0` | 16:39:31 → 17:41:51Z (62 min) | three boots (C3, control K3, C4), each ~13–14 min to READY + prewarm (824 requests, 0 errors), bench, then in-window gates: default-thinking probe, smoke 10/10, TP probe 0 mismatches | 17:47:16Z (5 min after recheck; affinity holds traffic where the prefixes are) | `box-c/qa/sglang-takes-20261031/ROLL-box-a.md` |
| blue (#310), 10-31 | box-b → `9a9ede7ae0` | 17:49:32 → 18:06:10Z (16 min 38 s) | one restart incl. prewarm, 13 min 10 s; smoke 10/10; TP probe 1,920 checks, 0 mismatches | 18:09:48Z; 19 decisions settled, mean TTFT 185 ms | `box-c/qa/box-b-roll-20261031/NOTES.md` |

While box-b was out, box-a took every routing decision (339 in the window, none to box-b). A rollback drill on box-a
timed 13 min 21 s from restart to READY + prewarm (`box-c/qa/sglang-takes-20261031/NOTES.md`). Router-side, the
known-bad test (a TP rank frozen 9.9 s) had the row DOWN on every dev pod in ≤ 5.6 s (`stability.md`).

## Caveats

- **Restart time is engine load, not ops.** ~12 of the ~14 minutes are the engine loading 510 GB of weights and
  capturing graphs; the controller's own work is seconds.
- **Streams in flight at a stop are cut.** A console stop drains nothing; the hold stops new picks only. Cycle 1
  cut 4 streams and paged (fixed by #191's inhibition); cycle 3 cut 7 synthetic streams and paged nothing.
- **Load split is driven by cache affinity**, which is right for cost (a re-prefill per moved conversation) but
  uneven after any out span. The decode switch's benefit (first token and decode vs staying) is not yet measured;
  the source lists it as open.
- **Synthetic fan-out ran once or twice per shape**, one after another, on production with live traffic beside it.
- **A provider stop may not come back**: a stopped instance waits for its GPUs and "may wait indefinitely"; box-b has no
  static IP. That is why only one box is stopped at a time, and why the drill stopped box-b: the user judged that
  box-a, on a lower spot rate, "may not start up again" (`DRILL.md`).
