# Stability on 4× B300: hangs, rank desync, serving hours, restarts

Machine-readable rows: `data/measurements.csv` (area `stability`). Statistics: with 0 events in T hours, the
one-sided 95% upper bound on the event rate is −ln(0.05) / T ≈ 2.996 / T per hour (`methodology.md`).

## Why this needed measuring

The megakernel is a persistent kernel spanning the four GPUs with its own cross-rank synchronisation; a rank that
diverges from the others (a different sampled token, a different batch) can deadlock the step. Public reports
before we started: a community 4× B200 port of the megakernel "hung twice in 12.5 h in the published build"
(`box-c/qa/window1-20261027/briefs/brief-v2.md` item 7), about one hang per 6 h
(`box-c/qa/box-a-megakernel-20261029/README.md` "Hangs"). NVIDIA's own article publishes no stability figures.

## The safety layer we run (all B300 serving since 2026-10-30)

| layer | what it does | source |
|---|---|---|
| TP-consistency probe (`SGLANG_MK_TP_PROBE=1`, every 100 decode steps in serving, every 10 in soaks; `ASSERT=1`) | each rank hashes its step state and compares across TP ranks; a mismatch fail-stops the engine instead of letting it hang or emit divergent text | `box-c/qa/box-a-megakernel-20261029/merge-latest.md` "TP-consistency probe" |
| `hangwatch.py --service --on-hang kill` | watches step progress and the probe; on a stall it kills the engine so the watchdog relaunches it | `box-c/qa/box-a-megakernel-20261029/README.md` "Hangs" |
| `watchdog.sh` | relaunches a dead engine and re-runs prewarm | `box-c/07-operations.md` |
| router liveness (the serving router) | a backend that misses its 2 s liveness goes DOWN on every router pod; tested with a TP rank `kill -STOP` for 9.9 s: row DOWN within ≤ 5.6 s on every dev pod, in-flight streams paused ~10 s and finished normally | `box-c/qa/box-a-megakernel-20261029/raw/box-a-return/NOTES.md` |
| sampler pin `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample` and upstream sglang #38565 (deterministic top-p renorm at TP > 1, in config X') | keeps the ranks' random draws identical | `fleet-history/box-a/RETRY-MEGAKERNEL.md` |

## Synthetic soaks (off pool)

| build | soak | loaded h | requests | hangs / probe mismatches | probe checks per rank | 95% bound | source |
|---|---|---|---|---|---|---|---|
| first boot, pre-megakernel (10-26) | 1,800 s resume soak | 0.5 | 5,472 resumes | 0 hangs, 0 corrupt, 0 retractions | — | — | `box-c/qa/box-a-bringup-20261026/README.md` |
| old plugin MK3 (10-28) | 30-min synthetic + live window | 1.08 (0.58 live) | 1,502 synthetic + 245 live | 0 / no probe | — | 2.77 per h | `box-c/qa/box-a-tuning-20261026/README.md` |
| config X `9ec65dacaf`, probe every 10 (S5) | 2 × 1 h synthmix (seeds 51, 52) | 1.445 | 13,410 | **0 / 0** | 852,300 | **2.07 per h** | `box-c/qa/box-a-megakernel-20261029/raw/s5-soak/NOTES.md`, `data/raw/megakernel/raw__s5-soak__soak-stats.txt` |
| MK-syncsample variant (S6b) | 1 h synthmix (seed 61) | 0.729 | 6,963 | 0 / 0 | 405,980 | 4.11 per h | `box-c/qa/box-a-megakernel-20261029/raw/s6b-syncsample/NOTES.md` |
| config X' on box-b, bring-up soak | 15 min | ~0.25 | — | 0 / 0; 0 aborts, 0 tracebacks | 28,408 probe_ok | — | `box-c/qa/box-b-bringup-20261030/NOTES.md` |

## Serving in the production pool (real traffic, megakernel)

Source: `fleet-history/box-a/RETRY-MEGAKERNEL.md` (hours by load band from vmsingle
`sglang:num_running_reqs`, 15 s samples; loaded hours are a floor) and `box-c/qa/box-a-outage-20261031/NOTES.md`.

| build | window (UTC) | wall in pool / loaded | hours at 0 / 1 / 2–4 / ≥ 5 running | peak running | requests | TP probe ok / mismatch | hang episodes / kills |
|---|---|---|---|---|---|---|---|
| X `9ec65dacaf`, box-a | 10-30 07:24:22–12:01:07 | 4.61 h / 0.650 h | 3.96 / 0.49 / 0.16 / 0 | 4 | 5,462 chat 200s | 283,044 / 0 | 0 / 0 |
| X' `4943d6ed90`, box-a | 10-30 12:51:00–14:48:44 | 1.96 h / 0.342 h | 1.62 / 0.27 / 0.08 / 0 | 3 | 1,946 chat 200s (+2 409s retried to 200, a cache-ref check, not the engine) | 114,256 / 0 | 0 / 0 |
| X' `4943d6ed90`, box-a, one engine process | 10-30 12:34:32 → 10-31 01:39:56 (13 h 05 min uninterrupted) | — | — | 6 seen at read | — | 568,072 / 0 | 0 / 0; watchdog: no relaunch |
| X' `4943d6ed90`, box-a, whole tenure (includes the two rows above) | 10-30 12:51:00 → 10-31 17:42:04 (last in-pool sample 16:39:37) | 20.61 h / **6.250 h** | 14.36 / 4.18 / 1.76 / 0.30 | 13 | — | — | 0 (below) |
| X' `4943d6ed90`, box-b | 10-30 17:56:00 → 10-31 17:49:32 | 22.56 h / **9.179 h** | 13.38 / 4.08 / 3.43 / 1.68 | 14 | — | — | 0 (below) |
| X' `9a9ede7ae0`, box-a | 10-31 17:42:04 → 11-01 03:44:52 (row held from 03:45:07) | 8.01 h / **3.287 h** | 4.73 / 2.16 / 1.08 / 0.05 | 7 | — | — | 0 (below) |
| X' `9a9ede7ae0`, box-b | 10-31 18:06:10 → 11-01 04:41:39 (the read) | 10.48 h / **5.471 h** | 5.01 / 3.06 / 2.16 / 0.25 | 14 | — | — | 0 (below) |

The last four rows (read 2026-11-01 04:41:44Z): prod sbr-vmsingle raw samples of
`sglang:num_running_reqs{backend="<box>"}` and `sbr_backend_health_state{job="sbr-prod",backend="<box>"}` (one
series per router pod), from 10-30 12:00Z. A sample counts when every router pod reads the row healthy (0) — held
(3) or DOWN (4) on any pod drops it, so out-of-rotation boots and benches never count — and loaded when
`num_running_reqs > 0`; each sample is 15 s. The method reproduces the 12:51–14:48Z window's 0.342 h exactly.
Tree windows from `box-c/qa/prod-merge-20261030/raw/gate/NOTES.md`, `box-c/qa/box-b-bringup-20261030/NOTES.md`,
`box-c/qa/sglang-takes-20261031/ROLL-box-a.md` and `box-c/qa/box-b-roll-20261031/NOTES.md`. Script, per-sample table
and summary: `data/scripts/xprime_loaded_hours.py`, `data/raw/x-prime-serving/` (the raw vmsingle answers carry
router pod addresses and door host names, so they are not copied). The in-pool load includes our own synthetic
fan-outs through the router (`fleet-ops.md` §3) beside the production traffic.

Hang events in those windows: 0. Every engine restart in them (a vmsingle scrape gap > 60 s: box-b 10-30 20:56Z,
23:03Z, 10-31 00:52Z, 17:50Z; box-a 10-31 03:02Z, the 09:46–17:33Z takes session, 18:12Z, 19:36–21:04Z) began
with the row held on every router pod, i.e. a deliberate restart; no pod read either row DOWN (4) while it was in
the pool (box-a's only DOWN samples, 10-31 09:49:25Z, fell inside a hold). box-b's hangwatch (`HANG` and
`watch.log`, read 11-01 04:43Z) holds one dump, 10-31 17:50:43Z, written by the roll's own restart under hold
(`box-c/qa/box-b-roll-20261031/NOTES.md`). box-a's hangwatch shows 0 episodes through 10-31 01:38:46Z
(`box-c/qa/box-a-outage-20261031/NOTES.md`); its later log was not read (ssh refused 11-01 04:42Z, the row held since
03:45:07Z), so after that the count rests on the vmsingle evidence above.

### The hang-rate bound, derived

Rule: with 0 events observed in T loaded hours, the one-sided 95% upper confidence bound on a Poisson rate is
λ ≤ −ln(0.05) / T = 2.996 / T per loaded hour (the "rule of three", 3 / T, rounded). A loaded hour is an hour with
≥ 1 running request (15 s vmsingle samples for serving; soak hours are all loaded). Builds are never pooled with
each other; soak and serving hours of the same build are.

| build | hours in the denominator | T (loaded h) | events | bound = 2.996 / T |
|---|---|---|---|---|
| config X `9ec65dacaf` | S5 soaks only (2 × 1 h synthmix, seeds 51 and 52) | 1.445 | 0 hangs, 0 probe mismatches | **≤ 2.07 per loaded h** |
| config X `9ec65dacaf` | S5 soaks 1.445 + production serving 10-30 07:24–12:01Z 0.650 | **2.095** | 0 | **≤ 1.43 per loaded h** (config X's best) |
| MK-syncsample (not served) | S6b soak, seed 61 | 0.729 | 0 | ≤ 4.11 per loaded h |
| config X' `4943d6ed90` | production serving 10-30 12:51–14:48Z (the first vmsingle window) | 0.342 | 0 | ≤ 8.76 per loaded h (the source prints 8.77, i.e. 3 / T) |
| config X' `4943d6ed90` | all of its production serving, box-a 6.250 + box-b 9.179 (10-30 12:51Z → 10-31 17:49Z) | **15.429** | 0 | **≤ 0.194 per loaded h** (the best bound on the record) |
| config X' `9a9ede7ae0` (serving now) | production serving, box-a 3.287 + box-b 5.471 (10-31 17:42Z → 11-01 04:41Z) | **8.758** | 0 | **≤ 0.342 per loaded h** |
| old plugin MK3 (10-28) | 30-min synthetic + 0.58 h live | 1.08 | 0 | ≤ 2.77 per loaded h |

The 13 h 05 min uninterrupted config X' engine process (10-30 12:34Z → 10-31 01:39Z, 0 hang episodes) lies inside
`4943d6ed90`'s 15.429 h. The two X' trees differ by four upstream picks (`fork-and-config.md`), so they are not
pooled; counted as one config they would give 24.188 h and ≤ 0.124 per loaded h, under the community port's ~1 per
6 h (0.167). Sources:
`fleet-history/box-a/RETRY-MEGAKERNEL.md` (the serving windows and the 2.095 h bound),
`box-c/qa/box-a-megakernel-20261029/raw/s5-soak/NOTES.md`, `box-c/qa/box-a-megakernel-20261029/raw/s6b-syncsample/NOTES.md`,
`box-c/qa/box-a-tuning-20261026/README.md`.

The 10-31 01:45Z "box-a engine down" report was a provider console telemetry dropout (it showed 0 °C on a running
B300); the engine, probe and hangwatch were all fine (`box-c/qa/box-a-outage-20261031/NOTES.md`).

## Restarts, boots and roll-outs (time out of rotation)

| event | box | measured | source |
|---|---|---|---|
| fresh rent → serving production | box-b | ~72 min (created 16:44Z, repack copy 16:59–17:29Z at ~290 MB/s for 510 GB, READY 17:43Z, prod 17:56Z) | `box-c/qa/box-b-bringup-20261030/NOTES.md` |
| engine boot to READY (config X') | box-a / box-b | 703–723 s (config X 795 s) | `box-c/qa/prod-merge-20261030/raw/gate/NOTES.md`, box-b NOTES |
| stop → start smoke, 3 cycles, no hand step | box-b ×2, box-a ×1 | start → back in on every router pod 14 min 8 s / 14 min 8 s / 14 min 22 s; 0 pages | `box-c/qa/autopool-20261030/SMOKE.md` |
| stop → start smokes after sessions | box-a | 14 min 28 s; 14.6 min; 13 min 56 s | `box-c/qa/box-b-roll-20261031/NOTES.md`, `box-c/qa/sglang-takes-20261031/NOTES.md` |
| hard cut and roll to a new tree | box-b | out 16 min 38 s; restart incl. prewarm (824 requests, 0 errors) 13 min 10 s; box-a carried all traffic meanwhile | `box-c/qa/box-b-roll-20261031/NOTES.md` |
| rollback drill | box-a | 13 min 21 s restart → READY + prewarm | `box-c/qa/sglang-takes-20261031/NOTES.md` |

## Caveats

- **Bounded, not proven low.** The best bound is ≤ 0.194 hangs per loaded hour (config X' `4943d6ed90`, 15.43
  h); the tree serving now is at ≤ 0.342 (8.76 h). Excluding the community port's ~1 per 6 h at 95% needs ~18
  loaded hang-free hours on one build: `4943d6ed90` stopped serving at 15.4, `9a9ede7ae0` needs ~9 more hang-free
  loaded hours.
- **Production load is light.** In-pool running requests peaked at 13–14, but 87% (`4943d6ed90`) and 97%
  (`9a9ede7ae0`) of the loaded hours were at 1–4 running, and some peaks may be our own synthetic fan-outs; the
  high-concurrency evidence is the synthetic soaks (synthmix up to the mix's widths) and the bench's C16.
- **Sampling coverage.** Every soak ran T 1.0 / top_p 1.0 / no top_k. Production drops top_k and clamps top_p
  to [0.95, 1.0] only when a client sets it; the top_p < 1 path (upstream #38565's) is covered by the fix in X',
  not by a soak.
- Probe checks are per rank; "0 mismatches" means no rank ever disagreed with the others at a probed step.
