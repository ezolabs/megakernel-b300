# Quality suite: the classes the earlier gates never tested

This suite runs on box-a against the engine at `http://127.0.0.1:8002` (model `DeepSeek-V4.1-Flash`;
the key is read from `/etc/environment`). Its purpose is to compare two arms, the megakernel (MK) and the
final, on the request classes the earlier gates never scored:

- gsm8k at full size
- long-context retrieval
- structured output
- tool-using agent sessions
- non-English prompts
- images
- quality while the engine is under load

The code is stdlib Python 3.8+, with no installs. Every suite is one command and fits inside 1 hour at 16
streams on the final. Each prints its expected wall time before it starts and a progress/ETA line every 30 s.
Each one writes two files:

- `OUT/<suite>-<cell>.jsonl`: one line per item, with the answer's tail, the score, tokens and wall time.
- `OUT/<suite>-<cell>.summary.json`: every metric as k/n with a Wilson 95% CI, and means with a bootstrap
  95% CI.

`compare.py` reads any two sets of these files.

- Box copy: `/workspace/mk29/harness/quality/`. Public data is cached in its `data/` folder: the gsm8k
  `test.jsonl` and the MGSM TSVs, already downloaded on box-a.
- Repo copy: `box-c/qa/box-a-megakernel-20261029/harness/quality/`.
- All prompts are synthetic or come from public datasets. No production data is used.

## Commands and expected wall time

Run these on the box from `/workspace/mk29/harness/quality`. `OUT` is the arm's raw folder, e.g.
`/workspace/mk29/raw/quality-final`. The wall times are estimates from the final's box-a speed: 3,990 tok/s
aggregate at C16, ~31k tok/s cold prefill, and gsm8k 100/16 in 14 s (`box-c/qa/box-a-ship1-20261027/README.md`).
A slower arm takes longer.

| # | suite | command | items | expected wall |
|---|---|---|---|---|
| 1 | gsm8k, full test set | `python3 gsm8k_full.py --out OUT` | 1,319 × {t0, prod} | ~3–5 min per cell |
| 2 | needles / retrieval | `python3 needles.py --out OUT` | 4 lengths × 5 depths × 4 reps = 80 | ~20–30 min, capped at 55 |
| 3 | JSON / response_format | `python3 json_format.py --out OUT` | 100 × {schema, object} | ~2–4 min |
| 4 | multi-turn agent sessions | `python3 agent_sessions.py --out OUT` | 100 sessions, 4–8 turns | ~4–8 min |
| 5 | non-English (MGSM) | `python3 mgsm.py --out OUT` | 6 langs × 20 = 120 | ~1–2 min |
| 6 | images (only if served) | `python3 images.py --out OUT` | 80 + 20 no-image controls | ~1–3 min, or seconds when not served |
| 7 | gsm8k under 16-stream load | `python3 under_load.py --out OUT` | 500 × {t0, prod} | ~6–9 min per cell + ~1 min |
| – | all of the above | `bash run_all.sh OUT TAG [suites]` | | ~50–80 min |
| 8 | compare two arms | `python3 compare.py dirs OUT_FINAL OUT_MK --labels final,mk --md cmp.md --json cmp.json` | | seconds |
| 9 | DSML gate, two-sample | `python3 ../dsml_compare.py FINAL_DSML MK_DSML --labels final,mk` | | seconds |

Every suite takes these options:

- `--concurrency/-c` (default 16; 2 for needles; 8 foreground for under_load)
- `--effort` (default `high`, the serving effort; `none` is sent and turns thinking off; `default` omits it)
- `--base-url`, `--model`, `--api-key`, `--env-file`
- `--tag` (stored in every summary)
- `--timeout`

Prod sampling means T 1.0 / top_p 1.0: DeepSeek's official sampler, which the DSML gate also uses.

## What each suite measures

1. **gsm8k_full.py.** All 1,319 test problems in two cells:
   - `t0`: T 0.
   - `prod`: prod sampling.
   - Settings: effort high, max_tokens 8192. The 4096 gate cap truncates a few traces; truncation still
     scores as wrong and is reported as `truncated`.
   - Prompt and scoring are `gsm8k.py`'s: the number on the last `#### n` line, else the last
     number in the reply.
   - Metrics: `exact` (primary), `format`, `truncated`.
   - `--n 500` takes the same seed-7 subset as `gsm8k.py`, so its ids pair with the full run.
2. **needles.py.** Synthetic haystacks of number-free prose at 100k / 250k / 500k / 900k prompt tokens,
   with 50 key/code records planted in each.
   - The question asks for 3 of the codes. The 3 asked records sit together at the item's depth (0.05,
     0.25, 0.5, 0.75 or 0.95); the other 47 are distractors.
   - Scoring is exact: `all3` (primary) and `key` overall, plus `all3@<len>`, `key@<len>` and `key@d<depth>`.
   - It reads the served context (`/v1/models` max_model_len, `/get_server_info` context_length) and skips
     any length that does not fit with max_tokens.
   - Prompt size comes from the served `tokenizer.json`. The box-a dry run measured 99,997 / 249,999 /
     500,002 / 899,999 tokens against the targets. Without the `tokenizers` package it falls back to a
     chars/token ratio calibrated on the engine's usage.
   - Each run carries a fresh nonce, so items are cold prefills.
   - It measures the first item of each length alone, then drops whole repetitions (never a depth) if the
     projection would pass `--budget-min` (55).
   - Setting: T 0.
   - `--dry-run` builds the prompts and sends nothing.
3. **json_format.py.** Five deterministic extraction families: person, invoice with total, event with a
   date pattern and an enum, sentiment, and a service config with an open `env` map.
   - Two cells: `schema` (`response_format: json_schema`, strict) and `object` (`json_object`). The schema is
     in the prompt in both.
   - Metrics: `parse`; `valid` (primary; checked by this file's own JSON Schema validator); `fields` (the
     facts are right); `parse_lenient` (diagnostic).
   - A one-request probe per cell checks that the server accepts that response_format. A 4xx writes
     `"supported": false` and skips the cell.
   - Setting: prod sampling.
4. **agent_sessions.py.** 100 sessions in five families: orders/refund, currency chain, files (TODO hunt,
   then write a note), weather (warmest, gap in °F, then a 4th city), calendar (list the day, then book the
   earliest free half-hour).
   - All 10 OpenAI tools are offered in every session. The harness plays every tool from the session's own
     state.
   - Metrics:
     - `call_valid`: the tool exists and the arguments parse and match its schema.
     - `step_tools`: the required tool was called with the right key arguments.
     - `step_answer`: the `ANSWER:` line is right, or the world state is right (the file written, the event
       booked).
     - `session_success` (primary): all of the above held for the session.
   - Setting: prod sampling. reasoning_content is sent back as returned (`--strip-reasoning` drops it).
5. **mgsm.py.** MGSM (public `google-research/url-nlp`) in zh/es/fr/de/ja/ru with native instructions.
   - Scoring is gsm8k's, after NFKC and locale separators.
   - Metrics: `exact` (primary), `exact@<lang>`, and `script@zh|ja|ru`. The script metric is a heuristic for
     the reply's script, not a language classifier.
   - Setting: T 0. `--cells t0,prod` adds a prod cell; `--per-lang 250` runs the whole set.
6. **images.py.** First it probes with one PNG (a red square). A 4xx, or a reply that names no colour, writes
   `"served": false` with the reason and stops.
   - box-a's checkpoint has a `vision_config`, but whether the engine serves images is decided only by the
     probe.
   - When served, it runs four pixel-only families, drawn by a stdlib PNG encoder: count, shape, colour,
     quadrant.
   - It also runs no-image `control` rows, which should land near chance (1/6, 1/3, 1/5, 1/4).
7. **under_load.py.** The gsm8k 500-subset, in both cells, at 8 foreground streams on top of 16 background
   streams.
   - The background loader is built in, because `harness/synthmix.py` did not exist when this was written.
     `--bg-cmd '<cmd>'` runs any external loader instead.
   - Each background stream sends back to back at prod sampling, with this mix:
     - 60% chat: a cacheable per-stream system prompt plus 300–3,000 fresh tokens.
     - 25% multi-step arithmetic.
     - 15% a fresh 20k–60k-token document (a cold prefill).
   - Record (`load-bg.json`, also in each summary's `background`):
     - requests, errors, background tok/s
     - in-flight samples every 1 s, and `sglang:num_running_reqs` from `/metrics` every 5 s
     - `load_held`: true when ≥ 15 of 16 streams were in flight in ≥ 95% of the samples
   - The cells are `t0-load16` and `prod-load16` under suite `gsm8k`.
8. **compare.py.** Compares A (the reference) against B, metric by metric:
   - the difference with a Newcombe hybrid-score 95% CI
   - Fisher exact and two-proportion z p-values
   - when both jsonl files exist, the primary metric paired on shared item ids, with exact McNemar and a
     paired bootstrap CI
   - verdict text per metric and per suite: `better` / `WORSE` when the CI excludes 0; otherwise
     `non-inferior` when the CI's bad end stays within `--margin` (default 3 pp), `inconclusive` when it
     reaches past it (`--margin 0` gives "no detectable difference")
   - a warning line for any secondary metric that is worse, including an `answered` drop (lost requests)

   It is also a gate:
   - **Equal n.** Both arms must be the same size, or the run is refused with exit 3 and nothing is compared.
     In `counts` mode that means N1 == N2. In `files`/`dirs` it means each pair sent the same number of items
     (the n of `answered`, errors included). `--allow-unequal` waives the check and adds a note to the report.
     With equal items, a metric's own n can still differ by the items an arm lost to errors. That is noted, and
     `answered` scores the loss.
   - **WORSE.** A metric is WORSE when its Newcombe 95% CI excludes 0 in the bad direction, or when the
     one-sided Fisher exact p (B worse than A) is < 0.05. The one-sided test is the stricter of the two: 3/470
     vs 10/470 failures has a CI that still reaches past 0, but p = 0.045.
   - **What gates.** Each pair gates on its primary metric and on `answered`. `--gate-all` makes every metric
     gate. That is off by default, because the sub-metrics (`exact@zh`, `key@d0.05`, ...) are many and some
     will be false alarms. Without it, a worse secondary metric is still printed as a WARNING line.
   - **Exit codes.** 0 when nothing gates WORSE, 2 when anything does, 3 on an unequal-n refusal, 1 on bad
     input (usage errors included, so 2 always means WORSE).
   - The report text is unchanged, with a `**Gate:**` line under each verdict and a final `GATE: PASS|WORSE`
     line. `--json` adds `gate` at the top and per result, and `gate`, `gate_why` and `fisher_p_worse` per
     metric.

   Modes:
   - `compare.py dirs A B` matches every (suite, cell).
   - `compare.py files A.summary.json B.summary.json` compares one pair, e.g. idle vs load:
     `files OUT/gsm8k-t0.summary.json OUT/gsm8k-t0-load16.summary.json`.
   - `compare.py counts K1 N1 K2 N2 [--lower-better] [--name dsml] --labels final,mk` compares raw counts,
     e.g. the DSML gate's pass counts out of 470. Example: `counts 1 470 10 470 --lower-better` gives
     +1.91 pp [+0.54, +3.67], Fisher p 0.011, and exits 2.
9. **../dsml_compare.py.** This is the DSML gate's two-sample read. It compares two arms' `dsml/analyze.py
   grade` outputs: A is the reference (the final), B is the arm under test (the MK).
   - **Why it exists.** analyze.py's own bar (`TODO_BAR`) is a one-arm catastrophe gate against a fixed null.
     A 2% defect passes one mode of 470 calls 66% of the time. Final 1/470 vs MK 10/470 passes that bar, yet
     the two-sample read says WORSE. The user's bar: compare both arms at equal n with a two-sample test, at
     prod sampling (T 1.0, top_p 1.0), never against a fixed bar alone. Run this beside the grade, not
     instead of it.
   - **Input.** Each side is one of:
     - a phase's `selfparse.json`
     - a saved grade stdout (`analyze.py grade ... > <phase>.grade.txt`; it reads the `PASS <phase> <case>
       <mode> sends=... defective raw=K/N parsed=K/N differ=K/N` row lines)
     - a directory holding either. It reads every `selfparse.json` in the directory or one level down, else its
       `*.grade.txt`. An example is `raw/s4-mk-b1/dsml/`.
     A (case, mode) graded twice in one directory (two phases) is refused (exit 1); name the phase instead.
   - **What it compares.** The REPEATS rows (`T1_todo_nested`, `T1_todo_nested_strictfalse`, each in stream and
     nonstream mode), matched by (case, mode). Per row it compares three metrics, chosen with `--metrics`
     (default `raw,parsed,differ`):
     - `raw`: defective calls in the model's own DSML
     - `parsed`: defective returned arguments
     - `differ`: sends whose raw and parsed sides differ
     All three count defects, so lower is better. Each comparison goes through `compare.compare_counts`, the same
     code as above: B - A with a Newcombe 95% CI, a two-sided Fisher p, and a one-sided Fisher p that B is worse.
   - **Gate.** The WORSE rule is compare.py's. There is no multiplicity correction: 4 rows × 3 metrics, with raw
     and parsed nearly always the same count. A healthy pair can therefore come out WORSE by chance; rerun
     both arms to settle it.
   - **Equal n.** A row's two arms must have the same `sends`. A send can carry more than one call, so the calls
     n may still differ (S4's strictfalse stream: 7/472 calls from 470 sends). That is compared and noted.
     Unequal sends are refused unless `--allow-unequal`. A row present in only one arm is refused unless
     `--allow-missing`.
   - **Output.** It prints a table and `DSML-COMPARE PASS|WORSE|REFUSED`; `--json` and `--md` also write it.
   - **Exit codes.** 0 when nothing is WORSE, 2 when anything is, 3 on a refusal, 1 on bad input.
   - It runs on the Mac or the box (stdlib Python, nothing sent), from `harness/`. It imports
     `quality/compare.py`, so copy both.

## Rules every suite follows

- **A failed request is an error, never a wrong answer.** Transport errors, non-2xx statuses and unreadable
  200s are recorded as `ok: null` with the error kind. They count only against `answered`; every summary
  carries `answered`, `errors`, `error_kinds` and `first_error`.
- **Conditions are recorded.** Every summary records its condition: sampler, effort, max_tokens, n, seed and
  concurrency. `compare.py` notes any condition that differs between the two sides.
- **The key never leaves the header.** It is not written to any file or printed.

## Offline test (Mac or box; never touches the engine)

`python3 test_quality.py` runs 21 tests.

- **Unit tests:** the intervals and tests against published values (Newcombe 1998 example (a), Fisher
  "tea" and 1/9/11/3, Wilson 81/263, McNemar), plus answer extraction, needle placement and parsing, the
  schema validator, tool-call validation, the agent world, and image grading.
- **End to end:** every suite runs against `fake_server.py`, a local oracle that rebuilds each suite's items
  and answers them. The tests check that:
  - the oracle scores 100%
  - `--bad-rate 0.3` makes every scorer count failures, and `--error-rate` failures count as errors
  - `--no-json` / `--no-images` / a 300k context skip cleanly
  - under_load holds 16 streams
  - compare.py reads the results and pairs them by id

`python3 ../dsml_compare.py --self-test` runs 15 tests of the gate in both scripts, on synthetic counts:

- the one-sided Fisher values (the tea table, 17/70), exact at gsm8k-full n
- `compare.py counts 1 470 10 470 --lower-better` exits 2 with its report text intact; equal counts exit 0
- unequal n exits 3, or is compared under `--allow-unequal`; a usage error exits 1
- `dirs`: lost requests gate through `answered`, a short run is refused, a worse secondary gates only under
  `--gate-all`
- dsml_compare.py: equal arms pass; a 1/470 vs 10/470 row, or a differ-only regression, exits 2; unequal sends
  and a missing row exit 3; selfparse.json, its dirs and the grade stdout read the same counts
- S4's recorded MKL1 grade (`raw/s4-mk-b1/dsml/`) parses to 8/470, 6/470, 7/472 and 3/470 from either file

`QUALITY_E2E=0` runs only the unit tests. The suite passed on the Mac (Python 3.15) and on box-a (Python
3.12.3, with PIL and `tokenizers` present).
