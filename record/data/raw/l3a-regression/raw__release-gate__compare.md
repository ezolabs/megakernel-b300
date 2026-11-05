## A. MultiChallenge @high, 266 items -- today L3a (box-b) vs then L3a (gpu1018b), UNPAIRED

graded 266/266 · truncated_thinking 0 · judge unparsed 0 / unscored 0 / errors 0 · judge a model judge (served a model judge), cap 16384 · complete True

| read | today L3a | then L3a | today - then, 95 % | read |
|---|---|---|---|---|
| **macro** (the row's score; bootstrap CIs, MOVER) | 67.7 [61.7, 73.6] | 66.2 [60.2, 72.1] | +1.48 pp [-6.95, +9.88] | SAME |
| micro (pooled 266; Newcombe) | 185/266 = 69.5 [63.8, 74.8] | 183/266 = 68.8 | +0.8 pp [-7.1, +8.6] | SAME |
| axis INFERENCE_MEMORY | 74/109 = 67.9 [58.6, 75.9] | 75/109 = 68.8 | -0.9 pp [-13.1, +11.3] | SAME |
| axis INSTRUCTION_RETENTION | 60/69 = 87.0 [77.0, 93.0] | 59/69 = 85.5 | +1.4 pp [-10.4, +13.3] | SAME |
| axis SELF_COHERENCE | 28/48 = 58.3 [44.3, 71.2] | 29/48 = 60.4 | -2.1 pp [-20.9, +17.0] | SAME |
| axis RELIABLE_VERSION_EDITING | 23/40 = 57.5 [42.2, 71.5] | 20/40 = 50.0 | +7.5 pp [-13.8, +27.9] | SAME |

## B. The release-gate bar -- today L3a vs then's same-session STOCK (UNPAIRED)

| read | today L3a | then STOCK | delta, 95 % | AZ §1.68.2 at 6.7669 pp (CI lower > -margin) |
|---|---|---|---|---|
| macro (MOVER) | 67.7 | 64.7 [58.6, 70.7] | +2.96 pp [-5.53, +11.47] | **pass** (then, paired: +1.47 [-3.88, +6.89] -> pass) |
| micro (Newcombe; SAME/BETTER/WORSE vs STOCK) | 185/266 = 69.5 [63.8, 74.8] | 182/266 = 68.4 | +1.1 pp [-6.7, +8.9] | SAME |

## C. Arena-Hard v2, 250 every-third slice -- generation only (no judge pass: no STOCK answers survive)

| effort | rows | answered | excluded (thinking at cap) | errors | output tokens mean / median | answer chars mean | ids sha256 | registered |
|---|---|---|---|---|---|---|---|---|
| @high | 250/250 | 250 | 0 | 0 | 1054 / 599 | 3586 | b31a49a17927 | False |
| @max | 250/250 | 250 | 0 | 0 | 1353 / 753 | 4720 | b31a49a17927 | False |

Then (gate-l3a/RESULTS.md:9): L3a vs same-session STOCK, 250/250 judged (0 excluded), 632 battles, style-controlled 0.4524 [0.4164, 0.4843], raw 0.4551 [0.3930, 0.5184]; bar SC CI lower >= 0.40 -> PASS. Today: not judged -- a head-to-head needs STOCK's answers on the slice, and none survive (NOTES.md §4e).
