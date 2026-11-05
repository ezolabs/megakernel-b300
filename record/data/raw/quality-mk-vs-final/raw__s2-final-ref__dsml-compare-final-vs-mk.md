## DSML two-sample gate: final (reference) vs mk

- final: raw/s2-final-ref/dsml
- mk: raw/s4-mk-b1/dsml
- K counts defects (lower is better); WORSE = Newcombe 95% CI excludes 0 upward, or one-sided Fisher p (B worse) < 0.05

| case | mode | metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | Fisher p (mk worse) | grade final/mk | gate |
|---|---|---|---|---|---|---|---|---|---|
| T1_todo_nested | nonstream | raw | 462/463 (99.78%) | 6/470 (1.28%) | -98.51 [-99.22, -96.72] | 3.671e-264 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested | nonstream | parsed | 469/470 (99.79%) | 6/470 (1.28%) | -98.51 [-99.22, -96.73] | 5.944e-266 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested | nonstream | differ | 20/470 (4.26%) | 0/470 (0.00%) | -4.26 [-6.48, -2.56] | 1.552e-06 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested | stream | raw | 459/460 (99.78%) | 8/470 (1.70%) | -98.08 [-98.94, -96.17] | 1.088e-259 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested | stream | parsed | 470/471 (99.79%) | 8/470 (1.70%) | -98.09 [-98.94, -96.19] | 6.121e-263 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested | stream | differ | 34/470 (7.23%) | 0/470 (0.00%) | -7.23 [-9.94, -5.07] | 6.269e-11 | 1 | EXPECTED-FAIL/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | raw | 1/470 (0.21%) | 3/470 (0.64%) | +0.43 [-0.64, +1.66] | 0.6242 | 0.3121 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | parsed | 1/470 (0.21%) | 3/470 (0.64%) | +0.43 [-0.64, +1.66] | 0.6242 | 0.3121 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | raw | 5/470 (1.06%) | 7/472 (1.48%) | +0.42 [-1.18, +2.08] | 0.773 | 0.3894 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | parsed | 5/470 (1.06%) | 7/472 (1.48%) | +0.42 [-1.18, +2.08] | 0.773 | 0.3894 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |

- note: T1_todo_nested nonstream raw: same sends, n differs (final 463, mk 470: sends with more than one call)
- note: T1_todo_nested stream raw: same sends, n differs (final 460, mk 470: sends with more than one call)
- note: T1_todo_nested stream parsed: same sends, n differs (final 471, mk 470: sends with more than one call)
- note: T1_todo_nested_strictfalse stream raw: same sends, n differs (final 470, mk 472: sends with more than one call)
- note: T1_todo_nested_strictfalse stream parsed: same sends, n differs (final 470, mk 472: sends with more than one call)
DSML-COMPARE PASS: 12 comparison(s) over 4 row(s), nothing WORSE
