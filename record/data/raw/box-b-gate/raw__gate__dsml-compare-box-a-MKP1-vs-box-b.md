## DSML two-sample gate: box-a-MKP1 (reference) vs box-b

- box-a-MKP1: qa/prod-merge-20261030/raw/gate/dsml/selfparse.json
- box-b: qa/box-b-bringup-20261030/raw/gate/dsml/selfparse.json
- K counts defects (lower is better); WORSE = Newcombe 95% CI excludes 0 upward, or one-sided Fisher p (B worse) < 0.05

| case | mode | metric | box-a-MKP1 | box-b | box-b - box-a-MKP1 (pp) [Newcombe 95%] | Fisher p | Fisher p (box-b worse) | grade box-a-MKP1/box-b | gate |
|---|---|---|---|---|---|---|---|---|---|
| T1_todo_nested | nonstream | raw | 4/470 (0.85%) | 5/470 (1.06%) | +0.21 [-1.24, +1.71] | 1 | 0.5 | PASS/PASS | pass |
| T1_todo_nested | nonstream | parsed | 4/470 (0.85%) | 5/470 (1.06%) | +0.21 [-1.24, +1.71] | 1 | 0.5 | PASS/PASS | pass |
| T1_todo_nested | nonstream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |
| T1_todo_nested | stream | raw | 4/470 (0.85%) | 3/470 (0.64%) | -0.21 [-1.59, +1.11] | 1 | 0.7743 | PASS/PASS | pass |
| T1_todo_nested | stream | parsed | 4/470 (0.85%) | 3/470 (0.64%) | -0.21 [-1.59, +1.11] | 1 | 0.7743 | PASS/PASS | pass |
| T1_todo_nested | stream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | raw | 3/470 (0.64%) | 2/470 (0.43%) | -0.21 [-1.47, +0.98] | 1 | 0.8132 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | parsed | 3/470 (0.64%) | 2/470 (0.43%) | -0.21 [-1.47, +0.98] | 1 | 0.8132 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | nonstream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | raw | 3/471 (0.64%) | 1/470 (0.21%) | -0.42 [-1.66, +0.64] | 0.6242 | 0.9376 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | parsed | 3/471 (0.64%) | 1/470 (0.21%) | -0.42 [-1.66, +0.64] | 0.6242 | 0.9376 | PASS/PASS | pass |
| T1_todo_nested_strictfalse | stream | differ | 0/470 (0.00%) | 0/470 (0.00%) | +0.00 [-0.81, +0.81] | 1 | 1 | PASS/PASS | pass |

- note: T1_todo_nested_strictfalse stream raw: same sends, n differs (box-a-MKP1 471, box-b 470: sends with more than one call)
- note: T1_todo_nested_strictfalse stream parsed: same sends, n differs (box-a-MKP1 471, box-b 470: sends with more than one call)
DSML-COMPARE PASS: 12 comparison(s) over 4 row(s), nothing WORSE
