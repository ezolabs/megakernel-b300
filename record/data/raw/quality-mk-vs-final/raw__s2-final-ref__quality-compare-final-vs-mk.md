## Quality comparison: final (reference) vs mk

Non-inferiority margin: 3.0 pp

### agent: final `prod` vs mk `prod` (primary `session_success`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -2.63 pp)
**Gate:** pass
- note: same 100 items, but metric n differs (final/mk: items lost to errors or not scored): call_valid 433/437
- paired on 100 shared items: both right 89, only final 1, only mk 6, neither 4; diff +5.00 pp [+0.00, +10.00] (paired bootstrap), McNemar exact p = 0.125

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 100/100 (100.00%) | 100/100 (100.00%) | +0.00 [-3.70, +3.70] | 1 | 1 | inconclusive |
| session_success* | 90/100 (90.00%) | 95/100 (95.00%) | +5.00 [-2.63, +12.96] | 0.2828 | 0.1795 | noninferior |
| call_valid | 433/433 (100.00%) | 437/437 (100.00%) | +0.00 [-0.87, +0.88] | 1 | 1 | noninferior |
| step_tools | 232/240 (96.67%) | 236/240 (98.33%) | +1.67 [-1.35, +4.93] | 0.3816 | 0.2422 | noninferior |
| step_answer | 236/240 (98.33%) | 239/240 (99.58%) | +1.25 [-0.91, +3.81] | 0.3724 | 0.1774 | noninferior |
| answer_line | 231/240 (96.25%) | 237/240 (98.75%) | +2.50 [-0.45, +5.83] | 0.1409 | 0.07941 | noninferior |
| all_calls_valid | 100/100 (100.00%) | 100/100 (100.00%) | +0.00 [-3.70, +3.70] | 1 | 1 | inconclusive |
| turns_4_to_8 | 88/100 (88.00%) | 92/100 (92.00%) | +4.00 [-4.60, +12.73] | 0.4804 | 0.3458 | inconclusive |
| session_success@orders | 11/20 (55.00%) | 16/20 (80.00%) | +25.00 [-3.89, +48.97] | 0.176 | 0.09143 | inconclusive |
| session_success@currency | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| session_success@files | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| session_success@weather | 19/20 (95.00%) | 19/20 (95.00%) | +0.00 [-19.06, +19.06] | 1 | 1 | inconclusive |
| session_success@calendar | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |

| mean | final | mk |
|---|---|---|
| turns | 5.51 [5.17, 5.84] | 5.55 [5.23, 5.88] |
| calls | 4.33 [3.88, 4.78] | 4.37 [3.94, 4.81] |
| step_answer_per_session | 0.98 [0.96, 1.00] | 1.00 [0.99, 1.00] |

### gsm8k: final `prod-load16` vs mk `prod-load16` (primary `exact`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -1.88 pp)
**Gate:** pass
- paired on 500 shared items: both right 481, only final 2, only mk 4, neither 13; diff +0.40 pp [-0.60, +1.40] (paired bootstrap), McNemar exact p = 0.6875

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 500/500 (100.00%) | 500/500 (100.00%) | +0.00 [-0.76, +0.76] | 1 | 1 | noninferior |
| exact* | 483/500 (96.60%) | 485/500 (97.00%) | +0.40 [-1.88, +2.70] | 0.8578 | 0.7193 | noninferior |
| format | 499/500 (99.80%) | 500/500 (100.00%) | +0.20 [-0.58, +1.12] | 1 | 0.3171 | noninferior |
| truncated | 1/500 (0.20%) | 0/500 (0.00%) | -0.20 [-1.12, +0.58] | 1 | 0.3171 | noninferior |

| mean | final | mk |
|---|---|---|
| completion_tokens | 412.14 [358.41, 475.43] | 408.38 [360.20, 462.52] |
| wall_s | 3.08 [2.66, 3.58] | 3.15 [2.76, 3.60] |

### gsm8k: final `prod` vs mk `prod` (primary `exact`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -1.87 pp)
**Gate:** pass
- paired on 1319 shared items: both right 1278, only final 10, only mk 2, neither 29; diff -0.61 pp [-1.14, -0.15] (paired bootstrap), McNemar exact p = 0.03857

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 1319/1319 (100.00%) | 1319/1319 (100.00%) | +0.00 [-0.29, +0.29] | 1 | 1 | noninferior |
| exact* | 1288/1319 (97.65%) | 1280/1319 (97.04%) | -0.61 [-1.87, +0.64] | 0.3966 | 0.3325 | noninferior |
| format | 1317/1319 (99.85%) | 1317/1319 (99.85%) | +0.00 [-0.41, +0.41] | 1 | 1 | noninferior |
| truncated | 1/1319 (0.08%) | 2/1319 (0.15%) | +0.08 [-0.29, +0.48] | 1 | 0.5635 | noninferior |

| mean | final | mk |
|---|---|---|
| completion_tokens | 409.63 [376.63, 444.46] | 416.90 [379.79, 456.47] |
| wall_s | 1.57 [1.44, 1.72] | 1.62 [1.47, 1.78] |

### gsm8k: final `t0-load16` vs mk `t0-load16` (primary `exact`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -2.23 pp)
**Gate:** pass
- paired on 500 shared items: both right 484, only final 1, only mk 1, neither 14; diff +0.00 pp [-0.60, +0.60] (paired bootstrap), McNemar exact p = 1

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 500/500 (100.00%) | 500/500 (100.00%) | +0.00 [-0.76, +0.76] | 1 | 1 | noninferior |
| exact* | 485/500 (97.00%) | 485/500 (97.00%) | +0.00 [-2.23, +2.23] | 1 | 1 | noninferior |
| format | 499/500 (99.80%) | 500/500 (100.00%) | +0.20 [-0.58, +1.12] | 1 | 0.3171 | noninferior |
| truncated | 1/500 (0.20%) | 0/500 (0.00%) | -0.20 [-1.12, +0.58] | 1 | 0.3171 | noninferior |

| mean | final | mk |
|---|---|---|
| completion_tokens | 393.97 [344.02, 452.51] | 379.54 [334.20, 429.97] |
| wall_s | 2.79 [2.42, 3.22] | 2.72 [2.39, 3.08] |

### gsm8k: final `t0` vs mk `t0` (primary `exact`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -1.08 pp)
**Gate:** pass
- paired on 1319 shared items: both right 1275, only final 4, only mk 7, neither 33; diff +0.23 pp [-0.23, +0.76] (paired bootstrap), McNemar exact p = 0.5488

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 1319/1319 (100.00%) | 1319/1319 (100.00%) | +0.00 [-0.29, +0.29] | 1 | 1 | noninferior |
| exact* | 1279/1319 (96.97%) | 1282/1319 (97.19%) | +0.23 [-1.08, +1.54] | 0.8172 | 0.7286 | noninferior |
| format | 1315/1319 (99.70%) | 1316/1319 (99.77%) | +0.08 [-0.40, +0.57] | 1 | 0.7051 | noninferior |
| truncated | 4/1319 (0.30%) | 3/1319 (0.23%) | -0.08 [-0.57, +0.40] | 1 | 0.7051 | noninferior |

| mean | final | mk |
|---|---|---|
| completion_tokens | 414.00 [373.44, 457.13] | 404.36 [365.94, 445.84] |
| wall_s | 1.60 [1.44, 1.76] | 1.58 [1.43, 1.73] |

### json: final `object` vs mk `object` (primary `valid`)

**Verdict:** mk non-inferior to final within 3.0 pp (CI bad end -2.79 pp)
**Gate:** pass
- paired on 100 shared items: both right 99, only final 0, only mk 1, neither 0; diff +1.00 pp [+0.00, +3.00] (paired bootstrap), McNemar exact p = 1

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 100/100 (100.00%) | 100/100 (100.00%) | +0.00 [-3.70, +3.70] | 1 | 1 | inconclusive |
| valid* | 99/100 (99.00%) | 100/100 (100.00%) | +1.00 [-2.79, +5.45] | 1 | 0.3161 | noninferior |
| parse | 100/100 (100.00%) | 100/100 (100.00%) | +0.00 [-3.70, +3.70] | 1 | 1 | inconclusive |
| fields | 99/100 (99.00%) | 100/100 (100.00%) | +1.00 [-2.79, +5.45] | 1 | 0.3161 | noninferior |
| parse_lenient | 100/100 (100.00%) | 100/100 (100.00%) | +0.00 [-3.70, +3.70] | 1 | 1 | inconclusive |
| valid@config | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@event | 19/20 (95.00%) | 20/20 (100.00%) | +5.00 [-11.63, +23.61] | 1 | 0.3112 | inconclusive |
| valid@invoice | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@person | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@sentiment | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |

| mean | final | mk |
|---|---|---|
| completion_tokens | 127.14 [116.67, 138.72] | 134.22 [123.66, 145.54] |

### json: final `schema` vs mk `schema` (primary `valid`)

**Verdict:** inconclusive: CI bad end -4.58 pp reaches past the 3.0 pp margin (more items needed to call it)
**Gate:** pass
- paired on 80 shared items: both right 80, only final 0, only mk 0, neither 0; diff +0.00 pp [+0.00, +0.00] (paired bootstrap), McNemar exact p = 1

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 80/100 (80.00%) | 80/100 (80.00%) | +0.00 [-11.10, +11.10] | 1 | 1 | inconclusive |
| valid* | 80/80 (100.00%) | 80/80 (100.00%) | +0.00 [-4.58, +4.58] | 1 | 1 | inconclusive |
| parse | 80/80 (100.00%) | 80/80 (100.00%) | +0.00 [-4.58, +4.58] | 1 | 1 | inconclusive |
| fields | 80/80 (100.00%) | 80/80 (100.00%) | +0.00 [-4.58, +4.58] | 1 | 1 | inconclusive |
| parse_lenient | 80/80 (100.00%) | 80/80 (100.00%) | +0.00 [-4.58, +4.58] | 1 | 1 | inconclusive |
| valid@event | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@invoice | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@person | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| valid@sentiment | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |

| mean | final | mk |
|---|---|---|
| completion_tokens | 104.53 [94.76, 114.92] | 103.81 [93.83, 114.33] |

### mgsm: final `t0` vs mk `t0` (primary `exact`)

**Verdict:** inconclusive: CI bad end -8.19 pp reaches past the 3.0 pp margin (more items needed to call it)
**Gate:** pass
- paired on 120 shared items: both right 112, only final 2, only mk 0, neither 6; diff -1.67 pp [-4.17, +0.00] (paired bootstrap), McNemar exact p = 0.5

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 120/120 (100.00%) | 120/120 (100.00%) | +0.00 [-3.10, +3.10] | 1 | 1 | inconclusive |
| exact* | 114/120 (95.00%) | 112/120 (93.33%) | -1.67 [-8.19, +4.70] | 0.7842 | 0.5818 | inconclusive |
| format | 120/120 (100.00%) | 117/120 (97.50%) | -2.50 [-7.09, +1.01] | 0.2469 | 0.08134 | inconclusive |
| exact@zh | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| script@zh | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| exact@es | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| exact@fr | 16/20 (80.00%) | 16/20 (80.00%) | +0.00 [-24.68, +24.68] | 1 | 1 | inconclusive |
| exact@de | 20/20 (100.00%) | 19/20 (95.00%) | -5.00 [-23.61, +11.63] | 1 | 0.3112 | inconclusive |
| exact@ja | 18/20 (90.00%) | 18/20 (90.00%) | +0.00 [-21.36, +21.36] | 1 | 1 | inconclusive |
| script@ja | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| exact@ru | 20/20 (100.00%) | 19/20 (95.00%) | -5.00 [-23.61, +11.63] | 1 | 0.3112 | inconclusive |
| script@ru | 20/20 (100.00%) | 19/20 (95.00%) | -5.00 [-23.61, +11.63] | 1 | 0.3112 | inconclusive |

| mean | final | mk |
|---|---|---|
| completion_tokens | 485.54 [377.52, 623.61] | 661.77 [442.86, 929.41] |

### needles: final `t0` vs mk `t0` (primary `all3`)

**Verdict:** inconclusive: CI bad end -6.75 pp reaches past the 3.0 pp margin (more items needed to call it)
**Gate:** pass
- note: conditions differ in: nonce
- paired on 80 shared items: both right 79, only final 1, only mk 0, neither 0; diff -1.25 pp [-3.75, +0.00] (paired bootstrap), McNemar exact p = 1

| metric | final | mk | mk - final (pp) [Newcombe 95%] | Fisher p | z p | verdict |
|---|---|---|---|---|---|---|
| answered | 80/80 (100.00%) | 80/80 (100.00%) | +0.00 [-4.58, +4.58] | 1 | 1 | inconclusive |
| all3* | 80/80 (100.00%) | 79/80 (98.75%) | -1.25 [-6.75, +3.45] | 1 | 0.3158 | inconclusive |
| key | 240/240 (100.00%) | 237/240 (98.75%) | -1.25 [-3.61, +0.53] | 0.2484 | 0.0823 | inconclusive |
| all3@100k | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| key@100k | 60/60 (100.00%) | 60/60 (100.00%) | +0.00 [-6.02, +6.02] | 1 | 1 | inconclusive |
| all3@250k | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| key@250k | 60/60 (100.00%) | 60/60 (100.00%) | +0.00 [-6.02, +6.02] | 1 | 1 | inconclusive |
| all3@500k | 20/20 (100.00%) | 20/20 (100.00%) | +0.00 [-16.11, +16.11] | 1 | 1 | inconclusive |
| key@500k | 60/60 (100.00%) | 60/60 (100.00%) | +0.00 [-6.02, +6.02] | 1 | 1 | inconclusive |
| all3@900k | 20/20 (100.00%) | 19/20 (95.00%) | -5.00 [-23.61, +11.63] | 1 | 0.3112 | inconclusive |
| key@900k | 60/60 (100.00%) | 57/60 (95.00%) | -5.00 [-13.70, +1.86] | 0.2437 | 0.07941 | inconclusive |
| key@d0.05 | 48/48 (100.00%) | 48/48 (100.00%) | +0.00 [-7.41, +7.41] | 1 | 1 | inconclusive |
| key@d0.25 | 48/48 (100.00%) | 48/48 (100.00%) | +0.00 [-7.41, +7.41] | 1 | 1 | inconclusive |
| key@d0.50 | 48/48 (100.00%) | 45/48 (93.75%) | -6.25 [-16.84, +2.22] | 0.2421 | 0.07845 | inconclusive |
| key@d0.75 | 48/48 (100.00%) | 48/48 (100.00%) | +0.00 [-7.41, +7.41] | 1 | 1 | inconclusive |
| key@d0.95 | 48/48 (100.00%) | 48/48 (100.00%) | +0.00 [-7.41, +7.41] | 1 | 1 | inconclusive |

| mean | final | mk |
|---|---|---|
| key_fraction_per_item | 1.00 [1.00, 1.00] | 0.99 [0.96, 1.00] |
| prompt_tokens | 437527.80 [372527.61, 504402.36] | 437528.00 [372527.95, 504402.47] |
| completion_tokens | 425.21 [385.85, 465.96] | 472.36 [397.84, 587.41] |
| wall_s@100k | 4.21 [3.84, 4.54] | 4.25 [3.85, 4.65] |
| wall_s@250k | 10.34 [9.33, 11.36] | 10.10 [9.00, 11.24] |
| wall_s@500k | 21.39 [19.67, 23.06] | 21.73 [19.84, 23.60] |
| wall_s@900k | 44.63 [41.14, 47.83] | 45.20 [40.83, 49.45] |

| suite | verdict |
|---|---|
| agent `prod` | mk non-inferior to final within 3.0 pp (CI bad end -2.63 pp) |
| gsm8k `prod-load16` | mk non-inferior to final within 3.0 pp (CI bad end -1.88 pp) |
| gsm8k `prod` | mk non-inferior to final within 3.0 pp (CI bad end -1.87 pp) |
| gsm8k `t0-load16` | mk non-inferior to final within 3.0 pp (CI bad end -2.23 pp) |
| gsm8k `t0` | mk non-inferior to final within 3.0 pp (CI bad end -1.08 pp) |
| json `object` | mk non-inferior to final within 3.0 pp (CI bad end -2.79 pp) |
| json `schema` | inconclusive: CI bad end -4.58 pp reaches past the 3.0 pp margin (more items needed to call it) |
| mgsm `t0` | inconclusive: CI bad end -8.19 pp reaches past the 3.0 pp margin (more items needed to call it) |
| needles `t0` | inconclusive: CI bad end -6.75 pp reaches past the 3.0 pp margin (more items needed to call it) |

GATE: PASS (9 compared; nothing WORSE by Newcombe 95% CI or one-sided Fisher p < 0.05)
