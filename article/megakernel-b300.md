---
title: "DeepSeek-V4.1-Flash breaks 1,000 tokens per second for one user: how we roughly doubled decode in five days"
slug: megakernel-b300
draft: false
featured: true
date: "2026-10-31"
updated: "2026-11-05"
description: "DeepSeek-V4.1-Flash topped 1,000 tokens a second for one user on B300: the fastest bench request hit 1,064, and just over half of one quiet node's live replies beat 1,024. The bench averages 870."
abstract: >-
  For one user, DeepSeek-V4.1-Flash now breaks 1,000 tokens per second on B300. The fastest bench request hit 1,064.
  Over 14 engine starts, the bench averaged 870. That is about twice our first boot's 439, five days earlier, though the
  steps used different benches. Live, our router timed just over half of one quiet node's replies above 1,024, and the
  engine averaged 873 there. The biggest step was NVIDIA's megakernel for SGLang. Built for Grace Blackwell Ultra
  (GB300), it refused B300 at launch. We patched it and added guards to stop the engine if its chips lose sync. One-user
  decode ran 33 to 39% faster than our tuned stack without the megakernel. Tests detected no quality loss, and 20,818
  requests ran with no hang.
authors:
  - name: EZO Labs Research
section: inference
tags:
  - inference
  - megakernel
  - b300
  - gpu-kernels
  - sglang
  - speculative-decoding
  - model-serving
  - latency
report_number: EZO-R-2026-07
lang: en
code:
  url: "https://github.com/ezolabs/megakernel-b300"
  note: "B300 patch, the measured record behind this post and a setup kit"
og:
  kicker: "EZO LABS // RESEARCH"
  title:
    - "DeepSeek-V4.1-Flash breaks"
    - "1,000 tokens a second"
    - "for one user."
  foot:
    - "Fastest bench request 1,064; bench average 870"
    - "Megakernel: +33 to 39% over our tuned stack."
display:
  verdict: "Fast for one user."
  finding: "The fastest one-user bench request hit 1,064 tokens a second, above 1,000. Just over half of one quiet node's live replies beat 1,024. The bench averages 870, about double our first boot."
  numbers:
    - value: "1,064 tok/s"
      label: "fastest of 208 one-user bench requests, 5 above 1,000; bench average 870"
    - value: "≈2×"
      label: "one-user decode in five days: 439 at first boot, 870 averaged over 14 boots"
    - value: "0 of 20,818"
      label: "requests hung; the build we serve is bounded at 0.19 hangs per loaded hour, 95%"
  assumptions:
    - text: "Each step of the climb was timed on its own bench, so 439 to 870 is about 2×, not an exact ratio."
    - text: "Live, our router timed each reply, short bursts of accepted drafts included: just over half of one quiet node's replies beat 1,024. The engine's mean gap between tokens there gives 873, beside the bench's 870."
    - text: "The megakernel baseline is our own tuned production stack without it, not stock SGLang."
    - text: "The 33 to 39% is two megakernel boots against one baseline boot, and speed varies from boot to boot."
cta:
  lead: "Running your own GPU inference cluster? We can tune yours the same way."
  label: "talk to us"
  href: "mailto:contact@ezolabs.io?subject=optimize%20our%20inference%20cluster"
  track: research_megakernel_cta
bibtex_key: ezolabs2026megakernelb300
revisions:
  - date: "2026-10-31"
    note: "Published."
  - date: "2026-11-01"
    note: "Trimmed repeated numbers and caveats, and dropped the short-answer list: the abstract is the summary. No figure changed."
  - date: "2026-11-03"
    note: "Plainer wording: terms defined where they first appear, the climb's steps say who did what, and the abstract names the megakernel's baseline. No number or figure changed."
  - date: "2026-11-05"
    note: "Linked the repo with the B300 patch, the measured record and a setup kit."
  - date: "2026-11-05"
    note: "Corrected the accuracy-check margin and the hang bound to the author's corrected record."
license: site-terms
---

On our B300 nodes, we serve DeepSeek-V4.1-Flash, a 1M-token-context open-weight mixture-of-experts model. Tokens are the word pieces the model writes.

We run it on SGLang, the open-source serving engine, with NVIDIA's low-latency "megakernel" plugin. A megakernel fuses the many small programs the model runs for each token into a few long-running ones on the graphics processor (GPU).

![Column chart drawn to scale from zero of one user's decode speed in tokens per second on 4×B300, step by step from 26 to 31 October 2026, each step on its own bench with its own before and after: first boot 439; engine tuning 443 to 516; fast host cores 514 to 614–639; the megakernel with our B300 patch 638.4 to 849.0 and 885.8 on two boots; the megakernel build as served, boot averages 840.9 to 906.7 over 14 boots, average 870.2, about twice the first boot. A dashed line marks 1,000 tokens per second; above it sits the fastest of 208 bench requests, 1,063.8. Last, live traffic on node 2 over 34.5 minutes on 30 October: 873.4, from the engine's mean gap between tokens; busier node 1 averaged 702.2.](megakernel-b300/figures/speed-journey.svg)

**Figure 1.** One user's decode speed, step by step, from first boot to the build we serve. A boot is one start of the serving engine. Decode is the phase where the model writes its reply token by token. Each step has its own bench, so read its own before → after. Only the fastest bench requests pass 1,000. Live, the quieter node's average, 873, sits beside the bench's 870.

Source: EZO Labs Research, single-stream decode speed runs on two 4×B300 nodes, 26–31 October 2026, and the engine's live mean gap between tokens on one node over 34.5 minutes on 30 October 2026.

## How we got here

Figure 1 shows five steps, each timed on that day's bench:

- **26 October, first boot: 439.** On our ramp bench, an early copy of the appendix's decode test.
- **27 October, engine tuning: 443 → 516.** We checked drafted tokens together in blocks, combined GPU results in a single operation and sped up prompt processing. We measured these changes with an older bench tool.
- **28–29 October, fast host cores: 514 → 614 to 639.** We restricted the scheduler, which assigns serving work, to the host's fast cores. We measured this on the ramp bench (see the tip below).
- **28–30 October, the megakernel: 638.4 → 885.8 and 849.0.** We compared two boots with our B300 patch against our tuned stack without the megakernel. We used our megakernel speed gate, a set of performance checks.
- **30–31 October, the build we serve: 870.2.** We merged newer engine code and measured no change in speed.

The megakernel step was the largest. Its gain depends on how many reply streams are decoding at once on the node (Figure 2).

![Bar chart drawn to scale from zero of the gain in per-stream decode speed from NVIDIA's megakernel over our tuned stack without it, on one 4×B300 node, by the number of streams decoding at once, for two boots of the same build: plus 38.8% and plus 33.0% at 1 stream, plus 29.6% and plus 29.8% at 2, plus 23.7% and plus 23.9% at 4, plus 11.7% and plus 14.7% at 8, and plus 1.1% and plus 0.7% at 16. At one stream, 638.4 tokens per second without the megakernel, 885.8 and 849.0 with it.](megakernel-b300/figures/lead.svg)

**Figure 2.** How much faster each stream decodes with the megakernel, by the number of streams decoding at once, for two boots of the same build. The gain is largest for one user and gone at 16.

Source: EZO Labs Research, decode speed runs on one 4×B300 node, the megakernel build's two boots against our tuned stack without it, 28–30 October 2026.

Live on 30 October, our router timed every reply on both nodes for 34.5 minutes. On our second node, just over half of the replies came in above 1,024 tokens per second; on the first, fewer than half did. The router's rate counts short bursts of accepted drafts, so short replies read high. Two more caveats sit with that share:

- Live prompts suit speculative decoding better. It drafts several tokens ahead and keeps the ones the model agrees with: about 4.0 per step live, against 3.3 on the bench.
- The second node was the quieter one, with 0.25 requests running on average against 0.84 on the first.

## What NVIDIA shipped, and why B300 was refused

The plugin is [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega), marked experimental. NVIDIA's team released it on 18 October 2026 and described it in [a post with the SGLang team](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode). NVIDIA built it on four GB300 (Grace Blackwell Ultra) GPUs, where one user's reply streamed 1.38 times faster. When NVIDIA replayed recorded agent sessions there, one user's decode rose from 671.2 to 928.4 tokens per second at the 90th percentile. With 8K-token inputs, the gain was 12 to 38% at 1 to 8 users. That test simulated 3.51 accepted draft tokens per step.

Plain B300 is the same chip family, but it has 148 streaming multiprocessors (SMs) where GB300 has 152. An SM is one of the chip's compute blocks. B300 also sits in an ordinary x86 server. At launch, the plugin required all 152 SMs and refused to start on B300. Its page promises "no silent fallback", so refusing B300 was a design choice, not a bug. We patched the check and added a safety layer, and the patched plugin has served production traffic since 30 October.

On 30 October, upstream merged an official B300 route, [pull request #35](https://github.com/sgl-project/sgl-nvidia-mega/pull/35). Opting in with a setting named `sm148` admits a 148-SM B300, and the kernels are unchanged. Upstream measured B300 with only one request running at a time. The median rose from 633 tokens per second without the megakernel to 823 with it. That was a 30% gain.

**Table 1.** GB300 and plain B300, as the plugin sees them.

| | GB300 | Plain B300 |
| --- | --- | --- |
| Chip family | Blackwell Ultra | Blackwell Ultra |
| SMs per GPU | 152 | 148 |
| Host processor | NVIDIA Grace (Arm) | x86 server processors |
| Plugin at launch, 18 October | Supported; where it was built and measured | Refused at startup |
| Plugin since 30 October | Supported | Supported with the `sm148` opt-in; upstream speed measured at one request |

## What we changed

### A small patch

Our patch, on the plugin version SGLang pinned on 28 October 2026, changes two settings and none of NVIDIA's kernels:

1. It sets the number of work blocks kept running on the GPU to 148 instead of 152, to match B300's SMs.
2. It declares that the plugin works with SGLang's HiCache. HiCache moves older cached conversation data to the server's main memory so we can serve long contexts. This feature is called host cache offload.

The offload setting matters for response time and total output. In an earlier trial with offload disabled, one stream still gained 15% in decode speed. But the first token took 3 to 6 times longer, and total output at 4 to 16 streams fell 6 to 12.5%. Our patch predates upstream's route and is unofficial. Upstream's change still lacks the offload flag, so we stay on ours until the next plugin update.

### Before you try it: two traps and a tip

- **Trap 1, the interface version.** The plugin moved to version 2 of the interface through which SGLang calls its megakernel. SGLang code that uses version 1 still starts with the plugin installed, shows no error and serves at the old speed. Our gate now checks that all 27 megakernel decode graphs, recorded sequences of GPU operations, were captured for reuse.
- **Trap 2, the `sm148` setting.** Upstream's B300 route uses `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`, which replaces the plugin's 39 default kernel slots instead of adding to them. Set it to `sm148` alone, and B300 starts but runs stock code on every step. Keep the default slots in the list, then check that the speed moved.
- **A tip for x86 hosts.** On one of our hosts, only 16 of 128 cores reach the higher clock speed. Those cores boost to 4,600 MHz; the rest stay at 2,300. Pinning SGLang's scheduler to the fast cores cut a one-stream step from about 7.1 to 5.4 milliseconds. One stream then made 20 to 24% more tokens per second. Grace hosts have no such split, and we have not re-measured it with the megakernel on.

## The safety layer

A megakernel step spans all four GPUs and keeps them in sync itself. If one GPU diverges, for example by randomly selecting a different token during sampling, the step can wait forever. A community port to four B200 GPUs hung twice in 12.5 hours, about once every 6 hours, and NVIDIA's post gives no stability figures. So we added four guards before serving real traffic (Figure 3):

- **A GPU-agreement probe.** Every 100 decode steps, the four GPUs compare hashes, compact fingerprints of their state. A mismatch deliberately stops the engine. It costs under about 1% per step.
- **Matched random draws.** All four GPUs draw the same random numbers, so sampling cannot split them.
- **A hang watchdog.** If steps stall, it kills the engine, and a supervisor restarts and warms it up.
- **Router health checks.** A node that misses its two-second check is marked down. When we froze one GPU's process for 9.9 seconds, every router dropped the node within 5.6 seconds, and its three streams paused, then finished normally.

Before serving traffic, every engine change must also pass a test gate. We compare output byte for byte on 32 prompts using greedy decoding, which chooses the most likely token at each step. We also run NVIDIA's accuracy check and test tool-call formatting and math with 16 streams running at once. This gate has already rejected one upstream change that left speed alone but shifted the model's output.

![Diagram in two parts. Left, the plugin as shipped: a startup check needs 152 streaming multiprocessors (SMs) on every GPU; GB300, with its Grace host, has 152 and starts; plain B300, on an x86 host, has 148 and is refused, on purpose, since the plugin promises no silent fallback. Right, our changes: the patch lowers the check from 152 to 148 to match B300 and turns on host cache offload; the safety layer adds four guards: a GPU-agreement probe every 100 steps that stops the engine on a mismatch, matched random draws on all four GPUs, a hang watchdog that kills a stalled engine so it restarts, and router health checks that mark a stalled node down within 5.6 seconds. With both, plain B300 serves with the megakernel.](megakernel-b300/figures/what-changed.svg)

**Figure 3.** What stood between the plugin and B300, and what we added. The patch changes two settings, and the safety layer turns a possible hang into a fast, visible stop.

Source: EZO Labs Research, from the plugin's published requirements and our serving setup on two 4×B300 nodes, 28–31 October 2026.

## Speed

### Gain by number of streams

Table 2 gives the numbers behind Figure 2. The baseline is our tuned production stack without the megakernel, on the same node, not stock SGLang.

**Table 2.** Decode speed per stream, in tokens per second, on one 4×B300 node. The two megakernel columns are two boots of the same build.

| Streams at once | Without megakernel | With, first boot | Gain | With, second boot | Gain |
| --- | --- | --- | --- | --- | --- |
| 1 | 638.4 | 885.8 | +38.8% | 849.0 | +33.0% |
| 2 | 530.6 | 687.9 | +29.6% | 688.5 | +29.8% |
| 4 | 450.0 | 556.6 | +23.7% | 557.5 | +23.9% |
| 8 | 350.9 | 392.0 | +11.7% | 402.4 | +14.7% |
| 16 | 294.6 | 297.9 | +1.1% | 296.7 | +0.7% |

The megakernel cuts the time each step takes. With few streams, that step time accounts for most of the wait. At 16 streams, processing them together dominates the work, leaving little gain. The same new code with the megakernel off matched the old stack on all 19 speed-gate checks, so the gain is the megakernel's.

### One stream, across many boots

Speed varies from boot to boot, even on the same code. Across 14 boots on both nodes, one stream averaged 870.2 tokens per second. Boot averages ran from 840.9 to 906.7, about 8% apart. Our two code versions ran the same, at 875.6 and 863.0. So we judge a change by many boots, or against the normal spread between boots.

Figure 4 compares the megakernel with stock kernels, the GPU programs used with the megakernel switched off. The stock run used the same new code and the tuned stack's own arrangement of experts across GPUs. Both were measured on the same bench. The megakernel ran 38 to 44% faster, and even its slowest request beat the fastest stock one. That tops Table 2's 33 to 39% because this stock boot averaged 615.3 against the old stack's 638.4, a gap inside the normal spread between boots.

![Dot plot of single-stream decode speed in tokens per second on 4×B300 nodes, one row per boot, each request a small dot and each boot's average a large one, on one axis from 560 to 1,080 with a reference line at 1,000. On the speed gate's bench, on one node with 8 requests per boot: stock kernels with the megakernel off average 615.3, requests 567.0 to 657.3; megakernel boot 1 averages 885.8, plus 44.0%, and boot 2 849.0, plus 38.0%, their requests 783.5 to 957.9; together 867.4, plus 41.0%. On the decode test, a separate bench, 14 megakernel boots on two nodes average 870.2, with boot averages from 840.9 to 906.7; the 8 boots of the first code version average 875.6, the 6 of the second 863.0. Pooled over its 208 requests, the 10th percentile is 784.6, the median 865.2, the 90th percentile 954.0 and the fastest request 1,063.8.](megakernel-b300/figures/decode-distribution.svg)

**Figure 4.** Single-stream decode speed, megakernel against stock kernels. On the speed gate's bench, the boot with the megakernel off averaged 615.3 tokens per second and the two megakernel boots 867.4, 41.0% faster. On the decode test, 14 megakernel boots averaged between 840.9 and 906.7. Across their 208 requests, the median was 865.2, the 90th percentile 954.0 and the fastest 1,063.8. Five passed 1,000.

Source: EZO Labs Research, single-stream speed runs on two 4×B300 nodes: the megakernel speed gate on one node, one boot with the megakernel off and two with it, 29–30 October 2026; the decode test, two code versions of the megakernel build, 30–31 October 2026.

Live traffic agrees with the bench. Over the 34.5-minute window above, the engine's mean gap between tokens on our quieter second node was 1.145 milliseconds. That is 873 tokens per second live, beside the bench's 870. On the busier first node, with more streams at once, it works out to 702.

### First token and long prompts

On an idle node, a short prompt's first token arrived in a median 39 milliseconds, and in 26.8 on our faster host. On our faster host, processing a never-seen 60,000-token prompt took 0.869 seconds while four other streams kept decoding. A 979,971-token prompt, near the 1M-token context limit, took 15.4 seconds to process, and the model found the facts hidden in it. That speed comes from our own changes to prompt processing; the megakernel left it unchanged.

## Quality

We detected no quality loss against our tuned stack in 9 paired tests using the same questions and random seeds, from grade-school math to multi-step agent sessions. Against our reference scores for the same model weights on an 8×B200 node, every difference in Table 3 is within measurement noise.

**Table 3.** Accuracy of today's serving build on 4×B300 against our 8×B200 reference. Each side ran every question twice, and the two runs are pooled. The range is a 95% interval for the difference, unpaired; pooling two runs of the same questions makes it narrower than the data support.

| Test | Questions × runs | 4×B300, megakernel | 8×B200 reference | Difference, points (95% interval) |
| --- | --- | --- | --- | --- |
| GPQA Diamond, graduate-level science | 198 × 2 | 0.914 | 0.907 | +0.8 (−3.3 to +4.8) |
| GSM8K slice, grade-school math | 300 × 2 | 0.978 | 0.975 | +0.3 (−1.5 to +2.2) |
| MATH-500 slice, competition math | 100 × 2 | 0.900 | 0.890 | +1.0 (−5.1 to +7.1) |
| HumanEval+, code | 163 × 2 | 0.951 | 0.945 | +0.6 (−2.9 to +4.2) |

NVIDIA's own accuracy check scores the output again to measure how much the probabilities assigned to tokens changed. It passed on every boot we checked, and the worst reading was about a third of its limit. Most of our accuracy sets have 100 to 300 items. Sets that size would miss a drop smaller than about 7 to 15 points.

## Stability

Two one-hour soak tests, sustained runs to check stability, served 13,410 requests with no hang. With the probe checking every 10 steps, they found no mismatch in 852,300 probe checks per GPU. Production on 30 October added 7,408 requests with no hang, and one engine ran 13 hours without a restart. The soak tests and that build's first production hours together served 18,872 requests in 2.095 loaded hours, counting only time with a request running.

Zero hangs over those 18,872 requests caps the soak build's rate at 1.43 per loaded hour (95% confidence). Production on the build we serve has since added 15.43 loaded hours on both nodes with zero hangs, capping its rate at 0.19 per loaded hour (95% confidence). That is a bound, not proof of zero, and it cannot yet rule out the community port's one hang every 6 hours. Ruling that out takes about 18 hang-free loaded hours on one build. A stopped node rejoins the serving pool automatically in about 14 minutes.

## What it costs

The plugin needs every GPU to hold a slice of every expert, the small networks a mixture-of-experts model picks from for each token. Our tuned stack gave each GPU whole experts instead. The plugin's layout shrinks the cache that holds each conversation's memory by 25%. The remaining cache is still far larger than one 1M-token request needs. It also nudges output: fed the same text, the build picks the old stack's top token at 94.3% of short-prompt positions. An engine without the megakernel but with that layout changes the same token choices, and no accuracy test we ran detected an effect. With the node kept busy, the one-stream speedup cuts cost per output token by 25 to 28%.

## What we do not claim

- We do not claim to be faster than GB300. Our 90th percentile is 954.0. It sits near GB300's published 928.4 and 945.96, but the workloads, weights, draft acceptance and statistics all differ. Our fastest request, 1,064, does not compare either. What carries over is the relative gain, on hardware the code was not written for.
- Our setup is not supported. The unofficial patch needs re-checking at each plugin update; upstream's `sm148` opt-in is the supported route.
- These speeds do not represent every workload. They include speculative decoding and come from short synthetic prompts, 300-token replies and an idle node. Production mostly ran 1 to 4 requests at once. Brief peaks reached 13 to 14, some likely our own test traffic. Soaks sampled at temperature 1.0 only.

## Appendix: method

**Hardware.** Two rented nodes, each with four B300 SXM6 GPUs (148 SMs, compute capability 10.3, 275,040 MiB each, NVLink between every pair, 1,100 W, 2,032 MHz SM clock) on x86 hosts.

**Serving stack.** Our SGLang fork at tensor parallel size 4 and expert parallel size 1, FlashMLA attention, an fp8 (e4m3) KV cache, MXFP4 expert weights and 5-token draft blocks with block verification. Context 1,048,576 tokens, chunked prefill 8,192, up to 64 running requests, decode graphs for batches of 1 to 64. HiCache host offload at ratio 2 with selective write-through, which needs our patch's offload flag; TensorRT-LLM all-reduce fusion; the plugin enabled with `--enable-nvidia-megakernel`. The baseline is the same fork tuned without the megakernel, at expert parallel size 4. The TP-consistency probe runs every 100 decode steps in serving and every 10 in soaks, with assert on.

**Decode speed.** Four synthetic reasoning prompt templates with random numbers, tens of tokens long; thinking on, temperature 1.0, top-p 1.0, 300 completion tokens per request. At concurrency C, C streams start together; each cell has 8 repetitions, seeded per C. Per request, decode speed is completion tokens over the time from first to last streamed token, so time to first token is excluded. A repetition's value is the median over its C streams; a cell is the mean over repetitions. Every node was out of the router pool and idle before each run. Accepted draft length is about 3.3 per step here and about 4.0 in live traffic.

**Speed gates.** Each gate sets a band from the baseline's own boot-to-boot (A/A) spread: about ±4.6 to 8.3% for decode, 8 to 32% for time to first token and 6.7% for cold fill. A cell outside the band is a gain or a regression; a gate passes with no regression. One boot per arm understates boot variance, which is why the single-stream figures use 14 boots.

**Speed history.** Figure 1's steps come from four benches, as logged at the time: the ramp bench (first boot and fast host cores), an older command-line tool (27 October), the megakernel speed gate (28–30 October, and Figure 4's three gate boots) and the decode test above (the 14 boots). The ramp bench is an earlier copy of the program the decode test wraps, and we do not hold its first-boot settings, so steps on different benches do not compare as numbers.

**Live speed.** Over 34.5 minutes on both nodes on 30 October 2026, our router timed each reply's decode rate over its visible stream and counted the rates in histogram buckets. Two of its percentiles for the second node fall inside the 1,024 to 2,048 bucket, where it reads them by straight-line interpolation; together they put 51.5% of that node's replies above 1,024. The rate includes bursts of accepted drafts, so it is not a sustained per-stream speed. That speed is the engine's mean gap between tokens over the same window: 1.145 milliseconds on the second node, 873 tokens per second, and 1.424 on the first, 702. The engine's percentile gaps are not used, since they fall inside its first histogram bucket.

**First token and cold fill.** Time to first token is client-side, from send to the first streamed delta, on an idle, warm node. Cold fill starts four background decode streams, then sends one never-seen 60,000-token prompt, 8 repetitions; fill speed is its length over time to first token. The 1M-token probe is one 979,971-token prompt with hidden facts, checked for the answer and the time.

**Numerical checks.** Greedy: 32 prompts at temperature 0, 256 tokens, compared byte for byte on an idle node, since batch composition changes decoding. Teacher-forced: the reference's output is fed to the candidate over 32 prompts and top-1 agreement recorded; the long-prompt half sits at a 94 to 95.5% noise floor, so the short half carries the signal. NVIDIA's KL test: temperature 0, the k3 estimator on the sampled token, re-scored by a prefill pass, the mean of per-prompt means; short set 8 LongBench-v2 prompts × 256 tokens, limit 0.03; long set 4 windows of 18,000 to 27,000 tokens × 128 tokens, limit 0.06; 3 repeats per boot. Readings ran 0.0034 to 0.0076 short and 0.0087 to 0.0205 long, the 0.0205 run overlapping its warm-up.

**Accuracy.** Newcombe 95% intervals for differences in proportions, McNemar's test for paired items. A build is non-inferior when the interval's lower end is above −3 points. The run against the 8×B200 reference used concurrency 4, temperature 1.0 and two repetitions pooled, unpaired.

**Hang bound.** With zero events in T loaded hours, the one-sided 95% upper bound on a Poisson rate is 2.996 / T. Builds are never pooled; a build's soak and serving hours are. The soak build bounds at 1.43 on 2.095 loaded hours. The build that took over production on 30 October bounds at 0.194 on its 15.43 loaded hours, and the build serving since 31 October at 0.342 on 8.76 loaded hours.

**Data handling.** Every speed and quality prompt is synthetic or from a public benchmark. Live traffic was read only as aggregate metrics.

### Sources

- SGLang, NVIDIA and AMD teams, [SGLang low-latency mode: kernels and runtime](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode), 18 October 2026.
- The plugin's code and requirements: [sgl-project/sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega).
- Upstream's B300 opt-in and its one-request measurement: [sgl-nvidia-mega pull request #35](https://github.com/sgl-project/sgl-nvidia-mega/pull/35), merged 30 October 2026.
- SGLang's pin to the tested plugin version and its GB300 screen: [SGLang pull request #47145](https://github.com/sgl-project/sglang/pull/47145), merged 28 October 2026.
