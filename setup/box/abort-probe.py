#!/usr/bin/env python3
"""Abort probe: proves that a client disconnect actually frees the box, for both a
still-queued/prefilling request and a mid-stream (already-decoding) one (box-
parametrized via boxenv.py, like tails-probe.py; stdlib only).

WHY. A client that gives up (a closed browser tab, a timed-out proxy hop, a retried
request) must not leave its slot occupied on the engine -- a fork that keeps prefilling
or decoding a request nobody is listening to any more is exactly the kind of silent
box-occupancy bug this tool set exists to catch (see soak.py's `abort_intentional` /
`abort_completed_anyway` counters, which classify this from the CLIENT's own timeout
but never check that the SERVER actually let the slot go). This probe checks the
server side directly: fire a large cold request, sever the TCP connection ourselves at
a chosen moment, then watch the engine to see whether it actually noticed and freed the
probe's own request inside an acceptable window.

CODE PATHS THIS PROVES. Two independent disconnect-handling paths in the (forked)
SGLang engine:
  (a) QUEUED / STILL PREFILLING -- tokenizer_manager.py's wait-for-response loop polls
      `await request.is_disconnected()` on its own ~4 s wait tick while a request has
      not yet produced its first token (queued behind others, or still being
      prefilled). Closing the socket during this window and never seeing the request
      freed inside a few ticks is exactly this path failing.
  (b) MID-STREAM -- once the first content token has been produced and is being
      streamed out, the ASGI response generator's next attempt to send further chunks
      to the now-closed socket fails; the engine's own `create_abort_task` path is
      what is supposed to notice that failure and abort the underlying generation
      immediately rather than silently running it to completion into the void.
A fork that broke either one shows up here as the probe's own request still held by
the engine past --accept-seconds.

MEASUREMENT -- EACH CASE IS JUDGED BY ITS OWN REQUEST. Every case's request carries a
unique `rid` (`abort-probe-<case>-<uuid hex>`, the body's `rid` field, which the engine
adopts as its request id and echoes as `x-request-id`). Whether the engine still holds
that request is read off tokenizer_manager's per-rid table, `rid_to_state`: an entry is
made when the request enters `generate_request` (after the chat conversion, before it is
tokenized and sent to the scheduler) and dropped only once the scheduler is done with it
-- its finished batch output, or the abort echo the scheduler sends back after freeing
the request's KV (waiting-queue pop / chunked-prefill teardown / running abort).

The fork exposes no READ-ONLY view of that table (or of any per-request state):
`GET /v1/loads` carries per-rank counts and token sums only (load_snapshot.py
LoadSnapshot), `/server_info`'s internal_states carries config, memory and topology,
`/metrics` carries box-wide gauges, and the serve log names a rid only on a chat
conversion or a short-prefill swap (the scheduler's abort lines are DEBUG). So the table
is read through a RID CHECK built so that it can never become a second request under a
live rid: a tiny non-streaming chat completion sent with the SAME rid and a `lora_path`
naming an adapter that is never loaded (`abort-probe-rid-check-never-loaded`).
tokenizer_manager.generate_request runs two refusals, in this order, both before
anything is tokenized for the scheduler or sent to it:
  1. `_init_req_state` refuses a rid still in `rid_to_state` -- HTTP 400 "Duplicate
     request ID detected: <rid>" (--held-pattern): state HELD;
  2. otherwise the check registers its own short-lived entry and
     `_validate_and_resolve_lora` refuses the adapter -- HTTP 400 "LoRA adapter '...'
     was requested, but LoRA is not enabled" (--free-pattern): state FREE. The failure
     path (`_release_req_states_on_failure`) deletes only the check's own undelivered
     entry; its `_release_lora_once` does nothing on an engine without LoRA.
LORA MUST BE OFF. On a LoRA-ENABLED engine the same check is refused later ("Got LoRA
adapter that has never been loaded"), and that failure path's `_release_lora_once`
(tokenizer_manager.py, `self.enable_lora and state.obj.lora_path`) schedules a
`lora_registry.release` for an adapter reference the check never acquired -- every
check would corrupt the engine's adapter refcounts. So the PREFLIGHT first reads
`/server_info` (the deprecated `/get_server_info` if that fails) and refuses, as a
PROBE FAULT and before any rid check is sent, unless the engine's resolved config
shows `enable_lora` false/null AND no `lora_paths` -- an unreadable answer, or one
without the `enable_lora` field, is refused the same way (LoRA-off unconfirmed). The
default --free-pattern matches only the not-enabled refusal, so a LoRA-enabled answer
can never count as FREE either.
A check is never dispatched, prefilled or decoded, so it can never stand beside the
probe's request in the scheduler, whatever phase that request is in. A 2xx answer
would mean the engine DISPATCHED a check -- state ADMITTED: the probe stops checking at
once and the case FAILs, and a PREFLIGHT check under a fresh never-used rid, sent once
(after the LoRA-off read) before any case runs, must come back FREE or nothing is
fired at all.

The verdict for a case is the time from our sever to the send of the first check that
found the rid free, against --accept-seconds (default 6 -- tokenizer_manager's own ~4 s
tick plus slack); the last check that still found it held is printed beside it, so the
release is bracketed to one --poll-interval (default 0.5 s).

Why not the box-wide `/metrics` sum: `sglang:num_running_reqs` + `sglang:num_queue_reqs`
back to its pre-request baseline is confounded by live traffic arriving or finishing
during the window -- the 2026-10-22 parity window failed the queued case twice (6.3 s,
9.37 s) while the serve log showed the probe's own queued request aborted and freed
~2-3 s after the sever; a live request finishing can equally drop it to baseline while
ours is still held. That sum, its time-to-baseline, the --abort-pattern log scan and any
serve-log line naming the rid are still printed, as INFO rows only -- never the verdict.

What a case must also prove before its release counts:
  - "midstream": the first content token arrived, a rid check sent after it and BEFORE
    the sever answered HELD, and the probe's own stream was still live around that
    check -- it keeps reading the stream (read-only: our own response) and waits up to
    --liveness-seconds for one more content chunk after the check's answer. A FREE
    answer while the stream was still producing tokens means the engine's rid table
    does not hold a decoding request on this topology (or the check reached another
    tokenizer worker under --tokenizer-worker-num > 1): no own-request signal, FAIL. An
    `x-request-id` echo naming a different rid also fails the case.
  - "queued": no check can be sent before the sever (the engine sends no headers before
    the first token, and the request registers its rid only after its ~300k-token chat
    conversion), so the case instead needs the FIRST post-sever check to answer held. A
    free first answer means the request was not yet registered or vanished inside one
    round trip -- the attempt is not judged and is retried with a fresh rid and prompt
    and a doubled --close-after, up to --attempts (default 3). The case therefore needs
    a prompt whose PREFILL OUTLASTS --close-after (doubled per retry): a prompt that
    prefills inside it (box-c at --prompt-tokens 20000) is already decoding at the sever,
    box-c frees a decoding request inside one check round trip, and every attempt is
    not judged -- the case FAILs as unproven, its ERROR line carrying that hint.
  - both: the request must not have ended on its own before the sever (a finish_reason,
    `[DONE]` or the server closing the stream). The request is sent with
    `ignore_eos: true` and `max_tokens: 4000` so this should not happen; if it does the
    attempt is not judged and is retried with a fresh rid and prompt, up to --attempts.
  - both: the sever must really have closed the probe's own connection. It is a
    shutdown(SHUT_RDWR) on the socket object the reader captured at connect time (NOT
    conn.sock: http.client sets that to None as soon as a `Connection: close` response
    head arrives -- uvicorn sends one -- while the response keeps the fd open), then the
    response, the socket and the connection are closed; the reader must stop within
    SEVER_JOIN_SECONDS (5 s) and the socket's fd must be closed. Otherwise the engine had
    no disconnect to notice, so the case FAILs as a PROBE FAULT ("no own-request
    signal, case unproven (not an engine verdict)"), never as the engine's.
Exit 0 only if the preflight is FREE and both cases pass. The final line,
`ABORT-PROBE PASS|FAIL: queued=..., midstream=...`, keeps its shape for the gate runners
that grade it.

GUARD -- this fires large cold prefills directly at the engine: one ~300k-token
(--prompt-tokens) cold prefill per case, plus up to --attempts minus 1 further cold
prefills per case when an attempt is not judged (each with a fresh prompt). Every
request is `ignore_eos` with `max_tokens: 4000`: on a fork that does not honor the
sever, that is up to 4000 tokens decoded into a closed socket per case. Every rid check
(the preflight, then one per --poll-interval, default 0.5 s, while a case is polled) is
one tiny chat-template render that the engine refuses before the scheduler -- no
prefill, no decode, no completion. The big prefills are the same shape of load
tails-probe.py and freeze-probe.py warn about. Do not point the production defaults at
box-c while it is serving live routed traffic; validate small (`--prompt-tokens 20000`)
first, or run inside a cutover's gateway-disable bracket (tools/cutover.sh), per this tool
set's own convention. A small run validates the MIDSTREAM case only: on box-c a 20000-
token prefill ends inside --close-after, so the queued case's attempts are all not
judged and it FAILs as unproven. Only a prompt whose prefill outlasts --close-after --
the default 300000 on box-c -- judges the queued case.

BASE URL / MODEL / API KEY / LOG CONVENTION (shared by every probe in this set, see
boxenv.py):
  - --base-url wins; else boxenv.resolve_engine_url() (the box file's ENGINE_PORT, or
    $ENGINE_URL).
  - --model wins; else boxenv.resolve_served_model().
  - --api-key wins; else --env-file (a first-choice override, read directly for
    VLLM_API_KEY=...); else boxenv.resolve_api_key() ($VLLM_API_KEY, else
    /etc/environment on the box, else the box file's ENV_FILE).
  - --log wins; else boxenv.serve_log_default() -- an INPUT file (the engine's own
    serve log, tailed after each case, info only), never written by this tool.
  - The resolved base URL is printed first, before anything else runs.

Usage:
  python3 abort-probe.py                                     # 300k-token cold prompt, both cases
  python3 abort-probe.py --prompt-tokens 20000                # small on-box validation: midstream only (see GUARD)
  python3 abort-probe.py --window 30 --accept-seconds 10      # looser windows for a slow/loaded box
  python3 abort-probe.py --self-test
  python3 abort-probe.py --help

--self-test: runs the exact same preflight and case logic against a small bundled
stdlib http.server mock instead of a real engine. The mock keeps the engine's per-rid
table and its two refusals (a held rid -> 400 "Duplicate request ID detected"; a free
rid on a LoRA-tagged check -> 400 "LoRA adapter ... not enabled"), serves its LoRA config
on /server_info, counts every rid check it received and every one it would have
dispatched, and can add or drop phantom "live" requests in its /metrics. Like
the engine behind uvicorn it sends the response head (with `Connection: close`) only at
the first token, and it notices a disconnect only through a REAL watch on its socket --
a readable socket whose peek returns EOF, checked every prefill tick and before every
output -- so only a client that really closed its connection is ever noticed.
Thirteen scenarios, each with an expected verdict per case:
  1. well-behaved (notices the severed socket within a short tick, frees the slot and
     the rid, logs an abort line) -> PASS both cases;
  2. stuck fork (catches the write failure but never frees the slot or the rid, logs
     nothing -- the bug class this tool exists to catch) -> FAIL both cases;
  3. well-behaved, but a live request ARRIVES with each probe request and stays, so the
     global running+queue sum never returns to baseline -> must still PASS (and the
     self-test checks the global info row would have said FAIL, so the scenario really
     does confound the old measure);
  4. stuck fork, but a live request FINISHES as each probe request arrives, so the
     global sum is back at baseline at once -> must still FAIL (and the global info row
     must have said PASS);
  5. well-behaved, but the request registers its rid only after the first --close-after
     -> the queued case's first attempt is unobserved and retried, and must PASS on a
     later attempt;
  6. well-behaved, but the rid table drops a request once it is decoding (what the
     2026-10-22 live run looked like: a mid-stream check came back free) -> queued
     PASS, midstream FAIL (no own-request signal), and no check ever dispatched;
  7. an engine that would DISPATCH a LoRA-tagged check (no LoRA refusal) -> the
     preflight is admitted, nothing is fired, both cases FAIL;
  8. well-behaved, but the first midstream request ends on its own right after its
     first token -> that attempt is not judged and the case must PASS on a later one;
  9. the pre-fix sever (the 2026-10-22 09:56Z live failure: through conn.sock, which
     http.client clears once a `Connection: close` response head has arrived, with the
     reader left draining the stream, so the connection never closed and the engine
     streamed on) -> queued still PASSes (severed before the head), midstream must FAIL
     as a PROBE FAULT, never as an engine verdict;
 10. a LoRA-enabled engine (`enable_lora: true` on /server_info) -> the preflight
     refuses as a PROBE FAULT before any rid check is sent, nothing is fired, both
     cases FAIL;
 11. LoRA adapters configured (`lora_paths` set, `enable_lora` null), served only on the
     deprecated /get_server_info -> the same refusal, read through the fallback route;
 12. no /server_info at all -> LoRA-off unconfirmed, the same refusal;
 13. well-behaved, but the prefill ends inside --close-after and a decoding request is
     freed inside one check round trip (the 2026-10-22 box-c run at --prompt-tokens
     20000) -> every queued attempt is not judged and the case FAILs with the
     prefill-outlasts-close-after hint in its ERROR line; midstream PASSes.
A self-test that could not also produce the FAIL verdict, or that a confounded global
sum could flip, would not prove this tool can tell a healthy fork from a broken one.
This proves the tool's own mechanics (rid on the request, sever-the-socket, the
LoRA-off read, preflight, rid checks, stream liveness, the sever's own effectiveness, retry, info rows, gate on
--accept-seconds) end to end; it is a plumbing check, not a substitute for the real run
against box-c, and takes about 50 s.
"""
from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import random
import re
import select
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import boxenv

CURL_UA = boxenv.CURL_UA

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu "
    "apple river stone cloud garden window silver copper marble velvet"
).split()

# Same stdlib-only, no-tokenizer estimate as tails-probe.py: a `word<9 digits>` token
# is ~3.6 tokens on DeepSeek-V4.1-Flash's tokenizer, not prose's ~1.3. Approximate on
# purpose -- nothing in the accept rule depends on this number being exact, it only
# makes --prompt-tokens mean roughly what it says.
TOKENS_PER_WORD = 3.6

DEFAULT_ABORT_PATTERN = r"(?i)abort"

# tokenizer_manager._init_req_state's refusal of a rid that is still in rid_to_state
# ("Duplicate request ID detected: <rid>"), returned as HTTP 400 by the chat route.
DEFAULT_HELD_PATTERN = r"(?i)duplicate request id"

# tokenizer_manager._validate_and_resolve_lora's refusal of the check's never-loaded
# adapter on an engine without LoRA -- "LoRA adapter '<name>' was requested, but LoRA is
# not enabled" -- the answer a check gets once the rid is free, before anything reaches
# the scheduler. Deliberately NOT the LoRA-enabled engine's "Got LoRA adapter that has
# never been loaded": that refusal's cleanup releases an adapter reference the check
# never acquired (see MEASUREMENT, "LORA MUST BE OFF"), so it must never read as FREE.
DEFAULT_FREE_PATTERN = r"(?i)lora adapter .*but lora is not enabled"

# The engine's config routes, in order: /server_info, then the deprecated alias.
SERVER_INFO_PATHS = ("/server_info", "/get_server_info")

# The adapter every rid check names. Never loaded anywhere, so the check is refused
# before it is tokenized for, or sent to, the scheduler.
CHECK_ADAPTER = "abort-probe-rid-check-never-loaded"

CASES = ("queued", "midstream")

# Tries of a case whose attempt is not judged (the queued case's first post-sever check
# already free; either case's request ending on its own before the sever). Each queued
# retry doubles --close-after.
DEFAULT_ATTEMPTS = 3

# Midstream: how long to wait, after the pre-sever check's answer, for one more content
# chunk on the probe's own stream (the proof the request was still decoding).
DEFAULT_LIVENESS_SECONDS = 2.0

# The streaming request's own generation length; `ignore_eos` keeps it from ending early.
PROBE_MAX_TOKENS = 4000

# --prompt-tokens' default: the size whose prefill outlasts --close-after on box-c, so the
# one that judges the queued case there (a small size validates midstream only).
DEFAULT_PROMPT_TOKENS = 300_000

# After the sever, how long the reader thread gets to stop before the sever is judged
# ineffective (a probe fault: the case FAILs as unproven, never as the engine's).
SEVER_JOIN_SECONDS = 5.0


# --------------------------------------------------------------------- resolving cfg


def resolve_key(api_key: str | None, env_file: str | None) -> str:
    """--api-key wins; else --env-file (a first-choice override, read directly for
    VLLM_API_KEY=...); else boxenv's own chain ($VLLM_API_KEY, /etc/environment, the
    box file's ENV_FILE)."""
    if api_key:
        return api_key
    if env_file:
        try:
            for line in open(env_file):
                if line.startswith("VLLM_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return boxenv.resolve_api_key(None)


def gen_words(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return [f"{rng.choice(WORDS)}{rng.randrange(10 ** 9)}" for _ in range(n)]


def build_cold_prompt(seed: int, target_tokens: int) -> str:
    """A unique, never-before-seen ~target_tokens-token prompt -- distinct random
    content per (seed, case) is the point: a re-run must never get a radix-cache hit
    off a previous run's prompt, or the "cold" premise (a real prefill, not a cached
    no-op) breaks."""
    n_words = max(8, int(target_tokens / TOKENS_PER_WORD))
    words = gen_words(seed, n_words)
    return (
        "[abort-probe cold prompt, seed "
        + str(seed)
        + "] "
        + " ".join(words)
        + "\n\nUsing only the context above, write a long, detailed essay."
    )


# ------------------------------------------------------------------------- /metrics


def http_get(base: str, headers: dict, path: str, timeout: float = 5.0) -> str:
    req = urllib.request.Request(base + path, headers={**headers, "User-Agent": CURL_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def metrics_running_queue(base: str, headers: dict, timeout: float = 5.0):
    """(running, queue) from `sglang:num_running_reqs` / `sglang:num_queue_reqs`, or
    (None, None) on any error -- the same tp_rank="0"-scoped line match ttft-probe.py
    and freeze-probe.py use for the same two metrics."""
    try:
        text = http_get(base, headers, "/metrics", timeout=timeout)
    except Exception:  # noqa: BLE001 - a metrics hiccup must not crash the probe
        return None, None
    running = queue = None
    for line in text.splitlines():
        if line.startswith("sglang:num_running_reqs{") and 'tp_rank="0"' in line:
            try:
                running = float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        elif line.startswith("sglang:num_queue_reqs{") and 'tp_rank="0"' in line:
            try:
                queue = float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return running, queue


def metrics_sum(running, queue):
    if running is None or queue is None:
        return None
    return running + queue


# ------------------------------------------------------------------------- log tail


def log_offset(path: str):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def log_scan_since(path: str, offset, pattern: "re.Pattern"):
    """Whether any line appended to `path` after byte `offset` matches `pattern`.
    Returns (matched, first_matching_line) -- `offset is None` (log unreadable when
    the case started) always reports no match rather than raising."""
    if offset is None:
        return False, None
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return False, None
    for line in data.decode("utf-8", "replace").splitlines():
        if pattern.search(line):
            return True, line.strip()[:200]
    return False, None


# ------------------------------------------------------------- the probe's own rid


def make_rid(case: str) -> str:
    """A rid no other request on the box can share. The scheduler aborts by PREFIX
    (`req.rid.startswith(abort.rid)`), so the uuid hex also keeps our rid from being a
    prefix of anyone else's, and anyone else's from being ours."""
    return f"abort-probe-{case}-{uuid.uuid4().hex}"


def rid_check(base, headers, model, rid, held_re, free_re, timeout):
    """One rid check: a tiny non-streaming chat completion sent with `rid` and the
    never-loaded CHECK_ADAPTER as its `lora_path`. The engine answers it with one of
    its two pre-scheduler refusals (see MEASUREMENT):
      held     -- HTTP 400 matching `held_re` (the rid is still in rid_to_state);
      free     -- HTTP 400 matching `free_re` (the rid is free; the adapter refused);
      admitted -- any 2xx: the engine DISPATCHED the check (it carries `max_tokens: 1`
                  so even that is one token) -- never expected; the caller stops;
      error    -- a transport error, another status, or a 400 matching neither, which
                  decides nothing.

    `sent` is the wall time the check left: both refusals run right after the tiny
    message's chat render, so a free answer bounds the release by the send."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "abort-probe rid check"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "reasoning_effort": "none",
            "stream": False,
            "rid": rid,
            "lora_path": CHECK_ADAPTER,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={**headers, "Content-Type": "application/json", "User-Agent": CURL_UA},
        method="POST",
    )
    sent = time.time()
    status = None
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            text = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            text = e.read().decode(errors="replace")
        except Exception:  # noqa: BLE001 - an unreadable error body still has a status
            text = ""
    except Exception as e:  # noqa: BLE001 - a transport hiccup is one undecided check
        done = time.time()
        return {"sent": sent, "done": done, "rtt": done - sent, "state": "error",
                "http": None, "detail": str(e)[:150]}
    done = time.time()
    if 200 <= status < 300:
        state = "admitted"
    elif status == 400 and held_re.search(text):
        state = "held"
    elif status == 400 and free_re.search(text):
        state = "free"
    else:
        state = "error"
    return {"sent": sent, "done": done, "rtt": done - sent, "state": state, "http": status,
            "detail": None if state in ("held", "free") else text.strip()[:150]}


def read_server_info(base, headers, timeout):
    """The engine's resolved config: GET /server_info, else the deprecated
    /get_server_info. Returns (path, info dict, None) or (None, None, why)."""
    whys = []
    for path in SERVER_INFO_PATHS:
        try:
            info = json.loads(http_get(base, headers, path, timeout=timeout))
        except urllib.error.HTTPError as e:
            whys.append(f"{path}: HTTP {e.code}")
            continue
        except Exception as e:  # noqa: BLE001 - transport or JSON trouble: try the next route
            whys.append(f"{path}: {str(e)[:100]}")
            continue
        if isinstance(info, dict):
            return path, info, None
        whys.append(f"{path}: not a JSON object")
    return None, None, "; ".join(whys)


def lora_config(info: dict) -> dict:
    """Every `enable_lora` / `lora_paths` value in a /server_info answer (the resolved
    top-level args, and any copy nested in internal_states), and whether any of them
    turns LoRA on: `enable_lora` truthy, or `lora_paths` non-empty -- the engine enables
    LoRA for configured paths even with `enable_lora` unset."""
    enable, paths = [], []

    def walk(node, depth):
        if depth > 6:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "enable_lora":
                    enable.append(v)
                elif k == "lora_paths":
                    paths.append(v)
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(info, 0)
    return {
        "found": bool(enable),
        "enabled": any(bool(v) for v in enable) or any(bool(v) for v in paths),
        "enable_lora": enable[0] if enable else None,
        "lora_paths": next((v for v in paths if v), paths[0] if paths else None),
    }


def lora_off_check(base, headers, timeout):
    """Reads the engine's config and confirms LoRA is off before any rid check is sent
    (see MEASUREMENT, "LORA MUST BE OFF"). Returns (ok, summary, reason): reason is a
    PROBE FAULT when LoRA is on or its state cannot be read."""
    path, info, why = read_server_info(base, headers, timeout)
    if info is None:
        return False, f"server_info unreadable ({why})", (
            f"PROBE FAULT: could not read the engine's config ({why}), so LoRA-off is "
            "unconfirmed -- a rid check on a LoRA-enabled engine releases an adapter "
            "reference it never acquired; refusing to send any rid check (not an engine "
            "verdict)"
        )
    lc = lora_config(info)
    summary = (f"{path} enable_lora={lc['enable_lora']!r} "
               f"lora_paths={json.dumps(lc['lora_paths'])[:120]}")
    if not lc["found"]:
        return False, summary, (
            f"PROBE FAULT: {path} carries no `enable_lora` field, so LoRA-off is "
            "unconfirmed -- refusing to send any rid check (not an engine verdict)"
        )
    if lc["enabled"]:
        return False, summary, (
            f"PROBE FAULT: LoRA is enabled on this engine ({summary}) -- a rid check's "
            "failure cleanup would release a LoRA adapter reference it never acquired "
            "(tokenizer_manager `_release_lora_once`), corrupting the adapter refcounts; "
            "refusing to send any rid check (not an engine verdict)"
        )
    return True, summary, None


def preflight(base, headers, model, held_re, free_re, timeout):
    """First the LoRA-off read (lora_off_check): refused, no rid check is sent at all.
    Then one rid check under a fresh rid nobody has used, before anything is fired: it
    must answer FREE -- proof that this engine refuses a check before the scheduler and
    that --free-pattern matches its refusal. Returns (ok, rid, check, reason, lora):
    rid and check are None when the LoRA read refused; lora is its summary line."""
    lora_ok, lora_summary, lora_reason = lora_off_check(base, headers, timeout)
    if not lora_ok:
        return False, None, None, lora_reason, lora_summary
    ok, rid, check, reason = _preflight_check(base, headers, model, held_re, free_re, timeout)
    return ok, rid, check, reason, lora_summary


def _preflight_check(base, headers, model, held_re, free_re, timeout):
    rid = f"abort-probe-preflight-{uuid.uuid4().hex}"
    check = rid_check(base, headers, model, rid, held_re, free_re, timeout)
    if check["state"] == "free":
        return True, rid, check, None
    if check["state"] == "admitted":
        reason = (
            f"the preflight rid check was ADMITTED (HTTP {check['http']}): this engine "
            f"dispatches a chat request naming the never-loaded adapter {CHECK_ADAPTER!r}, "
            "so a rid check could become a second request under a live rid -- refusing to "
            "fire anything"
        )
    elif check["state"] == "held":
        reason = (
            f"the preflight rid check under a fresh rid came back held (HTTP {check['http']}): "
            "--held-pattern matches an answer that is not the duplicate-rid refusal"
        )
    else:
        reason = (
            f"the preflight rid check came back {check['state']} (HTTP {check['http']}"
            + (f", {check['detail']!r}" if check.get("detail") else "")
            + "): not the expected pre-scheduler adapter refusal (--free-pattern) -- no "
            "dispatch-safe rid check on this engine"
        )
    return False, rid, check, reason


# --------------------------------------------------------------- fire + sever + read


def fire_and_sever(base, headers, model, case, rid, prompt_tokens, seed, close_after,
                   timeout, pre_sever=None, liveness_seconds=DEFAULT_LIVENESS_SECONDS,
                   legacy_sever=False):
    """Opens one streaming chat completion carrying `rid` and severs the TCP connection
    ourselves -- not a graceful client-side cancel -- at the moment `case` calls for:
      "queued":   close_after seconds after the request was sent, regardless of
                  whether anything has streamed back yet (the ~300k-token prefill
                  takes far longer than close_after, so this always lands while the
                  request is still queued/being prefilled).
      "midstream": once the first content-bearing delta is read (decoding has
                  started), `pre_sever()` (the held rid check) runs, then up to
                  `liveness_seconds` pass waiting for one more content chunk after its
                  answer, and the socket is severed.
    The reader keeps reading the stream until the sever, so it sees whether the request
    ended on its own first (a finish_reason, `[DONE]`, or the server closing the
    stream). Returns a dict: fire_wall / sever_wall (time.time() at send and at the
    actual shutdown), saw_first_token, content_chunks, ended_before_sever /
    finish_reason, http_status / error_body (a non-200 answer to the request itself),
    rid_echo (the response's x-request-id, None if no headers), sever_effective /
    sever_detail (whether the probe's own connection really closed: the reader stopped
    within SEVER_JOIN_SECONDS and the socket's fd is closed -- a False here is a PROBE
    fault, never the engine's), and for midstream pre_check (pre_sever()'s result),
    live_after_check and content_after_check_s.

    The sever is a shutdown(SHUT_RDWR) on the socket object the reader captured at
    connect time, then the response, the socket and the connection are closed.
    `legacy_sever` exists for the self-test only: it reproduces the broken pre-fix
    sever (through conn.sock, with the reader left draining the stream) so the
    sever_effective guard is proven to catch it.
    """
    parsed = urllib.parse.urlsplit(base)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    base_path = parsed.path.rstrip("/") if parsed.path else ""
    path = base_path + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": build_cold_prompt(seed, prompt_tokens)}],
            "max_tokens": PROBE_MAX_TOKENS,
            "ignore_eos": True,
            "temperature": 0.0,
            "reasoning_effort": "none",
            "stream": True,
            "stream_options": {"include_usage": True},
            "rid": rid,
        }
    ).encode()
    req_headers = {
        **headers,
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": CURL_UA,
    }

    first_token = threading.Event()
    severing = threading.Event()
    ended = threading.Event()
    reader_error: list[Exception | None] = [None]
    got_response: list[bool] = [False]
    http_status: list[int | None] = [None]
    error_body: list[str | None] = [None]
    rid_echo: list[str | None] = [None]
    finish_reason: list[str | None] = [None]
    stream_error: list[str | None] = [None]
    content_chunks: list[int] = [0]
    last_content_wall: list[float] = [0.0]
    sock_box: list = [None]
    resp_box: list = [None]

    def mark_ended(reason):
        if not severing.is_set() and not ended.is_set():
            finish_reason[0] = reason
            ended.set()

    def reader():
        resp = None
        try:
            # Connect explicitly and keep OUR OWN reference to the socket: once the
            # server answers `Connection: close` (uvicorn does, for our
            # `Connection: close` request), http.client's getresponse() calls
            # conn.close(), which sets conn.sock = None while the response's file
            # object keeps the fd open -- a sever through conn.sock is then a no-op
            # and the engine keeps streaming to a live client.
            conn.connect()
            sock_box[0] = conn.sock
            if severing.is_set():
                return
            conn.request("POST", path, body=body, headers=req_headers)
            resp = conn.getresponse()
            resp_box[0] = resp
            got_response[0] = True
            http_status[0] = resp.status
            rid_echo[0] = resp.getheader("x-request-id")
            if resp.status != 200:
                error_body[0] = resp.read(400).decode(errors="replace").strip()[:200]
                return
            while not (severing.is_set() and not legacy_sever):
                chunk = resp.readline()
                if not chunk:
                    mark_ended("stream closed by the server")
                    break
                buf = chunk.strip()
                if not buf.startswith(b"data:"):
                    continue
                d = buf[5:].strip()
                if d == b"[DONE]":
                    mark_ended(finish_reason[0] or "[DONE]")
                    break
                try:
                    o = json.loads(d)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(o, dict) and o.get("error"):
                    # The engine's in-stream error chunk: the request failed, it was
                    # not ended by us -- a hard error, never a retried attempt.
                    stream_error[0] = json.dumps(o["error"])[:200]
                    mark_ended("in-stream error")
                    break
                choices = o.get("choices") or []
                if not choices:
                    continue
                if (choices[0].get("delta") or {}).get("content"):
                    content_chunks[0] += 1
                    last_content_wall[0] = time.time()
                    first_token.set()
                fr = choices[0].get("finish_reason")
                if fr:
                    mark_ended(f"finish_reason={fr}")
        except Exception as e:  # noqa: BLE001 - severing the socket ourselves raises here too
            reader_error[0] = e
        finally:
            if resp is not None and not legacy_sever:
                try:
                    resp.close()  # drops the response's hold on the fd
                except Exception:  # noqa: BLE001
                    pass

    th = threading.Thread(target=reader, daemon=True)
    fire_wall = time.time()
    th.start()

    pre_check = None
    live_after_check = None
    content_after_check_s = None
    if case == "queued":
        time.sleep(close_after)
    else:
        # Wait for the first content token, but stop early if the reader already
        # ended without one (a refused request must not sit out the whole --timeout).
        deadline = fire_wall + timeout
        while not first_token.wait(timeout=0.1):
            if not th.is_alive() or time.time() >= deadline:
                break
        if first_token.is_set() and pre_sever is not None:
            pre_check = pre_sever()
            # Liveness: one more content chunk after the check's answer proves the
            # request was still decoding while the check ran.
            live_after_check = False
            live_deadline = time.time() + liveness_seconds
            while time.time() < live_deadline:
                if last_content_wall[0] > pre_check["done"]:
                    live_after_check = True
                    content_after_check_s = round(last_content_wall[0] - pre_check["done"], 3)
                    break
                if ended.is_set() or not th.is_alive():
                    break
                time.sleep(0.02)
    saw_first_token = first_token.is_set()

    chunks_at_sever = content_chunks[0]
    severing.set()
    sever_wall = time.time()
    # legacy_sever (self-test only) reproduces the pre-fix sever: through conn.sock,
    # which getresponse() already set to None on a `Connection: close` response.
    sock = getattr(conn, "sock", None) if legacy_sever else sock_box[0]
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)  # FIN now, whoever else holds the fd
        except OSError:
            pass
    if legacy_sever:
        conn.close()  # the pre-fix code's only close: a no-op once conn.sock is None
    th.join(timeout=SEVER_JOIN_SECONDS)
    reader_alive = th.is_alive()
    if not legacy_sever:
        for closer in (resp_box[0], sock_box[0], conn):
            if closer is None:
                continue
            try:
                closer.close()
            except Exception:  # noqa: BLE001
                pass
    own = sock_box[0]
    # The sever took iff our reader stopped and the socket's fd is really closed
    # (fileno() is -1 only after the last holder -- the response's file object
    # included -- let go). Never connected: nothing to sever.
    fd_closed = own is None or own.fileno() == -1
    sever_effective = (not reader_alive) and fd_closed

    return {
        "sever_effective": sever_effective,
        "sever_detail": None if sever_effective else (
            f"reader still receiving {SEVER_JOIN_SECONDS}s after the sever: {reader_alive}; "
            f"socket fd still open: {not fd_closed}; content chunks read in all: {content_chunks[0]}"
        ),
        "content_chunks_after_sever": content_chunks[0] - chunks_at_sever,
        "fire_wall": fire_wall,
        "sever_wall": sever_wall,
        "saw_first_token": saw_first_token,
        "content_chunks": content_chunks[0],
        "ended_before_sever": ended.is_set(),
        "finish_reason": finish_reason[0],
        "stream_error": stream_error[0],
        "got_response": got_response[0],
        "http_status": http_status[0],
        "error_body": error_body[0],
        "rid_echo": rid_echo[0],
        "reader_error": str(reader_error[0])[:150] if reader_error[0] else None,
        "pre_check": pre_check,
        "live_after_check": live_after_check,
        "content_after_check_s": content_after_check_s,
    }


# ---------------------------------------------------------------------- one full case


def _check_row(check: dict, t0: float) -> dict:
    row = {"t": round(check["sent"] - t0, 2), "state": check["state"], "http": check["http"],
           "rtt": round(check["rtt"], 3)}
    if check.get("detail"):
        row["detail"] = check["detail"]
    return row


def _blank_result(case: str, rid, close_after) -> dict:
    return {
        "case": case,
        "rid": rid,
        "close_after": close_after,
        "baseline_running": None,
        "baseline_queue": None,
        "baseline_sum": None,
        "saw_first_token": False,
        "content_chunks": 0,
        "ended_before_sever": False,
        "finish_reason": None,
        "stream_error": None,
        "sever_effective": None,
        "sever_detail": None,
        "content_chunks_after_sever": 0,
        "live_after_check": None,
        "content_after_check_s": None,
        "pre_sever_check": None,
        "checks": [],
        "held_after_sever": False,
        "last_held_s": None,
        "released_by_s": None,
        "unobserved": False,
        "admitted": False,
        "elapsed_to_baseline_s": None,
        "global_verdict_pass": None,
        "abort_seen": False,
        "abort_line": None,
        "rid_line": None,
        "samples": [],
        "error": None,
        "passed": False,
    }


def not_run_result(case: str, reason: str) -> dict:
    """A case that was never fired (the preflight refused): FAIL, with the reason."""
    r = _blank_result(case, None, None)
    r["error"] = f"not run: {reason}"
    r["attempt"] = 0
    r["attempts"] = 0
    r["prior_attempts"] = []
    return r


def run_case(base, headers, model, log_path, case, prompt_tokens, seed, close_after,
             window, poll_interval, accept_seconds, timeout, abort_re, held_re,
             free_re, check_timeout, liveness_seconds=DEFAULT_LIVENESS_SECONDS,
             legacy_sever=False):
    """One attempt at one case. The verdict is the probe's OWN request's release (see
    MEASUREMENT); the global running+queue baseline, its time-to-baseline, the abort-
    pattern log scan and a serve-log line naming the rid ride along as info only.
    `result["unobserved"]` marks an attempt that is not judged (the request ended on
    its own before the sever, or the queued case's first post-sever rid check was
    already free); run_case_attempts retries it."""
    rid = make_rid(case)
    rid_re = re.compile(re.escape(rid))
    running0, queue0 = metrics_running_queue(base, headers)
    baseline = metrics_sum(running0, queue0)
    offset = log_offset(log_path)

    def pre_sever():
        return rid_check(base, headers, model, rid, held_re, free_re, check_timeout)

    fire_info = fire_and_sever(
        base, headers, model, case, rid, prompt_tokens, seed, close_after, timeout,
        pre_sever=pre_sever if case == "midstream" else None,
        liveness_seconds=liveness_seconds, legacy_sever=legacy_sever,
    )

    result = _blank_result(case, rid, close_after)
    result.update(
        baseline_running=running0,
        baseline_queue=queue0,
        baseline_sum=baseline,
        **{k: v for k, v in fire_info.items() if k != "pre_check"},
    )

    if fire_info["http_status"] is not None and fire_info["http_status"] != 200:
        result["error"] = (
            f"the engine refused the probe's own request: HTTP {fire_info['http_status']} "
            f"{fire_info['error_body']!r} -- case unproven"
        )
        return result

    if fire_info["stream_error"] is not None:
        result["error"] = (
            f"the probe's own request failed in-stream before the sever: "
            f"{fire_info['stream_error']} -- case unproven"
        )
        return result

    if fire_info["ended_before_sever"]:
        result["unobserved"] = True
        result["error"] = (
            f"the probe's own request ended on its own before the sever "
            f"({fire_info['finish_reason']}, after {fire_info['content_chunks']} content "
            "chunk(s)) -- nothing left to abort, attempt not judged"
        )
        return result

    if not fire_info["sever_effective"]:
        # The probe's own connection never closed, so the engine had no disconnect to
        # notice: whatever the rid table says next is not about the abort path.
        result["error"] = (
            f"PROBE FAULT: the sever did not close the probe's own connection "
            f"({fire_info['sever_detail']}) -- no own-request signal, case unproven "
            "(not an engine verdict)"
        )
        return result

    if case == "midstream":
        if not fire_info["saw_first_token"]:
            result["error"] = "never observed a content token before --timeout -- case unproven"
            return result
        if fire_info["rid_echo"] is not None and fire_info["rid_echo"] != rid:
            result["error"] = (
                f"the engine answered under x-request-id={fire_info['rid_echo']!r}, not the "
                f"sent rid {rid!r} -- no own-request signal, case unproven"
            )
            return result
        check = fire_info["pre_check"]
        result["pre_sever_check"] = _check_row(check, fire_info["sever_wall"])
        if check["state"] == "admitted":
            result["admitted"] = True
            result["error"] = (
                f"UNSAFE: the pre-sever rid check was ADMITTED (HTTP {check['http']}) -- the "
                "engine dispatched a request under the probe's live rid; no further checks sent"
            )
            return result
        if check["state"] != "held":
            if check["state"] == "free" and fire_info["live_after_check"]:
                why = (
                    f"while the probe's own stream was still producing tokens (one more "
                    f"{fire_info['content_after_check_s']}s after the check's answer): the "
                    "engine's rid table does not hold a decoding request here (or >1 "
                    "tokenizer worker)"
                )
            elif check["state"] == "free":
                why = (
                    f"and the probe's own stream showed no further token within "
                    f"{liveness_seconds}s of it, so its liveness at the check is unproven"
                )
            else:
                why = "(an undecided check)"
            result["error"] = (
                f"the rid check sent after the first token and before the sever came back "
                f"{check['state']} (HTTP {check['http']}"
                + (f", {check['detail']}" if check.get("detail") else "")
                + f"), not held, {why} -- no own-request signal, case unproven"
            )
            return result

    t0 = fire_info["sever_wall"]
    released_by = None
    last_held = None
    held_after_sever = False
    elapsed_to_baseline = None
    while True:
        elapsed = time.time() - t0
        running, queue = metrics_running_queue(base, headers)
        cur = metrics_sum(running, queue)
        result["samples"].append({"t": round(elapsed, 2), "running": running, "queue": queue})
        if elapsed_to_baseline is None and cur is not None and baseline is not None and cur <= baseline:
            elapsed_to_baseline = elapsed
        if not result["abort_seen"]:
            seen, line = log_scan_since(log_path, offset, abort_re)
            if seen:
                result["abort_seen"], result["abort_line"] = True, line
        if result["rid_line"] is None:
            seen, line = log_scan_since(log_path, offset, rid_re)
            if seen:
                result["rid_line"] = line
        if released_by is None:
            check = rid_check(base, headers, model, rid, held_re, free_re, check_timeout)
            row = _check_row(check, t0)
            result["checks"].append(row)
            if check["state"] == "admitted":
                # Never expected (the preflight was free): stop sending checks at once.
                result["admitted"] = True
                break
            if check["state"] == "held":
                held_after_sever = True
                last_held = row["t"]
            elif check["state"] == "free":
                released_by = check["sent"] - t0
                if case == "queued" and not held_after_sever:
                    # Never seen held after the sever: registration not yet reached
                    # (the request's own chat conversion still running) or a release
                    # inside one round trip -- this attempt proves nothing.
                    result["unobserved"] = True
                    break
        elapsed = time.time() - t0
        # Released: keep sampling the info row up to --accept-seconds so it can say
        # what the global measure would have ruled; never past --window.
        if released_by is not None and (elapsed_to_baseline is not None or elapsed >= accept_seconds):
            break
        if elapsed >= window:
            break
        time.sleep(poll_interval)

    result["held_after_sever"] = held_after_sever
    result["last_held_s"] = last_held
    result["released_by_s"] = round(released_by, 2) if released_by is not None else None
    result["elapsed_to_baseline_s"] = round(elapsed_to_baseline, 2) if elapsed_to_baseline is not None else None
    result["global_verdict_pass"] = elapsed_to_baseline is not None and elapsed_to_baseline <= accept_seconds

    if result["admitted"]:
        last = result["checks"][-1]
        result["error"] = (
            f"UNSAFE: a rid check at t={last['t']}s was ADMITTED (HTTP {last['http']}) -- the "
            "engine dispatched a request under the probe's rid; no further checks sent"
        )
        return result
    if result["unobserved"]:
        result["error"] = (
            "the first post-sever rid check already found the rid free: the request was "
            "never observed held (not yet registered, or released inside one round trip) "
            "-- attempt not judged"
        )
        return result
    if released_by is None:
        states = [c["state"] for c in result["checks"]]
        if held_after_sever:
            result["error"] = (
                f"the probe's own request was still held at the window's end "
                f"({states.count('held')} held checks, the last at {last_held}s)"
            )
        else:
            last = result["checks"][-1] if result["checks"] else None
            result["error"] = (
                f"no rid check resolved inside --window ({len(states)} checks, last "
                f"{last['state'] if last else 'none'}"
                + (f": HTTP {last['http']} {last.get('detail')!r}" if last else "")
                + ") -- case unproven"
            )
        return result
    result["passed"] = released_by <= accept_seconds
    return result


def run_case_attempts(base, headers, model, log_path, case, prompt_tokens, seed, close_after,
                      window, poll_interval, accept_seconds, timeout, abort_re, held_re,
                      free_re, check_timeout, attempts, liveness_seconds=DEFAULT_LIVENESS_SECONDS,
                      legacy_sever=False):
    """run_case, retrying an attempt that is not judged (fresh rid, fresh prompt seed;
    the queued case also doubles close_after) up to `attempts` times."""
    prior = []
    ca = close_after
    r = None
    n = max(1, attempts)
    for i in range(n):
        r = run_case(
            base, headers, model, log_path, case, prompt_tokens, seed + 100 * i, ca,
            window, poll_interval, accept_seconds, timeout, abort_re, held_re, free_re,
            check_timeout, liveness_seconds, legacy_sever,
        )
        r["attempt"] = i + 1
        r["attempts"] = n
        r["prior_attempts"] = list(prior)
        if not r["unobserved"]:
            return r
        prior.append({"rid": r["rid"], "close_after": ca, "why": r["error"],
                      "saw_first_token": r["saw_first_token"], "checks": r["checks"][:3]})
        if case == "queued":
            ca *= 2
    r["error"] = (
        f"every one of {r['attempts']} attempt(s) was not judged (last: {r['error']}) -- "
        "case unproven"
    )
    if case == "queued":
        r["error"] += " -- " + queued_unjudged_hint(prompt_tokens, prior)
    return r


QUEUED_HINT_MARK = "the queued case needs a prompt whose prefill outlasts --close-after"


def queued_unjudged_hint(prompt_tokens: int, attempts: list[dict]) -> str:
    """The ERROR-line hint for a queued case whose every attempt was not judged: the
    usual cause is a prompt that prefills inside --close-after (see GUARD)."""
    n = len(attempts)
    decoding = sum(1 for a in attempts if a.get("saw_first_token"))
    last_ca = attempts[-1]["close_after"] if attempts else None
    if decoding:
        why = (
            f"its first token arrived before the sever on {decoding} of {n} attempt(s), so "
            f"--prompt-tokens {prompt_tokens} prefills inside --close-after (up to "
            f"{last_ca}s) and validates the MIDSTREAM case only"
        )
    else:
        why = (
            f"it was never seen held after the sever on any of {n} attempt(s) (close_after "
            f"up to {last_ca}s); a small --prompt-tokens (such as 20000) validates the "
            "MIDSTREAM case only"
        )
    return (
        f"hint: {QUEUED_HINT_MARK} (the default --prompt-tokens {DEFAULT_PROMPT_TOKENS} on "
        f"box-c); {why} -- judge the queued case at --prompt-tokens {DEFAULT_PROMPT_TOKENS}, "
        "inside a gateway-disable bracket (see GUARD)"
    )


def print_preflight(pf: dict) -> None:
    check = pf["check"]
    lora_refused = check is None
    # A refusal's reason already names what was read.
    print(f"preflight lora: REFUSED: {pf['reason']}" if lora_refused
          else f"preflight lora: {pf['lora']} -> LoRA off", flush=True)
    if lora_refused:
        print("preflight rid check: not sent (the LoRA-off read refused)", flush=True)
        return
    print(
        f"preflight rid check: rid={pf['rid']} {check['state']} (HTTP {check['http']}, rtt "
        f"{round(check['rtt'], 3)}s)" + ("" if pf["ok"] else f" -- REFUSED: {pf['reason']}"),
        flush=True,
    )


def print_case_result(r: dict, accept_seconds: float) -> None:
    verdict = "PASS" if r["passed"] else "FAIL"
    print(f"-- case {r['case']} --", flush=True)
    if r.get("attempt", 1) == 0:
        print(f"  ERROR: {r['error']}", flush=True)
        print(f"  {r['case'].upper()}-CASE {verdict} (not run)", flush=True)
        return
    print(
        f"  rid={r['rid']} attempt={r.get('attempt', 1)}/{r.get('attempts', 1)} "
        f"close_after={r['close_after']}s content_chunks_before_sever={r.get('content_chunks')}"
        + (" (first token arrived before the sever: ran as a mid-stream disconnect)"
           if r["case"] == "queued" and r.get("saw_first_token") else ""),
        flush=True,
    )
    for p in r.get("prior_attempts") or []:
        print(f"  attempt not judged: rid={p['rid']} close_after={p['close_after']}s "
              f"why={p['why']!r} first checks={json.dumps(p['checks'])}", flush=True)
    if r.get("pre_sever_check"):
        p = r["pre_sever_check"]
        print(
            f"  pre-sever rid check: {p['state']} (HTTP {p['http']}, rtt {p['rtt']}s); "
            f"own stream live after it: {r.get('live_after_check')}"
            + (f" (+{r['content_after_check_s']}s)" if r.get("content_after_check_s") is not None else ""),
            flush=True,
        )
    checks = r.get("checks") or []
    if checks:
        states = [c["state"] for c in checks]
        print(
            f"  own request: checks={len(checks)} (held={states.count('held')} "
            f"free={states.count('free')} admitted={states.count('admitted')} "
            f"error={states.count('error')}) "
            f"held_after_sever={r['held_after_sever']} last_held_s={r['last_held_s']} "
            f"released_by_s={r['released_by_s']}",
            flush=True,
        )
        errs = [c for c in checks if c["state"] == "error"]
        if errs:
            print(f"  first undecided check: {json.dumps(errs[0])}", flush=True)
    if r.get("error"):
        print(f"  ERROR: {r['error']}", flush=True)
    print(
        f"  {r['case'].upper()}-CASE {verdict} (own request released_by_s={r['released_by_s']}, "
        f"accept <= {accept_seconds}s)",
        flush=True,
    )
    gv = r.get("global_verdict_pass")
    print(
        f"  info (not the verdict): global baseline running={r.get('baseline_running')} "
        f"queue={r.get('baseline_queue')} sum={r.get('baseline_sum')} "
        f"time_to_baseline_s={r['elapsed_to_baseline_s']}"
        + ("" if gv is None else f" (the global measure alone would say {'PASS' if gv else 'FAIL'})")
        + f" abort_line_seen={r['abort_seen']}"
        + (f" ({r['abort_line']!r})" if r["abort_line"] else "")
        + f" rid_line_seen={r['rid_line'] is not None}"
        + (f" ({r['rid_line']!r})" if r["rid_line"] else ""),
        flush=True,
    )


def run_both_cases(base, headers, model, log_path, prompt_tokens, seed, close_after,
                   window, poll_interval, accept_seconds, timeout, abort_re, held_re,
                   free_re, check_timeout, attempts, liveness_seconds=DEFAULT_LIVENESS_SECONDS,
                   quiet=False, legacy_sever=False):
    """The preflight, then both cases. Returns (results, overall, preflight) with
    preflight = {"ok", "rid", "check", "reason", "lora"} (rid and check None when the
    LoRA-off read refused); a refused preflight fires nothing and fails both cases."""
    ok, pf_rid, pf_check, pf_reason, pf_lora = preflight(
        base, headers, model, held_re, free_re, check_timeout)
    pf = {"ok": ok, "rid": pf_rid, "check": pf_check, "reason": pf_reason, "lora": pf_lora}
    if not quiet:
        print_preflight(pf)
    results = []
    for i, case in enumerate(CASES):
        if ok:
            r = run_case_attempts(
                base, headers, model, log_path, case, prompt_tokens, seed + i, close_after,
                window, poll_interval, accept_seconds, timeout, abort_re, held_re, free_re,
                check_timeout, attempts, liveness_seconds, legacy_sever,
            )
        else:
            r = not_run_result(case, pf_reason)
        if not quiet:
            print_case_result(r, accept_seconds)
        results.append(r)
    overall = ok and all(r["passed"] for r in results)
    return results, overall, pf


# --------------------------------------------------------------- self-test mock server


class _MockState:
    """The mock engine's books: its own running slots, phantom "live" requests that
    only show in /metrics, the per-rid table a real tokenizer_manager keeps, and every
    request it dispatched (checks it admitted, streaming requests it started)."""

    def __init__(self, live: int = 0):
        self.lock = threading.Lock()
        self.running = 0.0
        self.queue = 0.0
        self.live = float(live)
        self.rids: set[str] = set()
        self.admitted: list[str] = []
        self.streams: list[str] = []
        self.finished_early = 0
        # Every rid check that reached the mock (refused or not).
        self.checks_received = 0
        # rid -> the streaming request's server-side socket (instant_release reads it).
        self.conns: dict = {}

    def start(self):
        with self.lock:
            self.running += 1.0

    def finish(self):
        with self.lock:
            self.running = max(0.0, self.running - 1.0)

    def live_add(self, n: float):
        with self.lock:
            self.live = max(0.0, self.live + n)

    def hold(self, rid: str) -> bool:
        """Registers rid; False if it is already held (the duplicate refusal)."""
        with self.lock:
            if rid in self.rids:
                return False
            self.rids.add(rid)
            return True

    def is_held(self, rid: str) -> bool:
        with self.lock:
            return rid in self.rids

    def release(self, rid: str):
        with self.lock:
            self.rids.discard(rid)

    def take_finish_early(self, limit: int) -> bool:
        with self.lock:
            if self.finished_early < limit:
                self.finished_early += 1
                return True
            return False

    def snapshot(self):
        with self.lock:
            return self.running + self.live, self.queue


class _MockCfg:
    """One scenario's mock behaviour (see SELF_TEST_SCENARIOS)."""

    def __init__(self, well_behaved=True, live_mode=None, live_initial=0, register_delay=0.0,
                 lora_gate=True, drop_rid_on_decode=False, finish_early_midstream=0,
                 server_info_route="/server_info", enable_lora=None, lora_paths=None,
                 prefill_seconds=None, instant_release=False):
        # The one route the mock serves its config on (None: neither), and the LoRA
        # fields it reports there -- box-c's own answer is enable_lora null, lora_paths null.
        self.server_info_route = server_info_route
        self.enable_lora = enable_lora
        self.lora_paths = lora_paths
        # Overrides the self-test's prefill length (None: SELF_TEST_PREFILL_SECONDS).
        self.prefill_seconds = prefill_seconds
        # A severed request's rid is gone by the time any later check is answered (box-c
        # freeing a decoding request inside one check round trip).
        self.instant_release = instant_release
        self.well_behaved = well_behaved
        self.live_mode = live_mode
        self.live_initial = live_initial
        self.register_delay = register_delay
        # True: a free rid's LoRA-tagged check is refused like a real engine without
        # the adapter; False: it is dispatched (admitted) -- an engine the probe must
        # refuse to run against.
        self.lora_gate = lora_gate
        # The rid leaves the table once its request starts decoding, the slot stays.
        self.drop_rid_on_decode = drop_rid_on_decode
        # How many midstream requests end on their own right after the first token.
        self.finish_early_midstream = finish_early_midstream


def _sock_gone(sock) -> bool:
    """Whether the peer closed its end of `sock`: readable, and a peek returns EOF (a
    `Connection: close` client sends nothing after its request body)."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def _make_mock_handler(state: "_MockState", cfg: "_MockCfg", log_path: str, prefill_seconds: float,
                       tick: float, decode_interval: float, release_delay: float):
    if cfg.prefill_seconds is not None:
        prefill_seconds = cfg.prefill_seconds

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A003 - silence default access log
            pass

        def _write_chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _client_gone(self) -> bool:
            """Whether the client closed its end of this request's socket."""
            return _sock_gone(self.connection)

        def _refuse(self, message: str):
            self._send_json(400, {"object": "error", "type": "BadRequest", "code": 400,
                                  "message": message})

        def do_GET(self):  # noqa: N802 - stdlib handler naming
            if cfg.server_info_route is not None and self.path.split("?", 1)[0] == cfg.server_info_route:
                # The resolved args at top level, and a copy in internal_states, as the
                # fork's /server_info answers.
                lora = {"enable_lora": cfg.enable_lora, "lora_paths": cfg.lora_paths}
                self._send_json(200, {**lora, "model_path": "mock-model",
                                      "internal_states": [dict(lora)], "version": "mock"})
            elif self.path.startswith("/metrics"):
                running, queue = state.snapshot()
                body = (
                    f'sglang:num_running_reqs{{name="sglang",tp_rank="0"}} {running}\n'
                    f'sglang:num_queue_reqs{{name="sglang",tp_rank="0"}} {queue}\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):  # noqa: N802 - stdlib handler naming
            if not self.path.startswith("/v1/chat/completions"):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = b""
            try:
                while length > 0:
                    chunk = self.rfile.read(min(length, 1 << 16))
                    if not chunk:
                        break
                    raw += chunk
                    length -= len(chunk)
            except Exception:  # noqa: BLE001
                pass
            try:
                req = json.loads(raw or b"{}")
            except ValueError:
                req = {}
            rid = req.get("rid") or f"mock-{time.time_ns()}"

            if not req.get("stream"):
                # A rid check, refused like tokenizer_manager.generate_request does:
                # the duplicate test first, then the adapter -- both before dispatch.
                with state.lock:
                    state.checks_received += 1
                    stream_sock = state.conns.get(rid)
                if cfg.instant_release and stream_sock is not None and _sock_gone(stream_sock):
                    state.release(rid)  # freed before this check is answered
                if state.is_held(rid):
                    self._refuse(f"Duplicate request ID detected: {rid}")
                elif cfg.lora_gate and req.get("lora_path"):
                    self._refuse(
                        f"LoRA adapter '{req['lora_path']}' was requested, but LoRA is not "
                        "enabled. Please launch the server with --enable-lora flag and preload "
                        "adapters using --lora-paths or /load_lora_adapter endpoint."
                    )
                else:
                    with state.lock:
                        state.admitted.append(rid)
                    self._send_json(200, {"id": rid, "object": "chat.completion",
                                          "choices": [{"index": 0, "finish_reason": "length",
                                                       "message": {"role": "assistant", "content": "ok"}}]})
                return

            # The probe's own streaming request. A real engine registers the rid only
            # after the prompt's chat conversion: register_delay mimics it.
            if cfg.register_delay > 0:
                time.sleep(cfg.register_delay)
            if not state.hold(rid):
                self._refuse(f"Duplicate request ID detected: {rid}")
                return
            with state.lock:
                state.streams.append(rid)
                state.conns[rid] = self.connection
            state.start()
            if cfg.live_mode == "arrives":
                state.live_add(1)   # live traffic lands with us and stays
            elif cfg.live_mode == "departs":
                state.live_add(-1)  # a live request finishes as ours lands
            finish_early = (
                "-midstream-" in rid and cfg.finish_early_midstream > 0
                and state.take_finish_early(cfg.finish_early_midstream)
            )
            aborted = False
            try:
                # "queued/prefilling" phase: nothing is sent (the engine sends no
                # headers before the first token); every tick the handler asks the
                # SOCKET whether the client is gone, like tokenizer_manager's own
                # periodic `request.is_disconnected()` wait-tick -- a real EOF watch,
                # so only a client that really closed its connection is noticed.
                t0 = time.time()
                while time.time() - t0 < prefill_seconds:
                    time.sleep(tick)
                    if self._client_gone():
                        raise ConnectionResetError("client disconnected while queued")
                if cfg.drop_rid_on_decode:
                    state.release(rid)  # the table forgets it; the slot stays busy
                # The response head goes out with the first token, `Connection: close`
                # included, as uvicorn answers a `Connection: close` request -- which is
                # what makes http.client hand the socket from the connection to the
                # response (see fire_and_sever).
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.send_header("x-request-id", rid)
                self.end_headers()
                # "decoding" phase: stream content tokens one at a time, checking for
                # a disconnect before each one (serving_chat's per-output
                # `is_disconnected()` check) as well as failing on a write error.
                for i in range(200):
                    if self._client_gone():
                        raise ConnectionResetError("client disconnected mid-stream")
                    payload = json.dumps({"choices": [{"delta": {"content": f"word{i} "},
                                                       "finish_reason": None}]}).encode()
                    self._write_chunk(b"data: " + payload + b"\n\n")
                    if finish_early:
                        break
                    time.sleep(decode_interval)
                fin = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop" if finish_early else "length"}]}).encode()
                self._write_chunk(b"data: " + fin + b"\n\n")
                self._write_chunk(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                aborted = True
                if not cfg.well_behaved:
                    # The bug class this tool exists to catch: the fork notices the
                    # write failed but never frees the slot or the rid and never logs
                    # anything -- it just silently stops writing, holding the box.
                    return
            finally:
                if aborted:
                    if cfg.well_behaved:
                        # The abort round trip (tokenizer -> scheduler -> echo) before
                        # the rid is dropped: slot first, then the rid, like the engine.
                        time.sleep(release_delay)
                        state.finish()
                        state.release(rid)
                        try:
                            with open(log_path, "a") as f:
                                f.write(
                                    f"[{time.strftime('%H:%M:%S')}] abort: request {rid} "
                                    "aborted (client disconnected)\n"
                                )
                        except OSError:
                            pass
                else:
                    state.finish()
                    state.release(rid)

    return Handler


def _start_mock_server(cfg: "_MockCfg", prefill_seconds: float, tick: float, decode_interval: float,
                       release_delay: float):
    state = _MockState(live=cfg.live_initial)
    fd, log_path = tempfile.mkstemp(prefix="abort-probe-mock-", suffix=".log")
    os.close(fd)
    handler_cls = _make_mock_handler(state, cfg, log_path, prefill_seconds, tick,
                                     decode_interval, release_delay)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.daemon_threads = True
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    port = server.server_address[1]
    return server, th, f"http://127.0.0.1:{port}", log_path, state


# self-test sizing: small and fast, but every knob is still exercised end to end.
SELF_TEST_PROMPT_TOKENS = 400
SELF_TEST_PREFILL_SECONDS = 1.0
SELF_TEST_TICK = 0.1
SELF_TEST_DECODE_INTERVAL = 0.1
SELF_TEST_RELEASE_DELAY = 0.3
SELF_TEST_CLOSE_AFTER = 0.4
SELF_TEST_REGISTER_DELAY = 0.6   # > CLOSE_AFTER, < 2 * CLOSE_AFTER and < PREFILL_SECONDS
SELF_TEST_WINDOW = 3.0
SELF_TEST_POLL_INTERVAL = 0.2
SELF_TEST_ACCEPT_SECONDS = 2.0
SELF_TEST_TIMEOUT = 10.0
SELF_TEST_CHECK_TIMEOUT = 5.0
SELF_TEST_LIVENESS_SECONDS = 1.0  # > DECODE_INTERVAL: a live mock stream always shows a chunk

# Each scenario: label, mock config, the expected verdict per case, the expected global
# info verdict per case (None: not asserted), the minimum attempt a passing case must
# have needed, whether the preflight must pass, and what the pre-sever midstream check
# must have said (None: not asserted).
SELF_TEST_SCENARIOS = (
    {"label": "well-behaved", "cfg": _MockCfg(),
     "expect": {"queued": True, "midstream": True}},
    {"label": "stuck-fork", "cfg": _MockCfg(well_behaved=False),
     "expect": {"queued": False, "midstream": False}},
    {"label": "well-behaved, live traffic keeps the global sum up",
     "cfg": _MockCfg(live_mode="arrives"),
     "expect": {"queued": True, "midstream": True},
     "expect_global": {"queued": False, "midstream": False}},
    {"label": "stuck-fork, live traffic drops the global sum to baseline",
     "cfg": _MockCfg(well_behaved=False, live_mode="departs", live_initial=len(CASES)),
     "expect": {"queued": False, "midstream": False},
     "expect_global": {"queued": True, "midstream": True}},
    {"label": "well-behaved, rid registers after close_after",
     "cfg": _MockCfg(register_delay=SELF_TEST_REGISTER_DELAY),
     "expect": {"queued": True, "midstream": True},
     "min_attempt": {"queued": 2}},
    {"label": "well-behaved, the rid table drops a decoding request",
     "cfg": _MockCfg(drop_rid_on_decode=True),
     "expect": {"queued": True, "midstream": False},
     "pre_sever": "free"},
    {"label": "an engine that dispatches a LoRA-tagged check",
     "cfg": _MockCfg(lora_gate=False),
     "expect": {"queued": False, "midstream": False},
     "preflight_ok": False},
    {"label": "well-behaved, the first midstream request ends on its own",
     "cfg": _MockCfg(finish_early_midstream=1),
     "expect": {"queued": True, "midstream": True},
     "min_attempt": {"midstream": 2}},
    # The 2026-10-22 09:56Z live failure, reproduced: the pre-fix sever went through
    # conn.sock, which http.client clears once the `Connection: close` response head
    # has arrived, and the reader kept draining -- the mock's EOF watch never fires.
    # Before the head (queued) it still worked. The guard must call it a PROBE FAULT.
    {"label": "the pre-fix sever (through conn.sock, reader left draining)",
     "cfg": _MockCfg(),
     "legacy_sever": True,
     "expect": {"queued": True, "midstream": False},
     "probe_fault": {"midstream"}},
    # LoRA on: every rid check's failure cleanup would release an adapter reference it
    # never acquired. The preflight must refuse before a single check reaches the mock.
    {"label": "a LoRA-enabled engine (enable_lora true)",
     "cfg": _MockCfg(enable_lora=True, lora_paths=[{"lora_name": "a", "lora_path": "/a"}]),
     "expect": {"queued": False, "midstream": False},
     "preflight_ok": False, "lora_refusal": "LoRA is enabled"},
    {"label": "LoRA adapters configured (lora_paths, enable_lora null), /get_server_info only",
     "cfg": _MockCfg(server_info_route="/get_server_info", lora_paths=["/adapters/a"]),
     "expect": {"queued": False, "midstream": False},
     "preflight_ok": False, "lora_refusal": "/get_server_info"},
    {"label": "no /server_info route (LoRA-off unconfirmed)",
     "cfg": _MockCfg(server_info_route=None),
     "expect": {"queued": False, "midstream": False},
     "preflight_ok": False, "lora_refusal": "could not read the engine's config"},
    # The 2026-10-22 box-c run at --prompt-tokens 20000: the prefill ends inside
    # --close-after, so the queued request is decoding at the sever and is freed before
    # the first check -- never seen held, every attempt not judged, the hint printed.
    {"label": "well-behaved, the prefill ends inside close_after (a small --prompt-tokens)",
     "cfg": _MockCfg(prefill_seconds=0.05, instant_release=True),
     "expect": {"queued": False, "midstream": True},
     "queued_hint": f"first token arrived before the sever on {DEFAULT_ATTEMPTS} of {DEFAULT_ATTEMPTS}"},
)


def _self_test_run(scenario: dict) -> bool:
    label = scenario["label"]
    cfg = scenario["cfg"]
    expect = scenario["expect"]
    expect_global = scenario.get("expect_global") or {}
    min_attempt = scenario.get("min_attempt") or {}
    preflight_ok = scenario.get("preflight_ok", True)
    server, th, base, log_path, state = _start_mock_server(
        cfg, SELF_TEST_PREFILL_SECONDS, SELF_TEST_TICK, SELF_TEST_DECODE_INTERVAL,
        SELF_TEST_RELEASE_DELAY,
    )
    try:
        headers = {"Authorization": "Bearer mock"}
        results, overall, pf = run_both_cases(
            base, headers, "mock-model", log_path,
            SELF_TEST_PROMPT_TOKENS, random.randrange(10 ** 6), SELF_TEST_CLOSE_AFTER,
            SELF_TEST_WINDOW, SELF_TEST_POLL_INTERVAL, SELF_TEST_ACCEPT_SECONDS, SELF_TEST_TIMEOUT,
            re.compile(DEFAULT_ABORT_PATTERN), re.compile(DEFAULT_HELD_PATTERN),
            re.compile(DEFAULT_FREE_PATTERN), SELF_TEST_CHECK_TIMEOUT, DEFAULT_ATTEMPTS,
            SELF_TEST_LIVENESS_SECONDS, quiet=True,
            legacy_sever=scenario.get("legacy_sever", False),
        )
    finally:
        server.shutdown()
        server.server_close()
        th.join(timeout=5.0)
        try:
            os.unlink(log_path)
        except OSError:
            pass

    expect_overall = preflight_ok and all(expect.values())
    problems = []
    if pf["ok"] != preflight_ok:
        problems.append(f"preflight {'free' if pf['ok'] else 'refused'}, expected "
                        f"{'free' if preflight_ok else 'refused'} ({pf['reason']})")
    if overall != expect_overall:
        problems.append(f"overall {'PASS' if overall else 'FAIL'}, expected {'PASS' if expect_overall else 'FAIL'}")
    # Safety: no check may ever be dispatched under one of the probe's case rids.
    case_rids_admitted = [a for a in state.admitted if not a.startswith("abort-probe-preflight-")]
    if case_rids_admitted:
        problems.append(f"the mock dispatched a check under a case rid: {case_rids_admitted}")
    if not preflight_ok:
        if state.streams:
            problems.append(f"the preflight was refused yet {len(state.streams)} request(s) were fired")
    elif state.admitted:
        problems.append(f"the mock dispatched {len(state.admitted)} check(s) on a LoRA-refusing engine")
    lora_refusal = scenario.get("lora_refusal")
    if lora_refusal:
        reason = pf["reason"] or ""
        if pf["check"] is not None or state.checks_received:
            problems.append(f"the LoRA-off read refused, yet {state.checks_received} rid check(s) "
                            "reached the mock")
        if "PROBE FAULT" not in reason or lora_refusal not in reason:
            problems.append(f"preflight reason {reason!r} is not the PROBE FAULT naming {lora_refusal!r}")
    elif pf["check"] is None:
        problems.append(f"the LoRA-off read refused unexpectedly: {pf['reason']!r}")
    queued_hint = scenario.get("queued_hint")
    for r in results:
        has_hint = QUEUED_HINT_MARK in (r.get("error") or "")
        if r["case"] == "queued" and queued_hint:
            if not has_hint or queued_hint not in r["error"]:
                problems.append(f"queued: ERROR line lacks the prefill hint {queued_hint!r}: {r.get('error')!r}")
        elif has_hint:
            problems.append(f"{r['case']}: unexpected prefill hint: {r.get('error')!r}")
    for r in results:
        case = r["case"]
        if r["passed"] != expect[case]:
            problems.append(f"{case} {'PASS' if r['passed'] else 'FAIL'}, expected {'PASS' if expect[case] else 'FAIL'}")
        eg = expect_global.get(case)
        if eg is not None and r["global_verdict_pass"] != eg:
            problems.append(
                f"{case}: the global info row said {'PASS' if r['global_verdict_pass'] else 'FAIL'}, "
                f"the scenario needs it to say {'PASS' if eg else 'FAIL'} (not confounded)"
            )
        if r["passed"] and r.get("attempt", 1) < min_attempt.get(case, 1):
            problems.append(f"{case} passed on attempt {r.get('attempt')}, expected >= {min_attempt[case]}")
        if r["passed"] and case == "queued" and not r["held_after_sever"]:
            problems.append("queued passed without a held check after the sever")
        if r["passed"] and case == "midstream" and (r.get("pre_sever_check") or {}).get("state") != "held":
            problems.append("midstream passed without a held pre-sever check")
        if not expect[case] and r["released_by_s"] is not None and cfg.well_behaved is False:
            problems.append(f"{case}: a stuck request was reported released at {r['released_by_s']}s")
        # A LoRA-off refusal is a PROBE FAULT on every case it kept from running.
        fault_expected = case in (scenario.get("probe_fault") or ()) or bool(lora_refusal)
        is_fault = "PROBE FAULT" in (r.get("error") or "")
        if fault_expected and not is_fault:
            problems.append(f"{case}: expected a PROBE FAULT, got {r.get('error')!r}")
        if not fault_expected and is_fault:
            problems.append(f"{case}: unexpected PROBE FAULT: {r.get('error')!r}")
        if pf["ok"] and not fault_expected and r.get("sever_effective") is False:
            problems.append(f"{case}: the probe's sever did not close its connection")
        if case == "midstream" and scenario.get("pre_sever"):
            got = (r.get("pre_sever_check") or {}).get("state")
            if got != scenario["pre_sever"]:
                problems.append(f"midstream pre-sever check said {got}, expected {scenario['pre_sever']}")
            elif "no own-request signal" not in (r.get("error") or ""):
                problems.append(f"midstream failed for another reason: {r.get('error')!r}")
            if not r.get("live_after_check"):
                problems.append("midstream: the own-stream liveness after the free check was not seen")
    ok = not problems
    print(
        f"self-test [{label}]: preflight={'free' if pf['ok'] else 'REFUSED'} cases="
        + ", ".join(
            f"{r['case']}={'PASS' if r['passed'] else 'FAIL'}"
            f"(attempt {r.get('attempt', 1)}, released_by_s={r['released_by_s']}, "
            f"global={'PASS' if r['global_verdict_pass'] else 'FAIL'})"
            for r in results
        )
        + f" overall={'PASS' if overall else 'FAIL'} (expected {'PASS' if expect_overall else 'FAIL'}) "
        + f"checks_received={state.checks_received} checks_dispatched={len(state.admitted)} "
        + ("OK" if ok else "MISMATCH: " + "; ".join(problems)),
        flush=True,
    )
    if not ok:
        for r in results:
            print(f"    {r['case']}: {json.dumps({k: v for k, v in r.items() if k != 'samples'})}", flush=True)
    return ok


def self_test() -> int:
    """Runs the real preflight and case logic (preflight / fire_and_sever / rid_check /
    run_case / run_case_attempts / run_both_cases) against the bundled mock in the thirteen
    SELF_TEST_SCENARIOS (see the module docstring)."""
    ok = True
    for scenario in SELF_TEST_SCENARIOS:
        ok = _self_test_run(scenario) and ok
    print("ABORT-PROBE SELF-TEST", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Abort probe: proves a client disconnect (queued/prefilling, and "
        "mid-stream) actually frees the probe's own request within --accept-seconds."
    )
    ap.add_argument("--base-url", default=None, help="engine base URL (default: boxenv.resolve_engine_url())")
    ap.add_argument("--model", default=None, help="served model name (default: boxenv.resolve_served_model())")
    ap.add_argument("--api-key", default=None, help="API key value (default: --env-file, else boxenv)")
    ap.add_argument(
        "--env-file", default=None,
        help="env file to read VLLM_API_KEY from directly when --api-key/$VLLM_API_KEY are unset "
             "(first-choice override before falling back to boxenv's own resolution)",
    )
    ap.add_argument("--log", default=None, help="serve log to scan for an abort line / the rid, info only (default: boxenv.serve_log_default())")
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS, help=f"approx tokens in each case's cold prompt (default {DEFAULT_PROMPT_TOKENS}); the queued case needs a prompt whose prefill outlasts --close-after -- a small one such as 20000 validates the midstream case only (see GUARD)")
    ap.add_argument("--seed", type=int, default=None, help="prompt content seed (default: derived from the current time, so re-runs are always cold)")
    ap.add_argument("--close-after", type=float, default=2.0, help="seconds after firing before severing the 'queued' case's socket; doubled on each queued retry (default 2)")
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help=f"tries of a case whose attempt is not judged -- the queued case's first post-sever check already free, or the request ending on its own before the sever (default {DEFAULT_ATTEMPTS})")
    ap.add_argument("--window", type=float, default=15.0, help="seconds to keep checking the rid (and sampling /metrics + the log) after severing, per case (default 15)")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="rid check + /metrics + log poll interval, seconds (default 0.5)")
    ap.add_argument("--accept-seconds", type=float, default=6.0, help="a case passes if its own request is released within this many seconds of the sever (default 6)")
    ap.add_argument("--liveness-seconds", type=float, default=DEFAULT_LIVENESS_SECONDS, help=f"midstream: how long to wait after the pre-sever rid check's answer for one more content chunk on the probe's own stream (default {DEFAULT_LIVENESS_SECONDS})")
    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP socket timeout, and the 'midstream' case's ceiling for observing a first content token, seconds (default 180)")
    ap.add_argument("--check-timeout", type=float, default=30.0, help="HTTP timeout of one rid check, seconds (default 30)")
    ap.add_argument("--held-pattern", default=DEFAULT_HELD_PATTERN, help=f"regex (re.search) a rid check's HTTP 400 body must match to count as 'still held' (default {DEFAULT_HELD_PATTERN!r})")
    ap.add_argument("--free-pattern", default=DEFAULT_FREE_PATTERN, help=f"regex (re.search) a rid check's HTTP 400 body must match to count as 'free' -- a LoRA-off engine's refusal of the never-loaded check adapter (default {DEFAULT_FREE_PATTERN!r})")
    ap.add_argument("--abort-pattern", default=DEFAULT_ABORT_PATTERN, help=f"regex (re.search) for the info-only serve-log abort scan (default {DEFAULT_ABORT_PATTERN!r})")
    ap.add_argument("--self-test", action="store_true", help="bounded run against a bundled local mock (thirteen scenarios) instead of a real engine")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if args.log is None:
        args.log = boxenv.serve_log_default()

    base = boxenv.resolve_engine_url(args.base_url)
    print(f"base_url={base}")
    model = boxenv.resolve_served_model(args.model)
    key = resolve_key(args.api_key, args.env_file)
    headers = {"Authorization": "Bearer " + key}

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2 ** 31)
    abort_re = re.compile(args.abort_pattern)
    held_re = re.compile(args.held_pattern)
    free_re = re.compile(args.free_pattern)

    print(
        f"prompt_tokens~{args.prompt_tokens} seed={seed} close_after={args.close_after}s "
        f"attempts={args.attempts} window={args.window}s poll_interval={args.poll_interval}s "
        f"accept_seconds={args.accept_seconds}s liveness_seconds={args.liveness_seconds}s "
        f"verdict=own-request-release check_adapter={CHECK_ADAPTER} log={args.log}"
    )

    results, overall, _pf = run_both_cases(
        base, headers, model, args.log, args.prompt_tokens, seed, args.close_after,
        args.window, args.poll_interval, args.accept_seconds, args.timeout, abort_re,
        held_re, free_re, args.check_timeout, args.attempts, args.liveness_seconds,
    )

    print(
        "ABORT-PROBE "
        + ("PASS" if overall else "FAIL")
        + ": "
        + ", ".join(f"{r['case']}={'PASS' if r['passed'] else 'FAIL'}" for r in results)
    )
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
