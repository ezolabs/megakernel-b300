#!/usr/bin/env python3
"""Speed-baseline driver for box-a through its engine door or its engine loopback port.

Companion to ramp.py and tails-probe.py for the three measurements
those tools cannot take as written (see baseline.md §1b, §2 and §3):

  think-ramp  per-stream decode tok/s with THINKING ON (the router's default effort), TTFT
              taken at the first delta of either kind (reasoning_content or content) and
              excluded from the decode rate. ramp.py sends reasoning_effort "none" and
              starts its clock at the first CONTENT delta, so on a thinking reply its
              "decode" would divide all tokens by the content-only tail.
  resume      resume-after-idle TTFT: seeds one ~87k-token (--ctx-tokens 70000) conversation per gap (prefix
              built by tails-probe.py's own build_prefix, imported, so the content shape
              is that tool's), then resumes each after its idle gap with one appended
              user turn and records TTFT + usage cached_tokens. tails-probe.py itself has
              no idle-gap mode and no TTFT (non-streamed, turn 1 never scored).
  coldfill    4 background thinking streams decode; once all are decoding, one unique
              ~60k-token (--fill-tokens 48300) cold prompt is sent. Reports background per-stream tok/s
              before / during / after the fill window and the fill's wall time.

and, for engine A/Bs (procedure: METHOD.md beside this file):

  compare     compare A.jsonl B.jsonl [A2.jsonl [B2.jsonl ...]] [--gate PCT]: reads the
              per-request records the three modes above write (--records), and per cell
              (mode + concurrency / gap / fill role) reports the median of each arm, B-A and
              its bootstrap 95% CI (rep-clustered percentile bootstrap, --boot resamples,
              >= 2000). Files are boots in boot order, alternating A and B: with a third file
              A2 (= A re-booted) it also reports the noise band |A2-A| (the point estimate)
              and B-A2; a fourth, B2 (= B re-booted, METHOD.md's default A, B, A2, B2), adds
              B2-A2, and a longer chain (A3, B3 ...) carries on. Verdict: ADOPT only if the CI
              lower bound of every B boot's gain over each adjacent A boot exceeds
              max(--gate, band) (band: the widest consecutive A pair); HOLD when the
              cell cannot be judged (under 6 reps, errored requests, requests refused because
              loads_before shows the box busier than the run's --max-live, settings or
              connection modes ("conn"; none = "new") differ).
              ADOPT also needs a second B boot (METHOD.md §2): a three-boot chain whose B
              clears the bar HOLDs, needing B2; one whose B misses it is NO (more B boots
              only add tests). --gate with two files is refused: a verdict needs A, B and A2.
              compare A.jsonl A2.jsonl --aa reports the noise band alone. Gains are signed so
              positive is better (decode tok/s up, TTFT down). Each cell scores its own
              metrics (think-ramp dec + ttft, resume ttft, coldfill fill ttft, coldfill
              bg-during dec); --metric only narrows that, names on stderr any it names that
              no cell carries, and exits non-zero if nothing is scored. A cell with no scorable value in an
              arm (e.g. every coldfill background stream ended inside the fill window) HOLDs
              with its excluded count.
  selftest    CPU unit test of the bootstrap / compare logic on synthetic data (no box,
              no key, no network). selftest --simulate instead re-runs METHOD.md §5's
              adoption-rate table (--trials, --null-trials, --boot, --boot-seed).

Per-request records (--records, default <--out>.req.jsonl when --out is given): one JSON
line per stream with kind "req", run id, arm label (--arm), mode, cell, rep, stream index,
decode tok/s excluding TTFT ("dec"), TTFT, tokens, finish, error, the run's --max-live and
the loads reading taken before it (coldfill background streams also carry
ended_before_fill_end). The row output of the three modes (stdout / --out) is unchanged by them.

Connections: requests share one LIFO pool of keep-alive connections per host; a request takes
the most recently returned idle one, or opens one when none is idle (so concurrent requests each
hold their own), and only a request on a new connection pays the TCP+TLS handshake and the cold
upload of its body. A pooled connection the far end closed while idle is retried once on a new
connection outside the pool, and every other idle one to that host is closed. HTTPS_PROXY and
the other proxy variables are ignored on this path (urllib honoured them). --new-connection keeps the old path (`Connection: close`, a new
connection per request: ~0.27 s more per request through a gateway, #300). Every record and row
carries "conn" ("keep-alive" | "new"); records also carry "conn_reused".

Every rung/probe snapshots the engine's /v1/loads: over ssh (read-only curl on the box
loopback), or with --on-box by running that curl locally when bench.py runs ON the box
(base URL then defaults to http://127.0.0.1:$ENGINE_PORT, key from $VLLM_API_KEY or
/etc/environment; no box file). Off the box, --base-url names the engine or its door and
the key comes from $VLLM_API_KEY or the box's /etc/environment over ssh.
The gate before each rung fails closed: if /v1/loads cannot be read after --gate-tries
attempts (backoff 2, 4, 8 ... s), the run stops with exit 2 rather than start a rung ungated.
Box access: boxes/$BOX.env (default box-a).
Stdlib only.  Key: $VLLM_API_KEY, else the box's /etc/environment (off-box, over ssh).
"""
import argparse
import http.client
import importlib.util
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "box"))
UA = "curl/8.7.1"
def box_env(box=None):
    """Box access from boxes/$BOX.env beside this tool set (BOX_IP, SSH_PORT, SSH_KEY, ENGINE_PORT; SSH_USER, default root)."""
    box = box or os.environ.get("BOX", "box-a")
    path = os.environ.get("BOX_ENV") or os.path.join(TOOLS, "boxes", f"{box}.env")
    if not os.path.isfile(path):
        sys.exit(f"bench.py: box file {path} not found (set BOX or BOX_ENV)")
    env = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("BOX_IP", "SSH_PORT", "SSH_KEY") if not env.get(k)]
    if missing:
        sys.exit(f"bench.py: {path} lacks {', '.join(missing)}")
    return env

# Box access is read lazily, so compare / selftest (and --on-box) need no box file.
_BOX = None
ON_BOX = False  # set by --on-box: bench.py runs on the box itself, /v1/loads read locally

def box():
    global _BOX
    if _BOX is None:
        _BOX = box_env()
    return _BOX

def engine_port():
    return (os.environ.get("ENGINE_PORT") if ON_BOX else box().get("ENGINE_PORT")) or "8002"

def ssh_cmd():
    b = box()
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR",
            "-i", os.path.expanduser(b["SSH_KEY"]), "-p", b["SSH_PORT"],
            f"{b.get('SSH_USER') or 'root'}@{b['BOX_IP']}"]

# The engine key lives on the box (/etc/environment), never in a tracked file. Same source
# order as engine_key(): $VLLM_API_KEY first, then /etc/environment.
def loads_cmd():
    return ('K="${VLLM_API_KEY:-$(grep ^VLLM_API_KEY= /etc/environment | cut -d= -f2- | tr -d \'"\')}"; '
            'curl -s -m5 -H "Authorization: Bearer $K" localhost:%s/v1/loads' % engine_port())

MODEL = "DeepSeek-V4.1-Flash"

def engine_key():
    """--on-box: the engine's own key, from $VLLM_API_KEY or the box's /etc/environment."""
    k = os.environ.get("VLLM_API_KEY")
    if k:
        return k
    try:
        for line in open("/etc/environment"):
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    sys.exit("bench.py --on-box: no VLLM_API_KEY in the environment or /etc/environment")

def remote_engine_key():
    """Off-box: the engine's own key, read from the box's /etc/environment over ssh (never a
    tracked file), or $VLLM_API_KEY in this shell."""
    k = os.environ.get("VLLM_API_KEY")
    if k:
        return k
    out = subprocess.run(ssh_cmd() + ['grep ^VLLM_API_KEY= /etc/environment | cut -d= -f2- | tr -d \'"\''],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    if not out:
        sys.exit("bench.py: no VLLM_API_KEY in the environment or in the box's /etc/environment")
    return out

def loads():
    try:
        cmd = ["sh", "-c", loads_cmd()] if ON_BOX else ssh_cmd() + [loads_cmd()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        a = json.loads(out[out.index("{"):])["aggregate"]
        return {"run": a["num_running_reqs"], "wait": a["num_waiting_reqs"], "kv": round(a["kv_usage"], 3)}
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:80]}

class LoadGateError(RuntimeError):
    """/v1/loads could not be read, so the live-load gate cannot be checked."""

def wait_quiet(max_running, log, tries=5, backoff=2.0):
    """Block until live #running-req <= max_running and return that reading.

    Fails closed: a read error is retried up to `tries` times in a row (sleeping backoff,
    2*backoff, 4*backoff ... between), then LoadGateError is raised so no rung starts ungated.
    """
    fails = 0
    while True:
        l = loads()
        if "err" in l:
            fails += 1
            if fails >= tries:
                raise LoadGateError(f"/v1/loads unreadable {fails}x in a row (last: {l['err']}); "
                                    f"refusing to start a rung without the #running-req <= {max_running} gate")
            delay = backoff * 2 ** (fails - 1)
            log(f"# /v1/loads read failed ({l['err']}); retry {fails}/{tries - 1} in {delay:g} s")
            time.sleep(delay)
            continue
        fails = 0
        if l["run"] <= max_running:
            return l
        log(f"live #running-req {l['run']} > {max_running}; waiting 20 s")
        time.sleep(20)

# Connection handling (METHOD.md §1). By default requests reuse pooled keep-alive connections:
# one shared pool per host, last in first out; a request takes the most recently returned idle
# connection (or opens one when none is idle, so concurrent requests each hold their own), and
# hands it back once the stream is read to the end, so a request on a reused connection skips the
# TCP+TLS handshake and the cold slow-start upload of its body (#300: ~0.27 s a request through
# a gateway). --new-connection keeps the old path: urllib, `Connection: close`, a new connection per
# request. Every record and row names the mode that ran ("conn": "keep-alive" | "new").
NEW_CONNECTION = False  # set by --new-connection
_POOL = {}  # (scheme, host, port) -> [idle http.client connections]
_POOL_LOCK = threading.Lock()

def conn_mode():
    return "new" if NEW_CONNECTION else "keep-alive"

def _conn_get(u, timeout, fresh=False):
    """An idle pooled connection to u's host (reused True), else a new one (reused False).
    fresh: always a new one, never from the pool."""
    key = (u.scheme, u.hostname, u.port)
    c = None
    if not fresh:
        with _POOL_LOCK:
            idle = _POOL.get(key)
            c = idle.pop() if idle else None
    if c is not None:
        c.timeout = timeout
        if c.sock is not None:
            c.sock.settimeout(timeout)
        return c, True
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port, timeout=timeout), False

def _conn_put(u, c):
    with _POOL_LOCK:
        _POOL.setdefault((u.scheme, u.hostname, u.port), []).append(c)

def _conn_flush(u):
    """Close every idle pooled connection to u's host: once one is found closed by the far end,
    the rest idled as long or longer (the far end's idle timeout) and are taken as stale too."""
    with _POOL_LOCK:
        idle = _POOL.pop((u.scheme, u.hostname, u.port), [])
    for c in idle:
        c.close()

# A pooled connection the far end closed while it sat idle fails on first use with one of these;
# the request is then sent once more on a NEW connection that bypasses the pool, and every other
# idle connection to that host is closed (_conn_flush). A failure on a new connection is an error.
_STALE = (http.client.RemoteDisconnected, http.client.CannotSendRequest, BrokenPipeError,
          ConnectionResetError, ConnectionAbortedError)

def _open_stream(url, data, headers, timeout):
    """Send the POST; return (response line iterator, finish(), t0, reused).

    finish(ok) is called once the stream is done: ok returns a fully read keep-alive connection
    to the pool, anything else closes it.
    """
    if NEW_CONNECTION:
        req = urllib.request.Request(url, data=data, headers={**headers, "Connection": "close"}, method="POST")
        t0 = time.time()
        r = urllib.request.urlopen(req, timeout=timeout)
        return r, lambda ok: r.close(), t0, False
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/") + ("?" + u.query if u.query else "")
    for attempt in (1, 2):
        c, reused = _conn_get(u, timeout, fresh=attempt == 2)
        t0 = time.time()
        try:
            c.request("POST", path, body=data, headers=headers)
            r = c.getresponse()
        except _STALE:
            c.close()
            if reused:
                _conn_flush(u)
                continue
            raise
        except BaseException:
            c.close()
            raise
        break
    if r.status >= 300:
        c.close()
        raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)

    def finish(ok):
        if ok:
            try:
                r.read()  # the rest of the body (past [DONE]) so the connection is clean for the next request
            except Exception:  # noqa: BLE001
                ok = False
        if ok and not r.will_close:
            _conn_put(u, c)
        else:
            c.close()
    return r, finish, t0, reused

def stream(base, key, body, timeout=900):
    """POST a streamed chat completion; return timing + per-chunk (t, chars) list."""
    url = base.rstrip("/") + "/v1/chat/completions"
    data = json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode()
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": UA}
    t0 = time.time()
    first = None
    chunks = []
    usage = None
    text = []
    finish = None
    reused = None
    try:
        r, done, t0, reused = _open_stream(url, data, headers, timeout)
        ok = False
        try:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:  # noqa: BLE001
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                ch = o.get("choices") or []
                if not ch:
                    continue
                finish = ch[0].get("finish_reason") or finish
                delta = ch[0].get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    chunks.append((now, len(piece)))
                    text.append(delta.get("content") or "")
            ok = True
        finally:
            done(ok)
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:200], "t0": t0, "conn_reused": reused}
    t_end = time.time()
    ct = (usage or {}).get("completion_tokens")
    out = {"t0": t0, "t_first": first, "t_end": t_end, "ttft": (first - t0) if first else None,
           "wall": t_end - t0, "ct": ct, "pt": (usage or {}).get("prompt_tokens"),
           "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
           "finish": finish, "chunks": chunks, "content": "".join(text), "conn_reused": reused}
    if first and ct and t_end > first:
        out["dec"] = ct / (t_end - first)
    return out

THINK_PROMPTS = [
    "A train leaves city A at {a} km/h and another leaves city B, {d} km away, at {b} km/h toward it. "
    "A bird flies between them at {c} km/h until they meet. Work out carefully how far the bird flies, "
    "checking your arithmetic twice. (ref {r})",
    "Find all integer solutions of x^2 - {a}y^2 = 1 with 0 < x < {d}, reasoning step by step and "
    "verifying each candidate. (ref {r})",
    "Write and reason through a Python function that returns the {a}th prime palindrome greater than {d}; "
    "explain the algorithm's complexity. (ref {r})",
    "A warehouse has {a} boxes of {b} kg and {c} boxes of {d} kg. Plan how to load them into trucks with a "
    "1,000 kg limit using as few trucks as possible; justify optimality. (ref {r})",
]

def think_prompt(rng):
    t = rng.choice(THINK_PROMPTS)
    return t.format(a=rng.randint(3, 97), b=rng.randint(3, 97), c=rng.randint(3, 97),
                    d=rng.randint(100, 999), r=rng.randrange(10 ** 12))

def record(args, mode, cell, rep, i, r, **extra):
    """Write one per-request record (kind "req") for stream result r to args.rec, if set.

    dec = completion tokens / (t_end - first delta), so TTFT is excluded from the rate.
    """
    if not getattr(args, "rec", None):
        return
    r = r or {"err": "no result"}
    o = {"kind": "req", "run": args.run_id, "arm": args.arm, "mode": mode, "cell": cell,
         "rep": rep, "i": i, "seed": args.seed, "seed_offset": args.seed_offset,
         "base": args.base_url, "max_live": args.max_live, "t0": round(r["t0"], 3) if r.get("t0") else None,
         "conn": conn_mode()}
    if r.get("conn_reused") is not None:
        o["conn_reused"] = r["conn_reused"]
    if "err" in r:
        o["err"] = r["err"]
    else:
        o.update({"dec": round(r["dec"], 2) if r.get("dec") else None,
                  "ttft": round(r["ttft"], 4) if r.get("ttft") is not None else None,
                  "ct": r.get("ct"), "pt": r.get("pt"), "cached": r.get("cached"), "finish": r.get("finish")})
    o.update(extra)
    args.rec(json.dumps(o))

def cmd_think_ramp(args, key, log):
    rng = random.Random(args.seed)
    rows = []
    for rep in range(1, args.reps + 1):
        for C in [int(x) for x in args.levels.split(",")]:
            l0 = wait_quiet(args.max_live, log, args.gate_tries)
            body = lambda: {"model": MODEL, "max_tokens": args.max_tokens, "temperature": 1.0, "top_p": 0.95,
                            "messages": [{"role": "user", "content": think_prompt(rng)}]}
            out = [None] * C
            bodies = [body() for _ in range(C)]
            ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, stream(args.base_url, key, bodies[i])))
                   for i in range(C)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            for i, r in enumerate(out):
                record(args, "think-ramp", f"C={C}", rep, i, r, C=C, max_tokens=args.max_tokens, loads_before=l0)
            ok = [r for r in out if r and "err" not in r and r.get("dec")]
            errs = [r["err"] for r in out if r and "err" in r]
            dec = sorted(r["dec"] for r in ok)
            ttft = sorted(r["ttft"] for r in ok)
            span = (max(r["t_end"] for r in ok) - min(r["t_first"] for r in ok)) if ok else 0
            row = {"rep": rep, "C": C, "t": time.strftime("%H:%M:%SZ", time.gmtime(min(r["t0"] for r in out if r))),
                   "n_ok": len(ok), "errs": len(errs), "first_err": errs[0] if errs else None,
                   "dec_med": round(st.median(dec), 1) if dec else None,
                   "dec_min": round(dec[0], 1) if dec else None, "dec_max": round(dec[-1], 1) if dec else None,
                   "agg_tok_s": round(sum(r["ct"] for r in ok) / span, 0) if span else None,
                   "ttft_med": round(st.median(ttft), 3) if ttft else None,
                   "ttft_max": round(ttft[-1], 3) if ttft else None,
                   "ct_med": st.median([r["ct"] for r in ok]) if ok else None,
                   "finish": sorted({r["finish"] for r in ok}), "loads_before": l0, "conn": conn_mode()}
            log(json.dumps(row))
            rows.append(row)
            time.sleep(args.pause)
    return rows

def load_tails_probe():
    spec = importlib.util.spec_from_file_location("tails_probe", os.path.join(TOOLS, "tails-probe.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, TOOLS)
    spec.loader.exec_module(mod)
    return mod

def cmd_resume(args, key, log):
    tp = load_tails_probe()
    tp.SEED_OFFSET = args.seed_offset
    gaps = [float(g) for g in args.gaps_min.split(",")]
    convs = []
    for i, g in enumerate(gaps):
        prefix = tp.build_prefix(i, args.ctx_tokens)
        user1 = (f"{prefix}\n\nInstruction: the paragraph above is unique to this conversation. "
                 f"Reply with exactly the words 'READY {i}' and nothing else.")
        msgs = [{"role": "user", "content": user1}]
        l0 = wait_quiet(args.max_live, log, args.gate_tries)
        r = stream(args.base_url, key, {"model": MODEL, "messages": msgs, "max_tokens": 16,
                                        "temperature": 0.0, "reasoning_effort": "none"})
        if "err" in r:
            log(json.dumps({"seed": i, "gap_min": g, "err": r["err"], "conn": conn_mode()}))
            continue
        msgs.append({"role": "assistant", "content": r["content"]})
        c = {"i": i, "gap_min": g, "msgs": msgs, "seeded_at": r["t_end"], "seed_pt": r["pt"]}
        convs.append(c)
        log(json.dumps({"seed": i, "gap_min": g, "pt": r["pt"], "cached": r["cached"],
                        "ttft": round(r["ttft"], 3), "loads": l0,
                        "at": time.strftime("%H:%M:%SZ", time.gmtime(r["t_end"])), "conn": conn_mode()}))
    rows = []
    for c in sorted(convs, key=lambda c: c["gap_min"]):
        due = c["seeded_at"] + 60 * c["gap_min"]
        while time.time() < due:
            time.sleep(min(30, max(0.5, due - time.time())))
        msgs = c["msgs"] + [{"role": "user", "content": f"Resume after {c['gap_min']:g} min idle: reply with "
                                                         f"exactly 'READY {c['i']}.2' and nothing else."}]
        lb = loads()
        r = stream(args.base_url, key, {"model": MODEL, "messages": msgs, "max_tokens": 16,
                                        "temperature": 0.0, "reasoning_effort": "none"})
        record(args, "resume", f"gap={c['gap_min']:g}m", 1, c["i"], r, gap_min=c["gap_min"],
               ctx_tokens=args.ctx_tokens, loads_before=lb)
        row = {"gap_min": c["gap_min"], "conv": c["i"], "loads": lb,
               "at": time.strftime("%H:%M:%SZ", time.gmtime(time.time())), "conn": conn_mode()}
        if "err" in r:
            row["err"] = r["err"]
        else:
            row.update({"actual_gap_s": round(r["t0"] - c["seeded_at"], 1), "pt": r["pt"], "cached": r["cached"],
                        "hit": round((r["cached"] or 0) / r["pt"], 4) if r["pt"] else None,
                        "ttft": round(r["ttft"], 3)})
        log(json.dumps(row))
        rows.append(row)
    return rows

def cmd_coldfill(args, key, log):
    tp = load_tails_probe()
    tp.SEED_OFFSET = args.seed_offset
    rng = random.Random(args.seed)
    l0 = wait_quiet(args.max_live, log, args.gate_tries)
    bg = [None] * args.bg
    bodies = [{"model": MODEL, "max_tokens": args.bg_tokens, "temperature": 1.0, "top_p": 0.95,
               "messages": [{"role": "user", "content": think_prompt(rng) +
                             " Be extremely thorough; explore several approaches in full."}]}
              for _ in range(args.bg)]
    ths = [threading.Thread(target=lambda i=i: bg.__setitem__(i, stream(args.base_url, key, bodies[i], 1800)))
           for i in range(args.bg)]
    for t in ths:
        t.start()
    time.sleep(args.lead_s)
    lmid = loads()
    fill_prompt = (tp.build_prefix(7000 + args.seed_offset, args.fill_tokens) +
                   "\n\nReply with exactly the word READY and nothing else.")
    fr = stream(args.base_url, key, {"model": MODEL, "messages": [{"role": "user", "content": fill_prompt}],
                                     "max_tokens": 16, "temperature": 0.0, "reasoning_effort": "none"})
    lfill = loads()
    for t in ths:
        t.join()
    record(args, "coldfill", "fill", 1, 0, fr, fill_tokens=args.fill_tokens, loads_before=l0)
    if "err" in fr:
        log(json.dumps({"fill_err": fr["err"], "conn": conn_mode()}))
        return {}
    f0, f1 = fr["t0"], fr["t_first"]

    def rate(r, a, b):
        tot = sum(n for _, n in r["chunks"])
        if not tot or not r["ct"]:
            return None
        cpt = tot / r["ct"]  # chars per token over the whole stream
        chars = sum(n for t, n in r["chunks"] if a <= t < b)
        return round(chars / cpt / (b - a), 1) if b > a else None

    def maxgap(r, a, b):
        ts = [t for t, _ in r["chunks"] if a - 5 <= t <= b + 5]
        g = [y - x for x, y in zip(ts, ts[1:]) if y >= a and x <= b]
        return round(max(g), 3) if g else None

    per = []
    for r in bg:
        if not r or "err" in r:
            per.append({"err": (r or {}).get("err")})
            record(args, "coldfill", "bg-during", 1, len(per) - 1, r, bg_tokens=args.bg_tokens, loads_before=l0)
            continue
        pre_a = max(r["t_first"] + 1.0, f0 - 15)
        post_b = min(r["t_end"], f1 + 15)
        per.append({"ct": r["ct"], "finish": r["finish"], "dec_whole": round(r.get("dec") or 0, 1),
                    "pre": rate(r, pre_a, f0), "during": rate(r, f0, f1), "post": rate(r, f1, post_b),
                    "max_gap_during_s": maxgap(r, f0, f1), "ended_before_fill_end": r["t_end"] < f1})
        # The background streams' rate DURING the fill window is what a prefill knob moves;
        # recorded as dec so compare scores it like any decode rate. A stream that ended inside
        # the window has no whole-window rate: it is recorded with ended_before_fill_end true and
        # no dec, and compare counts it as excluded.
        early = per[-1]["ended_before_fill_end"]
        record(args, "coldfill", "bg-during", 1, len(per) - 1,
               {"t0": r["t0"], "dec": None if early else per[-1]["during"], "ttft": None, "ct": r["ct"],
                "finish": r["finish"]},
               bg_tokens=args.bg_tokens, loads_before=l0, ended_before_fill_end=early,
               pre=per[-1]["pre"], post=per[-1]["post"], max_gap_during_s=per[-1]["max_gap_during_s"])
    row = {"t": time.strftime("%H:%M:%SZ", time.gmtime(f0)), "fill_pt": fr["pt"], "fill_cached": fr["cached"],
           "fill_ttft_s": round(fr["ttft"], 3), "fill_wall_s": round(fr["wall"], 3),
           "fill_prefill_tok_s": round((fr["pt"] - (fr["cached"] or 0)) / fr["ttft"], 0) if fr["pt"] else None,
           "bg": per, "loads_before": l0, "loads_bg_running": lmid, "loads_after_fill": lfill, "conn": conn_mode()}
    log(json.dumps(row))
    return row

# ---- compare: per-cell median differences with a rep-clustered bootstrap 95% CI ----------

METRICS = {"dec": True, "ttft": False}  # metric -> higher is better
# Which metric scores which cell (METHOD.md §4). A metric a cell does not carry, or carries only
# as a meaningless rate, is never scored there, whatever --metric names: resume and the coldfill
# fill request generate 16 tokens, so their "dec" is no decode rate, and a coldfill background
# stream has no TTFT of its own inside the fill window. Key (mode, cell) first, then (mode, None).
MODE_METRICS = {("think-ramp", None): ("dec", "ttft"),
                ("resume", None): ("ttft",),
                ("coldfill", "fill"): ("ttft",),
                ("coldfill", "bg-during"): ("dec",)}

def cell_metrics(mode, cell):
    """The metrics scored for (mode, cell); a mode this table does not know scores all of them."""
    return MODE_METRICS.get((mode, cell)) or MODE_METRICS.get((mode, None)) or tuple(METRICS)
CFG_KEYS = ("max_tokens", "ctx_tokens", "fill_tokens", "bg_tokens")  # must match across arms
MIN_BOOT = 2000
MIN_REPS = 6

def read_records(path):
    """The per-request records (kind "req") in a --records file; other lines are skipped."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if isinstance(o, dict) and o.get("kind") == "req":
                recs.append(o)
    if not recs:
        raise ValueError(f"{path}: no per-request records (kind \"req\"): pass the --records "
                         f"file (<out>.req.jsonl), not the --out rows")
    return recs

def group_cells(recs, metric, mode=None):
    """{(mode, cell): {"clusters": [[v, ...], ...], "n_err", "n_refused", "n_excluded", "cfg", "seeds", "conn"}}.

    A cluster is every value of one (run, rep): the streams of one rung share the batch, so
    they are not independent draws, and the bootstrap resamples whole rungs, not streams.
    Never scored, each counted:
      n_err       errored requests;
      n_refused   requests that started on a busy or unverified box: the record's
                  loads_before shows #running-req above that run's --max-live, or is
                  unreadable (records from before max_live was recorded are not checked);
      n_excluded  requests with no value for the metric: a coldfill background stream that
                  ended inside the fill window (ended_before_fill_end), a stream with no delta.
    """
    cells = {}
    for o in recs:
        if mode and o.get("mode") != mode:
            continue
        c = cells.setdefault((o.get("mode"), o.get("cell")),
                             {"by": {}, "n_err": 0, "n_refused": 0, "n_excluded": 0, "cfg": set(), "seeds": set(),
                              "conn": set()})
        c["cfg"].add(json.dumps({k: o[k] for k in CFG_KEYS if k in o}, sort_keys=True))
        # A record from before the keep-alive change has no "conn": it ran the per-request
        # `Connection: close` path, which is --new-connection's.
        c["conn"].add(o.get("conn") or "new")
        c["seeds"].add(o.get("seed"))
        if o.get("err"):
            c["n_err"] += 1
            continue
        if refused(o):
            c["n_refused"] += 1
            continue
        v = o.get(metric)
        if v is None:
            c["n_excluded"] += 1
            continue
        c["by"].setdefault((o.get("run"), o.get("rep")), []).append(float(v))
    out = {}
    for k, c in cells.items():
        out[k] = {"clusters": list(c["by"].values()), "n_err": c["n_err"], "n_refused": c["n_refused"],
                  "n_excluded": c["n_excluded"], "cfg": c["cfg"], "seeds": c["seeds"],
                  "conn": c["conn"]}
    return out

def refused(o):
    """True when a record's loads_before shows the box busier than its run's --max-live allowed
    (or the reading failed), so the request did not run on the box the method requires."""
    ml, lb = o.get("max_live"), o.get("loads_before")
    if ml is None or lb is None:
        return False
    if not isinstance(lb, dict) or "err" in lb or lb.get("run") is None:
        return True
    return lb["run"] > ml

def percentile(xs, q):
    """Linear-interpolated q-th percentile (0..100) of a non-empty list."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)

def bootstrap(a, b, higher_better=True, n_boot=5000, rng=None):
    """Median of arm b minus median of arm a, with a percentile bootstrap 95% CI.

    a, b: lists of clusters (lists of floats). Each resample draws len(a) clusters of a and
    len(b) clusters of b with replacement and takes the median of the pooled values.
    gain_pct is 100 * (med_b - med_a) / med_a, sign-flipped when lower is better, so a
    positive gain is always an improvement of b over a. CI fields are None when either arm
    has fewer than 2 clusters (no spread to resample).
    """
    if n_boot < MIN_BOOT:
        raise ValueError(f"--boot {n_boot} < {MIN_BOOT}: too few resamples for a 95% interval")
    rng = rng or random.Random(1)
    fa = [x for c in a for x in c]
    fb = [x for c in b for x in c]
    if not fa or not fb:
        raise ValueError("bootstrap needs at least one value in each arm")
    sign = 1.0 if higher_better else -1.0
    ma, mb = st.median(fa), st.median(fb)
    res = {"med_a": ma, "med_b": mb, "diff": mb - ma, "gain_pct": sign * 100.0 * (mb - ma) / ma if ma else None,
           "reps_a": len(a), "reps_b": len(b), "n_a": len(fa), "n_b": len(fb),
           "diff_lo": None, "diff_hi": None, "gain_lo": None, "gain_hi": None}
    if len(a) < 2 or len(b) < 2 or not ma:
        return res
    diffs, gains = [], []
    for _ in range(n_boot):
        ra = [x for _ in range(len(a)) for x in a[rng.randrange(len(a))]]
        rb = [x for _ in range(len(b)) for x in b[rng.randrange(len(b))]]
        da, db = st.median(ra), st.median(rb)
        diffs.append(db - da)
        gains.append(sign * 100.0 * (db - da) / da)
    res.update({"diff_lo": percentile(diffs, 2.5), "diff_hi": percentile(diffs, 97.5),
                "gain_lo": percentile(gains, 2.5), "gain_hi": percentile(gains, 97.5)})
    return res

def band_of(aa):
    """The A-A' noise band: |A' - A| as a gain %, the point estimate (median difference).

    The B-A interval already carries within-boot sampling noise; the band adds what a restart
    alone moved the medians, without stacking a second interval on the first.
    """
    if aa.get("gain_pct") is None:
        return None
    return abs(aa["gain_pct"])

def arm_names(n):
    """Role labels of n boots in boot order, alternating A and B from A: A, B, A2, B2, A3, B3 ...

    METHOD.md §2: A, B, A2, B2 is the default order; A, B, A2 the three-boot one; a longer chain
    (A, B, A2, B2, A3, B3 ...) carries on alternating. These labels are METHOD.md's boot names and
    its file names alike (A2 is the A' boot, B2 the B' boot).
    """
    return [("A" if i % 2 == 0 else "B") + ("" if i < 2 else str(i // 2 + 1)) for i in range(n)]

def score_chain(clusters, higher_better, n_boot, rng):
    """Every bootstrap a verdict over a chain of boots A, B, A2[, B2, A3 ...] needs.

    clusters: one list of clusters per boot, in boot order (A boots at even positions). Each B
    boot is tested against each ADJACENT A boot, the one before and then the one after it, if
    any ("tests", each tagged with its "b" and "a" labels); then each consecutive pair of A
    boots gives a restart band ("aas"). band_pct is the widest of those bands. "ab", "a2b" and
    "aa" name the first B's two tests and the first A pair, as in the three-boot order. The
    draw order (B-A, B-A2, then B2-A2 ..., then A-A2 ...) keeps a three-boot chain's numbers
    those of the three-boot compare before chains existed.
    """
    names = arm_names(len(clusters))
    tests = []
    for j in range(1, len(clusters), 2):
        for i in (j - 1, j + 1):
            if i < len(clusters):
                tests.append({"b": names[j], "a": names[i],
                              **bootstrap(clusters[i], clusters[j], higher_better, n_boot, rng)})
    aas = [{"a": names[i], "a2": names[i + 2], **bootstrap(clusters[i], clusters[i + 2], higher_better, n_boot, rng)}
           for i in range(0, len(clusters) - 2, 2)]
    bands = [band_of(x) for x in aas]
    return {"tests": tests, "aas": aas, "ab": tests[0], "a2b": tests[1], "aa": aas[0],
            "band_pct": None if any(b is None for b in bands) else max(bands)}

def compare_files(paths, metrics=None, gate=None, aa_only=False, n_boot=5000, seed=1, mode=None):
    """Rows, one per (cell, metric), comparing the boots in paths, in boot order.

    paths: A, B (intervals only, no verdict); or a chain A, B, A2[, B2[, A3 ...]] alternating
    A and B boots (arm_names), METHOD.md §2's default being A, B, A2, B2; or A, A2 with aa_only.
    metrics: None scores each cell's own metrics (cell_metrics); a list narrows that, and a
    metric a cell does not carry is never scored there.

    aa_only: paths are A, A2; each row reports A2-A and the noise band |A2-A|.
    With three or more paths each row carries every B boot vs each adjacent A boot (B-A, B-A2,
    B2-A2 ...: score_chain), the band (|A2-A|, the point estimate; the widest consecutive A pair
    in a longer chain) and a verdict:
      ADOPT  at least two B boots (A, B, A2, B2 or longer), and the CI lower bound of EVERY
             one of those B-vs-adjacent-A gains exceeds max(gate, band): B over A and over
             A2, B2 over A2 (and A3 ...), so both B boots clear the gate;
      NO     one does not (with one B boot too: a second can only add tests);
      HOLD   it cannot be judged: an arm under MIN_REPS reps, an errored or refused request
             in the cell, arms run with different settings, no interval, or an arm with no
             scorable value (an excluded-only cell, e.g. every coldfill background stream
             ended inside the fill window, HOLDs with its excluded count per arm); or the
             one B boot of A, B, A2 clears the bar, which a boot offset alone can do
             (METHOD.md §2, §5): it needs a second B boot.
    """
    if len(paths) < 2 or (aa_only and len(paths) != 2):
        raise ValueError("compare takes A.jsonl B.jsonl, a boot chain A.jsonl B.jsonl A2.jsonl [B2.jsonl ...], "
                         "or A.jsonl A2.jsonl --aa")
    if len(paths) >= 3 and gate is None:
        raise ValueError("compare with A2 needs --gate PCT (the item's gate: a minimum, never waived)")
    if len(paths) == 2 and gate is not None:
        raise ValueError("--gate needs three or more files: a verdict needs A, B and A2 (A re-booted) to know "
                         "the restart band; with two files compare only reports intervals")
    recs = [read_records(p) for p in paths]
    arm_labels = ("A", "A2") if aa_only else arm_names(len(paths))
    rows = []
    for metric in (metrics or tuple(METRICS)):
        hb = METRICS[metric]
        arms = [group_cells(r, metric, mode) for r in recs]
        for key in sorted(set().union(*arms), key=lambda k: (str(k[0]), _cell_order(k[1]))):
            if metric not in cell_metrics(*key):
                continue
            row = {"mode": key[0], "cell": key[1], "metric": metric}
            if any(key not in g for g in arms):
                row["missing_in"] = [p for p, g in zip(paths, arms) if key not in g]
                rows.append(row)
                continue
            cs = [g[key] for g in arms]
            row["n_err"] = [c["n_err"] for c in cs]
            row["n_refused"] = [c["n_refused"] for c in cs]
            row["n_excluded"] = [c["n_excluded"] for c in cs]
            notes = []
            if any(row["n_refused"]):
                notes.append("refused (started with #running-req above the run's --max-live, or loads "
                             "unreadable): " + ", ".join(f"{n} {k}" for n, k in zip(arm_labels, row["n_refused"])))
            if any(row["n_excluded"]):
                notes.append("excluded (no value: background stream ended inside the fill window, or no "
                             "delta): " + ", ".join(f"{n} {k}" for n, k in zip(arm_labels, row["n_excluded"])))
            if any(not c["clusters"] for c in cs):
                row["no_values_in"] = [p for p, c in zip(paths, cs) if not c["clusters"]]
                # Only cells that carry this metric reach here (cell_metrics), so an arm with no
                # value lost every request to an error, a refusal or an exclusion (all coldfill
                # background streams ended inside the fill window): never a silent skip.
                if len(paths) >= 3:
                    row["verdict"] = "HOLD"
                    notes.append("HOLD: no scorable values in " +
                                 ", ".join(n for n, c in zip(arm_labels, cs) if not c["clusters"]))
                if notes:
                    row["notes"] = notes
                rows.append(row)
                continue
            if len(set().union(*[c["cfg"] for c in cs])) > 1:
                notes.append("settings differ across arms: " + " vs ".join(sorted(set().union(*[c["cfg"] for c in cs]))))
            if len(set().union(*[c["conn"] for c in cs])) > 1:
                notes.append("connection modes differ across arms: " +
                             " vs ".join("/".join(sorted(c["conn"])) for c in cs))
            if key[0] == "think-ramp" and len(set().union(*[c["seeds"] for c in cs])) > 1:
                notes.append("think-ramp --seed differs across arms, so the prompt sets differ")
            rng = random.Random(seed)
            if aa_only:
                aa = bootstrap(cs[0]["clusters"], cs[1]["clusters"], hb, n_boot, rng)
                row.update({"aa": aa, "band_pct": band_of(aa),
                            "aa_excludes_0": aa["gain_lo"] is not None and (aa["gain_lo"] > 0 or aa["gain_hi"] < 0)})
            elif len(paths) == 2:
                row["ab"] = bootstrap(cs[0]["clusters"], cs[1]["clusters"], hb, n_boot, rng)
            else:
                row.update(score_chain([c["clusters"] for c in cs], hb, n_boot, rng))
                row["gate_pct"] = gate
                row["verdict"], why = _verdict(row, cs, gate, notes)
                notes.extend(why)
            low = [n for n, c in zip(arm_labels, cs) if len(c["clusters"]) < MIN_REPS]
            if low:
                notes.append(f"under {MIN_REPS} reps in {', '.join(low)}")
            if notes:
                row["notes"] = notes
            rows.append(row)
    return rows

def _cell_order(cell):
    """Sort C=2 before C=16 and gap=5m before gap=30m; other cells by name."""
    try:
        return (0, float(str(cell).split("=", 1)[1].rstrip("m")), "")
    except (IndexError, ValueError):
        return (1, 0.0, str(cell))

def _verdict(row, cs, gate, notes, min_b_boots=2):
    """(verdict, reasons) for a scored chain row (compare_files' docstring).

    min_b_boots: B boots an ADOPT needs (METHOD.md §2: two). sim_block alone passes 1, to
    measure the bare test of a three-boot chain (§5's table: how often one B boot clears it).
    """
    why = []
    if any(len(c["clusters"]) < MIN_REPS for c in cs):
        why.append("HOLD: fewer than %d reps in an arm" % MIN_REPS)
    if any(c["n_err"] for c in cs):
        why.append("HOLD: errored requests in this cell (scored arms are not like for like)")
    if any(c["n_refused"] for c in cs):
        why.append("HOLD: refused requests in this cell (the box was not idle; re-run the arm)")
    if any(n.startswith("settings differ") for n in notes):
        why.append("HOLD: arms ran with different settings")
    if any(n.startswith("connection modes differ") for n in notes):
        why.append("HOLD: arms ran with different connection modes (\"conn\"; a record without one ran "
                   "`Connection: close`, i.e. \"new\")")
    tests = row["tests"]
    if any(t["gain_lo"] is None for t in tests) or row["band_pct"] is None:
        why.append("HOLD: no interval (an arm has fewer than 2 reps)")
    if why:
        return "HOLD", why
    # Every B boot against every adjacent A boot: the weakest of those tests decides.
    worst = min(tests, key=lambda t: t["gain_lo"])
    lo = worst["gain_lo"]
    bar = max(gate, row["band_pct"])
    band = ("|A2-A| band" if len(row["aas"]) == 1 else "widest A-boot band") + f" {row['band_pct']:.2f}%"
    test = f"gain lower bound {lo:+.2f}% ({worst['b']}-{worst['a']})"
    if lo > bar:
        n_b = len({t["b"] for t in tests})
        if n_b < min_b_boots:
            return "HOLD", [f"HOLD: needs a second B boot: {test} > max(gate {gate:g}%, {band}) on one B boot, "
                            "and a boot offset alone can carry one B boot past the bar (METHOD.md §2, §5): "
                            "boot B2 and compare A B A2 B2"]
        return "ADOPT", [f"{test} > max(gate {gate:g}%, {band})"]
    return "NO", [f"{test} <= max(gate {gate:g}%, {band})"]

def _fmt_ci(r):
    if r.get("gain_pct") is None:
        return "n/a"
    if r["gain_lo"] is None:
        return f"{r['gain_pct']:+6.2f}% [no CI]"
    return f"{r['gain_pct']:+6.2f}% [{r['gain_lo']:+.2f}, {r['gain_hi']:+.2f}]"

def format_rows(rows, paths, aa_only):
    names = ["A", "A2"] if aa_only else arm_names(len(paths))
    out = ["# compare " + "  ".join(f"{n}={p}" for n, p in zip(names, paths)),
           "# gain: + is better (dec tok/s up, ttft down); [..] = bootstrap 95% CI; band = |A2-A|"
           + (" (the widest consecutive A pair)" if len(paths) > 4 else "") + "; reps/values/errors per arm"]
    per_cell = {(r["mode"], r["cell"]): r["n_refused"] for r in rows if "n_refused" in r}  # same for every metric
    tot = [sum(v[j] for v in per_cell.values()) for j in range(len(names))]
    out.append("# refused records (started with #running-req above the run's --max-live, or loads unreadable): "
               + ", ".join(f"{n} {k}" for n, k in zip(names, tot)))
    for r in rows:
        head = f"{r['mode']:<10} {r['cell']:<10} {r['metric']:<4}"
        if "missing_in" in r or "no_values_in" in r:
            out.append(f"{head}  {r.get('verdict', 'skipped')}: no {'cell' if 'missing_in' in r else 'values'} in "
                       f"{', '.join(r.get('missing_in') or r.get('no_values_in'))}")
            for n in r.get("notes", []):
                out.append(f"{'':<27}  - {n}")
            continue
        if "tests" in r:  # a chain: each boot's median from any test it is in
            by = {}
            for t in r["tests"]:
                by[t["a"]] = (t["med_a"], t["reps_a"], t["n_a"])
                by[t["b"]] = (t["med_b"], t["reps_b"], t["n_b"])
            arms = [by[n] for n in names]
        else:
            first = r["aa"] if aa_only else r["ab"]
            arms = [(first["med_a"], first["reps_a"], first["n_a"]), (first["med_b"], first["reps_b"], first["n_b"])]
        meds = "  ".join(f"{n} {m:8.3f} ({k}/{v}/{e})" for n, (m, k, v), e in zip(names, arms, r["n_err"]))
        if aa_only:
            tail = f"A2-A {_fmt_ci(r['aa'])}  band {r['band_pct']:.2f}%" if r["band_pct"] is not None \
                else f"A2-A {_fmt_ci(r['aa'])}"
            if r.get("aa_excludes_0"):
                tail += "  (CI excludes 0: restart-to-restart drift)"
        elif "tests" in r:
            band = f"{r['band_pct']:.2f}%" if r["band_pct"] is not None else "n/a"
            tail = "  ".join(f"{t['b']}-{t['a']} {_fmt_ci(t)}" for t in r["tests"]) + f"  band {band}  {r['verdict']}"
        else:
            tail = f"B-A {_fmt_ci(r['ab'])}"
        out.append(f"{head}  {meds}  {tail}")
        for n in r.get("notes", []):
            out.append(f"{'':<27}  - {n}")
    return "\n".join(out)

def cmd_compare(args):
    if len(args.files) < 2:
        sys.exit("bench.py compare: give A.jsonl B.jsonl [A2.jsonl [B2.jsonl ...]] in boot order, "
                 "or A.jsonl A2.jsonl --aa")
    metrics = [m.strip() for m in args.metric.split(",") if m.strip()] if args.metric else None
    bad = [m for m in metrics or () if m not in METRICS]
    if bad:
        sys.exit(f"bench.py compare: unknown --metric {', '.join(bad)} (have {', '.join(METRICS)})")
    try:
        rows = compare_files(args.files, metrics, args.gate, args.aa, args.boot, args.boot_seed, args.only_mode)
    except (ValueError, OSError) as e:
        sys.exit(f"bench.py compare: {e}")
    # A --metric no cell carries (dec on resume, say) scores nothing: say so, never a silent skip.
    unmatched = unmatched_metrics(rows, metrics)
    if unmatched:
        print(f"bench.py compare: no cell carries --metric {', '.join(unmatched)}"
              + (f" in mode {args.only_mode}" if args.only_mode else "")
              + " (METHOD.md §4: think-ramp dec+ttft, resume ttft, coldfill fill ttft, coldfill bg-during dec)",
              file=sys.stderr, flush=True)
    if not rows:
        sys.exit("bench.py compare: nothing scored: no cell in these files carries the metrics asked for"
                 + (f" in mode {args.only_mode}" if args.only_mode else ""))
    print("\n".join(json.dumps(r) for r in rows) if args.json else format_rows(rows, args.files, args.aa))

def unmatched_metrics(rows, metrics):
    """The metrics named with --metric that no row scores (no cell carries them); [] without --metric."""
    have = {r["metric"] for r in rows}
    return [m for m in metrics or () if m not in have]

def synth_arm(center, rel_sd, reps, per_rep, rng, rep_sd=0.0, boot_sd=0.0):
    """One arm (one boot): reps clusters of per_rep values ~ N(c*(1+rep shift), rel_sd*c).

    c = center * boot, boot ~ N(1, boot_sd) drawn ONCE per arm and shared by all its reps: the
    restart-to-restart offset no within-boot rep can see (METHOD.md's reason for A, B, A', B').
    rep shift ~ N(0, rep_sd) per rep. boot_sd=0 draws no offset, so the RNG stream (and every
    result pinned on it) is the same as without the parameter.
    """
    if boot_sd:
        center *= rng.gauss(1, boot_sd)
    out = []
    for _ in range(reps):
        shift = rng.gauss(0, rep_sd)
        out.append([rng.gauss(center * (1 + shift), rel_sd * center) for _ in range(per_rep)])
    return out

# METHOD.md §5's table: (per-request sd, per-rep sd, per-boot sd), C, true gain; each cell run
# with each order in SIM_BOOTS (3 = A, B, A2; 4 = A, B, A2, B2) at each of SIM_REPS reps.
# No A-A' pilot has been run yet, so the boot sd is an assumption: 4% (the restart offset the
# boot-noise self-test models) for every spread, and 5% for one cell to show how fast it bites.
# Replace it with the pilots' measured |A2-A| once there are some.
SIM_CELLS = [((0.05, 0.03, 0.04), 1, 0.10), ((0.05, 0.03, 0.04), 1, 0.20),
             ((0.05, 0.03, 0.04), 4, 0.10), ((0.05, 0.03, 0.04), 4, 0.20),
             ((0.08, 0.05, 0.04), 1, 0.20), ((0.08, 0.05, 0.04), 4, 0.20),
             ((0.05, 0.03, 0.05), 4, 0.20),
             ((0.05, 0.03, 0.04), 1, 0.0), ((0.05, 0.03, 0.04), 4, 0.0),
             ((0.08, 0.05, 0.04), 4, 0.0), ((0.05, 0.03, 0.05), 4, 0.0)]
SIM_BOOTS = (3, 4)
SIM_REPS = (6, 12, 20)
SIM_GATE = 5.0

def sim_block(spread, C, gain, reps, trials, n_boot, seed, boots=3):
    """Share of `trials` synthetic boot chains that compare's bootstrap and verdict ADOPT.

    boots arms in boot order A, B, A2[, B2 ...] (arm_names), scored by score_chain and _verdict
    as compare does. A boots are drawn at 500 tok/s, B boots at 500*(1+gain); every value of an
    arm is one decode rate (higher is better), C values per rep. spread is (per-request sd,
    per-rep sd) or, with a per-arm boot offset, (per-request sd, per-rep sd, boot sd)
    (synth_arm). Seeded from the block's own parameters, so a block's result does not depend on
    which process runs it or in what order; a 2-tuple spread keeps its pre-boot-offset seed and
    draws, and boots=3 the seed it had before B2 existed. With boots=3 the share is that chain's
    bare test (B over A and A2 clearing the bar), which compare itself HOLDs for a second B boot.
    """
    rel_sd, rep_sd, boot_sd = (tuple(spread) + (0.0,))[:3]
    key = f"{seed}|{rel_sd}|{rep_sd}|{C}|{gain}|{reps}" if not boot_sd \
        else f"{seed}|{rel_sd}|{rep_sd}|{boot_sd}|{C}|{gain}|{reps}"
    rng = random.Random(key if boots == 3 else f"{key}|boots={boots}")
    adopted = 0
    for _ in range(trials):
        arms = [synth_arm(500.0 * (1 + gain * (i % 2)), rel_sd, reps, C, rng, rep_sd, boot_sd)
                for i in range(boots)]
        row = score_chain(arms, True, n_boot, rng)
        cs = [{"clusters": x, "n_err": 0, "n_refused": 0} for x in arms]
        adopted += _verdict(row, cs, SIM_GATE, [], min_b_boots=1)[0] == "ADOPT"
    return adopted / trials

def cmd_simulate(trials, null_trials, n_boot, seed):
    """Print METHOD.md §5's adoption-rate table, re-run (one process per block, all cores)."""
    import concurrent.futures as cf
    if n_boot < MIN_BOOT:
        sys.exit(f"bench.py selftest --simulate: --boot {n_boot} < {MIN_BOOT}")
    jobs = [(cell, boots, reps) for cell in SIM_CELLS for boots in SIM_BOOTS for reps in SIM_REPS]
    t0 = time.time()
    argv = [(sp, C, g, reps, null_trials if g == 0 else trials, n_boot, seed, boots)
            for (sp, C, g), boots, reps in jobs]
    try:
        with cf.ProcessPoolExecutor() as ex:
            share = dict(zip(jobs, ex.map(sim_block, *zip(*argv))))
    except (OSError, NotImplementedError) as e:  # no process pool here (a sandbox without semaphores)
        print(f"# no process pool ({e}); running the blocks in this process", file=sys.stderr, flush=True)
        share = {job: sim_block(*a) for job, a in zip(jobs, argv)}
    print(f"# bench.py selftest --simulate --trials {trials} --null-trials {null_trials} --boot {n_boot} "
          f"--boot-seed {seed}: share of boot chains adopted at a {SIM_GATE:g}% gate "
          f"({time.time() - t0:.0f}s)")
    print("\n| spread | C | true gain | boots | trials | " + " | ".join(f"{r} reps" for r in SIM_REPS) + " |")
    print("| --- | -: | -: | --- | -: | " + " | ".join("-:" for _ in SIM_REPS) + " |")
    for cell in SIM_CELLS:
        (rel_sd, rep_sd, boot_sd), C, g = cell
        for boots in SIM_BOOTS:
            print(f"| {rel_sd * 100:g}% / {rep_sd * 100:g}% / {boot_sd * 100:g}% | {C} | {g * 100:+.0f}% | "
                  f"{', '.join(arm_names(boots))} | {null_trials if g == 0 else trials} | "
                  + " | ".join(f"{share[(cell, boots, r)]:.2f}" for r in SIM_REPS) + " |")

def cmd_selftest():
    """CPU unit test of percentile / bootstrap / compare on synthetic data."""
    import contextlib
    import io
    import shutil
    import tempfile
    import types
    import unittest

    arm = synth_arm

    def write(dirname, name, cells, seed=7, err=0, arm_label=None, max_tokens=300):
        path = os.path.join(dirname, name)
        with open(path, "w") as f:
            f.write("# think-ramp start (header line, skipped)\n")
            f.write(json.dumps({"rep": 1, "C": 1, "dec_med": 1.0}) + "\n")  # a row line, skipped
            for cell, (dec, ttft) in cells.items():
                for rep, (dc, tc) in enumerate(zip(dec, ttft), 1):
                    for i, (d, t) in enumerate(zip(dc, tc)):
                        f.write(json.dumps({"kind": "req", "run": name, "arm": arm_label, "mode": "think-ramp",
                                            "cell": cell, "rep": rep, "i": i, "seed": seed, "dec": d, "ttft": t,
                                            "max_tokens": max_tokens}) + "\n")
                for i in range(err):
                    f.write(json.dumps({"kind": "req", "run": name, "mode": "think-ramp", "cell": cell, "rep": 1,
                                        "i": 100 + i, "seed": seed, "err": "HTTP Error 502",
                                        "max_tokens": max_tokens}) + "\n")
        return path

    class T(unittest.TestCase):
        def setUp(self):
            self.rng = random.Random(20261026)
            self.dir = tempfile.mkdtemp(prefix="bench-selftest-")

        def tearDown(self):
            shutil.rmtree(self.dir, ignore_errors=True)

        def cells(self, dec_center, ttft_center, reps=6, rel_sd=0.05, rep_sd=0.02):
            return {f"C={c}": (arm(dec_center / c ** 0.3, rel_sd, reps, c, self.rng, rep_sd),
                               arm(ttft_center, rel_sd, reps, c, self.rng, rep_sd)) for c in (1, 4)}

        def test_percentile(self):
            self.assertEqual(percentile([3, 1, 2], 50), 2)
            self.assertAlmostEqual(percentile([0, 10], 25), 2.5)
            self.assertEqual(percentile([5], 97.5), 5)
            self.assertAlmostEqual(percentile(list(range(101)), 2.5), 2.5)

        def test_refuses_few_resamples(self):
            with self.assertRaises(ValueError):
                bootstrap([[1.0], [2.0]], [[1.0], [2.0]], n_boot=1999)

        def test_point_estimate_and_sign(self):
            a, b = [[100.0, 102.0], [98.0]], [[110.0], [111.0, 109.0]]
            r = bootstrap(a, b, True, 2000)
            self.assertEqual((r["med_a"], r["med_b"]), (100.0, 110.0))
            self.assertAlmostEqual(r["gain_pct"], 10.0)
            self.assertLessEqual(r["gain_lo"], r["gain_pct"])
            self.assertGreaterEqual(r["gain_hi"], r["gain_pct"])
            # lower is better (ttft): b slower is a negative gain
            self.assertAlmostEqual(bootstrap(a, b, False, 2000)["gain_pct"], -10.0)

        def test_single_cluster_has_no_interval(self):
            r = bootstrap([[1.0, 2.0, 3.0]], [[1.0], [2.0]], True, 2000)
            self.assertIsNone(r["gain_lo"])

        def test_deterministic_for_a_seed(self):
            a, b = arm(500, 0.08, 6, 4, self.rng), arm(520, 0.08, 6, 4, self.rng)
            r1 = bootstrap(a, b, True, 2000, random.Random(3))
            r2 = bootstrap(a, b, True, 2000, random.Random(3))
            self.assertEqual((r1["gain_lo"], r1["gain_hi"]), (r2["gain_lo"], r2["gain_hi"]))

        def test_aa_band_covers_zero(self):
            # Same distribution twice, reps differing ~10% as the 2026-10-26 baseline's C=1 did.
            pa = write(self.dir, "a.jsonl", self.cells(560, 0.36, rel_sd=0.03, rep_sd=0.05))
            pa2 = write(self.dir, "a2.jsonl", self.cells(560, 0.36, rel_sd=0.03, rep_sd=0.05))
            rows = compare_files([pa, pa2], ("dec", "ttft"), aa_only=True, n_boot=2000)
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertLessEqual(r["aa"]["gain_lo"], 0.0, r)
                self.assertGreaterEqual(r["aa"]["gain_hi"], 0.0, r)
                self.assertFalse(r["aa_excludes_0"])
                self.assertAlmostEqual(r["band_pct"], abs(r["aa"]["gain_pct"]))  # band = |A2-A| point
                self.assertNotIn("notes", r)

        def c1(self, dec_center, ttft_center, reps):
            """C=1 only: one value per rep, ~6% total spread (5% per request, 3% per rep)."""
            return {"C=1": (arm(dec_center, 0.05, reps, 1, self.rng, 0.03),
                             arm(ttft_center, 0.05, reps, 1, self.rng, 0.03))}

        def test_c1_true_20pct_win_adopts_at_12_reps(self):
            pa = write(self.dir, "a.jsonl", self.c1(500, 0.40, 12))
            pb = write(self.dir, "b.jsonl", self.c1(600, 0.32, 12))  # +20% decode, -20% ttft
            pa2 = write(self.dir, "a2.jsonl", self.c1(500, 0.40, 12))
            pb2 = write(self.dir, "b2.jsonl", self.c1(600, 0.32, 12))
            rows = compare_files([pa, pb, pa2, pb2], ("dec", "ttft"), gate=5, n_boot=2000)
            self.assertEqual([r["verdict"] for r in rows], ["ADOPT", "ADOPT"])
            for r in rows:
                self.assertAlmostEqual(r["band_pct"], abs(r["aa"]["gain_pct"]))
                self.assertGreater(min(r["ab"]["gain_lo"], r["a2b"]["gain_lo"]), max(5, r["band_pct"]))

        def test_c1_zero_effect_is_not_adopted_at_12_reps(self):
            pa = write(self.dir, "a.jsonl", self.c1(500, 0.40, 12))
            pb = write(self.dir, "b.jsonl", self.c1(500, 0.40, 12))
            pa2 = write(self.dir, "a2.jsonl", self.c1(500, 0.40, 12))
            rows = compare_files([pa, pb, pa2], ("dec", "ttft"), gate=5, n_boot=2000)
            self.assertEqual([r["verdict"] for r in rows], ["NO", "NO"])
            for r in rows:
                self.assertLessEqual(r["ab"]["gain_lo"], 0.0)  # the interval covers no gain

        def test_b_must_beat_a2_too(self):
            # A tight box (1% per request, 0.5% per rep) where A2 booted 4% faster than A: the band
            # (~4%) stays under the 5% gate, so the bar is the gate alone, and B's +8% over A clears
            # it. Only the clause "B must also beat A2 on the same test" (~+4% over A2) says NO.
            tight = lambda c: {"C=1": (arm(c, 0.01, 12, 1, self.rng, 0.005),
                                       arm(0.40, 0.01, 12, 1, self.rng, 0.005))}
            pa = write(self.dir, "a.jsonl", tight(500))
            pb = write(self.dir, "b.jsonl", tight(540))
            pa2 = write(self.dir, "a2.jsonl", tight(520))
            r = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)[0]
            self.assertLess(r["band_pct"], 5)  # the band does not raise the bar
            self.assertGreater(r["ab"]["gain_lo"], 5)  # B vs A alone would adopt
            self.assertLess(r["a2b"]["gain_lo"], 5)  # the A2 test is what fails
            self.assertEqual(r["verdict"], "NO")

        def test_noise_sized_gain_is_not_adopted(self):
            # +4% true gain under 10% rep-to-rep noise: point estimate may clear the gate, the
            # interval must not.
            pa = write(self.dir, "a.jsonl", self.cells(500, 0.4, rep_sd=0.10))
            pb = write(self.dir, "b.jsonl", self.cells(520, 0.4, rep_sd=0.10))
            pa2 = write(self.dir, "a2.jsonl", self.cells(500, 0.4, rep_sd=0.10))
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            self.assertEqual([r["verdict"] for r in rows], ["NO", "NO"])

        def test_band_beats_gate(self):
            # B clears a 5% gate but A and A2 differ by more than B's lead: not adopted.
            pa = write(self.dir, "a.jsonl", self.cells(500, 0.4, rel_sd=0.01, rep_sd=0.005))
            pb = write(self.dir, "b.jsonl", self.cells(560, 0.4, rel_sd=0.01, rep_sd=0.005))
            pa2 = write(self.dir, "a2.jsonl", self.cells(440, 0.4, rel_sd=0.01, rep_sd=0.005))
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            for r in rows:
                self.assertGreater(r["ab"]["gain_lo"], 5)
                self.assertGreater(r["band_pct"], r["ab"]["gain_lo"])
                self.assertEqual(r["verdict"], "NO")

        def test_holds(self):
            pa = write(self.dir, "a.jsonl", self.cells(500, 0.4, reps=3))
            pb = write(self.dir, "b.jsonl", self.cells(700, 0.3, reps=3))
            pa2 = write(self.dir, "a2.jsonl", self.cells(500, 0.4, reps=3))
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            self.assertTrue(all(r["verdict"] == "HOLD" for r in rows))
            self.assertTrue(all(any("under 6 reps" in n for n in r["notes"]) for r in rows))
            pb = write(self.dir, "b-err.jsonl", self.cells(700, 0.3), err=1)
            pa = write(self.dir, "a6.jsonl", self.cells(500, 0.4))
            pa2 = write(self.dir, "a26.jsonl", self.cells(500, 0.4))
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            self.assertTrue(all(r["verdict"] == "HOLD" and r["n_err"] == [0, 1, 0] for r in rows))
            pb = write(self.dir, "b-mt.jsonl", self.cells(700, 0.3), max_tokens=600)
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            self.assertTrue(all(r["verdict"] == "HOLD" for r in rows))
            # a record that started with the box busier than its run's --max-live: refused, HOLD
            pb = os.path.join(self.dir, "b-busy.jsonl")
            with open(write(self.dir, "b-ok.jsonl", self.cells(700, 0.3))) as f, open(pb, "w") as g:
                for n, line in enumerate(f):
                    if line.startswith('{"kind"'):
                        o = json.loads(line)
                        o.update({"max_live": 0, "loads_before": {"run": 3 if n % 5 == 0 else 0, "wait": 0, "kv": 0.1}})
                        line = json.dumps(o) + "\n"
                    g.write(line)
            rows = compare_files([pa, pb, pa2], ("dec",), gate=5, n_boot=2000)
            self.assertTrue(all(r["verdict"] == "HOLD" and r["n_refused"][1] > 0 and r["n_refused"][0] == 0
                                for r in rows))
            self.assertTrue(all(any(n.startswith("refused") for n in r["notes"]) for r in rows))
            txt = format_rows(rows, [pa, pb, pa2], False)
            self.assertIn("# refused records", txt)
            self.assertIn(f"B {sum(r['n_refused'][1] for r in rows)}", txt)

        def test_refused(self):
            self.assertFalse(refused({"max_live": 0, "loads_before": {"run": 0}}))
            self.assertTrue(refused({"max_live": 0, "loads_before": {"run": 1}}))
            self.assertTrue(refused({"max_live": 4, "loads_before": {"err": "timeout"}}))
            self.assertFalse(refused({"loads_before": {"run": 9}}))  # no max_live recorded: not checked
            self.assertFalse(refused({"max_live": 0}))  # no reading for this record

        def test_errors_are_counted_not_scored(self):
            recs = [{"kind": "req", "mode": "m", "cell": "c", "run": "r", "rep": 1, "dec": 10.0},
                    {"kind": "req", "mode": "m", "cell": "c", "run": "r", "rep": 1, "err": "timeout"},
                    {"kind": "req", "mode": "m", "cell": "c", "run": "r", "rep": 2, "dec": 12.0},
                    {"kind": "req", "mode": "m", "cell": "c", "run": "r2", "rep": 1, "dec": None}]
            g = group_cells(recs, "dec")[("m", "c")]
            self.assertEqual(sorted(g["clusters"]), [[10.0], [12.0]])
            self.assertEqual((g["n_err"], g["n_excluded"], g["n_refused"]), (1, 1, 0))

        def test_missing_cell_and_bad_input(self):
            pa = write(self.dir, "a.jsonl", self.cells(500, 0.4))
            only1 = {"C=1": self.cells(500, 0.4)["C=1"]}
            pb = write(self.dir, "b.jsonl", only1)
            rows = compare_files([pa, pb], ("dec",), n_boot=2000)
            self.assertEqual([r.get("missing_in") for r in rows], [None, [pb]])
            rows_txt = os.path.join(self.dir, "rows.jsonl")
            with open(rows_txt, "w") as f:
                f.write(json.dumps({"rep": 1, "C": 1, "dec_med": 500}) + "\n")
            with self.assertRaises(ValueError):
                compare_files([pa, rows_txt], ("dec",), n_boot=2000)
            with self.assertRaises(ValueError):
                compare_files([pa, pb, pa], ("dec",), n_boot=2000)  # no --gate
            with self.assertRaises(ValueError):
                compare_files([pa, pb], ("dec",), gate=5, n_boot=2000)  # a verdict needs A, B, A2
            self.assertIn("B-A", format_rows(rows, [pa, pb], False))

        # ---- producers: each mode's record() path, with stubbed engine responses ----------

        def run_mode(self, mode, arm_label, speed, runs, busy=0, fname=None):
            """Run cmd_<mode> `runs` times against a stub engine `speed`x as fast as A; return
            the records file and the rows each run logged."""
            g = globals()
            path = os.path.join(self.dir, fname or f"{mode}-{arm_label}.req.jsonl")
            rows = []

            def stub_stream(base, key, body, timeout=900):
                content = body["messages"][-1]["content"]
                r = random.Random(f"{arm_label}|{content}|{len(body['messages'])}")
                if body.get("reasoning_effort") == "none":  # resume turn / seed, coldfill fill
                    t0 = 1002.0 if "READY and nothing else" in content else 1000.0
                    ttft = (3.0 if t0 == 1002.0 else 0.3) / speed * r.gauss(1, 0.03)
                    return {"t0": t0, "t_first": t0 + ttft, "t_end": t0 + ttft + 0.05, "ttft": ttft,
                            "wall": ttft + 0.05, "ct": 4, "pt": 1200, "cached": 0, "finish": "stop",
                            "chunks": [(t0 + ttft, 8)], "content": "READY", "dec": 80.0}
                dec = 500.0 * speed * r.gauss(1, 0.04)
                # coldfill: background stream 0 stops early, inside the fill window
                ct = 500 if (mode == "coldfill" and content == first_bg) else body["max_tokens"]
                t0, ttft = 1000.0, 0.5
                step = 10  # tokens per chunk, 2 chars per token
                chunks = [(t0 + ttft + k / dec, 2 * step) for k in range(0, ct, step)]
                t_end = t0 + ttft + ct / dec
                return {"t0": t0, "t_first": t0 + ttft, "t_end": t_end, "ttft": ttft, "wall": t_end - t0,
                        "ct": ct, "pt": 90, "cached": 0, "finish": "length" if ct == body["max_tokens"] else "stop",
                        "chunks": chunks, "content": "", "dec": ct / (t_end - t0 - ttft)}

            quiet = {"run": 0, "wait": 0, "kv": 0.0}
            stubs = {"stream": stub_stream,
                     "wait_quiet": lambda max_running, log, tries=5, backoff=2.0: dict(quiet),
                     "loads": lambda: {"run": busy, "wait": 0, "kv": 0.0},
                     "load_tails_probe": lambda: types.SimpleNamespace(
                         SEED_OFFSET=0, build_prefix=lambda i, n: f"unique prefix {i} " * 8)}
            # cmd_coldfill draws its background prompts from random.Random(args.seed), in order
            first_bg = think_prompt(random.Random(4242)) + " Be extremely thorough; explore several approaches in full."
            saved = {k: g[k] for k in stubs}
            g.update(stubs)
            try:
                with open(path, "a") as fh:
                    for run in range(runs):
                        args = types.SimpleNamespace(
                            seed=4242, seed_offset=1000 * run + len(arm_label), base_url="http://stub",
                            run_id=f"{mode}-{arm_label}-{run}", arm=arm_label, max_live=0, gate_tries=1,
                            reps=6, levels="1,4", max_tokens=300, pause=0, gaps_min="0", ctx_tokens=1000,
                            bg=4, bg_tokens=4000, lead_s=0, fill_tokens=1000,
                            rec=lambda s: fh.write(s + "\n"))
                        fn = {"think-ramp": cmd_think_ramp, "resume": cmd_resume, "coldfill": cmd_coldfill}[mode]
                        fn(args, "stub-key", lambda s: rows.append(json.loads(s)) if s.startswith("{") else None)
            finally:
                g.update(saved)
            return path, rows

        def test_think_ramp_records_end_to_end(self):
            pa, rows = self.run_mode("think-ramp", "A", 1.0, 1)
            pb, _ = self.run_mode("think-ramp", "B", 1.3, 1)
            pa2, _ = self.run_mode("think-ramp", "A2", 1.0, 1)
            pb2, _ = self.run_mode("think-ramp", "B2", 1.3, 1)
            # the row output is unchanged by records
            self.assertEqual(set(rows[0]), {"rep", "C", "t", "n_ok", "errs", "first_err", "dec_med", "dec_min",
                                            "dec_max", "agg_tok_s", "ttft_med", "ttft_max", "ct_med", "finish",
                                            "loads_before", "conn"})
            self.assertEqual(len(rows), 12)
            recs = read_records(pa)
            self.assertEqual(len(recs), 6 * (1 + 4))
            o = recs[0]
            for k in ("run", "arm", "mode", "cell", "rep", "i", "seed", "dec", "ttft", "ct", "finish",
                      "max_live", "loads_before", "max_tokens"):
                self.assertIn(k, o)
            self.assertEqual((o["mode"], o["cell"], o["arm"], o["max_live"]), ("think-ramp", "C=1", "A", 0))
            rows_c = compare_files([pa, pb, pa2, pb2], ("dec",), gate=5, n_boot=2000)
            self.assertEqual([(r["cell"], r["verdict"]) for r in rows_c], [("C=1", "ADOPT"), ("C=4", "ADOPT")])
            self.assertEqual([r["ab"]["reps_a"] for r in rows_c], [6, 6])
            self.assertEqual([r["ab"]["n_a"] for r in rows_c], [6, 24])

        def test_resume_records_end_to_end_and_refusal(self):
            pa, rows = self.run_mode("resume", "A", 1.0, 6)
            pb, _ = self.run_mode("resume", "B", 1.5, 6)
            pa2, _ = self.run_mode("resume", "A2", 1.0, 6)
            pb2, _ = self.run_mode("resume", "B2", 1.5, 6)
            self.assertTrue(all("gap_min" in r for r in rows))
            r = compare_files([pa, pb, pa2, pb2], ("ttft",), gate=5, n_boot=2000)[0]
            self.assertEqual((r["mode"], r["cell"], r["verdict"], r["ab"]["reps_a"]), ("resume", "gap=0m", "ADOPT", 6))
            # the resume turn's loads reading showed 2 running with --max-live 0: refused, HOLD
            pbusy, _ = self.run_mode("resume", "B", 1.5, 6, busy=2, fname="resume-B-busy.req.jsonl")
            r = compare_files([pa, pbusy, pa2], ("ttft",), gate=5, n_boot=2000)[0]
            self.assertEqual((r["verdict"], r["n_refused"]), ("HOLD", [0, 6, 0]))

        def test_coldfill_records_end_to_end(self):
            pa, rows = self.run_mode("coldfill", "A", 1.0, 6)
            pb, _ = self.run_mode("coldfill", "B", 1.3, 6)
            pa2, _ = self.run_mode("coldfill", "A2", 1.0, 6)
            pb2, _ = self.run_mode("coldfill", "B2", 1.3, 6)
            self.assertEqual(len(rows), 6)
            self.assertEqual([b["ended_before_fill_end"] for b in rows[0]["bg"]], [True, False, False, False])
            recs = [o for o in read_records(pa) if o["cell"] == "bg-during"]
            self.assertEqual(sum(o["ended_before_fill_end"] for o in recs), 6)
            self.assertTrue(all(o["dec"] is None for o in recs if o["ended_before_fill_end"]))
            got = {(r["cell"], r["metric"]): r for r in compare_files([pa, pb, pa2, pb2], gate=5, n_boot=2000)}
            # each cell scores its own metric only: bg-during dec, fill ttft (16 output tokens: no rate)
            self.assertEqual(sorted(got), [("bg-during", "dec"), ("fill", "ttft")])
            bg = got[("bg-during", "dec")]
            self.assertEqual((bg["verdict"], bg["n_excluded"], bg["ab"]["n_a"]), ("ADOPT", [6, 6, 6, 6], 18))
            self.assertTrue(any(n.startswith("excluded") and "A 6, B 6, A2 6, B2 6" in n for n in bg["notes"]))
            self.assertEqual(got[("fill", "ttft")]["verdict"], "ADOPT")
            # --metric narrows, never widens: ttft on bg-during and dec on fill stay unscored
            got = {(r["cell"], r["metric"]) for r in compare_files([pa, pb, pa2], ("dec", "ttft"), gate=5,
                                                                     n_boot=2000)}
            self.assertEqual(sorted(got), [("bg-during", "dec"), ("fill", "ttft")])

        def test_excluded_only_cell_holds(self):
            # Every arm's background streams all ended inside the fill window: no dec anywhere,
            # nothing errored or refused. The cell HOLDs with its excluded count, never a skip.
            paths = []
            for name in ("A", "B", "A2"):
                p = os.path.join(self.dir, f"coldfill-{name}.req.jsonl")
                with open(p, "w") as f:
                    for run in range(6):
                        for i in range(4):
                            f.write(json.dumps({"kind": "req", "run": f"{name}-{run}", "arm": name,
                                                "mode": "coldfill", "cell": "bg-during", "rep": 1, "i": i,
                                                "dec": None, "ttft": None, "bg_tokens": 4000, "max_live": 0,
                                                "loads_before": {"run": 0}, "ended_before_fill_end": True})
                                    + "\n")
                paths.append(p)
            rows = compare_files(paths, gate=5, n_boot=2000)
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual((r["cell"], r["metric"], r["verdict"], r["n_excluded"], r["n_err"], r["n_refused"]),
                             ("bg-during", "dec", "HOLD", [24, 24, 24], [0, 0, 0], [0, 0, 0]))
            self.assertTrue(any(n.startswith("excluded") and "A 24, B 24, A2 24" in n for n in r["notes"]))
            self.assertTrue(any(n.startswith("HOLD: no scorable values in A, B, A2") for n in r["notes"]))
            txt = format_rows(rows, paths, False)
            self.assertIn("HOLD: no values in", txt)
            self.assertIn("A 24, B 24, A2 24", txt)
            # two files (intervals only): no verdict, but the excluded count is still shown
            r = compare_files(paths[:2], n_boot=2000)[0]
            self.assertNotIn("verdict", r)
            self.assertTrue(any(n.startswith("excluded") for n in r["notes"]))

        def test_cell_metrics(self):
            self.assertEqual(cell_metrics("think-ramp", "C=4"), ("dec", "ttft"))
            self.assertEqual(cell_metrics("resume", "gap=30m"), ("ttft",))
            self.assertEqual(cell_metrics("coldfill", "fill"), ("ttft",))
            self.assertEqual(cell_metrics("coldfill", "bg-during"), ("dec",))
            self.assertEqual(cell_metrics("other", "x"), tuple(METRICS))

        def test_simulate_block(self):
            # the §5 simulation's two ends, small: a +20% gain at C=4 adopts, a 0% effect never does
            self.assertGreaterEqual(sim_block((0.05, 0.03), 4, 0.20, 12, 10, 2000, 1), 0.8)
            self.assertEqual(sim_block((0.05, 0.03), 1, 0.0, 12, 10, 2000, 1), 0.0)
            # seeded from the block's own parameters (a string seed: not PYTHONHASHSEED), so a fixed
            # block gives this exact share in any process, and `--simulate`'s pool reproduces §5's table.
            # The boot-noise-free pins (2-tuple spreads, the default boots=3: A, B, A2) stay as they
            # were before the boot sd and B2 existed: a 2-tuple spread keeps its old seed and draws,
            # and boots=3 its old seed, so these shares must not move.
            self.assertEqual(sim_block((0.05, 0.03), 1, 0.10, 12, 20, 2000, 1), 0.15)
            self.assertEqual(sim_block((0.08, 0.05), 1, 0.20, 6, 20, 2000, 1), 0.25)
            # Pinned on §5's current cells (4% boot sd, A, B, A2, B2).
            self.assertEqual(sim_block((0.05, 0.03, 0.04), 1, 0.10, 12, 20, 2000, 1, boots=4), 0.15)
            self.assertEqual(sim_block((0.05, 0.03, 0.04), 1, 0.20, 6, 20, 2000, 1, boots=4), 0.30)

        # ---- restart-to-restart (boot) noise: one offset per arm, the reason for A, B, A', B' ----

        def boot_arm(self, dec, ttft, levels=(1, 4), reps=12, boot_sd=0.04):
            """One boot: each cell's dec and ttft carry an offset ~ N(1, boot_sd) shared by all
            reps, on top of 5% per request and 3% per rep."""
            return {f"C={c}": (arm(dec / c ** 0.3, 0.05, reps, c, self.rng, 0.03, boot_sd),
                               arm(ttft, 0.05, reps, c, self.rng, 0.03, boot_sd)) for c in levels}

        def test_synth_arm_boot_offset(self):
            # boot_sd=0 draws nothing: the stream every pinned result above rests on is unchanged
            self.assertEqual(arm(500, 0.05, 6, 4, random.Random(5), 0.03),
                             arm(500, 0.05, 6, 4, random.Random(5), 0.03, 0.0))
            # one offset per arm, shared by every rep: with no other noise all values are equal, off 500
            r = random.Random(5)
            a = arm(500, 0.0, 6, 4, r, 0.0, 0.04)
            self.assertEqual(len({x for c in a for x in c}), 1)
            self.assertNotEqual(a[0][0], 500)
            # across arms (boots) the offsets spread ~boot_sd
            offs = [arm(500, 0.0, 1, 1, r, 0.0, 0.04)[0][0] / 500 for _ in range(400)]
            self.assertAlmostEqual(st.pstdev(offs), 0.04, delta=0.006)
            self.assertAlmostEqual(st.mean(offs), 1.0, delta=0.006)

        def test_boot_offset_alone_is_not_adopted(self):
            # No true effect: A, B and A' differ only by their boots' ~4% offsets. Within-boot reps
            # cannot see an offset, so A vs A' reads as drift (its CI excludes 0) and B vs A alone
            # can clear the gate; with A' in the test (B beats A2 too, bar = max(gate, band)) no
            # trial adopts.
            drift = ab_alone = 0
            for t in range(4):
                pa = write(self.dir, f"a{t}.jsonl", self.boot_arm(500, 0.40))
                pb = write(self.dir, f"b{t}.jsonl", self.boot_arm(500, 0.40))
                pa2 = write(self.dir, f"a2{t}.jsonl", self.boot_arm(500, 0.40))
                for r in compare_files([pa, pb, pa2], ("dec", "ttft"), gate=5, n_boot=2000):
                    self.assertIn(r["verdict"], ("NO", "HOLD"), r)
                    drift += r["aa"]["gain_lo"] > 0 or r["aa"]["gain_hi"] < 0
                    ab_alone += r["ab"]["gain_lo"] > 5
            self.assertGreater(drift, 0)
            self.assertGreater(ab_alone, 0)
            # the same as a rate, through sim_block: 12 reps, 4% boot sd, no effect
            self.assertEqual(sim_block((0.05, 0.03, 0.04), 1, 0.0, 12, 20, 2000, 1), 0.0)
            # At C=4 the three-boot order is NOT free of false ADOPTs, and this tolerance knows it: B's
            # one boot can land high against both A and A2. Measured: this block adopts 0.05 (2 of
            # 40); METHOD.md §5's table (200 trials) 0.03 at 12 reps, 0.04 at 20, 0.04 at a 5% boot
            # sd. <= 0.10 bounds that known rate; the boot-robust order below brings it near 0.
            self.assertLessEqual(sim_block((0.05, 0.03, 0.04), 4, 0.0, 12, 40, 2000, 1), 0.10)
            # A second B boot (A, B, A2, B2: B2 must clear A2 too) needs both B boots to land high:
            # this block adopts none, §5's table 0.01 at 12 reps.
            self.assertEqual(sim_block((0.05, 0.03, 0.04), 4, 0.0, 12, 40, 2000, 1, boots=4), 0.0)

        def test_c4_true_20pct_win_adopts_at_12_reps_with_boot_offset(self):
            pa = write(self.dir, "a.jsonl", self.boot_arm(500, 0.40, levels=(4,)))
            pb = write(self.dir, "b.jsonl", self.boot_arm(600, 0.32, levels=(4,)))  # +20% dec, -20% ttft
            pa2 = write(self.dir, "a2.jsonl", self.boot_arm(500, 0.40, levels=(4,)))
            pb2 = write(self.dir, "b2.jsonl", self.boot_arm(600, 0.32, levels=(4,)))
            rows = compare_files([pa, pb, pa2, pb2], ("dec", "ttft"), gate=5, n_boot=2000)
            self.assertEqual([(r["cell"], r["verdict"]) for r in rows], [("C=4", "ADOPT"), ("C=4", "ADOPT")])
            # As a rate, at a 4% boot sd: the mid-range reading of a 3-5% restart offset, the one this
            # test takes. This block adopts 0.875 (0.80 with B2 as well); §5's table, at 120 trials,
            # 0.79 (0.81 with B2). Power falls with the boot sd: at 5% the table gives 0.65 (0.59
            # with B2) at 12 reps, so "a +20% win at C=4 adopts" holds at 4%, not across the range.
            self.assertGreaterEqual(sim_block((0.05, 0.03, 0.04), 4, 0.20, 12, 40, 2000, 1), 0.8)
            self.assertGreaterEqual(sim_block((0.05, 0.03, 0.04), 4, 0.20, 12, 40, 2000, 1, boots=4), 0.75)

        # ---- a second B boot: A, B, A2, B2 and longer chains ----

        def test_arm_names(self):
            self.assertEqual(arm_names(3), ["A", "B", "A2"])
            self.assertEqual(arm_names(6), ["A", "B", "A2", "B2", "A3", "B3"])

        def chain(self, decs, levels=(4,)):
            # boots with no boot offset (boot_sd 0), dec as given and ttft 0.40 * 500 / dec
            return [write(self.dir, f"{n}.jsonl", self.boot_arm(d, 0.40 * 500 / d, levels=levels, boot_sd=0.0))
                    for n, d in zip(arm_names(len(decs)), decs)]

        def test_both_b_boots_must_clear(self):
            # three boots whose one B boot clears the bar: never ADOPT (METHOD.md §2), HOLD for B2,
            # the bound it cleared still reported
            rows = compare_files(self.chain([500, 600, 500]), ("dec", "ttft"), gate=5, n_boot=2000)
            self.assertEqual([r["verdict"] for r in rows], ["HOLD", "HOLD"])
            for r in rows:
                held = [n for n in r["notes"] if n.startswith("HOLD: needs a second B boot")]
                self.assertEqual(len(held), 1, r["notes"])
                self.assertIn("gain lower bound", held[0])
                self.assertIn("compare A B A2 B2", held[0])
            self.assertIn("HOLD", format_rows(rows, self.chain([500, 600, 500]), False).splitlines()[-1])
            # ... and the bare test under it, the one sim_block measures for three boots, does clear
            self.assertEqual(_verdict(rows[0], [{"clusters": [[1.0]] * MIN_REPS, "n_err": 0, "n_refused": 0}] * 3,
                                      5, [], min_b_boots=1)[0], "ADOPT")
            # three boots whose B misses: NO, not HOLD -- a second B boot only adds tests
            rows = compare_files(self.chain([500, 510, 500]), ("dec",), gate=5, n_boot=2000)
            self.assertEqual(rows[0]["verdict"], "NO")
            self.assertFalse(any("second B boot" in n for n in rows[0].get("notes", [])))
            # four: B2 confirms, and the hold goes; every B is tested against its adjacent A boots
            paths = self.chain([500, 600, 500, 600])
            rows = compare_files(paths, ("dec", "ttft"), gate=5, n_boot=2000)
            self.assertEqual([r["verdict"] for r in rows], ["ADOPT", "ADOPT"])
            self.assertEqual([(t["b"], t["a"]) for t in rows[0]["tests"]], [("B", "A"), ("B", "A2"), ("B2", "A2")])
            self.assertFalse(any("second B boot" in n for n in rows[0].get("notes", [])))
            text = format_rows(rows, paths, False)
            self.assertIn("B2-A2 ", text)
            self.assertIn(f"B2={paths[3]}", text.splitlines()[0])
            # a second B boot that does not repeat B's gain: NO, decided by B2-A2
            rows = compare_files(self.chain([500, 600, 500, 505]), ("dec",), gate=5, n_boot=2000)
            self.assertEqual(rows[0]["verdict"], "NO")
            self.assertIn("(B2-A2)", rows[0]["notes"][-1])

        def test_chained_baseline(self):
            # A, B, A2, B2, A3: each B boot flanked by A boots, four tests, the band the widest A pair
            rows = compare_files(self.chain([500, 600, 500, 600, 510]), ("dec",), gate=5, n_boot=2000)
            r = rows[0]
            self.assertEqual(r["verdict"], "ADOPT")
            self.assertEqual([(t["b"], t["a"]) for t in r["tests"]],
                             [("B", "A"), ("B", "A2"), ("B2", "A2"), ("B2", "A3")])
            self.assertEqual([(x["a"], x["a2"]) for x in r["aas"]], [("A", "A2"), ("A2", "A3")])
            self.assertEqual(r["band_pct"], max(band_of(x) for x in r["aas"]))
            self.assertGreater(r["band_pct"], 1.0)  # the A2-A3 pair, not A-A2
            # a chain still needs the gate
            with self.assertRaises(ValueError):
                compare_files(self.chain([500, 600, 500, 600]), ("dec",))

        def test_single_pair_never_adopts(self):
            # B 50% faster than A, each its own boot, but no A': no verdict, and asking for one is refused
            pa = write(self.dir, "a.jsonl", self.boot_arm(500, 0.40))
            pb = write(self.dir, "b.jsonl", self.boot_arm(750, 0.20))
            rows = compare_files([pa, pb], ("dec", "ttft"), n_boot=2000)
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertNotIn("verdict", r)
                self.assertGreater(r["ab"]["gain_lo"], 5)  # a clear interval, still no verdict
            self.assertNotIn("ADOPT", format_rows(rows, [pa, pb], False))
            with self.assertRaises(ValueError):
                compare_files([pa, pb], ("dec", "ttft"), gate=5, n_boot=2000)
            args = types.SimpleNamespace(files=[pa, pb], metric=None, gate=5, aa=False, boot=2000,
                                         boot_seed=1, only_mode=None, json=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
                cmd_compare(args)
            self.assertIn("a verdict needs A, B and A2", str(cm.exception.code))
            self.assertEqual(out.getvalue(), "")

        def test_compare_unmatched_metric(self):
            # resume cells carry ttft only: --metric dec matches no cell, and nothing is scored
            pa, _ = self.run_mode("resume", "A", 1.0, 2)
            pb, _ = self.run_mode("resume", "B", 1.5, 2)
            args = types.SimpleNamespace(files=[pa, pb], metric="dec", gate=None, aa=False, boot=2000,
                                         boot_seed=1, only_mode=None, json=True)
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out), \
                    self.assertRaises(SystemExit) as cm:
                cmd_compare(args)
            self.assertIn("nothing scored", str(cm.exception.code))
            self.assertIn("no cell carries --metric dec", err.getvalue())
            self.assertEqual(out.getvalue(), "")
            # one of two matches: the other is named, the rows still print, no exit
            args.metric = "dec,ttft"
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                cmd_compare(args)
            self.assertIn("no cell carries --metric dec ", err.getvalue())
            self.assertEqual([json.loads(s)["metric"] for s in out.getvalue().splitlines()], ["ttft"])
            # each mode's own metrics (no --metric): nothing named, nothing to warn about
            self.assertEqual(unmatched_metrics(compare_files([pa, pb], n_boot=2000), None), [])

        def serve(self, drop_idle=False, status=200):
            """A loopback HTTP/1.1 stub engine; returns (base URL, seen) where seen lists
            (client port, Connection header) per request. drop_idle closes every connection after
            its response without saying so, as an idle-timeout on the far end does; status answers
            every request with that code."""
            import http.server
            seen = []
            sse = "".join(f"data: {json.dumps(o)}\n\n" for o in (
                {"choices": [{"delta": {"content": "READY"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})) + "data: [DONE]\n\n"

            class H(http.server.BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def do_POST(self):
                    self.rfile.read(int(self.headers["Content-Length"]))
                    seen.append((self.client_address[1], self.headers.get("Connection")))
                    b = sse.encode() if status == 200 else b"bad gateway"
                    self.send_response(status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    self.wfile.flush()
                    if drop_idle:
                        self.close_connection = True

                def log_message(self, *a):
                    pass

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
            srv.handle_error = lambda *a: None  # a client's reset of a dropped connection is expected
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.addCleanup(srv.server_close)
            self.addCleanup(srv.shutdown)
            return f"http://127.0.0.1:{srv.server_address[1]}", seen

        def conn(self, new):
            g = globals()
            saved = g["NEW_CONNECTION"]
            g["NEW_CONNECTION"] = new
            _POOL.clear()
            self.addCleanup(lambda: (g.__setitem__("NEW_CONNECTION", saved), _POOL.clear()))

        def test_keep_alive_reuses_one_connection(self):
            self.conn(False)
            base, seen = self.serve()
            rs = [stream(base, "k", {"model": MODEL, "messages": [{"role": "user", "content": "x"}]}) for _ in range(3)]
            self.assertTrue(all("err" not in r for r in rs), rs)
            self.assertEqual([(r["content"], r["ct"], r["finish"]) for r in rs], [("READY", 1, "stop")] * 3)
            self.assertEqual([r["conn_reused"] for r in rs], [False, True, True])
            self.assertEqual(len({p for p, _ in seen}), 1, seen)  # one TCP connection for all three
            self.assertTrue(all(h != "close" for _, h in seen), seen)
            # two concurrent workers each hold their own connection; both go back to the pool
            out = [None, None]
            ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, stream(base, "k", {"model": MODEL})))
                   for i in range(2)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            self.assertTrue(all("err" not in r for r in out), out)
            self.assertLessEqual(len({p for p, _ in seen}), 2)
            # the record names the mode and whether the connection was reused
            lines = []
            args = types.SimpleNamespace(rec=lines.append, run_id="r", arm="A", seed=1, seed_offset=0,
                                         base_url=base, max_live=0)
            record(args, "resume", "gap=0m", 1, 0, rs[1])
            o = json.loads(lines[0])
            self.assertEqual((o["conn"], o["conn_reused"]), ("keep-alive", True))

        def test_keep_alive_reconnects_after_far_end_closes(self):
            self.conn(False)
            base, seen = self.serve(drop_idle=True)
            rs = [stream(base, "k", {"model": MODEL}) for _ in range(3)]
            self.assertTrue(all("err" not in r for r in rs), rs)
            self.assertEqual(len({p for p, _ in seen}), 3, seen)  # a new connection each time, no error
            self.assertEqual(len(seen), 3)  # and no request sent twice to the far end

        def test_new_connection_is_the_old_path(self):
            self.conn(True)
            base, seen = self.serve()
            rs = [stream(base, "k", {"model": MODEL}) for _ in range(2)]
            self.assertTrue(all("err" not in r for r in rs), rs)
            self.assertEqual([h for _, h in seen], ["close", "close"])
            self.assertEqual(len({p for p, _ in seen}), 2)
            self.assertEqual(_POOL, {})
            self.assertEqual(conn_mode(), "new")

        def test_http_error_is_an_err_record(self):
            self.conn(False)
            base, seen = self.serve(status=502)
            rs = [stream(base, "k", {"model": MODEL}) for _ in range(2)]
            self.assertEqual([r.get("err") for r in rs], ["HTTP Error 502: Bad Gateway"] * 2)
            self.assertEqual(len({p for p, _ in seen}), 2)  # an errored connection is closed, never pooled
            self.assertEqual(_POOL, {})
            # a refused connect is an err record too, as on the old path
            r = stream("http://127.0.0.1:9", "k", {"model": MODEL}, timeout=2)  # discard port
            self.assertIn("err", r)

        def test_conn_modes_must_match(self):
            def with_conn(src, name, conn):
                p = os.path.join(self.dir, name)
                with open(src) as f, open(p, "w") as g:
                    for line in f:
                        if line.startswith('{"kind"'):
                            line = json.dumps({**json.loads(line), "conn": conn}) + "\n"
                        g.write(line)
                return p
            pa = write(self.dir, "a.jsonl", self.cells(500, 0.4))  # no "conn": an old record, i.e. "new"
            pb = write(self.dir, "b.jsonl", self.cells(700, 0.3))
            pa2 = write(self.dir, "a2.jsonl", self.cells(500, 0.4))
            pb2 = write(self.dir, "b2.jsonl", self.cells(700, 0.3))
            # an old record and an explicit "new" are one mode: no connection note
            rows = compare_files([pa, with_conn(pb, "b-new.jsonl", "new"), pa2, pb2], ("dec",), gate=5, n_boot=2000)
            self.assertTrue(rows)
            self.assertFalse(any("connection modes" in n for r in rows for n in r.get("notes", [])), rows)
            # a keep-alive B against "new" A boots: HOLD, naming why
            rows = compare_files([pa, with_conn(pb, "b-ka.jsonl", "keep-alive"), pa2, pb2], ("dec",),
                                 gate=5, n_boot=2000)
            self.assertTrue(rows)
            self.assertTrue(all(r["verdict"] == "HOLD" for r in rows), rows)
            self.assertTrue(all(any(n.startswith("HOLD: arms ran with different connection modes")
                                    for n in r["notes"]) for r in rows), rows)
            self.assertTrue(all(any(n == "connection modes differ across arms: new vs keep-alive vs new vs new"
                                    for n in r["notes"]) for r in rows), rows)

        def test_keep_alive_survives_a_pool_of_stale_connections(self):
            # A high-C rung leaves several idle connections in the pool; the far end closes them all
            # while they idle (drop_idle: closed after each response, unannounced). The next rung,
            # with fewer requests than idle connections, must record no err: the retry goes out on
            # a new connection, never on another pooled one, and the stale ones are closed.
            self.conn(False)
            base, seen = self.serve(drop_idle=True)
            out = [None] * 4
            ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, stream(base, "k", {"model": MODEL})))
                   for i in range(4)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            self.assertTrue(all("err" not in r for r in out), out)
            key = next(iter(_POOL))
            n_idle = len(_POOL[key])
            self.assertGreaterEqual(n_idle, 3, _POOL)
            time.sleep(0.2)  # let the far end's closes land
            before = len(seen)
            rs = [stream(base, "k", {"model": MODEL}) for _ in range(n_idle - 1)]
            self.assertEqual([r.get("err") for r in rs], [None] * (n_idle - 1), rs)
            self.assertEqual([r["conn_reused"] for r in rs], [False] * (n_idle - 1))  # each ran on a new one
            self.assertEqual(len(seen) - before, n_idle - 1)  # and reached the far end exactly once
            self.assertEqual(len(_POOL[key]), 1)  # the stale ones were closed; only the last one is back

    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    sys.exit(0 if res.wasSuccessful() else 1)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["think-ramp", "resume", "coldfill", "compare", "selftest"])
    p.add_argument("files", nargs="*", help="compare: A.jsonl B.jsonl [A2.jsonl [B2.jsonl ...]] in boot order "
                                                    "(per-request --records files)")
    p.add_argument("--base-url", default=None,
                   help="required off the box (the engine or its door origin); with --on-box it defaults to http://127.0.0.1:$ENGINE_PORT (8002)")
    p.add_argument("--on-box", action="store_true",
                   help="bench.py runs ON the box: /v1/loads read locally, engine key from $VLLM_API_KEY or "
                        "/etc/environment, loopback base URL; no box file (METHOD.md)")
    p.add_argument("--new-connection", action="store_true",
                   help="the old path: a new connection per request with `Connection: close` (each pays its "
                        "TCP+TLS handshake and a cold upload); default a shared LIFO pool of keep-alive "
                        "connections, reused (proxy variables ignored). Records and rows name the mode in \"conn\" (METHOD.md §1)")
    p.add_argument("--out", default=None, help="append JSON lines here as well as stdout")
    p.add_argument("--records", default=None,
                   help="append per-request records here (default <out minus .jsonl>.req.jsonl when --out is set)")
    p.add_argument("--arm", default=None, help="label stamped on every record (e.g. A, B, A2, B2)")
    p.add_argument("--max-live", type=int, default=4, help="wait until live #running-req <= this before a rung")
    p.add_argument("--gate-tries", type=int, default=5,
                   help="consecutive /v1/loads read failures before the run stops (fail closed)")
    p.add_argument("--seed", type=int, default=int(time.time()))
    p.add_argument("--seed-offset", type=int, default=int(time.time()) % 10 ** 6)
    p.add_argument("--levels", default="1,2,4,8,16")
    p.add_argument("--reps", type=int, default=6, help="think-ramp: reps of every level (METHOD.md: >= 6 per cell)")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--gaps-min", default="0.1,5,30,60,90")
    p.add_argument("--ctx-tokens", type=int, default=70000,
                   help="build_prefix target; 70000 lands at ~87k prompt tokens through the router (baseline)")
    p.add_argument("--bg", type=int, default=4)
    p.add_argument("--bg-tokens", type=int, default=16000)
    p.add_argument("--lead-s", type=float, default=8.0)
    p.add_argument("--fill-tokens", type=int, default=48300,
                   help="build_prefix target; 48300 lands at ~60k prompt tokens (60000 lands at ~74.5k)")
    p.add_argument("--gate", type=float, default=None,
                   help="compare with A2: the item's adoption gate in %% gain (a minimum)")
    p.add_argument("--aa", action="store_true", help="compare A.jsonl A2.jsonl: report the A/A noise band")
    p.add_argument("--metric", default=None,
                   help="compare: narrow the metrics scored (dec, ttft); default each mode's own: think-ramp "
                        "dec+ttft, resume ttft, coldfill fill ttft, coldfill bg-during dec (METHOD.md §4)")
    p.add_argument("--boot", type=int, default=None,
                   help=f"compare: bootstrap resamples (>= {MIN_BOOT}; default 5000, selftest --simulate {MIN_BOOT})")
    p.add_argument("--boot-seed", type=int, default=1, help="compare: bootstrap RNG seed (reproducible output)")
    p.add_argument("--only-mode", default=None, help="compare: score only records of this mode")
    p.add_argument("--json", action="store_true", help="compare: one JSON object per row instead of the table")
    p.add_argument("--simulate", action="store_true",
                   help="selftest: instead of the unit test, re-run METHOD.md §5's adoption-rate simulation "
                        "(--trials, --null-trials, --boot, --boot-seed)")
    p.add_argument("--trials", type=int, default=120, help="selftest --simulate: trials per non-null cell")
    p.add_argument("--null-trials", type=int, default=200, help="selftest --simulate: trials per 0%%-effect cell")
    args = p.parse_args()
    if args.boot is None:
        args.boot = MIN_BOOT if args.simulate else 5000
    if args.mode == "selftest" and args.simulate:
        cmd_simulate(args.trials, args.null_trials, args.boot, args.boot_seed)
        return
    if args.mode == "selftest":
        cmd_selftest()
    if args.mode == "compare":
        cmd_compare(args)
        return
    if args.files:
        p.error(f"{args.mode} takes no positional files")
    global ON_BOX, NEW_CONNECTION
    ON_BOX = args.on_box
    NEW_CONNECTION = args.new_connection
    if ON_BOX:
        key = engine_key()
        args.base_url = args.base_url or f"http://127.0.0.1:{engine_port()}"
    else:
        box()  # fail now, not at the first rung, if the box file is missing
        if not args.base_url:
            p.error("--base-url is required when not running on the box (--on-box)")
        key = remote_engine_key()
    fh = open(args.out, "a") if args.out else None
    if args.records is None and args.out:
        args.records = (args.out[:-6] if args.out.endswith(".jsonl") else args.out) + ".req.jsonl"
    if args.records and args.out and os.path.abspath(args.records) == os.path.abspath(args.out):
        p.error("--records must differ from --out (rows and per-request records are separate files)")
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    args.run_id = f"{args.mode}-{started}-{args.seed}-{args.seed_offset}"
    rh = open(args.records, "a") if args.records else None
    args.rec = None
    if rh:
        def rec(s):
            rh.write(s + "\n")
            rh.flush()
        args.rec = rec
        rec(f"# {args.mode} start {started} run={args.run_id} arm={args.arm} base={args.base_url} model={MODEL} "
            f"conn={conn_mode()}")
        print(f"# per-request records -> {args.records}", file=sys.stderr, flush=True)

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    log(f"# {args.mode} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} base={args.base_url} "
        f"model={MODEL} seed={args.seed} seed_offset={args.seed_offset} conn={conn_mode()}")
    try:
        {"think-ramp": cmd_think_ramp, "resume": cmd_resume, "coldfill": cmd_coldfill}[args.mode](args, key, log)
    except LoadGateError as e:
        log(f"# {args.mode} aborted {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {e}")
        sys.exit(2)
    log(f"# {args.mode} end {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

if __name__ == "__main__":
    main()
