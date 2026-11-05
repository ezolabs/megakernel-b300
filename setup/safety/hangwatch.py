#!/usr/bin/env python3
"""hangwatch.py -- engine stall watcher for box-a (runs ON THE BOX under `setsid nohup`). Python 3 stdlib only.

  setsid nohup python3 /workspace/mk29/harness/hangwatch.py --out /workspace/mk29/watch/<tag> \
      > /workspace/mk29/watch/<tag>.nohup 2>&1 < /dev/null &
  touch /workspace/mk29/watch/<tag>/stop          # ends it (within one poll)

Every --interval s (default 3) it reads /metrics (num_running_reqs, num_queue_reqs and the progress counters) and every
--health-every s (default: the same interval; --health-timeout 5 s) GET /health. PROGRESS is any change of:
  realtime_tokens_total (all modes: prefill_compute / prefill_cache / decode -- the scheduler bumps it at every prefill
  batch, chunk by chunk, and every decode step), generation_tokens_total, prompt_tokens_total (these two move only when
  a request FINISHES), or the count of `Prefill batch` / `Decode batch` lines in the serve log (the scheduler writes one
  per prefill batch and one every decode-log-interval steps).
IN FLIGHT is num_running_reqs + num_queue_reqs > 0. A zero in-flight reading counts as idle (it resets the stall
clock) only once it has held for --idle-hold-s (35) s, so a stale zero gauge cannot mask a stall; an ARRIVAL (in-flight
goes from a zero reading to > 0, whatever the gap) also resets it, so the stall clock starts no earlier than the request
it times (a 50 s prefill after a 20 s think gap is not 70 s without progress). A STALL is:
  no_token        running > 0 and no progress for --stall-s (60) s (a cold 900k prompt is ~30-60 s of prefill with
                  no finished request: --stall-s must exceed the longest single prefill batch);
  queue_wedge     running == 0, queue > 0 and no progress for --stall-s s (requests waiting, nothing admitted);
  health          /health failed (non-200, error or > --health-timeout s) --health-fails (2) times in a row;
  metrics_down    /metrics unreachable for --stall-s s.
On a stall it FIRST writes <out>/HANG (one JSON line, appended: one line per episode), then captures evidence into
<out>/hang-<UTC>/: reason.json, metrics.txt, health.txt, procs.txt (the sglang process tree with roles), one
pyspy-<role>-<pid>.txt per sglang process (scheduler/TP ranks first; `py-spy dump --pid`, plus a --native dump of each
scheduler rank), nvidia-smi.txt, serve-tail.txt (last 300 log lines, the engine key REDACTED), ps.txt. Where py-spy
cannot read a process (a container without CAP_SYS_PTRACE under yama ptrace_scope 1: 'Permission denied') it falls
back to the in-process stack dump (any rank whose py-spy dump shows no stack) when --dump-signal names one (default
none: off): for each scheduler/TP rank whose /proc/<pid>/status SigCgt shows that signal caught -- for USR2 a
faulthandler.register(SIGUSR2, all_threads=True) handler, e.g. from a sitecustomize on the engine's PYTHONPATH
(qa/sglang-takes-20261104/merge-b300/site/sitecustomize.py) -- it sends that signal, one rank at a time, and writes
the serve-log bytes the dump appended to faulthandler-<role>-<pid>.txt (every thread's Python stack, written by the
stuck process itself even when it sits in a C call); a rank without the handler is never signalled (the default action
of SIGUSR2 is to terminate). SigCgt cannot tell faulthandler from another catcher: in our engines LLVM's (Triton's)
SA_RESETHAND handler catches USR2, resets it to the default and re-raises, so the rank dies of the dump -- hence the
default none, and USR2 only for an engine whose faulthandler site file registers first. Apart from that dump signal
and --on-hang kill it never signals the engine and keeps watching; a new episode is captured only after progress
resumes (or --recapture-s passes), at most --max-captures times. A /health failure while the metrics move with work in
flight is a busy engine (/health is a real generation queued behind user work), never a `health` stall.

BOOT: an engine is booting while it has not answered /health 200 since this watcher killed it, or while its top-level
serve process (ps args matching --serve-pattern, --port = --serve-port when it carries one) is younger than
--boot-grace-s (2700) and has not answered /health 200 yet; no HANG and no kill then, every stall clock held at zero.

--on-hang kill (default none): when a stall is still on at a poll AFTER its capture finished (or --on-hang-after
(120) s into an episode that took no capture), SIGTERM the top-level serve process's process group (relaunch.sh execs
`sglang serve` under setsid, so it leads one; otherwise it and its descendants), SIGKILL what is left after
--on-hang-term-s (20), log `ON-HANG KILL pid=... at ...` in watch.log and a JSON line in <out>/KILLS, then keep
watching: watchdog.sh relaunches an engine with no `sglang serve` process at once. At most one kill per episode and
--on-hang-max (3) per --on-hang-window-s (3600), then only logged (KILLS outcome rate_limited; the window counts the
kills already in <out>/KILLS, so a restarted watcher remembers them); never while --hold-file
(/workspace/tools/watchdog.hold) is fresh (a deliberate restart: the watchdog relaunches nothing then), nor with zero
or several serve processes matching. Those refusals (held / no_target / ambiguous / ps_failed) are retried every
--on-hang-retry-s (60) while the stall lasts; rate_limited is retried once the window has room again.

--service: the long-running mode beside watchdog.sh (`setsid nohup`): one instance per --out (hangwatch.pid), a stale
stop file removed at start (the pid file at a clean stop), watch.log rotated to watch.log.1 at half of --log-max-mb
(100), a loop error logged and the loop re-entered, --max-captures 50 by default.

It also tails the serve log incrementally and counts `MK_TP_PROBE MISMATCH` and `MK_TP_PROBE ok` lines (the env-gated
TP-consistency probe); the first --keep-mismatch MISMATCH lines are kept (redacted) in <out>/mk_tp_probe_mismatch.txt.
<out>/state.json is rewritten every poll (counters, last metrics, episodes); <out>/watch.log gets a status line every 60 s.
The first good /metrics logs which metric names matched ("metrics matched: ..."); if the running-requests metric or every
progress counter is absent it logs one WARNING (and state.json carries metric_warning): no_token / queue_wedge
detection is then inert (a missing queue metric only makes queue_wedge inert, named in the same warning).

py-spy: uses --pyspy, else /opt/pyspy/bin/py-spy, else a py-spy on PATH; if none, it creates the separate venv
/opt/pyspy (`python3 -m venv /opt/pyspy && /opt/pyspy/bin/pip install py-spy`) at start -- never inside an engine
venv. --no-install skips that.

The engine key is read from $SGLANG_API_KEY / $VLLM_API_KEY / $API_KEY, else from VLLM_API_KEY= (or SGLANG_API_KEY=,
API_KEY=) in /etc/environment. It is only sent as a bearer header and is redacted from every file this script writes.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

KEY_NAMES = ("SGLANG_API_KEY", "VLLM_API_KEY", "API_KEY")


def load_key(path="/etc/environment"):
    for n in KEY_NAMES:
        if os.environ.get(n):
            return os.environ[n].strip()
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return ""
    for n in KEY_NAMES:
        for l in lines:
            l = l.strip()
            if l.startswith("export "):
                l = l[7:].strip()
            if l.startswith(n + "="):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def utc(fmt="%Y-%m-%dT%H:%M:%SZ", t=None):
    return time.strftime(fmt, time.gmtime(time.time() if t is None else t))


def metric(text, name, agg="sum"):
    """Value of a Prometheus metric across label sets. agg=sum adds every series, agg=max takes the largest; series
    labelled with a tp_rank other than 0 are skipped (sglang exports the same scheduler gauge per rank on some builds)."""
    vals = []
    for m in re.finditer(r"^" + re.escape(name) + r"(\{[^}]*\})?\s+([0-9.eE+-]+|NaN|\+Inf)\s*$", text, re.M):
        lab = m.group(1) or ""
        if "tp_rank" in lab and 'tp_rank="0"' not in lab:
            continue
        try:
            vals.append(float(m.group(2)))
        except ValueError:
            pass
    if not vals:
        return None
    return max(vals) if agg == "max" else sum(vals)


def first_metric(text, names, agg="sum"):
    for n in names:
        v = metric(text, n, agg)
        if v is not None:
            return v
    return None


def matched_name(text, names):
    """The first of `names` present in a /metrics body, or None."""
    for n in names:
        if re.search(r"^" + re.escape(n) + r"(\{[^}]*\})?\s", text, re.M):
            return n
    return None


RUN_NAMES = ("sglang:num_running_reqs", "sglang_num_running_reqs")
QUEUE_NAMES = ("sglang:num_queue_reqs", "sglang_num_queue_reqs")
GEN_NAMES = ("sglang:generation_tokens_total", "sglang_generation_tokens_total", "sglang:generation_tokens")
PROMPT_NAMES = ("sglang:prompt_tokens_total", "sglang_prompt_tokens_total", "sglang:prompt_tokens")
# per-step scheduler counter, labelled mode=prefill_compute|prefill_cache|decode (summed over modes)
REALTIME_NAMES = ("sglang:realtime_tokens_total", "sglang_realtime_tokens_total")
PROGRESS = (("realtime_tokens", REALTIME_NAMES), ("gen_tokens", GEN_NAMES), ("prompt_tokens", PROMPT_NAMES))


class Redactor:
    def __init__(self, key):
        self.keys = [k for k in {key} if k and len(k) >= 6]
        self.rx = re.compile(r"(?i)((?:api[_-]?key|authorization|bearer|x-api-key)[\"']?\s*[:=]?\s*[\"']?(?:bearer\s+)?)"
                             r"([A-Za-z0-9_\-\.]{12,})")

    def __call__(self, s):
        for k in self.keys:
            s = s.replace(k, "<REDACTED>")
        return self.rx.sub(lambda m: m.group(1) + "<REDACTED>", s)


# ---------------------------------------------------------------- process tree

def read_proc(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        stat = open(f"/proc/{pid}/stat").read()
        ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        comm = open(f"/proc/{pid}/comm").read().strip()
        return dict(pid=pid, ppid=ppid, comm=comm, cmd=cmd)
    except (OSError, ValueError, IndexError):
        return None


def role_of(p):
    """sglang sets process titles like `sglang::scheduler_TP0`, `sglang::scheduler_DP0_TP1`, `sglang::detokenizer`."""
    s = p["cmd"] + " " + p["comm"]
    m = re.search(r"sglang::scheduler[_A-Za-z0-9]*?TP(\d+)", s) or re.search(r"scheduler[_-]?TP(\d+)", s)
    if m:
        return f"tp{int(m.group(1))}"
    if "sglang::scheduler" in s:
        return "scheduler"
    if "detokenizer" in s:
        return "detokenizer"
    if "sglang.launch_server" in s or "sglang serve" in s or "launch_server" in s:
        return "launcher"
    if "sglang::" in s:
        return re.sub(r"[^a-z0-9]+", "_", s.split("sglang::", 1)[1].split()[0].lower())[:24] or "sglang"
    return "child"


def sig_num(name):
    """--dump-signal value -> signal number (None for 'none'): a name (USR2, SIGUSR2) or a number of a real signal.
    Raises ValueError for anything else; main() calls it once at start so a typo fails there, not mid-capture."""
    if not name or name.lower() == "none":
        return None
    try:
        if name.isdigit():
            return int(signal.Signals(int(name)))
        n = name.upper()
        return int(signal.Signals[n if n.startswith("SIG") else "SIG" + n])
    except (KeyError, ValueError):
        raise ValueError(f"unknown signal {name!r}") from None

def catches(pid, sig, proc="/proc"):
    """True when /proc/<pid>/status SigCgt has `sig` set (a handler is installed, so the signal cannot kill it)."""
    try:
        with open(f"{proc}/{pid}/status") as f:
            for ln in f:
                if ln.startswith("SigCgt:"):
                    return bool(int(ln.split()[1], 16) >> (sig - 1) & 1)
    except (OSError, ValueError, IndexError):
        pass
    return False

def sglang_procs():
    """Every process whose title/cmdline mentions sglang, plus all descendants of a launcher (TP ranks spawned by
    multiprocessing can carry a bare python cmdline until setproctitle runs)."""
    procs = {}
    if not os.path.isdir("/proc"):  # not Linux (offline test): no process tree
        return []
    for d in os.listdir("/proc"):
        if d.isdigit():
            p = read_proc(int(d))
            if p:
                procs[p["pid"]] = p
    me = os.getpid()
    hit = {pid for pid, p in procs.items()
           if pid != me and ("sglang" in p["cmd"] or "sglang" in p["comm"]) and "hangwatch" not in p["cmd"]
           and "synthmix" not in p["cmd"] and "py-spy" not in p["cmd"]}
    kids = {}
    for pid, p in procs.items():
        kids.setdefault(p["ppid"], []).append(pid)
    stack = list(hit)
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in hit and "py-spy" not in procs[c]["cmd"]:
                hit.add(c)
                stack.append(c)
    out = []
    for pid in sorted(hit):
        p = dict(procs[pid])
        p["role"] = role_of(p)
        out.append(p)
    order = {"tp": 0, "scheduler": 1, "launcher": 2, "detokenizer": 3}
    out.sort(key=lambda p: (order.get(re.sub(r"\d+$", "", p["role"]), 4), p["role"], p["pid"]))
    return out


# ---------------------------------------------------------------- the serve process (--on-hang kill, boot grace)

SERVE_PATTERN = r"sglang serve\b|sglang\.launch_server\b"  # watchdog.sh's liveness is `sglang serv[e]`

def etime_s(s):
    """ps `etime` ([[dd-]hh:]mm:ss) -> seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(float(x)) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def ps_table():
    """Every process as dict(pid, ppid, pgid, stat, age_s, args), from `ps` (procps on the box, BSD ps on the Mac)."""
    r = subprocess.run(["ps", "-ww", "-A", "-o", "pid=,ppid=,pgid=,stat=,etime=,args="],
                       capture_output=True, text=True, timeout=15)
    rows = []
    for ln in r.stdout.splitlines():
        f = ln.split(None, 5)
        if len(f) < 5:
            continue
        try:
            rows.append(dict(pid=int(f[0]), ppid=int(f[1]), pgid=int(f[2]), stat=f[3], age_s=etime_s(f[4]),
                             args=f[5] if len(f) > 5 else ""))
        except ValueError:
            continue
    return rows


def serve_targets(rows, pattern, port, me=None):
    """The top-level serve process(es) among ps rows: args match `pattern`, not a zombie, not this watcher (or a
    harness script), and -- when `port` is set and the args carry a --port -- that port; a match whose parent also
    matches (a forked child still showing the launcher's argv) is not top-level."""
    rx = re.compile(pattern)
    me = os.getpid() if me is None else me
    hit = {}
    for r in rows:
        a = r["args"]
        if r["pid"] == me or r["stat"].startswith("Z") or not rx.search(a) \
                or "hangwatch" in a or "synthmix" in a or "py-spy" in a:
            continue
        ports = re.findall(r"--port[ =](\d+)(?:\s|$)", a)
        if port and ports and str(port) not in ports:
            continue
        hit[r["pid"]] = r
    return [r for r in sorted(hit.values(), key=lambda r: r["pid"]) if r["ppid"] not in hit]


def descendants(rows, pid):
    kids = {}
    for r in rows:
        kids.setdefault(r["ppid"], []).append(r["pid"])
    out, stack = {pid}, [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


# ---------------------------------------------------------------- py-spy

def find_pyspy(explicit, install, log):
    for c in ([explicit] if explicit else []) + ["/opt/pyspy/bin/py-spy", shutil.which("py-spy") or ""]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    if not install:
        log("py-spy not found and --no-install: stalls will be captured without stacks")
        return None
    log("py-spy not found: creating separate venv /opt/pyspy")
    try:
        subprocess.run([sys.executable, "-m", "venv", "/opt/pyspy"], check=True, capture_output=True, timeout=180)
        r = subprocess.run(["/opt/pyspy/bin/pip", "install", "-q", "py-spy"], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log("py-spy install failed: " + (r.stderr or r.stdout)[-500:].replace("\n", " "))
    except Exception as e:  # noqa: BLE001 -- any failure just means no stacks
        log("py-spy venv failed: " + repr(e)[:300])
    p = "/opt/pyspy/bin/py-spy"
    return p if os.path.isfile(p) else None


# ---------------------------------------------------------------- watcher

class Watch:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        os.makedirs(self.out, exist_ok=True)
        self.key = load_key()
        self.red = Redactor(self.key)
        self.base = a.url.rstrip("/")
        self.pyspy = None
        self.state = dict(start=utc(), pid=os.getpid(), url=self.base, serve_log=a.serve_log, stall_s=a.stall_s,
                          idle_hold_s=a.idle_hold_s,
                          polls=0, metrics_ok=0, metrics_fail=0, health_ok=0, health_fail=0, max_running=0,
                          serve_batches=dict(prefill=0, decode=0), last=None, episodes=[], mk_tp_probe=dict(ok=0, mismatch=0, first_mismatch=None,
                                                                    last_mismatch=None),
                          on_hang=getattr(a, "on_hang", "none"), kills=[], booting=None)
        self.log_pos = None
        self.log_ino = None
        self.kept_mismatch = 0
        self.log_cap = int(getattr(a, "log_max_mb", 0) or 0) << 20
        self.retry_at = None  # set by on_hang_kill: when a refused kill may be tried again (inf: not this episode)
        self.kill_ts = self.seed_kills()  # times of the kills inside --on-hang-window-s, for the rate limit

    KILL_OUTCOMES = ("killed", "term", "kill_failed")

    def seed_kills(self):
        """Kill times still inside --on-hang-window-s, from <out>/KILLS (its last 4 MiB), so a restarted --service
        watcher keeps the rate limit across its own restart."""
        win = getattr(self.a, "on_hang_window_s", 0) or 0
        now = time.time()
        try:
            with open(os.path.join(self.out, "KILLS"), "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - (4 << 20)))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []
        ts = []
        for ln in lines:
            try:
                k = json.loads(ln)
                if k.get("outcome") in self.KILL_OUTCOMES and 0 <= now - float(k["ts"]) < win:
                    ts.append(float(k["ts"]))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue  # a torn or foreign line
        return sorted(ts)

    def log(self, msg):
        p = os.path.join(self.out, "watch.log")
        if self.log_cap:  # --service: two generations (watch.log, watch.log.1) of half the cap each
            try:
                if os.path.getsize(p) >= self.log_cap // 2:
                    os.replace(p, p + ".1")
            except OSError:
                pass
        with open(p, "a") as f:
            f.write(utc() + " " + self.red(msg) + "\n")

    def get(self, path, timeout):
        h = {"Authorization": "Bearer " + self.key} if self.key else {}
        rq = urllib.request.Request(self.base + path, headers=h)
        try:
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")[:2000]

    def save_state(self):
        tmp = os.path.join(self.out, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, os.path.join(self.out, "state.json"))

    # -- serve-log probe counter
    def scan_log(self):
        path = self.a.serve_log
        try:
            st = os.stat(path)
        except OSError:
            return
        if self.log_pos is None:
            # first sight: count from the start only if asked, else from the current end
            self.log_pos = 0 if self.a.scan_from_start else st.st_size
            self.log_ino = st.st_ino
        if st.st_ino != self.log_ino or st.st_size < self.log_pos:  # rotated or truncated
            self.log_pos, self.log_ino = 0, st.st_ino
        if st.st_size == self.log_pos:
            return
        pr = self.state["mk_tp_probe"]
        with open(path, "rb") as f:
            f.seek(self.log_pos)
            chunk = f.read(min(st.st_size - self.log_pos, 64 << 20))
        nl = chunk.rfind(b"\n")
        if nl < 0:
            return
        self.log_pos += nl + 1
        sb = self.state["serve_batches"]
        sb["prefill"] += chunk.count(b"Prefill batch", 0, nl)
        sb["decode"] += chunk.count(b"Decode batch", 0, nl)
        for raw in chunk[:nl].split(b"\n"):
            if b"MK_TP_PROBE" not in raw:
                continue
            line = raw.decode("utf-8", "replace")
            if "MK_TP_PROBE MISMATCH" in line:
                pr["mismatch"] += 1
                pr["last_mismatch"] = utc()
                pr["first_mismatch"] = pr["first_mismatch"] or pr["last_mismatch"]
                if self.kept_mismatch < self.a.keep_mismatch:
                    self.kept_mismatch += 1
                    with open(os.path.join(self.out, "mk_tp_probe_mismatch.txt"), "a") as f:
                        f.write(self.red(line)[:4000] + "\n")
            elif "MK_TP_PROBE ok" in line:
                pr["ok"] += 1

    # -- evidence
    def sh(self, d, name, cmd, timeout):
        try:
            r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
            txt = r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
        except Exception as e:  # noqa: BLE001
            txt = "capture failed: " + repr(e)
        with open(os.path.join(d, name), "w") as f:
            f.write(self.red(txt))

    def tail(self, path, n):
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - (4 << 20)))  # the last 4 MiB is plenty for N lines
                data = f.read().decode("utf-8", "replace")
            return "\n".join(data.splitlines()[-n:]) + "\n"
        except OSError as e:
            return "tail failed: " + repr(e) + "\n"

    def capture(self, reason, info):
        t = time.time()
        ep = dict(info, reason=reason, t=utc(t=t), n=len(self.state["episodes"]) + 1)
        d = os.path.join(self.out, "hang-" + utc("%Y%m%dT%H%M%SZ", t))
        ep["evidence"] = d
        with open(os.path.join(self.out, "HANG"), "a") as f:  # first: the marker other tools poll for
            f.write(json.dumps(ep) + "\n")
        self.log("HANG " + json.dumps(ep))
        self.state["episodes"].append(ep)
        self.save_state()
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "reason.json"), "w") as f:
            json.dump(dict(ep, mk_tp_probe=self.state["mk_tp_probe"], last=self.state["last"]), f, indent=1)
        for name, path, to in (("metrics.txt", "/metrics", 8), ("health.txt", "/health", 8)):
            try:
                st, body = self.get(path, to)
                txt = f"HTTP {st}\n{body}"
            except Exception as e:  # noqa: BLE001
                txt = "failed: " + repr(e)
            with open(os.path.join(d, name), "w") as f:
                f.write(self.red(txt))
        procs = sglang_procs()
        with open(os.path.join(d, "procs.txt"), "w") as f:
            for p in procs:
                f.write(self.red(f"{p['pid']:>8} {p['ppid']:>8} {p['role']:<12} {p['comm']:<16} {p['cmd'][:400]}") + "\n")
        if self.pyspy:
            for p in procs:
                tag = f"pyspy-{p['role']}-{p['pid']}.txt"
                self.sh(d, tag, [self.pyspy, "dump", "--pid", str(p["pid"])], self.a.pyspy_timeout)
                if p["role"].startswith("tp") or p["role"] == "scheduler":
                    self.sh(d, tag.replace(".txt", "-native.txt"),
                            [self.pyspy, "dump", "--native", "--pid", str(p["pid"])], self.a.pyspy_timeout)
        else:
            with open(os.path.join(d, "pyspy-MISSING.txt"), "w") as f:
                f.write("py-spy unavailable\n")
        self.fault_dumps(d, procs)
        self.sh(d, "nvidia-smi.txt", "nvidia-smi; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,"
                "clocks.sm,temperature.gpu --format=csv", 30)
        with open(os.path.join(d, "serve-tail.txt"), "w") as f:
            f.write(self.red(self.tail(self.a.serve_log, 300)))
        self.sh(d, "ps.txt", "ps -eo pid,ppid,stat,etime,pcpu,rss,wchan:24,cmd --sort=-pcpu | head -80", 10)
        ep["evidence_done"] = utc()
        self.log(f"evidence done {d} ({int(time.time() - t)} s)")
        self.save_state()

    def fault_dumps(self, d, procs):
        """In-process stack dumps (see the docstring): only for ranks py-spy could not read, only when the rank catches
        the dump signal, one rank at a time so the dumps never interleave in the serve log."""
        sig = sig_num(self.a.dump_signal)
        if sig is None:
            return
        for p in procs:
            if not (p["role"].startswith("tp") or p["role"] == "scheduler"):
                continue
            spy = os.path.join(d, f"pyspy-{p['role']}-{p['pid']}.txt")
            try:
                with open(spy) as f:
                    spy_txt = f.read()
            except OSError:
                spy_txt = ""
            if self.pyspy and re.search(r"^Thread ", spy_txt, re.M):
                continue  # py-spy read its stacks: no signal needed
            name = os.path.join(d, f"faulthandler-{p['role']}-{p['pid']}.txt")
            if not catches(p["pid"], sig):
                with open(name, "w") as f:
                    f.write(f"not sent: pid {p['pid']} does not catch signal {sig} (SigCgt), so it would die of it\n")
                continue
            try:
                start = os.path.getsize(self.a.serve_log)
                os.kill(p["pid"], sig)
                time.sleep(self.a.dump_wait)
                with open(self.a.serve_log, "rb") as f:
                    f.seek(start)
                    txt = f.read(4 << 20).decode("utf-8", "replace")
                # keep the dump's own lines (Thread/Current thread headers and File lines), not the log noise around it
                keep = [ln for ln in txt.splitlines()
                        if re.match(r'\s*(File "|Thread 0x|Current thread 0x|<no Python frame>)', ln)]
                txt = f"signal {sig} sent to pid {p['pid']} ({p['role']}) at {utc()}\n" + "\n".join(keep) + "\n"
            except Exception as e:  # noqa: BLE001
                txt = "fault dump failed: " + repr(e) + "\n"
            with open(name, "w") as f:
                f.write(self.red(txt))
            self.log(f"faulthandler dump {p['role']} pid {p['pid']}: {len(txt.splitlines()) - 1} stack lines")

    # -- --on-hang kill
    def kill_record(self, rec):
        """One JSON line per decision in <out>/KILLS (killed or not, and why); kept in state.json too."""
        with open(os.path.join(self.out, "KILLS"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.state["kills"] = (self.state["kills"] + [rec])[-100:]
        self.save_state()

    def hold_fresh(self):
        """watchdog.sh's own rule: a deliberate restart holds the engine while its hold file is <= --hold-max-s old
        (and while it holds, the watchdog relaunches nothing, so a kill then would leave the engine down)."""
        h = self.a.hold_file
        if not h or not os.path.exists(h):
            return False
        try:
            return time.time() - os.path.getmtime(h) <= self.a.hold_max_s
        except OSError:
            return True

    def alive(self, rows, pgid, pids):
        """pids of `pgid`'s group (pgid set) or of `pids`, still running (zombies are dead)."""
        return [r["pid"] for r in rows if not r["stat"].startswith("Z")
                and (r["pgid"] == pgid if pgid is not None else r["pid"] in pids)]

    def on_hang_kill(self, reason, ep_n, stalled_s):
        """SIGTERM the engine's top-level serve process -- its whole process group when it leads one (relaunch.sh
        execs `sglang serve` under setsid, so it does), else it and its descendants -- then SIGKILL whatever is left
        after --on-hang-term-s. Returns True when a kill was sent. Refuses (logs, records, returns False) past the
        rate limit, under a fresh watchdog hold, with no serve process or more than one. Sets self.retry_at: inf after
        a kill (one per episode), the time the window has room again after rate_limited, else now +
        --on-hang-retry-s (a hold, a ps failure or a missing/extra serve process can clear while the stall lasts)."""
        a, now = self.a, time.time()
        rec = dict(t=utc(t=now), reason=reason, episode=ep_n, stalled_s=round(stalled_s, 1))
        self.kill_ts = [t for t in self.kill_ts if now - t < a.on_hang_window_s]
        retry_s = getattr(a, "on_hang_retry_s", 60.0)
        self.retry_at = now + retry_s
        if len(self.kill_ts) >= a.on_hang_max:
            # room again once the oldest kill that keeps the count at the max leaves the window
            self.retry_at = (self.kill_ts[len(self.kill_ts) - a.on_hang_max] + a.on_hang_window_s) if a.on_hang_max \
                else float("inf")
            self.log(f"ON-HANG KILL suppressed: {len(self.kill_ts)} kills in the last {a.on_hang_window_s:g} s "
                     f"(--on-hang-max {a.on_hang_max}); only logging this stall"
                     + (f" (next try in {self.retry_at - now:.0f} s if it lasts)" if a.on_hang_max else ""))
            self.kill_record(dict(rec, ts=now, outcome="rate_limited", recent=len(self.kill_ts),
                                  retry_in_s=round(self.retry_at - now, 1) if a.on_hang_max else None))
            return False
        if self.hold_fresh():
            self.log(f"ON-HANG KILL skipped: {a.hold_file} is fresh (a deliberate restart holds the engine); "
                     f"retry in {retry_s:g} s if the stall lasts")
            self.kill_record(dict(rec, ts=now, outcome="held", retry_in_s=retry_s))
            return False
        try:
            rows = ps_table()
        except Exception as e:  # noqa: BLE001
            self.log("ON-HANG KILL skipped: ps failed: " + repr(e)[:200] + f"; retry in {retry_s:g} s")
            self.kill_record(dict(rec, ts=now, outcome="ps_failed", retry_in_s=retry_s))
            return False
        tg = serve_targets(rows, a.serve_pattern, a.serve_port)
        if len(tg) != 1:
            what = "no serve process" if not tg else f"{len(tg)} serve processes {[r['pid'] for r in tg]}"
            self.log(f"ON-HANG KILL skipped: {what} match --serve-pattern {a.serve_pattern!r} port {a.serve_port}; "
                     f"retry in {retry_s:g} s if the stall lasts")
            self.kill_record(dict(rec, ts=now, outcome="no_target" if not tg else "ambiguous",
                                  pids=[r["pid"] for r in tg], retry_in_s=retry_s))
            return False
        self.retry_at = float("inf")
        self.kill_ts.append(now)
        p = tg[0]
        pid = p["pid"]
        # signal the group only when the serve process leads it and it is not this watcher's own group
        pgid = p["pgid"] if p["pgid"] == pid and p["pgid"] != os.getpgrp() else None
        pids = descendants(rows, pid) if pgid is None else set()
        targets = self.alive(rows, pgid, pids)
        self.log(f"ON-HANG KILL pid={pid} pgid={pgid if pgid is not None else '-'} at {utc(t=now)}: "
                 f"{reason} episode {ep_n} stalled {stalled_s:.0f} s; SIGTERM to "
                 + (f"process group {pgid}" if pgid is not None else "the process and its descendants")
                 + f" ({len(targets)} procs: {targets[:16]}); cmd: {p['args'][:300]}")
        rec.update(ts=now, pid=pid, pgid=pgid, procs=len(targets), cmd=self.red(p["args"][:300]))

        def send(sig):
            if pgid is not None:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
                return
            for q in pids:
                try:
                    os.kill(q, sig)
                except ProcessLookupError:
                    pass

        def wait(secs):
            end = time.time() + secs
            while True:
                left = self.alive(ps_table(), pgid, pids)
                if not left or time.time() >= end:
                    return left
                time.sleep(0.5)
        try:
            send(signal.SIGTERM)
            left = wait(a.on_hang_term_s)
            outcome = "term"
            if left:
                self.log(f"ON-HANG KILL pid={pid}: {len(left)} procs left {a.on_hang_term_s:g} s after SIGTERM "
                         f"({left[:16]}): SIGKILL")
                send(signal.SIGKILL)
                left = wait(10)
                outcome = "killed" if not left else "kill_failed"
        except Exception as e:  # noqa: BLE001 -- e.g. EPERM: record it, keep watching
            left, outcome = None, "kill_failed"
            rec["err"] = repr(e)[:300]
        rec.update(outcome=outcome, left=left, done=utc(), took_s=round(time.time() - now, 1))
        self.log(f"ON-HANG KILL pid={pid} done: {outcome} in {rec['took_s']:g} s"
                 + (f", still alive: {left[:16]}" if left else "") + "; watchdog.sh relaunches the engine")
        self.kill_record(rec)
        return True

    def serve_proc(self):
        """The one top-level serve process (ps row) or None (none, several, or ps failed)."""
        try:
            tg = serve_targets(ps_table(), self.a.serve_pattern, self.a.serve_port)
        except Exception:  # noqa: BLE001
            return None
        return tg[0] if len(tg) == 1 else None

    def run(self):
        a = self.a
        self.log(f"start pid={os.getpid()} url={self.base} stall_s={a.stall_s} interval={a.interval} "
                 f"key={'yes' if self.key else 'NO'} serve_log={a.serve_log} on_hang={a.on_hang}"
                 + (f" (after capture or {a.on_hang_after:g} s, max {a.on_hang_max}/{a.on_hang_window_s:g} s, "
                    f"TERM->KILL {a.on_hang_term_s:g} s, hold {a.hold_file})" if a.on_hang == "kill" else "")
                 + f" serve_pattern={a.serve_pattern!r} serve_port={a.serve_port} boot_grace_s={a.boot_grace_s:g}"
                 + (f" service (watch.log cap {a.log_max_mb} MB, max_captures {a.max_captures})" if a.service else ""))
        self.pyspy = find_pyspy(a.pyspy, not a.no_install, self.log)
        self.state["pyspy"] = self.pyspy
        self.log(f"py-spy: {self.pyspy}")
        # engine-life state that outlives a --service re-entry of the loop
        self.kill_t = None        # time of this watcher's last on-hang kill
        self.kill_healthy = True  # /health 200 seen since that kill
        self.serve_pid = None     # the top-level serve process last seen, and whether it has answered /health 200
        self.serve_healthy = False
        while True:
            try:
                self.loop()
                break
            except Exception as e:  # noqa: BLE001
                if not a.service:
                    raise
                self.log("ERROR in the watch loop (--service: re-entering it with a fresh stall clock): " + repr(e)[:500])
                time.sleep(max(1.0, a.interval))
        self.state["stopped"] = utc()
        self.save_state()
        self.log("stop file seen, exiting")

    def loop(self):
        a = self.a
        stop = os.path.join(self.out, "stop")
        last_sig = None
        idle_since = None
        prev_inflight = None
        last_prog = last_ok = time.time()
        hfail = 0
        last_h = last_s = 0.0
        in_episode = False
        ep_at = 0.0
        stall_since = None    # when the current episode's stall was first raised
        ep_captured = False   # its HANG capture has completed
        ep_retry = None       # --on-hang: None = not tried this episode, else when it may try again (inf: killed)
        booting_was = None
        mx60 = 0
        metrics_checked = False
        if a.health_every is None:
            a.health_every = a.interval
        while not os.path.exists(stop):
            now = time.time()
            self.state["polls"] += 1
            stall = None
            health_ok = False
            moving = False  # this poll's metrics moved with work in flight
            try:  # before /metrics, so this poll's serve-log batch lines count as this poll's progress
                self.scan_log()
            except Exception as e:  # noqa: BLE001
                self.log("serve-log scan failed: " + repr(e)[:200])
            try:
                st, txt = self.get("/metrics", a.metrics_timeout)
                if st != 200:
                    raise RuntimeError(f"/metrics HTTP {st}")
                last_ok = now
                self.state["metrics_ok"] += 1
                if not metrics_checked:
                    metrics_checked = True
                    found = {k: matched_name(txt, names) for k, names in
                             (("running", RUN_NAMES), ("queue", QUEUE_NAMES)) + PROGRESS}
                    self.state["metric_names"] = found
                    self.log("metrics matched: " + json.dumps(found))
                    inert = []
                    if found["running"] is None or all(found[k] is None for k, _ in PROGRESS):
                        inert = ["no_token", "queue_wedge"]
                    elif found["queue"] is None:
                        inert = ["queue_wedge"]
                    missing = [k for k in found if found[k] is None]
                    if inert:
                        self.state["metric_warning"] = f"missing {missing}: {'/'.join(inert)} stall detection is INERT"
                        self.log(f"WARNING: /metrics has no {missing} metric (looked for "
                                 f"{list(RUN_NAMES) + list(QUEUE_NAMES) + [n for _, ns in PROGRESS for n in ns]}): "
                                 f"{'/'.join(inert)} stall detection is INERT"
                                 + ("; only health/metrics_down stalls will fire" if "no_token" in inert else ""))
                run = first_metric(txt, RUN_NAMES) or 0.0
                q = first_metric(txt, QUEUE_NAMES) or 0.0
                ctr = {k: first_metric(txt, names) for k, names in PROGRESS}
                sb = self.state["serve_batches"]
                sig = tuple(ctr[k] for k, _ in PROGRESS) + (sb["prefill"], sb["decode"])
                inflight = run + q
                if inflight > 0:
                    idle_since = None
                elif idle_since is None:
                    idle_since = now
                idle = idle_since is not None and now - idle_since >= a.idle_hold_s
                # a new arrival after any zero reading starts the stall clock afresh: it must not inherit the age of
                # the progress before an idle gap shorter than --idle-hold-s
                arrival = inflight > 0 and prev_inflight == 0
                prev_inflight = inflight
                # any change counts (a counter that drops = engine restarted, which is not a stall either)
                progressed = last_sig is None or sig != last_sig or idle or arrival
                moving = inflight > 0 and last_sig is not None and sig != last_sig
                if progressed:
                    last_prog = now
                last_sig = sig
                mx60 = max(mx60, run)
                self.state["max_running"] = max(self.state["max_running"], run)
                self.state["last"] = dict(t=utc(t=now), running=run, queue=q, **ctr,
                                          serve_prefill_batches=sb["prefill"], serve_decode_batches=sb["decode"],
                                          idle_s=0.0 if idle_since is None else round(now - idle_since, 1),
                                          since_progress_s=round(now - last_prog, 1))
                if inflight > 0 and now - last_prog > a.stall_s:
                    stall = ("no_token" if run > 0 else "queue_wedge",
                             dict(running=run, queue=q, **ctr, serve_prefill_batches=sb["prefill"],
                                  serve_decode_batches=sb["decode"], stalled_s=round(now - last_prog, 1)))
                elif progressed and inflight > 0 and in_episode:
                    in_episode = False
                    stall_since, ep_captured, ep_retry = None, False, None
                    self.log("progress resumed")
            except Exception as e:  # noqa: BLE001
                self.state["metrics_fail"] += 1
                if now - last_ok > a.stall_s:
                    stall = ("metrics_down", dict(err=self.red(repr(e))[:300], since_s=round(now - last_ok, 1)))
            if now - last_h >= a.health_every:
                last_h = now
                try:
                    st, _ = self.get("/health", a.health_timeout)
                    ok = st == 200
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    self.state["health_ok"] += 1
                    hfail = 0
                    health_ok = True
                    if in_episode and stall is None and last_ok == now and idle_since is not None \
                            and now - idle_since >= a.idle_hold_s:
                        in_episode = False
                        stall_since, ep_captured, ep_retry = None, False, None
                        self.log("idle and healthy again")
                else:
                    self.state["health_fail"] += 1
                    hfail += 1
                if hfail >= a.health_fails and stall is None:
                    if moving:  # /health is a real generation queued behind user work: busy, not dead (watchdog.sh)
                        if hfail == a.health_fails:
                            self.log(f"/health failed {hfail}x while the engine makes progress with work in flight: "
                                     "busy, not a stall")
                    else:
                        stall = ("health", dict(consecutive_fail=hfail))
            # BOOT: an engine that has not answered /health 200 since this watcher killed it, or whose serve process
            # is younger than --boot-grace-s and has not answered /health 200 yet (a watchdog relaunch or a first
            # boot: 10-27 min on box-a), is loading, not hung: no HANG, no kill, every stall clock held at zero.
            serve = self.serve_proc()
            if serve is not None and serve["pid"] != self.serve_pid:
                self.serve_pid, self.serve_healthy = serve["pid"], False
                self.log(f"serve process pid={serve['pid']} pgid={serve['pgid']} age {serve['age_s']} s: "
                         + self.red(serve["args"][:200]))
            if health_ok:
                self.serve_healthy = self.kill_healthy = True
            booting = None
            if self.kill_t is not None and not self.kill_healthy and now - self.kill_t < a.boot_grace_s:
                booting = f"killed by --on-hang {now - self.kill_t:.0f} s ago, no /health 200 since"
            elif serve is not None and not self.serve_healthy and serve["age_s"] < a.boot_grace_s:
                booting = f"serve pid {serve['pid']} is {serve['age_s']} s old, no /health 200 yet"
            if booting:
                if not booting_was:
                    self.log(f"engine booting ({booting}): HANG and --on-hang held until its /health 200, at most "
                             f"--boot-grace-s {a.boot_grace_s:g} s" + (f" (ignored: {stall[0]})" if stall else ""))
                stall = None
                last_prog = last_ok = now
                hfail, last_sig, idle_since, prev_inflight = 0, None, None, None
                in_episode, stall_since, ep_captured, ep_retry = False, None, False, None
            elif booting_was:
                self.log("engine up: /health 200, stall detection resumes" if health_ok else
                         f"boot grace over ({booting_was}; --boot-grace-s {a.boot_grace_s:g} s) without /health 200: "
                         "stall detection resumes")
            booting_was = booting
            self.state["booting"] = booting
            captured_now = False
            if stall and (not in_episode or now - ep_at > a.recapture_s):
                if not in_episode:
                    stall_since = now
                if len(self.state["episodes"]) < a.max_captures:
                    in_episode, ep_at = True, now
                    self.capture(*stall)
                    captured_now = ep_captured = True
                elif not in_episode:
                    in_episode, ep_at = True, now
                    self.log("stall (capture cap reached): " + json.dumps(dict(stall[1], reason=stall[0])))
            # --on-hang kill: on a LATER poll than the capture, the stall still on (so the evidence is on disk and the
            # stall outlived it), or --on-hang-after s into an episode that took no capture; a refusal that can clear
            # (hold, ps, serve-process match, rate limit) is tried again at ep_retry while the stall lasts
            if a.on_hang == "kill" and stall and in_episode and (ep_retry is None or now >= ep_retry) \
                    and not captured_now and (ep_captured or now - stall_since >= a.on_hang_after):
                n = self.state["episodes"][-1]["n"] if ep_captured else None
                killed = self.on_hang_kill(stall[0], n, now - stall_since)
                ep_retry = self.retry_at
                if killed:
                    self.kill_t, self.kill_healthy = time.time(), False
            if now - last_s >= 60:
                last_s = now
                pr = self.state["mk_tp_probe"]
                lst = self.state["last"] or {}
                self.log(f"ok run_max60={mx60:g} realtime={lst.get('realtime_tokens')} gen={lst.get('gen_tokens')} "
                         f"prompt={lst.get('prompt_tokens')} batches={lst.get('serve_prefill_batches')}/"
                         f"{lst.get('serve_decode_batches')} episodes={len(self.state['episodes'])} "
                         f"probe_ok={pr['ok']} probe_mismatch={pr['mismatch']} health_fail={self.state['health_fail']}")
                mx60 = 0
            self.save_state()
            time.sleep(max(0.2, a.interval - (time.time() - now)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir (HANG, hang-*/, state.json, watch.log; stop file)")
    ap.add_argument("--url", default=os.environ.get("ENGINE_URL", "http://127.0.0.1:" + os.environ.get("ENGINE_PORT", "8002")))
    ap.add_argument("--serve-log", default=os.environ.get("SERVE_LOG", "/workspace/sglang-serve.log"))
    ap.add_argument("--interval", type=float, default=3.0, help="metrics poll period, s (2-5)")
    ap.add_argument("--stall-s", type=float, default=60.0,
                    help="s in flight with no progress before a no_token/queue_wedge stall (> the longest prefill batch)")
    ap.add_argument("--idle-hold-s", type=float, default=35.0,
                    help="s a zero in-flight reading must hold before it counts as idle (resets the stall clock)")
    ap.add_argument("--health-every", type=float, default=None, help="s between /health checks (default --interval)")
    ap.add_argument("--health-timeout", type=float, default=5.0)
    ap.add_argument("--health-fails", type=int, default=2)
    ap.add_argument("--metrics-timeout", type=float, default=5.0)
    ap.add_argument("--recapture-s", type=float, default=600.0, help="re-capture a stall that is still on after this")
    ap.add_argument("--max-captures", type=int, default=None, help="evidence captures per run (default 5; 50 with --service)")
    ap.add_argument("--pyspy", default="", help="py-spy binary (default /opt/pyspy/bin/py-spy, then PATH)")
    ap.add_argument("--pyspy-timeout", type=float, default=20.0)
    ap.add_argument("--no-install", action="store_true", help="never create /opt/pyspy")
    # Default none: SigCgt shows USR2 caught on our engines' ranks, but the catcher is LLVM's SA_RESETHAND handler
    # (Triton), which resets USR2 to the default and re-raises, so the rank dies of the dump
    # (qa/sglang-takes-20261104/merge-b300/NOTES.md, M1/M1b). Pass USR2 only for an engine whose faulthandler site
    # file (merge-b300/site/sitecustomize.py) registers first.
    ap.add_argument("--dump-signal", default="none",
                    help="in-process stack-dump signal for ranks py-spy cannot read (sent only when caught); "
                         "default none = off; USR2 only where a faulthandler site file registers it first")
    ap.add_argument("--dump-wait", type=float, default=2.0, help="s to wait for a rank's dump to reach the serve log")
    ap.add_argument("--keep-mismatch", type=int, default=200, help="MISMATCH lines kept in mk_tp_probe_mismatch.txt")
    ap.add_argument("--scan-from-start", action="store_true", help="count MK_TP_PROBE lines already in the log too")
    ap.add_argument("--on-hang", choices=("none", "kill"), default="none",
                    help="none: never kill (default). kill: once a stall's capture is done (or --on-hang-after s into "
                         "one that took none) and it is still on, SIGTERM then SIGKILL the serve process group so "
                         "watchdog.sh relaunches it")
    ap.add_argument("--on-hang-after", type=float, default=120.0,
                    help="s of continuous stall before a kill when the episode took no capture (capture cap reached)")
    ap.add_argument("--on-hang-max", type=int, default=3, help="kills at most per --on-hang-window-s; then only log")
    ap.add_argument("--on-hang-window-s", type=float, default=3600.0)
    ap.add_argument("--on-hang-term-s", type=float, default=20.0, help="s from SIGTERM to SIGKILL")
    ap.add_argument("--on-hang-retry-s", type=float, default=60.0,
                    help="s before a refused kill (held / no_target / ambiguous / ps_failed) is tried again while the "
                         "stall lasts")
    ap.add_argument("--boot-grace-s", type=float, default=2700.0,
                    help="an engine with no /health 200 since our kill, or a serve process younger than this with no "
                         "/health 200 yet, is booting: no HANG, no kill (box-a boots in 10-27 min; watchdog.sh's 2700)")
    ap.add_argument("--serve-pattern", default=SERVE_PATTERN,
                    help="regex on ps args naming the engine's top-level serve process")
    ap.add_argument("--serve-port", type=int, default=None,
                    help="only a serve process whose --port is this (default: --url's port; one without --port matches)")
    ap.add_argument("--hold-file", default=os.path.join(os.environ.get("TOOLS_REMOTE", "/workspace/tools"), "watchdog.hold"),
                    help="no kill while this is fresh (a deliberate restart; watchdog.sh relaunches nothing then)")
    ap.add_argument("--hold-max-s", type=float, default=900.0, help="watchdog.sh's HOLD_MAX: an older hold is stale")
    ap.add_argument("--service", action="store_true",
                    help="long-running mode beside watchdog.sh: one instance per --out (hangwatch.pid), a stale stop "
                         "file removed at start, watch.log rotated (--log-max-mb), a loop error logged not fatal, "
                         "--max-captures 50")
    ap.add_argument("--log-max-mb", type=int, default=None,
                    help="cap on watch.log + watch.log.1 (default 100 with --service, else unbounded)")
    a = ap.parse_args(argv)
    try:  # validate now: fault_dumps first calls sig_num after the HANG marker, mid-capture
        sig = sig_num(a.dump_signal)
    except ValueError as e:
        ap.error(f"--dump-signal: {e} (a signal name such as USR2, a number, or none)")
    if sig in (signal.SIGKILL, signal.SIGSTOP):
        ap.error(f"--dump-signal: {a.dump_signal} cannot be caught, so no rank could ever take it")
    if a.dump_wait < 0:
        ap.error("--dump-wait must be >= 0")
    if a.on_hang_max < 0 or a.on_hang_after < 0 or a.on_hang_term_s < 0 or a.boot_grace_s < 0:
        ap.error("--on-hang-max/--on-hang-after/--on-hang-term-s/--boot-grace-s must be >= 0")
    if a.on_hang_retry_s <= 0:
        ap.error("--on-hang-retry-s must be > 0")
    try:
        re.compile(a.serve_pattern)
    except re.error as e:
        ap.error(f"--serve-pattern: {e}")
    if a.serve_port is None:
        a.serve_port = urllib.parse.urlsplit(a.url).port
    if a.max_captures is None:
        a.max_captures = 50 if a.service else 5
    if a.log_max_mb is None:
        a.log_max_mb = 100 if a.service else 0
    if a.service:
        os.makedirs(a.out, exist_ok=True)
        pidf = os.path.join(a.out, "hangwatch.pid")
        try:
            old = int(open(pidf).read().split()[0])
        except (OSError, ValueError, IndexError):
            old = None
        if old and old != os.getpid():
            try:
                rows = [r for r in ps_table() if r["pid"] == old and not r["stat"].startswith("Z")]
            except Exception:  # noqa: BLE001
                rows = []
            if rows and "hangwatch" in rows[0]["args"]:
                print(f"hangwatch --service already runs on {a.out} (pid {old}); `touch {a.out}/stop` ends it",
                      file=sys.stderr)
                return 1
        with open(pidf, "w") as f:
            f.write(f"{os.getpid()}\n")
    stale = False
    if a.service and os.path.exists(os.path.join(a.out, "stop")):
        os.remove(os.path.join(a.out, "stop"))
        stale = True
    try:
        w = Watch(a)
        if stale:
            w.log("removed a stale stop file (--service start)")
        if w.kill_ts:
            w.log(f"rate limit seeded from KILLS: {len(w.kill_ts)} kills in the last {a.on_hang_window_s:g} s")
        return w.run()
    finally:
        if a.service:  # ours only: a second instance that raced us to the file keeps its own
            try:
                if open(pidf).read().split()[0] == str(os.getpid()):
                    os.remove(pidf)
            except (OSError, IndexError):
                pass


if __name__ == "__main__":
    sys.exit(main())
