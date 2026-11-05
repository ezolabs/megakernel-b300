#!/usr/bin/env python3
"""Short in-process warm, run on EVERY engine boot before the front door opens (#379).

A fresh `sglang serve` process pays one-time in-memory costs on its first real requests: Triton
autotune results, cuBLASLt heuristics and pinned staging buffers live in the process, so the disk
compile caches (DeepGEMM cubins, torch extensions -- what prewarm-box-d.py sweeps, once per tree) do not
cover them. On 2026-10-27 they showed as 1.6-2.2 s first-use stalls (a 1.63 s cold fill, a 2183 ms
extend against 249-261 ms steady, a 612 ms background gap) that the box-c final's pre-un-drain cold fill
and three extends absorbed by hand. A watchdog relaunch boots a fresh process too, and nobody runs
anything by hand then, so the boot path itself runs this warm.

What it sends, on the loopback engine with the engine's own key, every request /v1/chat/completions:
  coldfill  ONE cold prompt of about --fill-words words (default 60000, ~60k tokens), its first line a
            fresh nonce so nothing hits the radix cache;
  extend    three turns on that conversation, each adding about --extend-words words (default 700), so
            each is a cached-prefix extend of the one before;
  image     ONE image turn on smoke.py's fixture frame (smoke_frame()), so the vision path's first use
            is paid here too.
Every step logs its wall time, prompt tokens and cached tokens (when the engine reports them).

--gate FLAG is the boot-path mode. relaunch.sh writes FLAG (`pid=<its own pid>` + `owner=relaunch`)
just before it execs into `sglang serve`, then starts this script double-forked with --engine-pid set
to that same pid. While FLAG exists the front door answers 503 (Caddy's @warming matcher, tls.sh), so
the serving router sees no READY backend. This script:
  1. takes the flag over (rewrites it with its own pid, owner=warm-short), so watchdog.sh can tell a
     live warm from a stale flag;
  2. waits for the engine's /health 200, exiting with the flag left CLOSED if the engine process dies
     first (the next launch rewrites it and starts its own warm; watchdog.sh clears a flag whose pid
     is gone);
  3. runs the warm under a hard --timeout (default 180 s, counted from /health 200);
  4. removes FLAG -- after the warm, or at the timeout with a WARNING line -- but only while FLAG
     still names this process (a newer launch's flag is never touched).
Exit: 0 warm done; 1 warm failed or timed out (gate opened anyway); 3 engine died before /health;
4 the flag was taken by a newer launch; 2 usage.

Without --gate it is a one-shot warm against --base-url (default boxenv's engine URL): exit 0 or 1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORDS = (
    "river stone garden window paper morning yellow silver market engine harbor winter quiet "
    "orange forest signal bridge copper letter ladder pocket thunder candle mirror valley rocket "
    "button marble planet shadow island basket violet meadow lantern feather pencil cobalt "
    "anchor saddle velvet harvest compass tunnel blanket falcon pepper canvas granite meteor "
    "puzzle ribbon summer timber walnut zephyr acorn beacon cactus dolphin ember fiddle glacier "
    "hammer igloo jasmine kettle lemon magnet nectar oyster parrot quartz raven spruce tulip "
    "umbrella vessel willow yarn zinc amber breeze cedar delta echo fable garnet hazel iris "
    "juniper kayak lotus maple nickel olive prism quill rapids sable topaz upland vapor wheat"
).split()

CURL_UA = "box-c-warm-short/1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def read_key(environment_file: str | None, cli_key: str | None) -> str:
    """--api-key, else VLLM_API_KEY from --environment-file (the file relaunch.sh gave the engine),
    else boxenv's resolution. The key never rides argv from relaunch.sh: it passes the file's path."""
    if cli_key:
        return cli_key
    if environment_file and os.path.isfile(environment_file):
        with open(environment_file) as f:
            for line in f:
                if line.startswith("VLLM_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    import boxenv  # tools/boxenv.py, beside this file
    return boxenv.resolve_api_key(None)


class Engine:
    def __init__(self, base: str, key: str | None):
        self.base = base.rstrip("/")
        self.key = key

    def _req(self, path: str, body: dict | None, timeout: float) -> tuple[int, bytes]:
        headers = {"User-Agent": CURL_UA}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, timeout)) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def health(self) -> int:
        try:
            return self._req("/health", None, 5)[0]
        except Exception:
            return 0

    def model(self, timeout: float) -> str:
        code, raw = self._req("/v1/models", None, timeout)
        if code != 200:
            raise RuntimeError(f"/v1/models -> {code}")
        return json.loads(raw)["data"][0]["id"]

    def chat(self, model: str, messages: list, max_tokens: int, timeout: float) -> dict:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "stream": False}
        code, raw = self._req("/v1/chat/completions", body, timeout)
        if code != 200:
            raise RuntimeError(f"/v1/chat/completions -> {code}: {raw[:200]!r}")
        return json.loads(raw)


def usage_str(resp: dict) -> str:
    u = resp.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return f"prompt_tokens={u.get('prompt_tokens', '?')} cached={cached if cached is not None else '?'}"


def image_url() -> str | None:
    try:
        import smoke  # tools/smoke.py: the fixture frame the smoke/quality checks send
        return smoke.smoke_frame()
    except Exception as e:  # smoke.py not staged beside this file: skip the turn, loudly
        log(f"WARNING: image turn skipped -- smoke.smoke_frame() unavailable: {e}")
        return None


def warm(eng: Engine, a, deadline: float) -> None:
    """The warm itself: raises on any failed request."""
    # Each request's socket timeout runs 5 s past the deadline, so --gate's join(timeout) is what ends
    # an overrun warm (and reports it as one), never a socket timeout racing it.
    left = lambda: deadline - time.monotonic() + 5  # noqa: E731
    model = a.model or eng.model(left())
    rng = random.Random(uuid.uuid4().int)
    nonce = f"warm-short {uuid.uuid4()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    log(f"warm start model={model} fill_words={a.fill_words} extend_words={a.extend_words} x3 + image")
    total = time.monotonic()

    msgs = [{"role": "user", "content": f"{nonce}\n{words(rng, a.fill_words)}\nReply with one word."}]
    t = time.monotonic()
    r = eng.chat(model, msgs, a.max_tokens, left())
    log(f"coldfill {time.monotonic() - t:.2f}s {usage_str(r)}")

    for i in range(1, 4):
        msgs.append({"role": "assistant", "content": "ok."})
        msgs.append({"role": "user", "content": f"{words(rng, a.extend_words)}\nReply with one word."})
        t = time.monotonic()
        r = eng.chat(model, msgs, a.max_tokens, left())
        log(f"extend {i}/3 {(time.monotonic() - t) * 1000:.0f}ms {usage_str(r)}")

    url = image_url()
    if url is not None:
        img = [{"role": "user", "content": [
            {"type": "text", "text": f"{nonce} image. What shape and colour is in the picture? One line."},
            {"type": "image_url", "image_url": {"url": url}}]}]
        t = time.monotonic()
        r = eng.chat(model, img, a.max_tokens, left())
        log(f"image {(time.monotonic() - t) * 1000:.0f}ms {usage_str(r)}")
    log(f"warm done in {time.monotonic() - total:.2f}s")


# ------------------------------------------------------------------------------------------- gate

def read_flag(path: str) -> dict:
    try:
        with open(path) as f:
            return dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
    except OSError:
        return {}


def write_flag(path: str, fields: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in fields.items()))
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def open_gate(path: str) -> bool:
    """Remove the flag only while it still names this process; False when a newer launch owns it."""
    if read_flag(path).get("pid") != str(os.getpid()):
        return False
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return True


def gate(eng: Engine, a) -> int:
    me = str(os.getpid())
    held = read_flag(a.gate)
    if held.get("pid") != str(a.engine_pid):
        log(f"WARNING: gate {a.gate} does not name engine pid {a.engine_pid} (holds {held or 'nothing'}) "
            f"-- a newer launch owns it; this warm exits without touching it")
        return 4
    write_flag(a.gate, {"pid": me, "owner": "warm-short", "engine": a.engine_pid,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"gate CLOSED ({a.gate}); waiting for engine pid {a.engine_pid} /health at {eng.base}")

    t0 = time.monotonic()
    while eng.health() != 200:
        if not pid_alive(a.engine_pid):
            log(f"engine pid {a.engine_pid} gone before /health 200 -- gate left closed for the next launch")
            return 3
        if read_flag(a.gate).get("pid") != me:
            log("gate taken by a newer launch while waiting for /health -- exiting")
            return 4
        time.sleep(a.poll)
    log(f"engine /health 200 after {time.monotonic() - t0:.0f}s; short warm, timeout {a.timeout:.0f}s")

    deadline = time.monotonic() + a.timeout
    err: list[BaseException] = []

    def run() -> None:
        try:
            warm(eng, a, deadline)
        except BaseException as e:  # noqa: BLE001 -- reported below, the gate opens either way
            err.append(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(a.timeout)
    if th.is_alive():
        rc, why = 1, f"WARNING: short warm did not finish in {a.timeout:.0f}s -- gate opened anyway"
    elif err:
        rc, why = 1, f"WARNING: short warm failed ({err[0]}) -- gate opened anyway"
    else:
        rc, why = 0, "short warm done"
    if open_gate(a.gate):
        log(f"{why}; gate OPEN ({a.gate} removed) {time.monotonic() - t0:.0f}s after the wait began")
    else:
        log(f"{why}; gate left alone -- a newer launch owns {a.gate}")
    sys.stdout.flush()
    os._exit(rc)  # a timed-out warm thread is still blocked in a socket read: do not wait for it


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=None, help="engine origin; default boxenv's engine_url()")
    p.add_argument("--api-key", default=None, help="default VLLM_API_KEY from --environment-file, then boxenv")
    p.add_argument("--environment-file", default=None, help="KEY=VALUE file holding VLLM_API_KEY (relaunch.sh's)")
    p.add_argument("--model", default=None, help="served model; default the engine's /v1/models first id")
    p.add_argument("--fill-words", type=int, default=60000, help="cold fill size in words (~tokens)")
    p.add_argument("--extend-words", type=int, default=700, help="each extend's added words (~tokens)")
    p.add_argument("--max-tokens", type=int, default=16, help="max_tokens per request")
    p.add_argument("--timeout", type=float, default=180.0, help="whole-warm deadline (s), from /health 200")
    p.add_argument("--gate", default=None, help="boot mode: the warm gate flag to take over and remove")
    p.add_argument("--engine-pid", type=int, default=None, help="with --gate: the engine pid the flag names")
    p.add_argument("--poll", type=float, default=2.0, help="with --gate: /health poll interval (s)")
    a = p.parse_args()
    if a.gate and not a.engine_pid:
        p.error("--gate needs --engine-pid")

    base = a.base_url
    if not base:
        import boxenv
        base = boxenv.resolve_engine_url(None)
    try:
        key = read_key(a.environment_file, a.api_key)
    except Exception as e:  # no key: the warm's requests fail on 401, and --gate still opens the door
        log(f"WARNING: no API key resolved ({e}); requests go unauthenticated")
        key = None
    eng = Engine(base, key)

    if a.gate:
        return gate(eng, a)
    try:
        warm(eng, a, time.monotonic() + a.timeout)
        return 0
    except Exception as e:
        log(f"WARNING: short warm failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
