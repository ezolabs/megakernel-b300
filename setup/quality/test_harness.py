#!/usr/bin/env python3
"""Offline tests for synthmix.py and hangwatch.py against a stdlib fake OpenAI-compatible engine (no network beyond
127.0.0.1, no real engine). Run on the Mac:  python3 test_harness.py  [-v]   (~60 s)

The fake engine streams SSE chat completions (reasoning_content, content, tool_calls split across chunks with every 9th
one malformed, usage with cached_tokens, an HTTP 500 every 37th streaming request), answers non-streaming calls (greedy probe),
and serves /metrics (sglang names, a tp_rank="1" decoy series) and /health. Switches: stall (running>0, every counter
flat), prefill (running>0, generation flat; the test moves prompt_tokens_total / realtime_tokens_total / the serve log by
hand), idle (nothing in flight, nothing moves), queued (queue>0, running 0, nothing moves), health (503), healthhang
(/health sleeps 10 s), nometrics (/metrics without the running/progress series), down (/metrics and /health 503).
The --on-hang kill tests stand in for `sglang serve` with a sleeper (its own session, like relaunch.sh's setsid exec,
with a `sleep` child in its group) whose argv carries a unique marker and --port."""
import contextlib, io, json, os, re, signal, subprocess, sys, tempfile, threading, time, types, unittest
from unittest import mock
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = "testkey-" + "Q7" * 10  # a fake engine key; every output must redact it
os.environ["VLLM_API_KEY"] = KEY
for _n in ("SGLANG_API_KEY", "API_KEY"):
    os.environ.pop(_n, None)
import synthmix  # noqa: E402
import hangwatch  # noqa: E402

STATE = {"gen": 0, "prompt": 0, "rt": 0, "mode": "ok", "running": 0, "n": 0}
LOCK = threading.Lock()


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        mode = STATE["mode"]
        if mode == "down":  # a dead or booting engine: nothing answers 200
            return self._send(503, "")
        if self.path == "/metrics":
            if mode == "nometrics":
                return self._send(200, "# nothing useful\nprocess_cpu_seconds_total 1.0\n", "text/plain")
            q = 0
            if mode in ("stall", "prefill"):
                run = 3 if mode == "stall" else 1
            elif mode == "idle":
                run = 0
            elif mode == "queued":
                run, q = 0, 2
            else:
                STATE["gen"] += 7
                if STATE["rt"] is not None:
                    STATE["rt"] += 7
                run = STATE["running"]
            t = (f'sglang:num_running_reqs{{tp_rank="0"}} {run}\nsglang:num_running_reqs{{tp_rank="1"}} 99\n'
                 f'sglang:num_queue_reqs{{tp_rank="0"}} {q}\nsglang:generation_tokens_total{{tp_rank="0"}} {STATE["gen"]}\n'
                 f'sglang:prompt_tokens_total{{tp_rank="0"}} {STATE["prompt"]}\n'
                 f'sglang:realtime_tokens_total_created{{mode="decode",tp_rank="0"}} {time.time()}\n')
            if STATE["rt"] is not None:  # mode-labelled per-step counter; tp_rank="1" decoy moves every scrape
                t += (f'sglang:realtime_tokens_total{{mode="prefill_compute",tp_rank="0"}} {STATE["rt"]}\n'
                      f'sglang:realtime_tokens_total{{mode="decode",tp_rank="0"}} 5\n'
                      f'sglang:realtime_tokens_total{{mode="decode",tp_rank="1"}} {time.time()}\n')
            return self._send(200, t, "text/plain")
        if self.path == "/health":
            if mode == "healthhang":
                time.sleep(10)
            return self._send(503 if mode == "health" else 200, "")
        self._send(404, "")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.headers.get("Authorization") != "Bearer " + KEY:
            return self._send(401, '{"error":"bad key"}')
        with LOCK:
            STATE["n"] += 1
            n = STATE["n"]
        if n % 37 == 0 and body.get("stream"):  # the greedy probe (non-streaming) never gets one
            return self._send(500, '{"error":"boom"}')
        msgs = body["messages"]
        last = msgs[-1].get("content") or ""
        m = re.search(r"(\d+) \* (\d+)", last) or re.search(r"(\d+) times (\d+)", last)
        content = f"The answer is {int(m.group(1)) * int(m.group(2)):,}." if m else "Here is a short answer."
        mm = re.search(r"code of unit (\S+)\?", last)
        if mm:
            content = re.search(r"unit " + re.escape(mm.group(1)) + r" is (\d+)", last).group(1)
        if body.get("response_format"):
            content = json.dumps({"name": "X", "city": "Y", "year": int(re.search(r"in (\d{4})", last).group(1)),
                                  "tags": ["a"]})
        think = body.get("reasoning_effort") != "none"
        if not body.get("stream"):
            return self._send(200, json.dumps({"choices": [{"message": {
                "role": "assistant", "reasoning_content": "r" if think else None, "content": "greedy " + last[:20]},
                "finish_reason": "stop"}]}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def ev(o):
            d = ("data: " + json.dumps(o) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d))
            self.wfile.flush()
        with LOCK:
            STATE["running"] += 1
        ntok = 0
        try:
            time.sleep(0.01)
            if think:
                for w in ("Let", " me", " think."):
                    ev({"choices": [{"index": 0, "delta": {"reasoning_content": w}}]})
                    ntok += 1
            if body.get("tools") and n % 4:
                args = json.dumps({"path": "src/a.py"}) if n % 9 else '{"path": "src/a.py"'
                ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                                       "function": {"name": "read_file", "arguments": ""}}]}}]})
                for i in range(0, len(args), 5):
                    ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 5]}}]}}]})
                    ntok += 1
                fin = "tool_calls"
            else:
                for i in range(0, len(content), 4):
                    ev({"choices": [{"index": 0, "delta": {"content": content[i:i + 4]}}]})
                    ntok += 1
                fin = "stop"
            ev({"choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
            ev({"choices": [], "usage": {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": ntok,
                                         "prompt_tokens_details": {"cached_tokens": 5}}})
            d = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(d), d))
            self.wfile.flush()
        finally:
            with LOCK:
                STATE["running"] -= 1


class Server(ThreadingHTTPServer):
    request_queue_size = 128  # the default backlog of 5 resets connections at C16 on older Pythons


SRV = Server(("127.0.0.1", 0), Fake)
SRV.daemon_threads = True
URL = "http://127.0.0.1:%d" % SRV.server_address[1]
threading.Thread(target=SRV.serve_forever, daemon=True).start()


def wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


SLEEPER = r'''
import os, signal, subprocess, sys, time
d, mode = sys.argv[1], sys.argv[2]
kid = subprocess.Popen(["sleep", "600"])  # in the sleeper's process group, like the engine's TP ranks
open(os.path.join(d, "kid"), "w").write(str(kid.pid))
def term(*_):
    open(os.path.join(d, "term"), "w").write("TERM")
    if mode == "exit":
        kid.wait()
        sys.exit(0)
signal.signal(signal.SIGTERM, term)  # mode "ignore": notes the TERM and keeps running (SIGKILL's case)
open(os.path.join(d, "ready"), "w").close()
while True:
    time.sleep(1)
'''


def sleeper(marker, port, mode="exit"):
    """A stand-in for `sglang serve`: own session/process group, argv `... <marker> --port <port>`; reaped by a thread
    so a dead one leaves no zombie. Returns (Popen, its dir)."""
    d = tempfile.mkdtemp(prefix="sleeper-")
    p = subprocess.Popen([sys.executable, "-c", SLEEPER, d, mode, marker, "--port", str(port)],
                         start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=p.wait, daemon=True).start()
    assert wait_for(lambda: os.path.exists(os.path.join(d, "ready")), 15), "sleeper did not start"
    return p, d


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


def reap(*ps):
    for p in ps:
        if p.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)


class SynthmixTest(unittest.TestCase):
    def setUp(self):
        STATE["mode"] = "ok"
        self.out = tempfile.mkdtemp(prefix="synthmix-")

    def test_sample_is_deterministic_and_covers_kinds(self):
        r1 = subprocess.run([sys.executable, os.path.join(HERE, "synthmix.py"), "sample", "--n", "40",
                             "--sample-streams", "3", "--long-max", "20000"], capture_output=True, text=True, check=True)
        r2 = subprocess.run([sys.executable, os.path.join(HERE, "synthmix.py"), "sample", "--n", "40",
                             "--sample-streams", "3", "--long-max", "20000"], capture_output=True, text=True, check=True)
        self.assertEqual(r1.stdout, r2.stdout)
        rows = [json.loads(l) for l in r1.stdout.splitlines()]
        self.assertEqual({r["kind"] for r in rows}, set(synthmix.KINDS))
        self.assertTrue(all(128 <= r["max_tokens"] <= 2048 for r in rows))
        self.assertTrue(any(r["thinking"] for r in rows) and any(not r["thinking"] for r in rows))

    def test_long_doc_size(self):
        import random
        for tgt in (8000, 120000):
            doc, facts = synthmix.gen_doc(random.Random(tgt), tgt)
            est = len(doc) / 4.3
            self.assertTrue(0.95 * tgt <= est <= 1.1 * tgt, (tgt, est))
            for unit, code in facts:
                self.assertIn(f"unit {unit} is {code}", doc)

    def test_soak_with_idle_gaps(self):
        rc = synthmix.main(["soak", "--out", self.out, "--url", URL, "--duration", "24", "--streams", "1,4,16",
                            "--phase-s", "6", "--idle-prob", "1", "--idle-min", "1", "--idle-max", "2",
                            "--think-max", "0.1", "--long-max", "20000", "--error-backoff-s", "0.2",
                            "--max-error-rate", "0.1"])  # the fake injects a 500 every 37th (2.7%)
        self.assertEqual(rc, 0)
        S = json.load(open(os.path.join(self.out, "summary.json")))
        A = S["all"]
        self.assertTrue(S["gate"]["ok"])
        self.assertGreater(S["gate"]["fail_rate"], 0)
        self.assertFalse(synthmix.gate(S, 0.0)["ok"], "the default-strict gate must fail a run with errors")
        print("\nsoak summary:", json.dumps({k: A[k] for k in ("n", "errs", "hangs", "empty", "bad_marker", "finish",
                                                             "ttft_p50", "ttft_p99", "tps_p50", "tool_parse_ok",
                                                             "answer_ok", "json_ok")}),
              "| by_C n:", {c: v["n"] for c, v in S["by_C"].items()}, "| idle phases:",
              sum(1 for p in S["phases"] if "idle_s" in p), "| errors:", S["error_kinds"])
        self.assertGreater(A["n"], 100)
        self.assertEqual(set(S["by_C"]), {"1", "4", "16"})
        self.assertTrue(any("idle_s" in p for p in S["phases"]))
        self.assertEqual(S["harness_errors"], 0)
        # every error is the fake's injected 500; malformed tool calls are caught, correct answers are scored right
        self.assertTrue(S["error_kinds"] and all("boom" in k for k in S["error_kinds"]), S["error_kinds"])
        ok, n = map(int, A["tool_parse_ok"].split("/"))
        self.assertTrue(0 < ok < n)
        # denominators are every request sent with the check: the fake's 500s land in `error`, nothing else misses
        for kind, key, outs in (("json", "json_ok", "json_outcomes"), ("qa", "answer_ok", "answer_outcomes")):
            B = S["by_kind"][kind]
            ok, n = map(int, B[key].split("/"))
            self.assertGreater(n, 0)
            self.assertEqual(sum(B[outs].values()), n, (kind, B[outs]))
            self.assertEqual(B[outs]["ok"], ok)
            self.assertEqual(B[outs]["ok"] + B[outs]["error"], n, (kind, B[outs]))
            self.assertEqual(B[outs]["error"], sum(1 for r in [json.loads(l) for l in open(
                os.path.join(self.out, "requests.jsonl"))] if r["kind"] == kind and r["err"]
                and ("answer_ok" if kind == "qa" else "json_ok") in r))
        self.assertEqual(A["bad_marker"], 0)
        recs = [json.loads(l) for l in open(os.path.join(self.out, "requests.jsonl"))]
        self.assertTrue(all(r["ttft"] is not None for r in recs if not r["err"]))
        self.assertTrue(any(r.get("agent_turn", 0) >= 2 for r in recs))

    def test_load_greedy_under_load_and_stop(self):
        rc = synthmix.main(["load", "--out", self.out, "--url", URL, "--streams", "4", "--duration", "120",
                            "--greedy-rounds", "2", "--warmup-s", "1", "--stop-after-greedy", "--long-max", "20000",
                            "--max-error-rate", "0"])
        S = json.load(open(os.path.join(self.out, "summary.json")))
        bad = S["all"]["errs"] + S["all"]["empty"] + S["harness_errors"]
        self.assertEqual(rc, 1 if bad else 0, S["gate"])  # --max-error-rate 0: any failed request fails the exit
        print("\nload summary: n=%d wall_s=%s stopped_early=%s" % (S["all"]["n"], S["wall_s"], S["stopped_early"]))
        self.assertTrue(S["stopped_early"])
        self.assertLess(S["wall_s"], 60)
        for k in (1, 2):
            rows = [json.loads(l) for l in open(os.path.join(self.out, f"greedy-r{k}.jsonl"))]
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(r.get("sha") for r in rows))
        self.assertTrue(os.path.exists(os.path.join(self.out, "done")))


class ScoreTest(unittest.TestCase):
    @staticmethod
    def m(text, finish="stop", err=None, reasoning="", tool_calls=()):
        return dict(text=text, finish=finish, err=err, reasoning=reasoning, tool_calls=list(tool_calls))

    def test_empty_vs_truncated(self):
        # only a max_tokens cut inside the reasoning is truncated; every other no-answer finish is empty
        tc = [{"id": "c", "name": "read_file", "arguments": "{}"}]
        cases = [(self.m("", finish="length", reasoning="Let me think"), False),
                 (self.m("", finish="length"), True),  # cut off having produced nothing at all
                 (self.m("  ", finish="length", reasoning=" \n"), True),
                 (self.m("", finish="stop", reasoning="Let me think"), True),  # finished, no answer
                 (self.m("", finish="abort", reasoning="Let me think"), True),
                 (self.m("", finish=None, reasoning="Let me think"), True),  # stream ended with no finish reason
                 (self.m("", finish="tool_calls", tool_calls=tc), False),
                 (self.m("", finish="length", err="HTTP 500"), False),  # an error is counted as an error, not empty
                 (self.m("24", finish="length"), False)]
        for m, want in cases:
            self.assertEqual(synthmix.score({}, m)["empty"], want, m)

    def test_stream_reader_fills_reasoning(self):
        # the empty rule reads m["reasoning"]: the SSE reader must fill it from delta.reasoning_content
        STATE["mode"] = "ok"
        c = synthmix.Client(URL, KEY, 30)
        body = dict(model="x", stream=True, messages=[{"role": "user", "content": "hi"}], max_tokens=64)
        for _ in range(3):  # the fake answers every 37th streaming request with a 500
            m = c.stream_chat(dict(body, reasoning_effort="high"), 30)
            if not m["err"]:
                break
        self.assertIsNone(m["err"])
        self.assertEqual((m["reasoning"], m["text"], m["finish"]), ("Let me think.", "Here is a short answer.", "stop"))
        for _ in range(3):
            m = c.stream_chat(dict(body, reasoning_effort="none"), 30)
            if not m["err"]:
                break
        self.assertEqual(m["reasoning"], "")

    def test_exact_number_match(self):
        self.assertTrue(synthmix.answer_match("2401", "The answer is 2401."))
        self.assertFalse(synthmix.answer_match("2401", "The answer is 24010."))
        self.assertFalse(synthmix.answer_match("2401", "12401"))
        self.assertFalse(synthmix.answer_match("2401", "2401.5"))
        self.assertFalse(synthmix.answer_match("2401", "v2401"))
        for t in ("7,203", "7.203", "7 203", "7\u202f203", "= 7203\n", "答案是7203。", "**7203**", "7203.0"):
            self.assertTrue(synthmix.answer_match("7203", t), t)
        self.assertFalse(synthmix.answer_match("7203", "72,03"))
        self.assertFalse(synthmix.answer_match("123456", "codes 123456789 and 654321"))
        self.assertTrue(synthmix.answer_match("654321", "codes 123456 654321"))

    def test_number_sign_and_big_integers(self):
        self.assertEqual(synthmix.numbers("x = -5, then 2024-10-29, y-3, \u22127 and -7,203"), [-5, 2024, 10, 29, 3, -7, -7203])
        self.assertTrue(synthmix.answer_match("-5", "The answer is -5."))
        self.assertTrue(synthmix.answer_match("-5", "The answer is \u22125."))
        self.assertFalse(synthmix.answer_match("-5", "The answer is 5."))
        self.assertFalse(synthmix.answer_match("5", "The answer is -5."))
        big = 2 ** 53 + 1  # float(big) == float(big - 1): a float compare would pass the wrong answer
        self.assertTrue(synthmix.answer_match(str(big), f"The answer is {big}."))
        self.assertFalse(synthmix.answer_match(str(big), f"The answer is {big - 1}."))
        self.assertFalse(synthmix.answer_match(str(10 ** 20), f"{10 ** 20 + 1}"))
        self.assertEqual(synthmix.numbers(str(big)), [big])
        self.assertIsInstance(synthmix.numbers("2401")[0], int)
        self.assertTrue(synthmix.answer_match("0.5", "half is 0.5"))
        self.assertTrue(synthmix.answer_match("1e3", "1000"))
        self.assertFalse(synthmix.answer_match("0.5", "half is 0.50001"))
        self.assertFalse(synthmix.answer_match("2.5", "9" * 400))  # an int too big for a float: no match, no crash

    def test_outcomes_keep_every_request_in_the_denominator(self):
        chk = {"expect": "2401"}
        cases = [(self.m("2401"), "ok"), (self.m("24010"), "wrong"), (self.m("  "), "empty"),
                 (self.m("24", finish="length"), "truncated"), (self.m("", finish=None), "truncated"),
                 (self.m("", finish=None, err="HTTP 500"), "error")]
        for m, want in cases:
            s = synthmix.score(chk, m)
            self.assertEqual(s["answer_outcome"], want, m)
            self.assertEqual(s["answer_ok"], want == "ok")
        js = {"json_keys": ["name", "year"], "json_year": 1999}
        self.assertEqual(synthmix.score(js, self.m('{"name": "a", "year": 1999}'))["json_outcome"], "ok")
        self.assertEqual(synthmix.score(js, self.m('{"name": "a"'))["json_outcome"], "wrong")
        self.assertEqual(synthmix.score(js, self.m('{"name": "a"', finish="length"))["json_outcome"], "truncated")
        self.assertFalse(synthmix.score(js, self.m('{"name": "a", "year": 2000}'))["json_year_ok"])
        bug = {"bug_fn": "load_items"}
        self.assertTrue(synthmix.score(bug, self.m("The bug is in `load_items()`."))["bug_found"])
        self.assertEqual(synthmix.score(bug, self.m("see load_items_v2"))["bug_outcome"], "wrong")

    def test_qlib_answer_line_fallback_is_explicit(self):
        sys.path.insert(0, os.path.join(HERE, "quality"))
        import qlib
        self.assertEqual(qlib.extract_answer("so 5 + 3 = 8\n#### 8", fallback=False), (8.0, 8.0))
        self.assertEqual(qlib.extract_answer("the answer is 42.", fallback=False), (None, None))
        self.assertEqual(qlib.extract_answer("the answer is 42."), (42.0, None))  # gsm8k.py's rule, the default
        self.assertEqual(qlib.extract_answer("step 3 gives 7\n#### 9\nchecked with 11"), (9.0, 9.0))

    def test_gate(self):
        S = dict(all=dict(n=100, errs=2, empty=0), harness_errors=0)
        self.assertTrue(synthmix.gate(S, 0.02)["ok"])
        S["all"]["empty"] = 1
        self.assertFalse(synthmix.gate(S, 0.02)["ok"])
        self.assertFalse(synthmix.gate(dict(all=dict(n=0, errs=0, empty=0), harness_errors=0), 0.02)["ok"])


class HangwatchUnitTest(unittest.TestCase):
    """In-process: sig_num, catches, fault_dumps' skip logic, the --dump-signal check in main(), scan_log counts."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="hangwatch-unit-")
        self.log = os.path.join(self.d, "serve.log")
        open(self.log, "w").close()

    def watch(self, **kw):
        a = dict(out=os.path.join(self.d, "w"), url="http://127.0.0.1:9", serve_log=self.log, stall_s=60.0,
                 idle_hold_s=35.0, dump_signal="USR2", dump_wait=0.0, keep_mismatch=200, scan_from_start=False,
                 on_hang="kill", on_hang_max=3, on_hang_window_s=3600.0, on_hang_term_s=20.0, hold_file="",
                 hold_max_s=900.0, serve_pattern=hangwatch.SERVE_PATTERN, serve_port=None, log_max_mb=0)
        a.update(kw)
        return hangwatch.Watch(types.SimpleNamespace(**a))

    def test_sig_num(self):
        usr2 = int(signal.SIGUSR2)
        for v in ("USR2", "usr2", "SIGUSR2", "SigUsr2", str(usr2)):
            self.assertEqual(hangwatch.sig_num(v), usr2, v)
        for v in ("none", "None", "", None):
            self.assertIsNone(hangwatch.sig_num(v), v)
        for v in ("USR3", "SIGFOO", "SIG_DFL", "_DFL", "0", "999", "-12", "12.0"):
            with self.assertRaises(ValueError, msg=v):
                hangwatch.sig_num(v)

    def test_main_rejects_a_bad_dump_signal_before_watching(self):
        for bad in (["--dump-signal", "USR3"], ["--dump-signal", "KILL"], ["--dump-wait", "-1"]):
            out = os.path.join(self.d, "never")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                hangwatch.main(["--out", out, "--url", "http://127.0.0.1:9", "--no-install", *bad])
            self.assertEqual(cm.exception.code, 2, bad)
            self.assertIn("--dump-", err.getvalue())
            self.assertFalse(os.path.exists(out), "it must fail before the watcher starts")

    def test_catches(self):
        proc = os.path.join(self.d, "proc")
        for pid, cgt in ((100, 1 << 11), (101, 1 << 9), (102, None), (103, "zz")):
            os.makedirs(os.path.join(proc, str(pid)))
            with open(os.path.join(proc, str(pid), "status"), "w") as f:
                f.write("Name:\tpython3\nSigBlk:\t0000000000000000\nSigIgn:\t0000000001001000\n")
                if cgt is not None:
                    f.write(f"SigCgt:\t{cgt if isinstance(cgt, str) else format(cgt, '016x')}\n")
        self.assertTrue(hangwatch.catches(100, 12, proc))  # bit 11 = signal 12 (Linux SIGUSR2)
        self.assertFalse(hangwatch.catches(100, 10, proc))
        self.assertTrue(hangwatch.catches(101, 10, proc))
        self.assertFalse(hangwatch.catches(102, 12, proc))  # no SigCgt line
        self.assertFalse(hangwatch.catches(103, 12, proc))  # malformed mask
        self.assertFalse(hangwatch.catches(999, 12, proc))  # no such process

    def test_fault_dumps_skip_logic(self):
        w = self.watch()
        w.pyspy = "/opt/pyspy/bin/py-spy"
        ev = os.path.join(self.d, "hang")
        os.makedirs(ev)
        procs = [dict(pid=10, role="tp0"), dict(pid=11, role="tp1"), dict(pid=12, role="scheduler"),
                 dict(pid=13, role="tp2"), dict(pid=14, role="detokenizer"), dict(pid=15, role="launcher")]
        # tp0: py-spy read its stacks; tp1, scheduler: py-spy refused; tp2: no py-spy file at all
        with open(os.path.join(ev, "pyspy-tp0-10.txt"), "w") as f:
            f.write("Process 10: sglang::scheduler_TP0\nThread 0x7F01 (active): \"MainThread\"\n    event_loop\n")
        for p in ("pyspy-tp1-11.txt", "pyspy-scheduler-12.txt"):
            with open(os.path.join(ev, p), "w") as f:
                f.write("\n--- stderr ---\nPermission denied (os error 13)\n")
        sig = hangwatch.sig_num("USR2")
        kills = []

        def fake_kill(pid, s):  # the rank's faulthandler writes its stacks into the serve log, among log noise
            kills.append((pid, s))
            with open(self.log, "a") as f:
                f.write(f"[2026-10-29 21:00:00 TP1] Decode batch, #running-req: 3 key={KEY}\n"
                        f"Thread 0x00007f{pid:04d} (most recent call first):\n"
                        f'  File "/sglang/scheduler.py", line 9 in event_loop_{pid} {KEY}\n'
                        f"Current thread 0x00007f{pid:04d} (most recent call first):\n  <no Python frame>\n")
        with mock.patch.object(hangwatch, "catches", lambda pid, s: pid in (11, 13)), \
                mock.patch.object(hangwatch.os, "kill", fake_kill):
            w.fault_dumps(ev, procs)
        self.assertEqual(kills, [(11, sig), (13, sig)])  # one at a time, in rank order; never 10, 12, 14, 15
        files = sorted(f for f in os.listdir(ev) if f.startswith("faulthandler-"))
        self.assertEqual(files, ["faulthandler-scheduler-12.txt", "faulthandler-tp1-11.txt", "faulthandler-tp2-13.txt"])
        self.assertIn("not sent: pid 12 does not catch signal", open(os.path.join(ev, files[0])).read())
        for f, pid in ((files[1], 11), (files[2], 13)):
            lines = open(os.path.join(ev, f)).read().splitlines()
            self.assertTrue(lines[0].startswith(f"signal {sig} sent to pid {pid} "), lines[0])
            # only that rank's dump (each read starts at the log size before its own signal), no log noise, key redacted
            self.assertEqual(lines[1:], [f"Thread 0x00007f{pid:04d} (most recent call first):",
                                         f'  File "/sglang/scheduler.py", line 9 in event_loop_{pid} <REDACTED>',
                                         f"Current thread 0x00007f{pid:04d} (most recent call first):",
                                         "  <no Python frame>"])
        # py-spy absent: even a rank with a stale py-spy file showing stacks is dumped
        kills.clear()
        w.pyspy = None
        with mock.patch.object(hangwatch, "catches", lambda pid, s: True), \
                mock.patch.object(hangwatch.os, "kill", fake_kill):
            w.fault_dumps(ev, procs[:1])
        self.assertEqual(kills, [(10, sig)])
        # --dump-signal none: nothing is signalled or written
        ev2 = os.path.join(self.d, "hang2")
        os.makedirs(ev2)
        w = self.watch(dump_signal="none")
        with mock.patch.object(hangwatch, "catches", lambda pid, s: True), \
                mock.patch.object(hangwatch.os, "kill", fake_kill):
            w.fault_dumps(ev2, procs)
        self.assertEqual((kills, os.listdir(ev2)), ([(10, sig)], []))

    def test_serve_targets_and_etime(self):
        self.assertEqual([hangwatch.etime_s(s) for s in ("00:07", "12:34", "02:03:04", "1-02:03:04")],
                         [7, 754, 7384, 93784])
        R = lambda pid, ppid, args, stat="S": dict(pid=pid, ppid=ppid, pgid=pid, stat=stat, age_s=5, args=args)
        rows = [R(100, 1, "/opt/v/bin/python3 /opt/v/bin/sglang serve --model-path /m --port 8002 --config k"),
                R(101, 100, "/opt/v/bin/python3 /opt/v/bin/sglang serve --model-path /m --port 8002 --config k"),
                R(200, 1, "python3 -m sglang.launch_server --port 8003"),  # another engine's port
                R(300, 1, "python3 hangwatch.py --serve-pattern 'sglang serve'"),
                R(400, 1, "sglang serve --port 8002", stat="Z"),
                R(500, 1, "bash -c sglang serveX --nccl-port 8002"),
                R(600, 1, "python3 -m sglang.launch_server --nccl-port 9000 --model-path /m")]  # no --port
        tg = hangwatch.serve_targets(rows, hangwatch.SERVE_PATTERN, 8002, me=1)
        self.assertEqual([r["pid"] for r in tg], [100, 600])  # top-level only; never the forked child, 8003, us
        self.assertEqual([r["pid"] for r in hangwatch.serve_targets(rows, hangwatch.SERVE_PATTERN, None, me=1)],
                         [100, 200, 600])
        self.assertEqual(hangwatch.descendants(rows, 100), {100, 101})

    def test_on_hang_kill_group_escalation_rate_limit_hold(self):
        marker = "hw-fake-serve-unit-%d" % os.getpid()
        w = self.watch(serve_pattern=marker, serve_port=18002, on_hang_max=2, on_hang_term_s=1.0)
        a, ad = sleeper(marker, 18002)
        b = c = None
        kid = int(open(os.path.join(ad, "kid")).read())
        try:
            self.assertTrue(w.on_hang_kill("no_token", 1, 130.0))
            self.assertEqual(w.retry_at, float("inf"))  # one kill per episode
            self.assertTrue(wait_for(lambda: a.returncode is not None, 10))
            self.assertEqual((open(os.path.join(ad, "term")).read(), a.returncode), ("TERM", 0))
            self.assertTrue(wait_for(lambda: not pid_alive(kid), 10), "the group's other member must get it too")
            # SIGTERM ignored: SIGKILL after --on-hang-term-s
            b, bd = sleeper(marker, 18002, mode="ignore")
            t = time.time()
            self.assertTrue(w.on_hang_kill("health", 2, 5.0))
            self.assertTrue(wait_for(lambda: b.returncode is not None, 10))
            self.assertEqual(b.returncode, -signal.SIGKILL)
            self.assertTrue(os.path.exists(os.path.join(bd, "term")))
            self.assertGreaterEqual(time.time() - t, 1.0)
            # --on-hang-max 2 per window: the third is only logged, the process lives
            c, _ = sleeper(marker, 18002)
            self.assertFalse(w.on_hang_kill("no_token", 3, 200.0))
            time.sleep(1)
            self.assertIsNone(c.returncode)
            # a fresh watchdog hold: no kill (watchdog.sh would relaunch nothing); a stale one does not hold
            hold = os.path.join(self.d, "watchdog.hold")
            open(hold, "w").close()
            w2 = self.watch(out=os.path.join(self.d, "w2"), serve_pattern=marker, serve_port=18002, hold_file=hold)
            t = time.time()
            self.assertFalse(w2.on_hang_kill("no_token", 1, 130.0))
            self.assertAlmostEqual(w2.retry_at, t + 60.0, delta=5)  # a hold can clear: tried again in the episode
            # another port, or no match at all: nothing to kill
            w3 = self.watch(out=os.path.join(self.d, "w3"), serve_pattern=marker, serve_port=18003)
            self.assertFalse(w3.on_hang_kill("no_token", 1, 130.0))
            time.sleep(1)
            self.assertIsNone(c.returncode)
            os.utime(hold, (time.time() - 1000, time.time() - 1000))
            self.assertTrue(w2.on_hang_kill("no_token", 2, 130.0))
            self.assertTrue(wait_for(lambda: c.returncode is not None, 10))
        finally:
            reap(*[p for p in (a, b, c) if p])
        K = [json.loads(l) for l in open(os.path.join(w.out, "KILLS"))]
        self.assertEqual([(k["outcome"], k.get("pid")) for k in K],
                         [("term", a.pid), ("killed", b.pid), ("rate_limited", None)])
        self.assertEqual(K[0]["pgid"], a.pid)
        self.assertEqual([k["outcome"] for k in json.load(open(os.path.join(w.out, "state.json")))["kills"]],
                         ["term", "killed", "rate_limited"])
        self.assertEqual([json.loads(l)["outcome"] for l in open(os.path.join(w2.out, "KILLS"))], ["held", "term"])
        self.assertEqual([json.loads(l)["outcome"] for l in open(os.path.join(w3.out, "KILLS"))], ["no_target"])
        log = open(os.path.join(w.out, "watch.log")).read()
        self.assertRegex(log, r"ON-HANG KILL pid=%d pgid=%d at 20\d\d-" % (a.pid, a.pid))
        self.assertIn("SIGKILL", log)
        self.assertIn("ON-HANG KILL suppressed: 2 kills in the last 3600 s", log)
        print(f"\nhangwatch on-hang unit: TERM -> exit 0 (group child gone), TERM ignored -> SIGKILL after 1 s, "
              f"3rd kill rate-limited, fresh hold refused, other port no_target")

    def test_rate_limit_seeded_from_kills_and_refusal_retry_times(self):
        out = os.path.join(self.d, "w")
        os.makedirs(out)
        now = time.time()
        with open(os.path.join(out, "KILLS"), "w") as f:  # a previous watcher's decisions in this --out
            for outcome, age in (("killed", 4000), ("term", 100), ("held", 90), ("kill_failed", 50), ("rate_limited", 10)):
                f.write(json.dumps(dict(t="x", ts=now - age, outcome=outcome)) + "\n")
            f.write('{"outcome": "term", "ts": \n[]\n{"outcome": "term"}\n')  # torn, foreign, no ts
        marker = "hw-no-such-serve-%d" % os.getpid()
        w = self.watch(serve_pattern=marker, on_hang_max=2, on_hang_retry_s=7.0)
        self.assertEqual([round(now - t) for t in w.kill_ts], [100, 50])  # in the window, kills only
        self.assertFalse(w.on_hang_kill("no_token", 1, 130.0))
        k = json.loads(open(os.path.join(out, "KILLS")).read().splitlines()[-1])
        self.assertEqual((k["outcome"], k["recent"]), ("rate_limited", 2))
        self.assertAlmostEqual(w.retry_at, now - 100 + 3600, delta=2)  # room again once the older one leaves
        self.assertAlmostEqual(k["retry_in_s"], 3500, delta=5)
        w1 = self.watch(serve_pattern=marker, on_hang_max=1)
        self.assertFalse(w1.on_hang_kill("no_token", 1, 130.0))
        self.assertAlmostEqual(w1.retry_at, now - 50 + 3600, delta=2)  # max 1: both must leave
        w3 = self.watch(serve_pattern=marker, on_hang_max=3, on_hang_retry_s=7.0)
        t = time.time()
        self.assertFalse(w3.on_hang_kill("no_token", 1, 130.0))  # room for a third: on to the ps check
        k = json.loads(open(os.path.join(out, "KILLS")).read().splitlines()[-1])
        self.assertEqual((k["outcome"], k["retry_in_s"]), ("no_target", 7.0))
        self.assertAlmostEqual(w3.retry_at, t + 7.0, delta=2)
        with mock.patch.object(hangwatch, "ps_table", mock.Mock(side_effect=OSError("ps: fork"))):
            t = time.time()
            self.assertFalse(w3.on_hang_kill("no_token", 1, 130.0))
        self.assertEqual(json.loads(open(os.path.join(out, "KILLS")).read().splitlines()[-1])["outcome"], "ps_failed")
        self.assertAlmostEqual(w3.retry_at, t + 7.0, delta=2)
        w0 = self.watch(serve_pattern=marker, on_hang_max=0)
        self.assertFalse(w0.on_hang_kill("no_token", 1, 130.0))
        self.assertEqual(w0.retry_at, float("inf"))  # --on-hang-max 0 never kills
        self.assertEqual(self.watch(out=os.path.join(self.d, "fresh")).kill_ts, [])  # no KILLS file

    def test_service_log_rotation_and_single_instance(self):
        w = self.watch(log_max_mb=1)
        for i in range(1200):
            w.log(f"line {i} " + "x" * 1000)
        p = os.path.join(w.out, "watch.log")
        self.assertTrue(os.path.exists(p + ".1"))
        self.assertLessEqual(os.path.getsize(p) + os.path.getsize(p + ".1"), (1 << 20) + 2048)
        self.assertIn("line 1199 ", open(p).read().splitlines()[-1])
        # a live hangwatch already named in --out/hangwatch.pid: a second --service exits 1 before watching
        out = os.path.join(self.d, "svc")
        os.makedirs(out)
        live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "hangwatch.py"])
        try:
            with open(os.path.join(out, "hangwatch.pid"), "w") as f:
                f.write(f"{live.pid}\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = hangwatch.main(["--out", out, "--url", "http://127.0.0.1:9", "--no-install", "--service"])
            self.assertEqual(rc, 1)
            self.assertIn("already runs", err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(out, "watch.log")))
        finally:
            live.kill()
            live.wait()

    def test_scan_log_counts_mismatch_lines(self):
        # the probe commit's real line shapes (logger.error / logger.info behind sglang's log prefix)
        mis = ("[2026-10-29 21:04:11 TP{r} EP{r}] MK_TP_PROBE MISMATCH it=4200 rank={r} "
               "fields=('running', 'seq_lens') mine=(3, 812) rank0=(3, 811) key=" + KEY)
        ok = "[2026-10-29 21:04:11 TP0 EP0] MK_TP_PROBE ok it=4100 rank=0 checks=40 state=(3, 811)"
        with open(self.log, "w") as f:
            f.write(mis.format(r=1) + "\n" + ok + "\n")
        # default (no --scan-from-start): what is already in the log is not counted
        w = self.watch(keep_mismatch=2)
        w.scan_log()
        self.assertEqual((w.state["mk_tp_probe"]["ok"], w.state["mk_tp_probe"]["mismatch"]), (0, 0))
        with open(self.log, "a") as f:
            f.write("[2026-10-29 21:04:12 TP0 EP0] Prefill batch, #new-seq: 1\n" + mis.format(r=2) + "\n" + ok + "\n"
                    + mis.format(r=3) + "\n" + mis.format(r=1)[:40])  # the last line is still being written
        w.scan_log()
        pr = w.state["mk_tp_probe"]
        self.assertEqual((pr["ok"], pr["mismatch"]), (1, 2))
        self.assertIsNotNone(pr["first_mismatch"])
        self.assertEqual(w.state["serve_batches"]["prefill"], 1)
        with open(self.log, "a") as f:
            f.write(mis.format(r=1)[40:] + "\n")
        w.scan_log()
        self.assertEqual((pr["ok"], pr["mismatch"]), (1, 3))  # the partial line counted once, when complete
        kept = open(os.path.join(w.out, "mk_tp_probe_mismatch.txt")).read().splitlines()
        self.assertEqual(kept, [mis.format(r=2).replace(KEY, "<REDACTED>"), mis.format(r=3).replace(KEY, "<REDACTED>")])
        # --scan-from-start counts the existing log too
        w2 = self.watch(out=os.path.join(self.d, "w2"), scan_from_start=True)
        w2.scan_log()
        self.assertEqual((w2.state["mk_tp_probe"]["ok"], w2.state["mk_tp_probe"]["mismatch"]), (2, 4))


class HangwatchTest(unittest.TestCase):
    def setUp(self):
        STATE["mode"] = "ok"
        STATE["rt"] = 0
        self.out = tempfile.mkdtemp(prefix="hangwatch-")
        self.log = os.path.join(self.out, "serve.log")
        # > 4 MiB so the tail's seek path (size - 4 MiB) is exercised
        with open(self.log, "w") as f:
            for i in range(60000):
                f.write(f"[2026-10-29 TP0] line {i:05d} decode batch ok api_key={KEY} " + "x" * 40 + "\n")
            f.write("MK_TP_PROBE ok step=1\n")
        self.w = os.path.join(self.out, "w")

    def start(self, *extra):
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "hangwatch.py"), "--out", self.w, "--url", URL,
                                   "--serve-log", self.log, "--interval", "0.5", "--stall-s", "3",
                                   "--idle-hold-s", "1", "--no-install",
                                   *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def stop(self):
        open(os.path.join(self.w, "stop"), "w").close()
        self.p.wait(timeout=15)

    def wlog(self):
        return open(os.path.join(self.w, "watch.log")).read()

    def hangs(self):
        p = os.path.join(self.w, "HANG")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    def test_stall_health_tail_probe_redaction(self):
        self.start("--scan-from-start", "--health-timeout", "1")
        self.assertTrue(wait_for(lambda: "metrics matched" in (open(os.path.join(self.w, "watch.log")).read()
                                                               if os.path.exists(os.path.join(self.w, "watch.log")) else ""), 10))
        with open(self.log, "a") as f:
            f.write(f"MK_TP_PROBE MISMATCH rank=2 field=ACCEPT_GREEDY key={KEY}\nMK_TP_PROBE ok step=2\n")
        STATE["mode"] = "stall"
        self.assertTrue(wait_for(lambda: len(self.hangs()) >= 1, 15))
        self.assertTrue(wait_for(lambda: "evidence done" in self.wlog(), 15))
        STATE["mode"] = "ok"
        self.assertTrue(wait_for(lambda: "idle and healthy again" in self.wlog() or "progress resumed" in self.wlog(), 10))
        STATE["mode"] = "healthhang"  # /health sleeps 10 s; the 1 s timeout must turn it into failures quickly
        t = time.time()
        self.assertTrue(wait_for(lambda: len(self.hangs()) >= 2, 12))
        health_s = time.time() - t
        STATE["mode"] = "ok"
        time.sleep(1.5)
        self.stop()
        H = self.hangs()
        self.assertEqual([h["reason"] for h in H], ["no_token", "health"])
        ev = H[0]["evidence"]
        tail = open(os.path.join(ev, "serve-tail.txt")).read().splitlines()
        self.assertEqual(len(tail), 300)
        self.assertEqual(tail[-1], "MK_TP_PROBE ok step=2")
        self.assertEqual(tail[-2], "MK_TP_PROBE MISMATCH rank=2 field=ACCEPT_GREEDY key=<REDACTED>")
        self.assertTrue(tail[-4].startswith("[2026-10-29 TP0] line 59999 decode batch ok api_key=<REDACTED>"), tail[-4])
        self.assertTrue(tail[0].startswith("[2026-10-29 TP0] line 59703 "), tail[0])
        for f in ("reason.json", "metrics.txt", "health.txt", "procs.txt", "nvidia-smi.txt", "ps.txt"):
            self.assertTrue(os.path.exists(os.path.join(ev, f)), f)
        st = json.load(open(os.path.join(self.w, "state.json")))
        self.assertEqual(st["mk_tp_probe"]["ok"], 2)
        self.assertEqual(st["mk_tp_probe"]["mismatch"], 1)
        self.assertEqual(st["metric_names"]["running"], "sglang:num_running_reqs")
        self.assertEqual(st["metric_names"]["realtime_tokens"], "sglang:realtime_tokens_total")
        self.assertEqual(st["metric_names"]["prompt_tokens"], "sglang:prompt_tokens_total")
        self.assertNotIn("metric_warning", st)
        leaks = [os.path.join(d, f) for d, _, fs in os.walk(self.w) for f in fs
                 if KEY in open(os.path.join(d, f), errors="replace").read()]
        self.assertEqual(leaks, [])
        print(f"\nhangwatch: episodes={[h['reason'] for h in H]} serve-tail lines={len(tail)} last={tail[-1]!r} "
              f"probe={st['mk_tp_probe']['ok']} ok / {st['mk_tp_probe']['mismatch']} mismatch, "
              f"health stall caught {health_s:.1f}s after /health hung, key leaks=0")

    def wait_started(self):
        self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(self.w, "watch.log"))
                                 and "metrics matched" in self.wlog(), 10))

    def hold(self, secs, step=None):
        """Sleep `secs`, calling step() every 0.5 s; return the HANG reasons seen meanwhile."""
        end = time.time() + secs
        while time.time() < end:
            if step:
                step()
            time.sleep(0.5)
        return [h["reason"] for h in self.hangs()]

    def test_long_prefill_is_progress_not_a_stall(self):
        # running > 0 with generation_tokens_total flat for 2x --stall-s, three ways a healthy long prefill shows up
        self.start()
        self.wait_started()
        STATE["mode"] = "prefill"

        def chunk():  # realtime_tokens_total{mode=prefill_compute}: bumped by the scheduler at every prefill chunk
            STATE["rt"] += 8192
        self.assertEqual(self.hold(6, chunk), [], "realtime prefill counter must count as progress")
        STATE["rt"] = None  # a build without the per-step counter: prompt_tokens_total moves only at request finish
        t0 = time.time()
        bumps = []

        def finish():
            if len(bumps) < 2 and time.time() - t0 >= 2.0 * (len(bumps) + 1):
                bumps.append(1)
                STATE["prompt"] += 900000
        self.assertEqual(self.hold(6, finish), [], "a prompt_tokens_total jump must count as progress")

        def batch_line():  # neither counter moves; the scheduler's per-chunk serve-log line does
            with open(self.log, "a") as f:
                f.write("[2026-10-29 16:50:22 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #running-req: 0\n")
        self.assertEqual(self.hold(6, batch_line), [], "serve-log Prefill batch lines must count as progress")
        # nothing moves at all with running > 0: that IS a stall (the watcher is not inert in this mode)
        self.assertTrue(wait_for(lambda: len(self.hangs()) >= 1, 8))
        self.stop()
        H = self.hangs()
        self.assertEqual([h["reason"] for h in H], ["no_token"])
        st = json.load(open(os.path.join(self.w, "state.json")))
        self.assertGreaterEqual(st["serve_batches"]["prefill"], 8)
        print(f"\nhangwatch prefill: no stall across 18 s of gen-flat prefill (stall_s 3); then {H[0]['reason']} "
              f"after {H[0]['stalled_s']} s flat; serve prefill batches counted={st['serve_batches']['prefill']}")

    def test_arrival_after_short_idle_restarts_the_stall_clock(self):
        # box-a soak at 1/10 scale: --stall-s 60 / --idle-hold-s 35, a 20 s think gap, then a 50 s prefill with every
        # counter flat. The gap is shorter than --idle-hold-s, so only the 0 -> >0 arrival may reset the clock; without
        # it the request inherits the progress age from before the gap (20 + 50 > 60) and trips a false HANG.
        STATE["rt"] = None
        self.start("--stall-s", "6", "--idle-hold-s", "3.5")
        self.wait_started()
        time.sleep(1)  # "ok" mode: counters move, so the last progress is the last ok poll
        STATE["mode"] = "idle"
        self.assertEqual(self.hold(2), [])
        STATE["mode"] = "prefill"  # running 1, nothing moves
        t = time.time()
        self.assertEqual(self.hold(5), [], "an arrival after a short idle gap must not inherit the old progress age")
        # still not inert: the same flat request does stall once --stall-s passes from its arrival
        self.assertTrue(wait_for(lambda: len(self.hangs()) >= 1, 6))
        hang_s = time.time() - t
        self.stop()
        H = self.hangs()
        self.assertEqual([h["reason"] for h in H], ["no_token"])
        self.assertLess(H[0]["stalled_s"], 7.5)  # timed from the arrival, not from before the idle gap (~8 s)
        print(f"\nhangwatch arrival: 2 s idle + 5 s flat prefill, no HANG (stall_s 6, idle_hold 3.5); "
              f"no_token {hang_s:.1f}s after the arrival, stalled_s={H[0]['stalled_s']}")

    def test_queue_only_wedge_hangs_and_idle_does_not(self):
        STATE["rt"] = None
        self.start()
        self.wait_started()
        STATE["mode"] = "idle"  # nothing in flight, nothing moves: idle, never a stall
        self.assertEqual(self.hold(6), [])
        STATE["mode"] = "queued"  # queue > 0, running 0, nothing moves: the scheduler admits nothing
        t = time.time()
        self.assertTrue(wait_for(lambda: len(self.hangs()) >= 1, 10))
        wedge_s = time.time() - t
        self.assertTrue(wait_for(lambda: "evidence done" in self.wlog(), 15))
        STATE["mode"] = "idle"
        self.assertTrue(wait_for(lambda: "idle and healthy again" in self.wlog(), 10))
        self.stop()
        H = self.hangs()
        self.assertEqual([h["reason"] for h in H], ["queue_wedge"])
        self.assertEqual((H[0]["running"], H[0]["queue"]), (0, 2))
        print(f"\nhangwatch queued: idle 6 s no stall; queue_wedge caught {wedge_s:.1f}s after the queue wedged")

    def test_missing_metrics_warns_once(self):
        STATE["mode"] = "nometrics"
        self.start()
        self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(self.w, "watch.log")) and "WARNING" in self.wlog(), 10))
        time.sleep(2)
        self.stop()
        self.assertEqual(self.wlog().count("WARNING"), 1)
        st = json.load(open(os.path.join(self.w, "state.json")))
        self.assertIn("INERT", st["metric_warning"])
        print("\nhangwatch missing-metrics:", st["metric_warning"])

    def test_service_on_hang_kill_after_capture_boot_grace_rate_limit(self):
        marker = "hw-fake-serve-loop-%d" % os.getpid()
        port = SRV.server_address[1]  # --serve-port defaults to --url's port
        a, ad = sleeper(marker, port)
        b = c = None
        try:
            os.makedirs(self.w)
            open(os.path.join(self.w, "stop"), "w").close()  # left by an earlier run: --service removes it
            self.start("--service", "--on-hang", "kill", "--serve-pattern", marker, "--boot-grace-s", "4",
                       "--on-hang-max", "2", "--on-hang-term-s", "2", "--health-timeout", "1",
                       "--hold-file", os.path.join(self.out, "watchdog.hold"))
            self.wait_started()
            self.assertIn("removed a stale stop file", self.wlog())
            r = subprocess.run([sys.executable, os.path.join(HERE, "hangwatch.py"), "--out", self.w, "--url", URL,
                                "--no-install", "--service"], capture_output=True, text=True, timeout=30)
            self.assertEqual((r.returncode, "already runs" in r.stderr), (1, True))
            STATE["mode"] = "down"  # the engine wedges: /metrics and /health stop answering
            self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(ad, "term")), 20), "no kill")
            t_kill = time.time()
            log = self.wlog()
            self.assertLess(log.index("evidence done"), log.index(f"ON-HANG KILL pid={a.pid} pgid={a.pid} at "))
            b, bd = sleeper(marker, port)  # watchdog.sh's relaunch; it stays down (booting) past the grace
            self.assertEqual(self.hold(max(0.0, t_kill + 3 - time.time())), ["health"], "a HANG while booting")
            self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(bd, "term")), 20), "no kill after the grace")
            self.assertGreaterEqual(time.time() - t_kill, 4.0)
            c, _ = sleeper(marker, port)
            self.assertTrue(wait_for(lambda: "ON-HANG KILL suppressed" in self.wlog(), 25), "no rate limit")
            self.assertIsNone(c.returncode)
            STATE["mode"] = "ok"
            pidf = os.path.join(self.w, "hangwatch.pid")
            self.assertEqual(open(pidf).read().strip(), str(self.p.pid))
            self.stop()
        finally:
            reap(*[p for p in (a, b, c) if p])
        self.assertEqual(self.p.returncode, 0)
        self.assertFalse(os.path.exists(pidf), "a clean stop removes hangwatch.pid")
        self.assertEqual([h["reason"] for h in self.hangs()], ["health", "health", "health"])
        K = [json.loads(l) for l in open(os.path.join(self.w, "KILLS"))]
        self.assertEqual([(k["outcome"], k.get("pid")) for k in K], [("term", a.pid), ("term", b.pid), ("rate_limited", None)])
        log = self.wlog()
        for s in ("engine booting (killed by --on-hang", "boot grace over", "ON-HANG KILL suppressed: 2 kills"):
            self.assertIn(s, log)
        print(f"\nhangwatch --service --on-hang kill: kills={[(k['outcome'], k.get('took_s')) for k in K]} "
              f"hangs={len(self.hangs())}, no HANG inside the 4 s boot grace, 3rd kill rate-limited")

    def test_refused_kill_is_retried_within_the_episode(self):
        # a fresh watchdog hold refuses the kill; the stall goes on, the hold goes stale: the SAME episode kills
        marker = "hw-fake-serve-retry-%d" % os.getpid()
        hold = os.path.join(self.out, "watchdog.hold")
        open(hold, "w").close()
        a, ad = sleeper(marker, SRV.server_address[1])
        try:
            self.start("--on-hang", "kill", "--serve-pattern", marker, "--boot-grace-s", "30", "--on-hang-retry-s", "1",
                       "--on-hang-term-s", "2", "--health-timeout", "1", "--hold-file", hold)
            self.wait_started()
            time.sleep(1.5)  # a few healthy polls: the young sleeper has answered /health 200, so it is not booting
            self.assertNotIn("engine booting", self.wlog())
            STATE["mode"] = "down"
            kills = lambda: [json.loads(l) for l in open(os.path.join(self.w, "KILLS"))] \
                if os.path.exists(os.path.join(self.w, "KILLS")) else []
            self.assertTrue(wait_for(lambda: len(kills()) >= 2, 25), "a held kill must be tried again")
            self.assertIsNone(a.returncode)
            os.utime(hold, (time.time() - 1000, time.time() - 1000))
            self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(ad, "term")), 10), "no kill once the hold went stale")
            STATE["mode"] = "ok"
            self.stop()
        finally:
            reap(a)
        K = kills()
        self.assertEqual(len(self.hangs()), 1)
        self.assertEqual(K[-1]["outcome"], "term")
        self.assertEqual({k["outcome"] for k in K[:-1]}, {"held"})
        self.assertEqual({k["episode"] for k in K}, {1})  # one episode throughout
        gaps = [K[i + 1]["ts"] - K[i]["ts"] for i in range(len(K) - 1)]
        self.assertGreaterEqual(min(gaps), 0.9)  # every --on-hang-retry-s, not every poll
        print(f"\nhangwatch retry: {len(K) - 1} held refusals {[round(g, 1) for g in gaps]} s apart, then "
              f"{K[-1]['outcome']} in the same episode")

    def test_booting_engine_is_held_until_health_and_none_never_kills(self):
        marker = "hw-fake-serve-boot-%d" % os.getpid()
        d, _ = sleeper(marker, SRV.server_address[1])
        try:
            STATE["mode"] = "down"  # a fresh serve process loading weights: nothing answers yet
            self.start("--serve-pattern", marker, "--boot-grace-s", "30", "--health-timeout", "1")
            self.assertTrue(wait_for(lambda: os.path.exists(os.path.join(self.w, "watch.log"))
                                     and f"engine booting (serve pid {d.pid} is" in self.wlog(), 10))
            self.assertEqual(self.hold(4), [], "a booting engine is not a hang")  # > stall_s 3 and 2 health fails
            STATE["mode"] = "ok"
            self.assertTrue(wait_for(lambda: "engine up: /health 200" in self.wlog(), 10))
            STATE["mode"] = "down"  # up once, then dead well inside the 30 s grace: a hang now
            self.assertTrue(wait_for(lambda: len(self.hangs()) >= 1, 10))
            time.sleep(1.5)
            self.assertIsNone(d.returncode, "--on-hang none must never kill")
            STATE["mode"] = "ok"
            self.stop()
        finally:
            reap(d)
        self.assertEqual([h["reason"] for h in self.hangs()], ["health"])
        self.assertFalse(os.path.exists(os.path.join(self.w, "KILLS")))
        print("\nhangwatch boot: held while the young serve process had no /health 200; HANG after it had one")


if __name__ == "__main__":
    unittest.main(warnings="ignore")  # the tests' quick open().read() calls are not under test
