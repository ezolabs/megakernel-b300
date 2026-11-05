#!/usr/bin/env python3
"""under_load.py -- gsm8k quality while a background load holds 16 streams on the engine.

Starts the background load, waits until all its streams are in flight (plus --warm-s), runs a gsm8k subset
(default the 500-problem seed-7 subset of gsm8k_full.py, so every id pairs with the idle full-set run) in each
cell (default t0 and prod) at --concurrency foreground streams (default 8: 16 + 8 = 24 in flight, under the
engine's 64), then stops the load. The foreground is gsm8k_full.py's own prompt and scoring.

Background load (built in; harness/synthmix.py did not exist when this was written -- point --bg-cmd at it, or
any loader, to use that instead): --bg-streams threads (default 16), each sending back to back, never idle,
synthetic requests only, at prod sampling (T 1.0 / top_p 1.0) and effort high:
  chat    60%  per-stream fixed system prompt (~400 tokens, cacheable like a real client's) + 300-3,000 tokens
               of fresh synthetic prose + "summarise in N bullets", max_tokens 256-1,024
  reason  25%  a fresh multi-step arithmetic task, max_tokens 2,048
  long    15%  a fresh 20k-60k-token synthetic document (cold prefill) + a question, max_tokens 512
Its record (OUT/load-bg.json, and in each summary's `background`): requests, errors, completion tokens and
tok/s, the in-flight count sampled every second during each foreground cell (min / median / max), and
sglang:num_running_reqs from /metrics every 5 s. A cell is flagged `load_held: false` unless at least
streams-1 were in flight in >= 95% of the samples (a stream is out of flight for a moment between requests).

    python3 under_load.py --out OUT [--cells t0,prod] [--n 500] [-c 8] [--bg-streams 16] [--warm-s 30]
                          [--bg-cmd 'python3 ../synthmix.py ...']

Writes OUT/gsm8k-<cell>-load16.jsonl / .summary.json (suite gsm8k, cell `<cell>-load<streams>`) and
OUT/load-bg.json. Compare against the idle run with
`compare.py files OUT/gsm8k-t0.summary.json OUT/gsm8k-t0-load16.summary.json` (paired by problem id).
Expected wall on the final: ~6-9 min per cell plus ~1 min warm-up/drain.
"""
import argparse
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402
import gsm8k_full  # noqa: E402

WORDS = ('river stone lantern harbour meadow copper willow beacon orchard granite tide canvas ember thistle '
         'compass quarry falcon marble ribbon cedar glacier saffron anchor velvet prairie summit harvest '
         'cobalt meridian parcel archive kettle ledger mosaic paddle quill rampart sorrel timber umber vessel').split()


def prose(rnd, tokens):
    words = []
    n = int(tokens / 1.3)
    while len(words) < n:
        s = [rnd.choice(WORDS) for _ in range(rnd.randint(8, 16))]
        s[0] = s[0].capitalize()
        words += s
        words[-1] += '.'
    return ' '.join(words)


class BgLoad:
    def __init__(self, conn, streams, seed=1):
        self.conn, self.streams, self.seed = conn, streams, seed
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.inflight = 0
        self.log = []
        self.threads = []
        self.systems = [prose(random.Random(f'sys-{seed}-{i}'), 400) for i in range(streams)]

    def request(self, rnd, sid):
        kind = rnd.choices(['chat', 'reason', 'long'], [60, 25, 15])[0]
        nonce = uuid.uuid4().hex[:12]
        if kind == 'chat':
            msgs = [{'role': 'system', 'content': 'You are a concise assistant. Context notes: ' + self.systems[sid]},
                    {'role': 'user', 'content': f'[{nonce}] ' + prose(rnd, rnd.randint(300, 3000))
                     + f'\n\nSummarise the passage above in {rnd.randint(3, 7)} bullet points.'}]
            mt = rnd.randint(256, 1024)
        elif kind == 'reason':
            a, b, c, d = (rnd.randint(12, 999) for _ in range(4))
            msgs = [{'role': 'user', 'content': f'[{nonce}] A depot holds {a} crates. Each day {b} arrive and '
                     f'{c} leave, and every {rnd.randint(2, 6)} days {d} more are audited out. Work out, step by '
                     f'step, how many crates remain after {rnd.randint(5, 30)} days, then check your arithmetic.'}]
            mt = 2048
        else:
            msgs = [{'role': 'user', 'content': f'[{nonce}] ' + prose(rnd, rnd.randint(20000, 60000))
                     + '\n\nWhich three words occur most often in the document above? Explain briefly.'}]
            mt = 512
        return kind, msgs, mt

    def worker(self, sid):
        rnd = random.Random(f'bg-{self.seed}-{sid}')
        while not self.stop.is_set():
            kind, msgs, mt = self.request(rnd, sid)
            with self.lock:
                self.inflight += 1
            t0 = time.time()
            r = qlib.chat(self.conn, msgs, max_tokens=mt, temperature=qlib.PROD_TEMPERATURE, top_p=qlib.PROD_TOP_P,
                          effort=qlib.SERVING_EFFORT, timeout=900)
            with self.lock:
                self.inflight -= 1
                self.log.append({'t0': t0, 't1': time.time(), 'kind': kind, 'error': r.error,
                                 'completion_tokens': r.usage.get('completion_tokens') if not r.error else 0,
                                 'prompt_tokens': r.usage.get('prompt_tokens') if not r.error else 0})
            if r.error:
                time.sleep(1.0)   # a failing engine is not hammered in a tight loop

    def start(self):
        for i in range(self.streams):
            t = threading.Thread(target=self.worker, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def halt(self, wait_s=180):
        self.stop.set()
        t_end = time.time() + wait_s
        for t in self.threads:
            t.join(max(0.1, t_end - time.time()))
        return sum(1 for t in self.threads if t.is_alive())


class Sampler:
    """In-flight samples (1 s, the built-in loader's own count) and /metrics num_running_reqs (5 s)."""

    def __init__(self, conn, bg):
        self.conn, self.bg = conn, bg
        self.stop = threading.Event()
        self.inflight, self.running = [], []
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        k = 0
        while not self.stop.is_set():
            if self.bg is not None:
                self.inflight.append(self.bg.inflight)
            if k % 5 == 0:
                v = running_reqs(self.conn)
                if v is not None:
                    self.running.append(v)
            k += 1
            self.stop.wait(1.0)


def running_reqs(conn):
    import urllib.request
    try:
        req = urllib.request.Request(conn.base + '/metrics', headers=conn.headers())
        txt = urllib.request.urlopen(req, timeout=5).read().decode('utf-8', 'replace')
    except Exception:
        return None
    vals = [float(m) for m in re.findall(r'^sglang:num_running_reqs(?:\{[^}]*\})?\s+([0-9.eE+-]+)', txt, re.M)]
    return sum(vals) if vals else None


def dist(xs):
    if not xs:
        return None
    return {'n': len(xs), 'min': min(xs), 'median': statistics.median(xs), 'max': max(xs)}


def bg_stats(log, t0, t1):
    win = [e for e in log if e['t1'] >= t0 and e['t0'] <= t1]
    done = [e for e in log if t0 <= e['t1'] <= t1]
    toks = sum(e['completion_tokens'] or 0 for e in done if not e['error'])
    kinds = {}
    for e in done:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    return {'requests_completed': len(done), 'requests_overlapping': len(win),
            'errors': sum(1 for e in done if e['error']),
            'first_error': next((e['error'] for e in done if e['error']), None),
            'completion_tokens': toks, 'completion_tok_per_s': round(toks / max(1e-9, t1 - t0), 1), 'kinds': kinds}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap, concurrency=8)
    ap.add_argument('--cells', default='t0,prod')
    ap.add_argument('--n', type=int, default=500, help='gsm8k problems (default the 500 seed-7 subset)')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=None, help='foreground request seed (default none)')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--bg-streams', type=int, default=16)
    ap.add_argument('--warm-s', type=float, default=30.0, help='seconds of load before the foreground starts')
    ap.add_argument('--bg-cmd', default=None, help='external background loader command (replaces the built-in one)')
    a = ap.parse_args()
    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    items = gsm8k_full.load_items(a.n, a.dataset)
    est_cell = items and len(items) * gsm8k_full.MEAN_TOKENS_EST / (a.concurrency * 0.7 * qlib.EST_STREAM_TPS)
    print(f'under_load: {len(items)} gsm8k problems x {cells} at {a.concurrency} streams over {a.bg_streams} background '
          f'streams; expected wall ~{qlib.fmt_s(est_cell * len(cells) + a.warm_s + 60)} on the final -> {out}',
          flush=True)

    bg, proc = None, None
    if a.bg_cmd:
        proc = subprocess.Popen(a.bg_cmd, shell=True, start_new_session=True,
                                stdout=open(os.path.join(out, 'load-bg-cmd.log'), 'w'), stderr=subprocess.STDOUT)
        print(f'under_load: external loader pid {proc.pid}: {a.bg_cmd}', flush=True)
    else:
        bg = BgLoad(conn, a.bg_streams)
        bg.start()
    t_warm = time.time()
    while bg is not None and bg.inflight < a.bg_streams and time.time() - t_warm < 120:
        time.sleep(0.5)
    time.sleep(a.warm_s)
    if proc is not None and proc.poll() is not None:
        sys.exit(f'under_load: the external loader exited early (rc {proc.returncode}); see load-bg-cmd.log')
    print(f'under_load: background warm ({time.time() - t_warm:.0f}s, in flight '
          f'{bg.inflight if bg else "n/a (external)"}); foreground starts', flush=True)

    record = {'streams': a.bg_streams, 'loader': a.bg_cmd or 'built-in BgLoad', 'cells': {}}
    for cell in cells:
        if cell not in gsm8k_full.CELLS:
            ap.error(f'unknown cell {cell!r}')
        smp = Sampler(conn, bg)
        smp.t.start()
        t0 = time.time()
        recs, wall = qlib.run_items(items, gsm8k_full.make_scorer(conn, cell, a.effort, a.max_tokens, a.seed),
                                    concurrency=a.concurrency,
                                    out_jsonl=os.path.join(out, f'gsm8k-{cell}-load{a.bg_streams}.jsonl'),
                                    label=f'gsm8k-{cell}-load{a.bg_streams}', est_s=est_cell)
        t1 = time.time()
        smp.stop.set()
        smp.t.join(10)
        bgrec = {'inflight': dist(smp.inflight), 'num_running_reqs': dist(smp.running), 'window_s': round(t1 - t0, 1)}
        if bg is not None:
            bgrec.update(bg_stats(bg.log, t0, t1))
            # A stream is briefly out of flight between its requests, so 'held' = at least streams-1 in flight
            # in >= 95% of the 1 s samples.
            full = sum(1 for x in smp.inflight if x >= a.bg_streams - 1)
            bgrec['frac_samples_full'] = round(full / len(smp.inflight), 4) if smp.inflight else None
            bgrec['load_held'] = bool(smp.inflight) and full / len(smp.inflight) >= 0.95
        else:
            bgrec['load_held'] = proc.poll() is None
        record['cells'][cell] = bgrec
        print(f'under_load: background during {cell}: {json.dumps(bgrec)}', flush=True)
        gsm8k_full.summarize(out, f'{cell}-load{a.bg_streams}', recs, wall, a, conn,
                             extra={'background': bgrec, 'foreground_concurrency': a.concurrency})
    if bg is not None:
        left = bg.halt()
        record['total'] = bg_stats(bg.log, t_warm, time.time())
        record['threads_still_running_at_exit'] = left
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    qlib.write_json(os.path.join(out, 'load-bg.json'), record)
    print(f'under_load: background record -> {os.path.join(out, "load-bg.json")}')


if __name__ == '__main__':
    main()
