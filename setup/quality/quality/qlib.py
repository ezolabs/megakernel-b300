#!/usr/bin/env python3
"""qlib.py -- the shared half of the quality suite (stdlib only, Python >= 3.8).

Every suite in this folder imports it for the same five things, so every suite sends, records, scores and
summarises the same way:

  * the engine connection: base URL (--base-url, $ENGINE_URL, default http://127.0.0.1:8002), model
    (--model, $MODEL, default DeepSeek-V4.1-Flash) and key (--api-key, $VLLM_API_KEY, else the
    VLLM_API_KEY= line of --env-file, default /etc/environment). The key rides only in the Authorization
    header; it is never written to a file or printed.
  * one chat call (`chat`) that never raises: it returns a Reply whose `error` names the failure KIND --
    `transport:` (no HTTP response: refused, reset, timeout), `status:` (a non-2xx answer, with its body's
    first 300 chars), `response:` (a 200 this code cannot read). A failed request is not a wrong answer:
    every suite records it as ok=None and counts it under `answered`, never under the quality metric.
  * a thread-pool runner (`run_items`) that writes one jsonl line per item as it completes, prints a
    progress/ETA line every 30 s, and rewrites the file sorted by id at the end.
  * the intervals: Wilson 95% for proportions, percentile bootstrap (5,000 resamples, fixed seed) for
    means; and the two-sample pieces compare.py uses (Newcombe hybrid-score CI for a difference of
    proportions, Fisher exact, two-proportion z, exact McNemar for paired items).
  * the summary file every suite writes next to its jsonl: <out>/<suite>-<cell>.summary.json, the shape
    compare.py reads (see `write_summary`).

Prod sampling (the DSML gate's and DeepSeek's official sampler) is T 1.0 / top_p 1.0; the serving effort is
`high`. Both are named constants here so every suite states the same cell.
"""
import concurrent.futures
import datetime
import json
import math
import os
import random
import re
import socket
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request

DEFAULT_BASE = 'http://127.0.0.1:8002'
DEFAULT_MODEL = 'DeepSeek-V4.1-Flash'
DEFAULT_ENVFILE = '/etc/environment'
PROD_TEMPERATURE = 1.0
PROD_TOP_P = 1.0
SERVING_EFFORT = 'high'
# 'none' is SENT and switches thinking off (the engine's explicit off switch, our internal rules box-c.mdc);
# 'default' sends no reasoning_effort at all, so the engine's own default (high) applies.
EFFORTS = ('none', 'low', 'high', 'xhigh', 'max', 'default')
UA = 'curl/8.5.0'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('QUALITY_DATA_DIR') or os.path.join(HERE, 'data')

# The final's measured speed on box-a (box-c/qa/box-a-ship1-20261027/README.md, FINAL column): the pre-run
# wall-time estimates every suite prints are built from these, and say so.
EST_AGG_TPS_C16 = 3990.0      # aggregate decode tok/s at 16 streams
EST_STREAM_TPS = 290.0        # per-stream decode tok/s at 16 streams
EST_PREFILL_TPS = 31000.0     # cold prefill tok/s (979,973 tokens in 31.36 s)


# ----------------------------------------------------------------------------------------------- args
def add_common_args(ap, concurrency=16):
    ap.add_argument('--base-url', default=None, help=f'engine base URL (default $ENGINE_URL, else {DEFAULT_BASE})')
    ap.add_argument('--model', default=None, help=f'served model id (default $MODEL, else {DEFAULT_MODEL})')
    ap.add_argument('--api-key', default=None, help='API key (default $VLLM_API_KEY, else --env-file)')
    ap.add_argument('--env-file', default=DEFAULT_ENVFILE, help=f'file holding VLLM_API_KEY= (default {DEFAULT_ENVFILE})')
    ap.add_argument('--out', default=None, help='output dir (default ./quality-out-<UTC stamp>)')
    ap.add_argument('--tag', default='', help='free text stored in every summary (arm name, boot id ...)')
    ap.add_argument('--concurrency', '-c', type=int, default=concurrency, help=f'requests in flight (default {concurrency})')
    ap.add_argument('--effort', default=SERVING_EFFORT, choices=EFFORTS,
                    help=f"top-level reasoning_effort (default {SERVING_EFFORT}, the serving effort; 'none' is sent "
                         "and turns thinking off; 'default' omits the field)")
    ap.add_argument('--timeout', type=float, default=1800.0, help='per-request HTTP timeout, s (default 1800)')
    return ap


class Conn:
    def __init__(self, base, model, key, timeout=1800.0):
        self.base = base.rstrip('/')
        self.model = model
        self.key = key
        self.timeout = timeout

    def headers(self):
        h = {'Content-Type': 'application/json', 'User-Agent': UA}
        if self.key:
            h['Authorization'] = 'Bearer ' + self.key
        return h


def read_key(env_file):
    try:
        with open(env_file, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('VLLM_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def conn_from_args(a):
    base = a.base_url or os.environ.get('ENGINE_URL') or DEFAULT_BASE
    model = a.model or os.environ.get('MODEL') or DEFAULT_MODEL
    key = a.api_key or os.environ.get('VLLM_API_KEY') or read_key(a.env_file)
    if not key:
        print(f'WARNING: no API key (no --api-key, $VLLM_API_KEY or VLLM_API_KEY= in {a.env_file}); '
              'requests go out unauthenticated', file=sys.stderr)
    return Conn(base, model, key, a.timeout)


def out_dir(a):
    d = a.out or os.path.join(os.getcwd(), 'quality-out-' + utc_stamp())
    os.makedirs(d, exist_ok=True)
    a.out = d
    return d


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ----------------------------------------------------------------------------------------------- HTTP
class Reply:
    __slots__ = ('status', 'body', 'error', 'wall')

    def __init__(self, status=None, body=None, error=None, wall=0.0):
        self.status, self.body, self.error, self.wall = status, body, error, wall

    @property
    def message(self):
        try:
            return self.body['choices'][0]['message'] or {}
        except Exception:
            return {}

    @property
    def content(self):
        return self.message.get('content') or ''

    @property
    def reasoning(self):
        m = self.message
        return m.get('reasoning_content') or m.get('reasoning') or ''

    @property
    def tool_calls(self):
        return self.message.get('tool_calls') or []

    @property
    def finish_reason(self):
        try:
            return self.body['choices'][0].get('finish_reason')
        except Exception:
            return None

    @property
    def usage(self):
        return (self.body or {}).get('usage') or {}

    def stats(self):
        u = self.usage
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        return {'wall_s': round(self.wall, 3), 'prompt_tokens': u.get('prompt_tokens'),
                'completion_tokens': u.get('completion_tokens'), 'cached_tokens': cached,
                'finish_reason': self.finish_reason, 'reasoning_chars': len(self.reasoning)}


def http_json(conn, method, path, body=None, timeout=None):
    """One HTTP call returning a Reply; never raises."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(conn.base + path, data=data, headers=conn.headers(), method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout or conn.timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode('utf-8', 'replace')
        except Exception:
            txt = ''
        return Reply(e.code, None, f'status: HTTP {e.code} {" ".join(txt.split())[:300]}', time.time() - t0)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError, Exception) as e:
        return Reply(None, None, f'transport: {type(e).__name__}: {e}'[:300], time.time() - t0)
    wall = time.time() - t0
    try:
        txt = raw.decode('utf-8', 'replace')
        obj = json.loads(txt) if txt.strip() else {}
    except Exception as e:
        return Reply(status, {'_raw': raw[:2000].decode('utf-8', 'replace')}, f'response: not JSON ({e})'[:300], wall)
    return Reply(status, obj, None, wall)


def chat(conn, messages, *, max_tokens, temperature=0.0, top_p=1.0, effort=SERVING_EFFORT, seed=None,
         tools=None, tool_choice=None, response_format=None, extra=None, timeout=None):
    body = {'model': conn.model, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': temperature, 'top_p': top_p, 'stream': False}
    if effort and effort != 'default':
        # SGLang DSV4.1: the TOP-LEVEL reasoning_effort sets thinking (gsm8k.py); 'none' turns it off.
        body['reasoning_effort'] = effort
    if seed is not None:
        body['seed'] = seed
    if tools:
        body['tools'] = tools
        if tool_choice:
            body['tool_choice'] = tool_choice
    if response_format is not None:
        body['response_format'] = response_format
    if extra:
        body.update(extra)
    r = http_json(conn, 'POST', '/v1/chat/completions', body, timeout)
    if r.error is None:
        try:
            r.body['choices'][0]['message']
        except Exception:
            r.error = 'response: 200 without choices[0].message: ' + json.dumps(r.body)[:200]
    return r


def server_info(conn):
    """What the engine says about itself: context length, model card, multimodal hints. GETs only."""
    info = {'context_length': None, 'sources': {}}
    r = http_json(conn, 'GET', '/v1/models', timeout=30)
    if r.error is None and isinstance(r.body, dict):
        data = r.body.get('data') or []
        info['sources']['v1_models'] = data[:4]
        for m in data:
            if m.get('max_model_len'):
                info['context_length'] = int(m['max_model_len'])
    else:
        info['sources']['v1_models_error'] = r.error
    r = http_json(conn, 'GET', '/get_server_info', timeout=30)
    if r.error is None and isinstance(r.body, dict):
        b = r.body
        keep = {k: b.get(k) for k in ('context_length', 'max_total_num_tokens', 'model_path', 'served_model_name',
                                        'grammar_backend', 'tool_call_parser', 'reasoning_parser', 'is_multimodal',
                                        'disable_multimodal', 'chat_template', 'max_running_requests', 'version')
                if k in b}
        for k in ('internal_states',):
            if isinstance(b.get(k), list) and b[k]:
                st = b[k][0]
                if isinstance(st, dict) and 'max_total_num_tokens' in st:
                    keep['max_total_num_tokens'] = st['max_total_num_tokens']
        info['sources']['get_server_info'] = keep
        if b.get('context_length'):
            cl = int(b['context_length'])
            info['context_length'] = min(info['context_length'] or cl, cl)
    else:
        info['sources']['get_server_info_error'] = r.error
    return info


# ----------------------------------------------------------------------------------------------- runner
def run_items(items, fn, *, concurrency, out_jsonl, label, est_s=None, progress_every=30.0):
    """Run fn(item) -> record (a dict with 'id') over items at `concurrency`, streaming records to out_jsonl
    (rewritten sorted by id at the end). Returns (records sorted, wall seconds). An exception inside fn is
    caught and recorded as an error record, so one bad item never kills the run."""
    n = len(items)
    lock = threading.Lock()
    recs = []
    t0 = time.time()
    last = [t0]
    tmp = out_jsonl + '.partial'
    fh = open(tmp, 'w')

    def one(it):
        try:
            rec = fn(it)
        except Exception as e:   # a bug in a scorer must not lose the run
            rec = {'id': it.get('id'), 'ok': None, 'error': f'harness: {type(e).__name__}: {e}'[:300]}
        with lock:
            recs.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            now = time.time()
            if now - last[0] >= progress_every:
                last[0] = now
                done = len(recs)
                el = now - t0
                rem = el / done * (n - done) if done else float('nan')
                errs = sum(1 for r in recs if r.get('ok') is None)
                okk = sum(1 for r in recs if r.get('ok') is True)
                print(f'[{label}] {done}/{n} done, {okk} ok, {errs} errors, {el:.0f}s elapsed, ~{rem:.0f}s left',
                      flush=True)
        return rec

    if est_s is not None:
        print(f'[{label}] {n} items at concurrency {concurrency}: expected wall ~{fmt_s(est_s)} '
              f'(estimate from the final\'s box-a speed; a slower arm takes longer)', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        list(ex.map(one, items))
    fh.close()
    wall = time.time() - t0
    recs.sort(key=lambda r: natural_key(str(r.get('id'))))
    with open(out_jsonl, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.remove(tmp)
    print(f'[{label}] finished {n} items in {fmt_s(wall)} -> {out_jsonl}', flush=True)
    return recs, wall


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def fmt_s(s):
    s = float(s)
    if s < 120:
        return f'{s:.0f}s'
    if s < 7200:
        return f'{s / 60:.1f} min'
    return f'{s / 3600:.2f} h'


def est_decode_s(n_items, mean_tokens, concurrency, prefill_tokens=0):
    """Pre-run wall estimate: decode at the final's aggregate rate (scaled down below 16 streams) plus cold
    prefill. A rough guide printed before a run, never a measurement."""
    c = max(1, min(concurrency, 16))
    agg = EST_AGG_TPS_C16 * c / 16.0 if c < 16 else EST_AGG_TPS_C16
    agg = max(agg, EST_STREAM_TPS)
    return n_items * mean_tokens / agg + n_items * prefill_tokens / EST_PREFILL_TPS


# ----------------------------------------------------------------------------------------------- stats
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2, z=Z95):
    """Difference p2 - p1 with Newcombe's hybrid score interval (1998, method 10). Returns (d, lo, hi)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, max(-1.0, lo), min(1.0, hi)


def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no more probable than the observed)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    lo, hi = max(0, c1 - r2), min(r1, c1)
    if n == 0:
        return 1.0

    def lp(x):
        return _lchoose(r1, x) + _lchoose(r2, c1 - x) - _lchoose(n, c1)
    obs = lp(a)
    tot = 0.0
    for x in range(lo, hi + 1):
        v = lp(x)
        if v <= obs + 1e-7 * abs(obs) + 1e-12:
            tot += math.exp(v)
    return min(1.0, tot)


def two_prop_z(k1, n1, k2, n2):
    """Pooled two-proportion z test: (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k2 / n2 - k1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p from the discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.exp(_lchoose(m, i) - m * math.log(2)) for i in range(k + 1))
    return min(1.0, 2 * tail)


def bootstrap_mean(xs, reps=5000, seed=1, alpha=0.05):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return float('nan'), float('nan'), float('nan')
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, m, m
    rnd = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return m, ms[int(alpha / 2 * reps)], ms[min(reps - 1, int((1 - alpha / 2) * reps))]


def bootstrap_diff_means(xa, xb, reps=5000, seed=1, alpha=0.05):
    xa = [float(x) for x in xa if x is not None]
    xb = [float(x) for x in xb if x is not None]
    if not xa or not xb:
        return float('nan'), float('nan'), float('nan')
    d = sum(xb) / len(xb) - sum(xa) / len(xa)
    rnd = random.Random(seed)
    na, nb = len(xa), len(xb)
    ds = sorted(sum(xb[rnd.randrange(nb)] for _ in range(nb)) / nb - sum(xa[rnd.randrange(na)] for _ in range(na)) / na
                for _ in range(reps))
    return d, ds[int(alpha / 2 * reps)], ds[min(reps - 1, int((1 - alpha / 2) * reps))]


def prop(k, n):
    lo, hi = wilson(k, n)
    return {'k': k, 'n': n, 'p': (k / n if n else None), 'lo': lo if n else None, 'hi': hi if n else None,
            'method': 'wilson95'}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    m, lo, hi = bootstrap_mean(xs)
    return {'n': len(xs), 'mean': m if xs else None, 'lo': lo if xs else None, 'hi': hi if xs else None,
            'method': 'bootstrap95-percentile-5000'}


# ----------------------------------------------------------------------------------------------- summary
def metric_from_records(records, key, where=None):
    """Proportion over the records that are answered (ok is not None) and carry a boolean `metrics[key]`."""
    k = n = 0
    for r in records:
        if r.get('ok') is None:
            continue
        if where is not None and not where(r):
            continue
        v = (r.get('metrics') or {}).get(key)
        if v is None:
            continue
        n += 1
        k += 1 if v else 0
    return prop(k, n)


def answered(records):
    n = len(records)
    k = sum(1 for r in records if r.get('ok') is not None)
    return prop(k, n)


def write_summary(out, suite, cell, records, *, primary, metrics, means=None, condition=None, extra=None,
                  wall_s=None, tag='', conn=None):
    """<out>/<suite>-<cell>.summary.json:
       {suite, cell, primary, tag, started/finished, wall_s, condition, n_items, errors, error_kinds,
        metrics: {name: {k, n, p, lo, hi, method}}, means: {name: {n, mean, lo, hi, method}}, ...extra}
    `metrics` always holds `answered` (non-error items / all items) besides the suite's own."""
    errs = [r for r in records if r.get('ok') is None]
    kinds = {}
    for r in errs:
        kind = (r.get('error') or 'unknown').split(':', 1)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
    m = {'answered': answered(records)}
    m.update(metrics)
    s = {'suite': suite, 'cell': cell, 'primary': primary, 'tag': tag, 'finished': utc_iso(),
         'wall_s': round(wall_s, 1) if wall_s is not None else None,
         'engine': {'base_url': conn.base, 'model': conn.model} if conn else None,
         'condition': condition or {}, 'n_items': len(records), 'errors': len(errs), 'error_kinds': kinds,
         'first_error': (errs[0].get('error') if errs else None), 'metrics': m, 'means': means or {}}
    if extra:
        s.update(extra)
    path = os.path.join(out, f'{suite}-{cell}.summary.json')
    with open(path, 'w') as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    print_summary(s)
    print(f'summary -> {path}')
    return s


def fmt_prop(m):
    if not m or not m.get('n'):
        return 'n=0'
    return f"{m['k']}/{m['n']} = {m['p']:.4f}  [{m['lo']:.4f}, {m['hi']:.4f}]"


def print_summary(s):
    print(f"== {s['suite']} / {s['cell']}  ({s['n_items']} items, {s['errors']} errors"
          f"{', ' + json.dumps(s['error_kinds']) if s['errors'] else ''}, wall {fmt_s(s['wall_s'] or 0)})")
    if s.get('first_error'):
        print(f"   first error: {s['first_error']}")
    for k, v in s['metrics'].items():
        star = ' *' if k == s.get('primary') else ''
        print(f"   {k:<28} {fmt_prop(v)}{star}")
    for k, v in (s.get('means') or {}).items():
        if v.get('n'):
            print(f"   {k:<28} mean {v['mean']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]  (n={v['n']})")


# ----------------------------------------------------------------------------------------------- answers
NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
STRICT_RE = re.compile(r'####\s*\$?\s*(-?[\d][\d,.\s\u00a0\u202f\u2009\']*)')


def norm_text(s):
    return unicodedata.normalize('NFKC', s or '')


def parse_number(tok):
    """'1,234' / '1.234' (thousands) / '1 234' / '12.5' / '-3' -> float, else None."""
    t = (tok or '').strip().replace('\u00a0', '').replace('\u202f', '').replace('\u2009', '').replace(' ', '')
    t = t.replace("'", '').rstrip('.,')
    if not t:
        return None
    if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+', t) and not re.fullmatch(r'-?\d+[.,]\d{1,2}', t):
        t = re.sub(r'[.,]', '', t)
    elif re.fullmatch(r'-?\d+,\d+', t):
        t = t.replace(',', '.')
    else:
        t = t.replace(',', '')
    try:
        return float(t)
    except ValueError:
        return None


def extract_answer(content, *, fallback=True):
    """(pred, strict). The ANSWER-line rule: `strict` is the number on the last '#### <n>' line, and `pred` is it.
    With no answer line, `strict` is None and `pred` is the last number anywhere in the content ONLY under
    fallback=True (gsm8k.py's rule, kept as the default because gsm8k_full.py / mgsm.py score `exact` with it for
    comparability; their `format` metric = strict is not None is the answer-line rate). fallback=False scores the
    answer line alone: no '####' line -> (None, None)."""
    txt = norm_text(content)
    strict = None
    ms = STRICT_RE.findall(txt)
    if ms:
        strict = parse_number(ms[-1])
    if strict is not None:
        return strict, strict
    if not fallback:
        return None, None
    nums = NUM_RE.findall(txt)
    return (parse_number(nums[-1]) if nums else None), None


def num_eq(a, b, tol=1e-4):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def load_or_download(path, url, timeout=120):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        with open(path + '.tmp', 'wb') as f:
            f.write(raw)
        os.replace(path + '.tmp', path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
