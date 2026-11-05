#!/usr/bin/env python3
"""fake_server.py -- an offline stand-in for the engine, for testing this suite on the Mac. NEVER a model.

A local OpenAI-compatible server (stdlib) that answers every suite's requests as an ORACLE: it rebuilds the
suites' own items (gsm8k/MGSM from the cached data files, needles by reading the planted records, JSON asks,
agent sessions and images from their generators) and answers them right, so a clean run must score ~100%.
--bad-rate P makes a deterministic share P of the answers wrong in each suite's own way (a wrong number,
broken JSON, broken tool-call arguments, a wrong key), so the scorers are seen to count failures; --no-json
and --no-images refuse response_format / image parts with HTTP 400 like a server without them;
--error-rate P answers a share P with HTTP 500 (scored as errors, not wrong answers).

    python3 fake_server.py --port 18002 --key test-key [--ctx 1048576] [--bad-rate 0] [--latency 0]

Endpoints: GET /v1/models, /get_server_info, /metrics; POST /v1/chat/completions, /flush_cache.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402
import agent_sessions  # noqa: E402
import images  # noqa: E402
import json_format  # noqa: E402
import mgsm  # noqa: E402
import gsm8k_full  # noqa: E402

A = None
RUNNING = [0]
LOCK = threading.Lock()
IDX = {}
SEEN = {}


def build_index():
    q = {}
    p = os.path.join(qlib.DATA_DIR, 'gsm8k_test.jsonl')
    if os.path.exists(p):
        for it in gsm8k_full.load_items(None, p):
            q[it['question'].strip()] = it['gold']
    for lang in mgsm.INSTR:
        p = os.path.join(qlib.DATA_DIR, f'mgsm_{lang}.tsv')
        if os.path.exists(p):
            for it in mgsm.load_items([lang], 10 ** 6):
                q[it['question'].strip()] = it['gold']
    IDX['qa'] = q
    IDX['json'] = {it['prompt']: it for cell in ('schema', 'object') for it in json_format.make_items(200, cell)}
    IDX['agent'] = {s['steps'][0]['user']: s for s in agent_sessions.make_sessions(200)}
    img = {}
    for it in images.make_items(40, 0):
        img[hashlib.sha256(it['png']).hexdigest()] = (it['family'], it['gold'])
    IDX['img'] = img


def bad(text):
    """Deterministic: is this request one of the --bad-rate share?"""
    if A.bad_rate <= 0:
        return False
    h = int(hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < A.bad_rate


def flat(content):
    if isinstance(content, str):
        return content
    return '\n'.join(p.get('text', '') for p in content or [] if isinstance(p, dict) and p.get('type') == 'text')


def gen_json_value(s):
    """A schema-valid instance (used for the json_object probe and any unindexed ask)."""
    t = s.get('type')
    if 'enum' in s:
        return s['enum'][0]
    if t == 'object':
        return {k: gen_json_value(v) for k, v in (s.get('properties') or {}).items()}
    if t == 'array':
        return [gen_json_value(s.get('items') or {'type': 'string'}) for _ in range(max(1, s.get('minItems', 1)))]
    if t == 'integer':
        return int(s.get('minimum', 1))
    if t == 'number':
        return float(s.get('minimum', 0))
    if t == 'boolean':
        return True
    return 'x'


def json_answer(it):
    g, fam = it['gold'], it['family']
    if fam == 'person':
        return g
    if fam == 'invoice':
        n = g['n_items']
        per = round(g['total'] / n, 2)
        items = [{'name': f'item{i}', 'quantity': 1, 'unit_price': per} for i in range(n)]
        return {'customer': g['customer'], 'items': items, 'total': g['total']}
    if fam == 'event':
        return g
    if fam == 'sentiment':
        return {'sentiment': g['sentiment'], 'confidence': 0.9, 'keywords': ['k']}
    if fam == 'config':
        return {'service': {'name': g['name'], 'port': g['port'], 'replicas': g['replicas']}, 'env': g['env'],
                'tags': ['a', 'b']}
    return {}


def agent_reply(sess, msgs, is_bad):
    users = [i for i, m in enumerate(msgs) if m.get('role') == 'user']
    step = len(users) - 1
    st = sess['steps'][step]
    since = msgs[users[-1] + 1:]
    k = sum(1 for m in since if m.get('role') == 'assistant' and m.get('tool_calls'))
    calls = st['oracle']['calls']
    if k < len(calls):
        c = calls[k]
        args = json.dumps(c['arguments'])
        if is_bad:
            args = args[:-1]           # truncated JSON: an invalid call
        return {'role': 'assistant', 'content': None, 'reasoning_content': 'need a tool',
                'tool_calls': [{'id': f'call_{step}_{k}', 'type': 'function',
                                'function': {'name': c['name'], 'arguments': args}}]}, 'tool_calls'
    ans = st['oracle']['answer']
    if is_bad:
        ans = 'I am not sure.\nANSWER: unknown'
    return {'role': 'assistant', 'content': ans, 'reasoning_content': 'done'}, 'stop'


def answer(body):
    msgs = body.get('messages') or []
    last_user = next((m for m in reversed(msgs) if m.get('role') == 'user'), {})
    content = last_user.get('content')
    text = flat(content)
    allt = ''.join(flat(m.get('content')) for m in msgs)
    is_bad = bad(allt)
    # images
    if isinstance(content, list) and any(p.get('type') == 'image_url' for p in content if isinstance(p, dict)):
        if A.no_images:
            return 400, {'object': 'error', 'message': 'image input is not supported by this model'}
        url = next(p['image_url']['url'] for p in content if p.get('type') == 'image_url')
        png = base64.b64decode(url.split(',', 1)[1])
        key = hashlib.sha256(png).hexdigest()
        is_bad = bad(key)          # per image: a family's items share one question text
        fam, gold = IDX['img'].get(key, ('color', 'red'))
        if is_bad and not text.startswith('What colour is the square in the image'):   # never spoil the probe
            gold = {'count': '9', 'shape': 'hexagon', 'color': 'purple', 'position': 'centre'}[fam]
        return 200, ({'role': 'assistant', 'content': f'The answer is {gold}.'}, 'stop')
    if body.get('tools'):
        first = next((m for m in msgs if m.get('role') == 'user'), {})
        sess = IDX['agent'].get(flat(first.get('content')))
        if sess:
            # per session, keyed on its id: hashing the growing transcript would redraw every turn
            return 200, agent_reply(sess, msgs, bad(sess['id']))
        return 200, ({'role': 'assistant', 'content': 'No session.'}, 'stop')
    rf = body.get('response_format')
    if rf:
        if A.no_json:
            return 400, {'object': 'error', 'message': 'response_format is not supported'}
        it = IDX['json'].get(text)
        if it:
            obj = json_answer(it)
            txt = json.dumps(obj)
            if is_bad:
                txt = txt[:-1] if rf.get('type') == 'json_object' else json.dumps({'wrong': True})
            return 200, ({'role': 'assistant', 'content': txt}, 'stop')
        sch = (rf.get('json_schema') or {}).get('schema') or {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
        return 200, ({'role': 'assistant', 'content': json.dumps(gen_json_value(sch))}, 'stop')
    if 'access code for key' in text:
        keys = re.findall(r'K-([0-9A-F]{6})', text.split('</document>')[-1])
        lines = []
        for k in keys:
            m = re.search(r'key K-' + k + r' is (\d{7})', text)
            v = m.group(1) if m else '0000000'
            if is_bad and k == keys[0]:
                v = str(int(v) + 1)
            lines.append(f'K-{k}: {v}')
        return 200, ({'role': 'assistant', 'content': '\n'.join(lines)}, 'stop')
    q = text.split('\n\n')[0].strip()
    gold = IDX['qa'].get(q)
    if gold is None:
        q2 = text.rsplit('\n\n', 1)[0].strip()
        gold = IDX['qa'].get(q2)
    if gold is not None:
        g = gold + (1 if is_bad else 0)
        gs = str(int(g)) if float(g).is_integer() else str(g)
        return 200, ({'role': 'assistant', 'content': f'Working it out step by step.\n#### {gs}',
                      'reasoning_content': 'thinking'}, 'stop')
    return 200, ({'role': 'assistant', 'content': 'Fine. ' * 20}, 'length' if body.get('max_tokens', 99) < 64 else 'stop')


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send(self, code, obj, ctype='application/json'):
        data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authed(self):
        if self.headers.get('Authorization') != 'Bearer ' + A.key:
            self.send(401, {'error': 'bad key'})
            return False
        return True

    def do_GET(self):
        if not self.authed():
            return
        if self.path == '/v1/models':
            return self.send(200, {'object': 'list', 'data': [{'id': A.model, 'object': 'model',
                                                                'max_model_len': A.ctx}]})
        if self.path == '/get_server_info':
            return self.send(200, {'context_length': A.ctx, 'served_model_name': A.model, 'grammar_backend': 'fake',
                                   'internal_states': [{'max_total_num_tokens': 1234567}]})
        if self.path == '/metrics':
            return self.send(200, f'# fake\nsglang:num_running_reqs{{model_name="{A.model}"}} {RUNNING[0]}.0\n',
                             'text/plain')
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(n)
        if not self.authed():
            return
        if self.path == '/flush_cache':
            return self.send(200, 'ok', 'text/plain')
        if self.path != '/v1/chat/completions':
            return self.send(404, {'error': 'not found'})
        body = json.loads(raw)
        if body.get('model') != A.model:
            return self.send(404, {'object': 'error', 'message': f'model {body.get("model")} not found'})
        with LOCK:
            RUNNING[0] += 1
        try:
            if A.latency:
                time.sleep(A.latency)
            if A.error_rate:
                key = hashlib.sha256(raw).hexdigest()
                with LOCK:
                    SEEN[key] = SEEN.get(key, 0) + 1      # a retry of the same body draws again
                    tries = SEEN[key]
                h = int(hashlib.sha256(f'err{key}{tries}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                if h < A.error_rate:
                    return self.send(500, {'object': 'error', 'message': 'fake engine error'})
            code, res = answer(body)
            if code != 200:
                return self.send(code, res)
            msg, fin = res
            chars = sum(len(json.dumps(m.get('content'))) for m in body.get('messages') or [])
            out = {'id': 'fake', 'object': 'chat.completion', 'model': A.model,
                   'choices': [{'index': 0, 'message': msg, 'finish_reason': fin}],
                   'usage': {'prompt_tokens': chars // 4, 'completion_tokens': len(msg.get('content') or '') // 4 + 5,
                             'prompt_tokens_details': {'cached_tokens': 0}}}
            self.send(200, out)
        finally:
            with LOCK:
                RUNNING[0] -= 1


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18002)
    ap.add_argument('--key', default='test-key')
    ap.add_argument('--model', default=qlib.DEFAULT_MODEL)
    ap.add_argument('--ctx', type=int, default=1048576)
    ap.add_argument('--bad-rate', type=float, default=0.0)
    ap.add_argument('--error-rate', type=float, default=0.0)
    ap.add_argument('--latency', type=float, default=0.0)
    ap.add_argument('--no-json', action='store_true')
    ap.add_argument('--no-images', action='store_true')
    A = ap.parse_args()
    build_index()
    srv = ThreadingHTTPServer(('127.0.0.1', A.port), H)
    srv.daemon_threads = True
    print(f'fake server on 127.0.0.1:{A.port} (oracle; bad-rate {A.bad_rate})', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
