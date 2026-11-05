#!/usr/bin/env python3
"""agent_sessions.py -- scripted multi-turn agent sessions with OpenAI tools; the harness plays the tools.

100 sessions by default (--n), 20 of each of five families, every one deterministic from a seed: the user
asks 2-3 things in turn, the model has the same 10 tools in every session (so picking the right one is part
of the test), and a simulated world answers each tool call from the session's own state:

  orders    lookup_order -> status + ETA; then refund the most expensive item (refund_item, right sku)
  currency  get_exchange_rate -> convert; convert a second amount; then chain into a third currency
  files     list_dir + read_file -> which file holds the TODO and what it says; then write_file a DONE note
  weather   get_weather x3 -> warmest; warmest-coldest gap in Fahrenheit; then a 4th city -> coldest of four
  calendar  list_events -> list the day; then create_event at the earliest free half-hour slot

A session is 4-8 assistant turns when the model works as intended (each step is >= 1 tool round plus an
answer). The system prompt asks for a final line `ANSWER: <answer>` on each step's answer; the answer check
reads that line (the whole content when the line is missing, `answer_line` says so; a number answer is then
the reply's LAST number, never any number it names). A booking is proved by the world state, and the reply
must name its start time in any clock form (9:30, 09:30, 9:30 AM).

Scored per session (primary: `session_success`), plus pooled rates:
  call_valid       a tool call names a real tool and its arguments parse as a JSON object that satisfies
                   the tool's schema (required keys, types, enums) -- pooled over every call
  step_tools       every REQUIRED call of the step (tool + key arguments) was made during that step
  step_answer      the step's answer is right (a number within tolerance, the right city/file/TODO text, or
                   the world state: the file written with the exact line, the event booked at the right time)
  session_success  every step's tools and answer right, and every call valid
A failed request ends the session as an error (ok=None), never as a wrong answer. Sampling defaults to prod
(T 1.0 / top_p 1.0) at effort high, the condition agents run under; the assistant turn is sent back as
returned (content, tool_calls and reasoning_content; --strip-reasoning drops the last).

    python3 agent_sessions.py --out OUT [--n 100] [-c 16] [--temperature 1.0]

Writes OUT/agent-<cell>.jsonl (each session's transcript of calls, tool results and answers; cell = prod or
t0) and OUT/agent-<cell>.summary.json. Expected wall on the final: ~4-8 min (sessions are sequential turns).
"""
import argparse
import ast
import datetime
import json
import operator
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

SYSTEM = ('You are a careful assistant with tools. Use the tools to look up any fact a tool can provide; never '
          'guess such values. Call tools with exactly the parameters their schemas define. When you have the final '
          'answer to the user\'s latest request, reply in plain text and end with one line of the form '
          '"ANSWER: <answer>".')


def _fn(name, desc, props, required):
    return {'type': 'function', 'function': {'name': name, 'description': desc, 'parameters': {
        'type': 'object', 'properties': props, 'required': required}}}


S = {'type': 'string'}
TOOLS = [
    _fn('lookup_order', 'Look up an order by id: status, ETA and line items.', {'order_id': S}, ['order_id']),
    _fn('refund_item', 'Refund one item (by sku) of an order.', {'order_id': S, 'sku': S, 'reason': S},
        ['order_id', 'sku', 'reason']),
    _fn('get_exchange_rate', 'Exchange rate: how many units of `quote` one unit of `base` buys.',
        {'base': {'type': 'string', 'description': 'ISO currency code, e.g. EUR'},
         'quote': {'type': 'string', 'description': 'ISO currency code, e.g. USD'}}, ['base', 'quote']),
    _fn('calculator', 'Evaluate an arithmetic expression (+ - * / ** and parentheses).', {'expression': S},
        ['expression']),
    _fn('list_dir', 'List the entries of a directory in the workspace.', {'path': S}, ['path']),
    _fn('read_file', 'Read a text file from the workspace.', {'path': S}, ['path']),
    _fn('write_file', 'Create or overwrite a text file in the workspace.', {'path': S, 'content': S},
        ['path', 'content']),
    _fn('get_weather', 'Current weather for a city.',
        {'city': S, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, ['city']),
    _fn('list_events', 'List calendar events on a date (YYYY-MM-DD).', {'date': S}, ['date']),
    _fn('create_event', 'Create a calendar event. Times are 24h HH:MM.',
        {'title': S, 'date': S, 'start': S, 'end': S}, ['title', 'date', 'start', 'end']),
]
TOOL_SCHEMAS = {t['function']['name']: t['function']['parameters'] for t in TOOLS}

PRODUCTS = ['ceramic mug', 'wool scarf', 'desk lamp', 'notebook set', 'water bottle', 'phone case', 'tea sampler',
            'bike light', 'cutting board', 'umbrella', 'backpack', 'candle set']
USD = {'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'JPY': 0.0067, 'CHF': 1.12, 'CAD': 0.73, 'AUD': 0.66, 'SEK': 0.094}
CITIES = ['Oslo', 'Lima', 'Cairo', 'Hanoi', 'Denver', 'Perth', 'Quito', 'Dublin', 'Nairobi', 'Seoul', 'Tunis',
          'Bogota', 'Helsinki', 'Manila', 'Lisbon', 'Reno']
CONDS = ['clear', 'cloudy', 'light rain', 'windy', 'fog', 'snow showers']
TODOS = ['rotate the staging API keys', 'add retry to the upload client', 'remove the legacy CSV exporter',
         'write tests for the date parser', 'bump the request timeout to 30 seconds', 'split the config loader',
         'document the cache eviction policy', 'drop support for Python 3.8']
FILES = {
    'README.md': '# proj\n\nA small service. See main.py for the entry point.\n',
    'main.py': 'from utils import load\n\n\ndef main():\n    cfg = load("config.yaml")\n    print(cfg)\n',
    'utils.py': 'import yaml\n\n\ndef load(path):\n    with open(path) as f:\n        return yaml.safe_load(f)\n',
    'notes.txt': 'Meeting notes: agreed to keep the API stable through Q3.\n',
    'config.yaml': 'port: 8080\nworkers: 4\n',
    'client.py': 'import requests\n\n\ndef upload(url, data):\n    return requests.post(url, data=data)\n',
}
EVENT_TITLES = ['Standup', 'Design review', 'Vendor call', '1:1 with Sam', 'Budget sync', 'Interview loop',
                'Lunch & learn', 'Ops handoff']
PROJECTS = ['atlas', 'beacon', 'cinder', 'dune', 'ember', 'fjord', 'grove', 'harbor']
NEW_TITLES = ['Roadmap chat', 'Pairing session', 'Customer debrief', 'Hiring sync', 'Security review']


# ----------------------------------------------------------------------------------------------- matching
def _ieq(a, b):
    return isinstance(a, str) and a.strip().lower() == str(b).strip().lower()


def _path(p):
    if not isinstance(p, str):
        return None
    p = p.strip()
    if not p.startswith('/'):
        p = '/' + p
    return p.rstrip('/') or '/'


def _time(t):
    m = re.fullmatch(r'\s*(\d{1,2}):(\d{2})\s*', str(t))
    return f'{int(m.group(1)):02d}:{m.group(2)}' if m else None


def match_arg(val, spec):
    kind, want = next(iter(spec.items()))
    if kind == 'ieq':
        return _ieq(val, want)
    if kind == 'path':
        return _path(val) == _path(want)
    if kind == 'time':
        return _time(val) == _time(want)
    if kind == 'contains':
        return isinstance(val, str) and want.lower() in val.lower()
    if kind == 'strip_eq':
        return isinstance(val, str) and val.strip() == want.strip()
    raise ValueError(kind)


def call_matches(call, exp):
    if call['name'] != exp['tool'] or not isinstance(call.get('args'), dict):
        return False
    for alt in exp.get('alts') or [exp.get('args') or {}]:
        if all(match_arg(call['args'].get(k), sp) for k, sp in alt.items()):
            return True
    return False


def answer_text(content):
    lines = [l for l in (content or '').splitlines() if re.match(r'\s*\**\s*ANSWER\s*\**\s*:', l, re.I)]
    if not lines:
        return None
    return re.sub(r'^\s*\**\s*ANSWER\s*\**\s*:\s*', '', lines[-1], flags=re.I).strip()


def _numbers(txt):
    out = []
    for tok in re.findall(r'-?\d[\d,]*(?:\.\d+)?', txt or ''):
        v = qlib.parse_number(tok)
        if v is not None:
            out.append(v)
    return out


_TIME_RE = re.compile(r'(?<![\d:])(\d{1,2})(?:[:.h](\d{2}))?\s*([ap])\.?\s*m\b\.?'      # 9:30 AM, 10am, 1.00 p.m.
                      r'|(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])', re.I)                 # 9:30, 09:30


def _times(txt):
    """Every clock time named in the text, as 24h HH:MM: 9:30, 09:30, 9:30 AM, 1:00 p.m., 10am, noon."""
    out = []
    for g in _TIME_RE.findall(txt or ''):
        h, mi, ap = int(g[0] or g[3]), int(g[1] or g[4] or 0), g[2].lower()
        if ap == 'p' and h < 12:
            h += 12
        elif ap == 'a' and h == 12:
            h = 0
        if h < 24 and mi < 60:
            out.append(f'{h:02d}:{mi:02d}')
    if re.search(r'\bnoon\b', txt or '', re.I):
        out.append('12:00')
    return out


def check_answer(spec, text, world, whole=False):
    """`text` is the ANSWER line's value, or (whole=True) the whole final reply when it has no ANSWER line."""
    t = qlib.norm_text(text or '')
    low = t.lower()
    kind = spec['type']
    if kind == 'all_contains':
        return all(any(v.lower() in low for v in group) for group in spec['values'])
    if kind == 'number':
        nums = _numbers(t)
        if whole:          # no ANSWER line: only the reply's last number is its answer, not any number it names
            nums = nums[-1:]
        return any(abs(x - spec['value']) <= spec['tol'] for x in nums)
    if kind == 'file':
        c = world.fs.get(spec['path'])
        return c is not None and c.strip() == spec['content'].strip()
    if kind == 'event':
        ok = any(_ieq(e['title'], spec['title']) and e['start'] == spec['start'] and e['end'] == spec['end']
                 for e in world.events.get(spec['date'], []) if e.get('created'))
        # the world state proves the booking; the reply must name its start in any clock form (9:30, 9:30 AM)
        return ok and _time(spec['start']) in _times(t)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------- world
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError('exponent too large')
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError('unsupported expression')
    return ev(ast.parse(expr.replace('^', '**').replace(',', ''), mode='eval'))


def rate(base, quote):
    return round(USD[base] / USD[quote], 6)


class World:
    def __init__(self, st):
        self.orders = json.loads(json.dumps(st.get('orders', {})))
        self.fs = dict(st.get('fs', {}))
        self.weather = dict(st.get('weather', {}))
        self.events = json.loads(json.dumps(st.get('events', {})))
        self.refunds = []

    def call(self, name, args):
        """-> result string (JSON) for the tool message."""
        h = getattr(self, 't_' + name, None)
        if h is None:
            return json.dumps({'error': f'unknown tool {name}'})
        try:
            return json.dumps(h(**args))
        except TypeError as e:
            return json.dumps({'error': f'bad arguments: {e}'})
        except Exception as e:
            return json.dumps({'error': str(e)})

    def t_lookup_order(self, order_id):
        o = self.orders.get(str(order_id).strip().lstrip('#').upper())
        if not o:
            return {'error': f'no order {order_id}'}
        return {'order_id': o['id'], 'status': o['status'], 'eta': o['eta'],
                'items': [{'sku': i['sku'], 'name': i['name'], 'unit_price': i['price'], 'quantity': 1}
                          for i in o['items']]}

    def t_refund_item(self, order_id, sku, reason):
        o = self.orders.get(str(order_id).strip().lstrip('#').upper())
        if not o:
            return {'error': f'no order {order_id}'}
        it = next((i for i in o['items'] if i['sku'].lower() == str(sku).strip().lower()), None)
        if not it:
            return {'error': f'order {o["id"]} has no sku {sku}'}
        self.refunds.append({'order_id': o['id'], 'sku': it['sku'], 'reason': reason})
        return {'refund_id': 'RF-' + o['id'][1:] + it['sku'][-2:], 'sku': it['sku'], 'amount': it['price'],
                'currency': 'USD', 'status': 'approved'}

    def t_get_exchange_rate(self, base, quote):
        b, q = str(base).strip().upper(), str(quote).strip().upper()
        if b not in USD or q not in USD:
            return {'error': f'unknown currency pair {base}/{quote}'}
        return {'base': b, 'quote': q, 'rate': rate(b, q)}

    def t_calculator(self, expression):
        v = safe_eval(str(expression))
        return {'expression': expression, 'result': round(v, 10)}

    def t_list_dir(self, path):
        p = _path(path)
        names = sorted(k[len(p) + 1:] for k in self.fs if k.startswith(p + '/') and '/' not in k[len(p) + 1:])
        if not names:
            return {'error': f'no such directory {path}'}
        return {'path': p, 'entries': names}

    def t_read_file(self, path):
        p = _path(path)
        if p not in self.fs:
            return {'error': f'no such file {path}'}
        return {'path': p, 'content': self.fs[p]}

    def t_write_file(self, path, content):
        p = _path(path)
        if not p.startswith('/work/'):
            return {'error': 'writes are allowed under /work only'}
        self.fs[p] = str(content)
        return {'path': p, 'bytes': len(str(content).encode()), 'written': True}

    def t_get_weather(self, city, unit='celsius'):
        key = next((c for c in self.weather if c.lower() == str(city).strip().lower()), None)
        if key is None:
            return {'error': f'unknown city {city}'}
        c, cond = self.weather[key]
        if str(unit).lower() == 'fahrenheit':
            return {'city': key, 'temperature': round(c * 9 / 5 + 32, 1), 'unit': 'fahrenheit', 'conditions': cond}
        if str(unit).lower() not in ('celsius', ''):
            return {'error': f'unknown unit {unit}'}
        return {'city': key, 'temperature': c, 'unit': 'celsius', 'conditions': cond}

    def t_list_events(self, date):
        d = str(date).strip()
        evs = sorted(self.events.get(d, []), key=lambda e: e['start'])
        return {'date': d, 'events': [{'title': e['title'], 'start': e['start'], 'end': e['end']} for e in evs]}

    def t_create_event(self, title, date, start, end):
        d, s, e = str(date).strip(), _time(start), _time(end)
        if not s or not e or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            return {'error': 'date must be YYYY-MM-DD and times HH:MM'}
        if e <= s:
            return {'error': 'end must be after start'}
        for ev in self.events.get(d, []):
            if s < ev['end'] and ev['start'] < e:
                return {'error': f'conflicts with "{ev["title"]}" {ev["start"]}-{ev["end"]}'}
        self.events.setdefault(d, []).append({'title': str(title), 'start': s, 'end': e, 'created': True})
        return {'created': True, 'title': title, 'date': d, 'start': s, 'end': e}


# ----------------------------------------------------------------------------------------------- families
def _hm(i):          # half-hour index from 09:00 -> HH:MM
    m = 9 * 60 + 30 * i
    return f'{m // 60:02d}:{m % 60:02d}'


def _date_variants(iso):
    y, m, d = iso.split('-')
    mon = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
           'November', 'December'][int(m) - 1]
    return [iso, f'{mon} {int(d)}', f'{int(d)} {mon}', f'{mon[:3]} {int(d)}', f'{int(d)} {mon[:3]}',
            f'{int(m)}/{int(d)}', f'{int(d)}/{int(m)}']


def fam_orders(rnd, sid):
    oid = f'A{10000 + sid * 397 + rnd.randint(0, 396)}'   # unique per session (sid < 226)
    prods = rnd.sample(PRODUCTS, 3)
    prices = rnd.sample([9.99, 14.5, 19.99, 24.0, 32.75, 45.0, 59.99, 74.25], 3)
    items = [{'sku': f'SKU-{rnd.randint(1000, 9999)}', 'name': p, 'price': pr} for p, pr in zip(prods, prices)]
    status = rnd.choice(['shipped', 'processing'])
    eta = f'2026-11-{rnd.randint(3, 27):02d}'
    top = max(items, key=lambda i: i['price'])
    state = {'orders': {oid: {'id': oid, 'status': status, 'eta': eta, 'items': items}}}
    steps = [
        {'user': f"What's the status of my order {oid}, and when is it expected to arrive?",
         'expect': [{'tool': 'lookup_order', 'args': {'order_id': {'ieq': oid}}, 'required': True}],
         'check': {'type': 'all_contains', 'values': [[status], _date_variants(eta)]},
         'oracle': {'calls': [{'name': 'lookup_order', 'arguments': {'order_id': oid}}],
                    'answer': f'Your order is {status}; expected {eta}.\nANSWER: {status}, {eta}'}},
        {'user': 'Please refund the most expensive item in that order (by unit price). Reason: arrived damaged. '
                 'How much will I get back?',
         'expect': [{'tool': 'refund_item', 'args': {'order_id': {'ieq': oid}, 'sku': {'ieq': top['sku']},
                                                     'reason': {'contains': 'damage'}}, 'required': True}],
         'check': {'type': 'number', 'value': top['price'], 'tol': 0.005},
         'oracle': {'calls': [{'name': 'refund_item', 'arguments': {'order_id': oid, 'sku': top['sku'],
                                                                    'reason': 'arrived damaged'}}],
                    'answer': f'Refund approved.\nANSWER: {top["price"]:.2f} USD'}},
    ]
    return state, steps


def fam_currency(rnd, sid):
    base, quote, q2 = rnd.sample([c for c in USD if c != 'JPY'], 3)
    amt, amt2 = rnd.randint(120, 4800), rnd.randint(50, 900)
    r1, r2 = rate(base, quote), rate(quote, q2)
    g1, g2 = round(amt * r1, 2), round(amt2 * r1, 2)
    g3 = round(amt2 * r1 * r2, 2)

    def tol(g):
        return max(0.02, 0.002 * abs(g))
    steps = [
        {'user': f'How much is {amt} {base} in {quote} at the current rate?',
         'expect': [{'tool': 'get_exchange_rate', 'args': {'base': {'ieq': base}, 'quote': {'ieq': quote}},
                     'required': True}],
         'check': {'type': 'number', 'value': g1, 'tol': tol(g1)},
         'oracle': {'calls': [{'name': 'get_exchange_rate', 'arguments': {'base': base, 'quote': quote}}],
                    'answer': f'ANSWER: {g1:.2f} {quote}'}},
        {'user': f'And what would {amt2} {base} be in {quote}?',
         'expect': [],
         'check': {'type': 'number', 'value': g2, 'tol': tol(g2)},
         'oracle': {'calls': [], 'answer': f'ANSWER: {g2:.2f} {quote}'}},
        {'user': f'Now convert that {quote} amount (the {amt2} {base} one) into {q2}.',
         'expect': [{'tool': 'get_exchange_rate', 'required': True,
                     'alts': [{'base': {'ieq': quote}, 'quote': {'ieq': q2}},
                              {'base': {'ieq': base}, 'quote': {'ieq': q2}}]}],
         'check': {'type': 'number', 'value': g3, 'tol': tol(g3)},
         'oracle': {'calls': [{'name': 'get_exchange_rate', 'arguments': {'base': quote, 'quote': q2}}],
                    'answer': f'ANSWER: {g3:.2f} {q2}'}},
    ]
    return {}, steps


def fam_files(rnd, sid):
    names = ['README.md'] + rnd.sample([k for k in FILES if k != 'README.md'], 4)
    target = rnd.choice(names[1:])
    todo = rnd.choice(TODOS)
    root = f'/work/{PROJECTS[sid % len(PROJECTS)]}-{sid:03d}'   # unique per session
    fs = {}
    for n in names:
        c = FILES[n]
        if n == target:
            lines = c.rstrip('\n').split('\n')
            pre = '# ' if n.endswith(('.py', '.yaml')) else ''
            lines.insert(rnd.randint(1, len(lines)), f'{pre}TODO: {todo}')
            c = '\n'.join(lines) + '\n'
        fs[root + '/' + n] = c
    line = f'Resolved: {todo}'
    state = {'fs': fs}
    steps = [
        {'user': f'Which file in {root} contains a TODO, and what exactly does the TODO say? Quote it verbatim.',
         'expect': [{'tool': 'read_file', 'args': {'path': {'path': root + '/' + target}}, 'required': True}],
         'check': {'type': 'all_contains', 'values': [[target], [todo]]},
         'oracle': {'calls': [{'name': 'list_dir', 'arguments': {'path': root}},
                              {'name': 'read_file', 'arguments': {'path': root + '/' + target}}],
                    'answer': f'ANSWER: {target}: "TODO: {todo}"'}},
        {'user': f'Create {root}/DONE.md whose entire content is exactly this one line: {line}',
         'expect': [{'tool': 'write_file', 'args': {'path': {'path': root + '/DONE.md'}, 'content': {'strip_eq': line}},
                     'required': True}],
         'check': {'type': 'file', 'path': root + '/DONE.md', 'content': line},
         'oracle': {'calls': [{'name': 'write_file', 'arguments': {'path': root + '/DONE.md', 'content': line + '\n'}}],
                    'answer': f'Done.\nANSWER: {root}/DONE.md written'}},
    ]
    return state, steps


def fam_weather(rnd, sid):
    cs = rnd.sample(CITIES, 4)
    temps = rnd.sample(range(-6, 36), 4)
    weather = {c: (t, rnd.choice(CONDS)) for c, t in zip(cs, temps)}
    three = cs[:3]
    t3 = {c: weather[c][0] for c in three}
    warm = max(three, key=lambda c: t3[c])
    cold3 = min(three, key=lambda c: t3[c])
    gap_f = round((t3[warm] - t3[cold3]) * 1.8, 1)
    cold4 = min(cs, key=lambda c: weather[c][0])
    steps = [
        {'user': f'Which is warmest right now: {three[0]}, {three[1]} or {three[2]}? Use Celsius.',
         'expect': [{'tool': 'get_weather', 'args': {'city': {'ieq': c}}, 'required': True} for c in three],
         'check': {'type': 'all_contains', 'values': [[warm]]},
         'oracle': {'calls': [{'name': 'get_weather', 'arguments': {'city': c, 'unit': 'celsius'}} for c in three],
                    'answer': f'ANSWER: {warm}'}},
        {'user': 'How many Fahrenheit degrees warmer is the warmest of those three than the coldest of them? '
                 '(The size of the gap, in Fahrenheit degrees.)',
         'expect': [],
         'check': {'type': 'number', 'value': gap_f, 'tol': 0.11},
         'oracle': {'calls': [], 'answer': f'ANSWER: {gap_f}'}},
        {'user': f'Also check {cs[3]}. Of all four cities, which is the coldest right now? Give just the city.',
         'expect': [{'tool': 'get_weather', 'args': {'city': {'ieq': cs[3]}}, 'required': True}],
         'check': {'type': 'all_contains', 'values': [[cold4]]},
         'oracle': {'calls': [{'name': 'get_weather', 'arguments': {'city': cs[3], 'unit': 'celsius'}}],
                    'answer': f'ANSWER: {cold4}'}},
    ]
    return {'weather': weather}, steps


def fam_calendar(rnd, sid):
    date = (datetime.date(2026, 11, 2) + datetime.timedelta(days=sid)).isoformat()   # unique per session
    free = rnd.randint(1, 6)                 # the first free half-hour index (>= 1: 09:00 is taken)
    titles = rnd.sample(EVENT_TITLES, 4)
    evs = []
    # occupy [0, free) with 1-2 events
    cut = rnd.randint(1, free) if free > 1 else free
    evs.append({'title': titles[0], 'start': _hm(0), 'end': _hm(cut)})
    if cut < free:
        evs.append({'title': titles[1], 'start': _hm(cut), 'end': _hm(free)})
    a = free + 1 + rnd.randint(0, 2)
    evs.append({'title': titles[2], 'start': _hm(a), 'end': _hm(a + rnd.randint(1, 3))})
    new = rnd.choice(NEW_TITLES)
    s, e = _hm(free), _hm(free + 1)
    steps = [
        {'user': f'What meetings do I have on {date}?',
         'expect': [{'tool': 'list_events', 'args': {'date': {'ieq': date}}, 'required': True}],
         'check': {'type': 'all_contains', 'values': [[ev['title']] for ev in evs], 'whole_ok': True},
         'oracle': {'calls': [{'name': 'list_events', 'arguments': {'date': date}}],
                    'answer': 'You have: ' + '; '.join(f'{ev["title"]} {ev["start"]}-{ev["end"]}' for ev in evs)
                              + '\nANSWER: ' + ', '.join(ev['title'] for ev in evs)}},
        {'user': f'Book a 30-minute meeting titled "{new}" on {date} in the earliest free slot that starts at or '
                 f'after 09:00 (slots start on the hour or half hour). What time did you book?',
         'expect': [{'tool': 'create_event', 'args': {'title': {'ieq': new}, 'date': {'ieq': date},
                                                      'start': {'time': s}, 'end': {'time': e}}, 'required': True}],
         'check': {'type': 'event', 'date': date, 'title': new, 'start': s, 'end': e},
         'oracle': {'calls': [{'name': 'create_event', 'arguments': {'title': new, 'date': date, 'start': s,
                                                                     'end': e}}],
                    'answer': f'Booked {s}-{e}.\nANSWER: {s}'}},
    ]
    return {'events': {date: evs}}, steps


FAMILIES = [('orders', fam_orders), ('currency', fam_currency), ('files', fam_files), ('weather', fam_weather),
            ('calendar', fam_calendar)]


def make_sessions(n=100, seed=20261029):
    out = []
    for i in range(n):
        fam, f = FAMILIES[i % len(FAMILIES)]
        rnd = random.Random(f'{seed}-{i}')
        state, steps = f(rnd, i)
        out.append({'id': f'agent-{i:03d}', 'family': fam, 'state': state, 'steps': steps})
    return out


# ----------------------------------------------------------------------------------------------- driving
def parse_call(tc):
    fn = tc.get('function') or {}
    name = fn.get('name')
    raw = fn.get('arguments')
    args, why = None, None
    if isinstance(raw, dict):
        args = raw
    else:
        try:
            args = json.loads(raw if raw is not None else '')
        except Exception as e:
            why = f'arguments not JSON: {e}'
    if args is not None and not isinstance(args, dict):
        why, args = 'arguments not an object', None
    if why is None:
        if name not in TOOL_SCHEMAS:
            why = f'unknown tool {name!r}'
        else:
            sch = TOOL_SCHEMAS[name]
            for k in sch['required']:
                if k not in args:
                    why = f'missing {k}'
                    break
            if why is None:
                for k, v in args.items():
                    ps = sch['properties'].get(k)
                    if ps is None:
                        why = f'unexpected argument {k}'
                        break
                    if ps.get('type') == 'string' and not isinstance(v, str):
                        why = f'{k} not a string'
                        break
                    if 'enum' in ps and v not in ps['enum']:
                        why = f'{k}={v!r} not in enum'
                        break
    return {'id': tc.get('id'), 'name': name, 'args': args, 'raw': raw if isinstance(raw, str) else json.dumps(raw),
            'valid': why is None, 'why': why}


def run_session(conn, sess, a):
    world = World(sess['state'])
    msgs = [{'role': 'system', 'content': SYSTEM}]
    steps_out, n_turns, calls_all = [], 0, []
    for si, st in enumerate(sess['steps']):
        msgs.append({'role': 'user', 'content': st['user']})
        calls, final = [], None
        for rnd_i in range(a.max_rounds):
            r = qlib.chat(conn, msgs, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
                          effort=a.effort, tools=TOOLS, tool_choice='auto')
            if r.error:
                return {'id': sess['id'], 'family': sess['family'], 'ok': None, 'error': r.error,
                        'failed_at': f'step {si} round {rnd_i}', 'steps': steps_out, 'turns': n_turns}
            n_turns += 1
            m = r.message
            tcs = m.get('tool_calls') or []
            am = {'role': 'assistant', 'content': m.get('content') or ''}
            if tcs:
                am['tool_calls'] = [{'id': tc.get('id') or f'call_{si}_{rnd_i}_{j}', 'type': 'function',
                                     'function': {'name': (tc.get('function') or {}).get('name'),
                                                  'arguments': (tc.get('function') or {}).get('arguments')
                                                  if isinstance((tc.get('function') or {}).get('arguments'), str)
                                                  else json.dumps((tc.get('function') or {}).get('arguments'))}}
                                    for j, tc in enumerate(tcs)]
            if m.get('reasoning_content') and not a.strip_reasoning:
                am['reasoning_content'] = m['reasoning_content']
            msgs.append(am)
            if not tcs:
                final = {'content': m.get('content') or '', 'finish_reason': r.finish_reason}
                break
            for tc in am['tool_calls']:
                pc = parse_call(tc)
                result = world.call(pc['name'], pc['args']) if pc['args'] is not None else \
                    json.dumps({'error': pc['why']})
                pc['result'] = result[:600]
                calls.append(pc)
                msgs.append({'role': 'tool', 'tool_call_id': tc['id'], 'name': pc['name'] or '', 'content': result})
        calls_all += calls
        tools_ok = all(any(call_matches(c, e) for c in calls) for e in st['expect'] if e.get('required'))
        if final is None:
            ans_ok, line = False, None
        else:
            line = answer_text(final['content'])
            ans_ok = check_answer(st['check'], line if line is not None else final['content'], world,
                                  whole=line is None)
            if not ans_ok and line is not None and st['check'].get('whole_ok'):
                # a check no listing can fool (every title of the day) may read the whole reply
                ans_ok = check_answer(st['check'], final['content'], world)
        steps_out.append({'step': si, 'tools_ok': tools_ok, 'answer_ok': bool(ans_ok), 'answer_line': line is not None,
                          'final': (final or {}).get('content', '')[-400:], 'gave_up': final is None,
                          'calls': [{k: c[k] for k in ('name', 'raw', 'valid', 'why', 'result')} for c in calls]})
    all_valid = all(c['valid'] for c in calls_all)
    success = all_valid and all(s['tools_ok'] and s['answer_ok'] for s in steps_out)
    return {'id': sess['id'], 'family': sess['family'], 'ok': success, 'turns': n_turns,
            'metrics': {'session_success': success, 'all_calls_valid': all_valid,
                        'all_steps_tools': all(s['tools_ok'] for s in steps_out),
                        'all_steps_answer': all(s['answer_ok'] for s in steps_out),
                        'turns_4_to_8': 4 <= n_turns <= 8},
            'n_calls': len(calls_all), 'n_valid_calls': sum(1 for c in calls_all if c['valid']), 'steps': steps_out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--max-rounds', type=int, default=6, help='model calls allowed per user step (default 6)')
    ap.add_argument('--temperature', type=float, default=qlib.PROD_TEMPERATURE)
    ap.add_argument('--top-p', type=float, default=qlib.PROD_TOP_P)
    ap.add_argument('--strip-reasoning', action='store_true', help='do not send reasoning_content back')
    a = ap.parse_args()
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    cell = 'prod' if (a.temperature, a.top_p) == (qlib.PROD_TEMPERATURE, qlib.PROD_TOP_P) else \
        ('t0' if a.temperature == 0 else f'T{a.temperature:g}-p{a.top_p:g}')
    sessions = make_sessions(a.n, a.seed)
    # ~6 sequential turns of ~350 tokens each per session, sessions in parallel
    est = a.n * 6 * (350 / qlib.EST_STREAM_TPS + 0.3) / max(1, a.concurrency)
    recs, wall = qlib.run_items(sessions, lambda s: run_session(conn, s, a), concurrency=a.concurrency,
                                out_jsonl=os.path.join(out, f'agent-{cell}.jsonl'), label=f'agent-{cell}', est_s=est)
    ans = [r for r in recs if r.get('ok') is not None]
    steps = [s for r in ans for s in r['steps']]
    nc = sum(r['n_calls'] for r in ans)
    nv = sum(r['n_valid_calls'] for r in ans)
    metrics = {'session_success': qlib.metric_from_records(recs, 'session_success'),
               'call_valid': qlib.prop(nv, nc),
               'step_tools': qlib.prop(sum(1 for s in steps if s['tools_ok']), len(steps)),
               'step_answer': qlib.prop(sum(1 for s in steps if s['answer_ok']), len(steps)),
               'answer_line': qlib.prop(sum(1 for s in steps if s['answer_line']), len(steps)),
               'all_calls_valid': qlib.metric_from_records(recs, 'all_calls_valid'),
               'turns_4_to_8': qlib.metric_from_records(recs, 'turns_4_to_8')}
    for fam, _ in FAMILIES:
        metrics[f'session_success@{fam}'] = qlib.metric_from_records(recs, 'session_success',
                                                                     where=lambda r, f=fam: r['family'] == f)
    means = {'turns': qlib.mean_ci([r['turns'] for r in ans]), 'calls': qlib.mean_ci([r['n_calls'] for r in ans]),
             'step_answer_per_session': qlib.mean_ci([sum(s['answer_ok'] for s in r['steps']) / len(r['steps'])
                                                      for r in ans if r['steps']])}
    cond = {'temperature': a.temperature, 'top_p': a.top_p, 'reasoning_effort': a.effort, 'max_tokens': a.max_tokens,
            'max_rounds': a.max_rounds, 'n': a.n, 'seed': a.seed, 'strip_reasoning': a.strip_reasoning,
            'concurrency': a.concurrency, 'tools': len(TOOLS)}
    qlib.write_summary(out, 'agent', cell, recs, primary='session_success', metrics=metrics, means=means,
                       condition=cond, wall_s=wall, tag=a.tag, conn=conn)


if __name__ == '__main__':
    main()
