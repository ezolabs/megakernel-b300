#!/usr/bin/env python3
"""json_format.py -- structured output: `response_format` json_schema and json_object, validated by schema.

200 extraction asks by default (--n 100 per cell), five deterministic families (a person record, an invoice
with line items and a total, an event with a date pattern and an enum, a review's sentiment, a service config
with an open `env` map), each a short synthetic text plus the JSON Schema the answer must satisfy. Two cells:

  schema  response_format {"type": "json_schema", "json_schema": {"name", "schema", "strict"}}: strict true
          only for a schema that fits strict mode (`strict_fits`: closed objects, every property required);
          the config family's open `env` map does not, so it goes strict false (still grammar-enforced:
          no family schema may use a keyword xgrammar cannot enforce, e.g. uniqueItems, or the engine 400s).
          Each record carries `strict`; `valid@strict` / `valid@nonstrict` split the primary metric
  object  response_format {"type": "json_object"}

The schema is in the prompt in both cells, so `schema` measures the engine's grammar enforcement on top of
the same ask. Metrics per item: `parse` (the whole content is one JSON value), `valid` (parse AND it
satisfies the schema; primary), `fields` (valid AND the deterministic facts are right: names, numbers, the
total, the date, the enum), and `parse_lenient` (a JSON object can be dug out of fences/prose; diagnostic
only). Before each cell one probe request checks the server accepts that response_format: a 4xx writes the
cell's summary with "supported": false and skips it. Sampling defaults to prod (T 1.0 / top_p 1.0), the
condition grammar enforcement matters most under; reasoning_effort high.

    python3 json_format.py --out OUT [--cells schema,object] [--n 100] [-c 16]

Writes OUT/json-<cell>.jsonl and OUT/json-<cell>.summary.json. Expected wall on the final: ~2-4 min.
The validator is this file's own (a JSON Schema subset: type, properties, required, additionalProperties,
enum, const, minimum/maximum/exclusive*, minLength/maxLength, pattern, items, minItems/maxItems,
uniqueItems), so it runs on any stdlib Python.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

MEAN_TOKENS_EST = 700

# ----------------------------------------------------------------------------------------------- validator
_TYPES = {
    'string': lambda v: isinstance(v, str),
    'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
    'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    'boolean': lambda v: isinstance(v, bool),
    'array': lambda v: isinstance(v, list),
    'object': lambda v: isinstance(v, dict),
    'null': lambda v: v is None,
}


def validate(v, s, path='$'):
    """Errors (a list of strings, empty when valid) of value v against schema s."""
    errs = []
    if s is True or s == {}:
        return errs
    if s is False:
        return [f'{path}: no value allowed']
    t = s.get('type')
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](v) for x in ts):
            return [f'{path}: expected {t}, got {type(v).__name__}']
    if 'const' in s and v != s['const']:
        errs.append(f'{path}: != const')
    if 'enum' in s and v not in s['enum']:
        errs.append(f'{path}: {v!r} not in enum')
    if isinstance(v, str):
        if 'minLength' in s and len(v) < s['minLength']:
            errs.append(f'{path}: shorter than {s["minLength"]}')
        if 'maxLength' in s and len(v) > s['maxLength']:
            errs.append(f'{path}: longer than {s["maxLength"]}')
        if 'pattern' in s and not re.search(s['pattern'], v):
            errs.append(f'{path}: does not match {s["pattern"]}')
    if _TYPES['number'](v):
        if 'minimum' in s and v < s['minimum']:
            errs.append(f'{path}: < {s["minimum"]}')
        if 'maximum' in s and v > s['maximum']:
            errs.append(f'{path}: > {s["maximum"]}')
        if 'exclusiveMinimum' in s and v <= s['exclusiveMinimum']:
            errs.append(f'{path}: <= {s["exclusiveMinimum"]}')
        if 'exclusiveMaximum' in s and v >= s['exclusiveMaximum']:
            errs.append(f'{path}: >= {s["exclusiveMaximum"]}')
    if isinstance(v, list):
        if 'minItems' in s and len(v) < s['minItems']:
            errs.append(f'{path}: fewer than {s["minItems"]} items')
        if 'maxItems' in s and len(v) > s['maxItems']:
            errs.append(f'{path}: more than {s["maxItems"]} items')
        if s.get('uniqueItems'):
            seen = [json.dumps(x, sort_keys=True) for x in v]
            if len(set(seen)) != len(seen):
                errs.append(f'{path}: items not unique')
        if isinstance(s.get('items'), dict):
            for i, x in enumerate(v):
                errs += validate(x, s['items'], f'{path}[{i}]')
    if isinstance(v, dict):
        props = s.get('properties') or {}
        for k in s.get('required') or []:
            if k not in v:
                errs.append(f'{path}: missing {k}')
        ap = s.get('additionalProperties', True)
        for k, x in v.items():
            if k in props:
                errs += validate(x, props[k], f'{path}.{k}')
            elif ap is False:
                errs.append(f'{path}: unexpected key {k}')
            elif isinstance(ap, dict):
                errs += validate(x, ap, f'{path}.{k}')
    return errs


# ----------------------------------------------------------------------------------------------- items
FIRST = ['Amara', 'Bjorn', 'Chiara', 'Dmitri', 'Esme', 'Farid', 'Greta', 'Hiro', 'Ines', 'Jonas', 'Keiko', 'Luca',
         'Maren', 'Nikhil', 'Odile', 'Pablo', 'Quinn', 'Rania', 'Soren', 'Tamsin']
LAST = ['Okafor', 'Lindqvist', 'Moretti', 'Volkov', 'Hartley', 'Haddad', 'Brandt', 'Tanaka', 'Duarte', 'Weber',
        'Sato', 'Romano', 'Nilsen', 'Rao', 'Laurent', 'Ortega', 'Byrne', 'Nasser', 'Kjaer', 'Wells']
CITIES = ['Lisbon', 'Tromso', 'Valencia', 'Kraków', 'Nagoya', 'Montreal', 'Adelaide', 'Porto', 'Ghent', 'Leipzig',
          'Bergen', 'Bologna', 'Cusco', 'Tbilisi', 'Hobart']
JOBS = ['veterinarian', 'sound engineer', 'pastry chef', 'structural engineer', 'librarian', 'ferry captain',
        'translator', 'glaciologist', 'florist', 'tax auditor', 'nurse', 'carpenter']
LANGS = ['English', 'Portuguese', 'Norwegian', 'Spanish', 'Polish', 'Japanese', 'French', 'German', 'Italian',
         'Quechua', 'Georgian', 'Arabic']
PRODUCTS = ['ceramic mug', 'wool scarf', 'desk lamp', 'notebook', 'water bottle', 'phone case', 'tea sampler',
            'bike light', 'cutting board', 'umbrella', 'backpack', 'candle set']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
TITLES = ['Budget review', 'Quarterly planning', 'Design critique', 'Vendor call', 'Incident retro', 'Hiring sync',
          'Roadmap walkthrough', 'Security drill']
POS = ['the battery easily lasts two days', 'setup took five minutes', 'the screen is gorgeous', 'support was quick '
       'and kind', 'it feels solid and well made', 'I would happily buy it again']
NEG = ['it stopped charging after a week', 'the hinge cracked on day two', 'support never answered my emails',
       'the app crashes constantly', 'it arrived scratched and dented', 'I returned it and want my money back']
SERVICES = ['ledger-api', 'image-resizer', 'auth-gateway', 'search-indexer', 'billing-worker', 'notify-hub']
ENVK = ['LOG_LEVEL', 'REGION', 'CACHE_TTL', 'FEATURE_FLAGS', 'DB_POOL', 'TIMEOUT_MS']
ENVV = {'LOG_LEVEL': ['debug', 'info', 'warn'], 'REGION': ['eu-west-1', 'us-east-2', 'ap-south-1'],
        'CACHE_TTL': ['60', '300', '900'], 'FEATURE_FLAGS': ['beta,dark-mode', 'none', 'search-v2'],
        'DB_POOL': ['8', '16', '32'], 'TIMEOUT_MS': ['1500', '3000', '5000']}


def _str(**kw):
    d = {'type': 'string'}
    d.update(kw)
    return d


def fam_person(rnd):
    name = f'{rnd.choice(FIRST)} {rnd.choice(LAST)}'
    age = rnd.randint(19, 88)
    city, job = rnd.choice(CITIES), rnd.choice(JOBS)
    langs = rnd.sample(LANGS, rnd.randint(1, 3))
    text = (f'{name} is a {age}-year-old {job} who lives in {city}. Colleagues say {name.split()[0]} is meticulous. '
            f'{name.split()[0]} speaks {", ".join(langs[:-1]) + " and " + langs[-1] if len(langs) > 1 else langs[0]}.')
    schema = {'type': 'object', 'additionalProperties': False,
              'required': ['name', 'age', 'city', 'occupation', 'languages'],
              'properties': {'name': _str(minLength=1), 'age': {'type': 'integer', 'minimum': 0, 'maximum': 130},
                             'city': _str(), 'occupation': _str(),
                             'languages': {'type': 'array', 'minItems': 1, 'items': _str()}}}
    gold = {'name': name, 'age': age, 'city': city, 'occupation': job, 'languages': langs}
    return 'person', 'Extract the person record from the text.', text, schema, gold


def fam_invoice(rnd):
    cust = f'{rnd.choice(FIRST)} {rnd.choice(LAST)}'
    items = []
    for p in rnd.sample(PRODUCTS, rnd.randint(2, 4)):
        items.append({'name': p, 'quantity': rnd.randint(1, 5), 'unit_price': rnd.choice([4.5, 7.99, 12.0, 15.25,
                                                                                         19.99, 24.0, 39.5])})
    total = round(sum(i['quantity'] * i['unit_price'] for i in items), 2)
    lines = '; '.join(f'{i["quantity"]} x {i["name"]} at ${i["unit_price"]:.2f} each' for i in items)
    text = f'Order for {cust}: {lines}. No discounts, no tax, no shipping.'
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['customer', 'items', 'total'],
              'properties': {'customer': _str(minLength=1),
                             'items': {'type': 'array', 'minItems': 1, 'items': {
                                 'type': 'object', 'additionalProperties': False,
                                 'required': ['name', 'quantity', 'unit_price'],
                                 'properties': {'name': _str(), 'quantity': {'type': 'integer', 'minimum': 1},
                                                'unit_price': {'type': 'number', 'minimum': 0}}}},
                             'total': {'type': 'number', 'minimum': 0}}}
    gold = {'customer': cust, 'n_items': len(items), 'total': total}
    return 'invoice', 'Extract the invoice, computing the total as the sum of quantity x unit_price.', text, schema, gold


def fam_event(rnd):
    title = rnd.choice(TITLES)
    m, d = rnd.randint(1, 12), rnd.randint(1, 28)
    pr = rnd.choice(['low', 'medium', 'high'])
    att = rnd.sample(FIRST, rnd.randint(2, 4))
    text = (f'Reminder: "{title}" is set for {MONTHS[m - 1]} {d}, 2027. Treat it as {pr} priority. '
            f'Invited: {", ".join(att)}.')
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['title', 'date', 'priority', 'attendees'],
              'properties': {'title': _str(minLength=1), 'date': _str(pattern=r'^\d{4}-\d{2}-\d{2}$'),
                             'priority': {'type': 'string', 'enum': ['low', 'medium', 'high']},
                             'attendees': {'type': 'array', 'minItems': 1, 'items': _str()}}}
    gold = {'title': title, 'date': f'2027-{m:02d}-{d:02d}', 'priority': pr, 'attendees': att}
    return 'event', 'Extract the event; write the date as YYYY-MM-DD.', text, schema, gold


def fam_sentiment(rnd):
    pos = rnd.random() < 0.5
    parts = rnd.sample(POS if pos else NEG, 3)
    text = f'Customer review: "{parts[0].capitalize()}, {parts[1]}, and {parts[2]}."'
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['sentiment', 'confidence', 'keywords'],
              'properties': {'sentiment': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                             'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                             'keywords': {'type': 'array', 'minItems': 1, 'maxItems': 5, 'items': _str()}}}
    gold = {'sentiment': 'positive' if pos else 'negative'}
    return 'sentiment', 'Classify the review.', text, schema, gold


def fam_config(rnd):
    name = rnd.choice(SERVICES)
    port = rnd.choice([8080, 8443, 9000, 5432, 6379, 3000, 7070])
    rep = rnd.randint(1, 12)
    env = {k: rnd.choice(ENVV[k]) for k in rnd.sample(ENVK, rnd.randint(2, 3))}
    tags = rnd.sample(['prod', 'canary', 'gpu', 'batch', 'internal', 'edge'], 2)
    text = (f'Deploy {name} listening on port {port} with {rep} replicas. Environment: '
            + ', '.join(f'{k}={v}' for k, v in env.items()) + f'. Tag it {tags[0]} and {tags[1]}.')
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['service', 'env', 'tags'],
              'properties': {'service': {'type': 'object', 'additionalProperties': False,
                                         'required': ['name', 'port', 'replicas'],
                                         'properties': {'name': _str(minLength=1),
                                                        'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                                                        'replicas': {'type': 'integer', 'minimum': 1, 'maximum': 100}}},
                             'env': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                             # no uniqueItems: xgrammar refuses the whole grammar over it even with strict
                             # false (400 "keyword(s) uniqueItems would be ignored", raw/s2-final-ref)
                             'tags': {'type': 'array', 'items': _str()}}}
    gold = {'name': name, 'port': port, 'replicas': rep, 'env': env}
    return 'config', 'Turn the deploy note into a service config; env values are strings.', text, schema, gold


FAMILIES = [fam_person, fam_invoice, fam_event, fam_sentiment, fam_config]


def make_items(n, cell, seed=20261029):
    rnd = random.Random(f'{seed}-{cell}')
    items = []
    for i in range(n):
        fam, task, text, schema, gold = FAMILIES[i % len(FAMILIES)](rnd)
        prompt = (f'{task}\n\nText:\n{text}\n\nReply with only a JSON object that satisfies this JSON Schema '
                  f'(no prose, no code fences):\n{json.dumps(schema)}')
        items.append({'id': f'json-{cell}-{i:03d}', 'family': fam, 'prompt': prompt, 'schema': schema, 'gold': gold})
    return items


def _ci(a, b):
    return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()


def check_fields(fam, obj, gold):
    try:
        if fam == 'person':
            return (_ci(obj['name'], gold['name']) and obj['age'] == gold['age'] and _ci(obj['city'], gold['city'])
                    and _ci(obj['occupation'], gold['occupation'])
                    and {x.lower() for x in obj['languages']} == {x.lower() for x in gold['languages']})
        if fam == 'invoice':
            return (_ci(obj['customer'], gold['customer']) and len(obj['items']) == gold['n_items']
                    and abs(float(obj['total']) - gold['total']) <= 0.011)
        if fam == 'event':
            return (_ci(obj['title'], gold['title']) and obj['date'] == gold['date']
                    and obj['priority'] == gold['priority']
                    and {x.lower() for x in obj['attendees']} == {x.lower() for x in gold['attendees']})
        if fam == 'sentiment':
            return obj['sentiment'] == gold['sentiment']
        if fam == 'config':
            s = obj['service']
            return (_ci(s['name'], gold['name']) and s['port'] == gold['port'] and s['replicas'] == gold['replicas']
                    and obj['env'] == gold['env'])
    except Exception:
        return False
    return False


def lenient_parse(txt):
    t = txt.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', t, re.S)
    if m:
        t = m.group(1).strip()
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


# The strict subset, as deepseek-recipe 0.2.0 deserializes it (src/protocol/json_schema.rs): a node is a `type`
# (one name, or a list of the simple names only), an `anyOf` of single-`type` nodes, or a `$ref`; nothing else.
_STRICT_SIMPLE = {'string', 'number', 'integer', 'boolean', 'null'}
_STRICT_TYPES = _STRICT_SIMPLE | {'object', 'array'}
_STRICT_FORMATS = {'email', 'hostname', 'ipv4', 'ipv6', 'uuid'}
_STRICT_INT_KEYS = ('const', 'default', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf')
# Keywords strict mode has no place for: recipe refuses a node that has only these, and silently drops them
# beside a `type`, so the strict constraint would not be the schema as written.
_STRICT_NONE = ('oneOf', 'allOf', 'not', 'if', 'then', 'else', 'prefixItems', 'patternProperties')
# Definition maps: `$defs` (2019-09+), `definitions` (draft-07) and recipe's own `$def` (the only one it keeps).
_STRICT_DEFS = ('$defs', 'definitions', '$def')

def strict_fits(s):
    """Whether `s` fits strict mode, so `strict: true` is safe to send. The top level is an object, and every
    object is closed: non-empty `properties`, every property in `required` (once), `additionalProperties`
    false. Types are single names or lists of the simple ones (`["object", "null"]` is outside the subset);
    combinators are `anyOf` only; definition maps ($defs, definitions, $def) are checked like the rest.
    deepseek-recipe refuses any other schema with strict: true (400 "Invalid json schema: ..."), so such an
    item goes strict: false, where the engine still grammar-enforces the raw schema."""
    return isinstance(s, dict) and s.get('type') == 'object' and _strict_node(s)

def _strict_node(s):
    if not isinstance(s, dict) or any(k in s for k in _STRICT_NONE):
        return False
    t, subs = s.get('type'), []
    if isinstance(t, list):
        if not t or not all(isinstance(x, str) and x in _STRICT_SIMPLE for x in t):
            return False
    elif t is not None:
        if t not in _STRICT_TYPES:
            return False
        if t == 'object':
            props, req = s.get('properties'), s.get('required')
            if not isinstance(props, dict) or not props or s.get('additionalProperties') is not False:
                return False
            if not isinstance(req, list) or len(req) != len(set(req)) or set(req) != set(props):
                return False
            subs += props.values()
        elif t == 'array':
            if not isinstance(s.get('items'), dict):
                return False
            subs.append(s['items'])
        elif t == 'string':
            if 'enum' in s and not (isinstance(s['enum'], list) and all(isinstance(x, str) for x in s['enum'])):
                return False
            if 'format' in s and s['format'] not in _STRICT_FORMATS:
                return False
        elif t == 'integer':
            if any(k in s and (isinstance(s[k], bool) or not isinstance(s[k], int)) for k in _STRICT_INT_KEYS):
                return False
    elif 'anyOf' not in s and not isinstance(s.get('$ref'), str):
        return False
    if 'anyOf' in s:
        br = s['anyOf']
        if not isinstance(br, list) or not br or not all(isinstance(b, dict) and isinstance(b.get('type'), str)
                                                          for b in br):
            return False
        subs += br
    for k in _STRICT_DEFS:
        if k in s:
            if not isinstance(s[k], dict):
                return False
            subs += s[k].values()
    return all(_strict_node(x) for x in subs)


def response_format(cell, schema):
    if cell == 'schema':
        return {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema,
                                                       'strict': strict_fits(schema)}}
    return {'type': 'json_object'}


def make_scorer(conn, cell, a):
    def score(it):
        rf = response_format(cell, it['schema'])
        r = qlib.chat(conn, [{'role': 'user', 'content': it['prompt']}], max_tokens=a.max_tokens,
                      temperature=a.temperature, top_p=a.top_p, effort=a.effort, response_format=rf)
        rec = {'id': it['id'], 'family': it['family']}
        if cell == 'schema':
            rec['strict'] = rf['json_schema']['strict']
        if r.error:
            rec.update(ok=None, error=r.error, wall_s=round(r.wall, 3))
            return rec
        txt = r.content
        obj, parse_ok = None, False
        try:
            obj = json.loads(txt)
            parse_ok = True
        except Exception:
            pass
        errs = validate(obj, it['schema']) if parse_ok else ['not JSON']
        valid = parse_ok and not errs
        fields = valid and check_fields(it['family'], obj, it['gold'])
        len_obj = lenient_parse(txt) if not parse_ok else obj
        rec.update(ok=valid, metrics={'parse': parse_ok, 'valid': valid, 'fields': bool(fields),
                                      'parse_lenient': len_obj is not None},
                   schema_errors=errs[:5], content_head=txt[:400], **r.stats())
        return rec
    return score


def probe(conn, cell, effort):
    sch = {'type': 'object', 'additionalProperties': False, 'required': ['ok'], 'properties': {'ok': {'type': 'boolean'}}}
    for attempt in range(3):   # a 4xx is an answer; a transport error or 5xx is retried twice
        r = qlib.chat(conn, [{'role': 'user', 'content': 'Reply with the JSON object {"ok": true} and nothing else.'}],
                      max_tokens=2048, temperature=0.0, effort=effort, response_format=response_format(cell, sch),
                      timeout=300)
        if not r.error or r.error.startswith('status: HTTP 4'):
            break
    if r.error and r.error.startswith('status: HTTP 4'):
        return False, r.error
    return True, r.error


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap)
    ap.add_argument('--cells', default='schema,object')
    ap.add_argument('--n', type=int, default=100, help='items per cell (default 100)')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--temperature', type=float, default=qlib.PROD_TEMPERATURE)
    ap.add_argument('--top-p', type=float, default=qlib.PROD_TOP_P)
    a = ap.parse_args()
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    est = qlib.est_decode_s(a.n, MEAN_TOKENS_EST, a.concurrency)
    print(f'json: cells {cells}, {a.n} items each; expected wall ~{qlib.fmt_s(est * len(cells))} on the final -> {out}')
    for cell in cells:
        if cell not in ('schema', 'object'):
            ap.error(f'unknown cell {cell!r}')
        items = make_items(a.n, cell)
        cond = {'response_format': response_format(cell, {'...': '...'})['type'], 'temperature': a.temperature,
                'top_p': a.top_p, 'reasoning_effort': a.effort, 'max_tokens': a.max_tokens, 'n': a.n,
                'concurrency': a.concurrency}
        ok, why = probe(conn, cell, a.effort)
        if not ok:
            print(f'json-{cell}: response_format refused by the server ({why}); cell skipped')
            qlib.write_summary(out, 'json', cell, [], primary='valid', metrics={}, condition=cond, tag=a.tag,
                               conn=conn, extra={'supported': False, 'probe_error': why})
            continue
        recs, wall = qlib.run_items(items, make_scorer(conn, cell, a), concurrency=a.concurrency,
                                    out_jsonl=os.path.join(out, f'json-{cell}.jsonl'), label=f'json-{cell}', est_s=est)
        metrics = {k: qlib.metric_from_records(recs, k) for k in ('valid', 'parse', 'fields', 'parse_lenient')}
        for fam in sorted({it['family'] for it in items}):
            metrics[f'valid@{fam}'] = qlib.metric_from_records(recs, 'valid', where=lambda r, f=fam: r['family'] == f)
        if cell == 'schema':   # valid@strict is like for like with a pre-patch run, which answered only those
            for name, val in (('strict', True), ('nonstrict', False)):
                metrics[f'valid@{name}'] = qlib.metric_from_records(recs, 'valid',
                                                                    where=lambda r, v=val: r.get('strict') is v)
        means = {'completion_tokens': qlib.mean_ci([r.get('completion_tokens') for r in recs if r.get('ok') is not None])}
        qlib.write_summary(out, 'json', cell, recs, primary='valid', metrics=metrics, means=means, condition=cond,
                           wall_s=wall, tag=a.tag, conn=conn, extra={'supported': True, 'probe_error': why})


if __name__ == '__main__':
    main()
