#!/usr/bin/env python3
"""test_quality.py -- offline tests for the quality suite (stdlib unittest; never touches a real engine).

    python3 test_quality.py            # unit tests + end-to-end runs against fake_server.py on a free port
    QUALITY_E2E=0 python3 test_quality.py   # unit tests only (no fake server, no dataset download)

The end-to-end half starts fake_server.py (an oracle) on 127.0.0.1, runs every suite with small counts into a
temp dir and checks the summaries: the oracle scores 100%, --bad-rate makes each scorer count failures,
--no-json / --no-images make the suites skip cleanly, under_load holds its background streams, and compare.py
reads the result. gsm8k/MGSM data are downloaded once into $QUALITY_DATA_DIR (default a temp dir).
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('QUALITY_DATA_DIR', os.path.join(tempfile.gettempdir(), 'quality-test-data'))
import qlib  # noqa: E402
import needles  # noqa: E402
import json_format  # noqa: E402
import agent_sessions  # noqa: E402
import images  # noqa: E402
import compare  # noqa: E402


class Stats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = qlib.wilson(81, 263)
        self.assertAlmostEqual(lo, 0.2553, places=3)
        self.assertAlmostEqual(hi, 0.3662, places=3)
        self.assertEqual(qlib.wilson(0, 10)[0], 0.0)
        self.assertAlmostEqual(qlib.wilson(10, 10)[1], 1.0)

    def test_newcombe(self):
        # Newcombe (1998) Table II example (a): 56/70 vs 48/80, method 10 -> 0.0524 to 0.3339 (for p1 - p2)
        d, lo, hi = qlib.newcombe_diff(48, 80, 56, 70)   # our order: p_B - p_A with B = 56/70
        self.assertAlmostEqual(d, 0.2, places=4)
        self.assertAlmostEqual(lo, 0.0524, places=3)
        self.assertAlmostEqual(hi, 0.3339, places=3)

    def test_fisher(self):
        self.assertAlmostEqual(qlib.fisher_exact(3, 1, 1, 3), 0.4857, places=4)   # the lady tasting tea
        self.assertAlmostEqual(qlib.fisher_exact(1, 9, 11, 3), 0.002759, places=5)
        self.assertAlmostEqual(qlib.fisher_exact(5, 5, 5, 5), 1.0, places=6)

    def test_z_and_mcnemar(self):
        z, p = qlib.two_prop_z(48, 80, 56, 70)
        self.assertAlmostEqual(z, 2.6502, places=3)
        self.assertAlmostEqual(p, 0.008045, places=5)
        self.assertAlmostEqual(qlib.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(qlib.mcnemar_exact(1, 9), 0.021484375, places=8)

    def test_bootstrap(self):
        m, lo, hi = qlib.bootstrap_mean([1, 2, 3, 4, 5] * 20)
        self.assertAlmostEqual(m, 3.0)
        self.assertTrue(lo < 3.0 < hi and hi - lo < 0.7)

    def test_compare_verdicts(self):
        r = compare.compare_counts(950, 1000, 900, 1000, 0.03)
        self.assertEqual(r['class'], 'worse')
        r = compare.compare_counts(950, 1000, 951, 1000, 0.03)
        self.assertEqual(r['class'], 'noninferior')
        r = compare.compare_counts(19, 20, 18, 20, 0.03)
        self.assertEqual(r['class'], 'inconclusive')
        r = compare.compare_counts(10, 470, 30, 470, 0.0, lower_better=True)
        self.assertEqual(r['class'], 'worse')
        r = compare.compare_counts(19, 20, 18, 20, 0.0)
        self.assertEqual(r['class'], 'nodiff')


class Answers(unittest.TestCase):
    def test_parse_number(self):
        for s, v in [('1,234', 1234), ('1.234', 1234), ('1 234', 1234), ('12.5', 12.5), ('-3', -3), ('3,5', 3.5),
                     ('18.', 18), ('70000', 70000), ('1.234.567', 1234567), ('0.25', 0.25)]:
            self.assertEqual(qlib.parse_number(s), v, s)

    def test_extract(self):
        self.assertEqual(qlib.extract_answer('so 5 + 3 = 8\n#### 8'), (8.0, 8.0))
        self.assertEqual(qlib.extract_answer('the answer is 42.'), (42.0, None))
        self.assertEqual(qlib.extract_answer('#### $1,250'), (1250.0, 1250.0))
        self.assertEqual(qlib.extract_answer('答案\n#### １８'), (18.0, 18.0))
        self.assertEqual(qlib.extract_answer('#### 1.234'), (1234.0, 1234.0))
        self.assertEqual(qlib.extract_answer('no numbers'), (None, None))


class Needles(unittest.TestCase):
    def test_score_variants(self):
        ask = [('3F9A0C', '4829173'), ('00AB12', '1000000'), ('FFFFFF', '9999999')]
        txt = 'K-3F9A0C: 4829173\n**K-00AB12** is 1000000\n- `K-FFFFFF` = 9999999'
        per, _ = needles.score_content(txt, ask)
        self.assertEqual(per, [True, True, True])
        per, _ = needles.score_content('K-3F9A0C: 4829174\nK-00AB12: 1000000', ask)
        self.assertEqual(per, [False, True, False])
        per, _ = needles.score_content('The codes are 4829173, 1000000 and 9999999.', ask)
        self.assertEqual(per, [True, True, True])

    def test_build(self):
        sz = needles.Sizer(None)
        for L in (3000, 20000):
            it = needles.make_item(L, 0.5, 0, 1)
            text, est = needles.build_prompt(it, sz, 'abc')
            self.assertTrue(all(f'K-{k} is {v}.' in text for k, v in it['ask']))
            self.assertEqual(sum(text.count(f'K-{k} is {v}.') for k, v in it['records']), needles.N_RECORDS)
            self.assertLess(abs(est - L) / L, 0.05)
            # the asked records sit near the item's depth
            pos = [text.index(f'K-{k} is') / len(text) for k, _ in it['ask']]
            self.assertTrue(all(abs(p - 0.5) < 0.05 for p in pos), pos)

    def test_build_900k_speed(self):
        t0 = time.time()
        it = needles.make_item(900000, 0.95, 0, 1)
        text, est = needles.build_prompt(it, needles.Sizer(None), None)
        self.assertLess(time.time() - t0, 30)
        self.assertGreater(len(text), 3_000_000)


class Json(unittest.TestCase):
    def test_validator(self):
        it = json_format.make_items(10, 'schema')
        for x in it:
            self.assertEqual(json_format.validate(json_format_answer(x), x['schema']), [], x['family'])
        s = it[0]['schema']
        self.assertTrue(json_format.validate({'name': 'a'}, s))
        bad = dict(json_format_answer(it[0]))
        bad['age'] = '30'
        self.assertTrue(json_format.validate(bad, s))
        bad = dict(json_format_answer(it[0]))
        bad['extra'] = 1
        self.assertTrue(json_format.validate(bad, s))
        ev = next(x for x in it if x['family'] == 'event')
        b2 = dict(json_format_answer(ev))
        b2['date'] = 'March 3'
        self.assertTrue(json_format.validate(b2, ev['schema']))
        self.assertEqual(json_format.validate(True, {'type': 'integer'}), ['$: expected integer, got bool'])

    def test_strict_only_inside_strict_subset(self):
        # deepseek-recipe 0.2.0 refuses strict: true outside DeepSeek's strict subset with a 400
        # ("Invalid json schema: invalid type: map, expected a boolean"); those schemas go strict: false
        fams = {}
        for x in json_format.make_items(10, 'schema'):
            fams[x['family']] = json_format.response_format('schema', x['schema'])['json_schema']['strict']
        self.assertEqual(fams, {'person': True, 'invoice': True, 'event': True, 'sentiment': True, 'config': False})
        # strict false is still compiled by xgrammar, which 400s a schema with a keyword it would ignore
        self.assertFalse(any('uniqueItems' in json.dumps(x['schema']) for x in json_format.make_items(10, 'schema')))
        self.assertFalse(json_format.strict_fits({'type': 'object', 'properties': {'a': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'m': {'type': 'object', 'additionalProperties': {'type': 'string'}}}}}}}))
        def obj(props, **kw):
            return {'type': 'object', 'additionalProperties': False, 'required': list(props), 'properties': props,
                    **kw}
        fits = json_format.strict_fits
        ok = {'type': 'boolean'}
        self.assertTrue(fits(obj({'ok': ok})))
        self.assertFalse(fits({'type': 'object', 'additionalProperties': False, 'properties': {}, 'required': []}))
        # required unset: recipe would fill it in, but strict mode as specified (DeepSeek, OpenAI) needs it listed
        self.assertFalse(fits({'type': 'object', 'additionalProperties': False, 'properties': {'ok': ok}}))
        self.assertFalse(fits({'type': 'object', 'required': ['ok'], 'properties': {'ok': ok}}))  # not closed
        self.assertFalse(fits(obj({'ok': ok, 'n': {'type': 'integer'}}, required=['ok'])))     # one optional
        self.assertFalse(fits({'type': 'array', 'items': obj({'ok': ok})}))                    # top level not an object
        # definition maps: every spelling is checked, a $ref into them is fine
        bad = {'type': 'object', 'additionalProperties': {'type': 'string'}}
        for k in ('$defs', 'definitions', '$def'):
            self.assertTrue(fits(obj({'r': {'$ref': f'#/{k}/D'}}, **{k: {'D': obj({'ok': ok})}})), k)
            self.assertFalse(fits(obj({'r': {'$ref': f'#/{k}/D'}}, **{k: {'D': bad}})), k)
        # combinators: anyOf branches are checked; oneOf/allOf/not are outside strict mode
        self.assertTrue(fits(obj({'u': {'anyOf': [obj({'ok': ok}), {'type': 'null'}]}})))
        self.assertFalse(fits(obj({'u': {'anyOf': [bad, {'type': 'null'}]}})))
        for k in ('oneOf', 'allOf'):
            self.assertFalse(fits(obj({'u': {k: [obj({'ok': ok}), {'type': 'null'}]}})), k)
        self.assertFalse(fits(obj({'u': {'type': 'string', 'not': {'enum': ['x']}}})))
        # type lists: simple names only; a nullable object is spelled anyOf
        self.assertTrue(fits(obj({'s': {'type': ['string', 'null']}})))
        self.assertFalse(fits(obj({'o': {'type': ['object', 'null'], 'properties': {'ok': ok}, 'required': ['ok'],
                                         'additionalProperties': False}})))
        self.assertFalse(fits(obj({'o': {'type': ['object', 'null'], 'additionalProperties': {'type': 'string'}}})))
        # nodes recipe cannot type, and keyword values it cannot deserialize
        self.assertFalse(fits(obj({'x': {}})))
        self.assertFalse(fits(obj({'x': {'enum': ['a', 'b']}})))
        self.assertFalse(fits(obj({'d': {'type': 'string', 'format': 'date'}})))
        self.assertTrue(fits(obj({'d': {'type': 'string', 'format': 'uuid'}})))
        self.assertFalse(fits(obj({'n': {'type': 'integer', 'minimum': 0.5}})))
        self.assertFalse(fits(obj({'a': {'type': 'array'}})))

    def test_fields(self):
        for x in json_format.make_items(10, 'object'):
            self.assertTrue(json_format.check_fields(x['family'], json_format_answer(x), x['gold']), x['family'])


def json_format_answer(x):
    import fake_server
    return fake_server.json_answer(x)


class Agents(unittest.TestCase):
    def test_oracle_sessions_pass(self):
        for s in agent_sessions.make_sessions(25):
            w = agent_sessions.World(s['state'])
            for st in s['steps']:
                calls = []
                for c in st['oracle']['calls']:
                    pc = agent_sessions.parse_call({'id': 'x', 'function': {'name': c['name'],
                                                                            'arguments': json.dumps(c['arguments'])}})
                    self.assertTrue(pc['valid'], pc)
                    res = json.loads(w.call(pc['name'], pc['args']))
                    self.assertNotIn('error', res, (s['id'], c, res))
                    calls.append(pc)
                for e in st['expect']:
                    if e.get('required'):
                        self.assertTrue(any(agent_sessions.call_matches(c, e) for c in calls), (s['id'], e))
                line = agent_sessions.answer_text(st['oracle']['answer'])
                self.assertTrue(agent_sessions.check_answer(st['check'], line, w), (s['id'], st['check'], line))

    def test_invalid_calls(self):
        pc = agent_sessions.parse_call({'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"'}})
        self.assertFalse(pc['valid'])
        pc = agent_sessions.parse_call({'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo", "unit": "kelvin"}'}})
        self.assertFalse(pc['valid'])
        pc = agent_sessions.parse_call({'function': {'name': 'teleport', 'arguments': '{}'}})
        self.assertFalse(pc['valid'])
        pc = agent_sessions.parse_call({'function': {'name': 'read_file', 'arguments': {'path': '/proj/a'}}})
        self.assertTrue(pc['valid'])

    def test_wrong_answers_fail(self):
        s = agent_sessions.make_sessions(4)[3]           # weather
        w = agent_sessions.World(s['state'])
        st = s['steps'][0]
        others = [c for c in s['state']['weather'] if c.lower() not in st['oracle']['answer'].lower()]
        self.assertFalse(agent_sessions.check_answer(st['check'], others[0], w))
        self.assertEqual(agent_sessions.safe_eval('2*(3+4)'), 14)
        with self.assertRaises(ValueError):
            agent_sessions.safe_eval('__import__("os")')


class Images(unittest.TestCase):
    def test_png_and_grade(self):
        its = images.make_items(3, 4)
        self.assertTrue(its[0]['png'].startswith(b'\x89PNG'))
        self.assertTrue(images.grade('count', '3', 'There are **3** red squares.'))
        self.assertFalse(images.grade('count', '3', 'I count 4.'))
        self.assertTrue(images.grade('count', '2', 'two'))
        self.assertTrue(images.grade('position', 'top-left', 'It is in the upper left quadrant.'))
        self.assertTrue(images.grade('shape', 'triangle', 'The third shape is a triangle.'))
        self.assertTrue(images.grade('color', 'green', 'Green.'))

    def test_png_decodes(self):
        try:
            from PIL import Image   # optional cross-check
        except Exception:
            self.skipTest('PIL not installed')
        import io
        im = Image.open(io.BytesIO(images.make_items(1, 0)[2]['png']))
        self.assertEqual(im.size, (256, 256))


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipIf(os.environ.get('QUALITY_E2E') == '0', 'QUALITY_E2E=0')
class EndToEnd(unittest.TestCase):
    procs = []

    @classmethod
    def server(cls, *flags):
        port = free_port()
        p = subprocess.Popen([sys.executable, os.path.join(HERE, 'fake_server.py'), '--port', str(port),
                              '--key', 'test-key', *flags], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=dict(os.environ))
        cls.procs.append(p)
        for _ in range(200):
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/models',
                                             headers={'Authorization': 'Bearer test-key'})
                urllib.request.urlopen(req, timeout=1).read()
                return f'http://127.0.0.1:{port}'
            except Exception:
                time.sleep(0.1)
        raise RuntimeError('fake server did not start: ' + p.stdout.read().decode())

    @classmethod
    def setUpClass(cls):
        # The data download happens here, once (gsm8k test.jsonl + MGSM TSVs from GitHub).
        import gsm8k_full
        import mgsm
        gsm8k_full.load_items(5)
        mgsm.load_items(['zh', 'es', 'fr', 'de', 'ja', 'ru'], 2)
        cls.good = cls.server()
        cls.bad = cls.server('--bad-rate', '0.3', '--error-rate', '0.05')
        cls.none = cls.server('--no-json', '--no-images', '--ctx', '300000')
        cls.tmp = tempfile.mkdtemp(prefix='quality-e2e-')

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
            p.wait(10)

    def run_suite(self, script, base, out, *args, timeout=600):
        env = dict(os.environ, VLLM_API_KEY='test-key')
        r = subprocess.run([sys.executable, os.path.join(HERE, script), '--base-url', base, '--out', out, *args],
                           capture_output=True, text=True, timeout=timeout, env=env)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return r.stdout

    def summ(self, out, name):
        with open(os.path.join(out, name + '.summary.json')) as f:
            return json.load(f)

    def test_all_suites(self):
        good, bad = os.path.join(self.tmp, 'good'), os.path.join(self.tmp, 'bad')
        for out, base in ((good, self.good), (bad, self.bad)):
            self.run_suite('gsm8k_full.py', base, out, '--n', '60', '-c', '8')
            self.run_suite('needles.py', base, out, '--lengths', '4000,12000', '--depths', '0.1,0.5,0.9',
                           '--reps', '2', '--budget-min', '5')
            self.run_suite('json_format.py', base, out, '--n', '20')
            self.run_suite('agent_sessions.py', base, out, '--n', '20')
            self.run_suite('mgsm.py', base, out, '--per-lang', '5', '--cells', 't0,prod')
            self.run_suite('images.py', base, out, '--per-family', '5', '--controls', '4')
        for name, metric in (('gsm8k-t0', 'exact'), ('gsm8k-prod', 'exact'), ('needles-t0', 'all3'),
                             ('json-schema', 'valid'), ('json-object', 'valid'), ('agent-prod', 'session_success'),
                             ('mgsm-t0', 'exact'), ('images-t0', 'exact')):
            g = self.summ(good, name)
            self.assertEqual(g['errors'], 0, name)
            self.assertEqual(g['metrics'][metric]['p'], 1.0, (name, g['metrics'][metric], g.get('first_error')))
            b = self.summ(bad, name)
            self.assertLess(b['metrics'][metric]['p'], 1.0, (name, b['metrics'][metric]))
        g = self.summ(good, 'agent-prod')
        self.assertEqual(g['metrics']['call_valid']['p'], 1.0)
        self.assertEqual(g['metrics']['turns_4_to_8']['p'], 1.0)
        self.assertLess(self.summ(bad, 'agent-prod')['metrics']['call_valid']['p'], 1.0)
        self.assertGreater(self.summ(bad, 'gsm8k-t0')['errors'], 0)       # the 500s are errors, not wrong answers
        self.assertEqual(self.summ(good, 'json-schema')['metrics']['fields']['p'], 1.0)
        js = self.summ(good, 'json-schema')['metrics']
        self.assertEqual((js['valid@strict']['n'], js['valid@nonstrict']['n']), (16, 4))   # config is every 5th item
        self.assertEqual(self.summ(good, 'images-t0')['metrics']['control']['n'], 4)
        # compare: good vs bad shows B worse on gsm8k (paired), and runs over every common file
        r = subprocess.run([sys.executable, os.path.join(HERE, 'compare.py'), 'dirs', good, bad, '--labels',
                            'final,mk', '--json', os.path.join(self.tmp, 'cmp.json')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)   # B is degraded on purpose: a WORSE verdict exits 2
        cmp_ = json.load(open(os.path.join(self.tmp, 'cmp.json')))
        self.assertEqual(len(cmp_['results']), 9)
        g8 = next(x for x in cmp_['results'] if x['suite'] == 'gsm8k' and x['cell_a'] == 't0')
        self.assertIn('paired', g8)
        self.assertGreater(g8['paired']['a_only'], 0)
        self.assertIn(g8['class'], ('worse', 'inconclusive'))

    def test_unsupported(self):
        out = os.path.join(self.tmp, 'none')
        self.run_suite('json_format.py', self.none, out, '--n', '5')
        self.assertFalse(self.summ(out, 'json-schema')['supported'])
        self.run_suite('images.py', self.none, out)
        self.assertFalse(self.summ(out, 'images-t0')['served'])
        o = self.run_suite('needles.py', self.none, out, '--lengths', '4000,250000,500000', '--depths', '0.5',
                           '--reps', '1')
        s = self.summ(out, 'needles-t0')
        self.assertEqual(s['condition']['skipped_lengths'], [500000])
        self.assertEqual(s['metrics']['all3']['n'], 2)
        self.assertIn('skipped over context', o)

    def test_under_load(self):
        out = os.path.join(self.tmp, 'load')
        base = self.server('--latency', '0.2')
        self.run_suite('gsm8k_full.py', base, out, '--n', '40', '--cells', 't0')
        o = self.run_suite('under_load.py', base, out, '--n', '40', '--cells', 't0', '--warm-s', '2', '-c', '4')
        s = self.summ(out, 'gsm8k-t0-load16')
        self.assertEqual(s['metrics']['exact']['p'], 1.0)
        bg = s['background']
        self.assertTrue(bg['load_held'], bg)
        self.assertGreater(bg['requests_completed'], 16)
        self.assertGreaterEqual(bg['num_running_reqs']['max'], 16)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'compare.py'), 'files',
                            os.path.join(out, 'gsm8k-t0.summary.json'), os.path.join(out, 'gsm8k-t0-load16.summary.json'),
                            '--labels', 'idle,load'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('paired on 40 shared items', r.stdout)
        self.assertIn('load_held', open(os.path.join(out, 'load-bg.json')).read())
        _ = o


if __name__ == '__main__':
    unittest.main(verbosity=2)
