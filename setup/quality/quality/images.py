#!/usr/bin/env python3
"""images.py -- image input, ONLY if the served model takes it; otherwise it writes 'not served' and stops.

Serving check first: GET /v1/models and /get_server_info are recorded, then ONE probe request sends a
256x256 PNG of a single red square and asks its colour. A 4xx (or a 200 that names no colour at all, i.e. the
image never reached the model) writes OUT/images-t0.summary.json with "served": false and the reason, and
exits 0 -- the suite is skipped, not failed.

When served: 4 synthetic families x 20 (--per-family), images drawn by this file (stdlib PNG encoder, no PIL),
every answer derivable only from the pixels (no text is rendered):
  count     N red squares (1..6) among blue circles -> how many red squares
  shape     three shapes in a row (square/circle/triangle) -> the kind of the third (rightmost)
  color     one large shape in red/green/blue/yellow/black -> its colour
  position  one red disc in a quadrant -> top-left/top-right/bottom-left/bottom-right
plus `control` rows (--controls, default 20): the same questions with NO image, whose accuracy should sit
near chance (count 1/6, shape 1/3, colour 1/5, position 1/4); a control well above chance means the text leaks
the answer. Metrics: `exact` (primary, image rows), `exact@<family>`, `control` (no-image rows). The image
goes as an OpenAI `image_url` content part with a base64 data URI. T 0, effort high, max_tokens 4096.

    python3 images.py --out OUT [--per-family 20] [--controls 20] [-c 16]

Writes OUT/images-t0.jsonl and OUT/images-t0.summary.json. Expected wall on the final: ~1-3 min when served.
"""
import argparse
import base64
import os
import random
import re
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

W = H = 256
COLORS = {'red': (220, 30, 30), 'green': (30, 160, 50), 'blue': (30, 70, 220), 'yellow': (240, 210, 20),
          'black': (10, 10, 10)}
WHITE = (255, 255, 255)
NUMWORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
            'nine': 9, 'ten': 10}


class Canvas:
    def __init__(self, w=W, h=H, bg=WHITE):
        self.w, self.h = w, h
        self.px = bytearray(bytes(bg) * (w * h))

    def put(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i:i + 3] = bytes(c)

    def rect(self, x0, y0, x1, y1, c):
        for y in range(max(0, y0), min(self.h, y1)):
            i0 = (y * self.w + max(0, x0)) * 3
            n = max(0, min(self.w, x1) - max(0, x0))
            self.px[i0:i0 + 3 * n] = bytes(c) * n

    def disc(self, cx, cy, r, c):
        for y in range(cy - r, cy + r + 1):
            dx = int((r * r - (y - cy) ** 2) ** 0.5) if r * r >= (y - cy) ** 2 else -1
            if dx >= 0:
                self.rect(cx - dx, y, cx + dx + 1, y + 1, c)

    def triangle(self, cx, cy, r, c):   # upward isoceles in the box [cx-r, cx+r] x [cy-r, cy+r]
        for y in range(cy - r, cy + r + 1):
            half = int((y - (cy - r)) / 2)
            self.rect(cx - half, y, cx + half + 1, y + 1, c)

    def png(self):
        raw = b''.join(b'\x00' + bytes(self.px[y * self.w * 3:(y + 1) * self.w * 3]) for y in range(self.h))

        def chunk(t, d):
            return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xFFFFFFFF)
        return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', self.w, self.h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def data_uri(png):
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def _free_spot(rnd, placed, r):
    for _ in range(500):
        x, y = rnd.randint(r + 4, W - r - 5), rnd.randint(r + 4, H - r - 5)
        if all((x - a) ** 2 + (y - b) ** 2 >= (r + rb + 6) ** 2 for a, b, rb in placed):
            placed.append((x, y, r))
            return x, y
    raise RuntimeError('no free spot')


def draw(kind, x, y, r, c, cv):
    if kind == 'square':
        cv.rect(x - r, y - r, x + r, y + r, c)
    elif kind == 'circle':
        cv.disc(x, y, r, c)
    else:
        cv.triangle(x, y, r, c)


def item_count(rnd):
    n = rnd.randint(1, 6)
    cv, placed = Canvas(), []
    for _ in range(n):
        x, y = _free_spot(rnd, placed, 14)
        cv.rect(x - 14, y - 14, x + 14, y + 14, COLORS['red'])
    for _ in range(rnd.randint(2, 5)):
        x, y = _free_spot(rnd, placed, 13)
        cv.disc(x, y, 13, COLORS['blue'])
    return cv.png(), 'How many red squares are in the image? Answer with a single digit.', str(n)


def item_shape(rnd):
    kinds = [rnd.choice(['square', 'circle', 'triangle']) for _ in range(3)]
    cv = Canvas()
    for i, k in enumerate(kinds):
        draw(k, 48 + i * 80, 128, 30, COLORS['black'], cv)
    return (cv.png(), 'There are three shapes in a row. What kind of shape is the third (rightmost) one? Answer with '
            'one word: square, circle or triangle.', kinds[2])


def item_color(rnd):
    name = rnd.choice(sorted(COLORS))
    cv = Canvas()
    draw(rnd.choice(['square', 'circle', 'triangle']), 128, 128, 70, COLORS[name], cv)
    return cv.png(), 'What colour is the shape? Answer with one word: red, green, blue, yellow or black.', name


def item_position(rnd):
    q = rnd.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
    cx = rnd.randint(30, 98) + (128 if 'right' in q else 0)
    cy = rnd.randint(30, 98) + (128 if 'bottom' in q else 0)
    cv = Canvas()
    cv.disc(cx, cy, 20, COLORS['red'])
    return (cv.png(), 'Which quadrant of the image is the red dot in? Answer with one of: top-left, top-right, '
            'bottom-left, bottom-right.', q)


FAMILIES = [('count', item_count), ('shape', item_shape), ('color', item_color), ('position', item_position)]


def make_items(per_family, controls, seed=20261029):
    items = []
    for fam, f in FAMILIES:
        rnd = random.Random(f'{seed}-{fam}')
        for i in range(per_family):
            png, q, gold = f(rnd)
            items.append({'id': f'img-{fam}-{i:02d}', 'family': fam, 'png': png, 'question': q, 'gold': gold,
                          'control': False})
    rnd = random.Random(f'{seed}-control')
    for i in range(controls):
        fam, f = FAMILIES[i % len(FAMILIES)]
        _, q, gold = f(rnd)
        items.append({'id': f'img-control-{i:02d}', 'family': fam, 'png': None, 'question': q, 'gold': gold,
                      'control': True})
    return items


def grade(fam, gold, content):
    t = qlib.norm_text(content).lower()
    t = re.sub(r'\*', '', t)
    if fam == 'count':   # the LAST standalone integer or number word, in text order
        toks = re.findall(r'(?<![\d.,])\b(\d+|' + '|'.join(NUMWORDS) + r')\b(?![.,]?\d)', t)
        return bool(toks) and str(NUMWORDS[toks[-1]] if toks[-1] in NUMWORDS else int(toks[-1])) == gold
    if fam == 'position':   # the LAST quadrant named, in text order (rfind, as for shape/colour)
        t = re.sub(r'[\s_]+', '-', t).replace('upper', 'top').replace('lower', 'bottom')
        quads = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        pos = [(t.rfind(p), p) for p in quads if p in t]
        return bool(pos) and max(pos)[1] == gold
    vocab = ['square', 'circle', 'triangle'] if fam == 'shape' else sorted(COLORS)
    pos = [(t.rfind(v), v) for v in vocab if v in t]
    return bool(pos) and max(pos)[1] == gold


def user_msg(question, png):
    if png is None:
        return {'role': 'user', 'content': question}
    return {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_uri(png)}},
                                        {'type': 'text', 'text': question}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    qlib.add_common_args(ap)
    ap.add_argument('--per-family', type=int, default=20)
    ap.add_argument('--controls', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--save-images', action='store_true', help='also write the PNGs under OUT/images/')
    a = ap.parse_args()
    conn = qlib.conn_from_args(a)
    out = qlib.out_dir(a)
    info = qlib.server_info(conn)
    cond = {'temperature': 0.0, 'reasoning_effort': a.effort, 'max_tokens': a.max_tokens,
            'per_family': a.per_family, 'controls': a.controls, 'concurrency': a.concurrency}
    cv = Canvas()
    cv.rect(88, 88, 168, 168, COLORS['red'])
    for attempt in range(3):   # a 4xx is an answer; a transport error or 5xx is retried twice
        pr = qlib.chat(conn, [user_msg('What colour is the square in the image? Answer with one word.', cv.png())],
                       max_tokens=a.max_tokens, temperature=0.0, effort=a.effort, timeout=600)
        if not pr.error or pr.error.startswith('status: HTTP 4'):
            break
    served, why = True, None
    if pr.error:
        served, why = False, pr.error
        if not pr.error.startswith('status: HTTP 4'):
            print(f'images: the probe failed without a 4xx ({pr.error}); recorded as not served, re-run to retry')
    elif not any(c in pr.content.lower() for c in COLORS):
        served, why = False, f'probe answered without naming a colour: {pr.content[:200]!r}'
    if not served:
        print(f'images: NOT SERVED -- {why}')
        qlib.write_summary(out, 'images', 't0', [], primary='exact', metrics={}, condition=cond, tag=a.tag, conn=conn,
                           extra={'served': False, 'reason': why, 'server_info': info})
        return
    print(f'images: served (probe answered {pr.content.strip()[:60]!r})')
    items = make_items(a.per_family, a.controls)
    if a.save_images:
        os.makedirs(os.path.join(out, 'images'), exist_ok=True)
        for it in items:
            if it['png']:
                with open(os.path.join(out, 'images', it['id'] + '.png'), 'wb') as f:
                    f.write(it['png'])

    def score(it):
        r = qlib.chat(conn, [user_msg(it['question'], it['png'])], max_tokens=a.max_tokens, temperature=0.0,
                      effort=a.effort)
        rec = {'id': it['id'], 'family': it['family'], 'control': it['control'], 'gold': it['gold']}
        if r.error:
            rec.update(ok=None, error=r.error, wall_s=round(r.wall, 3))
            return rec
        ok = grade(it['family'], it['gold'], r.content)
        rec.update(ok=ok, metrics={('control' if it['control'] else 'exact'): ok}, content_tail=r.content[-200:],
                   **r.stats())
        return rec
    est = qlib.est_decode_s(len(items), 500, a.concurrency)
    recs, wall = qlib.run_items(items, score, concurrency=a.concurrency, out_jsonl=os.path.join(out, 'images-t0.jsonl'),
                                label='images', est_s=est)
    metrics = {'exact': qlib.metric_from_records(recs, 'exact'), 'control': qlib.metric_from_records(recs, 'control')}
    for fam, _ in FAMILIES:
        metrics[f'exact@{fam}'] = qlib.metric_from_records(recs, 'exact', where=lambda r, f=fam: r['family'] == f)
    qlib.write_summary(out, 'images', 't0', recs, primary='exact', metrics=metrics, condition=cond, wall_s=wall,
                       tag=a.tag, conn=conn, extra={'served': True, 'probe_answer': pr.content[:200],
                                                    'server_info': info})


if __name__ == '__main__':
    main()
