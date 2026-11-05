#!/usr/bin/env python3
"""Two-path TTFT probe: decomposes end-to-end latency across the deployment's network
hops so overhead measured end-to-end can be attributed to a specific hop.

Two paths, tested independently (each is skipped gracefully with a printed note if it
errors):
  loopback  the box's own engine origin (boxenv.engine_url(), also --base-url)
  door      the box's configured Caddy door (boxenv.door_url(), also --door-url)

Per path this runs, in order:
  (a) a 13-token prompt TTFT x3 (baseline TTFT with no prefill cost)
  (b) a streaming ~300-new-token completion (decode tok/s)
  (c) a fresh ~57k-token prompt, cold then warm (prefix-cache effect)
then prints a table with the per-hop delta door-loopback -- this is how
end-to-end overhead decomposes into "which hop actually adds the latency".

--deep runs the original, deeper single-path battery instead (mutated-prefix cache
invalidation, TTFT under background load, parallel /tokenize serialization) against exactly
one --paths entry; --self-test runs a small bounded version of it against loopback only.

Box selection and every URL/key/model default come from boxenv.py ($BOX or $BOX_ENV --
see that module's docstring); every value here also takes a CLI override. API key
(loopback + door): --api-key, else --env-file, else boxenv's own resolution ($VLLM_API_KEY,
/etc/environment, the box file's ENV_FILE). The resolved URL of every requested path is printed.

Ported and merged from the deep-dive scratch scripts probe1.py (a/b/c/e), probe2.py's
under_load ("under N streams") and probe3.py (8-parallel /tokenize).

Usage:
  python3 ttft-probe.py --api-key $VLLM_API_KEY             # loopback+door, box defaults
  python3 ttft-probe.py --paths loopback --api-key $VLLM_API_KEY --deep
  python3 ttft-probe.py --self-test --api-key $VLLM_API_KEY  # <=8 streams, <=200 tok, <=30s
  python3 ttft-probe.py --help
"""
import argparse
import json
import random
import statistics as st
import sys
import threading
import time
import urllib.error
import urllib.request

import boxenv

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu "
    "apple river stone cloud garden window silver copper marble velvet"
).split()


def _read_env_file_key(path):
    try:
        for line in open(path):
            line = line.strip()
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def load_key(api_key, env_file):
    if api_key:
        return api_key
    if env_file:
        k = _read_env_file_key(env_file)
        if k:
            return k
    return boxenv.resolve_api_key(None)


def gen_words(seed, n_words):
    rng = random.Random(seed)
    return [f"{rng.choice(WORDS)}{rng.randint(0, 999999)}" for _ in range(n_words)]


def tokens_to_words(n_tokens):
    """Empirical words-per-token ratio for WORDS + a random suffix (see gen_words)."""
    return max(1, int(n_tokens / 3.47))


def post(base, headers, path, body, timeout=600):
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json", "Connection": "close",
                 "User-Agent": boxenv.CURL_UA}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(base, headers, timeout=5):
    req = urllib.request.Request(base.rstrip("/") + "/metrics", headers={**headers, "User-Agent": boxenv.CURL_UA})
    out = {}
    try:
        text = urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception as e:
        return {"err": str(e)[:80]}
    for line in text.splitlines():
        for k in ("num_running_reqs", "num_queue_reqs", "spec_accept_length", "gen_throughput"):
            if line.startswith("sglang:" + k + "{") and 'tp_rank="0"' in line:
                out[k] = float(line.rsplit(" ", 1)[1])
    return out


def tokenize(base, headers, model, text, timeout=120):
    t0 = time.time()
    r = json.load(post(base, headers, "/tokenize", {"model": model, "prompt": text}, timeout=timeout))
    dt = time.time() - t0
    n = r.get("count") or len(r.get("tokens", []))
    return n, dt


def stream(base, headers, model, prompt, max_tokens=150, label="", quiet=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}
    m0 = metrics(base, headers)
    t0 = time.time()
    ttft = None
    usage = None
    last = t0
    maxgap = 0
    try:
        with post(base, headers, "/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                ch = o.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        maxgap = max(maxgap, now - last)
                    last = now
    except urllib.error.HTTPError as e:
        res = {"label": label, "err": str(e)[:120]}
        if not quiet:
            print(json.dumps(res), flush=True)
        return res
    except Exception as e:
        res = {"label": label, "err": str(e)[:120]}
        if not quiet:
            print(json.dumps(res), flush=True)
        return res
    wall = time.time() - t0
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    dec = ct / (wall - ttft) if ttft and ct and wall > ttft else None
    res = {"label": label, "t": time.strftime("%H:%M:%S", time.gmtime()),
           "ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 2),
           "prompt_tokens": pt, "completion_tokens": ct,
           "decode_tok_s": round(dec, 1) if dec else None, "max_itl_gap_s": round(maxgap, 3),
           "running_before": m0.get("num_running_reqs"), "queue_before": m0.get("num_queue_reqs")}
    if not quiet:
        print(json.dumps(res), flush=True)
    return res


def under_load(base, headers, model, C, seed, dur, fire_at, bg_tokens, ctx_words):
    ctx = " ".join(gen_words(seed, ctx_words))
    stream(base, headers, model, ctx + "\n\nSay OK.", 5, "prime_bg_ctx", quiet=True)
    stop = [False]
    bg = []
    lock = threading.Lock()

    def loop(i):
        k = 0
        while not stop[0]:
            tail = " ".join(gen_words(seed * 1000 + i * 100 + k, 30))
            k += 1
            r = stream(base, headers, model,
                       ctx + "\n\nNote " + tail + "\n\nWrite a 400-word story about a lighthouse keeper.",
                       bg_tokens, f"bg{i}", quiet=True)
            with lock:
                bg.append((i, r))

    ths = [threading.Thread(target=loop, args=(i,)) for i in range(C)]
    tstart = time.time()
    for t in ths:
        t.start()
    time.sleep(fire_at)
    fresh = " ".join(gen_words(seed + 1, ctx_words)) + "\n\nIgnoring everything above, write a 300-word story about a lighthouse keeper."
    tf = time.time()
    r = stream(base, headers, model, fresh, 150, f"fresh_under_load_C{C}", quiet=True)
    tfe = time.time()
    time.sleep(max(0, dur - (time.time() - tstart)))
    stop[0] = True
    for t in ths:
        t.join()
    ok = [x for i, x in bg if "err" not in x and x.get("ttft_s")]
    during = [x["max_itl_gap_s"] for x in ok]
    row = {"tag": f"ttft_under_load_C{C}", "t": time.strftime("%H:%M:%S", time.gmtime(tf)),
           "fresh_prompt_tokens": r.get("prompt_tokens"),
           "fresh_ttft_s": r.get("ttft_s") or r.get("err"),
           "fresh_dec_tok_s": r.get("decode_tok_s"),
           "bg_reqs": len(ok),
           "bg_maxgap_during_med": round(st.median(during), 2) if during else None,
           "bg_maxgap_during_max": round(max(during), 2) if during else None}
    print(json.dumps(row), flush=True)
    return row


def probe_path(name, base, headers, model, seed, target_tokens, decode_tokens):
    """The three-path battery for one path: 13-tok TTFT x3, ~decode_tokens streaming decode,
    a fresh ~target_tokens prompt cold then warm. Returns a result dict, or {"err": ...} on
    the first failed call (treated as this path being unreachable)."""
    print(f"-- path {name}: {base} model={model} --", flush=True)
    result = {"path": name, "base": base}

    ttfts = []
    for i in range(3):
        r = stream(base, headers, model, "Reply with the single word ONLINE and nothing else.",
                   20, f"{name}_13tok_rep{i + 1}")
        if "err" in r:
            result["err"] = r["err"]
            return result
        ttfts.append(r["ttft_s"])
    result["ttft_13tok_med"] = round(st.median(ttfts), 3)

    r = stream(base, headers, model, "Write a detailed essay about the history of lighthouses.",
               decode_tokens, f"{name}_decode{decode_tokens}")
    if "err" in r:
        result["err"] = r["err"]
        return result
    result["decode_tok_s"] = r.get("decode_tok_s")

    words = gen_words(seed, tokens_to_words(target_tokens))
    text = " ".join(words) + "\n\nIgnoring everything above, write a 300-word story about a lighthouse keeper."
    r_cold = stream(base, headers, model, text, 150, f"{name}_bigprompt_cold")
    if "err" in r_cold:
        result["err"] = r_cold["err"]
        return result
    r_warm = stream(base, headers, model, text, 150, f"{name}_bigprompt_warm")
    if "err" in r_warm:
        result["err"] = r_warm["err"]
        return result
    result["big_prompt_tokens"] = r_cold.get("prompt_tokens")
    result["ttft_big_cold"] = r_cold.get("ttft_s")
    result["ttft_big_warm"] = r_warm.get("ttft_s")
    return result


def print_table(rows):
    ok = {r["path"]: r for r in rows if "err" not in r}
    for r in rows:
        if "err" in r:
            print(f"skip {r['path']} ({r['base']}): {r['err']}", flush=True)
    order = [p for p in ("loopback", "door") if p in ok]
    if not order:
        print("no reachable paths to compare", flush=True)
        return

    metrics_list = [
        ("ttft_13tok_med", "ttft_13tok_s"),
        ("decode_tok_s", "decode_tok_s"),
        ("ttft_big_cold", "ttft_big_cold_s"),
        ("ttft_big_warm", "ttft_big_warm_s"),
    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v) if v is not None else "-"

    print("\n=== path comparison ===", flush=True)
    print("metric".ljust(18) + "".join(p.ljust(14) for p in order), flush=True)
    for key, label in metrics_list:
        vals = [ok[p].get(key) for p in order]
        print(label.ljust(18) + "".join(fmt(v).ljust(14) for v in vals), flush=True)

    print(flush=True)
    for a_name, b_name, tag in (("loopback", "door", "door-loopback"),):
        if a_name in ok and b_name in ok:
            for key, label in metrics_list:
                a, b = ok[a_name].get(key), ok[b_name].get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    print(f"delta {tag} {label}: {b - a:+.3f}", flush=True)


def run(base, headers, model, seed, target_tokens, mutate_offset, under_load_streams,
        max_tokens, tokenize_parallel, under_load_dur, under_load_fire_at):
    print("start", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "metrics", metrics(base, headers), flush=True)
    words = gen_words(seed, max(1, int(target_tokens / 3.47)))
    text = " ".join(words)
    n, dt = tokenize(base, headers, model, text)
    print(f"calib: {len(words)} words -> {n} tokens, /tokenize {dt:.3f}s", flush=True)
    nw = max(1, int(len(words) * target_tokens / n)) if n else len(words)
    words = words[:nw]
    text = " ".join(words)
    n, dt = tokenize(base, headers, model, text)
    print(f"prompt: {nw} words -> {n} tokens; /tokenize {dt:.3f}s", flush=True)
    n13, dt13 = tokenize(base, headers, model, "Reply with the single word ONLINE and nothing else.")
    print(f"  /tokenize 13-tok: {dt13:.4f}s", flush=True)
    tpw = n / nw if nw else 1
    wi = min(nw - 1, max(0, int(mutate_offset / tpw))) if tpw else 0
    npre, _ = tokenize(base, headers, model, " ".join(words[:wi])) if wi else (0, 0)
    print(f"mutation word index {wi} = token ~{npre}", flush=True)
    prompt_a = text + "\n\nIgnoring everything above, write a 300-word story about a lighthouse keeper."
    words_b = list(words)
    if words_b:
        words_b[wi] = "MUTATED" + words_b[wi]
    prompt_b = " ".join(words_b) + "\n\nIgnoring everything above, write a 300-word story about a lighthouse keeper."

    for i in range(3):
        stream(base, headers, model, "Reply with the single word ONLINE and nothing else.", 20, f"a_13tok_rep{i + 1}")
    stream(base, headers, model, prompt_a, max_tokens, "b_cold")
    stream(base, headers, model, prompt_a, max_tokens, "b_warm1")
    stream(base, headers, model, prompt_a, max_tokens, "b_warm2")
    stream(base, headers, model, prompt_b, max_tokens, "c_mut_first")
    stream(base, headers, model, prompt_b, max_tokens, "c_mut_again")
    tail = " ".join(gen_words(seed + 99, 350))
    stream(base, headers, model, prompt_a + "\n\nAdditional note: " + tail, max_tokens, "e_warm_plus800new")
    stream(base, headers, model, prompt_a + "\n\nAdditional note: " + tail, max_tokens, "e_warm_plus800new_again")

    out = [None] * tokenize_parallel
    T0 = time.time()

    def tok_one(i):
        json.load(post(base, headers, "/tokenize", {"model": model, "prompt": text + f" tail{i}"}, timeout=120))
        out[i] = time.time() - T0

    ths = [threading.Thread(target=tok_one, args=(i,)) for i in range(tokenize_parallel)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    print(f"{tokenize_parallel} parallel /tokenize of {n} tokens: completion times (s since fire): {sorted(round(o, 2) for o in out)}", flush=True)

    if under_load_streams > 0:
        under_load(base, headers, model, under_load_streams, seed + 500, under_load_dur, under_load_fire_at,
                   200, max(200, int(target_tokens / 3.47) // 4))

    print("end", time.strftime("%H:%M:%S UTC", time.gmtime()), "metrics", metrics(base, headers), flush=True)


def self_test(base, headers, model):
    start = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    t0 = time.time()
    print(f"self-test window start {start} (bounded: <=3 streams, <=15 tokens each, real-traffic TTFT can dominate)", flush=True)
    stream(base, headers, model, "Reply with the single word ONLINE and nothing else.", 10, "self_13tok")
    words = gen_words(1234, 400)
    text = " ".join(words)
    n, dt = tokenize(base, headers, model, text)
    print(f"self-test calib: {len(words)} words -> {n} tokens, /tokenize {dt:.3f}s", flush=True)
    prompt = text + "\n\nSay OK in one word."
    stream(base, headers, model, prompt, 15, "self_small_cold")
    out = [None] * 3

    def tok_one(i, T0):
        json.load(post(base, headers, "/tokenize", {"model": model, "prompt": text + f" tail{i}"}, timeout=60))
        out[i] = time.time() - T0

    T0 = time.time()
    ths = [threading.Thread(target=tok_one, args=(i, T0)) for i in range(3)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    print(f"self-test: 3 parallel /tokenize of {n} tokens: {sorted(round(o, 2) for o in out)}", flush=True)
    end = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"self-test window end {end} ({time.time() - t0:.1f}s elapsed)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", default=None,
                    help="comma-separated subset of loopback,door (default: both)")
    p.add_argument("--base-url", default=None,
                    help="loopback origin; default the box's engine URL (see boxenv.py)")
    p.add_argument("--door-url", default=None,
                    help="door origin; default the box's configured Caddy door (see boxenv.py)")
    p.add_argument("--api-key", default=None, help="bearer token for loopback/door")
    p.add_argument("--env-file", default=None, help="path to a VLLM_API_KEY=... env file")
    p.add_argument("--model", default=None, help="model for loopback/door (default: the box's served model)")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--target-tokens", type=int, default=57000,
                    help="size of the fresh cold/warm prompt (default: 57000)")
    p.add_argument("--decode-tokens", type=int, default=300,
                    help="new tokens streamed for the decode tok/s measurement (default: 300)")
    p.add_argument("--mutate-offset", type=int, default=1500, help="[--deep] token offset at which one word is mutated (default: 1500)")
    p.add_argument("--under-load-streams", type=int, default=8, help="[--deep] background streams for the under-load TTFT case (default: 8; 0 to skip)")
    p.add_argument("--under-load-duration", type=float, default=12.0)
    p.add_argument("--under-load-fire-at", type=float, default=3.0)
    p.add_argument("--max-tokens", type=int, default=150, help="[--deep] max tokens for the deep single-path battery")
    p.add_argument("--tokenize-parallel", type=int, default=8)
    p.add_argument("--deep", action="store_true",
                    help="run the original deep single-path battery (mutated-prefix cache "
                         "test, TTFT under background load, parallel /tokenize) against ONE "
                         "path named in --paths, instead of the three-path comparison")
    p.add_argument("--self-test", action="store_true", help="bounded run: small prompt, <=4 streams, <=30 max-tokens, well under 30s total, loopback only")
    args = p.parse_args()

    args.model = boxenv.resolve_served_model(args.model)

    is_default_paths = args.paths is None
    requested = [x.strip() for x in (args.paths.split(",") if args.paths else
                                      ["loopback", "door"]) if x.strip()]
    for name in requested:
        if name not in ("loopback", "door"):
            print(f"error: unknown path {name!r} (choose from loopback,door)", file=sys.stderr)
            sys.exit(2)

    path_urls = {}
    if "loopback" in requested:
        path_urls["loopback"] = boxenv.resolve_engine_url(args.base_url)
        print(f"base-url (loopback): {path_urls['loopback']}", flush=True)
    if "door" in requested:
        path_urls["door"] = boxenv.resolve_door_url(args.door_url)
        print(f"door-url: {path_urls['door']}", flush=True)

    vllm_key = None
    if "loopback" in requested or "door" in requested:
        vllm_key = load_key(args.api_key, args.env_file)

    def headers_for(name):
        return {"Authorization": "Bearer " + vllm_key}

    if args.self_test:
        if "loopback" in path_urls:
            base = path_urls["loopback"]
        else:
            base = boxenv.resolve_engine_url(args.base_url)
            print(f"base-url (loopback): {base}", flush=True)
        key = vllm_key or load_key(args.api_key, args.env_file)
        self_test(base, {"Authorization": "Bearer " + key}, args.model)
        return

    if not requested:
        print("error: no reachable path requested (all skipped)", file=sys.stderr)
        sys.exit(2)

    if args.deep:
        if len(requested) != 1:
            print("error: --deep requires exactly one path in --paths", file=sys.stderr)
            sys.exit(2)
        name = requested[0]
        run(path_urls[name], headers_for(name), args.model, args.seed, args.target_tokens,
            args.mutate_offset, args.under_load_streams, args.max_tokens, args.tokenize_parallel,
            args.under_load_duration, args.under_load_fire_at)
        return

    rows = []
    for name in ("loopback", "door"):
        if name not in requested:
            continue
        base = path_urls[name]
        try:
            row = probe_path(name, base, headers_for(name), args.model, args.seed,
                              args.target_tokens, args.decode_tokens)
        except Exception as e:
            row = {"path": name, "base": base, "err": str(e)[:160]}
        rows.append(row)

    print_table(rows)


if __name__ == "__main__":
    main()
