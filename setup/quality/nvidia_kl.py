#!/usr/bin/env python3
"""nvidia_kl.py -- NVIDIA's decode-vs-prefill KL test, ported to a standalone client.

Source: github.com/sgl-project/sgl-nvidia-mega at e34ae8b,
test/e2e/test_deepseek_v41_server.py: KL_BOUND / LONG_KL_BOUND (lines 55-59),
test_covered_steps_match_prefill + test_long_prompts_match_prefill (lines 280-310), and the
helpers they call in sglang.test.kl_test_utils (get_input_ids, _generate, _get_input_logprobs,
compare_kl_divergence; sglang main, last touched 4b384df0c9 2026-10-08) and the test's own
_logprobs / _long_prompts. Needs only the Python stdlib + requests; it talks to an SGLang
server's native API (/generate, /flush_cache, /v1/tokenize).
Re-read against both upstream files on 2026-10-29 (the test at e34ae8b, kl_test_utils.py on sglang
main): the bounds, both request bodies, the first 8 of 48 prompts for short, the 48-prompt join for
long, logr = input (prefill) - output (decode) logprob and the mean of per-prompt means all match.

Method (both modes, exactly NVIDIA's):
  1. POST /flush_cache (params timeout=30).
  2. ONE /generate request carrying every prompt of the mode as a batch of `input_ids`:
       sampling_params = {"temperature": 0.0, "max_new_tokens": N, "ignore_eos": True}
       return_logprob = True, return_text_in_logprobs = False, logprob_start_len = -1
     The "sampled" token is the greedy one (temperature 0.0); its decode-side logprob is
     meta_info.output_token_logprobs[k][0].
  3. POST /flush_cache, then ONE /generate batch of prompt + output_ids with
       sampling_params = {"temperature": 0.0, "max_new_tokens": 0, "ignore_eos": True}
       return_logprob = True, return_text_in_logprobs = False, logprob_start_len = 0
     i.e. the generated sequence is re-scored by the normal prefill path; the prefill-side
     logprob of each generated token is the last len(output) entries of
     meta_info.input_token_logprobs[k][0].
  4. Per prompt: k3 KL estimator on the sampled token, mean over its generated positions,
       logr = prefill_logprob - decode_logprob ; k3 = (exp(logr) - 1) - logr
     then avg_kl = the plain mean of the per-prompt means. Pass iff avg_kl < bound.

  short (test_covered_steps_match_prefill): the first 8 of the 48 cached LongBench-v2 prompts,
         256 new tokens, bound KL_BOUND = 0.03.
  long  (test_long_prompts_match_prefill): the 48 cached prompts' tokens joined end to end and
         cut into windows of 18000, 21000, 24000, 27000 tokens, each starting n//2 after the
         previous window's start; 128 new tokens; bound LONG_KL_BOUND = 0.06.

Prompts (kl_test_utils.get_input_ids, DEFAULT_PROMPT_TOKENS=3000, DEFAULT_NUM_SAMPLES=48): the
first 48 records of THUDM/LongBench-v2 split train (its data.json, in file order, which is what
datasets' streaming load yields), text = f"{context} {question}", tokens =
tokenizer.encode(text) (special tokens added: the BOS), truncated to
int(3000 * random.uniform(0.5, 1.5)) tokens, one draw per record in order.
Deviations, all forced by "stdlib + requests" or by reproducibility, none touching sampling:
  * tokenization goes through the server's own tokenizer (POST /v1/tokenize,
    add_special_tokens=True == tokenizer.encode's default) instead of a local HF tokenizer;
  * a context longer than TEXT_CAP characters is cut to TEXT_CAP before tokenizing (the kept
    span is <= 4500 tokens, far inside TEXT_CAP characters, so the kept tokens are the same);
  * NVIDIA's truncation draws from the unseeded global `random`, so their prompt lengths differ
    run to run; here the draws come from random.Random(--seed) (default 0) and the token lists
    are written once to --prompts (a JSON list of 48 token lists, the same shape as their
    cache file) so every boot is scored on byte-identical prompts.
  * every request carries `Authorization: Bearer <key>` (our engines require it).

usage:
  nvidia_kl.py [--url http://127.0.0.1:8002] [--mode short|long|both] [--repeats N]
               [--prompts PATH] [--seed 0] [--out OUT.json] [--label NAME]
  key: --api-key, else $VLLM_API_KEY, else VLLM_API_KEY= in /etc/environment.
The output JSON has one record per (repeat, mode) with avg_kl, the bound, pass, and per-prompt
values (prompt tokens, generated tokens, mean/max k3, the output token ids' sha256, the
largest |logr| position). Exit 0 when every run passed, 1 when one failed its bound.
"""

import argparse
import codecs
import hashlib
import json
import math
import os
import random
import sys
import time

import requests

KL_BOUND = 0.03
LONG_PROMPT_TOKENS = (18000, 21000, 24000, 27000)
LONG_KL_BOUND = 0.06
SHORT_SAMPLES = 8
SHORT_NEW_TOKENS = 256
LONG_NEW_TOKENS = 128
NUM_SAMPLES = 48
PROMPT_TOKENS = 3000
TEXT_CAP = 100_000
DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"


def api_key(arg):
    if arg:
        return arg
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    try:
        for line in open("/etc/environment"):
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


class Client:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.h = {"Authorization": "Bearer " + key} if key else {}

    def flush(self):
        r = requests.post(self.url + "/flush_cache", params={"timeout": 30}, headers=self.h, timeout=40)
        r.raise_for_status()

    def generate(self, input_ids, max_new_tokens, logprob_start_len):
        body = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens, "ignore_eos": True},
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": logprob_start_len,
        }
        r = requests.post(self.url + "/generate", json=body, headers=self.h, timeout=3600)
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, list) or len(out) != len(input_ids):
            raise RuntimeError("batch /generate returned %r" % (str(out)[:500],))
        return out

    def tokenize(self, text):
        r = requests.post(self.url + "/v1/tokenize", json={"prompt": text, "add_special_tokens": True},
                          headers=self.h, timeout=600)
        r.raise_for_status()
        d = r.json()
        toks = d.get("tokens") if isinstance(d, dict) else None
        if not isinstance(toks, list) or not toks or not isinstance(toks[0], int):
            raise RuntimeError("/v1/tokenize returned %r" % (str(d)[:300],))
        return toks

    def server_info(self):
        for path in ("/server_info", "/get_server_info"):
            try:
                r = requests.get(self.url + path, headers=self.h, timeout=30)
                if r.ok:
                    return r.json()
            except requests.RequestException:
                pass
        return {}


def stream_records(url, n):
    """The first n records of a JSON-array file, parsed while it streams (no full download)."""
    dec = json.JSONDecoder()
    udec = codecs.getincrementaldecoder("utf-8")()
    buf, pos, need, recs = "", 0, 0, []
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        chunks = r.iter_content(chunk_size=1 << 20)
        started = False
        for chunk in chunks:
            buf += udec.decode(chunk)
            if len(buf) < need:
                continue
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if not started:
                    if pos >= len(buf):
                        break
                    if buf[pos] != "[":
                        raise RuntimeError("dataset file is not a JSON array")
                    started, pos = True, pos + 1
                    continue
                if pos >= len(buf):
                    break
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    need = len(buf) + max(1 << 20, (len(buf) - pos) // 2)
                    break
                recs.append(obj)
                if len(recs) >= n:
                    return recs
                buf, pos, need = buf[end:], 0, 0
    if len(recs) < n:
        raise RuntimeError("dataset ended after %d records" % len(recs))
    return recs


def build_prompts(cli, path, seed, data_url):
    rng = random.Random(seed)
    recs = stream_records(data_url, NUM_SAMPLES)
    ids, meta = [], []
    for i, ex in enumerate(recs):
        context, question = ex.get("context", ""), ex.get("question", "")
        capped = len(context) > TEXT_CAP
        text = f"{context[:TEXT_CAP] if capped else context} {question}"
        toks = cli.tokenize(text)
        cut = int(PROMPT_TOKENS * rng.uniform(0.5, 1.5))
        if capped and len(toks) < cut:
            raise RuntimeError("record %d: TEXT_CAP too small (%d tokens < %d)" % (i, len(toks), cut))
        ids.append(toks[:cut])
        meta.append({"i": i, "_id": ex.get("_id"), "context_chars": len(context), "capped": capped,
                     "tokens_full_or_capped": len(toks), "truncate_len": cut, "kept": len(ids[-1])})
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(ids, f)
    os.replace(tmp, path)
    with open(path + ".meta.json", "w") as f:
        json.dump({"data_url": data_url, "seed": seed, "num_samples": NUM_SAMPLES,
                   "max_prompt_tokens": PROMPT_TOKENS, "text_cap_chars": TEXT_CAP,
                   "tokenizer": "server /v1/tokenize add_special_tokens=True", "records": meta}, f, indent=1)
    return ids


def long_prompts(ids):
    text = [t for p in ids for t in p]
    out, at = [], 0
    for n in LONG_PROMPT_TOKENS:
        out.append(text[at:at + n])
        at += n // 2
    if not all(len(p) == n for p, n in zip(out, LONG_PROMPT_TOKENS)):
        raise RuntimeError("corpus too short")
    return out


def k3_stats(inp, outp):
    vals, worst = [], (0.0, -1)
    for j, (a, b) in enumerate(zip(inp, outp)):
        logr = a - b
        vals.append(math.expm1(logr) - logr)
        if abs(logr) > abs(worst[0]):
            worst = (logr, j)
    return sum(vals) / len(vals), max(vals), worst


def run_mode(cli, mode, prompts):
    n_new = SHORT_NEW_TOKENS if mode == "short" else LONG_NEW_TOKENS
    bound = KL_BOUND if mode == "short" else LONG_KL_BOUND
    cli.flush()
    t0 = time.time()
    dec = cli.generate(prompts, n_new, -1)
    t1 = time.time()
    outputs = [[x[0] for x in r["meta_info"]["output_token_logprobs"]] for r in dec]
    full = [p + r["output_ids"] for p, r in zip(prompts, dec)]
    cli.flush()
    pre = cli.generate(full, 0, 0)
    t2 = time.time()
    inputs = [[x[0] for x in r["meta_info"]["input_token_logprobs"]][-len(o):] for r, o in zip(pre, outputs)]
    per = []
    for k, (p, r, i, o) in enumerate(zip(prompts, dec, inputs, outputs)):
        mean, mx, (wl, wj) = k3_stats(i, o)
        mi = r["meta_info"]
        per.append({
            "k": k, "prompt_tokens": len(p), "gen_tokens": len(o), "kl": mean, "k3_max": mx,
            "worst_logr": wl, "worst_pos": wj,
            "out_sha256": hashlib.sha256(json.dumps(r["output_ids"]).encode()).hexdigest()[:16],
            "finish": (mi.get("finish_reason") or {}).get("type") if isinstance(mi.get("finish_reason"), dict) else mi.get("finish_reason"),
            "cached_tokens": mi.get("cached_tokens"),
            "spec_verify_ct": mi.get("spec_verify_ct"),
            "spec_accept_length": mi.get("spec_accept_length"),
        })
    avg = sum(x["kl"] for x in per) / len(per)
    return {"mode": mode, "n_prompts": len(prompts), "max_new_tokens": n_new, "avg_kl": avg,
            "bound": bound, "pass": avg < bound, "decode_s": round(t1 - t0, 2),
            "prefill_s": round(t2 - t1, 2), "per_prompt": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", choices=("short", "long", "both"), default="both")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--prompts", default="longbench-v2-input_ids-3000-48-seed0.json",
                    help="token-list cache; built from LongBench-v2 through the server when absent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-url", default=DATA_URL)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    cli = Client(a.url, api_key(a.api_key))
    if os.path.exists(a.prompts):
        ids = json.load(open(a.prompts))
    else:
        print("building %s from %s (seed %d)" % (a.prompts, a.data_url, a.seed), flush=True)
        ids = build_prompts(cli, a.prompts, a.seed, a.data_url)
    if len(ids) != NUM_SAMPLES:
        raise SystemExit("prompt cache holds %d prompts, want %d" % (len(ids), NUM_SAMPLES))
    psha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    si = cli.server_info()
    keep = ("model_path", "tp_size", "ep_size", "speculative_algorithm", "speculative_dspark_block_size",
            "kv_cache_dtype", "max_total_num_tokens", "version", "enable_nvidia_megakernel")
    info = {k: si.get(k) for k in keep if k in si}
    if "max_total_num_tokens" not in info:
        for st in si.get("internal_states") or []:
            if isinstance(st, dict) and "max_total_num_tokens" in st:
                info["max_total_num_tokens"] = st["max_total_num_tokens"]
                break

    modes = ("short", "long") if a.mode == "both" else (a.mode,)
    runs = []
    for rep in range(a.repeats):
        for m in modes:
            prompts = ids[:SHORT_SAMPLES] if m == "short" else long_prompts(ids)
            res = run_mode(cli, m, prompts)
            res["repeat"] = rep
            runs.append(res)
            print("%s rep%d %-5s avg_kl=%.6f bound=%.2f %s  per=[%s]  (decode %.1fs, prefill %.1fs)" % (
                a.label, rep, m, res["avg_kl"], res["bound"], "PASS" if res["pass"] else "FAIL",
                " ".join("%.4f" % x["kl"] for x in res["per_prompt"]), res["decode_s"], res["prefill_s"]),
                flush=True)
    doc = {"label": a.label, "url": a.url, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": "sgl-project/sgl-nvidia-mega@e34ae8b test/e2e/test_deepseek_v41_server.py",
           "sampling": {"decode": {"temperature": 0.0, "ignore_eos": True, "logprob_start_len": -1,
                                   "return_text_in_logprobs": False},
                        "prefill": {"temperature": 0.0, "max_new_tokens": 0, "ignore_eos": True,
                                    "logprob_start_len": 0}, "batch": "one /generate per phase, every prompt"},
           "prompts": {"path": os.path.basename(a.prompts), "sha256_16": psha, "seed": a.seed},
           "server": info, "runs": runs}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=1)
    sys.exit(0 if all(r["pass"] for r in runs) else 1)


if __name__ == "__main__":
    main()
