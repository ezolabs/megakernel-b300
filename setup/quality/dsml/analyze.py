#!/usr/bin/env python3
"""analyze.py -- read the raw transcripts probe.py recorded and judge them. Nothing here sends a request. Stdlib only
(the parse itself runs in a child under --detector-py), so `python3 -I` is safe.

  python3 -I analyze.py report <dir>
      #2537's report: the phases fix/ control/ fix3/ fix4/ under <dir> (the gpu1018b records) -> <dir>/results.json +
      <dir>/LIVE.generated.md (the per-check table its LIVE.md quotes).
  python3 -I analyze.py grade <phase dir>... (--detector-tree <sha> [--sglang-repo DIR] | --detector-src DIR)
                       [--detector-py PY] [--tool-parser deepseekv41] [--reasoning-parser deepseek-v41]
                       [--tokenizer <checkpoint>/tokenizer.json] [--tree <sha>] [--expect-unset pass|fail [--force-expect]]
      THE DSML GATE, as the user defined it: each request is graded INSIDE ITSELF. Its own raw output (the sampled
      token ids the engine returned in sglext.output_ids, detokenized by the engine: probe.py's <mode>.raw.txt) is
      re-parsed IN ONE SHOT by the served tree's own parser -- serving_chat's non-stream path: the reasoning parser's
      parse_non_stream, then FunctionCallParser(tools, <tool parser>, tool_choice).parse_non_stream, i.e. the DSML
      detector's detect_and_parse -- and that re-parse must give the same calls, names and arguments, BYTE FOR BYTE,
      as the request returned: a stream's concatenated tool_calls deltas, a non-stream response's tool_calls. It does
      not depend on the model generating the same text twice. And the returned arguments must be the MODEL'S OWN
      (EMITTED): each a JSON object whose keys and values are the parameters the model wrote in its DSML, read by
      emitted_calls with the encoder's grammar and no served parser -- a parser that re-reads its own misreading
      byte-equal (box-e's deepseek_recipe 0.2.0 serves `name="first name"` as `{"first: ...}`) FAILs here. Beside it, per case: valid JSON, router replay clean,
      the case's own bars (one write_file call and its first tool_calls delta within 5 s, declared parameter names
      whole, ...). The cross-send comparison (stream vs non-stream of one case) is an `info` row only. Prints one row
      per request and DSML-SELFPARSE PASS|FAIL; writes <phase>/selfparse.json; exit 0 only on PASS.
      Every expected case x mode must carry a graded row (expected_rows: each case x mode the phase's probe.log START
      lines asked for -- a phase without a probe.log, every case dir x every mode recorded in it -- and either
      T1_todo_nested arm brings the other): a missing one prints `FAIL <phase> <case> <mode> MISSING: <why>`, lands in
      selfparse.json's "missing" and FAILs the grade, never a skip.
      A REPEATS case (probe.py's T1_todo_nested, <case>/repeat.json + one s<i>/ per send: Cosmos's real todo_write, a
      nested `todos` array, n >= 369 per mode at T 1.0 / top_p 1.0) prints one row per MODE, graded by count
      (todo_verdict): every send keeps its own bars above (HTTP, stream, router replay, self-parse byte-equal), and each
      call is read twice -- RAW, the model's DSML from its own output ids decoded by --tokenizer (else the engine's
      /detokenize text), and PARSED, the returned arguments -- for padded (`"id "`), garbled (`"id: 20, "`), invented
      (undeclared: `"notes_file_placehold"`) keys and unparseable arguments, plus the sends where the two sides differ;
      the mode passes only if each of the three counts is <= todo_max_bad(its calls, its null) -- a binomial bar at
      <= 0.1 % false-fail per count, from the calls the count ACTUALLY carries. Each side's defective calls: TODO_BAR,
      a CATASTROPHE gate, not a regression detector: p0 = the probe's pooled healthy arms (TODO_HEALTHY_ARMS, the
      config under test's own arms excluded: 48 of 6588 calls, 0.73 %; 470 calls: 10, 369: 9), >= 99 % power at a 5 %
      defect across both modes (86 % at 2.5 %), but a 2 % defect passes one mode of 470 calls 66 % of the time, and a
      count whose calls no bar can give 99 % power (< 234 calls) FAILs as too few calls for the gate. The raw-vs-parsed count: its OWN null, TODO_DIFFER_BAR
      (TODO_HEALTHY_DIFFER: 1 differing send of 7520 in the same healthy arms, 0.013 %; 470 sends: 2, 333: 1), so a
      parser-only defect fails at 3 of 470. It prints each count's n, k and power at 5 % and 2.5 %, the sides' one-sided
      binomial p at p0 and Fisher p against the pooled healthy arms (evidence beside the verdict, not the verdict),
      and the forms found.
      T1_todo_nested has two arms, each its own rows: T1_todo_nested (the tool's `strict` unset, straight to the
      engine: the tree's default) and T1_todo_nested_strictfalse (`"strict": false`, what the router posts for every
      unset tool, i.e. prod traffic), each graded by the bar above. The strict:false arm must PASS on every build.
      The unset arm must PASS on a tree carrying fork #901 (--expect-unset pass); on any tree without #901 it FAILs
      by design (~99.7 % defective on box-a's 819756d4). The expectation FOLLOWS THE TREE UNDER TEST (grade_expect):
      --tree <sha> (else --detector-tree's; the two naming different commits exits 1) is asked, read-only, of the
      fork checkout --sglang-repo (nothing is fetched) whether a STRICT_FIX_COMMITS commit (601f5614, faabdcae) is
      its ancestor -- a fix is pass, none fail -- and an --expect-unset that contradicts it exits 1 before any
      grading, --force-expect or not; a tree the checkout cannot place exits 1 unless --expect-unset is explicit
      with --force-expect; with no tree named, `fail` (the value that excuses) needs --force-expect and `pass` is
      the default. It prints `EXPECT-UNSET <how it was decided>`, written to selfparse.json as "expect_unset_from".
      On a no-fix tree (expect fail) the arm's FAIL prints EXPECTED-FAIL and does not fail the gate (a PASS there
      still passes, with a note). Only the defect-rate bars are excused, with the raw-vs-parsed bar (emitted diffs
      included) when one of them failed too -- the same defect showing, garbled keys parsing differently (box-a's
      819756d4: differ 13 / 16 of 369 over a bar of 2; alone it is a parser defect and still FAILs): a send error (HTTP,
      stream, router replay, self-parse), too few sends or calls, or off-prod sampling still FAILs the unset arm
      under --expect-unset fail (todo_unexcused).
      The parser runs under --detector-py (default <HOME>/workspace/sglang/.venv/bin/python, or $DSML_DETECTOR_PY: any
      python with sglang's dependencies; on a box, /opt/sglang-up/bin/python3) from the served tree's python/ dir:
      --detector-src <tree>/python, or --detector-tree <sha>, which `git archive`s python/sglang at that commit from
      --sglang-repo (default <HOME>/workspace/sglang) into ~/.cache/dsml-detector/sglang-<sha>/ once.
      The tree picks the path (tree_kind): box-d and before render with entrypoints/openai/encoding_dsv41.py and parse
      as above; box-e on (parser/deepseek_recipe_output.py present) parse as serving_deepseek_recipe's
      parse_chat_response -- DeepSeekRecipeOutputParser over the sampled ids (<mode>.output_ids.json when they decode
      to the raw text under --tokenizer, else the raw text encoded) -- under a detector python that also holds the
      deepseek_recipe package (on a box, /opt/sglang-box-e/bin/python3), with the checkpoint's tokenizer.json
      (--tokenizer, or $DSML_TOKENIZER; a directory means its tokenizer.json). CPU proof of both shapes on the box-e
      cutover's own outputs: tests/test_grade_trees.py.
  python3 -I analyze.py self-test --out DIR [--records <phase dir>...] [--real <phase dir>...] (--detector-tree <sha> | ...)
      With neither --records nor --real it runs on the tracked fixtures, tests/fixtures/1402ef5be8/{names,probecases}
      (box-c's own raw outputs on tree 1402ef5be8: `--detector-tree 1402ef5be80d91dc60002fce2c297f880518103a`), as both.
      The gate's own proof on recorded transcripts (which carry no raw output): each recorded stream's calls are
      rendered back into the DSML text the model emits (encoding_dsv41's own templates: the served tree's encoder) as
      that request's raw output, and the grade must PASS; mutated copies (one changed byte in a value, a dropped
      fragment, a renamed call, a missing raw text, a reasoning block quoting DSML) must FAIL or PASS as they should;
      and probe.py runs end to end against a local replay server that returns sglext.output_ids and answers
      /detokenize, and its output must grade PASS. --real names transcripts that DID record each request's raw output
      (#2539's inreq.py runs on box-c): graded as recorded they must PASS, agree with their inreq.json, and a changed byte
      must FAIL. Prints SELF-TEST PASS|FAIL.

Per case it reads <phase>/<case>/stream.sse + stream.times.tsv (raw SSE, arrival times) and nonstream.json (or #2539
inreq.py's untimed stream.raw / nonstream.raw / <mode>.detok.txt, the layout of the tracked fixtures), then:
  * timing: first SSE line, first chunk carrying a tool_calls delta, first non-empty arguments fragment, last
    reasoning_content chunk, last line; the largest gap between consecutive tool_calls chunks
  * per tool-call index: name(s), id(s), the arguments fragments in order and their concatenation
  * stream vs non-stream: call count, names, arguments compared as UTF-8 bytes (sha256 + first differing offset),
    content and finish_reason
  * the router's restatement rule (the serving router crates/sbr-wire/src/sglang/stream.rs:473-494,
    ClientArguments::delta) replayed over the fragments of each index: a fragment that starts with everything sent
    so far and is longer is read as a cumulative restatement (only its suffix forwarded); a fragment equal to
    everything sent after such a restatement is withheld. The replay's client text must equal the raw concatenation.
"""
import ast
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# The tracked fixtures: real box-c records that carry each request's own raw output (tests/: never staged to a box).
FIXTURE_PHASES = [os.path.join(HERE, "tests", "fixtures", "1402ef5be8", p) for p in ("names", "probecases")]
FIRST_DELTA_BAR_S = 5.0     # "a few seconds, not tens"
MIN_ARG_TOKENS = 5000


def sha(b):
    return hashlib.sha256(b).hexdigest()


def router_replay(fragments):
    """the serving router ClientArguments::delta, replayed. Returns (client_text, events)."""
    sent, restated, client, events = "", False, [], []
    for i, a in enumerate(fragments):
        if sent and a.startswith(sent) and len(a) > len(sent):
            suffix = a[len(sent):]
            events.append({"fragment": i, "kind": "read-as-restatement", "fragment_len": len(a), "sent_len": len(sent)})
            sent, restated = a, True
            client.append(suffix)
            continue
        if restated and a == sent:
            events.append({"fragment": i, "kind": "withheld-as-repeat", "fragment_len": len(a)})
            continue
        sent += a
        client.append(a)
    return "".join(client), events


def sse_path(d, name="stream"):
    """probe.py's <name>.sse, or #2539 inreq.py's <name>.raw (the same SSE bytes, untimed)."""
    for p in (os.path.join(d, name + ".sse"), os.path.join(d, name + ".raw")):
        if os.path.isfile(p):
            return p
    return None


def parse_stream(d, name="stream"):
    sp = sse_path(d, name) or os.path.join(d, name + ".sse")
    raw = open(sp, "rb").read()
    times = []
    tp = os.path.join(d, name + ".times.tsv")
    if os.path.isfile(tp):
        for row in open(tp).read().splitlines()[1:]:
            t, off, ln = row.split("\t")
            times.append((float(t), int(off), int(ln)))
    else:   # untimed: every line, time unknown
        off = 0
        for line in raw.splitlines(keepends=True):
            times.append((None, off, len(line)))
            off += len(line)
    r = {"lines": len(times), "bytes": len(raw), "t_first_line": times[0][0] if times else None,
         "t_last_line": times[-1][0] if times else None, "done": False, "errors": [], "content": "",
         "reasoning_len": 0, "finish_reason": None, "t_first_tool_delta": None, "t_first_args": None,
         "t_first_content": None, "t_last_reasoning": None, "tool_chunk_times": [], "calls": {}, "bad_lines": 0,
         "output_ids": None}
    for t, off, ln in times:
        line = raw[off:off + ln].rstrip(b"\r\n")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            r["done"] = True
            continue
        try:
            ch = json.loads(payload.decode("utf-8"))
        except Exception:
            r["bad_lines"] += 1
            continue
        if "error" in ch and not ch.get("choices"):
            r["errors"].append(ch["error"])
            continue
        oi = (ch.get("sglext") or {}).get("output_ids")
        if isinstance(oi, list) and oi and isinstance(oi[0], list):
            r["output_ids"] = len(oi[0])
        for c in ch.get("choices") or []:
            dl = c.get("delta") or {}
            if dl.get("content"):
                r["content"] += dl["content"]
                if r["t_first_content"] is None:
                    r["t_first_content"] = t
            if dl.get("reasoning_content"):
                r["reasoning_len"] += len(dl["reasoning_content"])
                r["t_last_reasoning"] = t
            tcs = dl.get("tool_calls") or []
            if tcs:
                r["tool_chunk_times"].append(t)
                if r["t_first_tool_delta"] is None:
                    r["t_first_tool_delta"] = t
            for tc in tcs:
                i = tc.get("index", 0)
                e = r["calls"].setdefault(i, {"names": [], "ids": [], "fragments": [], "frag_times": []})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    e["names"].append(fn["name"])
                if tc.get("id"):
                    e["ids"].append(tc["id"])
                a = fn.get("arguments")
                if a is not None:
                    e["fragments"].append(a)
                    e["frag_times"].append(t)
                    if a and r["t_first_args"] is None:
                        r["t_first_args"] = t
            if c.get("finish_reason"):
                r["finish_reason"] = c["finish_reason"]
    ts = r.pop("tool_chunk_times")
    r["tool_chunks"] = len(ts)
    r["timed"] = bool(times) and times[0][0] is not None
    r["max_gap_between_tool_chunks_s"] = round(max((b - a for a, b in zip(ts, ts[1:])), default=0.0), 3) if r["timed"] else None
    r["t_last_tool_chunk"] = ts[-1] if ts else None
    return r


def _keys(a):
    try:
        o = json.loads(a or "")
        return list(o.keys()) if isinstance(o, dict) else None
    except Exception:
        return None


def parse_nonstream(d):
    p = os.path.join(d, "nonstream.json")
    if not os.path.isfile(p):
        p = os.path.join(d, "nonstream.raw")   # #2539 inreq.py's layout
    if not os.path.isfile(p):
        return None
    raw = open(p, "rb").read()
    try:
        j = json.loads(raw)
    except Exception as e:
        return {"error": "not JSON: %r" % e, "raw_head": raw[:300].decode("utf-8", "replace")}
    if "choices" not in j:
        return {"error": j}
    m = j["choices"][0].get("message") or {}
    calls = [{"name": (c.get("function") or {}).get("name"), "arguments": (c.get("function") or {}).get("arguments"),
              "id": c.get("id"), "arg_keys": _keys((c.get("function") or {}).get("arguments"))} for c in (m.get("tool_calls") or [])]
    return {"content": m.get("content") or "", "reasoning_len": len(m.get("reasoning_content") or ""),
            "finish_reason": j["choices"][0].get("finish_reason"), "calls": calls, "usage": j.get("usage")}


def nonascii_profile(s):
    return {"non_ascii_chars": sum(1 for ch in s if ord(ch) > 127),
            "cjk_chars": sum(1 for ch in s if 0x4E00 <= ord(ch) <= 0x9FFF),
            "latin_accented_chars": sum(1 for ch in s if 0xC0 <= ord(ch) <= 0x17F),
            "emoji_chars": sum(1 for ch in s if ord(ch) > 0xFFFF)}


def analyze_case(d, tokens):
    s = parse_stream(d) if sse_path(d) else None
    n = parse_nonstream(d)
    out = {"stream": None, "nonstream": None, "compare": None}
    meta = {}
    for m in ("stream", "nonstream"):
        p = os.path.join(d, m + ".meta.json")
        if os.path.isfile(p):
            meta[m] = json.load(open(p))
    out["meta"] = meta
    if s is not None:
        calls = []
        for i in sorted(s["calls"]):
            e = s["calls"][i]
            cat = "".join(e["fragments"])
            replay, events = router_replay(e["fragments"])
            keys = None
            try:
                o = json.loads(cat)
                valid = True
                keys = list(o.keys()) if isinstance(o, dict) else None
            except Exception:
                valid = False
            calls.append({"index": i, "names": e["names"], "ids": sorted(set(e["ids"])), "fragments": len(e["fragments"]),
                          "args_chars": len(cat), "args_bytes": len(cat.encode("utf-8")),
                          "args_sha256": sha(cat.encode("utf-8")), "args_valid_json": valid, "arg_keys": keys,
                          "t_first_fragment": e["frag_times"][0] if e["frag_times"] else None,
                          "t_last_fragment": e["frag_times"][-1] if e["frag_times"] else None,
                          "router_replay_equal": replay == cat, "router_events": events,
                          "first_fragments": [f[:80] for f in e["fragments"][:4]],
                          "_args": cat})
        s["calls"] = calls
        if s["t_first_tool_delta"] is not None and s["t_last_reasoning"] is not None:
            s["first_tool_delta_after_reasoning_s"] = round(s["t_first_tool_delta"] - s["t_last_reasoning"], 3)
        out["stream"] = s
    out["nonstream"] = n
    if s is not None and n is not None and "calls" in n:
        sa = [c["_args"] for c in s["calls"]]
        na = [c["arguments"] or "" for c in n["calls"]]
        per = []
        for i in range(max(len(sa), len(na))):
            a = sa[i].encode("utf-8") if i < len(sa) else None
            b = na[i].encode("utf-8") if i < len(na) else None
            diff = None
            if a is not None and b is not None and a != b:
                diff = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            per.append({"index": i, "equal_bytes": a == b, "stream_sha256": sha(a) if a is not None else None,
                        "nonstream_sha256": sha(b) if b is not None else None,
                        "stream_bytes": len(a) if a is not None else None, "nonstream_bytes": len(b) if b is not None else None,
                        "first_diff_offset": diff,
                        "context": None if diff is None else {"stream": a[max(0, diff - 60):diff + 60].decode("utf-8", "replace"),
                                                              "nonstream": b[max(0, diff - 60):diff + 60].decode("utf-8", "replace")}})
        out["compare"] = {
            "calls_stream": len(sa), "calls_nonstream": len(na),
            "names_stream": [c["names"][0] if c["names"] else None for c in s["calls"]],
            "names_nonstream": [c["name"] for c in n["calls"]],
            "per_call": per,
            "all_args_equal": len(sa) == len(na) and all(p["equal_bytes"] for p in per),
            "content_equal": s["content"] == n["content"],
            "finish_reason": [s["finish_reason"], n["finish_reason"]],
        }
        if len(sa) == len(na) and sa:
            prof = nonascii_profile(sa[0])
            out["compare"]["args0_profile"] = prof
            try:
                o = json.loads(na[0])
                if isinstance(o, dict) and "content" in o:
                    out["compare"]["write_file"] = {"path": o.get("path"), "content_chars": len(o["content"]),
                                                    "content_lines": o["content"].count("\n") + 1}
            except Exception:
                pass
    if s is not None and os.path.isfile(os.path.join(d, "stream2.sse")):
        s2 = parse_stream(d, "stream2")
        a1 = ["".join(s["calls"][i]["_args"] for _ in [0]) for i in range(len(s["calls"]))]
        a2 = ["".join(s2["calls"][i]["fragments"]) for i in sorted(s2["calls"])]
        out["stream2"] = {"t_first_tool_delta": s2["t_first_tool_delta"], "t_last_tool_chunk": s2["t_last_tool_chunk"],
                          "finish_reason": s2["finish_reason"], "calls": len(a2),
                          "args_equal_stream": a1 == a2,
                          "args_equal_nonstream": n is not None and "calls" in n and a2 == [c["arguments"] or "" for c in n["calls"]],
                          "args_sha256": [sha(x.encode("utf-8")) for x in a2]}
    tk = tokens.get(os.path.basename(d))
    if tk:
        out["tokens"] = tk
    for c in (out["stream"] or {}).get("calls", []):
        c.pop("_args", None)
    return out


def verdicts(ph, res):
    v = {}

    def g(case):
        return res.get(case) or {}

    def ok_long(case, high):
        r = g(case)
        s, c, tk = r.get("stream") or {}, r.get("compare") or {}, r.get("tokens") or {}
        why = []
        at = (tk.get("arguments_tokens") or [0])[0] if tk.get("arguments_tokens") else 0
        if at < MIN_ARG_TOKENS:
            why.append("arguments %s tokens < %d" % (at, MIN_ARG_TOKENS))
        t = s.get("t_first_tool_delta")
        if t is None:
            why.append("no tool_calls delta in the stream")
        else:
            lat = s.get("first_tool_delta_after_reasoning_s", t) if high else t
            if lat is None or lat > FIRST_DELTA_BAR_S:
                why.append("first tool_calls delta %.2f s %s > %.1f s" % (lat, "after the reasoning ended" if high else "after the request", FIRST_DELTA_BAR_S))
        if not c.get("all_args_equal"):
            why.append("streamed arguments != non-stream arguments")
        if c.get("calls_stream") != 1 or c.get("calls_nonstream") != 1:
            why.append("calls stream/non-stream = %s/%s, want 1/1" % (c.get("calls_stream"), c.get("calls_nonstream")))
        if any(not x.get("args_valid_json") for x in s.get("calls", [])):
            why.append("streamed arguments not valid JSON")
        return ("PASS" if not why else "FAIL"), why

    if ph.startswith("fix"):
        for case, high in (("L1_en_off", False), ("L2_nonascii_off", False), ("L3_code_off", False), ("L5_csv_off", False), ("L4_nonascii_high", True), ("L6_squares_high", True), ("M1_powers_off", False), ("M2_getallen_off", False), ("M3_labels_off", False)):
            if case in res:
                v[case] = ok_long(case, high)
        for case, want in (("S1_short_off", "get_weather"), ("S2_noarg_off", "get_server_time")):
            r = g(case)
            c, s = r.get("compare") or {}, r.get("stream") or {}
            why = []
            if c.get("names_stream") != [want] or c.get("names_nonstream") != [want]:
                why.append("names stream %s / non-stream %s, want [%s]" % (c.get("names_stream"), c.get("names_nonstream"), want))
            if not c.get("all_args_equal"):
                why.append("arguments differ")
            if any(not x.get("args_valid_json") for x in s.get("calls", [])):
                why.append("streamed arguments not valid JSON")
            if case in res:
                v[case] = ("PASS" if not why else "FAIL", why)
        r = g("U1_undeclared_off")
        if r:
            s, n = r.get("stream") or {}, r.get("nonstream") or {}
            why = []
            if s.get("tool_chunks"):
                why.append("the stream carried %d tool_calls chunk(s) (names %s)" % (s["tool_chunks"], [x.get("names") for x in s.get("calls", [])]))
            if n.get("calls"):
                why.append("the non-stream response carried tool_calls %s" % [x["name"] for x in n["calls"]])
            v["U1_undeclared_off"] = ("PASS" if not why else "FAIL", why)
        r = g("C1_cfg_off")
        if r:
            s, c = r.get("stream") or {}, r.get("compare") or {}
            why = []
            for x in s.get("calls", []):
                if x["router_events"]:
                    why.append("router replay events on index %s: %s" % (x["index"], x["router_events"]))
                if not x["router_replay_equal"]:
                    why.append("router replay text != raw concatenation on index %s" % x["index"])
            if not c.get("all_args_equal"):
                why.append("streamed arguments != non-stream arguments")
            if c.get("names_stream") != ["cfg"]:
                why.append("stream names %s, want ['cfg']" % c.get("names_stream"))
            v["C1_cfg_off"] = ("PASS" if not why else "FAIL", why)
        for case in ("P1_pl_auto_off", "P2_pl_required_off"):
            r = g(case)
            if not r:
                continue
            s, n, c = r.get("stream") or {}, r.get("nonstream") or {}, r.get("compare") or {}
            why = []
            if c.get("names_stream") != ["zapisz_notatke"] or c.get("names_nonstream") != ["zapisz_notatke"]:
                why.append("names stream %s / non-stream %s" % (c.get("names_stream"), c.get("names_nonstream")))
            if not c.get("all_args_equal"):
                why.append("streamed arguments != non-stream arguments")
            for label, args in (("stream", [x.get("arg_keys") for x in s.get("calls", [])]),
                                ("non-stream", [x.get("arg_keys") for x in n.get("calls", [])])):
                for k in args:
                    if k != ["tytuł", "treść"]:
                        why.append("%s argument keys %s, want ['tytuł', 'treść'] whole" % (label, k))
            v[case] = ("PASS" if not why else "FAIL", why)
        rr = [x for case in res for x in ((res[case].get("stream") or {}).get("calls") or []) if x["router_events"]]
        v["router_replay_all_cases"] = ("PASS" if not rr else "FAIL", ["%d call(s) with router replay events" % len(rr)] if rr else [])
    return {k: {"verdict": a, "why": b} for k, (a, b) in v.items()}


# =====================================================================================================================
# The DSML gate, graded INSIDE each request (the user's definition): re-parse each request's own raw output in one
# shot with the served tree's parser and compare it, byte for byte, with what that request returned.
# =====================================================================================================================
DETECTOR_PY = os.environ.get("DSML_DETECTOR_PY") or os.path.expanduser("<HOME>/workspace/sglang/.venv/bin/python")
SGLANG_REPO = os.path.expanduser("<HOME>/workspace/sglang")
DETECTOR_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "dsml-detector")
MECH_MIN_ARG_BYTES = 12000   # #2539's bar: a mechanical long call is >= 12000 argument bytes
PL_KEYS = ["tytuł", "treść"]
SHIM_PACKAGES = ("sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai",
                 "sglang.srt.function_call", "sglang.srt.parser")


def detector_src_for_tree(sha, repo=SGLANG_REPO):
    """The served tree's python/ dir for commit <sha>: `git archive <sha> python/sglang` from <repo>, extracted once
    under DETECTOR_CACHE/sglang-<sha>/ (read-only on the repo)."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise SystemExit("--detector-tree takes the served tree's full 40-hex commit, got %r" % sha)
    dest = os.path.join(DETECTOR_CACHE, "sglang-" + sha)
    py = os.path.join(dest, "python")
    if os.path.isfile(os.path.join(py, "sglang", "srt", "function_call", "deepseekv32_detector.py")):
        return py
    if subprocess.run(["git", "-C", repo, "cat-file", "-e", sha + "^{commit}"], stderr=subprocess.DEVNULL).returncode:
        raise SystemExit("commit %s is not in %s: fetch it first (the fork remote stage.sh uses)" % (sha, repo))
    os.makedirs(DETECTOR_CACHE, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".x-" + sha[:12] + "-", dir=DETECTOR_CACHE)
    git = subprocess.Popen(["git", "-C", repo, "archive", sha, "python/sglang"], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", tmp], stdin=git.stdout)
    git.stdout.close()
    if git.wait() or tar.returncode:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit("git archive %s python/sglang from %s failed" % (sha, repo))
    try:
        os.rename(tmp, dest)
    except OSError:   # a concurrent extraction won; its copy is the same commit
        shutil.rmtree(tmp, ignore_errors=True)
    return py


def tree_kind(src):
    """Which output path the served tree's python/ dir <src> serves DeepSeek V4.1 through: "recipe" (box-e on: the
    chat-encoding adapter renders with the deepseek_recipe package and parses the sampled ids with
    parser/deepseek_recipe_output.py) or "encoder" (box-d and before: entrypoints/openai/encoding_dsv41.py renders,
    the reasoning parser + FunctionCallParser's DSML detector parse the text)."""
    srt = os.path.join(src, "sglang", "srt")
    if os.path.isfile(os.path.join(srt, "parser", "deepseek_recipe_output.py")):
        return "recipe"
    if os.path.isfile(os.path.join(srt, "entrypoints", "openai", "encoding_dsv41.py")):
        return "encoder"
    raise SystemExit("%s holds neither sglang/srt/parser/deepseek_recipe_output.py (box-e recipe) nor "
                     "sglang/srt/entrypoints/openai/encoding_dsv41.py (box-d encoder): not a served tree's python/ dir" % src)

def parse_worker(src, jobs_path, out_path, tokenizer=None):
    """Runs under the detector python (sglang's dependencies), from the served tree's python/ dir <src>. A recipe tree
    (tree_kind) runs recipe_parse instead, with the checkpoint's tokenizer.json <tokenizer>.

    The package __init__ files are bypassed (sglang/__init__.py imports triton, CUDA helpers and notebook utilities a
    parser never calls): each package on the import path is registered as a bare namespace over its own directory, so
    the parser modules themselves are the tree's files, unmodified. Where IPython is absent (sglang.utils imports
    IPython.display for notebook helpers at module level), an empty IPython.display stands in; nothing on the parse
    path calls it. Jobs:
      parse   {id, text, tools, tool_choice, thinking, tool_parser, reasoning_parser}: the engine's non-stream path --
              a trailing EOS dropped (the detokenizer trims the matched stop token from the engine's own text), the
              reasoning parser's parse_non_stream (force_reasoning = thinking), then _process_tool_calls' rule:
              FunctionCallParser(tools, tool_parser, tool_choice).parse_non_stream when the detector owns the format
              (or tool_choice is not required) and has_tool_call(text); else, for a required choice, the JSON path.
      render  {id, calls, reasoning, content}: the assistant turn as encoding_dsv41 renders it (the text the model is
              trained to emit), for the self-test.
      decode  {id, ids}: decode_jobs, on either tree kind.
    """
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import types
    sys.dont_write_bytecode = True   # -I ignores PYTHONDONTWRITEBYTECODE; never leave __pycache__ in a served tree
    sys.path.insert(0, src)
    for pkg in SHIM_PACKAGES:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(src, *pkg.split("."))]
        sys.modules[pkg] = m
    if importlib.util.find_spec("IPython") is None:
        class _Empty(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path, target=None):
                if name == "IPython" or name.startswith("IPython."):
                    return importlib.machinery.ModuleSpec(name, self, is_package=True)
                return None

            def create_module(self, spec):
                m = types.ModuleType(spec.name)
                m.__path__ = []
                m.__getattr__ = lambda a: type(a, (), {})
                return m

            def exec_module(self, m):
                pass
        sys.meta_path.append(_Empty())
    jobs = json.load(open(jobs_path, encoding="utf-8"))
    decoded = decode_jobs([j for j in jobs if j["op"] == "decode"], tokenizer)
    jobs = [j for j in jobs if j["op"] != "decode"]
    if tree_kind(src) == "recipe":
        out, info = recipe_parse(jobs, tokenizer)
        json.dump(dict(info, src=src, kind="recipe", results=out + decoded), open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False)
        return
    from sglang.srt.entrypoints.openai import encoding_dsv41 as enc
    from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
    from sglang.srt.function_call import deepseekv32_detector
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    out = []
    for j in jobs:
        try:
            if j["op"] == "render":
                tcl = [enc.tool_call_template.format(dsml_token=enc.dsml_token, tool_call_tag_name=enc.tool_call_tag_name,
                                                     name=c["name"], arguments=enc.encode_arguments_to_dsml(c))
                       for c in j["calls"]]
                tc = ("\n\n" + enc.tool_calls_template.format(dsml_token=enc.dsml_token, tool_calls="\n".join(tcl),
                                                              tc_block_name=enc.tool_calls_block_name)) if tcl else ""
                think = (j["reasoning"] + enc.thinking_end_token) if j.get("reasoning") is not None else ""
                out.append({"id": j["id"], "text": think + (j.get("content") or "") + tc + enc.eos_token})
                continue
            text, eos = j["text"], False
            if text.endswith(enc.eos_token):
                text, eos = text[: -len(enc.eos_token)], True
            rp = ReasoningParser(model_type=j["reasoning_parser"], stream_reasoning=False,
                                 force_reasoning=bool(j["thinking"]), tool_call_parser_active=True)
            reasoning, body = rp.parse_non_stream(text)
            tools = [Tool(**t) for t in j["tools"]]
            choice = j.get("tool_choice")
            if isinstance(choice, dict):
                choice = ToolChoice(**choice)
            required = choice == "required" or isinstance(choice, ToolChoice)
            p = FunctionCallParser(tools, j["tool_parser"], tool_choice=choice)
            owns = p.owns_tool_format()
            path, calls = "none", []
            if (not required or owns) and p.has_tool_call(body):
                path = "detector"
                _, items = p.parse_non_stream(body)
                calls = [{"name": c.name, "arguments": c.parameters} for c in items]
            elif required and not owns:
                path = "json"
                data = json.loads(body)
                data = [data] if isinstance(data, dict) else data
                calls = [{"name": c["name"], "arguments": json.dumps(c.get("parameters", {}), ensure_ascii=False)}
                         for c in data]
            out.append({"id": j["id"], "calls": calls, "path": path, "eos_stripped": eos,
                        "reasoning_chars": len(reasoning or ""), "body_chars": len(body or ""),
                        "detector": type(p.detector).__name__})
        except Exception as e:  # reported per job; the grade reads it as a FAIL
            out.append({"id": j["id"], "error": "%s: %s" % (type(e).__name__, e)})
    json.dump({"src": src, "kind": "encoder", "detector_file": deepseekv32_detector.__file__, "results": out + decoded},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


def decode_jobs(jobs, tokenizer):
    """decode {id, ids} jobs: the sampled ids decoded by the checkpoint's own tokenizer.json <tokenizer> (special tokens
    kept, as serving_chat decodes a tool-bearing request) -- a REPEATS case's raw text read from the ids themselves,
    independent of the engine's /detokenize. Each job answers {id, text} or {id, error}."""
    if not jobs:
        return []
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(tokenizer, "tokenizer.json") if os.path.isdir(tokenizer) else tokenizer)
    except Exception as e:
        return [{"id": j["id"], "error": "tokenizer %s: %s: %s" % (tokenizer, type(e).__name__, e)} for j in jobs]
    out = []
    for j in jobs:
        try:
            out.append({"id": j["id"], "text": tok.decode(j["ids"], skip_special_tokens=False)})
        except Exception as e:
            out.append({"id": j["id"], "error": "%s: %s" % (type(e).__name__, e)})
    return out


# DeepSeek V4.1's DSML spelling: encoding_dsv41's constants on box-d, deepseek_recipe's DeepseekV41Encoding on box-e
DSML_TOKEN, DSML_CALLS, DSML_INVOKE, DSML_PARAMETER = "｜DSML｜", " calls", " invoke", " parameter"
THINK_END, EOS_TOKEN = "</think>", "<｜end▁of▁sentence｜>"
# chat_encoding_adapters.ADAPTER_OUTPUT_PARSERS: the tool parser name -> the recipe chat encoding that owns its output
RECIPE_SPECS = {"deepseekv41": "dsv41", "deepseekv4": "dsv4"}

def render_dsml(calls, reasoning, content):
    """An assistant turn as encoding_dsv41 renders it (assistant_msg_template: reasoning + </think>, content, the calls
    block -- tool_calls_template over tool_call_template over encode_arguments_to_dsml -- and EOS): the text the model
    is trained to emit. A recipe tree has no Python encoder for one turn; this is the same template, for the self-test."""
    def arg(k, v):
        s = isinstance(v, str)
        return '<%s%s name="%s" string="%s">%s</%s%s>' % (DSML_TOKEN, DSML_PARAMETER, k, "true" if s else "false",
                                                         v if s else json.dumps(v, ensure_ascii=False), DSML_TOKEN, DSML_PARAMETER)
    tcl = []
    for c in calls:
        a = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]
        if not isinstance(a, dict):
            raise ValueError("Assistant tool call function.arguments must be a JSON object.")
        tcl.append('<%s%s name="%s">\n%s\n</%s%s>' % (DSML_TOKEN, DSML_INVOKE, c["name"], "\n".join(arg(k, v) for k, v in a.items()),
                                                     DSML_TOKEN, DSML_INVOKE))
    tc = ("\n\n<%s%s>\n%s\n</%s%s>" % (DSML_TOKEN, DSML_CALLS, "\n".join(tcl), DSML_TOKEN, DSML_CALLS)) if tcl else ""
    return ((reasoning + THINK_END) if reasoning is not None else "") + (content or "") + tc + EOS_TOKEN

def recipe_parse(jobs, tokenizer):
    """The box-e tree's non-stream output path, under the detector python: serving_deepseek_recipe's
    DeepSeekRecipeChatEncodingAdapter.parse_chat_response -- DeepSeekRecipe(<spec>, the checkpoint's tokenizer), then
    DeepSeekRecipeOutputParser(thinking_stage = parse_start(thinking), starts_in_tool_calls = prefills_tool_calls,
    parse_tool_calls) .finish(the sampled ids, the matched EOS) and whole_tool_calls. The ids are the request's own
    <mode>.output_ids.json when they decode, under <tokenizer>, to exactly its raw text; else the raw text encoded as
    parse_output_text does (encode_text). A render job is render_dsml. Returns (results, worker info)."""
    import deepseek_recipe
    from tokenizers import Tokenizer
    from sglang.srt.parser import deepseek_recipe_output as dro
    from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe
    parse = [j for j in jobs if j["op"] == "parse"]
    tok, recipes, out = None, {}, []
    if parse:
        if not tokenizer:
            raise SystemExit("a recipe tree parses token ids: name the checkpoint's tokenizer.json with --tokenizer "
                             "(or $DSML_TOKENIZER; on a box, $MODEL_DIR/tokenizer.json)")
        if os.path.isdir(tokenizer):
            tokenizer = os.path.join(tokenizer, "tokenizer.json")
        tok = Tokenizer.from_file(tokenizer)
    for j in jobs:
        try:
            if j["op"] == "render":
                out.append({"id": j["id"], "text": render_dsml(j["calls"], j.get("reasoning"), j.get("content"))})
                continue
            spec = RECIPE_SPECS.get(j["tool_parser"])
            if spec is None:
                raise ValueError("tool parser %r has no recipe chat encoding (%s)" % (j["tool_parser"], ", ".join(RECIPE_SPECS)))
            if spec not in recipes:
                recipes[spec] = DeepSeekRecipe(spec, tok)
            dr = recipes[spec]
            text, ids, source = j["text"], j.get("ids"), "output_ids"
            if not ids or tok.decode(ids, skip_special_tokens=False) != text:
                ids, source = dr.encode_text(text), "text"
            thinking = bool(j["thinking"])
            choice = j.get("tool_choice")
            # serving_deepseek_recipe.prefills_tool_calls: a forced choice ends a dsv41 chat prompt inside the calls block
            starts_in = (spec == "dsv41" and not thinking and bool(j["tools"]) and (choice == "required" or isinstance(choice, dict))
                         and j.get("last_role") != "assistant")
            eos = tok.token_to_id(EOS_TOKEN)
            eos_hit = bool(ids) and ids[-1] == eos
            p = dro.DeepSeekRecipeOutputParser(dr, thinking_stage=dr.parse_start(thinking=thinking), parse_tool_calls=True,
                                               starts_in_tool_calls=starts_in)
            d = p.finish(ids, {"type": "stop", "matched": eos} if eos_hit else {"type": "length"})
            out.append({"id": j["id"], "calls": [{"name": c.name, "arguments": c.arguments} for c in dro.whole_tool_calls(d)],
                        "path": "recipe", "eos_stripped": eos_hit, "reasoning_chars": len(d.reasoning or ""),
                        "body_chars": len(d.content or ""), "detector": "DeepSeekRecipeOutputParser(%s)" % spec,
                        "input": source, "starts_in_tool_calls": starts_in})
        except Exception as e:  # reported per job; the grade reads it as a FAIL
            out.append({"id": j["id"], "error": "%s: %s" % (type(e).__name__, e)})
    return out, {"detector_file": dro.__file__, "tokenizer": tokenizer,
                 "recipe": {"version": getattr(deepseek_recipe, "__version__", None), "file": deepseek_recipe.__file__}}

_INVOKE_RE = re.compile(r"<%s%s\s+name=\"([^\"]+)\"\s*>(.*?)</%s%s>" % (DSML_TOKEN, DSML_INVOKE, DSML_TOKEN, DSML_INVOKE), re.S)
_PARAM_RE = re.compile(r"<%s%s\s+name=\"([^\"]+)\"\s+string=\"([^\"]+)\"\s*>(.*?)</%s%s>" % (
    DSML_TOKEN, DSML_PARAMETER, DSML_TOKEN, DSML_PARAMETER), re.S)

def emitted_calls(text, thinking, tools, strip=True):
    """THE MODEL'S OWN CALLS, read from its raw output with the DSML grammar the encoder renders (stdlib, independent
    of any served parser): the reasoning (thinking on) ends at </think> or at the spaced calls tag; the calls block
    opens at <｜DSML｜ calls>, or at the first invoke when the prompt prefilled the opener (a forced choice); each closed
    invoke of a declared tool is one call, each closed parameter tag one argument. Returns [{name, params: [(key,
    string?, raw value)]}] -- the reference the served arguments must equal. strip=False keeps each parameter name as
    written (todo_verdict grades a padded one)."""
    if text.endswith(EOS_TOKEN):
        text = text[: -len(EOS_TOKEN)]
    opener = "<%s%s>" % (DSML_TOKEN, DSML_CALLS)
    invoke = "<%s%s" % (DSML_TOKEN, DSML_INVOKE)
    if thinking:
        k = text.find(THINK_END)
        text = text[k + len(THINK_END):] if k >= 0 else text[text.find(opener):] if opener in text else ""
    k = text.find(opener)
    if k >= 0:
        block = text[k + len(opener):]
    elif text.lstrip().startswith(invoke):
        block = text
    else:
        return []
    names = {(t.get("function") or {}).get("name") for t in tools or []}
    return [{"name": m.group(1).strip(), "params": [(p.group(1).strip() if strip else p.group(1), p.group(2) == "true", p.group(3))
                                                  for p in _PARAM_RE.finditer(m.group(2))]}
            for m in _INVOKE_RE.finditer(block) if m.group(1).strip() in names]

def emitted_diffs(emitted, calls):
    """Where the returned calls are not the model's emitted ones: names in order, then per call its arguments JSON --
    a JSON object whose keys are the emitted parameter names, in order, and whose values are the emitted values (a
    string parameter as written, or stripped as the DSML detector strips it; any other as JSON, or its stripped text
    where it is not JSON, the detector's fallback)."""
    if [c["name"] for c in emitted] != [c["name"] for c in calls]:
        return ["names: emitted %r, returned %r" % ([c["name"] for c in emitted], [c["name"] for c in calls])]
    out = []
    for i, (e, c) in enumerate(zip(emitted, calls)):
        want = [k for k, _, _ in e["params"]]
        try:
            got = json.loads(c["arguments"] or "")
        except ValueError as x:
            out.append("call %d %s: returned arguments are not JSON (%s): %r; the model emitted keys %r" % (
                i, c["name"], x, (c["arguments"] or "")[:120], want))
            continue
        if not isinstance(got, dict) or list(got) != want:
            out.append("call %d %s: returned keys %r, the model emitted %r" % (
                i, c["name"], list(got) if isinstance(got, dict) else type(got).__name__, want))
            continue
        for k, is_str, raw in e["params"]:
            v = got[k]
            if is_str:
                ok = isinstance(v, str) and v in (raw, raw.strip())
            else:
                try:
                    ok = json.dumps(v, sort_keys=True) == json.dumps(json.loads(raw.strip()), sort_keys=True)
                except ValueError:
                    ok = v == raw.strip()
            if not ok:
                out.append("call %d %s: %r is %r, the model emitted %r" % (i, c["name"], k, v if len(repr(v)) < 120 else repr(v)[:120], raw[:120]))
    return out

def run_worker(jobs, src, py=None, tokenizer=None):
    """Run parse_worker over <jobs> under the detector python; returns ({id: result}, worker info)."""
    py = py or DETECTOR_PY
    if not jobs:
        return {}, {}
    td = tempfile.mkdtemp(prefix="dsml-worker-")
    try:
        jp, op = os.path.join(td, "jobs.json"), os.path.join(td, "out.json")
        json.dump(jobs, open(jp, "w", encoding="utf-8"), ensure_ascii=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
        r = subprocess.run([py, "-I", os.path.abspath(__file__), "--parse-worker", src, jp, op] + ([tokenizer] if tokenizer else []),
                           env=env, cwd=td,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode or not os.path.isfile(op):
            raise SystemExit("detector worker failed (exit %s, %s): %s" % (
                r.returncode, py, r.stderr.decode("utf-8", "replace")[-1500:]))
        o = json.load(open(op, encoding="utf-8"))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    info = {k: o[k] for k in ("src", "kind", "detector_file", "tokenizer", "recipe") if k in o}
    return {x["id"]: x for x in o["results"]}, dict(info, python=py)


def load_raw(d, mode):
    """The request's own raw output text: probe.py's <mode>.raw.txt, or #2539 inreq.py's <mode>.detok.txt."""
    for name in (mode + ".raw.txt", mode + ".detok.txt"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return open(p, encoding="utf-8", newline="").read(), name
    return None, None


def returned_calls(d, mode):
    """What the request returned: (calls [{name, arguments}], stream analysis or None, non-stream analysis or None)."""
    if mode == "stream":
        s = parse_stream(d)
        if s["t_first_tool_delta"] is not None and s["t_last_reasoning"] is not None:
            s["first_tool_delta_after_reasoning_s"] = round(s["t_first_tool_delta"] - s["t_last_reasoning"], 3)
        calls = [{"name": e["names"][0] if e["names"] else None, "arguments": "".join(e["fragments"]),
                  "fragments": e["fragments"]} for _, e in sorted(s["calls"].items())]
        return calls, s, None
    n = parse_nonstream(d)
    if n is None or "calls" not in n:
        return None, None, n
    return [{"name": c["name"], "arguments": c["arguments"] or ""} for c in n["calls"]], None, n


def first_diff(a, b):
    return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))


def request_of(d):
    p = os.path.join(d, "request.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def thinking_of(req):
    """serving_chat's force_reasoning for the request: off only on an explicit reasoning_effort none (box-c thinks at
    high on a request with no thinking field)."""
    e = req.get("reasoning_effort")
    if e is None:
        e = (req.get("chat_template_kwargs") or {}).get("reasoning_effort")
    return e != "none"


def declared(req):
    out = {}
    for t in req.get("tools") or []:
        fn = t.get("function") or {}
        out[fn.get("name")] = list(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return out


def modes_in(d):
    return [m for m in ("stream", "nonstream") if (sse_path(d, m) if m == "stream" else
            (os.path.isfile(os.path.join(d, "nonstream.json")) or os.path.isfile(os.path.join(d, "nonstream.raw"))))]


def send_dirs(d):
    """A REPEATS case's send dirs (probe.py's <case>/s<i>/, in order; <case>/repeat.json marks the case), or None for a
    case sent once per mode."""
    if not os.path.isfile(os.path.join(d, "repeat.json")):
        return None
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))

def send_items(ph, case, d):
    """(send dir, mode, worker job id, send name or None) for every recorded send of the case at <d>."""
    subs = send_dirs(d)
    if subs is None:
        return [(d, m, "%s|%s|%s" % (ph, case, m), None) for m in modes_in(d)]
    return [(os.path.join(d, s), m, "%s|%s/%s|%s" % (ph, case, s, m), s) for s in subs for m in modes_in(os.path.join(d, s))]

def parse_jobs(phases, tool_parser, reasoning_parser, decode=False):
    """The worker's jobs: a one-shot parse of every send's raw text, and (<decode>: a --tokenizer was given) a decode of
    each REPEATS send's output ids."""
    jobs = []
    for ph in phases:
        for case in sorted(os.listdir(ph)):
            d = os.path.join(ph, case)
            req = request_of(d) if os.path.isdir(d) else None
            if req is None:
                continue
            for d, mode, jid, _ in send_items(ph, case, d):
                ip = os.path.join(d, mode + ".output_ids.json")
                ids = json.load(open(ip)) if os.path.isfile(ip) else None
                if decode and ids is not None and send_dirs(os.path.dirname(d)) is not None:
                    jobs.append({"id": "decode|" + jid, "op": "decode", "ids": ids})
                text, _ = load_raw(d, mode)
                if text is not None:
                    jobs.append({"id": jid, "op": "parse", "text": text, "ids": ids,
                                 "tools": req.get("tools") or [], "tool_choice": req.get("tool_choice"),
                                 "last_role": ((req.get("messages") or [{}])[-1] or {}).get("role"),
                                 "thinking": thinking_of(req), "tool_parser": tool_parser,
                                 "reasoning_parser": reasoning_parser})
    return jobs


def case_bars(case, mode, req, calls, s):
    """The case's own bars, each read inside this one request. Returns (why, info)."""
    why, info = [], []
    names = [c["name"] for c in calls]
    keys = []
    for c in calls:
        try:
            o = json.loads(c["arguments"] or "")
            keys.append(list(o.keys()) if isinstance(o, dict) else None)
        except ValueError:
            keys.append(None)
    choice = req.get("tool_choice")
    if choice == "required" and not calls:
        why.append("no tool call under tool_choice required")
    if case[:1] in ("M", "L"):
        if names != ["write_file"]:
            why.append("calls %r, want one write_file" % names)
        if case[:1] == "M" and calls and len(calls[0]["arguments"].encode("utf-8")) < MECH_MIN_ARG_BYTES:
            why.append("args %d bytes < %d: not a long call" % (len(calls[0]["arguments"].encode("utf-8")), MECH_MIN_ARG_BYTES))
        if s is not None:
            if not s.get("timed"):
                info.append("untimed transcript: first-delta bar not read")
            else:
                hi = thinking_of(req)
                t = s.get("first_tool_delta_after_reasoning_s") if hi else s.get("t_first_tool_delta")
                if hi and t is None:
                    t = s.get("t_first_tool_delta")
                if t is None:
                    why.append("no tool_calls delta in the stream")
                elif t > FIRST_DELTA_BAR_S:
                    why.append("first tool_calls delta %.2f s %s > %.1f s" % (
                        t, "after the reasoning ended" if hi else "after the request", FIRST_DELTA_BAR_S))
    elif case.startswith("P1") or case.startswith("P2"):
        if case.startswith("P2") and names != ["zapisz_notatke"]:
            why.append("calls %r, want one zapisz_notatke" % names)
        for k in keys:
            if k != PL_KEYS:
                why.append("argument keys %r, want exactly %r" % (k, PL_KEYS))
    elif case.startswith("S1"):
        if names != ["get_weather"]:
            why.append("calls %r, want [get_weather]" % names)
    elif case.startswith("S2"):
        if names != ["get_server_time"] or [c["arguments"] for c in calls] != ["{}"]:
            why.append("calls %r args %r, want get_server_time {}" % (names, [c["arguments"] for c in calls]))
    elif case.startswith("C1"):
        if names != ["cfg"]:
            why.append("calls %r, want [cfg]" % names)
    elif case.startswith("U1"):
        info.append("undeclared-tool case: recorded, not graded beyond the self-parse (calls %r)" % names)
    elif case.startswith("N"):   # #2539 inreq.py's declared-name cases: one tool, one declared parameter
        want = [list(v) for v in declared(req).values()]
        for k in keys:
            if [k] != want:
                why.append("argument keys %r, want exactly the declared %r" % (k, want[0] if want else None))
    dec = declared(req)
    for c, k in zip(calls, keys):
        if k is None:
            why.append("%s arguments are not a JSON object" % c["name"])
        elif c["name"] in dec and not case.startswith("U1"):
            extra = [x for x in k if x not in dec[c["name"]]]
            if extra:
                why.append("%s argument key(s) %r not declared" % (c["name"], extra))
    return why, info


# The REPEATS gate (probe.py's T1_todo_nested: Cosmos's real todo_write, a nested `todos` array, sent REPEATS = 470
# times per mode at T 1.0 / top_p 1.0 and graded on n >= 369 sends, TODO_MIN_SENDS). A defect at a few percent of
# calls -- prod's `"id: 20, "` entry key, box-e's `{"first: "abc123"}` -- is invisible to one T=0 send; graded by rate
# instead, per mode, on both sides of the parser.
#
# The bar is a binomial count whose null is the probe's OWN healthy arms (#1152, run review 32), and it is a
# CATASTROPHE gate: it fails a tree whose defect rate is several times the null's (a 5 % defect, #902's 28.2 %, a
# no-#901 tree's ~99 %), not one modestly above it -- a 2 % defect (2.7x p0) passes one mode of 470 calls 65.8 % of
# the time and both modes 43.4 %, and a mode row near the bar (9/470: binomial p 0.008 at p0) is evidence to chase,
# not a pass to cite. The first calibration (#931) took prod's live rate before the box-d -> box-e roll, 0.32 % of
# todo_write calls, as a healthy tree's; but this probe's case reads a healthy tree at ~0.73 % (TODO_HEALTHY_ARMS
# below: 14 mode rows of 470..472 calls on trees that carry fork #901, 48 defective of 6588 calls), and at that rate
# the old k <= 6 of 470 false-fails 5.89 % of mode rows (21.6 % of a four-row run). #1113 W2's config of record (v10 +
# env-box-e-w3: 9/470, 7/471 among its four rows) is NOT in the null: those are the runs under question, and pooling
# them into the bar they are graded by is circular (#1195, which also took the null from 0.84 % to 0.73 % and the bar
# at 470 from 11 to 10). So p0 = the pooled healthy rate, and a check (one count of one mode: raw or parsed) fails
# when its count is above todo_max_bad(its calls) = the least k with P(X > k | calls, p0) <= 0.1 % (TODO_BAR). The
# third count, the sends whose two sides differ (raw-vs-parsed), is held to the same rule at its OWN null,
# TODO_DIFFER_BAR (below: a healthy parser serves what the model wrote, 1 differing send of 7520), so a parser-only
# defect fails at 3 of 470 instead of hiding under the defect-rate bar of 10. A case carries 6 such checks (2 modes x
# 3), so it false-fails <= 0.6 % (union bound; ~0.33 % in fact at 470, the differ checks spending 0.004 % each), a
# two-arm run's 12 checks <= 1.2 %. The gate must catch a 5 % defect (TODO_CATCH_RATE) with >= 99 % power across
# both modes; a 2.5 % one it catches 86 % of the time at 470 calls (63 % in one mode) -- 99 % at 2.5 % would need
# ~849 calls per mode under this null, at 2 % ~1406 -- and a real defect (#902's 28.2 %, a no-#901 tree's ~99 %)
# fails with certainty. Exact sums (todo_binom_sf), p0 = 48/6588:
#   n per mode  max bad   false-fail per check at p0   power at 2.5 %: one mode / both   at 5 %, both modes   differ k (ff)
#   470 (#931)  6         5.89 %                          --                            --                    (6)  <- the old bar
#   200         6         0.073 %                      23.59 % / 41.62 %                  98.47 %              1 (0.035 %)
#   333         8         0.088 %                      45.33 % / 70.11 %                  99.98 %              1 (0.095 %)  <- 90 % call floor of 369
#   369         9         0.045 %                      44.26 % / 68.93 %                  99.99 %              2 (0.002 %)  <- TODO_MIN_SENDS
#   400         9         0.084 %                      54.37 % / 79.18 %                  99.998 %             2 (0.002 %)
#   423         10        0.034 %                      48.94 % / 73.93 %                  99.998 %             2 (0.003 %)  <- 90 % call floor of 470
#   470         10        0.081 %                      62.84 % / 86.19 %                  99.9999 %            2 (0.004 %)  <- probe.py's REPEATS
#   940         16        0.072 %                      93.43 % / 99.57 %                  ~100 %               2 (0.030 %)
# (differ k: todo_max_bad(n, TODO_DIFFER_RATE), its false-fail at that null; its power at 2.5 % is >= 99.8 % from
# n 200 and every n >= 77 reaches 99 % at 5 %, so the differ count is never the one too short to grade.)
# The bar is computed from the calls each count ACTUALLY carries, not from the sends asked for: todo_max_bad(n) is
# the least k that keeps the per-check budget (the k with the most power), and todo_power(n) the power it buys at
# TODO_CATCH_RATE. A mode reaching TODO_POWER at its own n misses a 5 % defect at most sqrt(1 %) = 10 % of the time,
# so two such modes keep the case at >= 99 % whatever their n (1 - m_a m_b >= 1 - max(m_a, m_b)^2). Every n >= 234
# reaches it (and 209, 210; the 369-send floor's 333 calls: k 8, 99.98 %); a count short of that FAILs as too few calls
# for the gate (todo_short), naming n, k, the power and the least n at or above it that reaches 99 %.
# Beside each count's verdict the row prints its one-sided binomial p, P(X >= k | n, p0), and its one-sided Fisher p
# against the pooled healthy arms (todo_fisher_p: the hypergeometric tail of an arm at least this much worse than
# 48/6588) -- evidence, not the verdict: the bar alone decides.
#
# The healthy arms the null pools: (record, tree, arm, mode, defective calls, calls) -- every recorded arm on a tree
# carrying #901 (raw = parsed defective counts in every one) EXCEPT a config under question: #1113 W2 (v10 +
# env-box-e-w3 on 5f117c89, stream 9/470, nonstream 4/471, strict:false stream 7/471, nonstream 3/470) is graded BY
# the bar, so it is not in it. The pool DOES include the candidate build's own passing arms: #756 mmrefs and #1113 W3
# ran on bdc3f67f, the cached-refs tree that was the config of record until prod 9b262265 carried it (#1161; copy a
# serves it until #1320); they stay as passing arms on the build
# under test, not the disputed FAIL (the manager's ruling on #1195). One pooled row FAILed the old k <= 6 (#756
# mmrefs nonstream 7/472); all PASS here. The arms on a tree WITHOUT #901 are the defect, not the null, and are left
# out (#750 copy a 070154af: stream 455/456, nonstream 458/458 raw). Re-graded under this bar (k 10 of 470..472):
# #1113 W2 9/470 (binomial p 0.0084, Fisher p 0.012), 4/471, 7/471 (0.060, 0.071), 3/470 PASS; #1151 copy a on
# bdc3f67f (v10 + cached refs) 5/470, 5/471, 4/472, 4/471 (binomial p 0.26..0.45) PASS. W2's stream 9/470 is the one
# row a 2 % defect would plausibly produce: the bar cannot tell it from noise, which is what a catastrophe gate means.
TODO_HEALTHY_ARMS = (
    # #750 copy b post-roll, tree 5f117c89 (.tmp/box-c-roll-20261029/dsml-b-post.out)
    ("#750 b", "5f117c89", "T1_todo_nested", "stream", 2, 470),
    ("#750 b", "5f117c89", "T1_todo_nested", "nonstream", 3, 470),
    # #756 mm-cached-refs tree (.tmp/box-c-mmrefs-756/notes.md 07:35Z) and its restored anchor (dsml-base.out)
    ("#756 mmrefs", "bdc3f67f", "T1_todo_nested", "stream", 3, 471),
    ("#756 mmrefs", "bdc3f67f", "T1_todo_nested", "nonstream", 7, 472),
    ("#756 anchor", "5f117c89", "T1_todo_nested", "stream", 3, 471),
    ("#756 anchor", "5f117c89", "T1_todo_nested", "nonstream", 4, 472),
    # #1113 W1 old args/env, W3 mm-cached-refs (.tmp/box-c-mmrefs-1113/w<i>-dsml-grade.txt); W2, the config of record
    # (v10 + env-box-e-w3), is the run under question and stays out (#1195)
    ("#1113 W1", "5f117c89", "T1_todo_nested", "stream", 2, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested", "nonstream", 4, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested_strictfalse", "stream", 2, 471),
    ("#1113 W1", "5f117c89", "T1_todo_nested_strictfalse", "nonstream", 4, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested", "stream", 2, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested", "nonstream", 4, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested_strictfalse", "stream", 3, 471),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested_strictfalse", "nonstream", 5, 470),
)
TODO_HEALTHY_BAD = sum(a[4] for a in TODO_HEALTHY_ARMS)      # 48
TODO_HEALTHY_CALLS = sum(a[5] for a in TODO_HEALTHY_ARMS)    # 6588
# THE BAR, a CATASTROPHE gate: (p0, the most one check may false-fail at p0). p0 = the pooled healthy arms' rate,
# 48 / 6588 = 0.729 %; k <= 10 of 470 calls, 9 of 369, 16 of 940 (todo_max_bad). It passes a 2 % defect 65.8 % of the
# time in one mode of 470 calls (43.4 % both modes): it catches a broken tree, not a modest regression.
# TODO_BLIND_RATE is that 2 %: every row prints the chance it would pass a defect at it.
TODO_BAR = (TODO_HEALTHY_BAD / TODO_HEALTHY_CALLS, 0.001)
TODO_BASE_RATE, TODO_CHECK_FALSE_FAIL = TODO_BAR
# The raw-vs-parsed count's own null: (record, tree, arm, mode, sends whose two sides differ, sends with a call) for
# every #901 arm whose grade recorded the count (the `differ=` field of its row, same sources as above). #756
# mmrefs's two rows are not in it: that grade (the box's /workspace/w756/dsml-b.grade.txt) stopped at SELF-PARSE
# under the wrong --detector-py, and its notes keep the defect counts only. #1113 W2's four rows stay in THIS null
# (#1195 took them out of the defect null only): their counts are all 0, so they make the differ bar stricter, never
# more lenient, and the differ bar's logic and values are unchanged.
TODO_HEALTHY_DIFFER = (
    ("#750 b", "5f117c89", "T1_todo_nested", "stream", 0, 470),
    ("#750 b", "5f117c89", "T1_todo_nested", "nonstream", 0, 470),
    ("#756 anchor", "5f117c89", "T1_todo_nested", "stream", 0, 470),
    ("#756 anchor", "5f117c89", "T1_todo_nested", "nonstream", 1, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested", "stream", 0, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested", "nonstream", 0, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested_strictfalse", "stream", 0, 470),
    ("#1113 W1", "5f117c89", "T1_todo_nested_strictfalse", "nonstream", 0, 470),
    ("#1113 W2", "5f117c89", "T1_todo_nested", "stream", 0, 470),
    ("#1113 W2", "5f117c89", "T1_todo_nested", "nonstream", 0, 470),
    ("#1113 W2", "5f117c89", "T1_todo_nested_strictfalse", "stream", 0, 470),
    ("#1113 W2", "5f117c89", "T1_todo_nested_strictfalse", "nonstream", 0, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested", "stream", 0, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested", "nonstream", 0, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested_strictfalse", "stream", 0, 470),
    ("#1113 W3", "bdc3f67f", "T1_todo_nested_strictfalse", "nonstream", 0, 470),
)
TODO_DIFFER_BAD = sum(a[4] for a in TODO_HEALTHY_DIFFER)     # 1
TODO_DIFFER_SENDS = sum(a[5] for a in TODO_HEALTHY_DIFFER)   # 7520
# THE DIFFER BAR: (its p0, 1 / 7520 = 0.013 %, the same per-check false-fail): k <= 2 of 470 or 369 sends, 1 of 333.
TODO_DIFFER_BAR = (TODO_DIFFER_BAD / TODO_DIFFER_SENDS, TODO_CHECK_FALSE_FAIL)
TODO_DIFFER_RATE = TODO_DIFFER_BAR[0]
TODO_CATCH_RATE = 0.05        # the defect rate the gate must catch with >= TODO_POWER, both modes
TODO_POWER = 0.99
TODO_SMALL_RATE = 0.025       # the smaller defect each row also prints its power at (both modes; not a target)
TODO_BLIND_RATE = 0.02        # the defect each row prints its PASS chance at (one mode): what the catastrophe gate misses
TODO_MIN_SENDS = 369          # the least sends per mode the case is graded on at all (probe.py's REPEATS sends 470)
TODO_MIN_CALL_SHARE = 0.9     # sends that must end in a todo_write call (else the case did not exercise the array)
TODO_FORMS = ("padded", "garbled", "invented", "unparseable")
_KEY_JUNK = set(':",{}[]<>=\\\n\r\t')

def todo_binom_sf(k, n, p):
    """P(X > k) for X ~ Binomial(n, p), summed exactly (stdlib only)."""
    return max(0.0, 1.0 - sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(min(k, n) + 1)))

def todo_max_bad(calls, p0=None):
    """The most defective calls (or differing sends) one check of the case may carry out of <calls>: the least k with
    P(X > k | calls, p0) <= the per-check false-fail of TODO_BAR. <p0>: TODO_BASE_RATE (470 calls: 10; 369: 9), or
    TODO_DIFFER_RATE for the raw-vs-parsed count (470 sends: 2; 333: 1)."""
    p0 = TODO_BASE_RATE if p0 is None else p0
    return next(k for k in range(calls + 1) if todo_binom_sf(k, calls, p0) <= TODO_CHECK_FALSE_FAIL)

def todo_binom_p(k, n, p=None):
    """The one-sided binomial p of <k> defective of <n>: P(X >= k | n, p) (p: TODO_BASE_RATE, the pooled healthy rate),
    the upper tail summed directly so a defect arm's p reads as tiny, not as 1 minus a rounded 1."""
    p = TODO_BASE_RATE if p is None else p
    if k <= 0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(k, n + 1)))

def todo_fisher_p(k, n, bad=None, calls=None):
    """The one-sided Fisher exact p of an arm's <k> defective of <n> against the pooled healthy arms (<bad> of <calls>,
    default TODO_HEALTHY_BAD / TODO_HEALTHY_CALLS): the hypergeometric chance the arm holds >= k of the k + bad
    defective calls when its n are drawn from the n + calls together -- small when the arm is worse than the pool."""
    bad = TODO_HEALTHY_BAD if bad is None else bad
    calls = TODO_HEALTHY_CALLS if calls is None else calls
    if k <= 0:
        return 1.0
    t, m = k + bad, n + calls
    lc = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    return min(1.0, sum(math.exp(lc(n, x) + lc(calls, t - x) - lc(m, t)) for x in range(k, min(t, n) + 1)))

def todo_power(n, p0=None, rate=TODO_CATCH_RATE):
    """The chance the case FAILs when one count carries <rate> (TODO_CATCH_RATE, or TODO_SMALL_RATE) in both modes, n
    calls per mode, under the bar todo_max_bad(n, p0) -- the most power any k within the false-fail budget reaches."""
    return 1 - (1 - todo_binom_sf(todo_max_bad(n, p0), n, rate)) ** 2

def todo_next_n(n, p0=None):
    """The least call count >= n whose bar at <p0> reaches TODO_POWER."""
    return next(m for m in itertools.count(n) if todo_power(m, p0) >= TODO_POWER)

def todo_short(label, n, p0=None):
    """Why a count of <n> calls cannot be graded at TODO_POWER under its null <p0>, or None when its bar reaches it."""
    pw = todo_power(n, p0)
    if pw >= TODO_POWER:
        return None
    return "%s: too few calls for the gate (n=%d, k %d, power %.2f%% < %g%% at a %g%% defect; %d call(s) reach it)" % (
        label, n, todo_max_bad(n, p0), 100 * pw, 100 * TODO_POWER, 100 * TODO_CATCH_RATE, todo_next_n(n, p0))

class _Pairs(list):
    """A JSON object as its (key, value) pairs, every key kept in order (a duplicate too)."""

class _NotJson(str):
    """A non-string DSML parameter whose text is not JSON."""

def loads_pairs(s):
    return json.loads(s, object_pairs_hook=_Pairs)

def _jtype(v):
    return "object" if isinstance(v, _Pairs) else "array" if isinstance(v, list) else "string" if isinstance(v, str) else \
        "null" if v is None else type(v).__name__

def key_form(k, allowed):
    """None for a declared key; else its form. padded: a declared key with whitespace around it (`"id "`); garbled: a key
    carrying JSON/DSML structure (`"id: 20, "`, `"first: "`), empty, a declared key's case-folded spelling, a truncated
    declared key (a word-boundary prefix of one) or a declared key run into junk past a non-identifier character;
    invented: any other undeclared key (`"notes_file_placehold"`)."""
    if k in allowed:
        return None
    s = k.strip()
    if s in allowed:
        return "padded"
    if not s or any(c in _KEY_JUNK for c in k) or s.lower() in {a.lower() for a in allowed}:
        return "garbled"
    for a in allowed:
        if (a.startswith(s) and not (a[len(s)].isalnum() or a[len(s)] == "_")) or \
           (s.startswith(a) and not (s[len(a)].isalnum() or s[len(a)] == "_")):
            return "garbled"
    return "invented"

def schema_keys(req):
    """{tool: (its declared parameter names, {parameter: its entry property names})}: an entry set for each parameter
    that is (or has an anyOf/oneOf branch that is) an array of objects with properties -- todo_write's `todos`."""
    out = {}
    for t in req.get("tools") or []:
        fn = t.get("function") or {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        nested = {}
        for k, p in props.items():
            for b in [p] + list(p.get("anyOf") or []) + list(p.get("oneOf") or []):
                it = b.get("items") if isinstance(b, dict) and b.get("type") == "array" else None
                if isinstance(it, dict) and it.get("properties"):
                    nested[k] = list(it["properties"])
        out[fn.get("name")] = (list(props), nested)
    return out

def arg_forms(obj, top, nested):
    """Every key defect in one call's arguments <obj> (a _Pairs): [(form, key, where)]. Top-level keys against <top>;
    each entry of a nested array parameter (or of its JSON-encoded string, the anyOf string branch) against its entry
    properties. An argument that is not an object, a nested parameter that is not an array of objects, or a non-string
    DSML value that is not JSON is `unparseable`."""
    if not isinstance(obj, _Pairs):
        return [("unparseable", None, "the arguments are a JSON %s, not an object" % _jtype(obj))]
    out = []
    for k, v in obj:
        f = key_form(k, top)
        if f:
            out.append((f, k, "arguments"))
        if isinstance(v, _NotJson):
            out.append(("unparseable", None, "%r is not JSON: %r" % (k, v[:60])))
            continue
        name = k if f is None else k.strip() if f == "padded" else None
        if name not in nested or v is None:
            continue
        if isinstance(v, str):
            try:
                v = loads_pairs(v)
            except ValueError:
                out.append(("unparseable", None, "%s is a string that is not JSON: %r" % (name, v[:60])))
                continue
        if not isinstance(v, list) or isinstance(v, _Pairs):
            out.append(("unparseable", None, "%s is a JSON %s, not an array" % (name, _jtype(v))))
            continue
        for i, e in enumerate(v):
            if not isinstance(e, _Pairs):
                out.append(("unparseable", None, "%s[%d] is a JSON %s, not an object" % (name, i, _jtype(e))))
                continue
            for k2, _ in e:
                f2 = key_form(k2, nested[name])
                if f2:
                    out.append((f2, k2, "%s[%d]" % (name, i)))
    return out

def parsed_forms(arguments, top, nested):
    """arg_forms of a returned call's arguments string."""
    try:
        obj = loads_pairs(arguments or "")
    except ValueError as e:
        return [("unparseable", None, "the arguments are not JSON (%s): %r" % (e, (arguments or "")[:60]))]
    return arg_forms(obj, top, nested)

def raw_forms(params, top, nested):
    """arg_forms of an emitted call (emitted_calls(..., strip=False) params): a string parameter's value as written, any
    other as JSON."""
    obj = _Pairs()
    for k, is_str, raw in params:
        if is_str:
            obj.append((k, raw))
            continue
        try:
            obj.append((k, loads_pairs(raw.strip())))
        except ValueError:
            obj.append((k, _NotJson(raw.strip())))
    return arg_forms(obj, top, nested)

def todo_verdict(mode, req, sends):
    """Grade one mode of a REPEATS case from its sends: [{send, hard: the send's own failed bars (HTTP, stream, router
    replay, self-parse), calls: [{name, arguments}] as returned, raw: its raw output text (the ids decoded by the
    tokenizer, else the engine's /detokenize text), raw_source: decode|engine, engine_raw, decode_error}]. Every call is
    read on both sides of the served parser -- RAW: the model's own DSML (emitted_calls, names unstripped); PARSED: the
    returned tool_calls arguments -- and counted for padded / garbled / invented (undeclared) keys and unparseable
    arguments (key_form, arg_forms). PASS needs >= TODO_MIN_SENDS sends at T 1.0 / top_p 1.0, no send failing its own
    bars, >= TODO_MIN_CALL_SHARE of sends ending in a call, and on EACH side defective calls <= todo_max_bad(calls)
    -- and the sends whose returned arguments are not the model's emitted ones (emitted_diffs) or whose defect
    forms differ between the sides <= todo_max_bad(sends with a call, TODO_DIFFER_RATE), the differ count's own null.
    Each bar is computed from the count's own n, and a count whose n no bar gives TODO_POWER (todo_short: below 234
    calls, 209 and 210 aside; 200 calls, k 6, 98.47 %) FAILs the mode as too few calls for the gate. Returns the row (`todo`
    holds the counts, each with its max_bad, power at TODO_CATCH_RATE, power_small at TODO_SMALL_RATE and pass_blind,
    the chance a TODO_BLIND_RATE defect passes its bar (the catastrophe gate's blind spot), `rate_why` the
    reasons in `why` that are a defect-rate bar: a side's padded / garbled / invented / unparseable calls over it, and
    `differ_why` the raw-vs-parsed bar's reason when the differ count is over it)."""
    sk, thinking, tools = schema_keys(req), thinking_of(req), req.get("tools")
    why, info = [], []
    n = len(sends)
    if n < TODO_MIN_SENDS:
        why.append("%d %s send(s): the case is graded on n >= %d per mode (probe.py's REPEATS; --repeat is a smoke run)" % (
            n, mode, TODO_MIN_SENDS))
    if req.get("temperature") != 1.0 or req.get("top_p") != 1.0:
        why.append("sent at temperature %r / top_p %r, not prod sampling 1.0 / 1.0" % (req.get("temperature"), req.get("top_p")))
    hard = [(s["send"], w) for s in sends for w in s["hard"]]
    if hard:
        why.append("%d send(s) failed their own bars: %s" % (len({x for x, _ in hard}), "; ".join("%s %s" % x for x in hard[:3])))
    side = {k: {"calls": 0, "defective": 0, "keys": dict.fromkeys(TODO_FORMS, 0), "examples": []} for k in ("raw", "parsed")}
    with_call, differ, src, rate_why = 0, [], {"decode": 0, "engine": 0}, []
    decode_differs = [s["send"] for s in sends if s["raw_source"] == "decode" and s.get("engine_raw") is not None
                      and s["raw"] != s["engine_raw"]]
    decode_errors = [(s["send"], s["decode_error"]) for s in sends if s.get("decode_error")]
    for s in sends:
        src[s["raw_source"]] += s["raw"] is not None
        emitted = [] if s["raw"] is None else emitted_calls(s["raw"], thinking, tools, strip=False)
        parsed = [c for c in s["calls"] if c["name"] in sk]
        with_call += bool(emitted or parsed)
        got = {}
        for name, calls, fn in (("raw", emitted, lambda c, t, nst: raw_forms(c["params"], t, nst)),
                                ("parsed", parsed, lambda c, t, nst: parsed_forms(c["arguments"], t, nst))):
            got[name] = []
            for i, c in enumerate(calls):
                top, nested = sk.get(c["name"], ([], {}))
                forms = fn(c, top, nested)
                got[name].append(sorted((f, k) for f, k, _ in forms))
                side[name]["calls"] += 1
                side[name]["defective"] += bool(forms)
                for f, k, where in forms:
                    side[name]["keys"][f] += 1
                    if len(side[name]["examples"]) < 6:
                        side[name]["examples"].append("%s call %d %s: %s %s" % (
                            s["send"], i, where, f, repr(k) if k is not None else ""))
        if s["raw"] is not None:
            ed = emitted_diffs(emitted_calls(s["raw"], thinking, tools), s["calls"])
            if ed or got["raw"] != got["parsed"]:
                differ.append("%s: %s" % (s["send"], ed[0][:200] if ed else "defects raw %r, parsed %r" % (got["raw"], got["parsed"])))
    if with_call < TODO_MIN_CALL_SHARE * n:
        why.append("only %d/%d send(s) ended in a call: the case did not exercise the nested array" % (with_call, n))
    for name in ("raw", "parsed"):
        x = side[name]
        rate = x["defective"] / x["calls"] if x["calls"] else 0.0
        x["rate"] = rate
        line = "%s: %d/%d call(s) defective (%.2f%%; padded %d, garbled %d, invented %d, unparseable %d key/value)%s" % (
            name, x["defective"], x["calls"], 100 * rate, x["keys"]["padded"], x["keys"]["garbled"], x["keys"]["invented"],
            x["keys"]["unparseable"], "; e.g. " + "; ".join(x["examples"]) if x["examples"] else "")
        x["max_bad"], x["power"] = todo_max_bad(x["calls"]), todo_power(x["calls"])
        x["power_small"] = todo_power(x["calls"], rate=TODO_SMALL_RATE)
        x["pass_blind"] = 1 - todo_binom_sf(x["max_bad"], x["calls"], TODO_BLIND_RATE)
        x["binom_p"], x["fisher_p"] = todo_binom_p(x["defective"], x["calls"]), todo_fisher_p(x["defective"], x["calls"])
        if x["defective"] > x["max_bad"]:
            line = line.replace("%.2f%%;" % (100 * rate), "%.2f%%; > %d allowed;" % (100 * rate, x["max_bad"]), 1)
        line = line.replace(" key/value)", " key/value; catastrophe bar k %d of n=%d at p0 %.2f%%, power %.2f%% at %g%% "
                            "(both modes), %.2f%% at %g%%, a %g%% defect passes %.2f%%; binomial p %.3g, Fisher p %.3g "
                            "vs pooled healthy %d/%d)" % (
            x["max_bad"], x["calls"], 100 * TODO_BASE_RATE, 100 * x["power"], 100 * TODO_CATCH_RATE,
            100 * x["power_small"], 100 * TODO_SMALL_RATE, 100 * TODO_BLIND_RATE, 100 * x["pass_blind"],
            x["binom_p"], x["fisher_p"], TODO_HEALTHY_BAD, TODO_HEALTHY_CALLS), 1)
        (why if x["defective"] > x["max_bad"] else info).append(line)
        if x["defective"] > x["max_bad"]:
            rate_why.append(line)
        short = todo_short(name, x["calls"])
        if short:
            why.append(short)
    drate = len(differ) / with_call if with_call else 0.0
    dmax, dpower = todo_max_bad(with_call, TODO_DIFFER_RATE), todo_power(with_call, TODO_DIFFER_RATE)
    dsmall = todo_power(with_call, TODO_DIFFER_RATE, TODO_SMALL_RATE)
    dline = ("raw vs parsed: %d/%d send(s) differ (%.2f%%%s; bar k %d of n=%d at its own p0 %.3f%%, power %.2f%% at %g%% "
             "(both modes), %.2f%% at %g%%; pooled healthy %d/%d)%s") % (
        len(differ), with_call, 100 * drate, "; > %d allowed" % dmax if len(differ) > dmax else "", dmax, with_call,
        100 * TODO_DIFFER_RATE, 100 * dpower, 100 * TODO_CATCH_RATE, 100 * dsmall, 100 * TODO_SMALL_RATE,
        TODO_DIFFER_BAD, TODO_DIFFER_SENDS, "; e.g. " + "; ".join(differ[:3]) if differ else "")
    (why if len(differ) > dmax else info).append(dline)
    differ_why = [dline] if len(differ) > dmax else []
    short = todo_short("raw vs parsed", with_call, TODO_DIFFER_RATE)
    if short:
        why.append(short)
    info.append("raw = %s" % ", ".join("%d %s" % (v, {"decode": "decoded from the output ids by --tokenizer",
                                                        "engine": "the engine's /detokenize text"}[k]) for k, v in src.items() if v)
                + ("" if src["decode"] else " (pass --tokenizer <checkpoint>/tokenizer.json to read the output ids themselves)"))
    if decode_differs:
        info.append("info, not graded: %d send(s) decode differently from the engine's /detokenize: %s" % (
            len(decode_differs), ", ".join(decode_differs[:5])))
    if decode_errors:
        info.append("tokenizer decode failed on %d send(s) (raw read from the engine's text): %s" % (
            len(decode_errors), "; ".join("%s %s" % x for x in decode_errors[:2])))
    first = [s["t_first_tool_delta"] for s in sends if s.get("t_first_tool_delta") is not None]
    return {"mode": mode, "ok": not why, "why": why, "info": info, "sends": n,
            "selfparse": {"equal": not any(w.startswith("SELF-PARSE") for _, w in hard), "sends": n},
            "todo": {"raw": side["raw"], "parsed": side["parsed"], "with_call": with_call, "differ": len(differ),
                     "differ_examples": differ[:10], "differ_max": dmax, "differ_power": dpower,
                     "differ_power_small": dsmall, "raw_source": src,
                     "rate_why": rate_why, "differ_why": differ_why,
                     "decode_differs": len(decode_differs), "first_delta_max_s": max(first) if first else None}}

# T1_todo_nested's two arms (probe.py CASES): the same todo_write schema, conversation and sampling, the tool's `strict`
# unset in the first and `"strict": false` in the second. The unset arm measures the TREE'S DEFAULT for an unset strict:
# it PASSes on a tree carrying the fork's strict-default fix (#901, `prod-tool-strict-default`: an unset strict is not
# strict) and fails by design on one without it, where upstream sglang #46133 grammar-constrains an unset strict and
# this schema's grammar admits invented keys (~99.7 % defective on box-a's 819756d4), so grade() takes the build's expectation from expect_unset (--expect-unset pass|fail) and prints a
# FAIL it expected as EXPECTED-FAIL, which does not fail the gate -- a FAIL on the defect-rate bars alone (todo_unexcused). The strict:false arm is what prod traffic carries
# (router release 12+ posts `strict: false` for every tool its caller left unset) and must PASS on every build.
TODO_UNSET_ARM = "T1_todo_nested"
TODO_ARMS = (TODO_UNSET_ARM, "T1_todo_nested_strictfalse")
EXPECT_UNSET = ("pass", "fail")
# The fork commits that carry the strict-default fix (#901): a tree under test has it iff one of them is its ancestor.
STRICT_FIX_COMMITS = ("601f5614", "faabdcae")


def tree_expect_unset(tree, repo=SGLANG_REPO, fixes=None):
    """What the unset arm must do on the tree under test <tree>, by a read-only git query on the fork checkout <repo>
    (nothing is fetched): ("pass", why) when a STRICT_FIX_COMMITS commit is an ancestor of <tree>, ("fail", why) when
    none is -- a fix commit absent from a full (non-shallow) clone that holds <tree> cannot be its ancestor -- and
    (None, why) when the checkout cannot tell: no such checkout, <tree> not in it, a fix commit missing from a shallow
    clone, a git error."""
    fixes = STRICT_FIX_COMMITS if fixes is None else fixes
    if not re.fullmatch(r"[0-9a-f]{7,40}", tree or ""):
        return None, "%r is not a commit sha (7 to 40 lowercase hex)" % (tree,)
    if not os.path.isdir(repo):
        return None, "no fork checkout at %s" % repo

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", repo] + list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               universal_newlines=True)
        except OSError:
            return 127, ""
        return r.returncode, r.stdout.strip()

    def commit(ref):
        rc, out = git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
        return out if rc == 0 and out else None

    full = commit(tree)
    if full is None:
        return None, "%s does not hold commit %s (nothing here fetches: fetch the fork there)" % (repo, tree)
    rc, shallow = git("rev-parse", "--is-shallow-repository")
    shallow = shallow if rc == 0 else "unknown"
    absent, unsure = [], []
    for f in fixes:
        fc = commit(f)
        if fc is None:
            (absent if shallow == "false" else unsure).append(f)
            continue
        rc, _ = git("merge-base", "--is-ancestor", fc, full)
        if rc == 0:
            return "pass", "fork commit %s (the strict-default fix) is an ancestor of %s in %s" % (f, full[:12], repo)
        if rc != 1:
            unsure.append(f)
    if unsure:
        return None, "%s cannot say whether %s %s an ancestor of %s (shallow clone: %s, or a git error)" % (
            repo, " / ".join(unsure), "is" if len(unsure) == 1 else "are", full[:12], shallow)
    return "fail", "none of the strict-default fix commits %s is an ancestor of %s in %s%s" % (
        " / ".join(fixes), full[:12], repo, " (%s absent from this full clone)" % " / ".join(absent) if absent else "")


def grade_expect(tree, given, force, repo=SGLANG_REPO):
    """(expectation, note) for the unset arm: <given> (--expect-unset, or None) checked against the tree under test
    <tree> (--tree, else --detector-tree, or None; main_grade refuses the two differing) by tree_expect_unset. A
    known tree decides it and a <given> that contradicts it is refused, --force-expect or not; a tree the checkout
    cannot place is refused unless <given> is explicit and <force> (--force-expect) is set. With no tree named, `fail`
    (the value that excuses the arm's FAIL) needs <force>; `pass` (the default) excuses nothing."""
    if given is not None and given not in EXPECT_UNSET:
        raise SystemExit("--expect-unset takes pass (a tree with fork #901) or fail (a tree without it), got %r" % given)
    if force and given is None:
        raise SystemExit("--force-expect takes an explicit --expect-unset pass|fail")
    if tree is None:
        if given == "fail" and not force:
            raise SystemExit("--expect-unset fail excuses the unset arm's FAIL: name the tree under test (--tree <sha> "
                             "[--sglang-repo DIR]) so it is checked, or add --force-expect to take it as given")
        e = given or "pass"
        return e, "--expect-unset %s (%s; no tree checked)" % (e, "--force-expect" if force else "given" if given else "the default")
    derived, why = tree_expect_unset(tree, repo)
    if derived is None:
        if given is None or not force:
            raise SystemExit("cannot tell whether tree %s carries the strict-default fix: %s; fetch the fork there, or "
                             "pass --expect-unset pass|fail --force-expect" % (tree, why))
        return given, "--expect-unset %s taken as given (--force-expect): %s" % (given, why)
    if given is not None and given != derived:
        raise SystemExit("--expect-unset %s contradicts tree %s: %s, so the unset arm must %s there (--expect-unset %s)"
                         % (given, tree, why, derived.upper(), derived))
    return derived, "--expect-unset %s from tree %s: %s" % (derived, tree, why)


_START_RE = re.compile(r"^START \S+ cases=(\[[^\]]*\]) modes=(\[[^\]]*\])")

def expected_rows(ph, present):
    """The (case, mode) rows phase <ph> must carry: every case x mode its probe.log START line(s) asked for (probe.py
    writes one per run, before its first send), else -- a phase recorded without a probe.log (the tracked #2539
    fixtures) -- every case dir holding a request x every mode recorded in the phase (<present>: the (case, mode) rows
    graded); and a T1_todo_nested arm brings the other arm in the same modes."""
    want = set()
    lp = os.path.join(ph, "probe.log")
    if os.path.isfile(lp):
        for line in open(lp, encoding="utf-8", errors="replace"):
            m = _START_RE.match(line)
            if m:
                want |= {(c, md) for c in ast.literal_eval(m.group(1)) for md in ast.literal_eval(m.group(2))}
    if not want:
        modes = {md for _, md in present} or {"stream", "nonstream"}
        want = {(c, md) for c in os.listdir(ph) if os.path.isdir(os.path.join(ph, c))
                and request_of(os.path.join(ph, c)) is not None for md in modes}
    return want | {(a, md) for c, md in want if c in TODO_ARMS for a in TODO_ARMS}

def todo_unexcused(row):
    """The reasons the unset arm's FAIL is NOT excused under --expect-unset fail: everything in its `why` but the
    defect-rate bars (todo_verdict's `rate_why`) and, when one of those failed, the raw-vs-parsed bar (`differ_why`:
    the sends whose emitted arguments or defect forms differ between the sides -- garbled keys parse differently, so
    it is the same defect showing; over its bar with no defect-rate bar failed it is a parser defect, not excused). A
    tree without #901 is expected to emit defective calls, never to fail a send (HTTP, stream, router replay,
    self-parse), run short of sends or calls, or leave prod sampling: any of those still FAILs the row. Empty: the
    FAIL is all the defect, an EXPECTED-FAIL."""
    todo = row.get("todo") or {}
    rate = todo.get("rate_why") or []
    excused = rate + (todo.get("differ_why") or []) if rate else []
    return [w for w in row["why"] if w not in excused]

def missing_why(ph, case, mode):
    d = os.path.join(ph, case)
    if not os.path.isdir(d):
        return "no %s/ in the phase: the case was never sent (probe.py stopped early, or ran without it)" % case
    if request_of(d) is None:
        return "%s/ holds no request.json" % case
    if send_dirs(d) is not None:
        return "no %s transcript in any of the case's %d send dir(s)" % (mode, len(send_dirs(d)))
    return "no %s transcript recorded" % mode

def print_todo_row(row):
    t = row["todo"]
    print("%s %-8s %-22s %-9s sends=%d with-call=%d defective raw=%d/%d parsed=%d/%d differ=%d/%d "
          "(CATASTROPHE gate: bar k <= %d/%d/%d at p0 %.2f/%.2f/%.3f%%, power %.2f/%.2f/%.2f%% at %g%% (both modes), "
          "%.2f/%.2f/%.2f%% at %g%%, a %g%% defect passes %.2f/%.2f%% (one mode); binomial p %.3g/%.3g, Fisher p "
          "%.3g/%.3g vs pooled healthy %d/%d, differ %d/%d)%s" % (
        row.get("verdict") or ("PASS" if row["ok"] else "FAIL"), row["phase"], row["case"], row["mode"], row["sends"],
        t["with_call"],
        t["raw"]["defective"], t["raw"]["calls"], t["parsed"]["defective"], t["parsed"]["calls"], t["differ"],
        t["with_call"], t["raw"]["max_bad"], t["parsed"]["max_bad"], t["differ_max"], 100 * TODO_BASE_RATE,
        100 * TODO_BASE_RATE, 100 * TODO_DIFFER_RATE, 100 * t["raw"]["power"], 100 * t["parsed"]["power"],
        100 * t["differ_power"], 100 * TODO_CATCH_RATE, 100 * t["raw"]["power_small"], 100 * t["parsed"]["power_small"],
        100 * t["differ_power_small"], 100 * TODO_SMALL_RATE, 100 * TODO_BLIND_RATE, 100 * t["raw"]["pass_blind"],
        100 * t["parsed"]["pass_blind"], t["raw"]["binom_p"],
        t["parsed"]["binom_p"], t["raw"]["fisher_p"], t["parsed"]["fisher_p"], TODO_HEALTHY_BAD, TODO_HEALTHY_CALLS,
        TODO_DIFFER_BAD, TODO_DIFFER_SENDS,
        ("" if row["ok"] else " | " + "; ".join(row["why"]))[:900]))
    for x in row["info"]:
        print("     %s" % x)

def grade(phases, src, py=None, tool_parser="deepseekv41", reasoning_parser="deepseek-v41", tokenizer=None, json_out=None,
          quiet=False, expect_unset="pass", expect_from=None):
    """Grade every request under <phases>. Returns (ok, rows). ok needs every row PASS (the unset T1 arm may be
    EXPECTED-FAIL under expect_unset "fail") and no expected case x mode row missing (expected_rows: a missing one is
    a FAIL line and an entry in <phase>/selfparse.json's "missing", never a skip). <expect_from> (grade_expect's
    note: how the expectation was decided) lands in selfparse.json as "expect_unset_from"."""
    if expect_unset not in EXPECT_UNSET:
        raise SystemExit("--expect-unset takes pass (a tree with fork #901) or fail (a tree without it), got %r" % expect_unset)
    results, winfo = run_worker(parse_jobs(phases, tool_parser, reasoning_parser, decode=bool(tokenizer)), src, py, tokenizer)
    rows, bad, missing = [], 0, []
    for ph in phases:
        prow = []
        for case in sorted(os.listdir(ph)):
            d = os.path.join(ph, case)
            req = request_of(d) if os.path.isdir(d) else None
            if req is None:
                continue
            per_mode, sends, items = {}, {}, send_items(ph, case, d)
            for d, mode, jid, sub in items:   # sub: a REPEATS send, graded by its own bars here and in todo_rows
                why, info = [], []
                meta = {}
                mp = os.path.join(d, mode + ".meta.json")
                if os.path.isfile(mp):
                    meta = json.load(open(mp))
                    if meta.get("status") != 200:
                        why.append("HTTP %s %s" % (meta.get("status"), meta.get("error", "")))
                try:
                    calls, s, n = returned_calls(d, mode)
                except Exception as e:
                    calls, s, n = None, None, None
                    why.append("unreadable %s transcript: %r" % (mode, e))
                if calls is None:
                    why.append("no %s response with choices (%r)" % (mode, (n or {}).get("error")))
                    calls = []
                if s is not None:
                    if s["errors"]:
                        why.append("stream errors %r" % s["errors"][:2])
                    if s["bad_lines"]:
                        why.append("%d unparseable SSE line(s)" % s["bad_lines"])
                    if not s["done"]:
                        why.append("stream had no [DONE]")
                    for i, c in enumerate(calls):
                        replay, events = router_replay(c["fragments"])
                        if replay != c["arguments"] or events:
                            why.append("router replay of call %d: %s" % (i, events or "text != raw concatenation"))
                # THE graded bar: this request's own raw output, re-parsed in one shot
                text, rawname = load_raw(d, mode)
                sp = {"raw": rawname, "equal": False}
                if text is None:
                    why.append("SELF-PARSE: no raw output recorded for this request (%s)" % (
                        (meta.get("raw") or {}).get("error") or "no %s.raw.txt: probe.py records it unless --no-raw" % mode))
                else:
                    r = results.get(jid) or {"error": "no worker result"}
                    if "error" in r:
                        why.append("SELF-PARSE: the re-parse raised %s" % r["error"])
                    else:
                        rp = r["calls"]
                        sp.update({"path": r["path"], "eos_stripped": r["eos_stripped"], "reasoning_chars": r["reasoning_chars"],
                                   "reparse_calls": len(rp), "returned_calls": len(calls)})
                        diffs = []
                        if [c["name"] for c in rp] != [c["name"] for c in calls]:
                            diffs.append("names: re-parse %r, returned %r" % ([c["name"] for c in rp], [c["name"] for c in calls]))
                        for i, (a, b) in enumerate(zip(rp, calls)):
                            x, y = (a["arguments"] or "").encode("utf-8"), (b["arguments"] or "").encode("utf-8")
                            if x != y:
                                k = first_diff(x, y)
                                diffs.append("call %d args differ at byte %d (returned %d B, re-parse %d B): returned %r / re-parse %r" % (
                                    i, k, len(y), len(x), y[max(0, k - 30):k + 30].decode("utf-8", "replace"),
                                    x[max(0, k - 30):k + 30].decode("utf-8", "replace")))
                        sp["args_sha256"] = [sha((c["arguments"] or "").encode("utf-8"))[:16] for c in calls]
                        if diffs:
                            why.extend("SELF-PARSE: " + x for x in diffs)
                        else:
                            sp["equal"] = True
                    # ... and the returned arguments are the model's own: a served parser that reproduces its own
                    # misreading (box-e's recipe on a spaced parameter name) is byte-equal above and caught here
                    if r.get("path") != "json" and sub is None:   # a REPEATS send: todo_verdict reads it by rate
                        ed = emitted_diffs(emitted_calls(text, thinking_of(req), req.get("tools")), calls)
                        sp["emitted_equal"] = not ed
                        why.extend("EMITTED: " + x for x in ed)
                if sub is not None:
                    dec = results.get("decode|" + jid) if tokenizer else None
                    sends.setdefault(mode, []).append({
                        "send": sub, "hard": why, "calls": calls, "engine_raw": text,
                        "raw": dec["text"] if dec and "text" in dec else text,
                        "raw_source": "decode" if dec and "text" in dec else "engine",
                        "decode_error": (dec or {}).get("error") if tokenizer else None,
                        "t_first_tool_delta": (s or {}).get("t_first_tool_delta")})
                    continue
                bw, bi = case_bars(case, mode, req, calls, s)
                why += bw
                info += bi
                row = {"phase": os.path.basename(ph.rstrip("/")), "case": case, "mode": mode, "ok": not why, "why": why,
                       "info": info, "selfparse": sp, "calls": [c["name"] for c in calls],
                       "args_bytes": [len((c["arguments"] or "").encode("utf-8")) for c in calls],
                       "t_first_tool_delta": (s or {}).get("t_first_tool_delta"),
                       "after_reasoning_s": (s or {}).get("first_tool_delta_after_reasoning_s"),
                       "_args": [c["arguments"] for c in calls]}
                per_mode[mode] = row
            for mode, ss in sends.items():
                per_mode[mode] = dict(todo_verdict(mode, req, ss), phase=os.path.basename(ph.rstrip("/")), case=case)
            if "stream" in per_mode and "nonstream" in per_mode and not sends:
                a, b = per_mode["stream"]["_args"], per_mode["nonstream"]["_args"]
                same = [x.encode("utf-8") for x in a] == [x.encode("utf-8") for x in b]
                where = None if same or not (a and b) else first_diff(a[0].encode("utf-8"), b[0].encode("utf-8"))
                per_mode["stream"]["info"].append("info, not graded: stream vs non-stream (two generations) %s" % (
                    "byte-equal" if same else "differ%s" % ("" if where is None else " at byte %d" % where)))
            for row in per_mode.values():
                row.pop("_args", None)
                row["verdict"] = "PASS" if row["ok"] else "FAIL"
                if case == TODO_UNSET_ARM:
                    row["expect"] = expect_unset
                    unexcused = todo_unexcused(row)
                    if expect_unset == "fail" and not row["ok"] and unexcused:
                        row["info"].insert(0, "FAIL under --expect-unset fail: EXPECTED-FAIL excuses only the defect-rate "
                                              "bars (padded / garbled / invented / unparseable calls over the bar, and the raw-vs-"
                                              "parsed bar beside them); "
                                              "%d other bar(s) failed: %s" % (len(unexcused), "; ".join(unexcused)[:400]))
                    elif expect_unset == "fail" and not row["ok"]:
                        row["verdict"] = "EXPECTED-FAIL"
                        row["info"].insert(0, "EXPECTED-FAIL (--expect-unset fail): a tree without fork #901 grammar-"
                                              "constrains an unset strict; the strict:false arm carries the gate")
                    elif expect_unset == "fail":
                        row["info"].insert(0, "PASS, though --expect-unset fail says this tree lacks fork #901: "
                                              "check the build (a #901 tree is graded --expect-unset pass)")
                    elif not row["ok"]:
                        row["info"].insert(0, "expected PASS (--expect-unset pass: a tree with fork #901); on a tree "
                                              "without #901 this arm fails by design: grade it --expect-unset fail")
                bad += row["verdict"] == "FAIL"
                prow.append(row)
                if not quiet and "todo" in row:
                    print_todo_row(row)
                elif not quiet:
                    em = row["selfparse"].get("emitted_equal")
                    print("%s %-8s %-22s %-9s calls=%s bytes=%s self-parse=%s%s%s" % (
                        row["verdict"], row["phase"], case, row["mode"], row["calls"], row["args_bytes"],
                        ("byte-equal" if row["selfparse"]["equal"] else "NOT-EQUAL")
                        + ("" if em is None else " emitted=%s" % ("equal" if em else "NOT-EQUAL")),
                        ("" if row["t_first_tool_delta"] is None else " first-delta=%.2fs" % row["t_first_tool_delta"])
                        + ("" if row["after_reasoning_s"] is None else " (%.3fs after reasoning)" % row["after_reasoning_s"]),
                        ("" if row["ok"] else " | " + "; ".join(row["why"]))[:900]))
                    for x in row["info"]:
                        print("     %s" % x)
        rows += prow
        # an expected case x mode with no graded row FAILs the grade, named: never a silent skip
        phase = os.path.basename(ph.rstrip("/"))
        pmiss = [{"phase": phase, "case": c, "mode": m, "why": missing_why(ph, c, m)}
                 for c, m in sorted(expected_rows(ph, {(r["case"], r["mode"]) for r in prow}) - {(r["case"], r["mode"]) for r in prow})]
        for x in pmiss:
            if not quiet:
                print("FAIL %-8s %-22s %-9s MISSING: %s" % (phase, x["case"], x["mode"], x["why"]))
        missing += pmiss
        # <phase>/selfparse.json beside the transcripts; the self-test names a prefix of its own instead
        jp = os.path.join(ph, "selfparse.json") if json_out is None else "%s.%s.json" % (json_out, phase)
        json.dump({"detector": winfo, "tool_parser": tool_parser, "reasoning_parser": reasoning_parser,
                   "expect_unset": expect_unset, "rows": prow, "missing": pmiss,
                   **({"expect_unset_from": expect_from} if expect_from else {})},
                  open(jp, "w"), indent=1, ensure_ascii=False)
    ok = bool(rows) and bad == 0 and not missing
    if not quiet:
        rep = [r for r in rows if "todo" in r]
        print("DSML-SELFPARSE %s: %d request(s) graded in %d phase(s), %d failed, %d expected row(s) missing%s%s (%s)" % (
            "PASS" if ok else "FAIL", len(rows) - len(rep) + sum(r["sends"] for r in rep), len(phases), bad, len(missing),
            " (%s)" % ", ".join("%s %s" % (x["case"], x["mode"]) for x in missing) if missing else "",
            "; %d of the rows are repeated-case modes (%s; --expect-unset %s)" % (len(rep), ", ".join("%s %s %s" % (
                r["case"], r["mode"], r["verdict"]) for r in rep), expect_unset) if rep else "",
            "detector %s" % winfo["detector_file"] if winfo else "no raw output to re-parse"))
    return ok, rows


def detector_args(argv):
    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    src = opt("--detector-src")
    if src is None:
        tree = opt("--detector-tree")
        if tree is None:
            raise SystemExit("name the served tree: --detector-tree <40-hex sha> [--sglang-repo DIR] or --detector-src <tree>/python")
        src = detector_src_for_tree(tree, opt("--sglang-repo", SGLANG_REPO))
    return {"src": os.path.abspath(src), "py": opt("--detector-py", DETECTOR_PY),
            "tool_parser": opt("--tool-parser", "deepseekv41"), "reasoning_parser": opt("--reasoning-parser", "deepseek-v41"),
            "tokenizer": opt("--tokenizer", os.environ.get("DSML_TOKENIZER"))}


VALUE_OPTS = ("--detector-src", "--detector-tree", "--sglang-repo", "--detector-py", "--tool-parser", "--reasoning-parser",
              "--tokenizer", "--expect-unset", "--tree")
FLAG_OPTS = ("--force-expect",)


def positional(argv):
    """grade's phase dirs: every word that is neither an option nor an option's value."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in VALUE_OPTS:
            skip = True
        elif a in FLAG_OPTS:
            pass
        elif a.startswith("--"):
            raise SystemExit("unknown option %s (grade takes %s)" % (a, ", ".join(VALUE_OPTS + FLAG_OPTS)))
        else:
            out.append(a)
    return out


def main_grade(argv):
    def opt(name):
        if name not in argv:
            return None
        v = (argv[argv.index(name) + 1:] or [None])[0]
        if v is None or v.startswith("--"):
            raise SystemExit("%s takes a value" % name)
        return v
    # the expectation first: a refused one grades nothing
    tree, dtree = opt("--tree"), opt("--detector-tree")
    if tree and dtree and not (tree.lower().startswith(dtree.lower()) or dtree.lower().startswith(tree.lower())):
        raise SystemExit("--tree %s and --detector-tree %s name different commits: the expectation would follow one tree "
                         "while the detector parses with another; name one tree (--detector-tree alone is both)"
                         % (tree, dtree))
    tree = tree or dtree
    expect_unset, expect_from = grade_expect(tree, opt("--expect-unset"), "--force-expect" in argv,
                                             opt("--sglang-repo") or SGLANG_REPO)
    print("EXPECT-UNSET " + expect_from)
    da = detector_args(argv)
    phases = [os.path.abspath(p) for p in positional(argv)]
    if not phases:
        raise SystemExit("usage: analyze.py grade <phase dir>... --detector-tree <sha>")
    for p in phases:
        if not os.path.isdir(p):
            raise SystemExit("no such phase dir: %s" % p)
    ok, _ = grade(phases, da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"],
                  expect_unset=expect_unset, expect_from=expect_from)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------------------------------------------------
# self-test: the gate's own proof on recorded transcripts
# ---------------------------------------------------------------------------------------------------------------------
def stream_texts(d):
    """(reasoning text or None, content text) of a recorded stream."""
    reasoning, content, saw = [], [], False
    for line in open(sse_path(d), "rb"):
        line = line.rstrip(b"\r\n")
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        try:
            ch = json.loads(line[5:].strip().decode("utf-8"))
        except ValueError:
            continue
        for c in ch.get("choices") or []:
            dl = c.get("delta") or {}
            if dl.get("reasoning_content") is not None:
                saw = True
                reasoning.append(dl["reasoning_content"])
            if dl.get("content"):
                content.append(dl["content"])
    return ("".join(reasoning) if saw else None), "".join(content)


def nonstream_texts(d):
    p = os.path.join(d, "nonstream.json")
    n = json.load(open(p if os.path.isfile(p) else os.path.join(d, "nonstream.raw"), encoding="utf-8"))
    m = n["choices"][0].get("message") or {}
    return m.get("reasoning_content"), m.get("content") or ""


def unrenderable(calls, finish):
    """Why a send's returned calls have no complete rendering, or None when every call's arguments are a whole JSON
    object. A send that finished at max_tokens (finish length) is a cut; any other send carrying arguments that do not
    parse was served that way by the tree under test (the 9d908424 N2 sends: `{"first: "abc123"}`)."""
    for i, c in enumerate(calls):
        try:
            v = json.loads(c["arguments"] or "")
        except ValueError:
            v = ValueError
        if isinstance(v, dict):
            continue
        what = "are not valid JSON" if v is ValueError else "are JSON but not an object"
        if finish == "length":
            return ("call %d %s's recorded arguments %s: the send was cut at max_tokens (finish length)"
                    % (i, c["name"], what))
        return "call %d %s's recorded arguments %s (finish %s, not a max_tokens cut): %s" % (
            i, c["name"], what, finish, json.dumps(c["arguments"], ensure_ascii=False))
    return None


def render_jobs(root, skipped):
    """One render job per recorded send under <root>/<phase>/<case>/: that send's own returned calls, reasoning and
    content, written back as the DSML text the model emits. Sends with a call whose arguments are not a whole JSON
    object are listed in <skipped> instead, as (case dir relative to <root>, mode, unrenderable()'s reason)."""
    jobs = []
    for ph in sorted(os.listdir(root)):
        pd = os.path.join(root, ph)
        if not os.path.isdir(pd):
            continue
        for case in sorted(os.listdir(pd)):
            d = os.path.join(pd, case)
            req = request_of(d) if os.path.isdir(d) else None
            if req is None:
                continue
            for mode in modes_in(d):
                calls, s, n = returned_calls(d, mode)
                if calls is None:
                    continue
                why = unrenderable(calls, (s or n or {}).get("finish_reason"))
                if why:
                    # no complete rendering: a call cut mid-arguments (finish length) really emitted an unclosed
                    # invoke, which the one-shot parse drops; arguments that do not parse otherwise are the tree's
                    # own mis-serving -- either way a FAIL the grade reports
                    skipped.append((os.path.relpath(d, root), mode, why))
                    continue
                reasoning, content = stream_texts(d) if mode == "stream" else nonstream_texts(d)
                if thinking_of(req) and reasoning is None:
                    reasoning = ""
                jobs.append({"id": "%s|%s" % (d, mode), "op": "render", "content": content,
                             "reasoning": reasoning if thinking_of(req) else None,
                             "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]})
    return jobs


def copy_case(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith((".raw.txt", ".detok.txt")) and f != "selfparse.json":
            shutil.copy2(p, os.path.join(dst, f))


class _Replay:
    """A local stand-in for the engine, for probe.py end to end: /v1/chat/completions replays a recorded send of the
    case whose last user message matches, adding sglext.output_ids (stream: a `choices: []` chunk before [DONE], as
    serving_chat sends it; non-stream: response.sglext); /detokenize maps those ids back to that send's raw text.
    Every request is logged (path, body, whether the bearer matched)."""

    def __init__(self, cases, key, fail_detok=()):
        self.cases, self.key, self.fail_detok, self.log, self.ids = cases, key, set(fail_detok), [], {}
        n = 0
        for name, c in cases.items():
            for mode in ("stream", "nonstream"):
                n += 1
                self.ids[(name, mode)] = [100000 + n * 10 + k for k in range(5)]

    def serve(self):
        import http.server
        import threading
        rp = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                auth_ok = self.headers.get("Authorization") == "Bearer " + rp.key
                j = json.loads(body.decode("utf-8")) if body else None
                rp.log.append({"path": self.path, "body": j, "auth_ok": auth_ok})
                if self.path == "/detokenize":
                    hit = [k for k, v in rp.ids.items() if v == j.get("tokens")]
                    if not hit or hit[0] in rp.fail_detok:
                        return self._send(404, b'{"error":"no such ids"}', "application/json")
                    text = rp.cases[hit[0][0]]["raw"][hit[0][1]]
                    return self._send(200, json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"), "application/json")
                name = next((n for n, c in rp.cases.items() if c["user"] == j["messages"][-1]["content"]), None)
                if name is None:
                    return self._send(400, b'{"error":"unknown case"}', "application/json")
                c = rp.cases[name]
                if j.get("stream"):
                    out = []
                    for line in open(c["sse"], "rb").read().splitlines(keepends=True):
                        if line.strip() == b"data: [DONE]" and j.get("return_output_ids_in_sglext"):
                            ext = {"id": "x", "object": "chat.completion.chunk", "choices": [],
                                   "sglext": {"output_ids": [rp.ids[(name, "stream")]]}}
                            out.append(b"data: " + json.dumps(ext).encode() + b"\n\n")
                        out.append(line)
                    return self._send(200, b"".join(out), "text/event-stream")
                n = json.load(open(c["nonstream"], encoding="utf-8"))
                if j.get("return_output_ids_in_sglext"):
                    n["sglext"] = {"output_ids": [rp.ids[(name, "nonstream")]]}
                return self._send(200, json.dumps(n, ensure_ascii=False).encode("utf-8"), "application/json")

            def _send(self, code, data, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.httpd.server_address[1]


def main_self_test(argv):
    da = detector_args(argv)
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None:
        raise SystemExit("self-test needs --out DIR (a scratch dir; recorded transcripts are copied there, never written)")
    def dirs_after(flag):
        out_ = []
        if flag in argv:
            i = argv.index(flag) + 1
            while i < len(argv) and not argv[i].startswith("--"):
                out_.append(os.path.abspath(argv[i]))
                i += 1
        return out_
    recs, real = dirs_after("--records"), dirs_after("--real")
    if not recs and not real:   # the tracked fixtures, as both the recorded and the real set
        recs = real = [p for p in FIXTURE_PHASES if os.path.isdir(p)]
    if not recs:
        raise SystemExit("no recorded phases (pass --records <phase dir>...)")
    for p in recs + real:
        if not os.path.isdir(p):
            raise SystemExit("no such phase dir: %s" % p)
    if os.path.exists(out):
        shutil.rmtree(out)
    N, F = [0], [0]

    def check(cond, what):
        N[0] += 1
        F[0] += not cond
        print("%s %s" % ("ok  " if cond else "FAIL", what))

    # 1. every recorded send, its own calls rendered back as its raw output: the self-parse must be byte-equal
    pos = os.path.join(out, "positive")
    labels = {}
    for r in recs:
        lab = "%s__%s" % (os.path.basename(os.path.dirname(r.rstrip("/"))), os.path.basename(r.rstrip("/")))
        labels[r] = lab
        for case in sorted(os.listdir(r)):
            if os.path.isdir(os.path.join(r, case)) and request_of(os.path.join(r, case)) is not None:
                copy_case(os.path.join(r, case), os.path.join(pos, lab, case))
    skipped = []
    rendered, _ = run_worker(render_jobs(pos, skipped), da["src"], da["py"], da["tokenizer"])
    for rel, mode, why in skipped:   # no complete rendering: those sends leave the positive set (their files are removed)
        print("     skip %s %s: %s" % (rel, mode, why))
        d = os.path.join(pos, rel)
        for f in os.listdir(d):
            if f.startswith(mode + ".") or (mode == "nonstream" and f == "nonstream.json"):
                os.remove(os.path.join(d, f))
    for jid, x in rendered.items():
        d, mode = jid.rsplit("|", 1)
        check("error" not in x, "render %s %s" % (os.path.relpath(d, pos), mode) + ("" if "error" not in x else ": " + x["error"]))
        if "error" not in x:
            open(os.path.join(d, mode + ".raw.txt"), "w", encoding="utf-8", newline="").write(x["text"])
    phases = [os.path.join(pos, lab) for lab in sorted(os.listdir(pos))]
    print("== positive: %d recorded send(s) in %d phase(s), each re-parsed from its own rendered raw output" % (len(rendered), len(phases)))
    _, rows = grade(phases, da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "positive"), quiet=True)
    for row in rows:
        check(row["selfparse"]["equal"], "self-parse byte-equal: %s/%s %s %s %s" % (
            row["phase"], row["case"], row["mode"], row["calls"], row["args_bytes"]))
    check(len(rows) == len(rendered) and len(rows) > 0, "every rendered send was graded (%d rows)" % len(rows))
    wi = json.load(open("%s.%s.json" % (os.path.join(out, "positive"), os.path.basename(phases[0])), encoding="utf-8"))["detector"]
    check(os.path.realpath(wi["detector_file"]).startswith(os.path.realpath(da["src"]) + os.sep),
          "the re-parse ran the named tree's own detector (%s)" % wi["detector_file"])
    div = [r for r in rows if r["mode"] == "stream" and any("differ" in i for i in r["info"])]
    check(all(r["selfparse"]["equal"] for r in div),
          "the %d case(s) whose stream and non-stream generated different text (the old cross-send bar's FAILs: %s) "
          "pass the in-request bar, the difference an info row" % (len(div), ", ".join("%s/%s" % (r["phase"], r["case"]) for r in div)))

    # 2. mutations: each must turn the self-parse (or the row) the way named
    def pick(case_prefix):
        for ph in phases:
            for case in sorted(os.listdir(ph)):
                if case.startswith(case_prefix) and os.path.isfile(os.path.join(ph, case, "stream.raw.txt")):
                    return ph, case
        return None, None

    mut = os.path.join(out, "mutated", "m")
    os.makedirs(mut)
    ph, case = pick("M2") if pick("M2")[0] else pick("M")
    want = {}
    if ph:
        src_d = os.path.join(ph, case)
        raw = open(os.path.join(src_d, "stream.raw.txt"), encoding="utf-8", newline="").read()
        k = raw.index('string="true">') + len('string="true">')
        k = next(i for i in range(k, len(raw)) if raw[i].isalnum())
        flip = "x" if raw[k] != "x" else "y"
        for tag, text, sse_edit in (("byte", raw[:k] + flip + raw[k + 1:], None),
                                    ("extra", raw.replace("</｜DSML｜ invoke>\n</｜DSML｜ calls>",
                                                          "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"write_file\">\n"
                                                          "<｜DSML｜ parameter name=\"path\" string=\"true\">x</｜DSML｜ parameter>\n"
                                                          "<｜DSML｜ parameter name=\"content\" string=\"true\">y</｜DSML｜ parameter>\n"
                                                          "</｜DSML｜ invoke>\n</｜DSML｜ calls>"), None),
                                    ("renamed", raw.replace('invoke name="write_file"', 'invoke name="read_file"', 1), None),
                                    ("dropped", raw, "drop"),
                                    ("noraw", None, None)):
            d = os.path.join(mut, "%s_%s" % (case, tag))
            copy_case(src_d, d)
            for f in os.listdir(d):
                if f.startswith("nonstream"):
                    os.remove(os.path.join(d, f))
            if sse_edit == "drop":   # remove one middle tool_calls fragment line from the stream (untimed copy)
                sp = sse_path(d)
                lines = open(sp, "rb").read().splitlines(keepends=True)
                frag = [i for i, l in enumerate(lines) if b'"arguments":"' in l and b'"arguments":""' not in l]
                del lines[frag[len(frag) // 2]]
                open(sp, "wb").write(b"".join(lines))
                if os.path.isfile(os.path.join(d, "stream.times.tsv")):
                    os.remove(os.path.join(d, "stream.times.tsv"))
            if text is not None:
                open(os.path.join(d, "stream.raw.txt"), "w", encoding="utf-8", newline="").write(text)
            want[os.path.basename(d)] = tag
    ph2, case2 = pick("L6")
    if ph2:   # thinking on: reasoning that quotes the unspaced V4 calls tag must stay reasoning (the V4.1 reasoning
        # parser ends reasoning at </think> or the spaced V4.1 calls tag only), so the re-parse still finds one call
        d = os.path.join(mut, "%s_reasoning" % case2)
        copy_case(os.path.join(ph2, case2), d)
        for f in os.listdir(d):
            if f.startswith("nonstream"):
                os.remove(os.path.join(d, f))
        raw = open(os.path.join(ph2, case2, "stream.raw.txt"), encoding="utf-8", newline="").read()
        raw = "I will not write <｜DSML｜tool_calls> by hand here; " + raw
        open(os.path.join(d, "stream.raw.txt"), "w", encoding="utf-8", newline="").write(raw)
        want[os.path.basename(d)] = "reasoning"
    _, mrows = grade([mut], da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "mutated"), quiet=True)
    by = {r["case"]: r for r in mrows}
    for c, tag in sorted(want.items()):
        r = by.get(c)
        if r is None:
            check(False, "mutation %s graded" % c)
            continue
        w = "; ".join(r["why"])
        if tag == "byte":
            check(not r["selfparse"]["equal"] and "args differ at byte" in w and not r["ok"], "one changed byte in a parameter value FAILs the self-parse (%s)" % w[:160])
        elif tag == "extra":
            check(not r["selfparse"]["equal"] and "names: re-parse" in w, "a second invoke in the raw output FAILs (%s)" % w[:160])
        elif tag == "renamed":
            check(not r["selfparse"]["equal"] and "names: re-parse" in w, "a call renamed to an undeclared tool is dropped by the parser and FAILs (%s)" % w[:160])
        elif tag == "dropped":
            check(not r["selfparse"]["equal"] and "args differ at byte" in w, "a stream missing one arguments fragment FAILs against its own raw output (%s)" % w[:160])
        elif tag == "noraw":
            check(not r["ok"] and "no raw output recorded" in w, "a request without its raw output FAILs, never passes by default (%s)" % w[:120])
        elif tag == "reasoning":
            check(r["selfparse"]["equal"] and r["selfparse"].get("reasoning_chars", 0) > 0,
                  "thinking on, reasoning quoting the V4 calls tag: the reasoning parser splits %d reasoning chars off before the detector, self-parse still byte-equal" % r["selfparse"].get("reasoning_chars", 0))

    # 2b. the declared-name cases (probe.py N1..N5 x required/auto), synthesized: a stream whose deltas build the call
    # the model rendered, its raw output that rendering; a stream that cut the name must FAIL both bars
    sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True
    import probe   # noqa: E402  (CASES only; nothing is sent)
    syn = os.path.join(out, "names", "n")
    os.makedirs(syn)
    sjobs, swant = [], {}
    for case in [c for c in probe.CASES if c.startswith("N")] + ["N5_required_cut"]:
        body = probe.CASES[case.replace("_cut", "")]
        name = list(body["tools"][0]["function"]["parameters"]["properties"])[0]
        sent = name[:-1] if case.endswith("_cut") else name
        d = os.path.join(syn, case)
        os.makedirs(d)
        json.dump(body, open(os.path.join(d, "request.json"), "w", encoding="utf-8"), ensure_ascii=False)
        frags = ["", "{" + json.dumps(sent, ensure_ascii=False) + ': "', 'abc123"}']
        lines, t = [], 0.2
        for i, a in enumerate(frags):
            tc = {"index": 0, "function": {"arguments": a}}
            if i == 0:
                tc.update(id="call_0", type="function")
                tc["function"]["name"] = "zapisz_pole"
            lines.append({"choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}]})
        lines.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        raw_sse, tsv, off = b"", "t_s\toffset\tlen\n", 0
        for ch in [json.dumps(x, ensure_ascii=False) for x in lines] + ["[DONE]"]:
            for ln in (("data: " + ch + "\n").encode("utf-8"), b"\n"):
                tsv += "%.4f\t%d\t%d\n" % (t, off, len(ln))
                raw_sse += ln
                off += len(ln)
                t += 0.01
        open(os.path.join(d, "stream.sse"), "wb").write(raw_sse)
        open(os.path.join(d, "stream.times.tsv"), "w").write(tsv)
        sjobs.append({"id": d, "op": "render", "content": "", "reasoning": None,
                      "calls": [{"name": "zapisz_pole", "arguments": json.dumps({name: "abc123"}, ensure_ascii=False)}]})
        swant[case] = not case.endswith("_cut")
    srend, _ = run_worker(sjobs, da["src"], da["py"], da["tokenizer"])
    for d, x in srend.items():
        open(os.path.join(d, "stream.raw.txt"), "w", encoding="utf-8", newline="").write(x["text"])
    _, nrows = grade([syn], da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "names"), quiet=True)
    for r in nrows:
        if swant[r["case"]]:
            check(r["ok"], "declared name kept whole: %s (%s)%s" % (r["case"], json.dumps(list(probe.CASES[r["case"]]["tools"][0]["function"]["parameters"]["properties"])[0], ensure_ascii=False),
                                                                  "" if r["ok"] else " | " + "; ".join(r["why"])[:200]))
        else:
            w = "; ".join(r["why"])
            check(not r["ok"] and "SELF-PARSE" in w and "want exactly the declared" in w,
                  "a stream that cut the declared name FAILs the self-parse and the key bar (%s)" % w[:200])
    check(len(nrows) == len(swant), "all %d declared-name requests graded" % len(swant))

    # 2c. REAL raw outputs (--real: transcripts that recorded each request's own detokenized output ids, e.g. #2539's
    # inreq.py run on box-c): graded as recorded they must PASS, agree with the run's own inreq.json verdicts where one
    # exists, and one changed byte in a real raw text must FAIL
    if real:
        rroot = os.path.join(out, "real")
        rph = []
        for r in real:
            dst = os.path.join(rroot, os.path.basename(r.rstrip("/")))
            shutil.copytree(r, dst, ignore=shutil.ignore_patterns("__pycache__", "selfparse.json"))
            rph.append(dst)
        _, rrows = grade(rph, da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "real"), quiet=True)
        check(len(rrows) > 0 and all(r["selfparse"]["raw"] for r in rrows), "%d real request(s) carry their own raw output" % len(rrows))
        for r in rrows:
            check(r["ok"], "real raw output graded PASS: %s/%s %s %s %s%s" % (r["phase"], r["case"], r["mode"], r["calls"], r["args_bytes"],
                                                                             "" if r["ok"] else " | " + "; ".join(r["why"])[:300]))
        for ph in rph:
            ij = os.path.join(ph, "inreq.json")
            if os.path.isfile(ij):
                theirs = json.load(open(ij, encoding="utf-8"))
                mine = {"%s/%s" % (r["case"], r["mode"]): r["ok"] for r in rrows if r["phase"] == os.path.basename(ph)}
                agree = [k for k in theirs if k in mine and bool(theirs[k]["ok"]) == mine[k]]
                check(len(agree) == len(theirs) == len(mine), "%s: every verdict agrees with the run's own inreq.json (%d/%d)" % (
                    os.path.basename(ph), len(agree), len(theirs)))
        victim = next((r for r in rrows if r["calls"] and r["selfparse"]["raw"]), None)
        if victim:
            src_d = os.path.join(rroot, victim["phase"], victim["case"])
            d = os.path.join(out, "real-mutated", "m", victim["case"] + "_byte")
            shutil.copytree(src_d, d)
            for f in os.listdir(d):
                if f.startswith("nonstream" if victim["mode"] == "stream" else "stream"):
                    os.remove(os.path.join(d, f))
            rawf = os.path.join(d, victim["selfparse"]["raw"])
            t = open(rawf, encoding="utf-8", newline="").read()
            k = t.index('string="true">') + len('string="true">')
            t = t[:k] + ("Z" if t[k] != "Z" else "Y") + t[k + 1:]
            open(rawf, "w", encoding="utf-8", newline="").write(t)
            _, mr = grade([os.path.dirname(d)], da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "real-mutated"), quiet=True)
            check(len(mr) == 1 and not mr[0]["selfparse"]["equal"], "one changed byte in a REAL raw output (%s %s) FAILs the self-parse" % (victim["case"], victim["mode"]))

    # 3. probe.py end to end against a local replay engine
    e2e_src = None
    for want_cases in (("M1_powers_off", "P2_pl_required_off", "S2_noarg_off", "C1_cfg_off"),):
        for ph in phases:
            if all(os.path.isfile(os.path.join(ph, c, "stream.raw.txt")) and os.path.isfile(os.path.join(ph, c, "nonstream.raw.txt")) for c in want_cases):
                e2e_src = (ph, want_cases)
                break
    check(e2e_src is not None, "a recorded phase holds M1, P2, S2 and C1 for the probe.py end-to-end run")
    if e2e_src:
        ph, cs = e2e_src
        model = request_of(os.path.join(ph, cs[0]))["model"]
        cases = {}
        for c in cs:
            d = os.path.join(ph, c)
            ns = os.path.join(d, "nonstream.json")
            cases[c] = {"user": request_of(d)["messages"][-1]["content"], "sse": sse_path(d),
                        "nonstream": ns if os.path.isfile(ns) else os.path.join(d, "nonstream.raw"),
                        "raw": {m: open(os.path.join(d, m + ".raw.txt"), encoding="utf-8", newline="").read() for m in ("stream", "nonstream")}}
        key = "selftest-key-" + sha(out.encode())[:20]
        envf = os.path.join(out, "probe.env")
        open(envf, "w").write('VLLM_API_KEY="%s"\n' % key)
        rp = _Replay(cases, key, fail_detok={("S2_noarg_off", "nonstream")})
        port = rp.serve()
        po = os.path.join(out, "probe-e2e", "live1")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "probe.py"), po, ",".join(cs)],
                           env=dict(os.environ, PROBE_PORT=str(port), PROBE_ENVFILE=envf, PROBE_MODEL=model),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rp.httpd.shutdown()
        check(r.returncode == 0, "probe.py ran against the replay engine (exit %d %s)" % (r.returncode, r.stderr.decode()[-300:]))
        chats = [x for x in rp.log if x["path"] == "/v1/chat/completions"]
        check(len(chats) == 2 * len(cs) and all(x["body"].get("return_output_ids_in_sglext") is True for x in chats),
              "every send (%d) carried return_output_ids_in_sglext: true" % len(chats))
        check(all(x["auth_ok"] for x in rp.log), "every request carried the bearer from the env file")
        check(not any(x["path"] == "/flush_cache" for x in rp.log), "no /flush_cache without --flush (a live run never flushes)")
        det = [x for x in rp.log if x["path"] == "/detokenize"]
        check(len(det) == 2 * len(cs) and all(x["body"].get("skip_special_tokens") is False for x in det),
              "each send's ids went to /detokenize with skip_special_tokens false (%d calls)" % len(det))
        leaked = [f for dp, _, fs in os.walk(po) for f in fs if key in open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()]
        check(not leaked, "the key is in no recorded file")
        for c in cs:
            req = request_of(os.path.join(po, c))
            check(req is not None and req.get("return_output_ids_in_sglext") is True and "stream" not in req,
                  "%s request.json is the body as sent (return_output_ids_in_sglext, no stream flag)" % c)
            for m in ("stream", "nonstream"):
                rawp = os.path.join(po, c, m + ".raw.txt")
                meta = json.load(open(os.path.join(po, c, m + ".meta.json")))
                if (c, m) == ("S2_noarg_off", "nonstream"):
                    check(not os.path.isfile(rawp) and meta.get("raw", {}).get("detokenize_status") == 404,
                          "a refused /detokenize is recorded in the meta (status 404), no raw text written")
                else:
                    ok = os.path.isfile(rawp) and open(rawp, encoding="utf-8", newline="").read() == cases[c]["raw"][m]
                    ids = json.load(open(os.path.join(po, c, m + ".output_ids.json")))
                    check(ok and ids == rp.ids[(c, m)] and meta["raw"]["output_ids"] == 5,
                          "%s %s: output ids read from the %s and their text written byte for byte" % (
                              c, m, "sglext chunk" if m == "stream" else "response sglext"))
        print("== probe.py end to end, graded")
        _, prow = grade([po], da["src"], da["py"], da["tool_parser"], da["reasoning_parser"], da["tokenizer"], json_out=os.path.join(out, "probe-e2e"), quiet=True)
        for row in prow:
            if (row["case"], row["mode"]) == ("S2_noarg_off", "nonstream"):
                check(not row["ok"] and any("no raw output recorded" in w for w in row["why"]),
                      "the send whose raw text is missing FAILs the grade (%s)" % "; ".join(row["why"])[:140])
            else:
                check(row["ok"], "graded PASS: %s %s %s %s%s" % (row["case"], row["mode"], row["calls"], row["args_bytes"],
                                                                  "" if row["ok"] else " | " + "; ".join(row["why"])[:300]))
    print("SELF-TEST %s: %d/%d checks passed (out %s)" % ("PASS" if F[0] == 0 else "FAIL", N[0] - F[0], N[0], out))
    sys.exit(0 if F[0] == 0 else 1)


def main_report(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        raise SystemExit("usage: analyze.py report <dir holding fix/ control/ fix3/ fix4/>")
    base = os.path.abspath(argv[0])
    out = {}
    for ph in ("fix", "control", "fix3", "fix4"):
        pd = os.path.join(base, ph)
        if not os.path.isdir(pd):
            continue
        tokens = {}
        tp = os.path.join(pd, "tokens.json")
        if os.path.isfile(tp):
            tokens = json.load(open(tp))
        res = {}
        for case in sorted(os.listdir(pd)):
            d = os.path.join(pd, case)
            if os.path.isdir(d):
                res[case] = analyze_case(d, tokens)
                for k in ("timed", "output_ids"):   # the grade's fields: #2537's results.json keeps its own shape
                    (res[case].get("stream") or {}).pop(k, None)
        out[ph] = {"cases": res, "verdicts": verdicts(ph, res)}
    json.dump(out, open(os.path.join(base, "results.json"), "w"), indent=1, ensure_ascii=False)
    lines = ["| phase | case | first byte s | first tool_calls delta s | after reasoning s | last tool chunk s | max gap s | tool chunks | arg tokens | stream == non-stream (bytes) | finish (s/ns) | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ph, o in out.items():
        for case, r in o["cases"].items():
            s, c, tk = r.get("stream") or {}, r.get("compare") or {}, r.get("tokens") or {}
            vd = o["verdicts"].get(case, {}).get("verdict", "-")

            def f(x):
                return "-" if x is None else ("%.2f" % x)
            eq = "-" if not c else ("yes" if c.get("all_args_equal") else "NO") + " (%s/%s calls)" % (c.get("calls_stream"), c.get("calls_nonstream"))
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                ph, case, f(s.get("t_first_line")), f(s.get("t_first_tool_delta")), f(s.get("first_tool_delta_after_reasoning_s")),
                f(s.get("t_last_tool_chunk")), f(s.get("max_gap_between_tool_chunks_s")), s.get("tool_chunks", "-"),
                ",".join(str(x) for x in tk.get("arguments_tokens", [])) or "-", eq,
                "/".join(str(x) for x in (c.get("finish_reason") or [s.get("finish_reason"), None])), vd))
    open(os.path.join(base, "LIVE.generated.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for ph, o in out.items():
        for k, x in o["verdicts"].items():
            print(ph, k, x["verdict"], "; ".join(x["why"]))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse-worker"]:
        parse_worker(*sys.argv[2:6])
    elif sys.argv[1:2] == ["grade"]:
        main_grade(sys.argv[2:])
    elif sys.argv[1:2] == ["self-test"]:
        main_self_test(sys.argv[2:])
    elif sys.argv[1:2] == ["report"]:
        main_report(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
