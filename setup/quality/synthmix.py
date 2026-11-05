#!/usr/bin/env python3
"""synthmix.py -- synthetic agent-shaped mixed load for box-a (runs ON THE BOX against the engine). Python 3 stdlib only.

Every prompt is GENERATED here, deterministically from --seed: no corpus, no production data. Request kinds (weights
--mix, default agent=35,code=20,qa=20,long=10,json=10,intl=5):
  agent  multi-turn coding-agent session: a generated system prompt + 6 tools in OpenAI `tools` format + a history of
         prior assistant tool calls and tool results (read_file / run_shell / search_code / edit_file / todo_write ...).
         Each stream walks its own session turn by turn, so the prefix grows the way a real agent's does. Checks: every
         streamed tool call has a known name and JSON-parseable arguments (tool_parse_ok); no DSML marker in text.
  code   refactor / bug-fix / test-writing ask over a generated Python module (60-400 lines).
  qa     short question; the arithmetic ones carry an expected answer (answer_ok).
  long   a generated document of --long-min..--long-max tokens (default 8k-120k) with planted facts; the ask
         retrieves one (answer_ok).
  json   JSON-mode ask (response_format json_object) with required keys (json_ok).
  intl   non-English prompts (fr, de, es, ja, zh, ru, pt).
Thinking: a --thinking-off fraction of requests sends reasoning_effort "none" (thinking off), the rest
reasoning_effort "high" (box-c's official default); --effort-on picks another level. max_tokens is drawn per kind,
thinking off / on: agent 512-1536 / 1024-2048, code 1024-2048 / 1536-2048, qa and intl 128-512 / 512-1024,
long 256-768 / 768-1536, json 256-512 / 768-1024; all capped at --max-tokens-cap (2048), so a 1 h soak stays sane.

Modes:
  soak    --duration S --streams 1,2,4,8,16 [--phase-s 240] [--idle-prob 0.5 --idle-min 5 --idle-max 90]
          Phases cycle through the stream counts until --duration; each phase is closed-loop (C streams, each sends
          its next request when the last one finished, after an agent-like --think-max pause). Between phases, with
          --idle-prob, the load drains to ZERO requests in flight and stays idle 5-90 s (the idle-desync case).
          Requests are never cancelled: a phase ends once its in-flight requests finish.
  load    --streams N --duration S   constant closed-loop background load (greedy-under-load / quality-under-load);
          default temperature 1.0. Ends at --duration or when <out>/stop appears (any mode honours it).
  sample  --n K   print K generated request bodies (messages abbreviated) without sending anything.

Outputs (--out DIR): requests.jsonl (one line per request: phase, C, stream, kind, thinking, ttft, dur, max_gap,
prompt/cached/out tokens, decode tok/s, finish_reason, error, tool/json/qa/needle checks, output sha), phases.jsonl,
summary.json (counts, errors, finish reasons, p50/p99 TTFT, per-stream decode tok/s, per-phase aggregate tok/s, check
rates, hangs = a gap > --hang-gap-s between streamed chunks), started (first request sent), done.
Scoring: answer_ok is an exact whole-number match ('2401' does not pass '24010'; 7,203 / 7.203 / 7 203 read as 7203);
the sign is kept ('5' does not pass '-5') and integers compare exactly, so one past 2**53 must match every digit.
Every request that carried a check is scored and counted (answer_outcome / json_outcome / bug_outcome: ok, wrong,
empty, truncated, error; the *_ok rates are ok / all sent). Exit status: 1 when (errors + empty outputs + harness
errors) / requests sent > --max-error-rate (0.02) or nothing was sent (summary.json `gate`), else 0. An empty output
is a request with no content and no tool call, whatever its finish reason, except a max_tokens cut inside the
reasoning (finish_reason 'length' with reasoning streamed, thinking on), which is truncated, not empty.
--save-text keeps the generated output text in requests.jsonl (synthetic prompts only, so it may be committed).

Engine key: $SGLANG_API_KEY / $VLLM_API_KEY / $API_KEY, else VLLM_API_KEY= (SGLANG_API_KEY=, API_KEY=) in
/etc/environment. It is never written to any output.
"""
import argparse, hashlib, http.client, json, math, os, random, re, sys, threading, time, urllib.parse

MODEL = "DeepSeek-V4.1-Flash"
KEY_NAMES = ("SGLANG_API_KEY", "VLLM_API_KEY", "API_KEY")
DSML_RX = re.compile(r"</think>|<｜DSML｜|<\|DSML\||<｜tool▁call|<\|tool_call")


def load_key(path="/etc/environment"):
    for n in KEY_NAMES:
        if os.environ.get(n):
            return os.environ[n].strip()
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return ""
    for n in KEY_NAMES:
        for l in lines:
            l = l.strip()
            if l.startswith("export "):
                l = l[7:].strip()
            if l.startswith(n + "="):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def sub_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


# ================================================================ text generators (deterministic from an rng)

NOUNS = ("cache scheduler buffer request worker queue token parser index ledger router shard replica snapshot "
         "manifest pipeline channel segment checkpoint allocator kernel stream session cursor batch window bucket "
         "gateway tenant invoice order customer account policy report metric alert widget catalog inventory "
         "warehouse shipment vendor contract budget forecast").split()
VERBS = ("allocates flushes rebalances validates serializes merges evicts compacts retries throttles indexes "
         "rewrites schedules prefetches aggregates reconciles publishes archives replays rotates").split()
ADJS = ("stale primary transient sparse dense nested idle hot cold pending durable volatile sharded ordered lazy "
        "eager remote local cached partial").split()
PLACES = ("Lisbon Oslo Nairobi Quito Hanoi Tbilisi Perth Halifax Tromso Cusco Valletta Bergen Accra Kyoto "
          "Riga Tallinn Porto Dakar Leeds Graz").split()
PEOPLE = ("Ada Bruno Chen Dalia Emeka Farah Goran Hana Ines Jonas Kenji Lena Mateo Nadia Omar Priya Quinn "
          "Rosa Sven Tariq Uma Viktor Wen Yara Zoltan").split()
TOPICS = ("supply chains", "tidal energy", "medieval trade routes", "glacier dynamics", "compiler design",
          "urban beekeeping", "coral reef recovery", "railway signalling", "bread fermentation", "satellite orbits",
          "jazz harmony", "volcanic soils", "public libraries", "desalination", "migratory birds")


def ident(rng, k=2):
    return "_".join(rng.choice(NOUNS) if i else rng.choice(ADJS + NOUNS) for i in range(k))


def camel(rng):
    return "".join(w.capitalize() for w in (rng.choice(ADJS), rng.choice(NOUNS), rng.choice(NOUNS)))


def sentence(rng):
    t = rng.randrange(6)
    a, b, c = rng.choice(NOUNS), rng.choice(NOUNS), rng.choice(ADJS)
    if t == 0:
        return f"The {c} {a} {rng.choice(VERBS)} every {b} within {rng.randint(2, 900)} milliseconds."
    if t == 1:
        return (f"In {rng.choice(PLACES)}, {rng.choice(PEOPLE)} measured {rng.randint(10, 9999)} {a}s per hour "
                f"while studying {rng.choice(TOPICS)}.")
    if t == 2:
        return f"When a {b} becomes {c}, the {a} {rng.choice(VERBS)} it before the next {rng.choice(NOUNS)} arrives."
    if t == 3:
        return (f"Records from {rng.randint(1801, 2025)} show that {rng.choice(TOPICS)} depended on "
                f"{rng.randint(2, 97)} distinct {a} types.")
    if t == 4:
        return f"Each {a} keeps a {c} copy of its {b}, which is why recovery takes {rng.randint(1, 60)} minutes."
    return f"{rng.choice(PEOPLE)} argued that the {c} {b} matters less than the {a} it {rng.choice(VERBS)}."


def paragraph(rng, n=None):
    return " ".join(sentence(rng) for _ in range(n or rng.randint(3, 8)))


def gen_module(rng, lines):
    """A plausible Python module of about `lines` lines; returns (source, function names, the planted bug's fn)."""
    out = ['"""%s utilities for the %s service."""' % (camel(rng), ident(rng)), "import json", "import time",
           "from dataclasses import dataclass, field", ""]
    fns = []
    bug_fn = None
    while len(out) < lines:
        r = rng.random()
        if r < 0.25:
            cls = camel(rng)
            fields = [ident(rng, 1) + "_" + rng.choice(NOUNS) for _ in range(rng.randint(2, 5))]
            out += ["", "@dataclass", f"class {cls}:"]
            types = ("int = 0", "str = ''", "float = 0.0", "list = field(default_factory=list)")
            out += ["    %s: %s" % (f, rng.choice(types)) for f in fields]
            out += ["", "    def to_json(self):", "        return json.dumps(self.__dict__, sort_keys=True)", ""]
        else:
            fn = ident(rng)
            while fn in fns:
                fn = ident(rng, 3)
            fns.append(fn)
            arg, lim = rng.choice(NOUNS) + "s", rng.randint(3, 64)
            body = [f"def {fn}({arg}, limit={lim}):",
                    f'    """{rng.choice(VERBS).capitalize()} the {rng.choice(ADJS)} {arg} up to limit."""',
                    "    result = []", "    seen = set()"]
            if bug_fn is None and rng.random() < 0.3:
                bug_fn = fn
                body += [f"    for i in range(1, len({arg})):", f"        item = {arg}[i]"]
            else:
                body += [f"    for i in range(len({arg})):", f"        item = {arg}[i]"]
            body += ["        key = item.get('id') if isinstance(item, dict) else item",
                     "        if key in seen:", "            continue", "        seen.add(key)",
                     "        result.append(item)", "        if len(result) >= limit:", "            break",
                     f"    time.sleep({rng.choice(['0', '0.001', '0.01'])})", "    return result", ""]
            out += [""] + body
    return "\n".join(out) + "\n", fns, bug_fn


def gen_doc(rng, target_tokens, n_facts=6):
    """A long synthetic report of ~target_tokens tokens with planted facts; returns (text, facts list)."""
    facts = []
    for i in range(n_facts):
        facts.append((f"{rng.choice(ADJS)}-{rng.choice(NOUNS)}-{rng.randint(10, 99)}", str(rng.randint(100000, 999999))))
    target_chars = int(target_tokens * 4.3)
    parts, size, sec = [], 0, 0
    slots = sorted(rng.sample(range(1, max(n_facts + 2, target_chars // 2000)), n_facts))
    k = 0
    while size < target_chars:
        sec += 1
        p = f"## Section {sec}: {rng.choice(TOPICS).title()} and the {rng.choice(ADJS)} {rng.choice(NOUNS)}\n\n"
        p += "\n\n".join(paragraph(rng) for _ in range(rng.randint(2, 4)))
        if k < n_facts and sec >= slots[k]:
            p += f"\n\nNote for the audit: the registered code of unit {facts[k][0]} is {facts[k][1]}."
            k += 1
        parts.append(p)
        size += len(p) + 2
    while k < n_facts:
        parts.append(f"Note for the audit: the registered code of unit {facts[k][0]} is {facts[k][1]}.")
        k += 1
    return "\n\n".join(parts), facts


# ================================================================ request builders

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the workspace, optionally a line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repo root."},
            "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact, unique occurrence of old_str with new_str in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
            "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command in the repo root and return stdout+stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout_s": {"type": "integer", "default": 120}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Regex search over the repository; returns path:line:text rows.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "glob": {"type": "string", "description": "e.g. *.py"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "todo_write", "description": "Replace the task list shown to the user.",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {
            "type": "object", "properties": {
                "id": {"type": "string"}, "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
            "required": ["id", "content", "status"]}}}, "required": ["todos"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def agent_system(rng):
    proj = camel(rng)
    rules = [f"{i + 1}. {r}" for i, r in enumerate(rng.sample([
        "Read a file before editing it; never guess its contents.",
        "Keep edits minimal and in the surrounding style.",
        "Run the relevant tests after every change and report their actual output.",
        "Never run destructive commands (rm -rf, git push --force) without being asked.",
        "Prefer search_code over reading whole directories.",
        "Use todo_write to keep a short plan when a task has more than two steps.",
        "Do not add dependencies; the project uses the standard library only.",
        "Explain each change in one sentence when you are done.",
        "If a tool fails, read its error and adjust; never repeat the identical call.",
        "Keep secrets out of files and command lines.",
        "Answer in the language the user writes in.",
        "When unsure about intent, choose the reversible option and say so."], 8))]
    about = "\n\n".join(paragraph(rng) for _ in range(rng.randint(3, 10)))
    layout = "\n".join(f"  src/{ident(rng)}.py" for _ in range(rng.randint(6, 18)))
    return (f"You are a careful software engineering agent working in the repository `{proj}`.\n"
            f"Today you have shell access through tools. Rules:\n" + "\n".join(rules) +
            f"\n\nRepository layout:\n{layout}\n  tests/\n\nBackground notes from the maintainers:\n{about}\n")


class AgentSession:
    """A deterministic synthetic agent trajectory: turn t sends the system prompt, tools, the task and the first t
    steps of (assistant tool calls -> tool results), so successive turns share a growing prefix."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.system = agent_system(rng)
        self.files = [f"src/{ident(rng)}.py" for _ in range(4)]
        tgt = self.files[0]
        self.task = rng.choice([
            f"The function in {tgt} that deduplicates records drops the first element. Find and fix it, then run the tests.",
            f"Add type hints to every public function in {tgt} and make sure the test suite still passes.",
            f"Refactor {tgt}: split the dataclasses into a new module models.py and update the imports.",
            f"Our CI says tests/test_{ident(rng, 1)}.py is flaky. Investigate and make it deterministic.",
            f"Write unit tests for {tgt} covering the limit argument and empty input.",
        ])
        self.steps = []
        n = rng.randint(4, 14)
        call_no = 0
        for k in range(n):
            calls, results = [], []
            for _ in range(1 if rng.random() < 0.75 else 2):
                call_no += 1
                cid = f"call_{seed % 100000:05d}_{call_no:03d}"
                name, args, res = self._step(rng, k)
                calls.append({"id": cid, "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
                results.append({"role": "tool", "tool_call_id": cid, "content": res})
            text = rng.choice(["", "", "Let me look at that.", "Checking the tests next.",
                               "I'll update the plan first.", "The error points at the loop bounds."])
            step = [{"role": "assistant", "content": text, "tool_calls": calls}] + results
            if rng.random() < 0.15:
                step.append({"role": "user", "content": rng.choice([
                    "Also keep the public API unchanged.", "Please don't touch the tests directory.",
                    "Quick note: we run Python 3.11 in CI.", "Can you also check the README example?"])})
            self.steps.append(step)
        self.turn = 0

    def _step(self, rng, k):
        f = rng.choice(self.files)
        r = rng.random()
        if k == 0 or r < 0.3:
            src, _, _ = gen_module(rng, rng.randint(40, 220))
            return "read_file", {"path": f}, "\n".join(f"{i + 1:4d}  {l}" for i, l in enumerate(src.splitlines()))
        if r < 0.5:
            cmd = rng.choice(["python -m pytest -q tests", f"python -m pytest -q tests -k {ident(rng, 1)}",
                              "git status --short", f"python -c 'import {f[4:-3]}'"])
            npass, nfail = rng.randint(5, 80), rng.randint(0, 3)
            out = "\n".join(f"tests/test_{ident(rng, 1)}.py::{ident(rng)} {'PASSED' if rng.random() > 0.1 else 'FAILED'}"
                            for _ in range(rng.randint(3, 25)))
            if nfail:
                out += (f"\n\n___ test_{ident(rng)} ___\n    assert len(result) == {rng.randint(2, 9)}\n"
                        f"E   AssertionError: assert {rng.randint(1, 8)} == {rng.randint(2, 9)}\n")
            return "run_shell", {"command": cmd}, out + f"\n{nfail} failed, {npass} passed in {rng.uniform(0.2, 9):.2f}s"
        if r < 0.65:
            pat = ident(rng, 1)
            rows = "\n".join(f"{rng.choice(self.files)}:{rng.randint(1, 300)}:    {pat} = {ident(rng)}(items)"
                             for _ in range(rng.randint(1, 12)))
            return "search_code", {"pattern": pat, "glob": "*.py"}, rows
        if r < 0.8:
            old = f"for i in range(1, len(items)):"
            return "edit_file", {"path": f, "old_str": old, "new_str": "for i in range(len(items)):"}, "ok: 1 replacement"
        if r < 0.9:
            todos = [{"id": str(i + 1), "content": f"{rng.choice(VERBS)} the {ident(rng)}",
                      "status": rng.choice(["pending", "in_progress", "completed"])} for i in range(rng.randint(2, 6))]
            return "todo_write", {"todos": todos}, "todo list updated"
        return "list_dir", {"path": "src"}, "\n".join(self.files + [f"src/{ident(rng)}.py" for _ in range(6)])

    def next_messages(self):
        """Messages for the next turn, or None when the trajectory is exhausted."""
        if self.turn > len(self.steps):
            return None
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": self.task}]
        for st in self.steps[:self.turn]:
            msgs += st
        self.turn += 1
        return msgs


QA_TEXT = [
    "Explain how a B-tree keeps itself balanced on insert, with a small worked example.",
    "What are the trade-offs between optimistic and pessimistic locking in a database?",
    "Give three reasons a TCP connection might stall even though both hosts are up.",
    "What is the difference between a process and a thread? Keep it short.",
    "Why does floating point addition fail to be associative? Give an example.",
    "Summarize how a Bloom filter works and when it gives a wrong answer.",
    "What does idempotent mean for an HTTP API, and why does it matter for retries?",
    "Name four sorting algorithms and their worst-case complexity.",
]
INTL = [
    ("fr", "Explique en quelques phrases pourquoi le ciel est bleu.", None),
    ("fr", "Combien font {a} plus {b} ? Réponds uniquement par le nombre.", "add"),
    ("de", "Erkläre kurz den Unterschied zwischen einem Stack und einer Queue.", None),
    ("de", "Was ist {a} mal {b}? Antworte nur mit der Zahl.", "mul"),
    ("es", "¿Cuáles son las ventajas de la energía solar? Responde en tres puntos.", None),
    ("es", "¿Cuánto es {a} más {b}? Responde solo con el número.", "add"),
    ("ja", "東京と大阪の違いを三つ、簡単に説明してください。", None),
    ("ja", "{a} 足す {b} はいくつですか？数字だけで答えてください。", "add"),
    ("zh", "请用三句话解释什么是机器学习。", None),
    ("zh", "{a} 乘以 {b} 等于多少？只回答数字。", "mul"),
    ("ru", "Кратко объясни, как работает фотосинтез.", None),
    ("pt", "Quanto é {a} vezes {b}? Responda apenas com o número.", "mul"),
]


def caps(lo_hi_off, lo_hi_on, thinking, rng, cap):
    lo, hi = lo_hi_on if thinking else lo_hi_off
    return min(cap, rng.randint(lo, hi))


def build_request(kind, rng, thinking, a, session=None):
    """Returns (body without model/stream fields, check dict). Checks are scored on the finished output."""
    chk = {}
    if kind == "agent":
        msgs = session.next_messages()
        body = {"messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                "max_tokens": caps((512, 1536), (1024, 2048), thinking, rng, a.max_tokens_cap)}
        chk["tools"] = True
        chk["agent_turn"] = session.turn - 1
    elif kind == "code":
        src, fns, bug = gen_module(rng, rng.randint(60, 400))
        name = f"{ident(rng)}.py"
        asks = [f"Rename every function in this module to camelCase, add type hints, and return the full file.",
                f"Write pytest unit tests for `{rng.choice(fns) if fns else 'the dataclasses'}` covering edge cases.",
                "Split this module into models.py (dataclasses) and ops.py (functions). Show both files.",
                "Review this module: list the three most important problems and show a patch for each."]
        if bug:
            asks.append(f"One function drops the first element of its input. Find it, explain the bug, and show the fix.")
            chk["bug_fn"] = bug
        ask = rng.choice(asks)
        body = {"messages": [{"role": "user", "content": f"File `{name}`:\n```python\n{src}```\n\n{ask}"}],
                "max_tokens": caps((1024, 2048), (1536, 2048), thinking, rng, a.max_tokens_cap)}
    elif kind == "qa":
        if rng.random() < 0.5:
            x, y = rng.randint(100, 9999), rng.randint(100, 999)
            q = rng.choice([f"What is {x} * {y}? Reply with just the number.",
                            f"Compute {x} times {y}. Give only the result."])
            chk["expect"] = str(x * y)
        else:
            q = rng.choice(QA_TEXT)
        body = {"messages": [{"role": "user", "content": q}],
                "max_tokens": caps((128, 512), (512, 1024), thinking, rng, a.max_tokens_cap)}
    elif kind == "long":
        lo, hi = a.long_min, max(a.long_min, a.long_max)
        tgt = int(round(lo * (hi / lo) ** rng.random())) if lo > 0 else hi
        doc, facts = gen_doc(rng, tgt)
        unit, code = rng.choice(facts)
        if rng.random() < 0.7:
            q = f"In the report above, what is the registered code of unit {unit}? Answer with the code only."
            chk["expect"] = code
        else:
            q = "Summarize the report above in five bullet points and list every registered unit code it mentions."
        chk["doc_tokens_target"] = tgt
        body = {"messages": [{"role": "system", "content": "You answer questions about long documents precisely."},
                             {"role": "user", "content": f"Report:\n\n{doc}\n\n---\n{q}"}],
                "max_tokens": caps((256, 768), (768, 1536), thinking, rng, a.max_tokens_cap)}
    elif kind == "json":
        who, city, year = rng.choice(PEOPLE), rng.choice(PLACES), rng.randint(1950, 2024)
        topic = rng.choice(TOPICS)
        text = (f"{who} moved to {city} in {year} to work on {topic}. {sentence(rng)} "
                f"Colleagues describe {who} as {rng.choice(ADJS)} and {rng.choice(ADJS)}.")
        body = {"messages": [{"role": "system", "content": "Reply with a single JSON object and nothing else."},
                             {"role": "user", "content": "Extract a JSON object with keys name (string), city (string), "
                              f"year (integer) and tags (list of short strings) from this text:\n{text}"}],
                "response_format": {"type": "json_object"},
                "max_tokens": caps((256, 512), (768, 1024), thinking, rng, a.max_tokens_cap)}
        chk["json_keys"] = ["name", "city", "year", "tags"]
        chk["json_year"] = year
    elif kind == "intl":
        lang, tmpl, op = rng.choice(INTL)
        x, y = rng.randint(12, 999), rng.randint(12, 99)
        q = tmpl.format(a=x, b=y)
        if op:
            chk["expect"] = str(x + y if op == "add" else x * y)
        chk["lang"] = lang
        body = {"messages": [{"role": "user", "content": q}],
                "max_tokens": caps((128, 512), (512, 1024), thinking, rng, a.max_tokens_cap)}
    else:
        raise ValueError(kind)
    return body, chk


# ================================================================ client

class Client:
    def __init__(self, url, key, read_timeout):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.https = u.scheme == "https"
        self.prefix = u.path.rstrip("/")
        self.key, self.read_timeout = key, read_timeout

    def _conn(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.read_timeout)

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def post_json(self, path, body):
        c = self._conn()
        try:
            c.request("POST", self.prefix + path, json.dumps(body).encode(), self._headers())
            r = c.getresponse()
            data = r.read().decode("utf-8", "replace")
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {data[:300]}")
            return json.loads(data)
        finally:
            c.close()

    def stream_chat(self, body, hang_gap_s):
        """POST a streaming chat completion; returns a metrics dict (never raises)."""
        m = dict(ttft=None, dur=None, max_gap=0.0, out_tok=None, prompt_tok=None, cached_tok=None, chunks=0,
                 finish=None, err=None, http=None, text="", reasoning="", tool_calls=[])
        t0 = time.time()
        tf = tl = None
        calls = {}
        c = self._conn()
        try:
            c.request("POST", self.prefix + "/v1/chat/completions", json.dumps(body).encode(), self._headers())
            r = c.getresponse()
            m["http"] = r.status
            text, reas = [], []
            if r.status != 200:
                m["err"] = f"HTTP {r.status}: " + r.read().decode("utf-8", "replace")[:300]
            while m["http"] == 200:
                raw = r.readline()
                if not raw:
                    break
                l = raw.decode("utf-8", "replace").strip()
                if not l.startswith("data:"):
                    continue
                l = l[5:].strip()
                if l == "[DONE]":
                    break
                try:
                    ev = json.loads(l)
                except ValueError:
                    continue
                if ev.get("error") or ev.get("object") == "error":
                    m["err"] = ("stream error: " + json.dumps(ev.get("error", ev))[:300])
                    continue
                now = time.time()
                u = ev.get("usage")
                if u:
                    m["out_tok"] = u.get("completion_tokens", m["out_tok"])
                    m["prompt_tok"] = u.get("prompt_tokens", m["prompt_tok"])
                    det = u.get("prompt_tokens_details") or {}
                    if det.get("cached_tokens") is not None:
                        m["cached_tok"] = det["cached_tokens"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    got = False
                    if d.get("content"):
                        text.append(d["content"]); got = True
                    if d.get("reasoning_content"):
                        reas.append(d["reasoning_content"]); got = True
                    for tc in d.get("tool_calls") or []:
                        got = True
                        slot = calls.setdefault(tc.get("index", len(calls)), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        m["finish"] = ch["finish_reason"]
                    if got:
                        m["chunks"] += 1
                        if tf is None:
                            tf = now
                        else:
                            m["max_gap"] = max(m["max_gap"], now - tl)
                        tl = now
            m["text"], m["reasoning"] = "".join(text), "".join(reas)
        except Exception as e:  # noqa: BLE001 -- a failed request is a data point, not a crash
            m["err"] = repr(e)[:300]
        finally:
            c.close()
        m["tool_calls"] = [calls[k] for k in sorted(calls)]
        m["dur"] = round(time.time() - t0, 3)
        m["ttft"] = round(tf - t0, 3) if tf else None
        m["max_gap"] = round(m["max_gap"], 3)
        m["hang"] = m["max_gap"] > hang_gap_s
        n = m["out_tok"] if m["out_tok"] is not None else m["chunks"]
        m["tps"] = round((n - 1) / (tl - tf), 1) if tf and tl and tl > tf and n and n > 1 else None
        return m


# a digit group: 7,203 / 7.203 / 7 203 (one separator kind, groups of exactly 3) -> 7203
GROUPED_RX = re.compile(r"(?<![\d.,])\d{1,3}(?:([,.\u00a0\u202f\u2009 ])\d{3})(?:\1\d{3})*(?!\d)")
# a whole number: not glued to a latin letter, digit, underscore or decimal point on either side (CJK text around it
# is fine: 答案是7203。 holds 7203); a leading '-' (or U+2212 minus) is its sign unless glued to a digit, letter, '_' or
# '.' (2024-10-29 and x-5 hold no negative number)
NUM_RX = re.compile(r"(?<![A-Za-z_\d.])[-\u2212]?\d+(?:\.\d+)?(?![A-Za-z_\d])")

def parse_num(t):
    """A number string -> int when it has no '.' or exponent (exact at any size), else float."""
    t = t.strip().replace("\u2212", "-")
    return float(t) if any(c in t for c in ".eE") else int(t)

def numbers(txt):
    """Every whole number in txt (digit-group separators removed, sign kept): ints, or floats where it has a '.'."""
    norm = GROUPED_RX.sub(lambda m: re.sub(r"\D", "", m.group(0)), txt)
    out = []
    for t in NUM_RX.findall(norm):
        try:
            out.append(parse_num(t))
        except ValueError:
            pass
    return out

def num_equal(a, b):
    """Two ints compare exactly; anything with a float in it within a 1e-9 relative/absolute tolerance."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    except OverflowError:  # an int too big for a float equals no float we compare against
        return False

def answer_match(expect, txt):
    """Exact number match: `expect` (a number string, sign kept) equals one whole number in txt; '2401' does not pass
    '24010', '5' does not pass '-5', and an integer past 2**53 must match every digit."""
    e = parse_num(str(expect))
    return any(num_equal(e, n) for n in numbers(txt))

def outcome(m, ok):
    """Outcome of one check on a request that carried it; every sent request lands in exactly one bucket."""
    if m["err"]:
        return "error"
    if m["finish"] != "stop":
        return "truncated"  # finish length (or the stream ended without a finish reason)
    if not m["text"].strip():
        return "empty"
    return "ok" if ok() else "wrong"

OUTCOMES = ("ok", "wrong", "empty", "truncated", "error")

def score(chk, m):
    """Content checks on a finished request. A check the request carried is always scored (`*_outcome` is one of
    OUTCOMES; the boolean is True only for ok), so errors, empty and truncated outputs stay in the denominator."""
    s = {}
    txt = m["text"]
    # empty = the engine gave no answer: no content and no tool call. The one exception is a max_tokens cut inside the
    # reasoning (finish 'length' with reasoning streamed: thinking on, the budget spent thinking), which is truncated.
    # Any other finish (stop, abort, a stream that ended with no finish reason) with no answer is empty, reasoning or not.
    s["empty"] = (not m["err"] and not txt.strip() and not m["tool_calls"]
                  and not (m["finish"] == "length" and m["reasoning"].strip()))
    s["bad_marker"] = bool(DSML_RX.search(txt))
    if chk.get("tools") or m["tool_calls"]:
        ok = True
        for tc in m["tool_calls"]:
            if tc["name"] not in TOOL_NAMES:
                ok = False
                continue
            try:
                if not isinstance(json.loads(tc["arguments"] or "{}"), dict):
                    ok = False
            except ValueError:
                ok = False
        s["n_tool_calls"] = len(m["tool_calls"])
        s["tool_parse_ok"] = ok if m["tool_calls"] else None
        s["tool_names"] = [tc["name"] for tc in m["tool_calls"]]
    if "expect" in chk:
        s["answer_outcome"] = outcome(m, lambda: answer_match(chk["expect"], txt))
        s["answer_ok"] = s["answer_outcome"] == "ok"
    if "json_keys" in chk:
        o = None
        t = txt.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)
        try:
            o = json.loads(t)
        except ValueError:
            pass
        s["json_outcome"] = outcome(m, lambda: isinstance(o, dict) and all(k in o for k in chk["json_keys"]))
        s["json_ok"] = s["json_outcome"] == "ok"
        s["json_year_ok"] = s["json_ok"] and str(o.get("year")) == str(chk["json_year"])
    if "bug_fn" in chk:
        s["bug_outcome"] = outcome(m, lambda: re.search(r"(?<!\w)" + re.escape(chk["bug_fn"]) + r"(?!\w)", txt) is not None)
        s["bug_found"] = s["bug_outcome"] == "ok"
    return s


# ================================================================ runner

KINDS = ("agent", "code", "qa", "long", "json", "intl")


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


class Runner:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.client = Client(a.url, load_key() if not a.no_key else "", a.read_timeout)
        self.lock = threading.Lock()
        self.reqf = open(os.path.join(a.out, "requests.jsonl"), "a")
        self.phf = open(os.path.join(a.out, "phases.jsonl"), "a")
        self.records, self.phases = [], []
        self.sessions, self.session_no = {}, {}
        self.req_no = {}
        self.stop_path = os.path.join(a.out, "stop")
        self.started = False
        self.ev = threading.Event()
        self.n_inflight = 0
        self.harness_errors = 0
        self.t0 = time.time()
        w = dict(x.split("=") for x in a.mix.split(","))
        self.mix = [(k, float(w.get(k, 0))) for k in KINDS if float(w.get(k, 0)) > 0]
        if not self.mix:
            raise SystemExit("--mix selects no kind")

    def stopped(self):
        return self.ev.is_set() or os.path.exists(self.stop_path)

    def inflight(self):
        return self.n_inflight

    def session(self, sid):
        s = self.sessions.get(sid)
        if s is None or s.turn > len(s.steps):
            n = self.session_no.get(sid, 0)
            self.session_no[sid] = n + 1
            s = self.sessions[sid] = AgentSession(sub_seed(self.a.seed, "session", sid, n))
        return s

    def one(self, phase, C, sid):
        a = self.a
        n = self.req_no.get(sid, 0)
        self.req_no[sid] = n + 1
        rng = random.Random(sub_seed(a.seed, "req", sid, n))
        kind = rng.choices([k for k, _ in self.mix], [w for _, w in self.mix])[0]
        thinking = rng.random() >= a.thinking_off
        body, chk = build_request(kind, rng, thinking, a, self.session(sid) if kind == "agent" else None)
        body.update(model=a.model, stream=True, stream_options={"include_usage": True},
                    temperature=a.temperature, top_p=a.top_p,
                    reasoning_effort=a.effort_on if thinking else "none")
        if a.request_seed:
            body["seed"] = rng.randrange(1 << 31)
        if not self.started:
            self.started = True
            open(os.path.join(a.out, "started"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        t_send = time.time()
        with self.lock:
            self.n_inflight += 1
        try:
            m = self.client.stream_chat(body, a.hang_gap_s)
        finally:
            with self.lock:
                self.n_inflight -= 1
        s = score(chk, m)
        out_sha = hashlib.sha256((m["reasoning"] + "\0" + m["text"] + "\0" +
                                  json.dumps(m["tool_calls"], sort_keys=True)).encode()).hexdigest()[:16]
        rec = dict(t=time.strftime("%H:%M:%SZ", time.gmtime(t_send)), rel_s=round(t_send - self.t0, 2), phase=phase,
                   C=C, stream=sid, n=n, kind=kind, thinking=thinking, max_tokens=body["max_tokens"],
                   ttft=m["ttft"], dur=m["dur"], max_gap=m["max_gap"], hang=m["hang"], prompt_tok=m["prompt_tok"],
                   cached_tok=m["cached_tok"], out_tok=m["out_tok"], chunks=m["chunks"], tps=m["tps"],
                   finish=m["finish"], err=m["err"], http=m["http"], reasoning_chars=len(m["reasoning"]),
                   text_chars=len(m["text"]), out_sha=out_sha,
                   **{k: v for k, v in chk.items() if k not in ("tools", "json_keys", "expect", "bug_fn")}, **s)
        if a.save_text:
            rec.update(text=m["text"], reasoning=m["reasoning"], tool_calls=m["tool_calls"])
        with self.lock:
            self.records.append(rec)
            self.reqf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.reqf.flush()
        return rec

    def run_phase(self, phase, C, until, think_max):
        """Closed loop: C streams until `until` (or stop); in-flight requests always finish."""
        t = time.time()
        rng = random.Random(sub_seed(self.a.seed, "think", phase))
        pauses = [random.Random(rng.randrange(1 << 62)) for _ in range(C)]
        before = len(self.records)

        def worker(sid, prng):
            while time.time() < until and not self.stopped():
                try:
                    if self.one(phase, C, sid)["err"]:
                        time.sleep(self.a.error_backoff_s)  # never hammer a failing engine
                except Exception as e:  # noqa: BLE001 -- a harness bug must not silently end a stream
                    with self.lock:
                        self.harness_errors += 1
                    print(f"harness error stream {sid}: {e!r}", file=sys.stderr, flush=True)
                    time.sleep(1)
                p = prng.uniform(0, think_max)
                if p > 0 and time.time() + p < until:
                    time.sleep(p)
        th = [threading.Thread(target=worker, args=(s, pauses[s]), daemon=True) for s in range(C)]
        for x in th:
            x.start()
        for x in th:
            x.join()
        wall = time.time() - t
        recs = self.records[before:]
        ph = dict(phase=phase, C=C, start=time.strftime("%H:%M:%SZ", time.gmtime(t)), wall_s=round(wall, 1),
                  n=len(recs), errs=sum(1 for r in recs if r["err"]), hangs=sum(1 for r in recs if r["hang"]),
                  out_tok=sum(r["out_tok"] or 0 for r in recs),
                  agg_tok_s=round(sum(r["out_tok"] or 0 for r in recs) / wall, 1) if wall > 0 else None,
                  ttft_p50=pct([r["ttft"] for r in recs], 50), ttft_p99=pct([r["ttft"] for r in recs], 99),
                  tps_p50=pct([r["tps"] for r in recs], 50))
        self.phases.append(ph)
        self.phf.write(json.dumps(ph) + "\n")
        self.phf.flush()
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} phase {phase} C{C} {json.dumps(ph)}", flush=True)
        return ph

    def idle(self, phase, secs):
        ph = dict(phase=phase, C=0, idle_s=round(secs, 1), start=time.strftime("%H:%M:%SZ", time.gmtime()))
        print(f"{ph['start']} idle {secs:.1f}s", flush=True)
        end = time.time() + secs
        while time.time() < end and not self.stopped():
            time.sleep(min(1.0, max(0.0, end - time.time())))
        self.phases.append(ph)
        self.phf.write(json.dumps(ph) + "\n")
        self.phf.flush()


# ================================================================ greedy probe (greedy-under-load)

GREEDY_T = ["What is {a} * {b}? Show your work briefly.",
            "Write a Python function that returns the {a}th Fibonacci number modulo {b}.",
            "Summarize the causes of World War {w} in three sentences.",
            "Translate to French: 'The {n} cats slept on the warm windowsill until noon.'",
            "Explain in plain words what a hash table is, using {a} buckets as an example.",
            "A train leaves at {a}:00 and travels {b} km at 80 km/h. When does it arrive?",
            "List {n} prime numbers greater than {b}.",
            "Write a haiku about the number {a}."]


def greedy_prompts():
    """The 32 fixed prompts of box-a-mk2/greedy.py (10-28), so shas compare with that raw."""
    P = []
    for i in range(32):
        a, b = 17 + i * 7, 101 + i * 13
        P.append(GREEDY_T[i % 8].format(a=a, b=b, w=1 + (i % 2), n=["three", "five", "seven", "nine"][i // 8]))
    return P


def greedy_round(client, model, path, max_tokens, runner=None):
    """Sequential T=0 non-streaming requests, body and sha exactly as greedy.py: sha256(reasoning + NUL + content)."""
    with open(path, "w") as f:
        for i, p in enumerate(greedy_prompts()):
            body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
                    "temperature": 0, "seed": 0}
            t = time.time()
            inflight = None
            if runner is not None:
                inflight = runner.inflight()
            try:
                r = client.post_json("/v1/chat/completions", body)
                msg = r["choices"][0]["message"]
                txt = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
                row = {"i": i, "sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "text": txt,
                       "finish": r["choices"][0].get("finish_reason"), "dur": round(time.time() - t, 2),
                       "load_inflight": inflight}
            except Exception as e:  # noqa: BLE001
                row = {"i": i, "sha": None, "err": repr(e)[:300], "load_inflight": inflight}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


# ================================================================ summary

def summarize(a, runner, mode, t_start):
    R = runner.records

    def block(rs):
        n = len(rs)
        tc = [r for r in rs if r.get("tool_parse_ok") is not None]
        ans = [r for r in rs if r.get("answer_ok") is not None]
        js = [r for r in rs if r.get("json_ok") is not None]
        bug = [r for r in rs if r.get("bug_found") is not None]

        def outs(key):
            return {o: sum(1 for r in rs if r.get(key) == o) for o in OUTCOMES}
        fin = {}
        for r in rs:
            fin[str(r["finish"])] = fin.get(str(r["finish"]), 0) + 1
        return dict(
            n=n, errs=sum(1 for r in rs if r["err"]), hangs=sum(1 for r in rs if r["hang"]),
            max_gap=max((r["max_gap"] for r in rs), default=0), empty=sum(1 for r in rs if r.get("empty")),
            bad_marker=sum(1 for r in rs if r.get("bad_marker")), finish=fin,
            out_tok=sum(r["out_tok"] or 0 for r in rs), prompt_tok=sum(r["prompt_tok"] or 0 for r in rs),
            cached_tok=sum(r["cached_tok"] or 0 for r in rs),
            ttft_p50=pct([r["ttft"] for r in rs], 50), ttft_p99=pct([r["ttft"] for r in rs], 99),
            ttft_max=pct([r["ttft"] for r in rs], 100),
            tps_p10=pct([r["tps"] for r in rs], 10), tps_p50=pct([r["tps"] for r in rs], 50),
            tps_p90=pct([r["tps"] for r in rs], 90),
            tool_calls=sum(r.get("n_tool_calls") or 0 for r in rs),
            tool_parse_ok=f"{sum(1 for r in tc if r['tool_parse_ok'])}/{len(tc)}",
            answer_ok=f"{sum(1 for r in ans if r['answer_ok'])}/{len(ans)}", answer_outcomes=outs("answer_outcome"),
            json_ok=f"{sum(1 for r in js if r['json_ok'])}/{len(js)}", json_outcomes=outs("json_outcome"),
            bug_found=f"{sum(1 for r in bug if r['bug_found'])}/{len(bug)}", bug_outcomes=outs("bug_outcome"))
    errs = {}
    for r in R:
        if r["err"]:
            k = re.sub(r"\d+", "#", r["err"])[:120]
            errs[k] = errs.get(k, 0) + 1
    S = dict(mode=mode, seed=a.seed, model=a.model, url=a.url, temperature=a.temperature, top_p=a.top_p,
             thinking_off=a.thinking_off, effort_on=a.effort_on, mix=a.mix, streams=a.streams,
             duration_s=a.duration, wall_s=round(time.time() - t_start, 1),
             start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
             end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), stopped_early=runner.stopped(),
             all=block(R), by_C={str(c): block([r for r in R if r["C"] == c]) for c in sorted({r["C"] for r in R})},
             by_kind={k: block([r for r in R if r["kind"] == k]) for k in KINDS if any(r["kind"] == k for r in R)},
             by_thinking={str(t): block([r for r in R if r["thinking"] == t]) for t in (True, False)
                          if any(r["thinking"] == t for r in R)},
             per_stream_tps={str(s): pct([r["tps"] for r in R if r["stream"] == s], 50)
                             for s in sorted({r["stream"] for r in R})},
             error_kinds=errs, harness_errors=runner.harness_errors, phases=runner.phases)
    S["gate"] = gate(S, a.max_error_rate)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    return S


def gate(S, max_rate):
    """The exit gate: failed requests (errors + empty outputs + harness errors) over all sent must be <= max_rate, and
    at least one request must have been sent, so a 0 exit with '0 errors' is evidence."""
    A = S["all"]
    bad = A["errs"] + A["empty"] + S["harness_errors"]
    rate = bad / A["n"] if A["n"] else 1.0
    return dict(ok=A["n"] > 0 and rate <= max_rate, n=A["n"], errs=A["errs"], empty=A["empty"],
                harness_errors=S["harness_errors"], fail_rate=round(rate, 4), max_error_rate=max_rate)

# ================================================================ modes

def run_soak(a, runner):
    levels = [int(x) for x in a.streams.split(",")]
    rng = random.Random(sub_seed(a.seed, "plan"))  # idle-gap draws: deterministic from the seed
    t_end = runner.t0 + a.duration
    k = 0
    while time.time() < t_end - 5 and not runner.stopped():
        C = levels[k % len(levels)]
        runner.run_phase(k, C, min(time.time() + a.phase_s, t_end), a.think_max)
        k += 1
        if time.time() >= t_end - 5 or runner.stopped():
            break
        if rng.random() < a.idle_prob:
            gap = rng.uniform(a.idle_min, a.idle_max)
            runner.idle(k, min(gap, max(0.0, t_end - time.time())))


def run_load(a, runner):
    C = int(a.streams.split(",")[0])
    t_end = runner.t0 + a.duration
    gth = None
    if a.greedy_rounds > 0:
        gc = Client(a.url, runner.client.key, a.read_timeout)

        def g():
            end = time.time() + a.warmup_s
            while time.time() < end and not runner.stopped():
                time.sleep(0.5)
            for k in range(a.greedy_rounds):
                if runner.stopped():
                    break
                p = os.path.join(a.out, f"greedy-r{k + 1}.jsonl")
                print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} greedy round {k + 1} -> {p}", flush=True)
                greedy_round(gc, a.model, p, a.greedy_max_tokens, runner)
            if a.stop_after_greedy:
                runner.ev.set()
        gth = threading.Thread(target=g, daemon=True)
        gth.start()
    runner.run_phase(0, C, t_end, a.think_max)
    if gth is not None:
        gth.join()


def run_sample(a):
    r = Runner.__new__(Runner)  # the generator half of Runner only: no client, no output files
    r.a, r.sessions, r.session_no, r.req_no = a, {}, {}, {}
    w = dict(x.split("=") for x in a.mix.split(","))
    r.mix = [(k, float(w.get(k, 0))) for k in KINDS if float(w.get(k, 0)) > 0]
    for i in range(a.n):
        sid = i % max(1, a.sample_streams)
        n = r.req_no.get(sid, 0)
        r.req_no[sid] = n + 1
        rng = random.Random(sub_seed(a.seed, "req", sid, n))
        kind = rng.choices([k for k, _ in r.mix], [w for _, w in r.mix])[0]
        thinking = rng.random() >= a.thinking_off
        body, chk = build_request(kind, rng, thinking, a, Runner.session(r, sid) if kind == "agent" else None)
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in body["messages"])
        short = [{"role": m["role"], "chars": len(m.get("content") or ""), "head": (m.get("content") or "")[:80],
                  **({"tool_calls": [t["function"]["name"] for t in m["tool_calls"]]} if m.get("tool_calls") else {})}
                 for m in body["messages"]]
        print(json.dumps(dict(i=i, stream=sid, n=n, kind=kind, thinking=thinking, max_tokens=body["max_tokens"],
                              est_tokens=chars // 4, tools=bool(body.get("tools")),
                              response_format=body.get("response_format"), check=chk, messages=short),
                         ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("soak", "load", "sample"))
    ap.add_argument("--out", help="output dir (required for soak/load)")
    ap.add_argument("--url", default=os.environ.get("ENGINE_URL", "http://127.0.0.1:" + os.environ.get("ENGINE_PORT", "8002")))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--duration", type=float, default=3600, help="seconds (soak/load)")
    ap.add_argument("--streams", default=None, help="soak: list, default 1,2,4,8,16; load: N (default 8)")
    ap.add_argument("--phase-s", type=float, default=240)
    ap.add_argument("--idle-prob", type=float, default=0.5)
    ap.add_argument("--idle-min", type=float, default=5)
    ap.add_argument("--idle-max", type=float, default=90)
    ap.add_argument("--think-max", type=float, default=2.0, help="per-stream pause between requests, uniform 0..s")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--thinking-off", type=float, default=0.3, help="fraction sent with reasoning_effort none")
    ap.add_argument("--effort-on", default="high", help="reasoning_effort when thinking is on")
    ap.add_argument("--mix", default="agent=35,code=20,qa=20,long=10,json=10,intl=5")
    ap.add_argument("--long-min", type=int, default=8000)
    ap.add_argument("--long-max", type=int, default=120000)
    ap.add_argument("--max-tokens-cap", type=int, default=2048)
    ap.add_argument("--request-seed", action="store_true", help="send a per-request sampling seed")
    ap.add_argument("--read-timeout", type=float, default=900, help="socket read timeout, s")
    ap.add_argument("--hang-gap-s", type=float, default=30)
    ap.add_argument("--error-backoff-s", type=float, default=2.0, help="a stream's pause after a failed request")
    ap.add_argument("--max-error-rate", type=float, default=0.02,
                    help="exit 1 when (errors + empty outputs + harness errors) / requests sent exceeds this")
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--no-key", action="store_true", help="send no Authorization header")
    ap.add_argument("--greedy-rounds", type=int, default=0, help="load: rounds of the 32 greedy prompts under load")
    ap.add_argument("--greedy-max-tokens", type=int, default=256)
    ap.add_argument("--warmup-s", type=float, default=30, help="load: wait before the first greedy round")
    ap.add_argument("--stop-after-greedy", action="store_true", help="load: end the load after the greedy rounds")
    ap.add_argument("--n", type=int, default=12, help="sample: bodies to print")
    ap.add_argument("--sample-streams", type=int, default=1)
    a = ap.parse_args(argv)
    if a.streams is None:
        a.streams = "1,2,4,8,16" if a.mode == "soak" else "8"
    if a.mode == "sample":
        run_sample(a)
        return 0
    if not a.out:
        ap.error("--out is required")
    runner = Runner(a)
    t_start = runner.t0
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items()}, f, indent=1)
    try:
        (run_soak if a.mode == "soak" else run_load)(a, runner)
    except KeyboardInterrupt:
        runner.ev.set()
    S = summarize(a, runner, a.mode, t_start)
    open(os.path.join(a.out, "done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    print(json.dumps(dict(all=S["all"], by_C={k: dict(n=v["n"], errs=v["errs"], ttft_p50=v["ttft_p50"],
                                                       ttft_p99=v["ttft_p99"], tps_p50=v["tps_p50"])
                                               for k, v in S["by_C"].items()}, gate=S["gate"])), flush=True)
    if not S["gate"]["ok"]:
        print(f"GATE FAIL: {json.dumps(S['gate'])}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
