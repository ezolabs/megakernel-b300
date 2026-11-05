#!/usr/bin/env python3
"""probe.py -- the DSML tool-call probe, run ON the box against its engine at 127.0.0.1:8002: the tool-call gate of a
box-c change's live battery (10-updating.md step 3), run after the router recheck.

Usage (on the box): [PROBE_MODEL=<served id>] python3 -I probe.py <out dir> <case|group>[,<case|group>...]
                    [--modes stream,nonstream] [--flush] [--no-raw] [--repeat N] [--concurrency C]
  The window kit's `dsml` gate (box-gates.sh, started by gates.sh) copies this file to the box and runs
  `probe.py <gates dir>/dsml live` with PROBE_MODEL set to the box file's SERVED_MODEL; the Mac then grades the fetched
  records with `analyze.py grade <dir> --detector-tree <the served tree's commit>` (gates.sh grade). Stdlib only, so
  `python3 -I` is safe. PROBE_PORT and PROBE_ENVFILE point it at another engine and env file (the offline tests).

Every case is sent once with "stream": true and once with the IDENTICAL body but "stream": false (T=0), one at a
time -- except the REPEATS cases (T1_todo_nested and T1_todo_nested_strictfalse: Cosmos's real todo_write tool, a
nested `todos` array of objects, at prod sampling T 1.0 / top_p 1.0; the two arms differ only in the tool's `strict`,
unset vs `"strict": false`, see CASES below): each is sent N times per mode (470; --repeat N overrides it for a smoke run, and
analyze.py fails the case below 369), C sends at a time (--concurrency, default 8), each send a fresh sample recorded
under <out dir>/<case>/s<i>/ with the files below, the body once in <case>/request.json and the run in
<case>/repeat.json; its sends always carry return_output_ids_in_sglext (even with --no-raw) and are never flushed.
--flush POSTs /flush_cache before each send (identical cold prefill for both sends): only on an IDLE engine,
never under live traffic (the engine refuses a flush while requests run, and a flush empties prod's prefix cache);
the graded bar does not need it.

The graded bar (the user's definition): each streamed request's OWN raw output, re-parsed in one shot, must give the
arguments that request streamed, byte for byte -- it does not depend on the model generating the same text twice.
So every body carries "return_output_ids_in_sglext": true (unless --no-raw): the engine ends the stream with a
`choices: []` chunk whose sglext.output_ids holds the sampled token ids (the non-stream response carries them in its
own sglext), and after each send this script detokenizes them with the engine's own POST /detokenize
(skip_special_tokens false, as serving_chat sets it for a tool-bearing request) -- outside the timed send.
analyze.py (on the Mac) re-parses that text with the served tree's DSML detector and compares.

For each send it writes, under <out dir>/<case>/:
  request.json            the body exactly as sent (stream flag aside; no key: the bearer rides only in the header)
  stream.sse              the raw SSE bytes as received (chunked transfer decoded), byte for byte
  stream.times.tsv        per SSE line: arrival seconds since the request was sent, byte offset, length
  nonstream.json          the raw non-stream response body
  <mode>.meta.json        status, wall seconds, sent_at; and `raw`: the output-id count and the detokenize status
  <mode>.output_ids.json  the sampled token ids the engine returned (a JSON list)
  <mode>.raw.txt          those ids detokenized by the engine, UTF-8, exactly as /detokenize returned the text
The API key is read from /etc/environment and goes only into the Authorization header -- never into a file.
Analysis is analyze.py's (on the Mac); this script only sends and records. Its last stdout line counts what it
recorded, `DSML-PROBE RECORDED <ok>/<n> sends ...` (ok = HTTP 200 with the request's own raw output written); it exits
0 whatever that count is (the gate reads the records, analyze.py grades them).
"""
import http.client
import json
import os
import sys
import time

HOST, PORT = "127.0.0.1", int(os.environ.get("PROBE_PORT", "8002"))   # PROBE_PORT/PROBE_ENVFILE: the Mac self-test only
ENVFILE = os.environ.get("PROBE_ENVFILE", "/etc/environment")
MODEL = os.environ.get("PROBE_MODEL", "DeepSeek-V4.1-Flash-1017")   # gpu1018b STOCK; box-c: DeepSeek-V4.1-Flash

WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a text file to the user's workspace, replacing it if it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file."},
                "content": {"type": "string", "description": "The full text content of the file."},
            },
            "required": ["path", "content"],
        },
    },
}
GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
GET_TIME = {
    "type": "function",
    "function": {
        "name": "get_server_time",
        "description": "Return the server's current UTC time. Takes no arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
}
CFG = {
    "type": "function",
    "function": {
        "name": "cfg",
        "description": "Apply a configuration object to the deployment.",
        "parameters": {
            "type": "object",
            "properties": {
                "cfg": {"type": "object", "description": "The configuration object to apply."},
                "note": {"type": "string", "description": "A short note for the change log."},
            },
            "required": ["cfg", "note"],
        },
    },
}

NOTATKA = {
    "type": "function",
    "function": {
        "name": "zapisz_notatke",
        "description": "Zapisz notatkę w notatniku użytkownika.",
        "parameters": {
            "type": "object",
            "properties": {
                "tytuł": {"type": "string", "description": "Tytuł notatki."},
                "treść": {"type": "string", "description": "Pełna treść notatki."},
            },
            "required": ["tytuł", "treść"],
        },
    },
}
PL_MSGS = [{"role": "user", "content": "Zapisz notatkę o tytule „Zakupy na sobotę” i treści: chleb żytni, "
                                       "masło, źdźbła szczypiorku, łosoś wędzony, ćwierć kilo sera, gęś na niedzielę."}]

SYS_TOOL = ("You are a coding agent. You act only by calling the tools you are given. "
            "When the user asks for a file, call write_file exactly once with the complete file; "
            "never print the file as chat text and never abbreviate or elide any part of it.")

LONG_EN = ("Create the file docs/operations-handbook.md with write_file. It is an operations handbook for a small "
           "data-center team, written in English Markdown. It must have exactly 45 numbered sections "
           "(## 1. ... through ## 45.), each about a different operational topic (power, cooling, change "
           "management, on-call, backups, and so on). Every section has two paragraphs of at least 70 words "
           "each, followed by a bulleted checklist of 4 items. Write all 45 sections in full; do not summarise, "
           "skip or abbreviate any section.")
LONG_NONASCII = ("Create the file phrasebook/nl-zh.md with write_file. It is a Dutch-Chinese travel phrasebook in "
                 "Markdown with exactly 140 numbered entries. Each entry is one line of the form "
                 "`N. <Dutch sentence> — <Simplified Chinese translation> — <English gloss> <one fitting emoji>`. "
                 "The Dutch sentences must use accented letters often (é, è, ë, ï, ö, ü, e.g. één, café, ruïne, "
                 "coördinatie, privé, reünie, België, geïnteresseerd, crème brûlée), and every entry needs a "
                 "different emoji (🚆 🏨 🍽️ 🧀 🚲 🌧️ and so on). Write all 140 entries in full; do not skip, "
                 "summarise or abbreviate any.")
LONG_CODE = ("Create the file src/textkit.py with write_file. It is a Python 3 module of exactly 36 small text "
             "utility functions. Every function has full type hints, a docstring of at least four lines with a "
             "doctest example, input validation that raises ValueError with a message in double quotes "
             "(e.g. raise ValueError(\"width must be > 0\")), and at least one regular expression written as a raw "
             "string with backslashes (r\"\\s+\", r\"\\d{3}-\\d{4}\", r\"\\\\\" and the like). Use tabs nowhere; "
             "use \\n escapes inside some string literals. Write all 36 functions in full; do not skip or abbreviate.")
LONG_CSV = ("Create the file data/cities.csv with write_file. It is a CSV file with the header row "
            "id,city,country,population,latitude,longitude,founded,description and then exactly 320 data rows, one per "
            "real city worldwide, ids 1 to 320 in order. The description column is one full sentence of 15-25 words in "
            "double quotes. Write all 320 rows in full; do not skip, summarise or abbreviate any row.")

LONG_SQUARES = ("Create the file data/powers.csv with write_file. It is a CSV with the header row n,square,cube,parity "
                "and then one row for every integer n from 1 to 700 in order (parity is the word even or odd). This is "
                "mechanical: do not work the numbers out or plan in your reasoning, just call write_file with the "
                "complete file. Write all 700 rows in full; do not skip or abbreviate any row.")
LONG_GETALLEN = ("Create the file data/getallen.md with write_file. It has one line for every integer n from 1 to 300 in "
                 "order, each line exactly of the form `n | <n written out as one Dutch word> | <n in Chinese numerals> "
                 "| één café 🚲 ruïne ✓` -- for example `1 | één | 一 | één café 🚲 ruïne ✓` and "
                 "`23 | drieëntwintig | 二十三 | één café 🚲 ruïne ✓`. No header, no other text. Write all 300 lines in "
                 "full; do not skip or abbreviate any line.")
LONG_LABELS = ("Create the file src/labels.py with write_file. It contains, for every integer n from 1 to 220 in order, "
               "exactly this function with n written as three digits (shown here for n = 7):\n"
               "def label_007() -> str:\n    \"\"\"Return the label for item 007.\"\"\"\n"
               "    return \"Item \\\"007\\\" \\\\ ok\\n\"\n\n"
               "No imports, no other text. Write all 220 functions in full; do not skip or abbreviate any.")


def long_case(prompt, effort, max_tokens):
    return {
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": effort,
        "tools": [WRITE_FILE],
        "messages": [{"role": "system", "content": SYS_TOOL}, {"role": "user", "content": prompt}],
    }


CASES = {
    # (2) long calls, thinking OFF (three) and once at high
    "L1_en_off": long_case(LONG_EN, "none", 16384),
    "L2_nonascii_off": long_case(LONG_NONASCII, "none", 16384),
    "L3_code_off": long_case(LONG_CODE, "none", 16384),
    "L4_nonascii_high": long_case(LONG_NONASCII, "high", 32768),
    # the thinking-high long call, run 3: a mechanical file, so the reasoning stays short (L4 reasoned past 32k)
    "L6_squares_high": long_case(LONG_SQUARES, "high", 65536),
    # run 4: mechanical (low-entropy) long calls, so T=0 decoding repeats across sends -- runs 2-3 showed this engine's
    # free-text generations diverge between identical sends (stream vs stream2 too), on the fix and on 7b0a2e8d93 alike
    "M1_powers_off": long_case(LONG_SQUARES, "none", 32768),
    "M2_getallen_off": long_case(LONG_GETALLEN, "none", 32768),
    "M3_labels_off": long_case(LONG_LABELS, "none", 32768),
    # a fourth thinking-off long call, in case one of L1-L3 comes in under 5k argument tokens
    "L5_csv_off": long_case(LONG_CSV, "none", 16384),
    # (3) a short call and a no-argument call
    "S1_short_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none",
        "tools": [GET_WEATHER, GET_TIME],
        "messages": [{"role": "user", "content": "What's the weather in Utrecht right now, in celsius? Use the tool."}],
    },
    "S2_noarg_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none",
        "tools": [GET_WEATHER, GET_TIME],
        "messages": [{"role": "user", "content": "What time is it on the server? Call get_server_time to find out."}],
    },
    # (4) a call to a tool the request does not declare
    "U1_undeclared_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none",
        "tools": [GET_WEATHER],
        "messages": [
            {"role": "system", "content": "You act only by calling tools. Besides the listed tools you also have a "
                                          "tool named delete_records(table: string, older_than_days: integer); "
                                          "call it like any other tool."},
            {"role": "user", "content": "Call delete_records with table \"audit_log\" and older_than_days 90. "
                                        "Do it now by calling the tool; do not answer in text."},
        ],
    },
    # (6) the user's Polish case: declared parameter names tytuł / treść, tool_choice auto AND required
    "P1_pl_auto_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none",
        "tools": [NOTATKA], "tool_choice": "auto", "messages": PL_MSGS,
    },
    "P2_pl_required_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none",
        "tools": [NOTATKA], "tool_choice": "required", "messages": PL_MSGS,
    },
    # (5) first parameter non-string, an object whose first key equals the parameter name (#2536 C2)
    "C1_cfg_off": {
        "model": MODEL, "temperature": 0, "max_tokens": 4096, "reasoning_effort": "none",
        "tools": [CFG],
        "messages": [
            {"role": "system", "content": "You act only by calling tools."},
            {"role": "user", "content": (
                "Call cfg. Its first argument cfg must be exactly this JSON object, keys in this order: "
                "{\"cfg\": \"primary\", \"region\": \"eu-west-1\", \"retries\": 3, \"timeout_s\": 45, "
                "\"replicas\": 6, \"endpoints\": [\"https://a.example.net/v1\", \"https://b.example.net/v1\", "
                "\"https://c.example.net/v1\", \"https://d.example.net/v1\", \"https://e.example.net/v1\", "
                "\"https://f.example.net/v1\"], \"limits\": {\"cpu\": \"4000m\", \"memory\": \"16Gi\", "
                "\"burst\": true}, \"labels\": {\"team\": \"infra\", \"tier\": \"gold\", \"owner\": \"ops-oncall\"}}. "
                "Its second argument note is \"roll out primary config to eu-west-1\".")},
        ],
    },
}


# Declared parameter names the 0046 grammar must keep whole (USER 12:35Z, #2539): one non-strict tool per name, a
# short value, tool_choice required (forced) AND auto. analyze.py grades each returned call's keys as exactly the
# declared name.
DECLARED_NAMES = ["user-id", "first name", "$ref", "1st", "名前"]


def name_case(name, choice):
    tool = {"type": "function", "function": {
        "name": "zapisz_pole", "description": "Save one field value.",
        "parameters": {"type": "object", "properties": {name: {"type": "string", "description": "The value."}},
                       "required": [name]}}}
    return {"model": MODEL, "temperature": 0, "max_tokens": 512, "reasoning_effort": "none", "tools": [tool],
            "tool_choice": choice,
            "messages": [{"role": "system", "content": "You act only by calling tools."},
                         {"role": "user", "content": 'Call zapisz_pole with its parameter "%s" set to "abc123".' % name}]}


for _i, _n in enumerate(DECLARED_NAMES):
    for _ch in ("required", "auto"):
        CASES["N%d_%s" % (_i + 1, _ch)] = name_case(_n, _ch)

# Cosmos's real todo_write tool, as Cosmos puts it on the wire (the internal ops repo `src/agent/tools.ts`, the
# `todo_write` registration: a `todos` union of an entry array -- every entry field optional, most `.nullish()`, so
# `["string", "null"]` -- and a JSON-encoded string), copied from the tool list a Cosmos agent receives: the MCP name
# omni namespaces it to, the description and the input schema byte for byte. The nested array of objects is where
# box-c served a padded/garbled entry key in production (`"id: 20, "`), the shape box-e's first roll served for a
# spaced parameter name (`{"first: "abc123"}`).
_NULLISH = ["string", "null"]
_IDS = {"type": ["string", "number"]}
TODO_ENTRY = {
    "additionalProperties": {},
    "properties": {
        "activeForm": {"description": "Its -ing form, shown while it runs.", "type": _NULLISH},
        "blocked": {"description": "Shorthand for status: blocked.", "type": "boolean"},
        "blockedOn": {"description": "With blocked: what it waits on.", "type": _NULLISH},
        "cancelReason": {"description": "With cancelled: why. A request-derived item closes only completed, blocked or "
                                         "cancelled with this.", "type": _NULLISH},
        "cancelled": {"description": "Shorthand for status: cancelled.", "type": "boolean"},
        "checks": {"description": "Extra to the recipe defaults (an omitted default returns). command: `run` (`{files}` = "
                                   "owns), `stage`, `timeout`, `requires` (paths, else unrunnable). file: `path`, "
                                   "`expect`, `text`. judge: `prompt` (criteria to refute). metric: `run` printing JSON, "
                                   "`metricPath` (JSON pointer), `baseline`, `tolerance`, `direction` (the worse way), "
                                   "`unit`.",
                   "items": {"additionalProperties": {}, "properties": {"kind": {"enum": ["command", "file", "judge", "metric"],
                                                                             "type": "string"}},
                             "required": ["kind"], "type": "object"},
                   "type": "array"},
        "content": {"description": "The step, imperative and specific. Omit with `id` to keep it.", "type": _NULLISH},
        "deep": {"description": "Subtle work: a lane on `auto` effort sends it at xhigh. `false` clears it.", "type": "boolean"},
        "deferred": {"description": "Shorthand: cancelled as deferred — work outside this request, kept under Deferred; the "
                                     "why. A by-id `pending` revives it.", "type": "string"},
        "deps": {"description": "Ids (`#n`) it waits on, finish-to-start. On `#n` it REPLACES the whole list: send every "
                                 "dep that stays; `[]` clears.", "items": _IDS, "type": "array"},
        "discoveredFrom": {"description": "The `#n` you found it under: ancestry and dispatch ownership, never a "
                                           "dependency. `null` with `id` clears.", "type": ["string", "number", "null"]},
        "id": {"description": "An existing `#n`.", "type": ["string", "number", "null"]},
        "kind": {"description": "research | plan | build | verify; `build` when it has checks, owns or deps. A `plan` item "
                                 "with no build parts goes to a planner.", "type": _NULLISH},
        "later": {"description": "ORDER, not skip: built after everything else ships; the reason (\"\" clears).",
                  "type": "string"},
        "noBatch": {"description": "Dispatch alone, never batched (a long benchmark). `false` clears.", "type": "boolean"},
        "notes": {"description": "Markdown: findings, the file that held the answer, what is left. Replaces the whole text, "
                                  "no copy kept (a cut under half is named); omit keeps, \"\" clears. A worker's notes go "
                                  "under notes its manager wrote. Out with a running worker: relayed to it, the item "
                                  "unchanged; \"\" is refused.", "type": _NULLISH},
        "owns": {"description": "Path globs its worker is fenced to, its checks run over them. On `#n` it ADDS, held or "
                                 "not; `[]` clears (to narrow: clear, then name what stays).",
                 "items": {"type": "string"}, "type": "array"},
        "reads": {"description": "Unwritten files whose NEW content it builds against (one only consulted goes in notes). "
                                  "One another item owns derives the ordering edge: no `deps` — only from a writer "
                                  "minted in an EARLIER write; a same-write or later writer is shown and never orders: "
                                  "write it first, or use `deps`.", "items": {"type": "string"}, "type": "array"},
        "reason": {"description": "With `verify: 'skip'`: why.", "type": "string"},
        "rootHeld": {"description": "The root runs it itself; never dispatched.", "type": "boolean"},
        "shared": {"description": "Files siblings conflict on (a lockfile): two naming one never run together.",
                   "items": {"type": "string"}, "type": "array"},
        "skipStages": {"description": "Stages (lint | test | e2e | boot) it cannot run: their defaults drop (`e2e` drops "
                                       "boot too). `[]` clears.", "items": {"type": "string"}, "type": "array"},
        "status": {"description": "pending (default) | in_progress | completed | blocked | cancelled; unchanged with `id`.",
                   "type": _NULLISH},
        "supersedes": {"description": "Ids it replaces: cancelled, their deps and dependents moved here.", "items": _IDS,
                       "type": "array"},
        "tests": {"description": "Its test files.", "items": {"type": "string"}, "type": "array"},
        "verify": {"description": "`skip` + `reason`: `completed` closes it on your word.", "enum": ["skip"], "type": "string"},
    },
    "type": "object",
}
TODO_WRITE = {
    "type": "function",
    "function": {
        "name": "mcp__cosmos__todo_write",
        "description": (
            "This chat's Tasks ledger: its memory across compaction, respawn and restart. Before the first change to "
            "the tree, put ONE item in_progress (one per scope; the first gets the user's prompt as its notes). `#n` "
            "patches that item; an entry with no id is a new item. A research ask is work: the question is the item, "
            "each angle a `research` item under it. File parts under the step that found them (`discoveredFrom`), "
            "cancel a dead angle with a `cancelReason`, re-plan with `supersedes`, set `deps` only where order "
            "matters. A DISCOVERY inside your owns is done now; one outside them is a Left undone line naming its "
            "files, which the harness files at the landing; a nit is a report line. Tick each item as it lands; write "
            "when the work's SHAPE changes, never per call. Keep a ✎ item's `#n`. An item out with a worker takes "
            "notes and a WIDENING of its `owns`, `reads` or `tests` (added, never removed); anything else goes to its "
            "worker by `mcp__cosmos__delegate_send`, except a pending part you filed under it before it went out and "
            "its worker never touched, which you may still re-parent or cancel. What `tests`, `checks` and `verify` "
            "do follows this chat's Verify mode."),
        "parameters": {
            "properties": {
                "patch": {"description": (
                    "Default with any `id`, a `discoveredFrom`/`deps`/`supersedes` on the ledger, in a worker’s scope, "
                    "or past 12 open items; a list that would drop an item in hand, a proposal or a dep edge also "
                    "applies as a patch. `false`: this list IS the ledger, omitted open items dropped; refused if one "
                    "has a worker or >max(3, ¼ of open)."), "type": "boolean"},
                "todos": {"anyOf": [{"items": TODO_ENTRY, "type": "array"}, {"type": "string"}],
                          "description": "The checklist, in order."},
            },
            "required": ["todos"],
            "type": "object",
        },
    },
}


def _todo_call(cid, todos):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": cid, "type": "function",
        "function": {"name": "mcp__cosmos__todo_write", "arguments": json.dumps({"todos": todos}, ensure_ascii=False)}}]}


# An agent turn three ledger writes in: two earlier todo_write calls in the history (a new list, then a by-id patch
# with numeric ids, the shape prod served as `"id: 20, "`), then a user update that needs one write of a nested array
# mixing patches (id + the fields they change) and new entries (content, discoveredFrom, owns, kind).
TODO_MSGS = [
    {"role": "system", "content": (
        "You are a coding agent working in a repository. You keep your work on a shared ledger with "
        "mcp__cosmos__todo_write: an entry with `id` patches that item and carries only the fields it changes; an "
        "entry with no `id` is a new item. Record every change the user reports in ONE todo_write call.")},
    {"role": "user", "content": (
        "Port the billing service's retry logic from the legacy Ruby worker to src/billing/retry.ts: read the old "
        "worker, write the new module with tests, then wire it into the queue consumer. Put it on the ledger first.")},
    _todo_call("call_01", [
        {"content": "Read the legacy Ruby retry worker (legacy/workers/billing_retry.rb)", "status": "in_progress",
         "activeForm": "Reading the legacy retry worker"},
        {"content": "Write src/billing/retry.ts with exponential backoff and jitter", "status": "pending",
         "activeForm": "Writing the retry module", "owns": ["src/billing/retry.ts"]},
        {"content": "Test the retry module in src/billing/retry.test.ts", "status": "pending",
         "activeForm": "Testing the retry module", "owns": ["src/billing/retry.test.ts"], "deps": ["#19"]},
        {"content": "Wire the retry module into the queue consumer", "status": "pending",
         "activeForm": "Wiring the retry module", "owns": ["src/queue/consumer.ts"], "deps": ["#19", "#20"]}]),
    {"role": "tool", "tool_call_id": "call_01", "content": (
        "Ledger: #18 [in_progress] Read the legacy Ruby retry worker · #19 [pending] Write src/billing/retry.ts with "
        "exponential backoff and jitter · #20 [pending] Test the retry module in src/billing/retry.test.ts · #21 "
        "[pending] Wire the retry module into the queue consumer")},
    _todo_call("call_02", [
        {"id": 18, "status": "completed", "notes": (
            "Backoff is 2^n seconds capped at 300 s, 6 attempts; a 4xx other than 429 is never retried "
            "(legacy/workers/billing_retry.rb L22-61).")},
        {"id": 19, "status": "in_progress"}]),
    {"role": "tool", "tool_call_id": "call_02", "content": "Patched #18 -> completed, #19 -> in_progress."},
    {"role": "user", "content": (
        "Update from the review: #19 is done -- the module is src/billing/retry.ts L1-142; note that its jitter is "
        "full jitter, not equal jitter. #20 starts now, and it already found two gaps: file both under #20, one for "
        "honouring the 429 Retry-After header (owns src/billing/retry.ts) and one for a clock-skew test helper "
        "(owns src/test-support/clock.ts). #21 is blocked on the queue team's consumer refactor. Also add a research "
        "item: does the payments API document idempotency keys for retried charges? Record all of it in one ledger "
        "write now.")},
]

CASES["T1_todo_nested"] = {
    # prod sampling (DeepSeek's recommended T 1.0 / top_p 1.0, what box-c's callers send) and box-c's default thinking
    "model": MODEL, "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384, "reasoning_effort": "high",
    "tools": [TODO_WRITE], "tool_choice": "auto", "messages": TODO_MSGS,
}
# Two arms of the one case, the same schema, conversation and sampling; only the tool's `strict` differs.
#   T1_todo_nested             no `strict`, straight to the engine: it measures the TREE'S DEFAULT for an unset strict.
#                              On a tree carrying the fork's strict-default fix (#901, `prod-tool-strict-default`: an
#                              unset strict is not strict) it must PASS; on any tree without it upstream sglang #46133
#                              grammar-constrains an unset strict, and this schema then fails by design: ~99.7 %
#                              defective on box-a's 819756d4 and the megakernel, 47.8 % on box-c's trim tree (#902).
#                              analyze.py grade --expect-unset pass|fail says which build it is.
#   T1_todo_nested_strictfalse `"strict": false` on the tool, what prod traffic carries: router release 12+ posts it for
#                              every tool its caller left unset. It must PASS on every build (#902 on box-c's trim tree:
#                              8/2986 pooled, ~0.27 %, the pre-box-e 0.32 %).
TODO_WRITE_STRICTFALSE = {"type": "function", "function": dict(TODO_WRITE["function"], strict=False)}
CASES["T1_todo_nested_strictfalse"] = dict(CASES["T1_todo_nested"], tools=[TODO_WRITE_STRICTFALSE])

# Cases sent many times, each send a fresh sample: a defect at a few percent of calls is invisible to one T=0 send.
# n per mode; --repeat N overrides it (a smoke run); analyze.py fails the case below TODO_MIN_SENDS (369) per mode.
# analyze.py sizes each count's bar from the calls it actually carries (analyze.todo_max_bad / todo_power), at <= 0.1 %
# false-fail per count. Each side's defective calls: analyze.TODO_BAR, a CATASTROPHE gate, p0 = the probe's pooled
# healthy arms less the config under question (48 of 6588 calls, 0.73 %): k <= 10 of 470 calls, >= 99 % power at a 5 %
# defect across both modes (86 % at 2.5 %; a 2 % defect passes one mode 66 % of the time), and a count of fewer than
# 234 calls (209 and 210 aside) FAILs the mode as too few for the gate. The raw-vs-parsed count: its own null,
# analyze.TODO_DIFFER_BAR (1 differing send of 7520 in the same arms, 0.013 %): k <= 2 of 470, so a parser-only
# defect fails at 3. 470 sends keep every bar's power down to the 90 % with-a-call floor (423 calls: k 10, 99.998 %),
# so a run with a few no-call sends still grades.
REPEATS = {"T1_todo_nested": 470, "T1_todo_nested_strictfalse": 470}

# `live`: the tool-call gate of a box-c change's live battery, run after the recheck (the window kit's `dsml` gate): the
# mechanical long calls (their bytes repeat, so the info row stays readable), the restatement shape, the Polish and
# declared-name cases, short/no-arg: 18 cases, each sent once streamed and once non-stream (36 sends); and the standing
# nested-array case in its two arms, T1_todo_nested (strict unset) and T1_todo_nested_strictfalse, 470 sends per mode
# each at prod sampling (4 x 470 = 1,880 sends). 1,916 sends.
GROUPS = {"live": ["M1_powers_off", "M2_getallen_off", "M3_labels_off", "C1_cfg_off", "P1_pl_auto_off",
                   "P2_pl_required_off", "S1_short_off", "S2_noarg_off"] + ["N%d_%s" % (i + 1, ch) for i in range(len(DECLARED_NAMES)) for ch in ("required", "auto")]
                  + ["T1_todo_nested", "T1_todo_nested_strictfalse"]}


def api_key():
    for line in open(ENVFILE, encoding="utf-8", errors="replace"):
        if line.startswith("VLLM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no VLLM_API_KEY in %s" % ENVFILE)


def post(path, body, key, timeout=900):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else b""
    c.putrequest("POST", path)
    c.putheader("Content-Type", "application/json")
    c.putheader("Authorization", "Bearer " + key)
    c.putheader("Content-Length", str(len(data)))
    c.endheaders()
    t0 = time.monotonic()
    c.send(data)
    return c, c.getresponse(), t0


def flush(key, log):
    """POST /flush_cache until it answers 200: the engine refuses (400) while the previous request is still being
    finished server-side, which is usually the case right after the client has read the last byte. A refused flush
    leaves the earlier send's prefix in the radix cache, so the next send's prefill differs (run 2's lesson)."""
    st = None
    for i in range(1, 61):
        try:
            c, r, _ = post("/flush_cache", None, key, timeout=60)
            txt = r.read().decode("utf-8", "replace")[:80]
            c.close()
            st = r.status
        except Exception as e:  # recorded, not fatal
            txt, st = repr(e), None
        if st == 200:
            break
        time.sleep(0.5)
    log.write("flush_cache %s after %d tries %s\n" % (st, i, txt.replace("\n", " ")))
    return st


def run_stream(body, d, key, name="stream"):
    b = dict(body, stream=True)
    c, r, t0 = post("/v1/chat/completions", b, key)
    status = r.status
    with open(os.path.join(d, name + ".sse"), "wb") as raw, open(os.path.join(d, name + ".times.tsv"), "w") as tt:
        tt.write("t_s\toffset\tlen\n")
        off = 0
        while True:
            line = r.readline()
            if not line:
                break
            t = time.monotonic() - t0
            raw.write(line)
            tt.write("%.4f\t%d\t%d\n" % (t, off, len(line)))
            off += len(line)
    wall = time.monotonic() - t0
    c.close()
    return {"status": status, "wall_s": round(wall, 4), "bytes": off}


def run_nonstream(body, d, key):
    b = dict(body, stream=False)
    c, r, t0 = post("/v1/chat/completions", b, key)
    data = r.read()
    wall = time.monotonic() - t0
    c.close()
    with open(os.path.join(d, "nonstream.json"), "wb") as f:
        f.write(data)
    meta = {"status": r.status, "wall_s": round(wall, 4), "bytes": len(data)}
    with open(os.path.join(d, "nonstream.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def sse_output_ids(path):
    """The sampled token ids of a recorded stream: the last `data:` chunk carrying sglext.output_ids (serving_chat
    sends it once, after the finish chunk, with choices []). Returns the ids of choice 0, or None."""
    ids = None
    for line in open(path, "rb"):
        line = line.rstrip(b"\r\n")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]" or b'"output_ids"' not in payload:
            continue
        try:
            ch = json.loads(payload.decode("utf-8"))
        except ValueError:
            continue
        oi = (ch.get("sglext") or {}).get("output_ids")
        if isinstance(oi, list) and oi and isinstance(oi[0], list):
            ids = oi[0]
    return ids


def nonstream_output_ids(path):
    try:
        j = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    oi = (j.get("sglext") or {}).get("output_ids") if isinstance(j, dict) else None
    return oi[0] if isinstance(oi, list) and oi and isinstance(oi[0], list) else None


def record_raw(mode, d, key):
    """Write <mode>.output_ids.json and <mode>.raw.txt for one finished send; returns the meta's `raw` entry."""
    src = os.path.join(d, mode + ".sse") if mode.startswith("stream") else os.path.join(d, "nonstream.json")
    ids = sse_output_ids(src) if mode.startswith("stream") else nonstream_output_ids(src)
    if ids is None:
        return {"output_ids": None, "error": "no sglext.output_ids in the response"}
    with open(os.path.join(d, mode + ".output_ids.json"), "w") as f:
        json.dump(ids, f)
    try:
        c, r, _ = post("/detokenize", {"model": MODEL, "tokens": ids, "skip_special_tokens": False}, key, timeout=120)
        data = r.read()
        c.close()
        status = r.status
    except Exception as e:  # recorded, not fatal: analyze.py grades a missing raw text as a FAIL
        return {"output_ids": len(ids), "detokenize_status": None, "error": repr(e)}
    if status != 200:
        return {"output_ids": len(ids), "detokenize_status": status, "error": data[:300].decode("utf-8", "replace")}
    try:
        text = json.loads(data.decode("utf-8"))["text"]
    except (ValueError, KeyError) as e:
        return {"output_ids": len(ids), "detokenize_status": status, "error": "bad /detokenize body: %r" % e}
    if not isinstance(text, str):
        return {"output_ids": len(ids), "detokenize_status": status, "error": "/detokenize text is not a string"}
    with open(os.path.join(d, mode + ".raw.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return {"output_ids": len(ids), "detokenize_status": status, "raw_chars": len(text),
            "skip_special_tokens": False}


def send_one(label, body, d, m, key, log, want_raw):
    """One send of <body> in mode <m>, recorded under <d>; returns whether it is a recorded HTTP 200 (with its own raw
    output when <want_raw>)."""
    ts = time.strftime("%FT%TZ", time.gmtime())
    try:
        meta = run_stream(body, d, key, m) if m.startswith("stream") else run_nonstream(body, d, key)
    except Exception as e:
        meta = {"error": repr(e)}
    if want_raw and "error" not in meta:
        meta["raw"] = record_raw(m, d, key)
    meta["sent_at"] = ts
    with open(os.path.join(d, m + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("%s %s %s %s\n" % (ts, label, m, json.dumps(meta)))
    return meta.get("status") == 200 and "error" not in meta and (
        not want_raw or ("error" not in meta["raw"] and os.path.isfile(os.path.join(d, m + ".raw.txt"))))


def send_repeated(n, body, d, modes, count, conc, key, log):
    """<count> sends per mode of a REPEATS case, <conc> at a time: send i of both modes under <d>/s<i>/ (two independent
    samples, never compared), the body in <d>/request.json once and <d>/repeat.json naming the run. Output ids are
    always recorded: the grade decodes them. Returns (sent, ok)."""
    import concurrent.futures
    width = max(3, len(str(count - 1)))
    subs = ["s%0*d" % (width, i) for i in range(count)]
    with open(os.path.join(d, "repeat.json"), "w") as f:
        json.dump({"case": n, "sends_per_mode": count, "modes": modes, "concurrency": conc, "sends": subs}, f, indent=1)
    jobs = []
    for s in subs:
        os.makedirs(os.path.join(d, s), exist_ok=True)
        jobs += [(s, m) for m in modes]
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda j: send_one("%s/%s" % (n, j[0]), body, os.path.join(d, j[0]), j[1], key, log, True), jobs))
    return len(res), sum(res)


def main():
    out, names = sys.argv[1], [c for n in sys.argv[2].split(",") for c in GROUPS.get(n, [n])]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit("unknown case(s) %s (known: %s; groups: %s)" % (unknown, ", ".join(CASES), ", ".join(GROUPS)))
    modes = ["stream", "nonstream"]
    do_flush = "--flush" in sys.argv
    want_raw = "--no-raw" not in sys.argv
    if "--modes" in sys.argv:
        modes = sys.argv[sys.argv.index("--modes") + 1].split(",")
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else None
    conc = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 8
    if (repeat is not None and repeat < 1) or conc < 1:
        raise SystemExit("--repeat and --concurrency take a positive count")
    key = api_key()
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "probe.log"), "a", buffering=1)
    import threading
    lock = threading.Lock()

    def log(line):
        with lock:
            logf.write(line)
    log("START %s cases=%s modes=%s flush=%s raw=%s model=%s repeat=%s concurrency=%d\n" % (
        time.strftime("%FT%TZ", time.gmtime()), names, modes, do_flush, want_raw, MODEL,
        {n: repeat or REPEATS[n] for n in names if n in REPEATS}, conc))
    sent = ok = 0
    for n in names:
        rep = n in REPEATS
        body = dict(CASES[n], return_output_ids_in_sglext=True) if want_raw or rep else CASES[n]
        d = os.path.join(out, n)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "request.json"), "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=1)
        if rep:   # never flushed: a flush per sample would empty prod's prefix cache hundreds of times
            s, k = send_repeated(n, body, d, modes, repeat or REPEATS[n], conc, key, log)
            sent, ok = sent + s, ok + k
            continue
        for m in modes:
            if do_flush:
                flush(key, logf)
            sent += 1
            ok += send_one(n, body, d, m, key, log, want_raw)
    log("DONE %s\n" % time.strftime("%FT%TZ", time.gmtime()))
    print("DSML-PROBE RECORDED %d/%d sends (HTTP 200%s) -> %s" % (
        ok, sent, " with the request's own raw output" if want_raw else "", out), flush=True)


if __name__ == "__main__":
    main()
