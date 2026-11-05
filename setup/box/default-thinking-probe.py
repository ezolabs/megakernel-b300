"""Default-thinking probe for the official-settings cutover. Runs ON the box, against loopback:

    BOX=box-c bash -c '. lib.sh; box_ssh "/opt/sglang/bin/python3 -I -"' \
        < default-thinking-probe.py

Three streamed chat calls with the same prompt, differing only in the thinking fields:
  omit  -- no reasoning_effort, no chat_template_kwargs (what the router sends for a request that
           states no thinking intent: the serving router crates/sbr-wire/src/sglang/request.rs, the
           "neither chat_template_kwargs nor reasoning_effort" assertion in
           thinking_is_turned_on_and_off_with_the_template_kwarg_it_is_read_from)
  none  -- reasoning_effort "none"
  high  -- reasoning_effort "high"
For each: prompt_tokens, reasoning chars, content chars, finish_reason, seconds to the first
streamed token of any kind, seconds to the first CONTENT token (the latency a user waits for), and
think_leak -- whether the joined content (all chunks, so a tag split across two deltas still counts)
carries a `</think>` or `<think>` tag, i.e. reasoning the parser did not split out.

Verdict after the cutover (SGLANG_DEFAULT_THINKING=1, SGLANG_DSV41_REASONING_EFFORT=high):
  omit thinks (reasoning > 0) and its prompt_tokens == high's  -> the default is thinking ON with an
                                                                  effort line rendered
  none does not think (reasoning == 0) and prompt_tokens < high's -> the off switch still works
  no case's content carries `</think>` or `<think>`               -> the reasoning parser split every
                                                                  reply (think_leak False in all three)
A thinking prompt whose reasoning leaks into content is 2026-10-18's failure (box-c/ISSUES.md): the
template rendered the thinking prompt but the parser treated the turn as non-thinking, so content read
the reasoning, a newline, `</think>`, then the answer. reasoning > 0 alone does not rule that out (a
parser that splits SOME reasoning and still leaves `</think>` in content), so the tag check is its own
gate.
Before the cutover the same probe reads omit == none (no reasoning, 37 prompt tokens).
What the prompt-token match cannot tell apart: every effort rung renders a line of the same length
(the serving notes: "37 prompt tokens against 63 with any rung set"), so "at high" rests on the gate that
reads SGLANG_DSV41_REASONING_EFFORT=high out of the engine's /proc/<pid>/environ (the cutover's Phase A
environ gate), not on this probe. The flag name default_thinking_on_at_high is kept for that gate's
expected VERDICT line.

Exit codes: 0 = post-cutover verdict (all three conditions above); 3 = any other verdict (today's shape,
a broken off switch, a `</think>`/`<think>` tag in any case's content, or a stream with no usage chunk to
compare); 4 = an HTTP/connection error (no verdict). The API key is
read from the box's /etc/environment and never printed.

PROBE_BASE / PROBE_ENV_FILE override the loopback URL and the key file, for the local stub test
(tests/default-thinking-probe.test.sh) only; nothing sets them over box_ssh, so on the box the defaults
apply. Plain urllib with no proxy handler: a proxy variable in the box's ssh environment (pam_env reads
/etc/environment) must never carry a loopback call off the box.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("PROBE_BASE") or "http://127.0.0.1:8002"
ENV_FILE = os.environ.get("PROBE_ENV_FILE") or "/etc/environment"
MODEL = "DeepSeek-V4.1-Flash"
PROMPT = "A train leaves at 09:40 and arrives at 13:05. How long is the trip? Answer in one line."
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api_key() -> str:
    with open(ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no VLLM_API_KEY in " + ENV_FILE)


def run(name: str, extra: dict, key: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8192,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first_any = first_content = None
    reasoning = content = 0
    content_text = []
    finish = None
    prompt_tokens = completion_tokens = None
    with OPENER.open(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens")
                completion_tokens = chunk["usage"].get("completion_tokens")
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                rc = delta.get("reasoning_content") or ""
                cc = delta.get("content") or ""
                now = time.monotonic() - t0
                if (rc or cc) and first_any is None:
                    first_any = now
                if cc and first_content is None:
                    first_content = now
                reasoning += len(rc)
                content += len(cc)
                content_text.append(cc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    joined = "".join(content_text)
    row = {
        "case": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_chars": reasoning,
        "content_chars": content,
        "finish": finish,
        "think_leak": "</think>" in joined or "<think>" in joined,
        "t_first_token_s": round(first_any, 3) if first_any is not None else None,
        "t_first_content_s": round(first_content, 3) if first_content is not None else None,
        "total_s": round(time.monotonic() - t0, 3),
    }
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    key = api_key()
    try:
        omit = run("omit", {}, key)
        none = run("none", {"reasoning_effort": "none"}, key)
        high = run("high", {"reasoning_effort": "high"}, key)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # HTTPError is a URLError; its str() is the status line, never the request headers.
        print("VERDICT error=%s: %s (no verdict)" % (type(exc).__name__, exc), flush=True)
        return 4
    pt = (omit["prompt_tokens"], none["prompt_tokens"], high["prompt_tokens"])
    have_usage = all(isinstance(t, int) for t in pt)
    default_on = have_usage and omit["reasoning_chars"] > 0 and omit["prompt_tokens"] == high["prompt_tokens"]
    off_works = have_usage and none["reasoning_chars"] == 0 and none["prompt_tokens"] < high["prompt_tokens"]
    leaks = (omit["think_leak"], none["think_leak"], high["think_leak"])
    no_leak = not any(leaks)
    print(
        "VERDICT default_thinking_on_at_high=%s off_switch_works=%s first_content_delay_vs_none=%ss"
        " prompt_tokens(omit,none,high)=%s,%s,%s think_leak(omit,none,high)=%s,%s,%s%s%s"
        % (
            default_on,
            off_works,
            None
            if omit["t_first_content_s"] is None or none["t_first_content_s"] is None
            else round(omit["t_first_content_s"] - none["t_first_content_s"], 3),
            pt[0],
            pt[1],
            pt[2],
            leaks[0],
            leaks[1],
            leaks[2],
            "" if no_leak else " think_leak=PRESENT (a </think> or <think> tag in content: reasoning not split)",
            "" if have_usage else " usage=MISSING (no prompt_tokens to compare: no verdict)",
        ),
        flush=True,
    )
    return 0 if (default_on and off_works and no_leak) else 3


if __name__ == "__main__":
    sys.exit(main())
