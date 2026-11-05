#!/usr/bin/env python3
"""SGLang smoke battery for DeepSeek-V4.1-Flash, box-parametrized: this
script never hardcodes a host, port, model or boot-shape number -- everything
comes from boxenv.py (the box file named by $BOX/$BOX_ENV), the box's live
sglang-args file, or a CLI override.

Run against loopback:

    python3 smoke.py [--reps N] [--skip a,b] [--metrics]

Offline, touching no engine and reading no box file:

    python3 smoke.py --self-test

BASE URL / MODEL / API KEY CONVENTION (shared by every probe in this tool
set, see boxenv.py):
  - --base-url defaults to boxenv.resolve_engine_url(): $ENGINE_URL, else
    http://127.0.0.1:<ENGINE_PORT> from the box file.
  - --model defaults to boxenv.resolve_served_model(): $SERVED_MODEL, else
    the box file's SERVED_MODEL.
  - --api-key defaults to boxenv.resolve_api_key(): $VLLM_API_KEY, else
    /etc/environment on the box, else the box file's ENV_FILE. --env-file
    (kept for back-compat) is checked first when given, ahead of that chain.
  - The resolved base URL is printed first, before anything else runs.

Checks, in order:
  1. GET /v1/models -- id present, max_model_len matches --context-length in
     the engine's sglang-args file. Also backs onto GET /health.
  2. GET /get_server_info -- asserts attention_backend=dsv4,
     moe_runner_backend=flashinfer_mxfp4, fp8_gemm_runner_backend=
     flashinfer_cutedsl, speculative_algorithm=DSPARK,
     speculative_dspark_block_size=5, and tp_size/ep_size/
     max_running_requests as the box's live sglang-args file configures them
     (8/8/64 on the 8xB200 boot line, 4/4/64 on the 4-GPU one, 8/1/64 on
     moetp-b200, 8/8/128 on adm128). Never a literal: a hardcoded 4/4
     reported the correct 8xB200 box as a mismatch, and a hardcoded 64 would
     report a box booted on args/adm128.args the same way.
  Checks 1 and 2 read those four values from --args-file ($ARGS_FILE, else the
  copy's own sglang-args-<id>.txt on a box that runs ENGINES copies -- $ENGINE's,
  else the copy whose port or host the base URL names, and a refusal naming
  ARGS_FILE/ENGINE when it names none -- else the box file's ARGS_FILE, else
  /workspace/tools/sglang-args.txt). When that file
  cannot be read where this script runs, they FAIL with an "unverified" reason
  naming the path instead of comparing against a deployment-wide guess: off-box
  the old literal fallback turned a correct 8xB200 box into
  `mismatches (got, want)={'tp_size': (8, 4), 'ep_size': (8, 4)}` and would
  equally have passed a wrong one. Run from the box, or pass --args-file a copy
  of the box's sglang-args.txt, for a green off-box run.
  3. Hang probe -- a bounded chat immediately after /get_server_info must not
     hang forever.
  4. Reasoning split -- three requests, each naming its effort (none is ever
     left out): reasoning_content non-empty at reasoning_effort "high" (allowed,
     not asserted, at "low"), empty at reasoning_effort="none", the answer in
     content at "low" and "high", and no <think> leaked into content at any of
     the three. An absent effort is default-thinking-probe.py's gate, not this
     check's.
  5. DSML tool call -- a weather tool call comes back as tool_calls, not
     leaked as DSML-tagged text.
  6. json_object -- response_format json_object returns valid, on-topic JSON.
  7. Needle -- a ~110k-token prompt, cold and warm, both recover the needle.
  8. Image round-trip -- a red circle PNG is described correctly.
  9. Greedy -- --reps greedy (temperature=0) completions.
  10. Sampled -- --reps sampled completions at temperature=1.0, top_p=0.95,
      reasoning_effort=high.

Every request goes through /v1/chat/completions and carries a top-level
reasoning_effort field ("low"/"high"/"none" as appropriate) plus a curl
User-Agent (Cloudflare's bot rule blocks Python's default urllib/requests UA
at a front door; harmless on loopback, required the moment any of these ever
run through a door instead).

--metrics reads /metrics and prints the lines matching this box's watched
counters: sglang:num_running_reqs, num_queue_reqs, spec_accept_length,
num_retracted_reqs (plus anything else matching spec/accept/running/queue/
retract, so a renamed metric still shows up rather than silently vanishing).

Stdlib only (urllib/json/base64/zlib/time); Pillow only for drawing the smoke
PNG, with an embedded base64 PNG fallback when Pillow is not installed on the
box. That fallback constant is decoded and walked at import time -- signature,
declared chunk lengths, every chunk's CRC32, IHDR, the IDAT stream's inflate
size, IEND -- so a truncated one cannot ship: the constant this file used to
carry decoded to 662 bytes whose last chunk declared len=1007 and ran off the
end, and the engine was the first thing to notice (HTTP 400, `Could not decode
image: broken data stream when reading image file`) -- which reads as a vision
failure when only the fallback was broken. --self-test re-proves the same
invariants offline.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxenv

CURL_UA = boxenv.CURL_UA

# 256x256 red circle on white PNG, used only when Pillow is not importable on
# the box. Generated once and frozen here; not regenerated at runtime. Frozen
# is not the same as trusted: this constant used to be base64 cut short
# mid-IDAT, so every run of this file re-proves it at import
# (_fallback_png_bytes below) and --self-test re-proves it offline.
FALLBACK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAD2klEQVR42u3XsRXCMBAFQRVB/5XR"
    "C2REkGHpYOfeVmD9CbwezoVv+QQOAOcAcA4A5wBwDgDnAHAOAOcAcA4A5wBwDgDnAHAOAOcAcA4A"
    "5wBwDgDnAHAOAPfx7rfbK18DgP9c9nfzbQEIzR0JAIweBgCMHgYA7J4EAOyeBADsngQA7J4EAEwf"
    "AwBMHwMATB8DAEwfAwBMHwMATB8DAEwfgzAAq2UgCsBYMYgCMFAMugDskoEuAItkIArAEDHoArA/"
    "BroALI+BKACDw6ALwM4Y6AKwMAa6AGyLgS4Aq2KgC8CeGIgCMCMMugCsh4EuALthoAvAYhgAQAAk"
    "AdgKA10AVsJAF4B9MNAFYBkMdAHYBANdANbAAAACIAnADhjoArAAdQF4ex03AIAAsH5VDSzrV9nA"
    "sn6VDQAgAKxfVQMACADrV9XAsn6VDQAgAKxfVQMACADrV9UAAALA+lU1AIAAsH5VDQAgAKxfVQMA"
    "CADrV9UAAALA+lU1AIAAAEAAWL+CBgAQAAAIAOtX0AAAAgAAAWD9ChoAQAAAIACsX0EDAAgAAAQA"
    "AALA+tUyAIAAAEAAACAArF8tAwAIAAAEAAACAAABAIAAsH4VDAAgAAAQAAAIAAAEAAACAAABAIAA"
    "AEAAACAAABAAAAgAAASA9evnDQAgAAAQAAAIAAbkJxgAAQCAAABAAAAgAAAQAAAIAAAEAAACAAAB"
    "4A0EgASA1APAgH53/QAIAAAEAAACAAABAIAAYECF9QMgAAAQAAAIAAbUWj8AAgAAAQCAAGBArfUD"
    "IAAAEAAMKLh+AAQAAAKAAQXXD4AAAEAAMKDg+gEQAAAIAAYUXD8AAuCy82Yavn4ABMCV5+U0ef0A"
    "CICLz/tp7PoBEADXn1fUzPUDIAC2nLfUwPUDIAB2nRfVtPUDIAA2nnfVqPXvBsCARq0fAAGw/byx"
    "hqwfAAFw4ry0Jqz/GAAGNGH9JwEwoOPrB0AAHD1vr7MLPAyAAeuvA2DA+gGwBgDCABiw/joABqy/"
    "DoAB668DYMD66wAYsP46AAasHwCLASAMgAHrrwNgwPrrABiw/joADEwfAAasPw+AAeuvA2DA+usA"
    "GLD+OgAGrL8OAAPTB4AB688DYMD66wAwMH0AGLD+PAAGrL8OAAPTBwAD0weAAesHAAPTBwAD0wcA"
    "A9MHAAPTBwAD0wcAA9MHAAPvDkBRglcGoCjBmwJQlOAFAShK8F4A5DB4FwByGHx/AEIkfFsA/tOJ"
    "rwGAcwA4B4BzADgHgHMAOAeAcwA4B4BzADgHgHMAOAeAcwA4B4BzADgHgHMAOAeAcwA49+6e1dCM"
    "dEEB+EoAAAAASUVORK5CYII="
)

PNG_SIG = b"\x89PNG\r\n\x1a\n"

# PNG colour type -> samples per pixel, and the bit depths the spec allows;
# IHDR's layout is fixed by the spec, not by this script.
PNG_COLOR_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
PNG_BIT_DEPTHS = (1, 2, 4, 8, 16)


def png_defect(data: bytes) -> str | None:
    """Why `data` is not a whole, decodable PNG -- or None if it is whole.

    A base64 constant cut short keeps a valid signature and IHDR, so every
    reader short of a real PNG walk calls it fine: the constant this file used
    to carry decoded to 662 bytes whose last chunk declared len=1007 and simply
    ran off the end, and the engine was the first thing to notice (HTTP 400,
    `Could not decode image: broken data stream when reading image file`). So
    walk it here: declared chunk lengths must fit inside the file, every
    chunk's CRC32 must match, IHDR must come first and describe a plain
    non-interlaced image, the concatenated IDAT stream must inflate to exactly
    the scanline total IHDR implies, and IEND must close the file with nothing
    after it. When Pillow is importable it gets the final word, so the answer
    is never less strict than the loader that would receive the payload.
    """
    if not data.startswith(PNG_SIG):
        return f"no 8-byte PNG signature (first bytes: {data[:8]!r})"
    pos = len(PNG_SIG)
    seen: list[str] = []
    idat = bytearray()
    ihdr = None
    iend = False
    while pos < len(data):
        if len(data) - pos < 8:
            return (
                f"truncated {len(data) - pos} byte(s) into a chunk header at offset {pos} "
                f"(chunks so far: {seen or ['<none>']})"
            )
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        name = ctype.decode("ascii", "replace")
        end = pos + 12 + length
        if end > len(data):
            return (
                f"chunk {name} at offset {pos} declares len={length} but only "
                f"{len(data) - pos - 12} payload byte(s) remain (file is {len(data)} bytes; "
                f"chunks so far: {seen or ['<none>']}) -- exactly what a base64 constant cut "
                f"short looks like"
            )
        payload = data[pos + 8:pos + 8 + length]
        stored = int.from_bytes(data[pos + 8 + length:end], "big")
        actual = zlib.crc32(ctype + payload) & 0xFFFFFFFF
        if actual != stored:
            return f"chunk {name} at offset {pos} has CRC32 {actual:#010x} != stored {stored:#010x} (payload damaged)"
        if not seen and name != "IHDR":
            return f"first chunk is {name}, not IHDR"
        if name == "IHDR":
            if ihdr is not None:
                return f"a second IHDR chunk at offset {pos}"
            ihdr = payload
        elif name == "IDAT":
            idat += payload
        elif name == "IEND":
            iend = True
            if length:
                return f"IEND carries {length} payload byte(s)"
            if end != len(data):
                return f"{len(data) - end} byte(s) follow IEND"
        seen.append(name)
        pos = end
    if ihdr is None:
        return f"no IHDR chunk (chunks: {seen or ['<none>']})"
    if not idat:
        return f"no IDAT chunk (chunks: {seen})"
    if not iend:
        return f"no IEND chunk -- the file ends after {seen}"
    if len(ihdr) != 13:
        return f"IHDR is {len(ihdr)} byte(s), not the 13 the spec fixes"
    width = int.from_bytes(ihdr[0:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    depth, color, compression, filt, interlace = ihdr[8], ihdr[9], ihdr[10], ihdr[11], ihdr[12]
    if not width or not height:
        return f"IHDR {width}x{height} (a zero dimension)"
    if color not in PNG_COLOR_CHANNELS:
        return f"IHDR colour type {color} is not one of {tuple(sorted(PNG_COLOR_CHANNELS))}"
    if depth not in PNG_BIT_DEPTHS:
        return f"IHDR bit depth {depth} is not one of {PNG_BIT_DEPTHS}"
    if (compression, filt, interlace) != (0, 0, 0):
        return f"IHDR compression/filter/interlace {(compression, filt, interlace)}, all of which must be 0"
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as e:
        return f"IDAT does not inflate ({e}) -- truncated or corrupt stream"
    stride = (width * depth * PNG_COLOR_CHANNELS[color] + 7) // 8
    want = height * (1 + stride)
    if len(raw) != want:
        return (
            f"IDAT inflates to {len(raw)} byte(s), not the {want} that "
            f"{width}x{height} depth-{depth} colour-{color} implies -- the stream is cut short"
        )
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Exception as e:
        return f"Pillow cannot decode it: {type(e).__name__}: {e}"
    return None


def _fallback_png_bytes(b64: str = FALLBACK_PNG_B64) -> bytes:
    """FALLBACK_PNG_B64 decoded and proved whole.

    Run at import, deliberately fatal: the fallback is reached only on a box
    without Pillow, which is exactly where a broken one turns the vision check
    into an HTTP 400 that reads like a model failure. Refusing to start is the
    only outcome that cannot ship a truncated constant again; --self-test calls
    this same function with a deliberately cut-short constant to show it bites.
    """
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise RuntimeError(
            f"FALLBACK_PNG_B64 is not valid base64 ({type(e).__name__}: {e})"
        ) from e
    defect = png_defect(data)
    if defect is not None:
        raise RuntimeError(
            f"FALLBACK_PNG_B64 does not decode to a whole PNG ({defect}); regenerate it from a "
            f"real PNG -- an engine answers HTTP 400 'Could not decode image: broken data stream "
            f"when reading image file' for one, which reads as a vision failure when vision is fine"
        )
    return data


# The validated bytes the image check posts when Pillow is missing: decoded,
# chunk-walked and (where Pillow exists) decoded again at import. smoke_frame()
# never re-derives this from the constant text, so the payload under test is
# the payload that was validated.
FALLBACK_PNG = _fallback_png_bytes()

# Every check name below is a valid --skip token.
CHECK_NAMES = [
    "models",
    "server_info",
    "hang_probe",
    "reasoning",
    "tool_call",
    "json_object",
    "needle",
    "image",
    "greedy",
    "sampled",
]


def _headers(key: str, content_type: bool) -> dict:
    h = {"Authorization": "Bearer " + key, "User-Agent": CURL_UA}
    if content_type:
        h["content-type"] = "application/json"
    return h


def call(base: str, key: str, path: str, body=None, timeout=900):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=_headers(key, body is not None),
        method="GET" if body is None else "POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def fetch_text(base: str, key: str, path: str, timeout=30) -> str:
    req = urllib.request.Request(base + path, headers=_headers(key, False))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def health(base: str, key: str, timeout=10) -> int:
    req = urllib.request.Request(base + "/health", headers=_headers(key, False))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def chat(base: str, key: str, model: str, messages, **kw):
    body = {"model": model, "messages": messages, "max_tokens": kw.pop("max_tokens", 600)}
    body.update(kw)
    r, wall = call(base, key, "/v1/chat/completions", body)
    ch = r["choices"][0]
    return ch, r.get("usage") or {}, wall


class ArgsFileUnreadable(RuntimeError):
    """The engine's launch-args file could not be read where this run happened.

    This script used to answer that with deployment-wide literals (tp_size=
    ep_size=4, context 1,048,576). Off-box that is not a weaker check, it is an
    invented one: it reported the correct 8xB200 8/8 boot as
    `mismatches (got, want)={'tp_size': (8, 4), 'ep_size': (8, 4)}`, and it
    would have passed a 4/4 box that was supposed to be 8/8 the same way. The
    file is the only honest source for the boot shape, so a check that cannot
    read it FAILS saying so (check_models / check_server_info) instead of
    comparing the live engine against a number nobody configured.
    """


def parse_args_text(text: str) -> list[str]:
    """The flag tokens of an args file's text, the words relaunch.sh launches.

    relaunch.sh reads the file as `A=$(grep -v '^#' FILE)` and then
    `eval "A=($A)"`. So a line starting with `#` is dropped, and the rest is
    split the way bash splits words: quotes are honoured, and a `#` that starts
    a word comments out the rest of its line. That covers an indented comment
    line and a trailing ` # ...` too. A `#` inside a word (`--x=a#b`), quoted
    or backslash-escaped stays literal, as in bash, and a backslash-newline
    outside single quotes joins lines. shlex's own comments=True cannot do
    that: it cuts at any unquoted `#`, mid-word included, so the comments are
    stripped here and shlex only splits. A plain split() did not: it read the
    flags out of the header prose too, and v1-b300-box-a.args' header line
    `--tp-size 8 --ep-size 8 -> 4 / 4` made check 2 want tp 8 from a box
    correctly serving TP4/EP4.
    """
    kept = "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))
    out: list[str] = []
    quote = ""  # "'" or '"' while inside that quote
    word_start = True  # the next char would begin a word (bash's only place for a comment)
    i, n = 0, len(kept)
    while i < n:
        c = kept[i]
        if quote == "'":
            if c == "'":
                quote = ""
        elif c == "\\" and i + 1 < n:
            if kept[i + 1] == "\n":  # line continuation: bash drops both
                i += 2
                continue
            out.append(kept[i : i + 2])
            word_start = False
            i += 2
            continue
        elif quote == '"':
            if c == '"':
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "#" and word_start:
            while i < n and kept[i] != "\n":
                i += 1
            continue
        out.append(c)
        word_start = not quote and c in " \t\n"
        i += 1
    return shlex.split("".join(out), comments=False)

def read_args_file(args_file: str) -> list[str]:
    """The launch-args file's flag tokens (parse_args_text: comments dropped).

    The same file, and for the same reason, as before: a rung may deliberately
    boot tp=8/ep=1 (args/moetp-b200.args) or max_running_requests=128
    (args/adm128.args), so neither a GPU count nor a literal can stand in for
    the flags the engine was actually launched with.
    """
    try:
        with open(args_file) as f:
            text = f.read()
    except OSError as e:
        raise ArgsFileUnreadable(
            f"cannot read the launch-args file {args_file!r} ({e.strerror or e}); pass --args-file "
            f"PATH with a readable copy of the box's sglang-args.txt (or set $ARGS_FILE) and re-run"
        ) from e
    try:
        return parse_args_text(text)
    except ValueError as e:
        raise ArgsFileUnreadable(
            f"cannot split the launch-args file {args_file!r} ({e}); relaunch.sh's eval would refuse "
            f"it too, so fix its quoting and re-run"
        ) from e


def _args_int(parts: list[str], flag: str) -> int | None:
    """The int following `flag` in an args-file token list, or None when the
    flag is absent or carries no int."""
    try:
        return int(parts[parts.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def expected_parallelism(args_file: str) -> tuple[int, int]:
    """The (tp_size, ep_size) the engine is actually configured to boot with.

    Read from the live SGLang launch-args file -- this pair lives in the args
    file, not the box file (see args/README.md's composition rule), because a
    rung may deliberately run tp=8/ep=1 (moetp-b200.args) on a box that still
    has 8 GPUs. So neither a GPU count nor a literal can stand in for it: the
    literals that used to be here were box-g's 4xB300 shape and reported the
    correct 8xB200 8/8 boot as
    `mismatches (got, want)={'tp_size': (8, 4), 'ep_size': (8, 4)}`.

    --ep-size falls back to the --tp-size read from the same file when the line
    does not carry it (SGLang's own default), so this is never a weaker check
    than the literals were. A file that cannot be read, or a line with no
    --tp-size, raises ArgsFileUnreadable -- never a deployment-wide 4/4.
    """
    parts = read_args_file(args_file)
    tp = _args_int(parts, "--tp-size")
    if tp is None:
        raise ArgsFileUnreadable(
            f"{args_file!r} carries no --tp-size, so the boot shape cannot be checked against it"
        )
    ep = _args_int(parts, "--ep-size")
    return tp, tp if ep is None else ep


def expected_max_running_requests(args_file: str) -> int | None:
    """--max-running-requests from the launch-args file, or None when the line
    does not pin it.

    None is not a default: it means this check cannot assert the value, and
    check_server_info says so in its own reason line instead of measuring the
    live engine against a deployment-wide literal. The 64 that used to be
    hard-coded here would have reported a box booted on args/adm128.args --
    which deliberately runs 128 -- as a mismatch of a knob that box's own line
    had chosen. Every line in args/ happens to pin it today, but the composition
    rule permits a line that leaves SGLang's own default in place, and that line
    must not be red-gated by a smoke literal.
    """
    return _args_int(read_args_file(args_file), "--max-running-requests")


def expected_context_length(args_file: str) -> int:
    """The window the engine is actually configured to serve.

    Read from the live SGLang launch-args file rather than hard-coded: a
    literal here goes stale the moment the box's --context-length changes, and
    off-box it invents an expectation for a box this run cannot see. An
    unreadable file, or a line with no --context-length, raises
    ArgsFileUnreadable.
    """
    ctx = _args_int(read_args_file(args_file), "--context-length")
    if ctx is None:
        raise ArgsFileUnreadable(
            f"{args_file!r} carries no --context-length, so the served window cannot be checked against it"
        )
    return ctx


def smoke_frame() -> str:
    """A 1280x827 PNG (the Cosmos capture-frame size) with a large red circle on
    white, as a data: URL. Falls back to the embedded 256x256 red-circle PNG
    (FALLBACK_PNG, decoded and walked at import) when Pillow is not installed on
    the box.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # The validated bytes, never the constant text: see _fallback_png_bytes.
        return "data:image/png;base64," + base64.b64encode(FALLBACK_PNG).decode()
    im = Image.new("RGB", (1280, 827), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((240, 63, 1040, 763), fill=(220, 20, 20))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def needle_prompt() -> str:
    """~105k-token needle prompt: random filler words plus a PELICAN-4471-style
    needle, scaled up from box-h-vllm-smoke.py's 70000-word/~100k-token recipe
    (70000 * 105/100 = 73500 words).
    """
    random.seed(7)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    filler = " ".join(random.choice(words) for _ in range(73500))
    needle = "The secret launch code is PELICAN-4471."
    cut = len(filler) // 2
    return (
        filler[:cut] + " " + needle + " " + filler[cut:]
        + "\n\nWhat is the secret launch code? Answer with the code only."
    )


def run(i, name, skip, fn, results):
    if name in skip:
        print(f"[{i}] {name}: SKIP")
        return
    try:
        ok, reason = fn()
    except Exception as e:
        ok, reason = False, f"exception: {type(e).__name__}: {e}"
    results.append(ok)
    print(f"[{i}] {name}: {'PASS' if ok else 'FAIL'} - {reason}")


def check_models(base: str, key: str, model: str, args_file: str) -> tuple[bool, str]:
    """Check 1. The window it expects comes from the box's own launch-args file,
    and an unreadable one FAILS with an "unverified" reason: off-box the old
    default (1,048,576) was a number this script chose rather than one the box
    was booted with, so a mismatch line naming it was evidence of nothing.
    """
    try:
        want = expected_context_length(args_file)
    except ArgsFileUnreadable as e:
        return False, f"unverified: {e}"
    hcode = health(base, key)
    m, _ = call(base, key, "/v1/models")
    ids = [x["id"] for x in m["data"]]
    mml = m["data"][0].get("max_model_len") if m["data"] else None
    ok = hcode == 200 and (model in ids) and (mml == want)
    return ok, f"health={hcode} ids={ids} max_model_len={mml} (expected {want} from {args_file})"


def check_server_info(base: str, key: str, args_file: str) -> tuple[bool, str]:
    """Assert the live engine's shape against what this box's launch-args file
    says it was booted with. tp_size/ep_size/max_running_requests come from that
    file rather than a literal, so the check still catches a REAL mismatch (a box
    serving 4/4 from an 8/8 boot line fails) without red-gating a box whose
    correct shape differs from another box's (8/8 on the B200 lines, 8/1 on
    moetp-b200, 128 on adm128). An unreadable launch-args file FAILS with an
    "unverified" reason instead of comparing the live engine against a
    deployment-wide 4/4.

    /get_server_info needs the bearer: unauthenticated it answers
    {"error": "Unauthorized"}, which parses as an empty dump and would read as
    a mismatch of every key rather than as a missing key.
    """
    try:
        tp, ep = expected_parallelism(args_file)
        mrr = expected_max_running_requests(args_file)
    except ArgsFileUnreadable as e:
        return False, f"unverified: {e}"
    dump_keys = [
        "attention_backend",
        "moe_runner_backend",
        "fp8_gemm_runner_backend",
        "speculative_algorithm",
        "speculative_dspark_block_size",
        "max_running_requests",
        "context_length",
        "chunked_prefill_size",
        "tp_size",
        "ep_size",
        "mem_fraction_static",
        "page_size",
        "version",
    ]
    want = {
        "attention_backend": "dsv4",
        "moe_runner_backend": "flashinfer_mxfp4",
        "fp8_gemm_runner_backend": "flashinfer_cutedsl",
        "speculative_algorithm": "DSPARK",
        "speculative_dspark_block_size": 5,
        "tp_size": tp,
        "ep_size": ep,
    }
    if mrr is not None:
        # A line that pins no --max-running-requests leaves SGLang's own default
        # in place, and this check cannot see what that default is -- so it
        # asserts nothing there and says so below, rather than red-gating a rung
        # for a knob its line never chose.
        want["max_running_requests"] = mrr
    info, _ = call(base, key, "/get_server_info")
    dump = {k: info.get(k, "<missing>") for k in dump_keys}
    print(f"    server_info: {dump}")
    mismatches = {k: (dump[k], v) for k, v in want.items() if dump[k] != v}
    ok = not mismatches
    mrr_note = (
        f"max_running_requests={mrr}" if mrr is not None
        else "max_running_requests NOT asserted (the launch line does not pin it)"
    )
    return ok, (
        f"matches dsv4/flashinfer_mxfp4/flashinfer_cutedsl/DSPARK/5/tp{tp}/ep{ep}/{mrr_note} "
        f"against {args_file}"
        if ok else f"mismatches (got, want)={mismatches}"
    )


def check_hang_probe(base, key, model):
    """The #34340 silent-hang probe: a bounded chat immediately after
    /get_server_info must not hang forever. FAIL on timeout, not an exception
    bubbling up uncaught.
    """
    try:
        ch, u, wall = chat(
            base, key, model,
            [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
            temperature=0.0,
            max_tokens=64,
            reasoning_effort="none",
            timeout=120,
        )
    except (TimeoutError, urllib.error.URLError, OSError) as e:
        return False, f"timeout/error within 120s bound: {type(e).__name__}: {e}"
    return True, f"finish={ch.get('finish_reason')} tokens={u.get('completion_tokens')} wall={wall:.2f}s"


def check_reasoning(base, key, model):
    prompt = (
        "A train leaves at 09:40 and arrives at 13:05 the same day. How many minutes is the "
        "trip? Reason it out, then answer with the number only on the last line."
    )
    want = "205"
    lines = []
    ok = True
    for effort in ("low", "high", "none"):
        kw: dict[str, object] = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4000}
        kw["reasoning_effort"] = effort
        ch, u, wall = chat(base, key, model, [{"role": "user", "content": prompt}], **kw)
        msg = ch["message"]
        rc = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        leak = "<think>" in content
        lines.append(
            f"effort={effort!r}: finish={ch.get('finish_reason')} reasoning_chars={len(rc)} "
            f"content={content[-60:]!r} leak={leak} tokens={u.get('completion_tokens')} wall={wall:.2f}s"
        )
        if effort == "high":
            ok &= len(rc) > 0 and not leak and want in content
        elif effort == "none":
            ok &= len(rc) == 0 and not leak
        else:  # "low"
            ok &= not leak and want in content
    print("    " + " | ".join(lines))
    return ok, (
        "reasoning_content non-empty at high (allowed, not asserted, at low); empty at "
        f"reasoning_effort=none; {want} in content at low and high; no <think> leak"
    )


def check_tool_call(base, key, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    ch, u, wall = chat(
        base, key, model,
        [{"role": "user", "content": "What is the weather in Tokyo right now? Use the tool."}],
        tools=tools,
        tool_choice="auto",
        reasoning_effort="low",
        temperature=1.0,
        top_p=0.95,
    )
    msg = ch["message"]
    tc = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    args = tc[0]["function"]["arguments"] if tc else None
    fn_name = tc[0]["function"]["name"] if tc else None
    ok = (
        ch.get("finish_reason") == "tool_calls"
        and bool(tc)
        and "Tokyo" in (args or "")
        and "DSML" not in content
    )
    return ok, f"finish={ch.get('finish_reason')} calls={len(tc)} fn={fn_name} args={args} dsml_leak={'DSML' in content} wall={wall:.2f}s"


def check_json_object(base, key, model):
    ch, u, wall = chat(
        base, key, model,
        [{"role": "user", "content": "Return a JSON object with keys city and country for Tokyo."}],
        response_format={"type": "json_object"},
        reasoning_effort="low",
        temperature=1.0,
        top_p=0.95,
    )
    content = ch["message"].get("content") or ""
    try:
        j = json.loads(content)
        jok = str(j.get("city", "")).lower().startswith("tokyo")
    except Exception:
        jok = False
    return jok, f"finish={ch.get('finish_reason')} valid={jok} content={content[:80]!r} wall={wall:.2f}s"


def check_needle(base, key, model):
    prompt = needle_prompt()
    cold_ch, cold_u, cold_wall = chat(
        base, key, model,
        [{"role": "user", "content": prompt}],
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=600,
    )
    warm_ch, warm_u, warm_wall = chat(
        base, key, model,
        [{"role": "user", "content": prompt}],
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=600,
    )
    cold_content = cold_ch["message"].get("content") or ""
    warm_content = warm_ch["message"].get("content") or ""
    cold_found = "PELICAN-4471" in cold_content
    warm_found = "PELICAN-4471" in warm_content
    ok = cold_found and warm_found
    return ok, (
        f"cold: prompt_tokens={cold_u.get('prompt_tokens')} found={cold_found} "
        f"finish={cold_ch.get('finish_reason')} completion_tokens={cold_u.get('completion_tokens')} wall={cold_wall:.2f}s | "
        f"warm: found={warm_found} finish={warm_ch.get('finish_reason')} "
        f"completion_tokens={warm_u.get('completion_tokens')} wall={warm_wall:.2f}s"
    )


def check_image(base, key, model):
    frame = smoke_frame()
    content = [
        {"type": "image_url", "image_url": {"url": frame}},
        {"type": "text", "text": "What shape and what colour? Two words."},
    ]
    ch, u, wall = chat(
        base, key, model,
        [{"role": "user", "content": content}],
        reasoning_effort="low",
        temperature=0.0,
        # The vision path is not run-to-run deterministic even at temperature 0
        # (the same request answered in 197 and 982 reasoning tokens on an idle
        # box), so the cap has to hold the long tail: at 400 the check failed on
        # an empty, length-cut reply while the answer itself was right.
        max_tokens=1500,
    )
    reply = ch["message"].get("content") or ""
    low = reply.lower()
    ok = "red" in low and "circle" in low
    return ok, f"finish={ch.get('finish_reason')} reply={reply[:120]!r} wall={wall:.2f}s"


def check_greedy(base, key, model, reps):
    lines = []
    ok = True
    for i in range(reps):
        ch, u, wall = chat(
            base, key, model,
            [{"role": "user", "content": "Give me a fully functional FastAPI server. Show the python code."}],
            temperature=0.0,
            max_tokens=2048,
            reasoning_effort="none",
        )
        ct = u.get("completion_tokens") or 0
        toks = ct / wall if wall > 0 else 0.0
        lines.append(f"rep{i + 1}: {toks:.1f} tok/s tokens={ct} wall={wall:.2f}s finish={ch.get('finish_reason')}")
        ok &= ch.get("finish_reason") is not None
    print("    " + " | ".join(lines))
    return ok, f"{reps} greedy reps completed"


def check_sampled(base, key, model, reps):
    lines = []
    ok = True
    for i in range(reps):
        ch, u, wall = chat(
            base, key, model,
            [{"role": "user", "content": "Give me a fully functional FastAPI server. Show the python code."}],
            temperature=1.0,
            top_p=0.95,
            reasoning_effort="high",
            max_tokens=2048,
        )
        ct = u.get("completion_tokens") or 0
        toks = ct / wall if wall > 0 else 0.0
        lines.append(f"rep{i + 1}: {toks:.1f} tok/s tokens={ct} wall={wall:.2f}s finish={ch.get('finish_reason')}")
        ok &= ch.get("finish_reason") is not None
    print("    " + " | ".join(lines))
    return ok, f"{reps} sampled reps completed"


def print_metrics(base, key):
    """Print the /metrics lines this box's smoke run watches: sglang's
    num_running_reqs, num_queue_reqs, spec_accept_length, num_retracted_reqs
    -- plus anything else matching spec/accept/running/queue/retract, so a
    renamed or added metric still surfaces instead of silently vanishing.
    """
    try:
        text = fetch_text(base, key, "/metrics")
    except Exception as e:
        print(f"metrics: FAIL to fetch - {type(e).__name__}: {e}")
        return
    pattern = re.compile(r"spec|accept|running|queue|retract", re.IGNORECASE)
    matched = [line for line in text.splitlines() if pattern.search(line)]
    print(f"metrics: {len(matched)} matching lines")
    for line in matched:
        print("    " + line)


def pick_args_file(cli_value, base_url, environ, table, tools_remote, host_of, legacy):
    """Which launch-args file checks 1-2 read, as a pure function of its inputs (--self-test drives it).

    --args-file, then $ARGS_FILE, win as given. On a box whose box file names ENGINES copies (`table`,
    boxenv.engines()), the legacy sglang-args.txt is the single-engine (TP8) line and not what any copy
    runs, so the copy's own <tools_remote>/sglang-args-<id>.txt is read: $ENGINE's (validated by the
    caller through boxenv.engine_id()), else the one copy whose host (host_of(id), the copy's door name)
    or port the base URL names. A base URL naming no copy, or several, refuses (SystemExit) naming
    ARGS_FILE / ENGINE / --args-file instead of comparing the copy against the TP8 line -- #750's copy-b
    smoke failed check 2 that way. With no ENGINES, `legacy()` (the box file's ARGS_FILE, else
    /workspace/tools/sglang-args.txt) is called.
    """
    if cli_value:
        return cli_value
    if environ.get("ARGS_FILE"):
        return environ["ARGS_FILE"]
    if table:
        root = (tools_remote or "/workspace/tools").rstrip("/")
        eid = environ.get("ENGINE") or None
        if eid is None:
            parsed = urllib.parse.urlparse(base_url or "")
            host = (parsed.hostname or "").lower()
            try:
                port = parsed.port
            except ValueError:
                port = None
            by_host = []
            for e in table:
                try:
                    if host and host == host_of(e).lower():
                        by_host.append(e)
                except Exception:
                    pass
            by_port = [e for e in table if port is not None and str(table[e].get("port")) == str(port)]
            hits = by_host or by_port
            if len(hits) != 1:
                names = " ".join(table)
                raise SystemExit(
                    f"smoke.py: this box runs ENGINES copies ({names}) and the base URL {base_url!r} names "
                    f"{'none' if not hits else 'more than one'} of them by port or host, so there is no copy "
                    f"args file to read for checks 1-2 (the legacy sglang-args.txt is the single-engine TP8 "
                    f"line, not a copy's): set ENGINE=<id> (lib.sh's engine_select), "
                    f"ARGS_FILE={root}/sglang-args-<id>.txt, or pass --args-file"
                )
            eid = hits[0]
        return f"{root}/sglang-args-{eid}.txt"
    return legacy()


def resolve_args_file(cli_value, base_url):
    """pick_args_file over boxenv: the box file is read only when no --args-file/$ARGS_FILE is given."""
    if cli_value or os.environ.get("ARGS_FILE"):
        return pick_args_file(cli_value, base_url, os.environ, {}, None, lambda e: "", lambda: "")
    boxenv.engine_id()  # validates $ENGINE against ENGINES (raises on an unknown id or a legacy box)
    table = boxenv.engines()
    return pick_args_file(
        None, base_url, os.environ, table, boxenv.load().get("TOOLS_REMOTE"),
        boxenv.engine_host, boxenv.args_file_default,
    )


# The checks that read the launch-args file: checks 1-2, CHECK_NAMES' first two. main() runs them
# under these same names (run(1, ARGS_FILE_CHECKS[0], ...), run(2, ARGS_FILE_CHECKS[1], ...)), so the
# skip gate below and the checks it gates are one table and cannot drift apart.
ARGS_FILE_CHECKS = tuple(CHECK_NAMES[:2])


def args_file_for(skip, cli_value, base_url):
    """resolve_args_file, but only when a check that reads the file runs: with checks 1-2 both in
    --skip the file is never read, so a base URL naming no copy (the fronting box-c door)
    must not refuse the whole battery. None when nothing needs it."""
    if all(name in skip for name in ARGS_FILE_CHECKS):
        return None
    return resolve_args_file(cli_value, base_url)


def _self_test() -> int:
    """Offline regression test for the two defects this file shipped with: a
    fallback PNG whose base64 was cut short (so the image check 400'd on a box
    without Pillow, reading as a vision failure) and boot-shape numbers that fell
    back to deployment-wide literals when the launch-args file was unreadable.

    Then the per-copy args-file pick (pick_args_file, pure), its wiring over
    boxenv (resolve_args_file / args_file_for) and the real box-c box file.

    No engine and no network: the PNG cases are memory-only, the args readers
    run through a temporary file (and its absence), the wiring cases through a
    temporary ENGINES box file (BOX_ENV), and, when this tree carries it, the
    tracked boxes/box-c.env is read so each of its copies' door name
    and port picks that copy's file. The environment and boxenv's cache are
    restored afterwards. PASS/FAIL per case, 0/1 back, like boxenv.py's
    --self-test.
    """
    import tempfile

    ok = True

    def case(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}{' -- ' + detail if detail else ''}")

    print("fallback PNG (posted by check_image on a box with no Pillow):")
    whole = _fallback_png_bytes(FALLBACK_PNG_B64)
    case(
        "FALLBACK_PNG_B64 decodes and walks as a whole PNG",
        whole == FALLBACK_PNG and png_defect(whole) is None,
        f"{len(whole)} bytes from {len(FALLBACK_PNG_B64)} base64 chars",
    )
    try:
        _fallback_png_bytes(FALLBACK_PNG_B64[:-40])
        case("base64 cut 40 chars short is rejected at import", False, "it decoded anyway")
    except RuntimeError as e:
        case("base64 cut 40 chars short is rejected at import", True, str(e)[:160])
    cut = FALLBACK_PNG[:-30]
    defect = png_defect(cut)
    case(
        "PNG bytes cut 30 short are named as truncated",
        defect is not None and "declares" in defect,
        defect or "no defect reported",
    )
    mid = len(FALLBACK_PNG) // 2
    damaged = FALLBACK_PNG[:mid] + bytes([FALLBACK_PNG[mid] ^ 0x01]) + FALLBACK_PNG[mid + 1:]
    defect = png_defect(damaged)
    case("one damaged byte is caught (CRC32 or inflate)", defect is not None, defect or "no defect reported")

    print()
    print("launch-args readers (checks 1-2's only source for the boot shape):")
    line = (
        "python -m sglang.launch_server --tp-size 8 --ep-size 1"
        " --max-running-requests 128 --context-length 262144\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sglang-args.txt")
        with open(path, "w") as f:
            f.write(line)
        case("--tp-size/--ep-size read from the line", expected_parallelism(path) == (8, 1))
        case("--max-running-requests read from the line", expected_max_running_requests(path) == 128)
        case("--context-length read from the line", expected_context_length(path) == 262144)

        with open(path, "w") as f:
            # The trailing comment sits on a line that is not the last: relaunch.sh's $(...) drops the
            # final newline, so one on the last line would comment out eval's closing `)`.
            f.write(
                "# header: --tp-size 4 --ep-size 4 -> 8 / 1\n"
                + line.replace(" --ep-size 1", " # trailing: --ep-size 2\n--ep-size 1")
            )
        case("comment lines and a trailing comment name no flag", expected_parallelism(path) == (8, 1))

        tricky = "--x=a#b \"c # d\" 'e #f' g\\ #h \\#i  # gone\n  # gone too\n--y 'j\nk' # gone\n"
        want = ["--x=a#b", "c # d", "e #f", "g #h", "#i", "--y", "j\nk"]
        got = parse_args_text(tricky)
        case("a # inside a word, quoted or escaped stays literal, as in bash", got == want, repr(got))

        with open(path, "w") as f:
            f.write(line.replace(" --ep-size 1", ""))
        case("a line with no --ep-size means ep = tp (SGLang's default)", expected_parallelism(path) == (8, 8))

        with open(path, "w") as f:
            f.write(line.replace(" --max-running-requests 128", ""))
        case(
            "a line that pins no --max-running-requests is unasserted, not defaulted",
            expected_max_running_requests(path) is None,
        )

        with open(path, "w") as f:
            f.write("python -m sglang.launch_server --context-length 1048576\n")
        try:
            expected_parallelism(path)
            case("a line with no --tp-size is unverified, never 4/4", False, "it returned a value")
        except ArgsFileUnreadable as e:
            case("a line with no --tp-size is unverified, never 4/4", True, str(e)[:140])

        absent = os.path.join(td, "absent-sglang-args.txt")
        for label, fn in (
            ("expected_parallelism", expected_parallelism),
            ("expected_max_running_requests", expected_max_running_requests),
            ("expected_context_length", expected_context_length),
        ):
            try:
                fn(absent)
                case(f"{label} on a missing file raises, not a literal", False, "it returned a value")
            except ArgsFileUnreadable as e:
                case(f"{label} on a missing file raises, not a literal", True, str(e)[:120])

    print()
    print("args-file pick (checks 1-2 read the copy's own line on a box with ENGINES copies):")
    table = {"a": {"port": 8002}, "b": {"port": 8012}}
    host_of = lambda e: f"box-c{e}.<DOMAIN>"
    legacy = lambda: "/workspace/tools/sglang-args.txt"
    pick = lambda cli, url, env, tbl=table: pick_args_file(cli, url, env, tbl, "/workspace/tools", host_of, legacy)
    case("--args-file wins", pick("/x.args", "http://127.0.0.1:8012", {}) == "/x.args")
    case("$ARGS_FILE wins over the copy pick", pick(None, "http://127.0.0.1:8012", {"ARGS_FILE": "/y"}) == "/y")
    case("$ENGINE picks its copy's file", pick(None, "http://127.0.0.1:8012", {"ENGINE": "a"}) == "/workspace/tools/sglang-args-a.txt")
    got = pick(None, "http://127.0.0.1:8012", {})
    case("a loopback base URL on copy b's port reads sglang-args-b.txt", got == "/workspace/tools/sglang-args-b.txt", got)
    got = pick(None, "https://box-ca.<DOMAIN>", {})
    case("copy a's door host reads sglang-args-a.txt", got == "/workspace/tools/sglang-args-a.txt", got)
    try:
        got = pick(None, "http://127.0.0.1:30000", {})
        case("a base URL naming no copy refuses, never the TP8 line", False, f"it returned {got}")
    except SystemExit as e:
        case("a base URL naming no copy refuses, never the TP8 line", "ARGS_FILE=" in str(e) and "ENGINE=" in str(e), str(e)[:140])
    got = pick(None, "http://127.0.0.1:30000", {}, {})
    case("a box with no ENGINES keeps the legacy default", got == "/workspace/tools/sglang-args.txt", got)

    print()
    print("args-file wiring (resolve_args_file over boxenv: ENGINES, engine_host, TOOLS_REMOTE):")
    saved = {k: os.environ.get(k) for k in ("BOX", "BOX_ENV", "ENGINE", "ARGS_FILE", "ENGINE_URL")}
    saved_cache = boxenv._CACHE

    def use_box(path):
        for k in saved:
            os.environ.pop(k, None)
        os.environ["BOX_ENV"] = path
        boxenv._CACHE = None

    try:
        with tempfile.TemporaryDirectory() as td:
            box = os.path.join(td, "engines.env")
            with open(box, "w") as f:
                f.write(
                    'ENGINES="a:0,1,2,3:8002:5557 b:4,5,6,7:8012:5567"   # two copies\n'
                    "HOSTNAME=wiretest.example.net\n"
                    "TOOLS_REMOTE=/opt/t # staged here\n"
                )
            use_box(box)
            got = resolve_args_file(None, "https://wiretestb.example.net")
            case("engine_host's door name picks the copy under TOOLS_REMOTE", got == "/opt/t/sglang-args-b.txt", got)
            got = resolve_args_file(None, "http://127.0.0.1:8002")
            case("an ENGINES port picks the copy", got == "/opt/t/sglang-args-a.txt", got)
            os.environ["ENGINE"] = "b"
            got = resolve_args_file(None, "https://wiretest.example.net")
            case("$ENGINE picks the copy whatever the URL", got == "/opt/t/sglang-args-b.txt", got)
            os.environ["ENGINE"] = "z"
            try:
                got = resolve_args_file(None, "http://127.0.0.1:8002")
                case("an $ENGINE not in ENGINES refuses", False, f"it returned {got}")
            except SystemExit as e:
                case("an $ENGINE not in ENGINES refuses", "unknown engine" in str(e), str(e)[:120])
            os.environ.pop("ENGINE")
            try:
                got = resolve_args_file(None, "https://wiretest.example.net")
                case("the fronting door (no copy) refuses checks 1-2", False, f"it returned {got}")
            except SystemExit as e:
                case("the fronting door (no copy) refuses checks 1-2", "ENGINE=" in str(e), str(e)[:120])
            got = args_file_for(set(ARGS_FILE_CHECKS), None, "https://wiretest.example.net")
            case("with checks 1-2 skipped the fronting door resolves nothing and runs", got is None, repr(got))
            try:
                got = args_file_for({"models"}, None, "https://wiretest.example.net")
                case("with only check 1 skipped check 2 still refuses", False, f"it returned {got}")
            except SystemExit:
                case("with only check 1 skipped check 2 still refuses", True)

        # The real box-c box file, when this tree has it: every copy's own door name and loopback port
        # must pick that copy's file -- the names engine_host derives, not invented ones.
        box_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boxes", "box-c.env")
        if os.path.isfile(box_file):
            use_box(box_file)
            table = boxenv.engines()
            root = (boxenv.load().get("TOOLS_REMOTE") or "/workspace/tools").rstrip("/")
            for eid, row in table.items():
                want = f"{root}/sglang-args-{eid}.txt"
                host = boxenv.engine_host(eid)
                got = resolve_args_file(None, f"https://{host}")
                case(f"box-c.env: {host} reads {want}", got == want, got)
                got = resolve_args_file(None, f"http://127.0.0.1:{row['port']}")
                case(f"box-c.env: port {row['port']} reads {want}", got == want, got)
            if not table:
                got = resolve_args_file(None, "http://127.0.0.1:8002")
                case("box-c.env sets no ENGINES: the legacy default", got == boxenv.args_file_default(), got)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        boxenv._CACHE = saved_cache

    print()
    print("SMOKE SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description="SGLang smoke battery for DeepSeek-V4.1-Flash (box-parametrized)")
    p.add_argument("--base-url", default=None, help="engine base URL (default: $ENGINE_URL, else http://127.0.0.1:<ENGINE_PORT> from the box file)")
    p.add_argument("--api-key", default=None, help="API key value (default: $VLLM_API_KEY, else read from --env-file, else the box's ENV_FILE/etc/environment)")
    p.add_argument(
        "--env-file", default=None,
        help="env file to read VLLM_API_KEY from first, ahead of the box-file chain, when --api-key/$VLLM_API_KEY are unset",
    )
    p.add_argument("--args-file", default=None, help="sglang launch-args file to read --context-length, --tp-size/--ep-size and --max-running-requests from (default: $ARGS_FILE, else on a box with ENGINES copies the copy's sglang-args-<id>.txt -- $ENGINE's, else the copy the base URL's port or host names, refusing when it names none and check 1 or 2 runs -- else the box file's ARGS_FILE, else /workspace/tools/sglang-args.txt). Unreadable: checks 1-2 FAIL saying they cannot verify, never a literal")
    p.add_argument("--model", default=None, help="served model id (default: $SERVED_MODEL, else the box file's SERVED_MODEL)")
    p.add_argument("--reps", type=int, default=3, help="greedy/sampled reps (default: %(default)s)")
    p.add_argument(
        "--skip",
        default="",
        help="comma-separated check names to skip: " + ",".join(CHECK_NAMES),
    )
    p.add_argument("--metrics", action="store_true", help="also print matching /metrics lines once")
    p.add_argument(
        "--dry-run", "--plan", dest="dry_run", action="store_true",
        help="print the ordered check list and exit 0 without resolving a box, an API key, "
             "or calling the engine at all",
    )
    p.add_argument(
        "--self-test", dest="self_test", action="store_true",
        help="offline regression test: decode and PNG-walk the embedded fallback image "
             "(including a deliberately truncated one, which must be rejected) and drive the "
             "launch-args readers against a temporary file, a missing one and one without "
             "--tp-size; then the per-copy args-file pick, against a temporary ENGINES box file "
             "and the tracked boxes/box-c.env (read only, when present); exits 0/1 "
             "without touching an engine or the network",
    )
    args = p.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    if args.dry_run:
        print("dry-run: smoke.py would run these checks, in order (skip any with --skip):")
        for i, name in enumerate(CHECK_NAMES, start=1):
            print(f"  [{i}] {name}")
        return

    base = boxenv.resolve_engine_url(args.base_url)
    print(f"base_url={base}")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(CHECK_NAMES)
    if unknown:
        raise SystemExit(f"unknown --skip name(s): {sorted(unknown)} (valid: {CHECK_NAMES})")

    key = args.api_key
    if not key and args.env_file and os.path.exists(args.env_file):
        with open(args.env_file) as f:
            for line in f:
                if line.startswith("VLLM_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    key = boxenv.resolve_api_key(key)

    model = boxenv.resolve_served_model(args.model)
    args_file = args_file_for(skip, args.args_file, base)
    results = []

    run(1, ARGS_FILE_CHECKS[0], skip, lambda: check_models(base, key, model, args_file), results)
    run(2, ARGS_FILE_CHECKS[1], skip, lambda: check_server_info(base, key, args_file), results)
    run(3, "hang_probe", skip, lambda: check_hang_probe(base, key, model), results)
    run(4, "reasoning", skip, lambda: check_reasoning(base, key, model), results)
    run(5, "tool_call", skip, lambda: check_tool_call(base, key, model), results)
    run(6, "json_object", skip, lambda: check_json_object(base, key, model), results)
    run(7, "needle", skip, lambda: check_needle(base, key, model), results)
    run(8, "image", skip, lambda: check_image(base, key, model), results)
    run(9, "greedy", skip, lambda: check_greedy(base, key, model, args.reps), results)
    run(10, "sampled", skip, lambda: check_sampled(base, key, model, args.reps), results)

    if args.metrics:
        print_metrics(base, key)

    ok = bool(results) and all(results)
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
