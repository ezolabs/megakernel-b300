#!/usr/bin/env python3
"""Shared box-parametrization for every Python probe under tools/.

Loads the box file named by $BOX_ENV, or boxes/$BOX.env next to this
directory when $BOX_ENV is unset (KEY=VALUE lines, blank lines and
#-comments ignored -- the same file lib.sh sources for the shell side),
into a dict, and exposes the handful of values a probe needs:

    engine_url()    -- $ENGINE_URL, else http://127.0.0.1:<ENGINE_PORT>
    door_url()      -- $DOOR_URL, else https://<HOSTNAME>:<APP_PORT_EXT>
    served_model()  -- $SERVED_MODEL, else the box file's SERVED_MODEL
    api_key()       -- $VLLM_API_KEY, else /etc/environment on the box,
                        else the box file's ENV_FILE (the dev box's secret
                        file, KEY=VALUE lines)
    box_name()      -- $BOX, or None
    out_path(probe, tag) -- where a probe should write its output file
    ssh_proxy_option() -- ['-o', 'ProxyCommand=...'] for every ssh a probe runs when this
                        session carries a proxy variable (a sandboxed worker); [] otherwise

On a box whose file sets ENGINES (box-c's 2xTP4: "a:0,1,2,3:8002:5557 b:4,5,6,7:8012:5567"),
$ENGINE=<id> points engine_url(), door_url(), serve_log_default() and args_file_default() at that
engine, the names lib.sh's `engine_select <id>` exports (port, <first label><id>.<domain>,
sglang-serve-<id>.log, sglang-args-<id>.txt). An unknown id, or an id on a box without ENGINES,
refuses; ENGINE unset is the legacy one-engine layout, unchanged.

A probe never hardcodes an ip, a port, a hostname or a box name -- those
live in the box file, and a fresh box is a new file under boxes/, nothing
else in this tool set changes. Every probe also takes --base-url /
--door-url CLI overrides on top of this (see resolve_engine_url /
resolve_door_url below), so a one-off run against an unlisted host never
needs a throwaway box file.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys

# Sent by every probe: Cloudflare's bot rule blocks Python's default
# urllib/requests User-Agent at a front door; harmless on loopback, required
# the moment any of these run through a door instead.
CURL_UA = "curl/8.7.1"


class BoxEnvError(SystemExit):
    """Raised (as a SystemExit) when the box file or a value it must supply
    is missing. A probe that hits this should exit non-zero with the
    message, not print a traceback."""


def _box_env_path() -> str:
    env = os.environ.get("BOX_ENV")
    if env:
        return env
    box = os.environ.get("BOX")
    if not box:
        raise BoxEnvError(
            "boxenv: no box selected -- set BOX=<name> (a file under "
            "tools/boxes/<name>.env) or BOX_ENV=<path to an env file>"
        )
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(tools_dir, "boxes", f"{box}.env")


def _parse_env_value(rest: str) -> str:
    """Extract the shell-assignment value from the raw text after `KEY=`, the
    way bash itself parses it when lib.sh sources this same file (`set -a;
    . "$BOX_ENV"`): single/double quotes are consumed as delimiters (their
    contents -- including any `#` -- kept literally, no escape processing
    beyond that), and the first UNQUOTED whitespace ends the value outright,
    so a trailing `# comment` (or anything else out there) is simply never
    part of it. A `#` glued to the value with no preceding whitespace is not
    a comment -- it is bash's own rule that a word only starts a comment at
    a word boundary, and by the time an unquoted char is reached here we are
    always mid-word (right after `=`, or inside the value already)."""
    out = []
    in_squote = False
    in_dquote = False
    for c in rest:
        if in_squote:
            if c == "'":
                in_squote = False
            else:
                out.append(c)
            continue
        if in_dquote:
            if c == '"':
                in_dquote = False
            else:
                out.append(c)
            continue
        if c == "'":
            in_squote = True
            continue
        if c == '"':
            in_dquote = True
            continue
        if c in " \t":
            break
        out.append(c)
    return "".join(out)


def _parse_env_file(path: str) -> dict:
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, rest = line.partition("=")
            out[k.strip()] = _parse_env_value(rest)
    return out


_CACHE: dict | None = None


def load() -> dict:
    """The box file as a dict, parsed once and cached for the process."""
    global _CACHE
    if _CACHE is None:
        path = _box_env_path()
        if not os.path.isfile(path):
            raise BoxEnvError(f"boxenv: no box file at {path}")
        _CACHE = _parse_env_file(path)
    return _CACHE


def _get(key: str, default=None):
    return load().get(key, default)


def box_name() -> str | None:
    return os.environ.get("BOX") or _get("BOX")


# ------------------------------------------------------------- engines (more than one per box)
# The box file MAY set ENGINES="a:0,1,2,3:8002:5557 b:4,5,6,7:8012:5567" (id:gpus:engine_port:kv_base,
# lib.sh's engines_check is the shell twin and refuses the same things). $ENGINE=<id> picks one; unset
# or empty is the legacy one-engine layout, where every function below returns exactly what it did
# before ENGINES existed.
_ENGINE_RE = re.compile(r"^([a-z][a-z0-9]*):((?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*):(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")

def engines() -> dict:
    """ENGINES parsed and validated: {id: {"gpus": [...], "port": int, "kv": int}}, in file order.
    {} when the box file sets none. Refuses a malformed entry, a repeated id, a shared GPU and any
    port (engine, engine+100, kv, kv+1, kv+2) used twice -- lib.sh's refusals, the same words."""
    raw = (_get("ENGINES") or "").split()
    out: dict = {}
    gpu_owner: dict = {}
    port_owner: dict = {}
    for e in raw:
        m = _ENGINE_RE.match(e)
        if not m:
            raise BoxEnvError(f"boxenv: ENGINES: malformed entry '{e}' (want id:gpu,gpu,...:engine_port:kv_base, e.g. a:0,1,2,3:8002:5557)")
        eid, gpus, port, kv = m.group(1), [int(g) for g in m.group(2).split(",")], int(m.group(3)), int(m.group(4))
        if eid in out:
            raise BoxEnvError(f"boxenv: ENGINES: engine id '{eid}' appears twice")
        for g in gpus:
            if g in gpu_owner:
                raise BoxEnvError(f"boxenv: ENGINES: GPU {g} is in engine '{gpu_owner[g]}' and engine '{eid}': two engines never share a GPU")
            gpu_owner[g] = eid
        for p in (port, port + 100, kv, kv + 1, kv + 2):
            if not 1 <= p <= 65535:
                raise BoxEnvError(f"boxenv: ENGINES: engine '{eid}' derives port {p}, outside 1-65535")
            if p in port_owner:
                raise BoxEnvError(f"boxenv: ENGINES: port {p} is used twice (engine '{port_owner[p]}' and engine '{eid}')")
            port_owner[p] = eid
        out[eid] = {"gpus": gpus, "port": port, "kv": kv}
    return out

def engine_id() -> str | None:
    """$ENGINE, validated against the box file's ENGINES; None for the legacy layout."""
    want = os.environ.get("ENGINE") or None
    if want is None:
        return None
    table = engines()
    if not table:
        raise BoxEnvError(f"boxenv: ENGINE={want} named, but the box file sets no ENGINES: this box runs one legacy engine -- unset ENGINE")
    if want not in table:
        raise BoxEnvError(f"boxenv: unknown engine '{want}' (ENGINES names: {' '.join(table)})")
    return want

def engine_host(eid: str) -> str:
    """<first label><id>.<rest of HOSTNAME>: box-ca.<DOMAIN> for id a."""
    hostname = _get("HOSTNAME") or ""
    if "." not in hostname:
        raise BoxEnvError(f"boxenv: engine '{eid}': HOSTNAME={hostname!r} has no domain to derive the engine's host name from")
    first, _, rest = hostname.partition(".")
    return f"{first}{eid}.{rest}"

def engine_url() -> str:
    eid = engine_id()
    if eid is not None:
        url = f"http://127.0.0.1:{engines()[eid]['port']}"
        env = (os.environ.get("ENGINE_URL") or "").rstrip("/")
        if env and env != url:
            raise BoxEnvError(f"boxenv: ENGINE={eid} is {url} but $ENGINE_URL says {env} -- unset one (lib.sh's engine_select {eid} exports the matching pair)")
        return url
    env = os.environ.get("ENGINE_URL")
    if env:
        return env.rstrip("/")
    port = _get("ENGINE_PORT")
    if not port:
        raise BoxEnvError("boxenv: ENGINE_PORT missing from the box file and $ENGINE_URL unset")
    return f"http://127.0.0.1:{port}"


def door_url() -> str:
    eid = engine_id()
    if eid is not None:
        # The engine's own name ($ENGINE_DOOR_URL, else derived); $DOOR_URL keeps naming the box's
        # legacy hostname, which fronts every copy, so it is not this engine's door.
        env = os.environ.get("ENGINE_DOOR_URL")
        if env:
            return env.rstrip("/")
        port = _get("APP_PORT_EXT")
        if not port:
            raise BoxEnvError("boxenv: APP_PORT_EXT missing from the box file and $ENGINE_DOOR_URL unset")
        return f"https://{engine_host(eid)}:{port}"
    env = os.environ.get("DOOR_URL")
    if env:
        return env.rstrip("/")
    hostname = _get("HOSTNAME")
    port = _get("APP_PORT_EXT")
    if not hostname or not port:
        raise BoxEnvError(
            "boxenv: HOSTNAME/APP_PORT_EXT missing from the box file and $DOOR_URL unset"
        )
    return f"https://{hostname}:{port}"


def served_model() -> str:
    env = os.environ.get("SERVED_MODEL")
    if env:
        return env
    model = _get("SERVED_MODEL")
    if not model:
        raise BoxEnvError("boxenv: SERVED_MODEL missing from the box file and $SERVED_MODEL unset")
    return model


def _env_file_path() -> str | None:
    path = _get("ENV_FILE")
    if not path:
        return None
    return os.path.expanduser(path)


def api_key() -> str:
    """VLLM_API_KEY, in order: $VLLM_API_KEY, then /etc/environment on the
    box itself, then the box file's ENV_FILE (the dev box's secret file,
    KEY=VALUE lines, never echoed)."""
    env = os.environ.get("VLLM_API_KEY")
    if env:
        return env
    for path in ("/etc/environment", _env_file_path()):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("VLLM_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    raise BoxEnvError(
        "boxenv: no VLLM_API_KEY -- not in $VLLM_API_KEY, /etc/environment, or the box's ENV_FILE"
    )


def out_path(probe: str, tag: str) -> str:
    """Where a probe should write its output file: /workspace/out-<BOX>-
    <probe>-<tag>.txt on the box, or ./out-<BOX>-<probe>-<tag>.txt when
    /workspace does not exist (the dev box, or an ad hoc off-box run)."""
    box = box_name() or "unknown"
    name = f"out-{box}-{probe}-{tag}.txt"
    base_dir = "/workspace" if os.path.isdir("/workspace") else "."
    return os.path.join(base_dir, name)


# ------------------------------------------------------------- CLI helpers


def resolve_engine_url(cli_value: str | None = None) -> str:
    """--base-url (or equivalent) wins; else engine_url()."""
    return (cli_value or engine_url()).rstrip("/")


def resolve_door_url(cli_value: str | None = None) -> str:
    """--door-url wins; else door_url()."""
    return (cli_value or door_url()).rstrip("/")


def resolve_api_key(cli_value: str | None = None) -> str:
    """--api-key (or equivalent) wins; else api_key()."""
    if cli_value:
        return cli_value
    return api_key()


def resolve_served_model(cli_value: str | None = None) -> str:
    return cli_value or served_model()


def serve_log_default() -> str:
    """The SGLang engine's own log file, tailed by probe-1m.py and read
    wholesale by loganal.py -- an INPUT, not something a probe writes.
    $SERVE_LOG, else the box file's SERVE_LOG, else the deployment-wide
    default path every box writes its serve log to.
    With $ENGINE=<id>: $SERVE_LOG, else /workspace/sglang-serve-<id>.log (lib.sh's ENGINE_SERVE_LOG)."""
    eid = engine_id()
    if eid is not None:
        return os.environ.get("SERVE_LOG") or f"/workspace/sglang-serve-{eid}.log"
    return os.environ.get("SERVE_LOG") or _get("SERVE_LOG") or "/workspace/sglang-serve.log"


def args_file_default() -> str:
    """The engine's launch-args file, read by smoke.py to learn the
    --context-length the box is actually configured to serve. $ARGS_FILE,
    else the box file's ARGS_FILE, else the deployment-wide default path.
    With $ENGINE=<id>: $ARGS_FILE, else <TOOLS_REMOTE>/sglang-args-<id>.txt (lib.sh's ENGINE_ARGS_FILE)."""
    eid = engine_id()
    if eid is not None:
        return os.environ.get("ARGS_FILE") or f"{_get('TOOLS_REMOTE') or '/workspace/tools'}/sglang-args-{eid}.txt"
    return os.environ.get("ARGS_FILE") or _get("ARGS_FILE") or "/workspace/tools/sglang-args.txt"


# ------------------------------------------------------------- the session ssh proxy
# A Python probe's ssh needs the transport lib.sh gives box_ssh/box_scp/fork_ssh. In a
# sandboxed session there is no direct route to the box: a bare
# `ssh root@<BOX_IP> ...` dies at once with "connect to host ... Operation not permitted" -- the sandbox
# refusing the connect, not the host being down, which is why the message has to be read before the
# host is declared dead. The only way out is the session HTTP proxy the environment already carries
# in $https_proxy (`http://srt.<id>:<password>@localhost:<port>`), spoken as an HTTP CONNECT tunnel
# carrying `Proxy-Authorization: Basic`, and ssh's own mechanism for that is -o ProxyCommand=<cmd>.
#
# That program already exists in this tree, and it is not written twice: it is the standing file
# `ssh-box-proxy.py` beside this one, and lib.sh mints the SAME file's bytes into $TMPDIR for its own
# ssh calls. `_proxy_helper_source()` below reads that file (never a heredoc, never a copy of its
# text here) so a probe's ssh and a script's ssh run one program; a second copy would be a second
# implementation of the CONNECT handshake to keep in step.
#
# ssh_proxy_option() returns ssh's two argv words -- ['-o', 'ProxyCommand=python3 -I <helper> %h %p
# <var>'] -- or [] when there is no proxy variable to tunnel through (an unsandboxed session, where
# ssh reaches the host directly and every ssh line must stay exactly what it was). Splice it into a
# command: cmd = ["ssh", *ssh_proxy_option(), "-o", "BatchMode=yes", ..., host, remote_cmd].
#
# The URL is a credential: it never appears in the option, only the NAME of the variable that holds
# it -- the helper reads the value from its own inherited environment, and `python3 -I` keeps a
# PYTHONPATH or user site out of that program. `%h %p` are ssh's own tokens, expanded by ssh for the
# host and port it settled on (a `root@<BOX_IP>`-style target has the user stripped before ssh applies
# them); ssh hands the ProxyCommand to /bin/sh, so the helper's path is shlex-quoted -- this repo
# lives under a path with a space in it. Everything a scheduler needs to know is in the returned
# option; nothing else about an ssh command line changes.
#
# Every refusal here -- a proxy variable set that no option can be built from -- is a warning on
# stderr and an empty option list, never an exception. No proxy variable at all is not a refusal: []
# and no warning, the bare command line being the right one there. A probe whose ssh then fails
# reports the sandbox's own message, which the warning has just explained.
PROXY_VARS = ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY")
_PROXY_HELPER_PATH: "str | None" = None    # the helper program, once written for this process
_PROXY_HELPER_DONE = False                 # tried already? one attempt, one warning, per process


def _proxy_warn(msg: str) -> None:
    print(f"boxenv: session proxy: {msg}", file=sys.stderr)


def ssh_proxy_var() -> "str | None":
    """The first proxy variable the ENVIRONMENT carries, by NAME only -- never by value, which is
    where the password is. os.environ is what a child process (ssh, and the helper ssh starts)
    inherits, so a name present there is a proxy that can actually carry the tunnel; an unset or
    empty one is not, and the order is lib.sh's own (box_ssh_proxy_var)."""
    for var in PROXY_VARS:
        if os.environ.get(var):
            return var
    return None


# The interpreter the option names is looked for on every call, but a missing one is explained ONCE
# per process -- lib.sh's box_ssh_proxy_setup() asks the same question once, at source time.
_PYTHON3_OFF_PATH_WARNED = False


def _python3_on_path(var: str) -> bool:
    """Is the `python3` this option tells ssh to run (`ProxyCommand=python3 -I <helper> ...`) on PATH?
    ssh hands that string to /bin/sh, which resolves the word against the environment ssh itself
    inherited -- exactly the PATH shutil.which reads -- so a missing one means the ProxyCommand can
    only fail, with sh's own `python3: not found`, which reads as a connect error rather than as the
    sandbox refusing the connect. False makes ssh_proxy_option() refuse the option and says why once
    per process, naming `$var` the way lib.sh's box_ssh_proxy_setup() does: a probe that ssh's ten
    times prints one warning, not ten, and a session without python3 still runs its ssh, just without
    the tunnel. lib.sh carries the identical guard for the shell side."""
    global _PYTHON3_OFF_PATH_WARNED
    if shutil.which("python3"):
        return True
    if not _PYTHON3_OFF_PATH_WARNED:
        _PYTHON3_OFF_PATH_WARNED = True
        _proxy_warn(f"${var} is set but python3 is not on PATH -- ssh calls run with no ProxyCommand")
    return False


def _proxy_helper_source() -> "str | None":
    """The ProxyCommand program: `ssh-box-proxy.py` beside this file, whole, exactly as lib.sh copies
    it into $TMPDIR for its own ssh calls. ONE copy of the tunnel program in the tree, not two -- the
    reason this reads that file rather than carrying its own text (or reading one out of a heredoc in
    lib.sh, the arrangement #270 retired). Read in binary and decoded, so what gets minted is that
    file's bytes: a text-mode read would rewrite a `\\r\\n` on the way in and silently hand ssh a
    program that differs from the one lib.sh mints. Returns None (after `_proxy_warn`) for any of the
    four unusable shapes: the file is not there; its bytes are not UTF-8 text (refused by the
    `except UnicodeDecodeError` branch below, whose own warning line is `boxenv: session proxy:
    <path> is not UTF-8 text (<exc>) -- ssh calls run with no ProxyCommand`); it does not look like
    that program; or it does not compile as one."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssh-box-proxy.py")
    try:
        with open(path, "rb") as f:
            source = f.read().decode("utf-8")
    except OSError as exc:
        _proxy_warn(f"cannot read {path} ({exc}) -- ssh calls run with no ProxyCommand")
        return None
    except UnicodeDecodeError as exc:
        _proxy_warn(f"{path} is not UTF-8 text ({exc}) -- ssh calls run with no ProxyCommand")
        return None
    if not source.startswith("#!/usr/bin/env python3") or "Proxy-Authorization" not in source:
        _proxy_warn(f"{path} is not the ssh ProxyCommand program -- ssh calls run with no ProxyCommand")
        return None
    try:
        compile(source, path, "exec")
    except SyntaxError as exc:
        _proxy_warn(f"the ssh ProxyCommand program in {path} does not compile ({exc}) -- ssh calls run with no ProxyCommand")
        return None
    return source


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _ssh_proxy_helper() -> "str | None":
    """That program, written to a private 0600 temp file for this process, and its path. mkstemp:
    one unique file per process, so a concurrent probe can never be handed another run's helper, and
    atexit removes it -- so the option belongs to the process that runs ssh: minting it in one
    process and ssh'ing from another would hand ssh a path that is already gone. Cached: the option
    may be asked for on every ssh a probe makes, the program is written once."""
    global _PROXY_HELPER_PATH, _PROXY_HELPER_DONE
    if _PROXY_HELPER_DONE:
        return _PROXY_HELPER_PATH
    _PROXY_HELPER_DONE = True
    source = _proxy_helper_source()
    if source is None:
        return None
    import atexit
    import tempfile

    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    try:
        fd, path = tempfile.mkstemp(prefix="boxenv-ssh-proxy-", suffix=".py", dir=tmpdir)
        with os.fdopen(fd, "wb") as fh:
            fh.write(source.encode("utf-8"))
    except OSError as exc:
        _proxy_warn(f"cannot write the ssh ProxyCommand helper into {tmpdir} ({exc}) -- ssh calls run with no ProxyCommand")
        return None
    atexit.register(_remove_quietly, path)
    _PROXY_HELPER_PATH = path
    return path


def ssh_proxy_option() -> list[str]:
    """ssh's session-proxy argument pair for this environment, or [] when there is no usable one.
    Splice it into every ssh argv a probe builds:
        cmd = ["ssh", *ssh_proxy_option(), "-o", "BatchMode=yes", host, remote_cmd]
    [] is SILENT when the session carries no proxy variable at all (unset, or set to "" -- the
    unsandboxed session, where the bare command line is right and there is nothing to warn about).
    Once a variable IS set, each refusal is a warning naming the reason and an empty list: a URL this
    HTTP CONNECT tunnel cannot speak (a `scheme://` URL whose scheme is not `http` -- a socks5h://
    proxy is exactly what the tunnel exists instead of -- or a value that names no proxy host), no
    python3 on PATH for the ProxyCommand to run, and a helper that cannot be read or written: the
    sibling `ssh-box-proxy.py` missing, its bytes not UTF-8 text (its own warning line, `boxenv:
    session proxy: <path> is not UTF-8 text (<exc>) -- ssh calls run with no ProxyCommand`), not the
    tunnel program or not compiling, or `$TMPDIR` unwritable. The URL is judged on each call, so its
    warning repeats; the missing interpreter and the unusable helper are explained once per process.
    See the section comment above
    for why the value never appears in the option."""
    var = ssh_proxy_var()
    if var is None:
        return []
    url = os.environ.get(var) or ""
    if "://" in url:
        scheme = url.split("://", 1)[0]
        if scheme != "http":
            _proxy_warn(f"${var} is {scheme}://, which this tunnel cannot speak (it is an HTTP CONNECT client) -- ssh calls run with no ProxyCommand")
            return []
    elif ":" not in url:
        _proxy_warn(f"${var} names no proxy host -- ssh calls run with no ProxyCommand")
        return []
    if not _python3_on_path(var):
        return []
    helper = _ssh_proxy_helper()
    if not helper:
        return []
    # `%%h %%p` are literal text here (ssh's tokens); the var NAME is from the fixed list above.
    return ["-o", "ProxyCommand=python3 -I %s %%h %%p %s" % (shlex.quote(helper), var)]


# --------------------------------------------------------------------------
# self-test: parse a temp file with inline comments and quoted values, the
# exact shape that broke _parse_env_file before this fix (BOX=<box>
# python3 loganal.py --self-test printed base_url with a trailing "# SGLang,
# loopback only" comment baked into the URL and failed with "URL can't
# contain control characters").
# --------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile

    cases = {
        # plain, no comment
        "HOSTNAME": ("box-c.<DOMAIN>", "box-c.<DOMAIN>"),
        # the exact bug repro: value then a wide gap then a trailing comment
        "ENGINE_PORT": ("8002              # SGLang, loopback only", "8002"),
        # `#` glued to the value (no preceding whitespace) is literal, not a comment
        "GLUED": ("bar#baz", "bar#baz"),
        # `#` right after `=` (still glued, no whitespace yet) is also literal
        "LEADING_HASH": ("#not-a-comment", "#not-a-comment"),
        # double-quoted value: the `#` inside stays, quotes themselves are stripped
        "QUOTED": ('"hello # not a comment" # real comment', "hello # not a comment"),
        # single-quoted value: same, plus an embedded space
        "SINGLE": ("'a#b c' # trailing", "a#b c"),
        # a completely empty value
        "EMPTY": ("", ""),
    }
    lines = ["# a leading comment line, ignored", ""]
    for k, (raw, _) in cases.items():
        lines.append(f"{k}={raw}")
    content = "\n".join(lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        parsed = _parse_env_file(path)
    finally:
        os.unlink(path)

    ok = True
    for k, (raw, want) in cases.items():
        got = parsed.get(k)
        status = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{status}  {k}={raw!r:60} -> {got!r} (want {want!r})")

    print()
    print("BOXENV SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _sys

    if "--self-test" in _sys.argv[1:]:
        _sys.exit(_self_test())
    print(__doc__)
