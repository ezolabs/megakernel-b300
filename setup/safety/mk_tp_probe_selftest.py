"""CPU self-test of the box-a TP-consistency probe (mk_tp_probe) as the scheduler
drives it around request_receiver.recv_requests (fbf4d51dc5 + a5c32f373c).

Run with the fork's python/ on PYTHONPATH and SGLANG_MK_TP_PROBE=1,
SGLANG_MK_TP_PROBE_EVERY=1 (the module reads its env at import).
"""
import importlib
import logging
import os
import struct
import sys
import zlib
from types import SimpleNamespace

os.environ["SGLANG_MK_TP_PROBE"] = "1"
os.environ["SGLANG_MK_TP_PROBE_EVERY"] = "1"
os.environ.pop("SGLANG_MK_TP_PROBE_ASSERT", None)

from sglang.srt.managers.scheduler_components import mk_tp_probe as probe

probe = importlib.reload(probe)


# Request ids a log line must never carry: the probe only ships their crc32.
RIDS = ("rid-7f3a-content", "rid-9c1d-content")


def scheduler(rids=RIDS, seqlen=5, kv=100):
    reqs = [SimpleNamespace(rid=r, seqlen=seqlen) for r in rids]
    return SimpleNamespace(
        running_batch=SimpleNamespace(reqs=reqs),
        forward_ct=7,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: kv),
        metrics_reporter=SimpleNamespace(
            spec_total_num_accept_tokens=3,
            spec_num_accept_tokens=1,
            spec_total_num_forward_ct=2,
            spec_num_forward_ct=1,
        ),
        model_worker=SimpleNamespace(),
    )


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


cap = Capture()
probe.logger.addHandler(cap)
probe.logger.setLevel(logging.INFO)
failures = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


assert probe.ENABLED and probe.EVERY == 1

# 1. The snapshot has one value per field, and before_recv returns one every EVERY iterations.
s0 = scheduler()
p0 = probe.before_recv(s0)
check("before_recv returns a probe", isinstance(p0, probe.MkTpProbe))
check("state has one value per FIELDS name", len(p0.state) == len(probe.FIELDS))

# 2. Rank 0's probe rides recv_reqs; after_recv strips it and real requests pass through, in order.
real = [object(), object()]
out = probe.after_recv(s0, [real[0], p0, real[1]], p0, tp_rank=0)
check("after_recv strips the probe, keeps requests in order", out == real)
check("a matching state logs no mismatch", not any("MISMATCH" in m for m in cap.records))

# 3. A follower whose state differs: `differ=[...]` names exactly the differing fields, and the line
#    (which also prints both full states: counters and crc32s) carries no request id. Both checks can
#    fail: a probe that listed every field, or shipped rids instead of their crc, would trip them.
s1 = scheduler(kv=99)
mine = probe.MkTpProbe(probe._state(s1))
out = probe.after_recv(s1, [p0], mine, tp_rank=1)
mism = [m for m in cap.records if "MISMATCH" in m]
check("a differing follower logs MISMATCH", len(mism) == 1)
check("differ=[...] names kv_available and nothing else",
      "differ=[kv_available: mine=99 rank0=100]" in mism[0])
check("the mismatch line carries no request id (rids ride as crc32 only)",
      not any(rid in mism[0] for rid in RIDS))
check("the probe is still stripped on mismatch", out == [])

# 4. No probe received: the list passes through untouched.
lst = [object()]
check("no probe: list unchanged", probe.after_recv(s0, lst, None, 0) is lst)

# 5. on_decode_result folds accepted lengths and tokens order-independently by rid.
def fold(order):
    probe._decode_steps = probe._accept_crc = probe._token_crc = probe._path_crc = 0
    probe._mk_widths = frozenset()
    reqs = [SimpleNamespace(rid=r) for r in order]
    toks = {"a": [1, 2], "b": [3]}
    batch = SimpleNamespace(reqs=reqs, forward_mode=SimpleNamespace(value=2))
    result = SimpleNamespace(can_run_cuda_graph=True)
    probe.on_decode_result(None, batch, result, [toks[r] for r in order], rows=2)
    return (probe._decode_steps, probe._accept_crc, probe._token_crc, probe._path_crc)

a, b = fold(["a", "b"]), fold(["b", "a"])
check("rolling crcs are batch-order independent", a == b and a[0] == 1)

# An int next_token_id (non-spec decode) folds as a one-token run, with a bare int forward_mode:
# compare against the crcs computed by hand, so a dropped or mis-packed token fails.
probe._decode_steps = probe._accept_crc = probe._token_crc = probe._path_crc = 0
probe._mk_widths = frozenset()
probe.on_decode_result(
    None,
    SimpleNamespace(reqs=[SimpleNamespace(rid="x")], forward_mode=1),
    SimpleNamespace(),
    [5],
    rows=1,
)
check("an int next_token_id folds as a one-token run", (
    probe._decode_steps,
    probe._accept_crc,
    probe._token_crc,
    probe._path_crc,
) == (
    1,
    zlib.crc32(b"x" + struct.pack("<i", 1)),
    zlib.crc32(struct.pack("<1q", 5)),
    zlib.crc32(struct.pack("<iiii", 1, False, 1, False)),
))

# 6. ASSERT mode raises on a mismatch.
os.environ["SGLANG_MK_TP_PROBE_ASSERT"] = "1"
probe = importlib.reload(probe)
try:
    probe.after_recv(s1, [probe.MkTpProbe(probe._state(s0))], None, tp_rank=1)
    check("ASSERT=1 raises on mismatch", False)
except AssertionError:
    check("ASSERT=1 raises on mismatch", True)

# 7. Off by default: without SGLANG_MK_TP_PROBE the scheduler never calls it.
os.environ.pop("SGLANG_MK_TP_PROBE")
probe = importlib.reload(probe)
check("disabled without SGLANG_MK_TP_PROBE", probe.ENABLED is False)

# 8. The scheduler wires it around recv_requests exactly as the commit intends.
src = open(sys.modules["sglang.srt.managers.scheduler_components.mk_tp_probe"].__file__.replace(
    "scheduler_components/mk_tp_probe.py", "scheduler.py")).read()
check("scheduler calls before_recv gated on ENABLED",
      "mk_tp_probe.before_recv(self) if mk_tp_probe.ENABLED else None" in src)
check("scheduler strips via after_recv before extending _deferred_input_requests",
      src.index("mk_tp_probe.after_recv(") < src.index("self._deferred_input_requests.extend(new_reqs)"))

print(f"{'FAILED' if failures else 'PASSED'}: {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
