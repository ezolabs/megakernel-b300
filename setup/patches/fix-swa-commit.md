# Fix commit: swa_branch_rebuild.startup_refusal on upstream's ChainAllocator (2026-10-29, #28)

CPU and git work only. No engine, GPU or serving process was touched. `/workspace/mk29/sglang` (S4's engine
tree: `41bb61ec` plus S4's uncommitted patch) was read and never written.

## The bug
Upstream `ccc71fa5` (#46976, "Use one allocator for unified KV sub-pools") deleted
`sglang/srt/mem_cache/allocator/unified_hybrid_swa.py`, which held `UnifiedSWATokenToKVPoolAllocator`; its
successor is `sglang/srt/mem_cache/kv/alloc/chain.py`'s `ChainAllocator`. `startup_refusal` in
`python/sglang/srt/mem_cache/swa_branch_rebuild.py` still imported the old class inside the function. The
scheduler calls it at init (`managers/scheduler.py:1397/1407`) whenever `SGLANG_SWA_BRANCH_REBUILD` is on, so
MK-latest boot 1 died there with a ModuleNotFoundError (20:58:36Z). `git grep` at `a5c32f373c` finds no other
reference to `unified_hybrid_swa` or `UnifiedSWATokenToKVPoolAllocator` under `python/` or `test/`. (The
`unified_hybrid_swa` local in `kv_cache_builder.py:241` is a bool variable, not the module.)

The old class and `ChainAllocator` both subclass `BaseTokenToKVPoolAllocator`, not
`SWATokenToKVPoolAllocator`. So the `isinstance(allocator, ChainAllocator)` arm is belt and braces: the
`not isinstance(allocator, SWATokenToKVPoolAllocator)` arm in front of it already refuses either one. That
matches how the original guarded the unified class.

## The commit
| | |
|---|---|
| branch | `box-a-mk-latest` |
| parent | `a5c32f373c259a7b0f0e7e13d1804eb03f0c2447` (also the GitHub tip: `git ls-remote` read it with the box's deploy key after the commit, so the push fast-forwards) |
| commit | **`9ec65dacaf5269d412a4098dd905debe3e05ed30`** `box-a-mk <box-a-mk@localhost>` |
| subject | `Import ChainAllocator in swa_branch_rebuild.startup_refusal (upstream #46976 deleted unified_hybrid_swa)` |
| files | `python/sglang/srt/mem_cache/swa_branch_rebuild.py` (+2 −4: S4's `raw/s4-mk-b1/fix-swa-branch-rebuild-import.patch` byte for byte, `git apply --check` clean), `test/srt/sbr/test_o_swa_branch_rebuild.py` (+59) |
| bundle | `/workspace/mk29/build/fix28/box-a-mk-latest-fix28.bundle` on box-a (prereq `a5c32f373c`, head `refs/heads/box-a-mk-latest` = `9ec65dac`, sha256 `fe91e1be2952ccbd1d1b990232732bcd5c4c36e210f6dda33ea4799403ed4785`). A copy sat in a scratch dir on the Mac |

The commit was made on the box in `/workspace/mk29/sglang-next`, which already held the tip. The Mac sandbox
could not clone the fork: `git clone `<SGLANG_FORK_URL>`` returned `Permission denied
(publickey)`.

### The test (added after `test_startup_gates`, with imports `importlib.util` and `ChainAllocator`)
`test_startup_refusal_on_the_chain_allocator`:
- asserts that `importlib.util.find_spec("sglang.srt.mem_cache.allocator.unified_hybrid_swa") is None`;
- builds the tree as a stand-in (`MagicMock(spec=UnifiedRadixCache)` with Full + SWA components, no mamba,
  and a `UnifiedTreeCore`-named core). It does not use the file's `_swa_cache`, because `_swa_cache` is red
  on upstream's `CacheInitParams` (#46991 removed `tree_components`), and so is the existing
  `test_startup_gates` that calls `startup_refusal` through it. That redness is why #21's CPU run never
  reached the import;
- uses `_gate_args()` for the args;
- checks three allocators. A real CPU `SWATokenToKVPoolAllocator` (new helper `_static_swa_allocator`) gives
  `None`. `MagicMock(spec=ChainAllocator)` gives `"the static SWA allocator"`. `object()` also gives
  `"the static SWA allocator"`.

## Test output (box-a, `/opt/sglang-mk29/bin/python -m pytest -q -p no:cacheprovider -W ignore`,
`PYTHONPATH=/workspace/mk29/sglang-next/python`, `CUDA_VISIBLE_DEVICES=9` (no such GPU), `timeout 1200 nice -n 19 ionice -c3`; box logs `/workspace/mk29/build/fix28/run.log`, `commit.log`)
```
## RED: HEAD a5c32f373c unpatched swa_branch_rebuild.py + the new test
E       ModuleNotFoundError: No module named 'sglang.srt.mem_cache.allocator.unified_hybrid_swa'
python/sglang/srt/mem_cache/swa_branch_rebuild.py:249: ModuleNotFoundError
FAILED test/srt/sbr/test_o_swa_branch_rebuild.py::test_startup_refusal_on_the_chain_allocator
1 failed in 10.59s
PATCH-APPLIED
EQUALS S4's patched box file
## GREEN: patched
1 passed in 10.29s
## whole file, patched
27 failed, 10 passed in 13.61s
## distinct E lines of the whole-file run (patched):
     27 E TypeError: CacheInitParams.__init__() got an unexpected keyword argument 'tree_components'
```
The whole file was 27 failed and 9 passed at `41bb61ec` (`unit-tests.log`). After the commit it is the same
27 reds, every one the pre-existing `tree_components` error, plus the new green test. No other test file
references `swa_branch_rebuild`.

## Box state after (box-a)
| path | state |
|---|---|
| `/workspace/mk29/sglang-next` | branch `box-a-mk-latest`, HEAD `9ec65dacaf52`, `git status --short` empty. Its `swa_branch_rebuild.py` sha256 is `8b975005b4ae29ac…`, identical (`cmp`) to S4's patched copy |
| `/workspace/mk29/sglang` | untouched: HEAD `41bb61ec53`, still ` M python/sglang/srt/mem_cache/swa_branch_rebuild.py` (S4's patch, sha256 `8b975005b4ae29ac…`) |
| `/workspace/mk29/build/fix28/` | `fix.patch`, `test_o.new.py`, `run.sh`/`run.log`, `commit.sh`/`commit.log`, the bundle |

## Push (not run from the worker sandbox)
The GitHub tip was `a5c32f373c` when this was written. The push must be fast-forward only, never `--force`.
Either of these works:
- from the box, with the fork deploy key that box-e-up carries:
  `ssh -p <SSH_PORT> root@<BOX_IP> 'cd /workspace/mk29/sglang-next && GIT_SSH_COMMAND="ssh -i /root/.ssh/box_c_fork_deploy -o IdentitiesOnly=yes" git push `<SGLANG_FORK_URL>` 9ec65dacaf5269d412a4098dd905debe3e05ed30:refs/heads/box-a-mk-latest'`
- from any clone that holds `a5c32f373c`:
  `git fetch /workspace/mk29/build/fix28/box-a-mk-latest-fix28.bundle box-a-mk-latest && git push `<SGLANG_FORK_URL>` 9ec65dacaf5269d412a4098dd905debe3e05ed30:refs/heads/box-a-mk-latest`

Once it is pushed, `/workspace/mk29/sglang` can be fast-forwarded to `9ec65dac`. Its dirty file equals the
committed one, so `git stash`/`checkout` would drop nothing new. That step belongs to whoever owns the engine
session there.
