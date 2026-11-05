# The SGLang fork, as a delta against upstream

The boxes served our SGLang fork, branch `prod`, at **`9a9ede7ae0183e7c43956395238e29c406d85fe8`** (the
commit of record; box-a ran it from 2026-10-31 17:42Z). That repository is private. This directory carries what a
reviewer needs instead: the whole delta from the upstream commit the fork sits on, the commit list, and how to
apply it.

| file | what |
|---|---|
| `fork-vs-upstream-3a3a8611a4.diff` | `git diff 3a3a8611a4 9a9ede7ae0` — 449 files, 64,558 insertions, 1,436 deletions, 2.7 MB |
| `fork-commits.txt` | `git log --no-merges --format='%h %ad %s' --date=short 3a3a8611a4..9a9ede7ae0` — 152 commits |
| `plugin-apply-check.txt` | the B300 patch's `git apply --check` result against plugin `54b4289` |

**The base.** `git merge-base 9a9ede7ae0183e7c43956395238e29c406d85fe8 upstream/main` is
**`3a3a8611a45f5a81fd07ffa6b0ed3ba33e7a4db9`** — the upstream `sgl-project/sglang` commit the fork's `prod` tip
still sits on. (The box files call it `FORK_BASE`; the fork's own history is cherry-picks on top of it, no merge.)

## Apply

```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang
git checkout 3a3a8611a45f5a81fd07ffa6b0ed3ba33e7a4db9
git apply /path/to/fork-vs-upstream-3a3a8611a4.diff
```

A three-way apply (`git apply --3way`) is the same delta with git's merge fallback; the plain form is what the
diff was cut for and is the one to try first. The tree this produces is the tree the boxes served — the fork's
`prod` tip at 2026-10-31 — not a side branch. Later `prod` commits (2026-11-04's `163591d9fc`, served by box-a
after the record's window) are **not** in this delta: the record covers `9a9ede7ae0`.

## What the delta is, in one paragraph

The fork's own work is the SGLang-side half of the B300/megakernel stack: EP1 support paths the plugin requires,
the TP-consistency probe (`SGLANG_MK_TP_PROBE*`, `patches/probe.…` — the probe's commit is
`../patches/merge-audit/probe-commit.patch`), the DSV4/DeepSeek-V4.1 engine changes (Engram host tables, the
exact decoder trim, SWA branch rebuild, DSpark draft-graph glue), the HiCache changes the `KV_OFFLOAD` path needs
(sidecar transfers over every source, the host eviction heap kept across calls), priority scheduling, the
short-prefill lane, and the `sbr-*` request-path work. Read `fork-commits.txt` for the per-commit list; the
per-merge "what each bought" table is `../../record/fork-and-config.md` §2, and each row's measured effect is in
`../../record/fork-and-config.md` §3 (the speed register).

## What is NOT here

- The git history's own objects for the private branch (only the diff and the commit list are).
- The fork's CI, its deploy key, and the private mirror path the boxes pull from.
- Anything after `9a9ede7ae0`; take a later tree from the fork itself, or re-diff against this base.
