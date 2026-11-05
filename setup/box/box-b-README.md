# box-b: box sheet

box-b is a copy of box-a's serving stack on a second 4× B300: one TP4 SGLang engine serving L3a
(`DeepSeek-V4.1-Flash`) as config X' (fork `prod`, at bring-up `9a9ede7ae0`, since 2026-11-04 17:42Z
`163591d9fc` (CANDFIX; the box file's `FORK_TREE` names what serves), + the megakernel plugin,
`args/stack-b300-box-a-mk2.args` + `env-box-a-mk2.txt`, the TP probe and `hangwatch.py` on). The user
asked for it on 2026-10-30 ("rent this 4xb300 box ASAP while it's available and set it up as box-b, same stack as
box-a"), to take box-c's traffic share before box-c is shut down. The bring-up record, every command and gate:
`../../box-c/qa/box-b-bringup-20261030/NOTES.md`. All times UTC.

> **STOPPED since 2026-11-05 01:57:49Z** (the provider `exited` 01:58:27Z; never destroyed), by the order given
> once both boxes ran the prod branch stably. box-a serves alone; rows `box-b` / `box-b-dev` stay pool members, parked by
> autopool (they keep the pools' `probe` designation, which the router reads only at config validation). Last tree
> served: fork `prod` `163591d9fc` + `stack-b300-box-a-mk2.args` + `env-box-a-mk2.txt` + the `/freeze_gc` timer. Way
> back, only on the user's word: `hostctl start instance <INSTANCE_ID>` (the slot may not come back), then autopool rejoins
> the rows once the doors answer. Stopped, it bills storage only. Record: `../../box-c/qa/box-b-shutdown-20261105/NOTES.md`.

> **Before the stop, box-b served beside box-a.** It served alone 2026-10-31 09:44Z to 16:21Z while box-a was stopped
> (`../../box-c/qa/box-a-maintenance-20261031/`). By the
> user's order of 2026-10-31, box-a is the test box and box-b the blue half of each roll: once a build is
> validated on box-a and box-a serves it, box-b is cut (window on `box-b` every pod + `box-b-dev`) and restarted on
> the same `prod` tree, so both serve one build; never both out. Never stop box-b without the user's word.
> box-b holds the pool probe designation (`box-b` / `box-b-dev`, #222), and the s5 failover counts any healthy
> door. Its roll onto `9a9ede7ae0`, with its own origin key (#135):
> `../../box-c/qa/box-b-roll-20261031/NOTES.md`.

> **Records.** Before the cut, the box's content-free records (serve-log metadata, hangwatch episodes, the
> installed args and env, onstart, `nvidia-smi -q`, the venv's package list, a `/metrics` snapshot) were copied
> to the Mac on 2026-11-05: `records-20261105/` in git, the bulk in
> `<HOME>/workspace/backups/box-b-20261105/`. What the box serves stands in its box file.

## Identity

| what | value |
| --- | --- |
| the provider | instance **<INSTANCE_ID>** (machine <MACHINE_ID>, host <HOST_ID>), label `box-b-dsv41-b300`, created 2026-10-30 16:44Z, contract end 2026-11-27T17:11Z |
| GPUs | 4× NVIDIA B300 SXM6 PC (sm_103), 275,040 MiB each, NV18 between every pair, 1100 W limit, 2032/3996 MHz max (box-a's SXM6 AC reads the same); driver 580.178.04 (box-a 580.173.02), CUDA 13.0 |
| host | AMD EPYC 9575F (256 threads, 2 NUMA nodes, every GPU on NUMA0), 3 TB RAM, 2,000 GB disk, everything under `/`; Ubuntu 24.04.4, kernel 6.8.0-139-generic, container hostname `<CONTAINER_HOSTNAME>` |
| image | `lmsysorg/sglang:dev-dsv41@sha256:4a5d132a…` (box-a's digest) |
| ssh | `root@<BOX_IP> -p <SSH_PORT>` (the direct mapping of container port 22). The provider's listed `ssh_port` is the proxy |
| front door | `https://box-b.<DOMAIN>:<APP_PORT_EXT>` (container 8001 → external <APP_PORT_EXT>), Caddy with a Let's Encrypt cert (notAfter 2027-01-28) behind the router allowlist; engine on loopback `:8002` |
| box file | [`boxes/box-b.env`](boxes/box-b.env) |
| engine | `max_total_num_tokens` 49,879,296 (box-a: 49,879,552); `/root/onstart.sh` = `build-onstart.sh` render for this box |
| watch | `watchdog.sh` (`/workspace/box-b-watchdog.log`), `hangwatch.py --service --on-hang kill` (`/workspace/mk29/watch/serving`), TP probe every 100 steps with assert |
| router rows | prod `box-b` in pool `box-c-prod` (config v262, live v264; KV and load feeds via the `box-b-kv` tunnel in sbr-prod), dev `box-b-dev` in `box-c-dev` (config v232, live v234); joined 2026-10-30 17:56Z (prod), 17:47Z (dev) |
| origin bearer | its own since #135 (sha256 prefix `036c9d8acdcb`; `/etc/environment` on the box, the `box-b` / `box-b-dev` rows re-sealed): `<HOME>/env/box-b.env` and the keyring on the Mac, `ENV_FILE` |
| standing order | stopped by the user's order (2026-11-05); never destroy <INSTANCE_ID>, start it again only on the user's word |

## How it was built

Copied, not rebuilt. The L3a repack (510,310,558,276 bytes, 56 files) came box to box from box-a and box-c at once
(2 GiB chunks, 24 ssh streams, 16:59 → 17:29Z) and passed `sha256sum -c` against box-a's manifest, 56/56.
`/opt/sglang-mk29`, the `/workspace/mk29` tree, plugin and harness, and the JIT and autotune caches were rsynced
from box-a, so the first boot compiled nothing (`new_cubins=0`). `stage.sh`, `cert-renew.sh`, `tls.sh` and
`dns.sh` ran as for any box. A temporary box-to-box key carried the copies and was removed afterwards.
