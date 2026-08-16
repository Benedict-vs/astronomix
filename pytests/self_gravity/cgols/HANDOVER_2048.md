# CGOLS 2048² × 4096 multi-node run — handover

**Written 2026-08-16. Intended to be picked up from ~September 2026, on HoreKa 2.**

Everything needed to run the 2048 campaign is implemented, committed and merged to
`main` (merge `7cae987`). Nothing has been run on a cluster yet. This file is the
complete set of instructions — you should not need the conversation that produced it.

The decision taken on 2026-08-16 was: **do not start on legacy HoreKa; wait for
HoreKa Jade (`gpu-b200`, expected 09/2026).** The reasoning is in §7.

---

## 1. What the run is

Schneider & Robertson 2018 (arXiv:1803.01008) CGOLS galactic-wind replication,
**adiabatic A-series, no cooling**, at 2048 × 2048 × 4096 cells in a 10 × 10 × 20 kpc
box, 75 Myr. This is 8× the cells of the completed 1024 run (which took 61.5 h on
4× H200 in one node, ~246 GPU-h).

It no longer fits in one node, so cgols was moved from *one process driving N devices*
to **one process per GPU under `srun`**.

---

## 2. What exists (all on `main`)

### Library changes (`astronomix/`) — affect all fork users

| file | change |
|---|---|
| `data_classes/simulation_helper_data.py` | `helper_data.r` built analytically from broadcast 1D axes (`_radius_from_axes`) instead of forcing the full meshgrid (208 GB at 2048, twice), and **sharded** (`_field_sharding`). With a sharding the build runs under `jax.jit(out_shardings=…)` so each process materialises only its shard. |
| `time_stepping/time_integration.py` | new `_place_params_on_mesh`, used at **both** placement sites: shards `params.gravitational_potential` (68.7 GB/device at 2048 if replicated) and restores `jnp.asarray` on replicated `device_put`. |
| `parallel/rank.py` (new) | `is_primary()` / `barrier(name)` / `broadcast_from_primary()` / `process_index()` / `process_count()`. All no-ops single-process. |
| `time_stepping/_progress_bar.py` | rank-guards the host side of both callbacks. |
| `_snapshotting/_orbax_storage.py` | `latest_step` is now barrier + rank-0 scan + broadcast. Module docstring documents the multi-process contract. |

### cgols changes (`pytests/self_gravity/cgols/cgols.py`)

- srun-aware import order; `DISTRIBUTED` auto-detected from `SLURM_NTASKS > 1` (or
  `CGOLS_DISTRIBUTED=1`). A bare `python cgols.py` is byte-identical to before.
- Auto near-square shard split when `CGOLS_SHARD_SPLIT` is unset:
  32 → `(1,8,4,1)`, 64 → `(1,8,8,1)`, 128 → `(1,16,8,1)`. z always unsharded.
- `_validate_shard_split()` enforces the divisibility rules (see §6).
- `CGOLS_IC_MODE=file|insitu`; **`insitu` whenever `DIM ≥ 2048` or distributed**, so
  *no IC prebuild is ever needed for this ladder* — `run_ic_horeka.sh` is not used.
- `RANK0` / `rprint`, rank guards + barriers on every host-side side effect.

### Scripts

| script | purpose |
|---|---|
| `_site_env.sh` | **sourced, not run.** Machine-specific env behind a `CGOLS_SITE` switch (`horeka` \| `jupiter` \| `local`). Provides `cgols_preflight`, `cgols_check_workspace`, `start_gpu_logger`/`stop_gpu_logger`. |
| `run_dist_sanity.sh` | ladder rungs 1–2. Runs `pytests/_dist_sanity.py`. |
| `run_cgols_multinode.sh` | the real launcher. Everything else delegates to it. |
| `run_cgols_bench.sh` | fixed-step calibration bench (rungs 6–7). |
| `run_horeka_2048.sh` | production parameter set; `exec`s the multinode launcher. |

`run_horeka_1024.sh` and `run_ic_horeka.sh` are the **legacy 1024 runners, deliberately
untouched** so that run stays reproducible. Do not model new work on them.

Every job's `.out` opens with a preflight banner printing jax / jaxlib / orbax / ptxas
versions, the git commit and branch, and `nvidia-smi -L`. **Read it on the first job.**

---

## 3. MUST CHANGE before running on HoreKa 2

All four runners were written against **legacy** HoreKa and are wrong on HoreKa 2.
Verified 2026-08-16 against <https://docs.nhr.kit.edu/clusters/horeka-2/>.

### 3.1 Partition names — in all four scripts

| scripts currently say | HoreKa 2 | hardware |
|---|---|---|
| `accelerated-h200` | `gpu-h200` | Ruby: 13 nodes × 4 H200-141GB, 4× IB NDR400 |
| `accelerated-h100` | `gpu-h100` | Teal: 22 nodes × 4 H100-94, 4× IB NDR200 |
| `accelerated-h200-8` | folded into Teal | the single 8× H200 node |
| `dev_accelerated-h100` | `dev-gpu-h100` | |
| `dev_accelerated` (A100/Green) | **gone** — but see below, it is no longer needed |
| — (new) | `gpu-b200` / `dev-gpu-b200` | **Jade: 75 nodes × 4 B200-186GB, 2 × Grace (ARM), 960 GiB host, 2× IB NDR200** |

HoreKa 2 has a **dev partition for every GPU type**: `dev-gpu-b200`, `dev-gpu-h200`,
`dev-gpu-h100`, `dev-cpu`. This is better than legacy, where only
`dev_accelerated-h100` (max 1 node) and `dev_accelerated` (A100) existed — the whole
cheap end of the ladder can now run on the *same hardware as production*, instead of
validating on A100s and hoping it carries over.

**Partition limits (max nodes, max walltime, exclusive vs shared) are NOT in the docs.**
Discover them on the system rather than guessing:

```bash
scontrol show partition dev-gpu-b200      # MaxNodes / MaxTime / OverSubscribe
scontrol show partition gpu-b200
sinfo -o "%20P %5a %12l %10D"             # drop %G/%N — node queries are often denied
sbatch --test-only --nodes=2 --partition=dev-gpu-b200 run_dist_sanity.sh
```

`sbatch --test-only` validates partition, node count and limits and reports when the job
*would* start, without submitting — use it before every new job shape.

### 3.2 Module system

`_site_env.sh` has `module load devel/cuda/12.9` (legacy naming). HoreKa 2 uses
EasyBuild/Lmod with `Name/Version-Toolchain`. Find it with `module spider CUDA`.

**It may be droppable entirely**: the `LD_LIBRARY_PATH` glob over
`$PYSITE/nvidia/*/lib` already provides every NVIDIA lib including NCCL, and ptxas
resolves from `site-packages/nvidia/cuda_nvcc/bin/ptxas`. The preflight banner reports
which ptxas is actually found — check it before assuming the module is needed.

### 3.3 Account

`--account=hk-project-pai00101` is hardcoded in all four. Confirmed still correct on
legacy as of 2026-08-16. Re-check on HoreKa 2:
`sacctmgr show associations user=$USER -n -P format=Account`.

### 3.4 Scheduling model

The HoreKa 2 **CPU** partition is shared-mode and requires explicit `--ntasks`,
`--cpus-per-task`, `--mem`. **It was not confirmed whether the GPU partitions are still
exclusive.** This matters: if `--gres=gpu:4` no longer implies a whole node, then
`--gpu-bind=none` and the `SLURM_LOCALID` device selection both break, because each rank
must see all of its node's GPUs. Check with `scontrol show partition gpu-b200` and look
at `OverSubscribe`.

### 3.5 Recommended: add a `horeka2` branch to `_site_env.sh`

Rather than editing four scripts, add `CGOLS_SITE=horeka2` to the `case` in
`_site_env.sh` and move the partition name into a variable the runners reference. That
is what the site switch was built for, and it lets both systems work during the overlap.

---

## 4. Jade is ARM — the environment must be rebuilt

**This is the largest single piece of work and gates everything else.** Jade's host CPU
is NVIDIA Grace (aarch64); the existing `astro` micromamba env is x86-64.

Needed:

1. A fresh aarch64 env with `jax[cuda12]` ≥ 0.10.1 and `orbax-checkpoint` 0.12.1.
   (Orbax **must** be ≥ 0.12.1 — older versions cannot restore under jax 0.10, and that
   failure would land at hour 45 of a 48 h leg.)
2. **Verify the Pallas/Triton path emits for Blackwell (sm_100).** cgols runs
   `backend=PALLAS` with `pallas_block_shape=(4,4,8)` and `pallas_use_triton=True`. A
   toolchain that cannot target sm_100 is a hard stop — there is a `NATIVE_JAX` fallback
   but it is far more memory-hungry and was never sized for this run.
3. Re-run ladder rungs 1–3 on the new env before trusting anything.

If the ARM env proves hard, **Ruby (`gpu-h200`, x86) remains a valid target** — the whole
plan was sized for it and the existing env works there. Jade is a scheduling and memory
win, not a requirement.

---

## 5. Verification ladder — run in order, cheap → expensive

Prerequisites, once:

```bash
cd <repo>/pytests/self_gravity/cgols
ws_allocate cgols 60 && ln -s "$(ws_find cgols)" data   # ≥2.5 TB, globally visible
micromamba activate astro && python -c "import jax, orbax.checkpoint as o; print(jax.__version__, o.__version__)"
```

Disk at 2048: one checkpoint = the full state = **344 GB**; `CGOLS_CHECKPOINT_KEEP=4`
plus one in flight ≈ 1.7 TB steady state; ~20 TB written over the run. Workspaces expire
after 60 days (3 extensions) and **have no backup**.

Use `sbatch --test-only …` first for anything new — it validates partition, limits and
node count without submitting.

**Rung 1 — intra-node NCCL P2P** (minutes, dev queue)
```bash
sbatch --partition=dev-gpu-b200 run_dist_sanity.sh
```
Expect `allgather = [0. 1. 2. 3.]` then `PASS`.

**Rung 2 — inter-node IB rendezvous** (minutes)
```bash
sbatch --test-only --nodes=2 --partition=dev-gpu-b200 run_dist_sanity.sh   # does it take 2 nodes?
sbatch            --nodes=2 --partition=dev-gpu-b200 run_dist_sanity.sh
```
Whether `dev-gpu-b200` allows 2 nodes is undocumented (§3.1) — the `--test-only` call
answers it in seconds. If it caps at 1 node, use a short `gpu-b200` job instead.
This rung is where a jax/NCCL version surprise surfaces, and it is the **first real test
of anything multi-node** — nothing in this project has ever run across nodes.

**Rung 3 — cgols smoke, small** (~30 min)
```bash
sbatch --nodes=1 --time=00:30:00 --partition=dev-gpu-b200 \
       --export=ALL,CGOLS_DIM=128,CGOLS_END_FRACTION=0.02 run_cgols_bench.sh
```

**Rung 4 — the multi-process I/O gate** (~2 h)
```bash
sbatch --nodes=2 --time=02:00:00 \
       --export=ALL,CGOLS_DIM=256,CGOLS_END_FRACTION=0.05,CGOLS_RUN_TAG=_rung4 \
       run_cgols_multinode.sh
```
Check, in order:
- frames in `cgols_snapshots_rung4/` written **exactly once**, with correct *global*
  content. The failure mode this guards against is a per-shard callback producing
  half-real / half-garbage planes.
- kill it, then resume with
  `--export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints_rung4,CGOLS_RUN_TAG=_rung4b`
  to prove the Orbax round-trip across processes.

The IC half of the original rung 4 is **already settled** — in-situ is bit-identical to
the file IC (§8). This rung is about I/O and collectives only.

**Rung 5 — "did the physics survive the port"** (~4 h)
```bash
sbatch --nodes=2 --time=04:00:00 \
       --export=ALL,CGOLS_DIM=1024,CGOLS_END_FRACTION=0.02,CGOLS_RUN_TAG=_rung5 \
       run_cgols_multinode.sh
```
Compare the `max|v|` / `min_rho` / `max_rho` trajectories in
`cgols_logs/cgols_diag_rung5.log` against the 1024 production diag log (local copies at
`/export/scratch/bschuber/horeka_1024/`). Different device counts reassociate reductions
— expect float32 agreement, **not** bit-identity.

**Rung 6 — calibration fit, three points at fixed global size**
```bash
for n in 2 4 8; do sbatch --nodes=$n --export=ALL,CGOLS_DIM=1024 run_cgols_bench.sh; done
```
Splits derive automatically: `(1,4,2,1)`, `(1,4,4,1)`, `(1,8,4,1)`.
Question to answer: **does the ~112 GB temp arena scale linearly with per-device cells,
or carry a fixed component?** If it has one, the 32-GPU option is the first to die.
Also read the strong-scaling efficiency across the node boundary — this matters more on
Jade than it did on Ruby (§6, interconnect).

**Rung 7 — go/no-go** (1 h)
```bash
sbatch run_cgols_bench.sh     # DIM=2048, 8 nodes, 20 fixed steps
```
Read per-device memory against the estimate. **The printer labels MiB as "MB"** — the
1024 run's `133,723.81 MB` was really 140.2 GB. Multiply s/step by ~135,000 for the wall
estimate, then size legs from the *late-run* decay, not leg 1.

---

## 6. Production

```bash
sbatch run_horeka_2048.sh                                          # leg 1
sbatch --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints,CGOLS_RUN_TAG=_leg2 \
       run_horeka_2048.sh                                          # leg 2, after TIMEOUT
```

Expect ~3 legs. Consider disabling `CGOLS_CHECKPOINT_KEEP` pruning for leg 1 — disk is
cheaper than a lost leg.

---

## 7. Facts you would otherwise have to re-derive

**Sizing.** Measured at 1024 on 4× H200: **140.2 GB/device**, **~3.13 s/step** (70,576
steps in 61.45 h; the earlier back-derived 2.46 s figure is wrong). Of that, ~17.35 GB
was *replicated global* data (potential 8.59 + `r` 8.76) — now sharded. The remaining
~491.5 GB aggregate scales with per-device cells.

| GPUs | nodes | split | per-device slice | est. GB/device |
|---|---|---|---|---|
| 32 | 8 × 4 | `(1, 8, 4, 1)` | 256×512×4096 | **~127–129** |
| 64 | 16 × 4 | `(1, 8, 8, 1)` | 256×256×4096 | ~64–65 |
| 128 | 32 × 4 | `(1, 16, 8, 1)` | 128×256×4096 | ~32–34 |

~133–140k steps at ~3.1–3.3 s/step ⟹ **~120–130 h ≈ 3 legs of 48 h, ~3,900–4,150 GPU-h.**
**32 GPUs is the minimum** — 16 would need ~250 GB/device, over even Jade's 186 GB.

**Why Jade over Ruby.** Ruby is 13 nodes total, so an 8-node 48 h slot is a multi-day
queue wait; Jade is 75 nodes. Jade also has 186 GB/GPU vs 141. **But Jade has 2× IB
NDR200 per node against Ruby's 4× NDR400 — roughly 4× less inter-node bandwidth — while
the GPUs are faster.** The comm/compute ratio is therefore materially worse than the
~3–4% estimated for Ruby. Rung 6's strong-scaling numbers are the check; if comm
dominates, prefer fewer, fatter shards (32 GPUs, not 64).

**Shard-count rule.** Per axis: the axis must divide evenly by its shard count **and** the
per-device slice must stay a multiple of `pallas_block_shape=(4,4,8)`. At 2048: `2048/gx`
divisible by 4 and `4096/gz` by 8 ⟹ every shard count must divide 512 ⟹ **must be a power
of two**. This rules out 48 GPUs and any 12/24/192-GPU configuration.
`_validate_shard_split()` enforces it and names the offending axis.

**Keep z unsharded.** The IC's vertical HSE pressure integral is a `cumsum` along z; a
z-split lowers it to a cross-device scan. Enforced.

**Traps that have already cost time:**

- `config.fixed_timestep=True` uses `dt = params.t_end / config.num_timesteps`. `dt_max`
  and the CFL number are **ignored**. A "1-step" A/B with `t_end=1.0, num_timesteps=1`
  takes one ~10 Myr step and diverges to NaN.
- Anything reading `config.grid_spacing` **before** `finalize_config` is suspect: the
  `SimulationConfig` default is 0.0025 and the real value is only derived once the state
  shape is known. `build_config()` now sets it and `_ic_axes` guards it — a wrong spacing
  fails *silently*, producing a plausible-looking state that then runs away.
- Never rank-guard an Orbax `save`/`restore`/`latest_step`, or any barrier — they are
  collectives and a guard on one rank deadlocks all the others. Rank-guard only
  astronomix's own bookkeeping.
- Never make `config` rank-dependent (e.g. `progress_bar=RANK0`) — it is a static
  argument, so that compiles a *different program* on rank 0.
- `srun`: use `--gpu-bind=none` and pass **no** `--ntasks` (inherit the allocation).
  `--gpus-per-task=1` cgroup-binds each task to a GPU that appears as ordinal 0, breaking
  NCCL P2P and deadlocking the topology exchange with "invalid device ordinal".
- Under Slurm, `$0` is the spool copy — resolve paths from `BASH_SOURCE`.

---

## 8. Already verified — do NOT redo

On the local 8×H200 box and on 4 simulated CPU devices, 2026-08-10:

- **cgols at DIM=128 on 1 GPU is bit-identical** before/after all library changes (same
  final state, same diag log).
- **`get_helper_data` is bit-identical** to the pre-change code across every geometry,
  dimensionality and wind combination (1D spherical/cylindrical FV, 1D/2D/3D Cartesian,
  stellar wind, cgols wind, `host_helper_data` on/off, padded and unpadded).
- **The in-situ IC is bit-identical to the stored `_d128.npy` state *and* potential** —
  so the analytic-coordinate and `dx = grid_spacing` substitutions are exact.
- `pytests/self_gravity/external_potential.py` and `hydrodynamics/shock_tube1D.py`
  reproduce exactly.
- 1-device vs 4-device (native backend): final state **4.2e-7**, frames 1e-8…6e-7,
  frame 0 bit-identical; checkpoint retention, sharded restore and the restart leg all
  exercised.

**What is NOT verified: the multi-node path has never run multi-node.** It is proven only
on simulated CPU devices. The sharded **Pallas** path is proven only intra-node (by the
completed 1024 run). That is what rungs 1, 2 and 4 exist for.

---

## 9. Open risks, ranked

1. **ARM env + Blackwell Pallas/Triton** (§4). Hard stop if the toolchain can't target
   sm_100. Fallback: Ruby.
2. **Memory extrapolation rests on a single measurement.** The 32-GPU estimate
   (~127–129 GB) sat ~12 GB under the *proven* 140.2 GB program on H200; on Jade's 186 GB
   there is much more headroom, which reduces this risk considerably. Rung 6 is still the
   gate.
3. **Inter-node bandwidth on Jade** (§7). Measure at rung 6 before committing.
4. **Orbax at `process_count > 1`.** `save`/`restore` barrier internally (confirmed from
   the 0.12.1 source), so the residual risk is our own pruning/`rmtree` logic racing
   them. Rung 4 is the proof.
5. **`jax.debug.callback` with sharded operands.** Explicit `P()` constraints are in
   place; rung 4 detects a failure as garbled half-planes.
6. **fp32 `cumsum` over a 4096-cell z-column** in `vertical_hse_pressure` — 2× the
   accumulation length of the validated run. Check the HSE residual in the IC
   diagnostics; if it drifts, do that reduction pairwise or in fp64.
7. **Queue reality / migration window.** Legacy HoreKa is decommissioned December 2026.
   Do not let a 48 h leg straddle the cutover — run the whole campaign on one system.

---

## 10. Reference

- Original plan: `~/.claude/plans/buzzing-floating-lobster.md`
- Multi-node gotchas: `.claude/skills/multi_node/MULTI_NODE.md` (updated with the
  sharded-global-leaf, rank-guard and debug-callback rules)
- Merge commit: `7cae987`; the multi-node commit itself is `afb24fb`
- HoreKa 2 docs: <https://docs.nhr.kit.edu/clusters/horeka-2/> and
  <https://docs.nhr.kit.edu/get-started/migration/>
