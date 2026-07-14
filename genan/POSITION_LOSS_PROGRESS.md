# GenAN Position loss — implementation progress

Status: **fully implemented and verified end-to-end, including live Isaac
Sim**. `compute_dynamics.py` produced real `M_inv`/`C`/`G`/`tau_target` data
against the actual robot model; feeding that real data through
`predict_next_position` reproduces the real recorded trajectory to within
~0.0001 rad; `train_genan.py --position_loss_weight >0` runs end-to-end
against real caches and produces a checkpoint with the exact same format as
before (so `play_genan.py` needs zero changes). Full plan:
`/home/ayush/.claude/plans/i-want-to-implement-zesty-allen.md`.

**Outstanding**: only the FULL 338,693-row dataset has not yet been run
through `compute_dynamics.py` (only an 8-row `--limit` smoke test has) — see
"Isaac Sim boot/shutdown behavior on this machine" below before doing that.

## What this adds

A Position loss for GenAN, on top of the existing Torque loss, without RL,
without a rollout, and without ever backpropagating through Isaac/PhysX:

1. **`compute_dynamics.py`** queries Isaac Lab's PhysX tensor API
   (`Articulation.root_physx_view.get_generalized_mass_matrices()` /
   `get_coriolis_and_centrifugal_compensation_forces()` /
   `get_gravity_compensation_forces()`) **once per recorded data point,
   offline** — a plain numeric query at a directly-written joint state, not a
   live simulation step. This is functionally RNEA
   (`tau = M(q)*qddot + C(q,qdot) + G(q)`).
2. Training (`train_genan.py`/`train_genan_single.py`) treats that query's
   output (`M_inv`, `C`, `G`, `tau_target`) as **constants** loaded from disk,
   and runs a closed-form, fully-differentiable one-step semi-implicit-Euler
   dynamics prediction using GenAN's own predicted torque, then takes a plain
   MSE against the real recorded next position. Gradients flow via ordinary
   autograd because the only tensor with a live graph is GenAN's own torque
   output.

This directly supersedes `DESIGN.md`'s Decision 1, whose stated reason for
rejecting a Position loss ("no exposed M(q)") turned out to be outdated —
Isaac Lab's PhysX tensor API exposes exactly that, confirmed used identically
elsewhere in Isaac Lab itself (`joint_impedance.py`, `factory_env.py`).

## Files created

| File | Isaac? | Purpose |
|---|---|---|
| `roto/genan/preprocess.py` | No | Loads `AlignedTrajectoryDataset`, `scipy.signal.savgol_filter`-smooths position per segment, gets ANALYTIC velocity/acceleration (`deriv=1`/`deriv=2`, not finite-difference), builds delta-histories via the existing `history.py`. Saves `cache/smoothed.npz`. |
| `roto/scripts/compute_dynamics.py` | **Yes** | Boots Isaac headless, builds a minimal cloned-env scene of `SHADOW_HAND_LITE_CFG` with gravity re-enabled (see caveat below), writes each recorded `(q, qdot)` directly into the sim, refreshes kinematics (`sim.physics_sim_view.update_articulations_kinematic()`), queries `M`/`C`/`G`, saves `cache/dynamics.npz` (`M_inv`, `C`, `G`, `tau_target`). |
| `roto/genan/dynamics_cache.py` | No | `DynamicsCache` — loads both `.npz` caches, exposes `.position_targets(dataset, t)` returning row-aligned `(tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid_mask)`. `valid_mask` excludes each trajectory segment's final row (next-row would spill into an unrelated segment/file). |
| `roto/genan/POSITION_LOSS_PROGRESS.md` | No | This file. |

## Files modified

| File | Change |
|---|---|
| `roto/genan/losses.py` | Added `predict_next_position(...)` (the semi-implicit-Euler step, shared) and `position_loss(...)` (MSE wrapper around it, full 16-joint). Existing `torque_loss` untouched. |
| `roto/genan/train_genan.py` | `train()` gained optional `position_loss_weight`/`dyn_cache` params — adds `position_loss_weight * position_loss(...)` to each ensemble member's loss when enabled (default `0.0` = fully inert, bit-for-bit identical to before). New CLI flags `--position_loss_weight`/`--preprocess_cache`/`--dynamics_cache`. Val loss also incorporates the position term (via ensemble-mean prediction) when enabled, so early-stopping/best-checkpoint selection reflects what's actually being optimized. |
| `roto/genan/train_genan_single.py` | Same idea, but **isolated**: the tested joint gets this script's own single-joint predicted torque; every OTHER joint gets its real/target torque (`tau_target`) substituted in, not any prediction ("rest kept still"). Still integrates the full 16×16 coupled dynamics step (ShadowLite's joints are physically coupled through the hand's rigid-body structure even though the tendon-pair coupling itself is software-only), but the loss narrows to just the tested joint's position column. |
| `roto/genan/agents/shadowlite/default.yaml` | Added `genan.position_loss_weight: 0.0`, `genan.preprocess_cache: null`, `genan.dynamics_cache: null`. |

## Key implementation decisions worth knowing about

- **Gradient-blocking bug found and routed around, not fixed in `model.py`**:
  `GenANEnsemble.forward()` silently breaks gradients — its final
  `label_scaler(..., inverse=True)` call defaults to `no_grad=True`
  (`RunningStandardScaler.forward`), so the de-standardized (physical-torque)
  output has no `grad_fn`. Fixed by calling
  `ensemble.label_scaler(pred_std, train=False, inverse=True, no_grad=False)`
  explicitly wherever a differentiable physical-torque value is needed.
  **Verified in isolation** (see Verification below) — this is a real,
  reproducible bug in the existing code, not a hypothetical.
- **Gravity caveat**: `SHADOW_HAND_LITE_CFG`'s spawn config sets
  `disable_gravity=True` on the robot's rigid bodies (used elsewhere for
  RL/eval numerical convenience). `compute_dynamics.py` builds its own patched
  copy of the config (`disable_gravity=False`) via chained `.replace()` calls
  — never mutating the shared module-level singleton in place. The script
  prints a sanity check (`max|G(q)|` over the first batch) specifically to
  catch a regression here.
- **Kinematic refresh after a direct state write**: `write_joint_state_to_sim`
  calls `root_physx_view.set_dof_positions/velocities(...)` directly, but
  PhysX's own tensor-API docs state link kinematics must be explicitly
  refreshed before any dependent query —
  `sim.physics_sim_view.update_articulations_kinematic()` is that refresh
  (confirmed by reading `isaaclab`'s `SimulationContext.forward()`, which
  makes the same call internally).
- **Coupling**: confirmed by direct inspection (grepped the binary USD for
  `mimic`/`gear`/`coupl` tokens — zero matches) that ShadowLite's active
  simulated model has 16 fully independent DOFs, no real tendon constraint —
  the "3 pairs share 1 motor" fact is software-only (`AlignedTrajectoryDataset`
  command-splitting). So a plain 16-DOF query is dynamically correct as-is.

## Verification done — all passed

- `preprocess.py`: ran end-to-end on the full real dataset (338,693 rows, 89
  segments) — no NaNs, sane output shapes.
- `losses.py`: unit-tested in isolation
  (`/tmp/.../scratchpad/test_grad.py`) — reproduced the `RunningStandardScaler`
  gradient bug with default args, confirmed the `no_grad=False` fix restores a
  live `grad_fn`, confirmed gradient flows all the way to ensemble member
  weights through `position_loss`.
- `train_genan.py` / `train_genan_single.py`: both regression-tested with the
  position loss **disabled** (default) against real data — reproduces prior
  behavior exactly (this is the safety net for the additive change).
- `compute_dynamics.py`: **ran successfully against live Isaac Sim**
  (`icra` env, `CUDA_VISIBLE_DEVICES=0`, `--num_envs 4 --limit 8`), producing
  real `M_inv`/`C`/`G`/`tau_target` for 8 recorded states. Sanity checks:
  - Gravity compensation is nonzero (`max|G(q)| ≈ 0.028`, confirming the
    `disable_gravity=False` patch works — this was the #1 correctness risk
    flagged in the plan).
  - `M(q)`'s condition number is ~123 across all 8 rows — well-conditioned,
    not near-singular (the very large raw `M_inv` magnitudes, ~4×10^5, are
    real physics: ShadowLite's finger links have genuinely tiny moments of
    inertia, not a numerical artifact).
  - `M_inv` is symmetric to within floating-point tolerance.
- **Integrator sanity check against real data** (`/tmp/.../scratchpad/test_integrator_real.py`):
  fed the REAL `tau_target`/`M_inv`/`C`/`G` through `predict_next_position`
  and compared to the REAL recorded next position — max error 0.000111 rad,
  mean error 0.000015 rad across all 8 rows. Confirms the whole pipeline
  (`preprocess.py`'s smoothing/derivatives → `compute_dynamics.py`'s dynamics
  query → `predict_next_position`'s semi-implicit-Euler step) is internally
  consistent and physically accurate on real hardware data.
- **`train_genan.py --position_loss_weight 0.1`** against the real (8-row)
  caches: ran 2 epochs without error, loss decreased both epochs, saved a
  checkpoint. Checkpoint keys (`ensemble_state_dict`, `input_dim`,
  `num_joints`, `ensemble_size`, `history_len`, `stride`, `joint_names`,
  `best_val_loss`) are byte-for-byte identical in structure to a normal
  Torque-loss-only checkpoint — `play_genan.py` needs zero changes to load it.

## Isaac Sim boot/shutdown behavior on this machine (for future runs)

Getting `compute_dynamics.py` to actually run surfaced a real environment
quirk worth documenting for next time: apparent "stalls" of 25-40+ minutes
across several attempts were **not boot hangs** — cross-checking file
timestamps after the fact showed the real computation had already completed
and `dynamics_test.npz` had already been written to disk (at 17:35) well
before the process was killed for appearing stuck (~17:30-ish check showed
"still running" only because I was watching stdout/process state, not the
output file). This matches `shadow_pd_id/DECISIONS.md`'s own documented
finding: Isaac Sim on this machine can hang for a long time **after**
finishing real work, during headless shutdown (`simulation_app.close()`), not
during boot. Their established workaround, which any future large
`compute_dynamics.py` run (e.g. the full 338,693-row dataset) should also use:

- Wrap the invocation in `timeout <N>` rather than waiting for a clean exit.
- Check for the **output file's existence/mtime**, not the process's exit
  code, to know whether the run actually succeeded.
- One additional real fix applied along the way: `compute_dynamics.py`
  initially only added `roto/genan` to `sys.path`, not `roto/` itself (the
  repo root), causing `ModuleNotFoundError: No module named 'roto.assets...'`
  — fixed by also inserting `_ROTO_ROOT`, matching `play_genan.py`'s
  established pattern.
- This machine is shared with other users/processes (observed: another
  user's long-running `optuna-dashboard` processes since April, and other
  GPU-memory-heavy jobs on GPU 1) — prefer `CUDA_VISIBLE_DEVICES=0` and check
  `nvidia-smi` before large runs.

## Next step: run the full dataset

Only the full 338,693-row dataset hasn't been run through
`compute_dynamics.py` yet (only the 8-row `--limit` smoke test above). Given
the shutdown-hang behavior above, recommend:

```
timeout 3600 python3 compute_dynamics.py \
  --in cache/smoothed.npz --out cache/dynamics.npz \
  --num_envs 256 --headless
# then check cache/dynamics.npz's mtime/existence, not the process exit code
```

Once that exists, retrain with `train_genan.py --position_loss_weight <w> \
--preprocess_cache cache/smoothed.npz --dynamics_cache cache/dynamics.npz`
on the real, full-size caches, and compare the resulting checkpoint's
sim-vs-real RMSE (via `play_genan.py`, unmodified) against a Torque-loss-only
baseline trained on the same data.
