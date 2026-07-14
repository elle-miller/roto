# GenAN for ShadowLite -- design decisions

This module ports the Generalized Actuator Network (GenAN) idea from
"Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator
Networks" (Schneider et al.) to the ShadowLite hand. It is deliberately a
**separate module** (`roto/genan/`) rather than a modification of
`roto/roto/tasks/uan_shadowlite/`: that task implements a different method
(an RL-trained residual torque, see its own module docstring and
`UAN_PROGRESS.md`) and is left untouched. GenAN reuses its dataset loader
(`AlignedTrajectoryDataset`) since the data format is identical, but trains
with a **supervised loss**, not PPO.

`roto/genan/{history,model,losses,train_genan,sweep_genan,config_utils}.py`
are pure PyTorch/NumPy (no `isaaclab` import), consistent with the existing
`dataset.py`/`features.py`/`reward.py` convention, so they are unit-testable
on CPU without booting Isaac Sim, and `sweep_genan.py` never needs to boot
Isaac Sim per-trial either (see Decision 4). `roto/scripts/play_genan.py` is
the one Isaac-dependent piece (evaluation only), reusing `uan_shadowlite`'s
own env.

Model construction reuses `multimodal_rl.models.mlp.MLP` (the same builder
`Encoder`/`GaussianPolicy` are built from) and
`multimodal_rl.models.running_standard_scaler.RunningStandardScaler` (the
same standardizer `Encoder`'s `state_preprocessor` and PPO's
`value_preprocessor` already use) rather than reimplementing either --
`roto/genan/model.py` is a thin composition of existing primitives, not a
new architecture stack.

## What ShadowLite has that PAMY2 doesn't (and vice versa)

The paper's setting: a 4-DoF, PAM-actuated, tendon-driven arm with no torque
sensor. GenAN learns the actuator dynamics; a known rigid-body simulator
(MJX) supplies arm dynamics, `M(q)`, and inverse dynamics for a
differentiable Position loss.

ShadowLite differs in three load-bearing ways:

1. **16 joints, 13 motors**, with 3 mechanically-coupled tendon pairs
   (FF/MF/RF J1+J2 sharing one motor). This coupling -- not muscle
   nonlinearity -- is ShadowLite's dominant hard-to-model actuation effect
   (backlash, sequencing, tendon slack; see `COUPLING_CODE_EXPLAINED.md`).
2. **No torque sensor with a known calibration.** `gt_effort` in the aligned
   recordings is *uncalibrated* motor effort (existing code already treats
   it as sign-only, never magnitude -- see `reward.py`'s docstring).
3. **No differentiable simulator.** The project runs on Isaac Lab / PhysX.
   There is no MJX/JAX-style differentiable step, no exposed `M(q)`, and
   `shadow_pd_id`'s own system-ID work explicitly resorted to gradient-free
   search for exactly this reason (see its `DECISIONS.md`).

These drive the two decisions below.

## Decision 1: Torque loss only -- no Position loss, no differentiable-dynamics stand-in

The paper trains two loss variants: a Torque loss (regress a torque label)
and a Position loss (differentiate through one simulator step, using
`M(q)`, to match the *resulting* position). We considered approximating the
Position loss with a hand-rolled, differentiable per-joint stand-in for
`M(q)` (e.g. a learnable diagonal effective inertia + semi-implicit Euler
step) purely so gradients would exist to backprop through. We rejected
that: it would be new physics fiction invented solely to manufacture a
gradient, duplicating a check the project already gets for free by running
the *actual* Isaac simulator and comparing sim-vs-real position directly --
exactly what `play_uan.py` already does for the RL residual, and what
`eval_genan.py` (see Decision 3) does here. A toy dynamics module would add
a second, unvalidated approximation on top of the one we're already making
by not having `M(q)`, for no accuracy benefit over just running Isaac.

So **this module trains with the Torque loss only**: standardized MSE
against `q_torque` (`gt_effort`). The known caveat (Decision-driving point
2 above) is that `gt_effort` is uncalibrated -- the network's raw
de-standardized output is torque in whatever unknown per-joint scale
`gt_effort` happens to be in, not verified true N*m.
`RunningStandardScaler`'s mean/std absorb that unknown scale during
training (the network only ever has to be internally self-consistent in
standardized space), but nothing here claims the de-standardized number is
calibrated. This mirrors how `uan_shadowlite/reward.py` already treats the
same field -- sign-agreement only, never magnitude. Trained-model accuracy
is judged empirically (Decision 3), not by trusting the torque scale.

## Decision 2: network inputs are delta-histories of `(q_meas, q_cmd)`, not `(pos_error, vel)`

The paper's GenAN is `tau_hat = f_theta(q_{t-H:t}, u_{t-H:t})` -- histories
of joint position and *control signal* (PAMY2's valve command), not the
`[pos_error, vel]` layout used by the repo's own reference implementation
(`UAN/athletic-loco-manipulation/rsl_rl/ppo/actuator_net.py`) or by
`uan_shadowlite/features.py`. We follow the paper: ShadowLite's PD position
target `q_cmd` (from `AlignedTrajectoryDataset.q_cmd`, already
coupling-split to 16 dims) is the natural analogue of PAMY2's control signal
`u`, and `q_meas` is the analogue of `q`. `history.py` builds

```
(x_t, x_{t-1} - x_t, ..., x_{t-H} - x_t)
```

per the paper's finding that delta-histories outperform raw sparse-strided
histories (their Appendix C) -- for both the `q` stream and the `u` stream,
standardized via `RunningStandardScaler` before concatenation. Default
`H=3, stride=1` (Table 1 of the paper).

`GenANEnsemble` holds `N=5` independently-seeded `GenAN` MLPs (2 hidden
layers, 512 units, tanh -- Table 1 verbatim, built via
`multimodal_rl.models.mlp.MLP`), each trained on a different bootstrap
permutation of the training trajectories. `sample_member`/`disagreement`
support the paper's per-step random-member rollout sampling and an
ensemble-disagreement signal, for potential downstream RL use (not wired up
here -- see "out of scope").

**Why not UAN's own 4-feature vocabulary (`joint_pos`/`joint_vel`/
`joint_pos_error`/`action`, `features.py`) instead?** Considered and
deliberately rejected, to keep GenAN's inputs matching the paper's own
`f_theta(q, u)` formulation rather than conflating two independent
differences (training method AND input representation) into one comparison
against UAN.

**Coupling in `q_meas`/`q_cmd` needs no GenAN-specific handling at all**,
because both streams come from the exact same `AlignedTrajectoryDataset`
instance UAN's own `task.py` builds (via `roto/genan/dataset_loader.py`'s
`sys.path` import of `dataset.py` directly -- not a reimplementation, the
identical file). `q_meas` is real per-DOF measured position (hardware
genuinely measures both J1 and J2 of a coupled pair independently, even
though one motor drives both). `q_cmd`'s 6 coupled-joint columns are already
split into physically-in-range per-joint targets by
`_build_cmd_from_action`/`_split_coupled_command` (`dataset.py`) before
`build_delta_history` ever sees them -- by the time GenAN's history is
built, the coupling problem has already been solved upstream, identically
for both methods.

## Decision 3: evaluation reuses `UANShadowLiteEnv` unmodified, `play_uan.py`-style -- with GenAN's torque made the SOLE torque, not additive

Rather than write a new Isaac task, `play_genan.py` (`roto/scripts/`,
alongside `play_uan.py`/`sweep.py`) builds the *existing*
`UANShadowLiteEnv`/`UANShadowLiteEnvCfg` with `uan.action_scale = 1.0` and a
large `uan.residual_clip`, so `_pre_physics_step`'s existing
`residual = clamp(actions * action_scale, -residual_clip, residual_clip)`
reduces to `residual == actions` -- no edits to `task.py` needed.

That alone is not enough, though: `_apply_action` *unconditionally* also
calls `set_joint_position_target`, which drives PhysX's own implicit-PD
(the identified Kp/Kd) and additively contributes its own torque regardless
of `actions`. Since GenAN was trained (Decision 1's Torque loss) to regress
the FULL recorded `gt_effort`, not a residual/correction on top of a
separate known controller, naively feeding its output through this path
would double-count torque: `tau_applied = tau_PD_auto + tau_GenAN_predicted`,
not `tau_GenAN_predicted` alone. (This was caught only after an initial
smoke-test rollout diverged wildly -- worth stating plainly since it looked
at first like "just an undertrained checkpoint," but was actually this
structural mismatch.)

`play_genan.py`'s `_zero_pd_stiffness`/`_restore_pd_stiffness` fix this --
but zeroing ONLY Kp, not Kd. An earlier version zeroed both, via the same
`robot.write_joint_stiffness_to_sim`/`write_joint_damping_to_sim` API
`shadow_pd_id/src/sim_rollout.py`'s `set_gains` already uses for its own PD
identification sweeps. That first attempt made the divergence *worse*
(mean RMSE ~22,000 rad, up from ~6,000): `shadow_pd_id`'s own system-ID
folds real, physically-always-present viscous friction into the identified
`Kd` (its `DECISIONS.md`: "on a position-commanded joint [PD damping and
viscous friction] both produce torque = -coefficient * velocity --
mathematically identical effects, so only their *sum* is identifiable from
motion data"). `Kd` is therefore not a pure control gain competing with
GenAN's output the way `Kp` is -- it's real passive dissipation. Zeroing it
removed ALL damping from the joints, so a 15-epoch network's inevitable
nonzero-mean torque bias integrated into unbounded runaway velocity (nothing
to dissipate it) until each joint hit its limit and the solver blew up --
exactly the pattern observed (small-range/mimic joints stayed bounded by
luck of scale; everything else diverged into the thousands).

Zeroing only `Kp` -- the active position-tracking term that actually
competes with GenAN's predicted torque -- while leaving `Kd` (real passive
friction) at its identified value keeps the sim numerically stable, and
matches the paper's own division of labor: MJX's rigid-body dynamics
already include passive joint damping as "known physics," independent of
whatever the learned actuator model contributes; GenAN only ever needs to
replace the unknown *active* part. `ImplicitActuator.reset()` is a
documented no-op ("no state to reset for implicit actuators"), so the
stiffness zeroing survives `env.reset()` without needing to reapply after
every reset; it's restored after the GenAN rollout so the function has no
side effects that outlive one call.

With Kp=0 (Kd untouched), PhysX's implicit-PD contributes zero *restoring*
torque but keeps its real damping, so `set_joint_effort_target(residual)`
-- GenAN's own prediction -- is the only *active* torque source, matching
what the network was actually trained to produce, while the sim retains the
passive dissipation it needs to stay stable under an imperfect prediction.

`play_genan.py` otherwise mirrors `play_uan.py` exactly: same
`AppLauncher`/`common_utils` boilerplate, same deterministic
same-start-state rollout, same two-column (`rmse_pd_only`/`rmse_pd_genan`)
RMSE table, one checkpoint per run.

## Decision 4: `sweep_genan.py` is Isaac-free and loads the dataset once, unlike `sweep.py`

`scripts/sweep.py` must boot Isaac Sim and build its env once, then reuse
that one env across every Optuna trial, because PPO training needs the
simulator. `train_genan.py`'s `train()` never touches Isaac at all, so
`sweep_genan.py` (`roto/genan/`, Isaac-free like `train_genan.py`) applies
the same "build once, reuse across trials" idea to the *dataset* instead:
load `AlignedTrajectoryDataset` once, pass it into every trial's `train()`
call. Otherwise it mirrors `sweep.py`'s structure directly: `TPESampler` +
`MedianPruner`, sqlite storage (`agents/shadowlite/default.yaml`'s
`sweeper.storage`, inspectable via `optuna-dashboard sqlite:///roto_genan.db`
the same way `sweep.py`'s own yaml documents for `roto_uan.db`), a
`--rerun-trial`/`--rerun-seeds` path, and a default path that runs the full
search then retrains the best trial on multiple seeds. `train_genan.py`'s
`train()` takes an optional `trial` (duck-typed: `.report()`/`.should_prune()`)
reported every epoch, so `optuna` itself stays a lazy import -- plain
single-run training never needs it installed.

What's actually searched: `lr`, `batch_size`, `history_len`, `stride`.
`ensemble_size` and the MLP architecture are Table-1-fixed values from the
paper, not swept.

All three scripts (`train_genan.py`, `sweep_genan.py`, `play_genan.py`) read
the same `roto/genan/agents/shadowlite/default.yaml`, mirroring
`uan_shadowlite/agents/shadowlite/default.yaml`'s layout (`dataset`/
`sweeper`/`experiment` sections) but dropping everything PPO/RL-specific
(`encoder`/`policy`/`value`/`agent`/`observations`/`trainer`) that doesn't
apply to a supervised actuator net -- see the yaml's own header comment.

## What this module does NOT do (future work)

- No Position loss / differentiable-simulator training path of any kind
  (Decision 1) -- not deferred as "needs a stand-in module later", just not
  pursued, because Isaac-rollout evaluation already answers the question a
  Position loss would otherwise be used to answer, without a second
  approximation layered on top.
- It does not wire GenAN's predicted torque into an RL *training* loop
  (only into the evaluation rollout in `play_genan.py`). Using GenAN's
  ensemble disagreement as an RL reward penalty, the way the paper does,
  would need a new task following `uan_shadowlite/task.py`'s pattern -- left
  for later.
- It does not recalibrate `gt_effort` (Decision 1).
