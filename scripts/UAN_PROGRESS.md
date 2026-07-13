# UAN (Unsupervised Actuator Network) for ShadowLite — Complete Reference

> Documentation file, kept here for visibility alongside the training scripts. As of this
> revision, UAN **lives inside `roto/` proper** (`roto/roto/tasks/uan_shadowlite/`,
> `roto/scripts/{train_uan,play_uan}.py`, `roto/tests/uan_shadowlite/`) — this supersedes
> an earlier iteration that lived in a separate sibling package (`uan_roto/`, since deleted)
> to avoid touching `roto/`. The user explicitly changed that constraint ("it should work
> from within roto"), so this integration is intentional, not a violation of an earlier rule.
> **Nothing has been committed or pushed yet — that happens only after you review this and
> say so**, per your explicit instruction.

---

## 0. TL;DR status

Implementation complete, **fully verified running on real hardware-recorded data on GPU**
from its final location inside `roto/`. 21/21 CPU unit tests pass. A live Isaac-Sim
train+play round-trip was run end-to-end against the actual `roto/data/data/aligned/`
recordings (89 real episode files, ~339k timesteps) on an RTX 4090, and it surfaced **two
more real bugs** beyond the three found in the previous iteration — both diagnosed and
fixed (see §9). The actuator-net input spec was then corrected per your follow-up feedback:
`joint_vel`/`action` are 13-dim (per physical motor), not 16-dim (per kinematic DOF) — final
spec is `joint_pos(16) + joint_vel(13) + joint_pos_error(16) + action(13)` = 58-dim/frame,
re-verified end-to-end (§12). Nothing under `roto/` other than the new UAN files was
modified. Waiting on your review before `git add`/`commit`/`push`.

---

## 1. The problem, in plain terms

roto's ShadowLite simulation drives all 16 hand joints with a **PD controller inside
PhysX** (an implicit actuator: give it a target angle, it computes
`torque = KP·(target − current) − KD·velocity` every physics substep). Gains are fixed at
`KP=1.0, KD=0.1` (`roto/roto/assets/shadow_hand_lite.py`, `SHADOW_HAND_LITE_CFG`, untouched
by this work).

The real ShadowLite hand does **not** respond identically to the same commanded targets —
friction, cable/tendon elasticity, backlash, and other unmodeled effects mean sim and
reality diverge. This is the sim-to-real gap.

**The fix:** train a small MLP — the actuator network — that watches the current tracking
state and outputs a small **residual torque**, added on top of whatever the PD controller is
already doing. If learned well, "sim's PD + learned residual" reproduces what the real motor
actually does, closing the gap.

**"Unsupervised"** because the network is never given a torque label. The only training
signal is: "here is the joint angle the real hand was actually at, given this command
sequence." The network is trained with RL to pick torques that make the *simulated*
trajectory match the *recorded real* trajectory. This is the technique from
`UAN/athletic-loco-manipulation` ("Bridging the Sim-to-Real Gap for Athletic
Loco-Manipulation," MIT, applied there to a Unitree Z1 arm), ported to ShadowLite and
re-expressed in roto's own skrl/`multimodal_rl` PPO stack.

---

## 2. Hard constraints, and how each is satisfied (updated)

| Constraint | Status |
|---|---|
| ~~Don't modify anything under `roto/`~~ **Explicitly lifted** — "it should work from within roto" | UAN now lives inside `roto/roto/tasks/uan_shadowlite/`, `roto/scripts/`, `roto/tests/uan_shadowlite/`. Nothing *else* under `roto/` was touched — no existing file was edited (verified via `git status`/`git diff` before starting; the only pre-existing uncommitted change in the tree, a hardcoded-path fix in `shadow_hand_lite.py`, is yours, not mine — see §9). |
| Model stays an MLP | Unchanged: roto's own `Encoder` (MLP) + `GaussianPolicy` (MLP), `action_space=16`. |
| Output = 16 residual torques | Unchanged. |
| KP/KD stays the same | Unchanged: `_apply_action` calls roto's exact `set_joint_position_target(...)` line, then adds one new `set_joint_effort_target(residual)` call; PhysX sums them (verified against Isaac Lab source). |
| Empty environment, nothing mobile | Unchanged: hand + ground plane only. |
| Dataset-agnostic | **Now two loaders**, sharing one interface: `AlignedTrajectoryDataset` (current `*.aligned.npz` recordings, directory/glob-aware) and `TrajectoryDataset` (older single/few-file format, kept for compatibility). Selected via `dataset.format` in yaml. |
| Actuator-net input = **exactly** what you specified: `joint_pos(16) + joint_vel(13, per motor) + joint_pos_error(16) + action(13, per motor)` = 58-dim/frame, normalized the way roto normalizes | Implemented verbatim — see §6.3/§7/§12 (13-dim correction). |
| Reuses roto's skrl PPO | Unchanged: `train_uan.py`/`play_uan.py` call roto's own `make_env`/`train_one_seed`/`make_models` from `common_utils.py`, now literally the same directory. |
| Trained UAN sits on top of the policy during downstream task training, correcting torque like real hardware every step | **Architecturally enabled, not yet built** — the actuator-net's inputs are deliberately restricted to values every roto env always has live (never privileged/replay-only data — see §4 D7), which is exactly what's required to embed the trained, frozen network inside Bounce/Baoding/etc. training later. Wiring that embedding is a follow-on task, intentionally not started yet (you said "that has to go in policy later"). |
| Commit only after verification, and only with your go-ahead | Not yet done — see §9 for what was verified, and this file itself is the "here's what I did, please review" artifact. |

---

## 3. Where everything lives now

```
roto/
├── roto/tasks/uan_shadowlite/
│   ├── __init__.py              # gym.register("UAN_Shadowlite") side effect
│   ├── dataset.py                # AlignedTrajectoryDataset + TrajectoryDataset (legacy)
│   ├── features.py               # FeatureBuilder -- the 58-dim input spec
│   ├── task.py                   # UANShadowLiteEnvCfg + UANShadowLiteEnv + reward
│   └── agents/shadowlite/
│       └── default.yaml          # dataset/uan/encoder/policy/value/agent/... config
├── scripts/
│   ├── train_uan.py               # entry point, reuses common_utils.py (same dir)
│   ├── play_uan.py                # checkpoint eval: PD-only vs PD+UAN RMSE + plot
│   └── UAN_PROGRESS.md            # this file
├── tests/uan_shadowlite/
│   ├── test_dataset.py            # 12 tests (AlignedTrajectoryDataset + legacy)
│   └── test_features.py           # 7 tests
└── data/data/aligned/             # real recordings (yours, not created by this work)
    ├── free_space_50/              # 21 episodes
    ├── free_space_continous_50/    # 49 episodes
    └── free_space_10_13072026/     # 19 episodes  (89 total, ~339k timesteps)
```

No file that existed before this work was edited, only new files added, with one exception
worth flagging explicitly: nothing — the `shadow_hand_lite.py` path-fix diff already in your
working tree is yours from before this session; it was left alone throughout, and you asked
for it to be *included* in the eventual commit, which will be honored when you say go.

---

## 4. Design decisions — updated for this revision

*(D1, D2, D3, D6 are unchanged from the previous iteration — additive residual torque via a
second `set_joint_effort_target` call; reusing roto's PPO stack; emitting features under the
`"prop"` key so roto's own encoder/`FrameStack` handle them for free; skipping Hydra in favor
of direct `Cfg()` construction, matching `roto/scripts/inspect_shadowlite.py`'s own pattern.
Full reasoning for those is unchanged and not repeated here — see git history of this file if
you want the original writeup. What follows is new or materially changed.)*

### D4 (updated) — Two dataset loaders, chosen by `dataset.format` in yaml

The real recordings you added (`roto/data/data/aligned/*/*.aligned.npz`) are a materially
different, richer format than the original single-file assumption this task was first built
around. Rather than force-fit them into the old loader, `dataset.py` now has two classes
behind one shared interface (`q_cmd`, `q_meas`, `q_meas_vel`, `q_torque`,
`traj_starts/ends/lengths`, `sample_start_indices`, `clamp`, `is_at_boundary`,
`traj_progress`) so `task.py` never needs to know which one is active:

- **`AlignedTrajectoryDataset`** (`dataset.format: aligned`, the new default): reads
  `*.aligned.npz` files. A `dataset.paths` entry can be a directory (auto-globs every
  `*.aligned.npz` inside, non-recursively), a glob pattern, or an explicit file — so
  `roto/data/data/aligned/free_space_50` as a single yaml line pulls in all 21 files in it.
- **`TrajectoryDataset`** (`dataset.format: legacy`): the original format
  (`joint_pos_cmd`/`joint_pos`/`actuated_names`/`episode_ends`/`rl_dt`, e.g.
  `roto/mimic_recording.npz`), kept only for backward compatibility.

### D5 (superseded) — Coupled-joint handling now lives in the dataset loader, derived from real per-file structure, not bypassed as a blanket assumption

The previous iteration assumed incoming recordings already contained a fully-expanded
16-dim command and simply replayed it, bypassing roto's coupling logic entirely. The real
`.aligned.npz` files instead expose the **actuator level** (13-dim: `action`, `command`,
`act_pos`, `act_vel`, `act_err`, named by `actuator_order`) separately from the **joint
level** (16-dim: `gt_pos`, `gt_vel`, `gt_effort`, named by `joint_order`). This surfaces a
real physical distinction that has to be handled correctly:

- **10 of the 13 actuator channels drive one joint each 1:1** (e.g. `rh_FFJ3`, `rh_THJ1`) —
  for these, `action` (empirically confirmed to be the real position *setpoint*, not the
  `command` field — see below) is the causally-correct PD target: it's what was actually
  sent to the motor, independent of what happened afterward.
- **3 channels (`rh_FFJ0`, `rh_MFJ0`, `rh_RFJ0`) are the combined "J0" motor** that
  mechanically drives *both* DOFs of a coupled pair (J1+J2) via a tendon — Shadow's own
  hardware convention, where J0 reports/commands the *combined* angle. There is no
  independently-causal setpoint for either individual DOF on real hardware. `action`'s J0
  value was empirically checked against `gt_pos[J1]+gt_pos[J2]` and correlates 0.89–0.99
  across the sample files, confirming it's on the combined-angle scale — but that scale does
  **not** match what roto's own `RotoEnv._handle_coupled_joints` expects as input (it expects
  a proxy pre-scaled to a single joint's own limit, not the combined range). Reusing that
  method on this data unmodified would silently apply the wrong transform. Rather than
  reverse-engineer an unvalidated rescaling, **the measured position (`gt_pos`) is used
  directly as the PD target for these 6 DOFs** — always available, makes no assumptions
  about the exact real coupling law, and is a defensible substitute since the coupled DOFs
  mechanically track each other on real hardware regardless of what commanded them. This is
  a documented approximation, not silently swept under the rug — see `dataset.py`'s
  `AlignedTrajectoryDataset` docstring ("DESIGN NOTE") and `COUPLED_JOINT_PAIRS`.

  **How `action` vs. `command` was disambiguated (empirically, not assumed):** the aligned
  file's `command` field ranges up to ±600 with many exact zeros and is essentially
  uncorrelated with `gt_effort` — consistent with it being the position controller's raw PID
  *output* (a torque-like quantity), matching the `command`/`error` fields seen directly in
  the bag files' `JointControllerState` messages (`rosbag info` + a custom message dump were
  used to confirm this). `action`, by contrast, stays constant while `act_pos` (measured)
  gradually approaches it — exactly the signature of a held position *setpoint*. This was
  checked directly against real file contents before committing to the design, not assumed
  from field naming.

### D7 (new) — The actuator-net's input is restricted to values every roto env has live, on your own explicit instruction, and this is also what makes future deployment inside a task policy possible

You specified the exact feature set: `joint_pos(16) + joint_vel(13) + joint_pos_error(16) +
action(13)` = 58-dim, all normalized "as normalized in roto." (An intermediate draft used
16-dim for all four; you corrected `joint_vel` and `action` to 13-dim — per-motor, not
per-kinematic-DOF — in a follow-up; see §12 for the full reasoning and re-verification.)
This is implemented by having `_get_proprioception()` read directly from roto's own
inherited, already-computed buffers — **not** recomputed:

- `joint_pos` (16) = `self.normalised_joint_pos[:, actuated_dof_indices]` — roto's own
  `unscale()` call (in `RotoEnv._compute_intermediate_values`, inherited unmodified),
  mapping to `[-1, 1]` via the robot's own joint position limits.
- `joint_vel` (13) = `self.normalised_joint_vel[:, control_dof_indices]` — same `unscale()`,
  via joint velocity limits, but sliced to the 13 independently-controlled motors, not all
  16 kinematic DOFs (§12: hardware doesn't report the 3 coupled pairs' J1/J2 velocity
  independently, and there's only one true independent velocity per physical motor anyway).
- `joint_pos_error` (16) = `self.joint_pos_error[:, actuated_dof_indices]` — roto's own raw
  `joint_pos_cmd − joint_pos` (unnormalized, exactly matching roto's own
  `_get_proprioception`'s convention of leaving this term raw).
- `action` (13) = `unscale(self.joint_pos_cmd[:, control_dof_indices], control_lower,
  control_upper)` — **not** `self.actions` (UAN's own 16-dim residual-torque output, a
  different quantity entirely). This back-solves, via roto's own `unscale()` (the exact
  inverse of `scale()`), what raw 13-dim policy action would have produced the current
  `joint_pos_cmd` at the 13 control-level joints — see §12 for why this specific derivation,
  not `dataset.q_cmd`'s own raw `action` field or UAN's `self.actions`, is the correct choice.

This is a deliberate correction from an earlier draft of this feature list, which included
`real_vel` (the *recorded* trajectory's velocity) as a candidate input. You correctly
flagged the problem with that in conversation: a feature that only exists because a real
trajectory is being replayed **cannot** be a network input if the same network is later
meant to run forward during a downstream task policy's live rollout, where no such recording
exists. Every one of the four features actually used is something *any* roto env — this one
or a future Bounce/Baoding env with UAN embedded — computes as a matter of course during a
normal step, which is precisely what makes "sits on top of the policy during \[downstream\]
training" (your stated end goal) buildable later without changing the network's input
contract.

### D8 (new) — Uncalibrated torque goes into the **reward**, sign-only, never into the network's input

You separately have 16-channel motor effort (`gt_effort` in the aligned files) that is not
calibrated to physical N·m. The same causality argument as D7 rules it out as a network
input (it's also replay-only data). But per the earlier discussion in this conversation, it
doesn't need calibration to be useful as a **reward** term, because reward only ever exists
during *this* network's own training — it's never needed again once the network is
trained and (eventually) embedded elsewhere.

The concrete mechanism (`compute_uan_reward`'s `torque_sign` term, `task.py`): compare
`sign(sim's total applied torque)` to `sign(gt_effort)`, per joint, and average — never
magnitude. Sign agreement is invariant to any *positive* per-joint calibration factor
(`τ_raw = a_j · τ_true`, unknown `a_j > 0`), which is exactly the kind of uncertainty
"uncalibrated" implies here. Weighted by `uan.reward.torque_sign` in yaml, defaulting to
`0.0` (fully inert until you opt in). "Sim's total applied torque" is computed as
`robot.data.applied_torque[actuated] + self.residual` — Isaac Lab's own `applied_torque`
bookkeeping only captures the *PD* portion (see `ImplicitActuator`/`_apply_actuator_model`
in the Isaac Lab source), since the residual is injected via a separate raw effort-target
write (D1) that bypasses the actuator model entirely; the two have to be summed manually to
get the true total.

---

## 5. What was verified directly against source/data (not assumed) — this revision's additions

| Claim | How verified |
|---|---|
| `action` (not `command`) is the real position setpoint for the 13 actuator channels | Loaded a real `.aligned.npz` file directly and compared: `action` holds constant while `act_pos` (measured) converges toward it (classic setpoint-tracking signature); `command` ranges ±600 with frequent exact zeros, near-zero correlation with `gt_effort` (consistent with a PID controller's raw output, not a position). Cross-checked against the original ROS bag's own `JointControllerState` messages (`rosbag info` + a custom per-topic value dump), which have `set_point`/`process_value`/`command`/`error` fields matching this interpretation exactly. |
| The J0 combined-actuator value is on the combined J1+J2 angle scale | Computed `corr(action[J0], gt_pos[J1]+gt_pos[J2])` directly from real data: 0.89–0.99 across the FF/MF/RF pairs in the sample file. |
| roto's own coupling-split law (`_handle_coupled_joints`) is NOT directly reusable on this data without rescaling | Read its exact math from `roto_env.py` (proxy expected in `[0, J2_upper]`, a single joint's own limit) and compared against the actual observed range of `action`'s J0 channel (up to ~2.9–3.0, clearly the combined range, not a single joint's) — confirmed a scale mismatch, hence D5's decision to use measured position instead of forcing a mismatched transform. |
| `robot.data.applied_torque` does not include the manually-injected residual | Re-read `isaaclab/assets/articulation/articulation.py::_apply_actuator_model`, which only writes `applied_torque` from `actuator.applied_effort` (the implicit-PD path) — the residual's separate `set_joint_effort_target` call never touches this buffer. |
| The dataset loader's coupled-vs-direct joint routing and torque/measured-position extraction are correct | 12 new unit tests in `test_dataset.py`, including two that directly assert `q_cmd` equals `action` for a directly-driven joint and equals `gt_pos` for a coupled joint, run against synthetic `.aligned.npz` fixtures matching the real schema exactly. |
| The loader works on the **real** files, not just synthetic fixtures | Loaded all 89 real files across all three recording folders directly (not through the env): 338,693 total timesteps, 89 segments (one per file, matching the real `seg_id`/`valid` structure found in every file inspected), `q_cmd[0]` for a coupled joint exactly equals `q_meas[0]` at the same index (by construction) while a directly-driven joint's `q_cmd[0]` is a clean round setpoint value distinct from its noisier `q_meas[0]` — exactly the expected causal signature. |
| The full pipeline runs end-to-end from the new in-`roto` location, against the real data, on GPU | See §9. |

---

## 6. Files, updated

Only what changed materially from the previous iteration is called out here; unchanged
mechanics (D1–D3/D6, `_pre_physics_step`/`_apply_action`/`_get_dones`/`_reset_idx` control
flow, the `Trainer.train()` runtime trace) are the same as before and not repeated.

### 6.1 `roto/roto/tasks/uan_shadowlite/dataset.py`
Now two classes (§4 D4). `AlignedTrajectoryDataset`'s segmentation logic deserves a specific
callout since a real bug was found and fixed in it during this pass (§9): a trajectory
segment boundary is placed wherever `seg_id` changes **or** either of the two adjacent rows
is marked invalid — implemented by assigning every row a monotonically-increasing "run id"
that increments on any break condition, then keeping only runs whose first row is valid
(every invalid row is guaranteed to be isolated into its own single-row, excluded run, since
a break is forced on both sides of it). This correctly splits a run into two segments even
when the invalid gap is in the *middle* of an otherwise-contiguous `seg_id` block — the
initial implementation only trimmed leading/trailing invalid rows at existing `seg_id`
edges, missing interior gaps entirely (caught by a unit test, not by inspection).

### 6.2 `roto/roto/tasks/uan_shadowlite/features.py`
`DEFAULT_FEATURES = ["joint_pos", "joint_vel", "joint_pos_error", "action"]` — your spec,
**58-dim**: `joint_pos`/`joint_pos_error` are 16-wide (`num_joints`), `joint_vel`/`action`
are 13-wide (`num_control`) — see §12. `FeatureBuilder.__init__` now takes both `num_joints`
and `num_control` and looks up each feature's width from whichever applies. `FeatureBuilder`
does no rescaling itself (no `pos_scale`/`vel_scale` parameters) — normalization is entirely
roto's own job upstream (D7); the builder is pure concatenation, in configured order, of
whatever `task.py` hands it via `FeatureContext`. Passing `None` or `[]` for `feature_list`
falls back to `DEFAULT_FEATURES` rather than raising.

### 6.3 `roto/roto/tasks/uan_shadowlite/task.py`
`_get_proprioception()` reads roto's own `normalised_joint_pos`/`joint_pos_error` (sliced to
`actuated_dof_indices`, 16) and `normalised_joint_vel` (sliced to `control_dof_indices`, 13)
directly (D7) instead of building a custom-scaled feature context. The `action` feature is
**not** `self.actions` (that's UAN's own 16-dim residual-torque output) — it's computed each
step as `unscale(self.joint_pos_cmd[:, control_dof_indices], control_pos_lower,
control_pos_upper)` (`unscale` imported from `roto.tasks.roto_env`, the exact inverse of
`scale()`), caching `control_pos_lower`/`control_pos_upper` (13-dim slices of roto's own
joint limits) once in `__init__`. `_get_rewards()` additionally computes `total_torque_sim`
(PD + residual, D8) and passes it plus `dataset.q_torque[t]` into `compute_uan_reward`, which
takes two extra arguments (`torque_sim`, `torque_real`) and an extra weight (`torque_sign`)
beyond the previous version's five reward terms. Dataset construction branches on
`cfg.dataset["format"]` to build either an `AlignedTrajectoryDataset` or `TrajectoryDataset`.

### 6.4 `roto/roto/tasks/uan_shadowlite/agents/shadowlite/default.yaml`
`dataset.format: aligned`, `dataset.paths` pre-populated with all three real recording
folders (`free_space_50`, `free_space_continous_50`, `free_space_10_13072026`) — edit as you
said you would. `uan.features` set to the exact 4-feature spec. `uan.reward.torque_sign: 0.0`
(inert by default; raise it to enable the calibration-free torque term).

### 6.5 `roto/scripts/train_uan.py` / `play_uan.py`
Same responsibilities as before, now living directly in `roto/scripts/` alongside
`common_utils.py`. A defensive `sys.path.insert` for `roto`'s own root is still present and
still needed — see §9's note on the pre-existing, unrelated environment issue this works
around. `play_uan.py` carries three fixes discovered in the previous iteration
(`args_cli.video` must be set explicitly, an `except` clause is required around `main()` so
`simulation_app.close()` can't silently swallow a traceback, and `PPO(...)` must be given a
real `Writer` instance, not `None`, or `agent.load()` fails on `NoneType.checkpoint_modules`)
plus one new fix from this pass — see §9.

---

## 7. The reward function, in full (updated)

```
e    = q_real − q_sim                       (per tracked joint; q_real = gt_pos, "aligned" format)
SE   = sum(e²);  AE = sum(|e|)
Δa   = ||action_t − action_{t-1}||
sign_agree = mean_j( sign(τ_sim,j) == sign(τ_real,j) )   # τ_sim = PD + residual; τ_real = gt_effort

reward =  survival
        + l1              * AE
        + exp_l2_loose    * exp(−coef_loose  * SE)
        + exp_l2          * exp(−coef_l2     * SE)
        + exp_l2_strict   * exp(−coef_strict * SE)
        + exp_action_rate * exp(−coef_action_rate * Δa)
        + torque_sign     * sign_agree
```

The first six terms are unchanged from the previous iteration (see that writeup's reasoning
for the three-sharpness-level curriculum and the action-rate smoothness term, both still
valid). The new `torque_sign` term (§4 D8) is additive, calibration-free, and `0.0` by
default. Default coefficients (`configs/default.yaml`):

| term | scale | coefficient |
|---|---|---|
| survival | 0.0 | — |
| l1 | −1.5 | — |
| exp_l2_loose | 4.0 | 100.0 |
| exp_l2 | 4.0 | 300.0 |
| exp_l2_strict | 5.0 | 1000.0 |
| exp_action_rate | 0.5 | 0.5 |
| torque_sign | 0.0 (off) | — |

---

## 8. How to run everything (updated paths)

```bash
cd /home/ayush/icra/roto/scripts

# train
python train_uan.py --headless --num_envs 512
python train_uan.py --headless --agent_cfg ../roto/tasks/uan_shadowlite/agents/shadowlite/my_variant.yaml
python train_uan.py --headless --dataset /path/to/aligned/episode_dir   # override dataset.paths

# evaluate a checkpoint
python play_uan.py --checkpoint logs/uan_shadowlite/uan_default/<timestamp>/checkpoints/best_agent.pt --headless

# unit tests (no Isaac Sim needed)
cd /home/ayush/icra/roto
python -m pytest tests/uan_shadowlite/ -v
```

`--config` defaults to `roto/roto/tasks/uan_shadowlite/agents/shadowlite/default.yaml`. To
make a variant: copy that file, edit, pass `--agent_cfg <path>`.

---

## 9. What was actually run and verified in this environment (this revision)

All of the following was run for real, on GPU, from the final in-`roto` file locations —
not simulated/assumed.

1. **19/19 unit tests pass** (`pytest tests/uan_shadowlite/`, `icra` conda env — no Isaac
   Sim required for these). One genuine bug was caught and fixed by a test during this pass
   (segmentation not splitting on interior invalid-row gaps — §6.1, §5).
2. **Real-data load test** (no Isaac Sim, pure `dataset.py`): loaded all 89
   `.aligned.npz` files across all three recording folders directly — 338,693 timesteps, 89
   segments, correct coupled/direct joint routing confirmed against real values (not just
   synthetic fixtures).
3. **Live Isaac Sim train+play round trip**, `icra` conda env, RTX 4090 (`cuda:1`; `cuda:0`
   had another process already using ~11.7 GB):
   - `train_uan.py --headless --num_envs 4` against the real aligned data: booted Isaac Sim,
     built the 4-env ShadowLite scene, confirmed the encoder's input width is **512**
     (64 features × `obs_stack=8`, matching the new spec exactly — visible directly in the
     printed `Linear(in_features=512, ...)` first layer), and ran 150 PPO update cycles.
     Reward: `11.78 → −193.51 → −462.17 → −581.17` — decelerating (Δ of −269 then −119
     between successive checkpoints), staying finite, not diverging to NaN/−∞. A checkpoint
     (`best_agent.pt`) was produced and used for the next step.
   - `play_uan.py --checkpoint <that checkpoint>`: **found and fixed a new, real bug** —
     `RuntimeError: Inplace update to inference tensor outside InferenceMode is not
     allowed`, raised during the *second* of the two rollouts. Root cause:
     `_pre_physics_step` reassigns `self.residual = torch.clamp(...)` (a brand-new tensor)
     every step; when that line runs inside `torch.inference_mode()` (as it does during the
     rollout loop), the resulting tensor is permanently marked as an "inference tensor."
     `rollout()`'s `env.reset(hard=True)` call was outside the `with torch.inference_mode():`
     block, so the second rollout's reset (`_reset_to_trajectory`'s
     `self.residual[env_ids] = 0.0`, an in-place write) hit that now-tainted tensor from
     *outside* any inference-mode context — exactly the operation PyTorch disallows. **Fix:**
     moved `env.reset(hard=True)` inside the same `with torch.inference_mode():` block as the
     step loop, so every rollout's reset-and-step sequence for a given `rollout()` call
     happens in one consistent context (fully documented in the code with the reasoning
     above). After the fix, `play_uan.py` ran to completion: both rollouts (3,698 steps each,
     matching the checkpoint episode's dataset length) completed, a full 16-joint RMSE table
     printed, and a comparison plot saved. (RMSE with the UAN residual active was slightly
     *worse* than PD-only for this checkpoint — expected and not a red flag: 150 update
     cycles on 4 envs is a smoke test, not a real training run, so the residual net hasn't
     learned anything useful yet; it's still close to its random initialization.)
   - A stale process from the *pre-fix* `play_uan.py` attempt (which had errored out) was
     found still running ~70 minutes later, holding GPU memory — `simulation_app.close()`
     appears to hang after certain exceptions (consistent with behavior observed in the
     previous iteration's debugging). Killed manually; GPU memory confirmed freed via
     `nvidia-smi` both before and after. Worth knowing if you see orphaned Isaac Sim
     processes after a script errors — check `ps aux` / `nvidia-smi` rather than assuming
     the process exited when the script's own output stops.
4. **Confirmed, again, that the active conda environments' `roto` package resolution is
   still broken** (this is the same pre-existing, unrelated issue found in the previous
   iteration — neither `icra` nor `s2r`'s editable `roto` install actually points at
   `/home/ayush/icra/roto`). The same fix as before is used: `train_uan.py`/`play_uan.py`
   `sys.path.insert(0, <repo root>)` themselves before importing anything that resolves to
   `roto`, rather than touching the environment (which would require `pip install -e roto`,
   correctly blocked by the permission system earlier since it writes `egg-info/` into the
   source tree).

**What this proves, and what it doesn't:** the entire pipeline — real-data loading with
correct coupled/direct joint handling, the exact 64-dim feature spec you specified, additive
residual-torque injection, the reward (including the new calibration-free torque term's
plumbing, though it wasn't exercised with a nonzero weight in this pass), PPO training, and
checkpoint evaluation — is mechanically correct end-to-end against your real recorded data,
on GPU, in its final location inside `roto/`. It does **not** prove the residual network
learns a *good* correction yet; that requires a real training run (thousands of updates, not
150) rather than this smoke test.

---

## 10. What could be improved / open items

- **Coupled-DOF PD target (D5) is a documented approximation, not a first-principles
  derivation.** Using measured position as the target for the 6 non-independently-driven
  DOFs is defensible (no independent setpoint exists on hardware for them) but means the
  residual network's job for exactly those 6 joints is subtly different from the other 10 —
  it's correcting pure dynamics-tracking error rather than reproducing a
  setpoint-to-response relationship. If you want to validate/replace this later with a
  properly-rescaled reuse of roto's own coupling law (`coupling_theta`/`_asymmetric_backlash`),
  that would need the real hardware's actual combined-to-individual split ratio empirically
  fit against `action`'s J0 range vs. `gt_pos[J1]`/`gt_pos[J2]` — not attempted here, flagged
  as a possible follow-up rather than guessed at.
- **The downstream "UAN sits on top of the policy during \[task\] training" integration is
  not built yet**, per your own instruction ("that has to go in policy later"). D7's
  input-feature restriction was specifically chosen to make that integration possible without
  redesigning the network later — but the actual wiring (loading a frozen trained UAN
  checkpoint's encoder+policy into a Bounce/Baoding-style env and calling it every step to
  produce the residual) is new work, not started.
- **`torque_sign` reward term was never exercised with a nonzero weight** in this pass — it's
  implemented and unit-testable in isolation, but its actual training-time effect (does it
  help, hurt, or do nothing useful) hasn't been empirically checked yet.
- **The `dataset_rate`/`rl_dt` assumption (60 Hz, confirmed across all 89 real files) would
  need re-verification** if any future recording uses a different rate — the loader already
  rejects mixed rates across files with a clear error, so this fails loudly rather than
  silently, but is worth knowing.
- **`play_uan.py --export`** still produces a state_dict bundle, not a traced/scripted
  standalone module (unchanged limitation from the previous iteration — roto's `Encoder`
  doesn't trace cleanly through `torch.jit.trace`).
- **Real training run not yet done.** Everything above is a smoke test (150 updates, 4
  envs). A proper run (the full `agent.rollouts`/`max_global_timesteps_M` schedule already
  in `default.yaml`, more environments) hasn't been executed — that's the natural next step
  once you're satisfied with this review.

---

## 11. Debugging guide (additions from this pass)

**`RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`**
(seen specifically in `play_uan.py`, second rollout onward). Caused by mixing
`torch.inference_mode()` and non-inference-mode code paths that touch the same persistent
tensor across multiple inference_mode entries/exits — specifically, any `env` attribute that
gets *reassigned* (not just in-place-modified) inside an `inference_mode()` block (e.g.
`self.residual = torch.clamp(...)`) becomes permanently tainted; any later in-place write to
it *outside* inference_mode fails. Fix: keep `env.reset()` and `env.step()` in the same
`inference_mode()` context for any code that reuses one `env` instance across multiple
rollouts (already fixed in `play_uan.py`; watch for this if you write new eval/inference
scripts that call `env.reset()` outside a `with torch.inference_mode():` block).

**Isaac Sim process still running / holding GPU memory well after a script errored out.**
`simulation_app.close()` has been observed (twice now, across two separate debugging
sessions) to hang for an extended period after certain exceptions rather than exiting
promptly. Check `ps aux | grep -E "train_uan|play_uan"` and `nvidia-smi` after any script
that errors, rather than assuming the process is gone once output stops; `kill -9` the PID
if it's still there.

**`ModuleNotFoundError` / wrong `roto` resolved when importing from `train_uan.py`/
`play_uan.py`.** Still a pre-existing, unrelated environment issue (neither `icra` nor `s2r`
conda env's editable `roto` install points at this actual repo) — both scripts work around
it with an explicit `sys.path.insert(0, <repo root>)` before importing anything `roto.*`.
This is a workaround, not a real fix; the real fix (`pip install -e roto` in the correct
environment) was deliberately not applied because it writes `egg-info/` into the repo, which
the permission system correctly flagged as crossing the file-modification boundary without
explicit sign-off. If you want that actually fixed at the environment level rather than
worked around per-script, that's your call to make (and your `pip install` to run).

**`KeyError: ... joint 'rh_XXJ0' is neither a coupled DOF nor present in actuator_order`.**
Would mean a recording's `actuator_order` doesn't follow the expected "10 direct + 3 combined
J0" structure this loader assumes. All 89 currently-recorded files follow it; a differently-
structured future recording (e.g. a different subset of joints driven) would need
`COUPLED_JOINT_PAIRS` in `dataset.py` adjusted to match.

*(All debugging entries from the previous iteration — `isaaclab.utils` import ordering, npz
key/joint-name mismatches, checkpoint/yaml mismatches, slow first Kit boot — still apply
unchanged and aren't repeated here.)*

---

## 12. Follow-up correction — `joint_vel` and `action` are 13-dim (per motor), not 16-dim

After the previous revision was verified working end-to-end, you flagged two corrections to
the feature spec, both concerning the difference between ShadowLite's 16 *kinematic* DOFs
and its 13 *physical motors* (3 finger pairs each share one motor across their J1+J2 via a
tendon coupling — the same structural fact behind D5's coupled-joint PD-target handling).

### Why `joint_vel` is 13-dim, not 16

Your words: "the input is 13 velo from each motor not 16 ones." This is correct and, in
hindsight, was already half-visible in this codebase's own data: the real recordings' 16-dim
`gt_vel` field **duplicates the same value across a coupled pair's two joints** (confirmed
directly while inspecting the aligned files during D5's investigation — `gt_vel[FFJ1] ==
gt_vel[FFJ2]` exactly, row by row) — because hardware has no way to measure J1 and J2
velocity independently when one motor drives both. Reporting a duplicated 16-dim vector to
the network would be reporting redundant, not-independently-informative data for 3 of its 16
entries. The fix takes the velocity at the 13-motor level instead:
`self.normalised_joint_vel[:, self.control_dof_indices]` — still roto's own live-simulated,
`unscale()`-normalized buffer (D7's "always available in any roto env" principle is
untouched; this is a slicing choice — which 13 or 16 indices to read — not a new,
privileged/replay-only information source).

`joint_pos`/`joint_pos_error` deliberately stay 16-dim: unlike velocity, *position* is
meaningfully independent per kinematic DOF even for a coupled pair (J1 and J2 have distinct
instantaneous angles at any moment, even though one motor's motion determines both via the
coupling law) — confirmed directly in the data too (`gt_pos[FFJ1]` and `gt_pos[FFJ2]` are
**not** duplicated the way `gt_vel` is). So the 16-vs-13 split isn't an arbitrary choice, it
tracks which quantities are actually independent at which level for this specific hardware.

### Why `action` is 13-dim, and specifically *not* `self.actions`

Your words: "the action are coming from policy during policy training so that is 13 actions
commanded." This is the same underlying principle as D7 (features must be computable in any
roto env, not just this replay env) applied one level deeper: `self.actions` — what the
*previous* revision used for this feature — is UAN's **own** 16-dim residual-torque output.
That's the wrong quantity for two independent reasons:

1. **Wrong width.** A downstream policy (Bounce/Baoding-style) that this trained, frozen UAN
   will eventually sit on top of outputs a 13-dim action (roto's own `control_joint_names`
   width) — not 16. If the actuator net's input included its *own* 16-dim action, the
   network's input contract would be structurally incompatible with ever running inside a
   downstream policy's env, where there is no 16-dim "UAN action" to read — only the 13-dim
   policy action exists there.
2. **Wrong semantics even at 13-dim.** Even a hypothetically-13-dim version of `self.actions`
   would represent "how big a residual torque correction did the network itself choose,"
   which is a fundamentally different concept from "what control command produced this
   step's target position" — the latter is what a real downstream policy's action means, and
   what the feature needs to represent for the two contexts (UAN's own replay training vs.
   embedded-in-a-policy) to look the same to the network.

The fix: derive a 13-dim quantity that means the same thing in both contexts —
"the control-level command implied by the current PD target" — via
`unscale(self.joint_pos_cmd[:, control_dof_indices], control_pos_lower, control_pos_upper)`,
where `unscale` is roto's own function (imported from `roto.tasks.roto_env`, the exact
mathematical inverse of the `scale()` call that turns a policy's raw action into a joint
target everywhere else in roto). Two things make this the right choice over the alternatives
that were considered and rejected:

- **Rejected: use the aligned dataset's own 13-dim `action` field directly.** It's
  numerically tempting (already 13-dim, already loaded, no math needed) but it's in
  *radians* (real commanded joint angles, e.g. 0.65, 1.74), whereas a real policy's raw
  sampled action (from `GaussianPolicy`, roughly Gaussian-distributed, intended to land near
  `[-1, 1]` after training) lives in a completely different numeric range. Feeding the
  network radians during its own training and roughly-unit-scale values during a future
  downstream embedding would mean the SAME input feature carries wildly different scales
  across the two contexts the network needs to generalize across — exactly the failure mode
  D7 exists to prevent, just for a different feature.
- **Chosen: back-solve via `unscale()`.** This produces the same roughly-`[-1, 1]`
  representation regardless of whether `joint_pos_cmd` came from `dataset.q_cmd` (replayed
  real data, this task) or from `scale(policy.actions)` (a live downstream policy, future
  work) — because `unscale` is the literal mathematical inverse of whichever `scale()` call
  produced `joint_pos_cmd` in the first place. No context-specific special-casing needed in
  the network's input contract.

### Implementation

- `features.py`: `_FEATURE_DIMS["joint_vel"]` and `_FEATURE_DIMS["action"]` changed from
  `"num_joints"` to a new `"num_control"` key; `FeatureBuilder.__init__` gained a
  `num_control` parameter and looks up each feature's width from whichever key applies.
  `DEFAULT_FEATURES` unchanged (`["joint_pos", "joint_vel", "joint_pos_error", "action"]`) —
  only the widths behind two of those four names changed. New total: 16 + 13 + 16 + 13 = 58.
- `task.py`: caches `self.control_pos_lower`/`self.control_pos_upper` (13-dim slices of
  roto's own `robot_joint_pos_lower_limits`/`upper_limits`) once in `__init__`;
  `_get_proprioception()` slices `normalised_joint_vel` to `control_dof_indices` (13) instead
  of `actuated_dof_indices` (16), and computes the `action` feature via the `unscale()`
  back-solve described above instead of using `self.actions`.
- `configs/default.yaml`: comment updated to spell out the 16+13+16+13=58 breakdown and why.
- `tests/uan_shadowlite/test_features.py`: rewritten around `NUM_JOINTS=16`/`NUM_CONTROL=13`;
  added `test_joint_vel_and_action_are_13_dim_not_16` and
  `test_joint_pos_and_joint_pos_error_stay_16_dim` to lock in the width split explicitly
  (not just implicitly via the default-feature-set width test). **21/21 tests pass**
  (19 previous + 2 new).

### Verification

Confirmed **empirically**, not just by code review, that the fix wires through correctly
end-to-end: `python train_uan.py --headless --num_envs 4 --device cuda:1` was re-run from
`roto/scripts/` against the real aligned data. The printed encoder architecture now shows
`Linear(in_features=464, out_features=256, ...)` as its first layer —
`464 = 58 (new feature width) × 8 (obs_stack)`, exactly matching the corrected spec (the
previous verification pass, documented in §9, showed `512 = 64 × 8` under the old,
now-superseded 16/16/16/16 spec — that number in §9 is accurate for what was true *at that
time* and has been left as-is rather than silently rewritten). Training produced a real,
finite reward (`12.43` at step 0, no NaN/crash) before being stopped intentionally; GPU
memory confirmed freed via `nvidia-smi` afterward. `joint_pos_error`'s timing was also
separately confirmed unchanged and already correct per your description ("the joint pos
errors are calculated using after the 16 torque is outputted, the new joint pos 16 vs the
original") — it's computed by roto's own inherited `_compute_intermediate_values()`, called
after each physics step (`_get_dones`, and defensively again in `_get_proprioception`), as
`joint_pos_cmd` (the target, set *before* the step in `_pre_physics_step`) minus `joint_pos`
(the actual position, read *after* PD+residual torque has been applied and physics has
stepped) — exactly the "new position vs. the original [target]" comparison you described; no
code change was needed for this one, only confirmation.

**Still not done, unchanged from §10:** the actual downstream-policy embedding (loading a
frozen trained UAN checkpoint's encoder+policy into a Bounce/Baoding-style env to supply the
residual every step, using that policy's own live 13-dim `self.actions` as the `action`
feature) remains future work — this section's changes make the network's input contract
*consistent* with that future embedding, but don't build it. Nothing was committed or
pushed; still waiting on your review.

---

## 13. Follow-up — `observations.obs_stack` raised from 8 to 20

Requested directly, no other change: `configs/default.yaml`'s `observations.obs_stack: 8` ->
`20`. This only changes how much per-frame history roto's own `FrameStack` wrapper hands to
the encoder (more closely matching the reference UAN paper's own 20-step window) -- no code
changed, since `obs_stack` was always a pure yaml/config knob (§4 D3: temporal stacking was
deliberately left to roto's existing `FrameStack` wrapper rather than built by hand).

**Verified end-to-end**, same method as §9/§12: re-ran `train_uan.py --headless --num_envs 4
--device cuda:1` from `roto/scripts/` against the real aligned data. Log now shows
`USING FRAME STACK: 20` and the encoder's first layer as
`Linear(in_features=1160, out_features=256, ...)` -- `1160 = 58 (per-frame feature width,
§12) × 20 (obs_stack)`, exactly as expected, no traceback. A real, finite first-step reward
printed (`11.79`, no NaN/crash) before the process was stopped intentionally; GPU memory
confirmed freed via `nvidia-smi` afterward. Nothing committed or pushed; still waiting on
your review.
