# Full-hand sim-vs-real validation — results summary (2026-07-10)

This documents the first full-hand (all 13 joints simultaneously) check of the
per-joint PD gains identified by `shadow_pd_id`, run against a real Baoding-style
no-ball hardware episode. Single-joint identification (see `DECISIONS.md`) only
ever excited one joint at a time with every other finger parked out of the way —
this was the first test with every finger moving together.

## Data used

- `data/raw/hw/real_noball_seed0001.npz` — real hardware log (13-joint policy
  order: `actions`, `joint_pos`, `joint_vel`, tactile, `with_ball=False`, `seed=1`).
- `data/raw/hw/sim_noball_seed0001.npz` — its paired sim recording (16-joint
  actuated order: `actions`, `joint_pos_cmd`, `joint_pos`, `source_episode=1`,
  `source_npz="policy_sim.npz"`).
- Confirmed these two are the SAME episode by bit-identical `actions` arrays
  (`diff == 0.0`): `sim_noball` was recorded once in sim via `record_policy.py`,
  then that exact command sequence was replayed open-loop on the real hand to
  produce `real_noball` — the same sim-command-then-hardware-replay pattern
  `replay_traj_hw.py` already uses for single joints, here for a whole episode.

## New tool

`roto/scripts/replay_traj_sim.py` — new script, replays a full-hand recorded
trajectory through Isaac Sim under `shadow_pd_id`'s identified gains, headless,
and compares simulated vs. real joint positions per joint. Takes `--sim_ref`
(source of the ground-truth command) and `--real_ref` (source of the
ground-truth real response).

## Config changes applied (all still in place)

1. **`roto/roto/assets/shadow_hand_lite.py`** — all 13 identified `{Kp, Kd}`
   values baked into the active `SHADOW_HAND_LITE_CFG` actuator cfg (was
   uniform placeholder Kp=1.0/Kd=0.1 per joint group). Required restructuring
   the dict into 13 literal joint-name keys, because Isaac Lab's actuator dict
   matching (`resolve_matching_names_values`) raises if two regex keys match
   the same joint — the old group regex (`rh_[MRF]FJ[1-4]`) would have
   double-matched every identified joint.
   - **This is a global change** — every task/robot script that spawns
     `SHADOW_HAND_LITE_CFG` (train/play/sweep for Bounce/Baoding/Peace on
     shadowlite) now uses these gains, not just this validation.
   - J1 mimic joints (FFJ1/MFJ1/RFJ1, never independently excited/identified)
     were first left at the old placeholder, then — per user decision — set
     equal to their own driver J2's identified Kp/Kd (same finger, same
     physical actuator/tendon driving both).
   - `effort_limit_sim` changed from a per-joint-group dict (~0.9–2.4 N·m) to a
     uniform `30.0` N·m, per user decision.
2. **`roto/roto/tasks/robots/shadowlite/shadowlite.py`**:
   - `couple_asymmetric_backward`: `True` → `False` (falls back to the
     `couple_gate_j1_on_measured=True` path, which stays `True` as before —
     just made explicit per user request).
   - `coupling_theta`: `0.785` rad (45°) → `0.8727` rad (50°), per user
     correction of the real hardware's coupling split point. Propagated to
     `roto/shadow_pd_id/config/joints.yaml` (3 entries), `roto/scripts/
     collect_traj_sim.py`, `roto/scripts/my_policy_node.py` (real-hardware ROS
     deployment script), `roto/scripts/plot_mimic_check.py`, and a stale
     docstring example in `roto/roto/tasks/roto_env.py`.
   - **Caveat:** these two coupling-mode changes only affect *future* RotoEnv
     rollouts (training, or a fresh `record_policy.py` recording). They have
     **no effect on the `sim_noball_seed0001.npz` reference file already on
     disk** — its `joint_pos_cmd` was generated under the old settings
     (asymmetric-backward on, 45°) and that's baked into the recorded numbers.
     Testing "what if" for the new coupling settings requires re-recording the
     sim reference against the same checkpoint/seed/episode, not done here.

## Test 1 — baseline (0.9–2.4 N·m effort limit, pre-existing round-trip bug)

Mean abs error vs. real hardware, rad:

| | identified gains | default (training-time) gains |
|---|---|---|
| **ALL 13 joints** | **0.1638** | 0.1879 |

~13% reduction. Biggest win: FFJ2 (0.36→0.20). Worse: MFJ2 (0.34 vs 0.33 default).
Max error ~1.55 rad at step 0 on several joints — an initial-pose mismatch (sim
starts exactly at the commanded pose; real hardware evidently started from a
different pose), not a steady-state tracking failure.

## Test 2 — effort limit raised to 30 N·m (same round-trip bug still present)

| | identified gains | default gains |
|---|---|---|
| **ALL 13 joints** | **0.1552** | 0.1879 |

Improved further (~17% reduction). The old 0.9 N·m cap on MFJ2 was saturating
its torque output given the identified Kp=16.5 — removing the cap fixed MFJ2
specifically (0.3355→0.2440, from *worse*-than-default to *better*-than-default).
Every other joint moved by <0.003 rad.

## Investigation: a major bug found in the replay script itself

While looking for non-gain factors behind the residual error, found that
`replay_traj_sim.py` was collapsing the recorded 16-DOF coupled-joint commands
(`joint_pos_cmd`, real per-joint J2 driver + J1 mimic values, computed by
`RotoEnv._handle_coupled_joints`'s stateful asymmetric-backlash coupling at
record time) down to a 13-dim mean-of-(J2,J1) "proxy", then **reconstructing a
different (J2,J1) pair from that mean using a simple linear-split formula**
never intended for this (copied from `sim_rollout.py`'s single-joint testbed).
This round-trip is lossy — measured:

```
rh_FFJ2: driver-only vs mean-of-pair differ by 0.34 rad on average (max 0.49)
rh_MFJ2: differ by 0.30 rad on average (max 0.52)
rh_RFJ2: differ by 0.09 rad on average (max 0.41)
```

— comparable to or larger than the entire reported tracking error on those
joints. Fixed by commanding the robot directly from the recorded 16-DOF
`joint_pos_cmd`, bypassing any coupling-formula reconstruction entirely.

## Test 3 — round-trip bug fixed + J1 gains = J2 gains

| joint | fixed round-trip | Test 2 (buggy round-trip) | default gains |
|---|---|---|---|
| rh_FFJ2 | **0.3605** | 0.1760 | 0.3600 |
| rh_MFJ2 | **0.3355** | 0.2440 | 0.3293 |
| rh_RFJ2 | 0.1532 | 0.1620 | 0.1350 |
| **ALL** | **0.1756** | 0.1552 | 0.1879 |

**Unexpectedly worse**, not better — FFJ2 and MFJ2 regressed back to roughly
default-gains-level error, and the overall mean went up (0.1552→0.1756).

### Why this is inconclusive, not a clean result

The retry bundled two changes at once:
1. Commanding from the true recorded J2+J1 pair instead of the lossy mean+resplit
   (a real correctness fix to the sim side).
2. Changing how the sim's own position is *read back* for reporting, from
   "J2 driver joint only" to "mean of J2+J1" — done to be internally consistent
   with how the target/reference values are read.

Change 2 means the comparison against `real_pos13` (from `real_noball_seed0001.npz`)
now uses a mean-based sim readout — but **it was never confirmed what convention
`real_noball_seed0001.npz`'s own `joint_pos` uses for its FFJ2/MFJ2/RFJ2 columns**
(driver-only encoder reading vs. some combined/averaged proxy — no script in this
repo writes that file, so it can't be checked from source). If real's convention
is driver-only, the new mean-based comparison is now mismatched in a way it
wasn't before, which would produce exactly this kind of regression even though
the underlying sim rollout is more physically correct.

**Open question, unresolved, left for later:** does `real_noball_seed0001.npz`'s
`joint_pos` represent the driver J2's own encoder reading or a combined/mean
proxy for the 3 coupled joints? Needed before trusting any FFJ2/MFJ2/RFJ2 number
in Tests 1–3 above. Per user decision, not chased further this session.

## Files

- Script: `roto/scripts/replay_traj_sim.py`
- Latest run output: `results/rollouts/full_hand/real_noball_seed0001_identified_gains_replay.{npz,png}`
  (overwritten on each retry — currently holds Test 3's results)
- Decisions log: `DECISIONS.md` (2026-07-10 entries)

## Known secondary factors (not yet investigated), for later

- Mimic joints' Kp/Kd are now copied from their driver, not independently
  identified — untested whether that's actually a good approximation.
- Single-joint PD-ID never saw whole-hand loading (self-contact, shared-actuator
  load from simultaneous finger motion) by construction.
- No armature (rotor inertia) or `velocity_limit_sim` modeled in
  `shadow_hand_lite.py` (both fields exist, both commented out).
- `ImplicitActuatorCfg` is a rigid PD — no tendon/cable compliance model for the
  real tendon-driven hand.
- Step-0 initial-pose mismatch (sim starts at the commanded pose; real hardware
  apparently started elsewhere) inflates max-error numbers but isn't a
  steady-state tracking issue.
