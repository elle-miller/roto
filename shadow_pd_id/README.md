# shadow_pd_id

**What this is:** identifies per-joint PD gains (`Kp`/`Kd`) plus friction terms
(Coulomb + viscous) for the Shadow Dexterous Hand Lite, so that a policy
trained in Isaac Sim/Isaac Lab (in the sibling `roto`/`multimodal_rl` packages
in this repo) behaves the same way on the real hand as it does in simulation.

**Why this approach:** the hand does not report true joint torque — only
uncalibrated strain-gauge differences — so directly calibrating a torque model
is not possible. Instead, this project matches *motion*: it finds the PD gains
and friction values that make a simulated joint move the way the real joint
actually moved when given the same commands ("sim-in-the-loop" black-box
optimization). A fast least-squares fit (done in an earlier, separate step)
gives the starting guess; this project refines that guess against the real
Isaac Sim physics engine.

See `DECISIONS.md` for the reasoning behind every non-obvious choice made along
the way — read that file, not just the code, to understand *why* something is
built the way it is.

## Environment

This project depends on the `roto`/`multimodal_rl` packages living in this
repo (`../roto`, `../multimodal_rl`). Use the `real2sim` conda env for anything
that touches Isaac Lab/Isaac Sim — it's the only env on this machine whose
editable `roto`/`multimodal_rl` installs point at this repo rather than a
sibling copy (see DECISIONS.md).

```bash
conda activate real2sim
```

Real-hardware data collection (`roto/scripts/collect_traj_hw.py`) needs
`rospy` and the `sr_hand` ROS stack, which live in a separate ROS/docker
environment on the robot machine — not in any conda env here. Sim-side work
(`roto/scripts/collect_traj_sim.py`, everything under `src/`) runs in
`real2sim`.

## Layout

```
shadow_pd_id/
  data/
    raw/
      commands/     # generated excitation trajectories (Step 0.5)
      ...           # paired real-hand logs (Step 0.5 output): commands + measured positions
    processed/       # filtered, resampled, train/held-out split (Step 1)
  config/
    joints.yaml       # joint order, limits, coupled-joint groups — single source of truth
    optim.yaml         # loss weights + candidate-search bounds/settings
  src/
    make_trajectories.py  # excitation trajectory generator (Step 0.5)
    load_data.py          # raw logs -> clean arrays (Step 1)
    sim_rollout.py         # SimRolloutEngine: gains -> simulated trajectory via Isaac Sim (Step 2)
    loss.py                # simulated vs. real trajectory -> scalar loss (Step 3)
    collect_rollouts.py    # samples candidate gains, simulates each, saves every result (Step 4a)
    optimize.py            # picks the best candidate from what's been collected (Step 4b)
    validate.py            # held-out evaluation + REPORT.md (Step 5)
    emit_config.py         # identified gains -> Isaac Lab config snippet (Step 6)
  results/
    rollouts/<joint>/  # every candidate_NNN.npz collect_rollouts.py has produced
    params/            # <joint>_gains.yaml selected by optimize.py
    plots/             # loss landscapes, rollout/validation overlays
    REPORT.md           # validate.py's accumulated per-joint report
    identified_gains_isaaclab_snippet.py  # emit_config.py's output
  DECISIONS.md          # dated log of every non-obvious choice — read this first
  README.md
```

## Status

**Steps 0–1 and 3 are fully done and verified. Steps 2/4/5/6 are written and
verified wherever that didn't require live Isaac Sim — but every
Isaac-Sim-dependent step still needs to actually be run by you**, for two
reasons: this machine has no ROS/hand connection (hardware collection), and
there's an unresolved, reproducible multi-minute Isaac Sim stall on this
machine that makes it impractical for me to run long simulation batches
interactively in this environment (see DECISIONS.md for the full
investigation — ruled out gains, rendering, network/telemetry, and GPU
throttling as causes; every run given enough patience does eventually
produce correct output, so this looks like a severe but bounded stall, not a
correctness bug).

Because of that stall, **this project does NOT use a live adaptive
optimizer** (the original plan's "propose gains -> simulate -> feedback"
loop). Instead: `collect_rollouts.py` simulates a batch of candidate gains
(Latin Hypercube sampled, see `config/optim.yaml`) one at a time, saving each
result to disk immediately — so it can be killed and re-run as many times as
a stall requires without losing completed work — and `optimize.py` picks the
best from whatever's been collected, entirely offline. Run
`collect_rollouts.py` with patience (it may stall; that's expected, just let
it keep going or re-run it), ideally in a terminal you can walk away from.

### Running the real-hand collection (do this on the robot/docker machine)

**Safety note**: non-excited fingers are now automatically parked out of the
way (curled toward the palm) during collection, instead of held straight at
zero — real-hardware testing found straight neighbors collided with the
excited finger (see DECISIONS.md). This is embedded in every command file as
`default_pose` and applied automatically by `collect_traj_hw.py` — no extra
flag needed, but if you're re-running collection against `.npz` files
generated before this fix, regenerate them first (`python
src/make_trajectories.py`) so they carry `default_pose`.

For every file in `data/raw/commands/` (and its `held_out/` subfolder),
run:

```bash
# from wherever roto/scripts/collect_traj_hw.py runs (inside the shadow docker)
python collect_traj_hw.py \
  --traj_file /path/to/shadow_pd_id/data/raw/commands/joint_00_rh_FFJ4_chirp.npz \
  --output_dir /path/to/shadow_pd_id/data/raw/hw
```

Repeat for every `.npz` under `data/raw/commands/` (52 files: 13 joints x
chirp/step/random/ramp) plus every file under `data/raw/commands/held_out/`
(13 more).

**IMPORTANT — use a DIFFERENT `--output_dir` for the two groups.** The
training `random` trajectory and the held-out `random` trajectory produce the
identical output filename (`joint_XX_name_random.npz`). Collecting both into
the same directory means the second one silently overwrites the first — this
already happened once (see DECISIONS.md) and cost a full re-collection of 13
training runs. Use separate output dirs, matching the input layout:

```bash
for f in /path/to/shadow_pd_id/data/raw/commands/*.npz; do
  python collect_traj_hw.py --traj_file "$f" --output_dir /path/to/shadow_pd_id/data/raw/hw
done
for f in /path/to/shadow_pd_id/data/raw/commands/held_out/*.npz; do
  python collect_traj_hw.py --traj_file "$f" --output_dir /path/to/shadow_pd_id/data/raw/hw/held_out
done
```

Once you have the resulting logs (same filenames, now containing
`actual_pos`/`actual_vel` alongside `cmd`), copy `data/raw/hw/` back to this
machine (or point `src/load_data.py` at it directly):

```bash
python src/load_data.py --log_dir data/raw/hw
```

`src/load_data.py` (Step 1) is written and verified against synthetic
fixture data — see DECISIONS.md — but has not yet run against real hand
logs, since none exist yet. Its low-pass filter cutoff (10 Hz) is a
placeholder to revisit once real sensor noise can be inspected.

## Pipeline (in order) — commands to run yourself

0-0.5. Done (trajectories generated + approved). Collect real-hand data per
   the "Running the real-hand collection" section above.

1. Load/clean/split the collected logs (no Isaac Sim needed):
   ```bash
   python src/load_data.py --log_dir data/raw/hw
   ```

2-4. Collect candidate gains for ALL 13 joints in one Isaac Sim boot, fitting
   each candidate against every available training excitation type
   (chirp/step/ramp/random combined, not just one — see DECISIONS.md for
   why), then select the best per joint:
   ```bash
   conda activate real2sim
   python src/collect_rollouts.py --headless --hw_dir data/raw/hw
   # ^ this can take a while and may stall (see Status above) -- if it hangs
   #   for a very long time, Ctrl-C and re-run the SAME command: already-
   #   collected candidates (and already-finished joints) are skipped, not
   #   redone. To do just one joint: add --joint_idx 0 (see config/joints.yaml
   #   policy_joint_order for the index -> name mapping).

   for j in rh_FFJ4 rh_MFJ4 rh_RFJ4 rh_THJ5 rh_FFJ3 rh_MFJ3 rh_RFJ3 rh_THJ4 \
            rh_FFJ2 rh_MFJ2 rh_RFJ2 rh_THJ2 rh_THJ1; do
     python src/optimize.py --rollout_dir results/rollouts/$j
   done
   ```

5. Validate each joint against its OWN held-out file (never used above):
   ```bash
   python src/validate.py --headless \
     --gains_file results/params/rh_FFJ4_gains.yaml \
     --held_out_file data/raw/hw/held_out/joint_00_rh_FFJ4_random.npz
   ```
   Repeat per joint (held-out files live under `data/raw/hw/held_out/`, kept
   in a separate directory from training logs specifically so the two can
   never collide on the shared `..._random.npz` filename — see DECISIONS.md);
   check `results/REPORT.md` after each one.

6. Once you're satisfied with the validated joints, emit the Isaac Lab
   config snippet:
   ```bash
   python src/emit_config.py
   ```
   Follow the printed instructions to drop the snippet into
   `roto/roto/assets/shadow_hand_lite.py` (stiffness/damping) and
   `roto/roto/tasks/roto_env.py` (friction, via a runtime call — see the
   generated file's comments).
