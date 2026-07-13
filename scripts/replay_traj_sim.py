#!/usr/bin/env python3
"""Replay a REAL full-hand hardware trajectory through Isaac Sim, under the
per-joint PD gains identified by shadow_pd_id/, and compare simulated vs real
joint positions.

WHY THIS FILE EXISTS: shadow_pd_id identified {Kp, Kd, Fc} one joint at a time
(single-joint excitation, all other fingers parked out of the way). This script
is the first full-hand check of those gains: it drives all 13 policy joints at
once through a real recorded motion (data/raw/hw/real_noball_seed0001.npz, a
real Baoding-style run with no ball) and asks whether the combined per-joint
gains reproduce the real hand's actual response when every finger moves
together -- something the single-joint identification never tested (no
inter-finger contact/coupling load on the actuators).

COMMAND SOURCE: real_noball_seed0001.npz's own `actions` field is the RAW
policy-network output (values run well outside any joint limit, e.g. ~[-5.5,
4.3]), not a usable joint-position command. The actual commanded target comes
from its paired file, data/raw/hw/sim_noball_seed0001.npz (`joint_pos_cmd`,
16-DOF actuated order) -- confirmed to be the SAME episode by their bit-identical
`actions` arrays (diff == 0.0): sim_noball was recorded once in sim via
record_policy.py (seed=1, source_episode=1), then that exact command sequence
was replayed open-loop on the real hand to produce real_noball (the same
sim-command-then-hardware-replay pattern replay_traj_hw.py uses for single
joints, here for a whole episode). So this script needs BOTH files: the sim
one for the ground-truth command, the real one for the ground-truth response.

Usage (from roto/scripts/):
    python replay_traj_sim.py \\
        --sim_ref  ../shadow_pd_id/data/raw/hw/sim_noball_seed0001.npz \\
        --real_ref ../shadow_pd_id/data/raw/hw/real_noball_seed0001.npz \\
        --headless
"""

import argparse
import os

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sim_ref", type=str, required=True,
                     help="record_policy.py-style .npz (e.g. sim_noball_seed0001.npz) -- source of the ground-truth "
                          "commanded target (`joint_pos_cmd`, 16-DOF actuated order).")
parser.add_argument("--real_ref", type=str, required=True,
                     help="Real hardware log .npz (e.g. real_noball_seed0001.npz) -- source of the ground-truth "
                          "real response (`joint_pos`, 13-DOF policy order) to compare the new sim rollout against.")
parser.add_argument("--settle_secs", type=float, default=1.0, help="Hold at the trajectory's first target before replay.")
parser.add_argument("--out_dir", type=str, default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "shadow_pd_id", "results", "rollouts", "full_hand"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG  # noqa: E402
from roto.tasks.physics import PHYSICS_DT, roto_sim_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Constants -- must match run_shadow.py / shadow_pd_id/config/joints.yaml
# ---------------------------------------------------------------------------

POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]

# Coupled driver (J2, policy-commanded) -> mimic (J1, coupling-derived). Only used
# for _to_proxy13's reporting/comparison convention below -- NOT for reconstructing
# commands (see main()'s docstring note on why the target is taken directly from
# the recorded 16-DOF joint_pos_cmd instead of re-deriving J1 from a J2 proxy).
COUPLED_DEP = {"rh_FFJ2": "rh_FFJ1", "rh_MFJ2": "rh_MFJ1", "rh_RFJ2": "rh_RFJ1"}

# Coulomb friction identified by shadow_pd_id/src/optimize.py (see
# shadow_pd_id/results/params/*_gains.yaml). Kp/Kd for these same joints are
# already baked into SHADOW_HAND_LITE_CFG's actuator cfg -- friction has no
# spawn-time cfg field on this Isaac Lab version, so it's applied here via a
# runtime call instead (same approach as shadow_pd_id/src/sim_rollout.py's
# SimRolloutEngine.set_gains).
IDENTIFIED_FRICTION = {
    "rh_FFJ4": 0.0618, "rh_MFJ4": 0.0618, "rh_RFJ4": 0.0618, "rh_THJ5": 0.0346,
    "rh_FFJ3": 0.0451, "rh_MFJ3": 0.0451, "rh_RFJ3": 0.0451, "rh_THJ4": 0.0807,
    "rh_FFJ2": 0.1277, "rh_MFJ2": 0.1277, "rh_RFJ2": 0.0155,
    "rh_THJ2": 0.0346, "rh_THJ1": 0.0085,
}


def _to_proxy13(data16: np.ndarray, actuated_names: list[str]) -> np.ndarray:
    """[T,16] actuated-order -> [T,13] policy-order proxy. Coupled joints (driver+mimic)
    collapse to their mean, matching plot_traj_compare.py / replay_traj_hw.py's convention
    (2*proxy = J2+J1 = the combined curl the hardware's single actuator produces)."""
    col = {n: i for i, n in enumerate(actuated_names)}
    out = []
    for jn in POLICY_JOINT_ORDER:
        mimic = COUPLED_DEP.get(jn)
        if mimic is not None:
            out.append(0.5 * (data16[:, col[jn]] + data16[:, col[mimic]]))
        else:
            out.append(data16[:, col[jn]])
    return np.stack(out, axis=1).astype(np.float32)


def _load_refs(sim_ref_path: str, real_ref_path: str):
    sim_ref = np.load(sim_ref_path, allow_pickle=True)
    real_ref = np.load(real_ref_path, allow_pickle=True)

    if list(sim_ref["actions"].shape) != list(real_ref["actions"].shape) or \
            not np.array_equal(sim_ref["actions"], real_ref["actions"]):
        raise ValueError(
            f"{sim_ref_path} and {real_ref_path} do not share identical `actions` -- they are not the same "
            "recorded episode (sim-command-then-hardware-replay pair). Refusing to mix mismatched files."
        )

    actuated_names = [str(n) for n in sim_ref["actuated_names"]]
    # Target is the recorded 16-DOF joint_pos_cmd AS-IS -- both the J2 driver's and
    # J1 mimic's own commanded values, exactly what RotoEnv._handle_coupled_joints
    # computed at record time (whatever coupling mode was active then). NOT
    # collapsed to a 13-dim mean-of-(J2,J1) proxy and re-split by a different
    # formula -- that round-trip was this script's original bug (see DECISIONS.md,
    # 2026-07-10): the mean and a re-derived split can differ from the true J2/J1
    # pair by ~0.3 rad on the coupled fingers, dwarfing the actual gain-fit error.
    target16 = sim_ref["joint_pos_cmd"].astype(np.float32)
    # sim_ref's OWN actual response, in the same 13-proxy reporting space real_pos13
    # is compared in -- whatever gains were active when that original record_policy.py
    # rollout ran (training-time defaults, not shadow_pd_id's identified values).
    # Kept as a reference line: answers "did the identified gains get this new
    # rollout closer to real than the old default gains were?"
    sim_ref_actual13 = _to_proxy13(sim_ref["joint_pos"], actuated_names)

    real_joint_names = [str(n) for n in real_ref["joint_names"]]
    if real_joint_names != POLICY_JOINT_ORDER:
        raise ValueError(f"{real_ref_path}'s joint_names {real_joint_names} != POLICY_JOINT_ORDER")
    real_pos13 = real_ref["joint_pos"].astype(np.float32)

    dt = float(real_ref["dt"])
    return target16, actuated_names, real_pos13, sim_ref_actual13, dt


def main():
    target16, actuated_names, real_pos13, sim_ref_actual13, dt = _load_refs(args_cli.sim_ref, args_cli.real_ref)
    T = target16.shape[0]
    control_rate_hz = 1.0 / dt
    steps_per_sample = round((1.0 / PHYSICS_DT) / control_rate_hz)
    print(f"[replay_traj_sim] {args_cli.real_ref}: {T} steps @ {control_rate_hz:.1f} Hz "
          f"({steps_per_sample} physics substeps/sample), command from {args_cli.sim_ref}")

    # -- Sim setup (identical physics config to training/shadow_pd_id) --------
    sim = sim_utils.SimulationContext(roto_sim_cfg)
    sim.set_camera_view(eye=(0, -0.6, 1.0), target=(0, -0.3, 0.5))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=1000.0)
    )
    robot = Articulation(SHADOW_HAND_LITE_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])

    joint_names_robot = list(robot.data.joint_names)
    n_robot_joints = len(joint_names_robot)
    name_to_ridx = {n: i for i, n in enumerate(joint_names_robot)}
    missing = [j for j in POLICY_JOINT_ORDER if j not in name_to_ridx]
    if missing:
        raise RuntimeError(f"Policy joints not found on robot: {missing}")

    # actuated_names (sim_ref's 16-DOF column order) -> this robot's own joint indices,
    # so target16[t] can be scattered into a full per-robot-joint command directly.
    actuated_ridx = [name_to_ridx[n] for n in actuated_names]

    # -- Apply identified Coulomb friction (Kp/Kd already baked into the cfg) -
    for joint_name, fc in IDENTIFIED_FRICTION.items():
        ridx = name_to_ridx[joint_name]
        robot.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=float(fc),
            joint_dynamic_friction_coeff=float(fc),
            joint_viscous_friction_coeff=0.0,
            joint_ids=[ridx],
        )

    def to_full_target(row16: np.ndarray) -> torch.Tensor:
        t = torch.zeros(1, n_robot_joints)
        t[0, actuated_ridx] = torch.as_tensor(row16, dtype=torch.float32)
        return t

    # -- Settle at the trajectory's first target before replay ----------------
    settle_steps = int(args_cli.settle_secs / PHYSICS_DT)
    settle_target = to_full_target(target16[0])
    for _ in range(settle_steps):
        robot.set_joint_position_target(settle_target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)

    # -- Replay ----------------------------------------------------------------
    sim_pos16 = np.zeros((T, len(actuated_names)), dtype=np.float32)
    for t in range(T):
        target = to_full_target(target16[t])
        for _ in range(steps_per_sample):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYSICS_DT)
        full_pos = robot.data.joint_pos[0].cpu().numpy()
        sim_pos16[t] = [full_pos[name_to_ridx[j]] for j in actuated_names]

    # Report/compare in the same 13-dim mean-of-(J2,J1) proxy space as sim_ref_actual13
    # and targets13 below, so old-gains vs new-gains vs real are all apples-to-apples.
    sim_pos13 = _to_proxy13(sim_pos16, actuated_names)
    targets13 = _to_proxy13(target16, actuated_names)

    # -- Save + plot BEFORE closing Isaac Sim -----------------------------------
    # Isaac Sim's headless shutdown (simulation_app.close(), called at the very end
    # of main()) is known to hang for 5-15+ minutes on this asset (see
    # shadow_pd_id/DECISIONS.md, 2026-07-08) -- writing outputs first means a
    # `timeout`-killed run still leaves usable results on disk.
    os.makedirs(args_cli.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args_cli.real_ref))[0]
    out_npz = os.path.join(args_cli.out_dir, f"{stem}_identified_gains_replay.npz")
    np.savez(out_npz, sim_pos=sim_pos13, real_pos=real_pos13, target=targets13,
             sim_ref_actual=sim_ref_actual13, joint_names=np.array(POLICY_JOINT_ORDER), dt=np.float32(dt))
    print(f"[replay_traj_sim] Saved {out_npz}")

    err_new = np.abs(sim_pos13 - real_pos13)
    err_old = np.abs(sim_ref_actual13 - real_pos13)
    print("\n--- mean abs error vs real (rad): identified-gains replay vs original sim_ref recording ---")
    print(f"{'joint':>10} | {'new gains':>10} {'old (sim_ref)':>14}")
    print("-" * 40)
    for j, jn in enumerate(POLICY_JOINT_ORDER):
        print(f"{jn:>10} | {err_new[:, j].mean():>10.4f} {err_old[:, j].mean():>14.4f}")
    print(f"{'ALL':>10} | {err_new.mean():>10.4f} {err_old.mean():>14.4f}")

    t_axis = np.arange(T) * dt
    ncols, nrows = 4, (13 + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), sharex=True)
    fig.suptitle(f"Full-hand sim replay vs real ({stem}, shadow_pd_id-identified gains)",
                 fontsize=13, fontweight="bold")
    axf = axes.flatten()
    for j, ax in enumerate(axf):
        if j >= 13:
            ax.set_visible(False)
            continue
        ax.plot(t_axis, targets13[:, j], "k--", lw=0.9, label="target (command)")
        ax.plot(t_axis, real_pos13[:, j], color="darkorange", lw=1.1, label="real actual")
        ax.plot(t_axis, sim_pos13[:, j], color="steelblue", lw=1.1, label="sim actual (identified gains)")
        ax.plot(t_axis, sim_ref_actual13[:, j], color="gray", lw=0.9, ls=":", label="sim actual (old/default gains)")
        ax.set_title(POLICY_JOINT_ORDER[j], fontsize=8, pad=2)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
    for ax in axf[(nrows - 1) * ncols:]:
        ax.set_xlabel("time (s)", fontsize=8)
    axf[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(args_cli.out_dir, f"{stem}_identified_gains_replay.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[replay_traj_sim] Saved {out_png}")

    simulation_app.close()


if __name__ == "__main__":
    main()
