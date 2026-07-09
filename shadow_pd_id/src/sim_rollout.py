#!/usr/bin/env python3
"""Replay a real command trajectory through Isaac Sim under candidate PD gains.

WHY THIS FILE EXISTS: this is the heart of "Method 2" (sim-in-the-loop
identification). Step 4's optimizer will call this hundreds of times with
different candidate {Kp, Kd, Fc} to see which gains make the SIMULATED joint
move the way the REAL joint actually moved, given the exact same commands.

Two hard constraints on how this must be built, both learned the hard way
while testing this exact Shadow Hand Lite asset in this project (see
DECISIONS.md):

  1. Loading the asset is slow (~1 minute) and Isaac Sim hangs for a long
     time on headless shutdown. So this file boots Isaac Sim and loads the
     robot asset EXACTLY ONCE (SimRolloutEngine.__init__), and every
     subsequent call to `.rollout()` reuses that same instance. Never
     construct a new SimRolloutEngine per optimizer iteration.

  2. Isaac Sim modules cannot be imported before `AppLauncher` runs (Carbonite
     plugin system requirement) -- so this file follows the same shape as
     roto/scripts/collect_traj_sim.py: parse minimal CLI args, launch the app,
     THEN import isaaclab.* and roto.*.

PARAMETERIZATION: gains are {Kp (stiffness), Kd (damping), Fc (Coulomb /
dynamic friction coefficient)} per joint -- 3 free parameters, not 4. Viscous
friction is deliberately fixed at 0 and folded into Kd; see DECISIONS.md for
why (on a position-commanded joint, viscous friction and damping are
mathematically identical -- torque = -coeff*velocity -- so they are not
separately identifiable from motion data, only their sum is).

Standalone test (this file's __main__ block, matching plan Step 2's
deliverable): roll out one joint's REAL recorded command trajectory twice
against the SAME persistent sim instance --  once with the asset's current
default gains, once with a deliberately different set of gains -- and plot
both against the reference. This proves two things at once: (a) the
plumbing works (no crashes, no NaNs, in-limits), and (b) set_gains() is
actually doing something (the two rollouts must differ from each other),
which a same-gains-only test would not have proven.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)  # so `from load_data import load_joint_config` works when run as a script

# IMPORT-SAFETY: this file is dual-use -- a standalone script
# (`python sim_rollout.py --reference_file ...`) AND a library imported for its
# SimRolloutEngine class by collect_rollouts.py / validate.py. The CLI parse and
# AppLauncher launch must therefore run ONLY when this file is the entrypoint. If
# they ran at import time (as they used to), `from sim_rollout import
# SimRolloutEngine` would (a) re-parse the IMPORTER's argv with this file's parser
# -- e.g. reject collect_rollouts.py's --hw_dir, failing AFTER the importer's own
# ~6s Isaac boot -- and (b) launch a SECOND Isaac Sim app on top of the one the
# importer already started. When imported, the importer runs AppLauncher before
# importing us, so the module-level isaaclab.* imports below still resolve.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_file", type=str, default=None,
                         help="A collected .npz (from collect_traj_sim.py/collect_traj_hw.py) to use "
                              "as the 'real' trajectory for the standalone plumbing test.")
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(_PROJECT_ROOT, "results", "plots", "sim_rollout_test"))
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG
from roto.tasks.physics import PHYSICS_DT, roto_sim_cfg

from load_data import load_joint_config  # noqa: E402  (must come after AppLauncher)


class SimRolloutEngine:
    """One persistent Isaac Sim instance, reused across every `.rollout()` call.

    Deterministic by construction: `roto_sim_cfg` already sets
    `enable_enhanced_determinism=True` (see roto/roto/tasks/physics.py), and
    every rollout starts from the same commanded-zero settle (see `.reset()`)
    rather than carrying over state from whatever the previous rollout ended
    at -- otherwise gains fit on iteration N would depend on the arbitrary
    ending state of iteration N-1, silently poisoning the optimization.
    """

    def __init__(self, cfg: dict, device: str = "cpu", settle_secs: float = 1.5, simulation_app=None):
        self.cfg = cfg
        self.device = device
        # The SimulationApp is owned by whoever launched AppLauncher -- this
        # file's __main__ block when run standalone, or collect_rollouts.py /
        # validate.py when they import this class. close() shuts it down through
        # this handle instead of a module-level global, which no longer exists
        # in the imported case (see the import-safety note at module top).
        self._simulation_app = simulation_app
        self.control_rate_hz = cfg["control_rate_hz"]
        self.policy_joint_order = cfg["policy_joint_order"]
        self.joint_limits = cfg["joint_limits_rad"]
        self.coupled_by_driver = {g["driver_joint"]: g for g in cfg["coupled_groups"]}
        self.settle_secs = settle_secs
        # 60 Hz command samples, held constant across this many 1/PHYSICS_DT
        # physics steps -- matches the real position controller only updating
        # its setpoint at 60 Hz (see collect_traj_sim.py, same convention).
        self.steps_per_sample = round((1.0 / PHYSICS_DT) / self.control_rate_hz)

        self.sim = sim_utils.SimulationContext(roto_sim_cfg)
        self.sim.set_camera_view(eye=(0, -0.6, 1.0), target=(0, -0.3, 0.5))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
            "/World/light", sim_utils.DomeLightCfg(intensity=1000.0)
        )

        robot_cfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/Robot")
        self.robot = Articulation(robot_cfg)
        self.sim.reset()
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])

        self.joint_names = list(self.robot.data.joint_names)
        self.n_robot_joints = len(self.joint_names)
        self.name_to_ridx = {n: i for i, n in enumerate(self.joint_names)}
        missing = [j for j in self.policy_joint_order if j not in self.name_to_ridx]
        if missing:
            raise RuntimeError(f"policy_joint_order names not found on robot: {missing}")

    def _resolve_pose_to_target(self, pose_13: np.ndarray) -> torch.Tensor:
        """Convert a 13-length policy-order pose (proxy values) into a full
        per-robot-joint torch target, resolving any of the 3 coupled drivers
        present in the pose into their (J2, J1) pair -- needed because a
        parked pose (see reset()/rollout() docstrings) can command a coupled
        driver to a nonzero curl (e.g. FFJ2/MFJ2/RFJ2 parked while exciting a
        thumb joint), not just whichever joint is actively being excited.
        """
        target = torch.zeros(1, self.n_robot_joints, device=self.device)
        for pi, joint_name in enumerate(self.policy_joint_order):
            proxy = float(pose_13[pi])
            coupled = self.coupled_by_driver.get(joint_name)
            if coupled is not None:
                upper = self.joint_limits[joint_name]["upper"]
                j1_upper = self.joint_limits[coupled["mimic_joint"]]["upper"]
                theta = coupled["coupling_theta_rad"]
                j2 = float(np.clip(proxy * (upper / theta), 0.0, upper))
                j1 = float(np.clip((proxy - theta) / (upper - theta) * j1_upper, 0.0, j1_upper))
                target[0, self.name_to_ridx[joint_name]] = j2
                target[0, self.name_to_ridx[coupled["mimic_joint"]]] = j1
            else:
                target[0, self.name_to_ridx[joint_name]] = proxy
        return target

    def set_gains(self, joint_name: str, kp: float, kd: float, fc: float = 0.0) -> None:
        """Set Kp/Kd/Coulomb-friction for one joint. Fv is intentionally not exposed (see module docstring)."""
        ridx = self.name_to_ridx[joint_name]
        self.robot.write_joint_stiffness_to_sim(float(kp), joint_ids=[ridx])
        self.robot.write_joint_damping_to_sim(float(kd), joint_ids=[ridx])
        # Coulomb friction is a SINGLE coefficient fc here, so static == dynamic == fc.
        # PhysX rejects the write (NpArticulationJointReducedCoordinate.cpp:209) if the
        # static effort < the dynamic effort -- setting static=0 while sweeping a nonzero
        # dynamic fc (as this used to) made PhysX silently drop the write, so the entire
        # fc dimension of the search never reached the sim. Keeping them equal both
        # satisfies PhysX and matches the intended Coulomb (velocity-independent) model.
        self.robot.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=float(fc),
            joint_dynamic_friction_coeff=float(fc),
            joint_viscous_friction_coeff=0.0,
            joint_ids=[ridx],
        )

    def reset(self, default_pose: np.ndarray | None = None) -> None:
        """Command `default_pose` (or all-zero if None) and let physics settle.

        `default_pose` is a 13-length policy-order pose -- normally the
        non-excited fingers' parked-out-of-the-way pose loaded from a
        collected log's `default_pose` field (see DECISIONS.md, 2026-07-08:
        holding neighboring fingers at zero let them collide with the finger
        being excited on real hardware). `None` preserves the old all-zero
        behavior for callers that don't have a pose yet (e.g. quick
        plumbing tests).

        Uses the same "soft reset via commanded target" pattern already
        proven in collect_traj_sim.py, rather than an unverified hard
        `write_joint_state_to_sim` call outside the DirectRLEnv framework
        this repo normally uses it in. If settle time becomes a bottleneck
        once Step 4 is running hundreds of rollouts, a hard reset is the
        first thing to revisit (and test properly) -- not before.
        """
        pose = default_pose if default_pose is not None else np.zeros(len(self.policy_joint_order), dtype=np.float32)
        target = self._resolve_pose_to_target(pose)
        settle_steps = int(self.settle_secs / PHYSICS_DT)
        for _ in range(settle_steps):
            self.robot.set_joint_position_target(target)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(PHYSICS_DT)

    def rollout(self, joint_idx: int, cmd: np.ndarray, default_pose: np.ndarray | None = None) -> np.ndarray:
        """Replay `cmd` (60 Hz proxy values for `joint_idx`) and return simulated position at 60 Hz.

        Non-excited joints are held at `default_pose` (all-zero if None) for
        every sample; the excited joint (+ its coupled mimic, if any) is
        overwritten with this step's commanded proxy on top of that baseline
        -- see `reset()`'s docstring for why this is no longer plain zero.

        Returns an array the SAME LENGTH as `cmd` (one simulated position per
        input command sample), obtained by sub-stepping physics
        `steps_per_sample` times per sample and keeping only the position at
        the end of each held segment -- i.e. sampled at the same 60 Hz the
        command was defined at, so it lines up directly with a real log's
        `q_filt` for loss computation.
        """
        pose = default_pose if default_pose is not None else np.zeros(len(self.policy_joint_order), dtype=np.float32)
        base_target = self._resolve_pose_to_target(pose)

        joint_name = self.policy_joint_order[joint_idx]
        limits = self.joint_limits[joint_name]
        coupled = self.coupled_by_driver.get(joint_name)
        if coupled is not None:
            j1_upper = self.joint_limits[coupled["mimic_joint"]]["upper"]
            theta = coupled["coupling_theta_rad"]
            mimic_ridx = self.name_to_ridx[coupled["mimic_joint"]]
        ridx = self.name_to_ridx[joint_name]

        sim_q = np.zeros(len(cmd), dtype=np.float32)

        for sample_idx in range(len(cmd)):
            proxy = float(np.clip(cmd[sample_idx], limits["lower"], limits["upper"]))
            target = base_target.clone()  # non-excited joints stay at their parked pose

            if coupled is not None:
                j2 = float(np.clip(proxy * (limits["upper"] / theta), 0.0, limits["upper"]))
                j1 = float(np.clip((proxy - theta) / (limits["upper"] - theta) * j1_upper, 0.0, j1_upper))
                target[0, ridx] = j2
                target[0, mimic_ridx] = j1
            else:
                target[0, ridx] = proxy

            for _ in range(self.steps_per_sample):
                self.robot.set_joint_position_target(target)
                self.robot.write_data_to_sim()
                self.sim.step(render=False)
                self.robot.update(PHYSICS_DT)

            sim_q[sample_idx] = self.robot.data.joint_pos[0, ridx].item()

        return sim_q

    def close(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app.close()


def _standalone_test():
    """Plan Step 2's deliverable: prove the plumbing works AND is gain-sensitive."""
    cfg = load_joint_config()
    engine = SimRolloutEngine(cfg, simulation_app=simulation_app)

    if args_cli.reference_file is None:
        print("[sim_rollout] No --reference_file given; nothing to test against. Exiting.")
        engine.close()
        return

    ref = np.load(args_cli.reference_file, allow_pickle=True)
    joint_idx = int(ref["joint_idx"])
    joint_name = str(ref["joint_name"])
    cmd = ref["cmd"]
    ref_q = ref["actual_pos"][:, joint_idx]
    default_pose = np.asarray(ref["default_pose"], dtype=np.float32) if "default_pose" in ref else None

    print(f"[sim_rollout] Testing joint {joint_idx} ({joint_name}), {len(cmd)} samples.")

    # Run 1: default asset gains (whatever SHADOW_HAND_LITE_CFG currently ships).
    engine.reset(default_pose)
    default_kp = float(engine.robot.data.joint_stiffness[0, engine.name_to_ridx[joint_name]])
    default_kd = float(engine.robot.data.joint_damping[0, engine.name_to_ridx[joint_name]])
    print(f"[sim_rollout] Default gains: Kp={default_kp:.3f}  Kd={default_kd:.3f}")
    q_default = engine.rollout(joint_idx, cmd, default_pose)

    # Run 2: deliberately different gains, same persistent instance, same
    # commands. If this doesn't diverge from run 1, set_gains() is not
    # actually reaching the sim -- this is the check a same-gains-only test
    # would have missed.
    engine.set_gains(joint_name, kp=3.0 * default_kp, kd=3.0 * default_kd, fc=0.0)
    engine.reset(default_pose)
    q_stiffer = engine.rollout(joint_idx, cmd, default_pose)

    engine.close()

    diff = np.abs(q_default - q_stiffer)
    print(f"[sim_rollout] max|q_default - q_stiffer| = {diff.max():.5f} rad "
          f"({'PASS: gains are having an effect' if diff.max() > 1e-4 else 'FAIL: no sensitivity to gains!'})")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(len(cmd)) / cfg["control_rate_hz"]
    ax.plot(t, cmd, "k--", lw=1, label="commanded")
    ax.plot(t, ref_q, lw=1.2, label="reference (this file's own recording)")
    ax.plot(t, q_default, lw=1.2, label=f"sim rollout, default gains (Kp={default_kp:.2f}, Kd={default_kd:.2f})")
    ax.plot(t, q_stiffer, lw=1.2, label="sim rollout, 3x stiffer gains")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (rad)")
    ax.set_title(f"{joint_name} — sim_rollout.py plumbing test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(args_cli.out_dir, f"joint_{joint_idx:02d}_{joint_name}_rollout_test.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[sim_rollout] Saved {out_path}")


if __name__ == "__main__":
    _standalone_test()
