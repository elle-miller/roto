#!/usr/bin/env python3
"""Query real rigid-body dynamics (M(q), C(q,qdot), G(q)) for GenAN's Position loss.

Isaac-dependent (boots Isaac Sim), placed in `scripts/` alongside every other
Isaac-booting script in this repo (`play_genan.py`, `train_uan.py`), matching
`shadow_pd_id/src/sim_rollout.py`'s proven pattern of loading
`SHADOW_HAND_LITE_CFG` standalone -- headless, no gym task, no RL.

This is the ONE non-differentiable simulator touch-point in the whole
Position-loss pipeline, and it happens exactly once per data point, offline:
for every (smoothed) recorded state `(q_t, qdot_t, qddot_t)` produced by
`preprocess.py`, this script queries Isaac Lab's PhysX tensor API
(`Articulation.root_physx_view`) for:

    M(q_t)      -- get_generalized_mass_matrices()
    C(q_t, qdot_t) -- get_coriolis_and_centrifugal_compensation_forces()
    G(q_t)      -- get_gravity_compensation_forces()

These are plain numeric queries against directly-written joint state, not
autograd operations -- train_genan.py/train_genan_single.py treat the saved
M_inv/C/G as CONSTANTS (no `grad_fn`), so nothing about PhysX being
non-differentiable matters once this script's output is on disk. This is
functionally RNEA (`tau = M(q)*qddot + C(q,qdot) + G(q)`), just computed via
three tensor queries against the real robot model instead of a literal
recursive Newton-Euler pass -- see
`.../omni.physx.demos.../InverseDynamicsTensorAPIDemo.py`'s
`_apply_inverse_dynamics` for the reference call sequence this mirrors.

GRAVITY CAVEAT (see roto/genan/DESIGN.md and this repo's SHADOW_HAND_LITE_CFG):
the shared robot spawn config sets `disable_gravity=True` on the articulation's
rigid bodies (used elsewhere for RL/eval numerical convenience) -- if left as
is, `get_gravity_compensation_forces()` would return zero, which is wrong
physics for matching real hardware (which experiences real gravity). This
script builds its own copy of the config with `disable_gravity=False`,
constructed via `.replace()` at every nesting level (never mutating the
shared module-level `SHADOW_HAND_LITE_CFG` singleton in place, since
`.replace()` is a shallow copy and mutating a nested field would leak into
any other code in the same process that imported the original config).

COUPLING: ShadowLite's active USD models all 16 joints as independent DOFs
(no PhysX mimic/gear/tendon constraint -- see DESIGN.md) -- confirmed by
direct inspection, zero `mimic`/`gear`/`coupl` tokens in the active USD. So a
plain 16-DOF query against this exact model is dynamically correct as-is; no
adjustment for the "3 pairs share 1 motor" software-only fact is needed here.

Usage:
    python compute_dynamics.py --in cache/smoothed.npz --out cache/dynamics.npz --headless
    python compute_dynamics.py --in cache/smoothed.npz --out cache/dynamics.npz --num_envs 256 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)
_GENAN_DIR = os.path.join(_ROTO_ROOT, "genan")

parser = argparse.ArgumentParser(description="Query M(q)/C(q,qdot)/G(q) for GenAN's Position loss via Isaac's PhysX tensor API.")
parser.add_argument("--in", dest="in_path", type=str, default=os.path.join(_GENAN_DIR, "cache", "smoothed.npz"),
                     help="preprocess.py's output .npz (q_meas_smooth/q_dot/q_ddot).")
parser.add_argument("--out", type=str, default=os.path.join(_GENAN_DIR, "cache", "dynamics.npz"),
                     help="Output .npz: M_inv/C/G/tau_target, row-aligned to --in.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel articulation replicas per batch.")
parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (debugging).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _unused = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, _ROTO_ROOT)
sys.path.insert(0, _GENAN_DIR)
from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG  # noqa: E402
from roto.tasks.physics import roto_sim_cfg  # noqa: E402


def _gravity_enabled_robot_cfg(prim_path: str):
    """A copy of SHADOW_HAND_LITE_CFG with gravity enabled on the robot's
    rigid bodies, built via chained `.replace()` calls (never mutating the
    shared singleton in place -- see module docstring's GRAVITY CAVEAT).
    """
    new_rigid_props = SHADOW_HAND_LITE_CFG.spawn.rigid_props.replace(disable_gravity=False)
    new_spawn = SHADOW_HAND_LITE_CFG.spawn.replace(rigid_props=new_rigid_props)
    return SHADOW_HAND_LITE_CFG.replace(prim_path=prim_path, spawn=new_spawn)


@configclass
class _DynamicsSceneCfg(InteractiveSceneCfg):
    """Minimal scene: just the robot, no ground/contacts/cameras -- this
    script only ever queries dynamics at directly-written joint states, it
    never needs anything to actually fall, collide, or render.
    """

    robot: object = _gravity_enabled_robot_cfg("/World/envs/env_.*/Robot")


def build_scene(num_envs: int, device: str) -> tuple[InteractiveScene, sim_utils.SimulationContext]:
    sim = sim_utils.SimulationContext(roto_sim_cfg.replace(device=device))
    scene_cfg = _DynamicsSceneCfg(num_envs=num_envs, env_spacing=1.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return scene, sim


def query_batch(
    robot: Articulation, sim: sim_utils.SimulationContext, q: torch.Tensor, qdot: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Write `(q, qdot)` (each (B, 16), B <= robot's articulation count) into
    the sim and read back M(q)/C(q,qdot)/G(q) for those exact states.

    No physics stepping happens here -- these are direct-write + numeric-read
    queries against PhysX's own analytic rigid-body relations at the written
    state, not a forward simulation. `write_joint_state_to_sim` calls
    `root_physx_view.set_dof_positions/velocities(...)` directly, but the
    PhysX tensor API's own docs (`update_articulations_kinematic`'s
    docstring, `omni.physics.tensors/impl/api.py`) state link kinematics
    must be explicitly refreshed after a direct joint-state write before any
    dependent query -- `sim.physics_sim_view.update_articulations_kinematic()`
    is exactly that refresh (the same call `SimulationContext.forward()` makes
    internally, called directly here rather than relying on that method's
    fabric-availability guard).
    """
    b = q.shape[0]
    env_ids = torch.arange(b, device=q.device)
    full_qdot = torch.zeros_like(q)
    full_qdot[:] = qdot
    robot.write_joint_state_to_sim(q, full_qdot, env_ids=env_ids)
    sim.physics_sim_view.update_articulations_kinematic()
    physx_view = robot.root_physx_view
    m = physx_view.get_generalized_mass_matrices()[:b]
    c = physx_view.get_coriolis_and_centrifugal_compensation_forces()[:b]
    g = physx_view.get_gravity_compensation_forces()[:b]
    return {"M": m.clone(), "C": c.clone(), "G": g.clone()}


def main() -> None:
    data = np.load(args_cli.in_path)
    q_all = torch.as_tensor(data["q_meas_smooth"], dtype=torch.float32)
    qdot_all = torch.as_tensor(data["q_dot"], dtype=torch.float32)
    qddot_all = torch.as_tensor(data["q_ddot"], dtype=torch.float32)
    num_rows = q_all.shape[0] if args_cli.limit is None else min(args_cli.limit, q_all.shape[0])
    num_joints = q_all.shape[1]

    scene, sim = build_scene(num_envs=args_cli.num_envs, device=args_cli.device)
    robot: Articulation = scene["robot"]
    print(f"[INFO] Scene built: {scene.num_envs} envs, {num_joints} joints, device={args_cli.device}.")

    m_inv_all = torch.zeros(num_rows, num_joints, num_joints)
    c_all = torch.zeros(num_rows, num_joints)
    g_all = torch.zeros(num_rows, num_joints)
    tau_target_all = torch.zeros(num_rows, num_joints)

    batch = args_cli.num_envs
    for start in range(0, num_rows, batch):
        end = min(start + batch, num_rows)
        q_batch = q_all[start:end].to(args_cli.device)
        qdot_batch = qdot_all[start:end].to(args_cli.device)
        qddot_batch = qddot_all[start:end].to(args_cli.device)
        result = query_batch(robot, sim, q_batch, qdot_batch)
        m_inv_batch = torch.linalg.inv(result["M"])
        tau_target_batch = torch.einsum("bij,bj->bi", result["M"], qddot_batch) + result["C"] + result["G"]

        m_inv_all[start:end] = m_inv_batch.cpu()
        c_all[start:end] = result["C"].cpu()
        g_all[start:end] = result["G"].cpu()
        tau_target_all[start:end] = tau_target_batch.cpu()
        if start == 0:
            g_nonzero = result["G"].abs().max().item()
            print(f"[INFO] Sanity check: max|G(q)| over first batch = {g_nonzero:.6f} "
                  f"({'OK, gravity is active' if g_nonzero > 1e-6 else 'WARNING: gravity compensation is all-zero!'})")
        print(f"[INFO] Processed rows {start}:{end} / {num_rows}")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez(
        args_cli.out,
        M_inv=m_inv_all.numpy(),
        C=c_all.numpy(),
        G=g_all.numpy(),
        tau_target=tau_target_all.numpy(),
    )
    print(f"[INFO] Saved M_inv/C/G/tau_target for {num_rows} rows to {args_cli.out}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING compute_dynamics.py:", err)
        raise
    finally:
        simulation_app.close()
