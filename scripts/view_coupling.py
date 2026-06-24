"""Visually verify the J1/J2 coupled-finger behaviour in the GUI.

Launches the sim with the viewer and slowly sweeps the three coupled fingers
(FF/MF/RF) from extended to fully curled and back — no policy involved, the J2
"curl proxy" actions are driven directly. Watch how the J1 mimic follows its J2
driver (and, with couple_gate_j1_on_measured enabled, whether J1 waits for J2 to
physically reach its stop). The live measured J2/J1 angles are printed.

Run WITHOUT --headless so the viewer opens:
    python view_coupling.py --task Baoding --robot shadowlite
    python view_coupling.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt --period 8
    python view_coupling.py --task Baoding --robot shadowlite --hold -1.0   # other joints extended
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise the coupled-finger J1/J2 motion in the GUI.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default=None,
                    help="Any agent_cfg that loads (obs/checkpoint don't matter — joints are driven directly).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--period", type=float, default=6.0, help="Seconds per full extend→curl→extend cycle.")
parser.add_argument("--hold", type=float, default=0.0,
                    help="Action [-1,1] held on all non-swept joints (-1 = lower limit / open).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from multimodal_rl.tools.writer import Writer


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    raw = env.env.unwrapped
    control_names = list(getattr(raw.cfg, "control_joint_names"))
    # Action indices of the J2 drivers we sweep (FF/MF/RF J2), in action order.
    driver_names = list(raw.cfg.coupled_joint_map.values())
    sweep_idx = [control_names.index(n) for n in driver_names]
    rl_dt = raw.cfg.sim.dt * raw.cfg.decimation

    print(f"[INFO] Sweeping drivers {driver_names} (action idx {sweep_idx})")
    print(f"[INFO] Gating: couple_gate_j1_on_measured={raw.couple_gate_j1_on_measured}  "
          f"couple_gate_lo_frac={raw.couple_gate_lo_frac}")
    print(f"[INFO] RL dt = {rl_dt:.4f}s, sweep period = {args_cli.period}s. Ctrl+C to stop.\n")
    print(f"  {'proxy':>6} | {'FFJ2':>6} {'FFJ1':>6} | {'MFJ2':>6} {'MFJ1':>6} | {'RFJ2':>6} {'RFJ1':>6}   (measured rad)")

    drv = raw.coupled_driver_indices       # [FFJ2, MFJ2, RFJ2] joint indices
    dep = raw.coupled_dependent_indices    # [FFJ1, MFJ1, RFJ1] joint indices

    step = 0
    with torch.inference_mode():
        env.reset(hard=True)
        while simulation_app.is_running():
            # Triangle wave in [-1, 1]: extended (-1) -> curled (+1) -> extended.
            phase = (step * rl_dt / args_cli.period) % 1.0
            s = 2.0 * abs(2.0 * phase - 1.0) - 1.0
            action = torch.full((raw.num_envs, len(control_names)), args_cli.hold,
                                dtype=torch.float32, device=env.device)
            action[:, sweep_idx] = s
            _, _, terminated, truncated, _ = env.step(action)

            if step % 15 == 0:
                mj2 = raw.robot.data.joint_pos[0, drv].cpu().numpy()
                mj1 = raw.robot.data.joint_pos[0, dep].cpu().numpy()
                print(f"  {s:6.2f} | {mj2[0]:6.2f} {mj1[0]:6.2f} | "
                      f"{mj2[1]:6.2f} {mj1[1]:6.2f} | {mj2[2]:6.2f} {mj1[2]:6.2f}")

            if bool(terminated[0].item()) or bool(truncated[0].item()):
                env.reset(hard=True)
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
