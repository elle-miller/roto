"""Interactive GUI to verify the stateful backlash coupling by hand.

Opens the viewer with an omni.ui control panel. Drag the "combined curl m" slider
slowly up and down, stop, and reverse to watch the J1 mimic freeze and unlock:

  • curl past 100°  -> J2 fills first, THEN J1 starts (strict forward)
  • uncurl          -> J2 unlocks EARLY at R, J1 unwinds to 0 at m=100° (overlap)
  • stop in (100°,R) and drag back up -> J1 shows FROZEN until m returns to R, then resumes

Per-finger sliders (unlink FF to move it alone = independence), an R override, an
asymmetric on/off toggle, and a Reset button are provided. Live J2/J1 (commanded &
measured), R, direction and the FROZEN indicator are shown per finger.

Run WITHOUT --headless:
    python view_coupling_slider.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt
    python view_coupling_slider.py --with_ball         # keep the balls in the palm
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive slider viewer for the backlash coupling.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--with_ball", action="store_true", default=False, help="Keep balls (default: free space).")
parser.add_argument("--stiffness", type=float, default=20.0,
                    help="Runtime joint stiffness for a crisp, non-jittery view (0 = leave default soft gains).")
parser.add_argument("--damping", type=float, default=2.0, help="Runtime joint damping (paired with --stiffness).")
parser.add_argument("--spread_deg", type=float, default=35.0,
                    help="FF/RF knuckle abduction at startup (deg) so fingers don't self-collide. "
                         ">20 widens the physical J4 limit (viewer-only). Set 0/20 to keep the real ±20°.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# force the GUI on (slider needs the Kit app)
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import types
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from common_utils import (
    LOG_PATH, load_hand_task_agent_cfg, make_env, register_hand_task_to_hydra,
    resolve_gym_env_id, set_seed, update_env_cfg,
)
from multimodal_rl.tools.writer import Writer

DEG = 180.0 / math.pi
RAD = math.pi / 180.0


def _no_dones(self):
    z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return z, z.clone()


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    agent_cfg["seed"] = args_cli.seed
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # clean kinematic view: no settle override, no DR/noise, no auto-reset
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0
    if hasattr(env_cfg, "settle_steps"):
        env_cfg.settle_steps = 0
    if not args_cli.with_ball:
        for bcfg in (env_cfg.ball_1_cfg, env_cfg.ball_2_cfg):
            bcfg.spawn.rigid_props.kinematic_enabled = True
            bcfg.init_state.pos = (5.0, 5.0, 5.0)

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped
    raw._get_dones = types.MethodType(_no_dones, raw)

    control_names = list(raw.cfg.control_joint_names)
    driver_names  = list(raw.cfg.coupled_joint_map.values())   # FFJ2 MFJ2 RFJ2
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    fingers = [n.replace("rh_", "").replace("J2", "") for n in driver_names]

    theta = raw.coupling_theta
    j2u   = raw.robot_joint_pos_upper_limits[drv][0].item()
    j1u   = raw.robot_joint_pos_upper_limits[dep][0].item()
    lower = raw.robot_joint_pos_lower_limits[drv][0].item()
    m_top_deg = (j2u + j1u) * DEG

    def proxy_for_m(m_rad):
        if m_rad <= j2u:
            return m_rad * theta / j2u
        return theta + (m_rad - j2u) / j1u * (j2u - theta)

    def action_for_m(m_deg):
        p = proxy_for_m(m_deg * RAD)
        return 2.0 * (p - lower) / (j2u - lower) - 1.0

    # ---- build the omni.ui panel (fallback to auto-sweep if unavailable) ------
    ui = None
    try:
        import omni.ui as ui
    except Exception as e:  # pragma: no cover - GUI only
        print(f"[WARN] omni.ui unavailable ({e}); falling back to auto-sweep.")

    state = {
        "master": 0.0, "link": [True, True, True], "finger": [0.0, 0.0, 0.0],
        "use_R": False, "R": 136.0, "asym": bool(getattr(raw, "couple_asymmetric_backward", True)),
        "reset": False,
    }
    labels = {}

    if ui is not None:
        # Use value-changed CALLBACKS (not per-frame polling): in a standalone Kit
        # loop, polling slider models is unreliable (they read stale 0), whereas
        # callbacks fire on the UI thread the moment you drag. This fixes "stuck at 0".
        fsliders = []

        def on_master(m):
            v = m.get_value_as_float()
            state["master"] = v
            for i in range(3):
                if state["link"][i]:
                    state["finger"][i] = v
                    fsliders[i].model.set_value(v)   # keep linked sliders visually synced

        def make_on_finger(i):
            def _cb(m):
                state["finger"][i] = m.get_value_as_float()
            return _cb

        def make_on_link(i):
            def _cb(m):
                state["link"][i] = m.get_value_as_bool()
            return _cb

        win = ui.Window("Backlash Coupling", width=470, height=580)
        with win.frame:
            with ui.VStack(spacing=6):
                ui.Label("Combined curl m (deg) — drag slowly, stop, reverse",
                         style={"font_size": 16})
                master = ui.FloatSlider(min=0.0, max=m_top_deg, height=26)
                master.model.set_value(0.0)
                master.model.add_value_changed_fn(on_master)

                ui.Spacer(height=4)
                with ui.HStack(height=24):
                    asym = ui.CheckBox(width=24)
                    asym.model.set_value(state["asym"])
                    asym.model.add_value_changed_fn(lambda m: state.update(asym=m.get_value_as_bool()))
                    ui.Label("asymmetric backlash ON (uncheck = strict frac=1 gate)")

                with ui.HStack(height=24):
                    useR = ui.CheckBox(width=24)
                    useR.model.add_value_changed_fn(lambda m: state.update(use_R=m.get_value_as_bool()))
                    ui.Label("override R:", width=80)
                    Rsl = ui.FloatSlider(min=100.0, max=179.0)
                    Rsl.model.set_value(state["R"])
                    Rsl.model.add_value_changed_fn(lambda m: state.update(R=m.get_value_as_float()))

                ui.Spacer(height=4)
                ui.Label("Per-finger (uncheck 'link' to move alone):", style={"font_size": 14})
                for i, f in enumerate(fingers):
                    with ui.HStack(height=22):
                        lk = ui.CheckBox(width=24); lk.model.set_value(True)
                        lk.model.add_value_changed_fn(make_on_link(i))
                        ui.Label(f"{f} link", width=70)
                        fs = ui.FloatSlider(min=0.0, max=m_top_deg)
                        fs.model.add_value_changed_fn(make_on_finger(i))
                        fsliders.append(fs)

                ui.Spacer(height=6)
                rst = ui.Button("Reset (re-sample R + tilt)", height=28)
                rst.set_clicked_fn(lambda: state.update(reset=True))

                ui.Spacer(height=8)
                ui.Label("Live state (cmd / measured, deg):", style={"font_size": 14})
                for f in fingers:
                    labels[f] = ui.Label("", style={"font_size": 14})

        def pull_ui():
            pass   # state is pushed by callbacks; nothing to poll
    else:
        def pull_ui():
            # auto-sweep fallback: triangle wave on master
            t = (pull_ui.k * 0.01) % 2.0
            pull_ui.k += 1
            state["master"] = (t if t <= 1.0 else 2.0 - t) * m_top_deg
            for i in range(3):
                state["finger"][i] = state["master"]
        pull_ui.k = 0

    # Stiffen the hand for a crisp, non-jittery view (visualization only; the soft
    # default gains make the position target ring).
    if args_cli.stiffness > 0:
        raw.robot.write_joint_stiffness_to_sim(float(args_cli.stiffness))
        raw.robot.write_joint_damping_to_sim(float(args_cli.damping))

    # Spread the fingers apart at startup so they don't self-collide during the mimic
    # test. FF/RF knuckle abduction (J4) is already at its ±20° limit in the default
    # pose, so to splay further we widen the J4 position limit (viewer-only, non-
    # physical) and command a larger abduction. Negative J4 = away from the middle.
    jn = raw.robot.joint_names
    if args_cli.spread_deg > 20.0:
        j4 = [jn.index(n) for n in ("rh_FFJ4", "rh_MFJ4", "rh_RFJ4")]
        W = (args_cli.spread_deg + 10.0) * RAD
        lim = torch.tensor([[-W, W]], device=raw.device).repeat(len(j4), 1)
        fn = (getattr(raw.robot, "write_joint_position_limit_to_sim", None)
              or getattr(raw.robot, "write_joint_limits_to_sim", None))
        if fn is not None:
            try:
                fn(lim, joint_ids=j4)
            except TypeError:
                fn(lim, joint_ids=j4, env_ids=None)
        raw.robot_joint_pos_lower_limits[j4] = -W
        raw.robot_joint_pos_upper_limits[j4] = W
        print(f"[INFO] widened J4 limit to ±{(args_cli.spread_deg+10):.0f}°, "
              f"spreading FF/RF to {-args_cli.spread_deg:.0f}°")

    # Hold non-coupled joints at the default catch pose, then apply the wider spread.
    ctrl = raw.control_dof_indices
    lo_c = raw.robot_joint_pos_lower_limits[ctrl]
    hi_c = raw.robot_joint_pos_upper_limits[ctrl]
    defpos = raw.robot.data.default_joint_pos[0, ctrl].clone()
    base_action = (2.0 * (defpos - lo_c) / (hi_c - lo_c) - 1.0)   # default pose as [-1,1]
    if args_cli.spread_deg > 20.0:
        sp = -args_cli.spread_deg * RAD
        for nm in ("rh_FFJ4", "rh_RFJ4"):
            ci = control_names.index(nm)
            base_action[ci] = 2.0 * (sp - lo_c[ci]) / (hi_c[ci] - lo_c[ci]) - 1.0

    print("[INFO] Drag the sliders in the 'Backlash Coupling' window. Ctrl+C to quit.")

    loop_step = 0
    with torch.inference_mode():
        env.reset(hard=True)
        while simulation_app.is_running():
            pull_ui()
            loop_step += 1

            if state["reset"]:
                env.reset(hard=True)
                if args_cli.stiffness > 0:               # reset re-applies default soft gains
                    raw.robot.write_joint_stiffness_to_sim(float(args_cli.stiffness))
                    raw.robot.write_joint_damping_to_sim(float(args_cli.damping))
                state["master"] = 0.0
                state["finger"] = [0.0, 0.0, 0.0]
                if ui is not None:
                    master.model.set_value(0.0)
                    for fs in fsliders:
                        fs.model.set_value(0.0)
                state["reset"] = False

            raw.couple_asymmetric_backward = bool(state["asym"])
            if state["use_R"]:
                raw.couple_release[:] = state["R"] * RAD

            # hold the whole hand at its default catch pose, then drive the 3 fingers
            action = base_action.unsqueeze(0).repeat(raw.num_envs, 1).clone()
            for i, idx in enumerate(sweep_idx):
                action[:, idx] = action_for_m(state["finger"][i])
            env.step(action)

            # refresh readouts
            j2c = raw.joint_pos_cmd[0, drv].cpu().numpy() * DEG
            j1c = raw.joint_pos_cmd[0, dep].cpu().numpy() * DEG
            j2m = raw.robot.data.joint_pos[0, drv].cpu().numpy() * DEG
            j1m = raw.robot.data.joint_pos[0, dep].cpu().numpy() * DEG
            Rv  = raw.couple_release[0].cpu().numpy() * DEG
            dirv = raw.couple_dir[0].cpu().numpy()
            frz  = raw.couple_frozen_flag[0].cpu().numpy()
            fval = raw.couple_frozen_val[0].cpu().numpy() * DEG
            mv   = raw.prev_m[0].cpu().numpy() * DEG
            for i, f in enumerate(fingers):
                d = "CURL " if dirv[i] > 0 else "UNCURL"
                fr = f"  FROZEN@{fval[i]:.0f}°" if frz[i] else ""
                txt = (f"{f}: m={mv[i]:5.1f}  J2 {j2c[i]:5.1f}/{j2m[i]:5.1f}  "
                       f"J1 {j1c[i]:5.1f}/{j1m[i]:5.1f}  R={Rv[i]:3.0f}  {d}{fr}")
                if ui is not None:
                    labels[f].text = txt
                elif loop_step % 30 == 0:
                    print("  " + txt)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
