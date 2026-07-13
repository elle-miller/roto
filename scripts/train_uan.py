#!/usr/bin/env python3
"""Train the UAN residual-torque policy for ShadowLite.

Boots Isaac Sim, builds the `UAN_Shadowlite` env directly from a yaml config
(no Hydra config-store registration -- see roto/scripts/UAN_PROGRESS.md D6
for why that's skipped), and reuses roto's own training stack (`make_env` /
`train_one_seed` from common_utils.py, same directory) end to end -- this is
the actual "reuse roto's skrl PPO" step: everything downstream of
`env_cfg`/`agent_cfg` here is identical to what train.py does for
Bounce/Baoding.

Usage:
    python train_uan.py --headless
    python train_uan.py --headless --num_envs 512
    python train_uan.py --headless --agent_cfg ../roto/tasks/uan_shadowlite/agents/shadowlite/my_variant.yaml
    python train_uan.py --headless --dataset /path/to/aligned/episode_dir
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Force line-buffered stdout -- see play_uan.py's matching comment for why
# this matters when Isaac Sim's shutdown sequence runs.
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)

parser = argparse.ArgumentParser(description="Train the UAN residual-torque policy for ShadowLite.")
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(_ROTO_ROOT, "roto", "tasks", "uan_shadowlite", "agents", "shadowlite", "default.yaml"),
    help="Path to the base agent yaml (dataset/uan/encoder/policy/value/agent/... sections).",
)
parser.add_argument(
    "--agent_cfg",
    type=str,
    default=None,
    help="Optional yaml merged OVER --config (only the keys present are overridden).",
)
parser.add_argument(
    "--dataset",
    type=str,
    action="append",
    default=None,
    help="Override dataset.paths (repeatable) -- directories, glob patterns, or explicit files.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
# NOTE: --device is intentionally NOT defined here -- AppLauncher.add_app_launcher_args()
# below already registers it and raises ValueError if a caller adds it again.
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _unused_hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything below touches isaaclab/omni and must be imported AFTER AppLauncher
# boots the simulator -- same ordering roto/scripts/train.py uses.
import yaml  # noqa: E402
from isaaclab.utils import update_dict  # noqa: E402

sys.path.insert(0, _THIS_DIR)  # so `import common_utils` resolves even if cwd differs
# `roto`'s own package is imported below (`from roto.tasks import ...`), which requires
# _ROTO_ROOT on sys.path. Prepending here is a defensive no-op when the active
# environment's `roto` editable install is already correct, but it's REQUIRED at least
# in this repo's `icra`/`s2r` conda envs, where the editable install was found to point
# at a different, stale directory entirely (discovered while first booting this script --
# see UAN_PROGRESS.md). Prepending here fixes resolution without touching site-packages.
sys.path.insert(0, _ROTO_ROOT)
from common_utils import LOG_PATH, make_env, train_one_seed, update_env_cfg  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401  (side effect: gym.register)
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402


def load_agent_cfg() -> dict:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.agent_cfg is not None:
        with open(args_cli.agent_cfg) as f:
            overlay = yaml.safe_load(f)
        agent_cfg = update_dict(agent_cfg, overlay)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    return agent_cfg


def main() -> None:
    agent_cfg = load_agent_cfg()
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = bool(agent_cfg["experiment"]["upload_videos"]) or args_cli.video
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = agent_cfg["dataset"]
    env_cfg.uan = agent_cfg["uan"]

    writer = Writer(agent_cfg)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"

    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING TRAINING:", err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
