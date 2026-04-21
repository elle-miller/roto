# =============================================================================
# INSPECTION UTILITY — Shadow Hand Lite / PeaceSign
# Run this standalone to check observation dims, action dims, and joint ordering.
# Does NOT modify common_utils.py.
# Usage: python inspect_shadowlite.py
# =============================================================================

import numpy as np


def inspect_peacesign_shadowlite(agent_cfg_entry_point: str = "rl_only_pt"):
    """
    Instantiate the PeaceSign_Shadowlite environment and print:
      - observation space keys and shapes
      - number of actions (policy output size)
      - joint names and ordering from env_cfg
      - encoder/policy input-output dimensions

    Args:
        agent_cfg_entry_point: which agent yaml to load (default: "rl_only_pt")
    """

    # ------------------------------------------------------------------ #
    # 1. Load env + agent config (mirrors what train.py does)
    # ------------------------------------------------------------------ #
    from common_utils import (
        load_hand_task_agent_cfg,
        resolve_gym_env_id,
        update_env_cfg,
    )
    from roto.tasks.peace.peace import PeaceSignCfg

    task_name = "PeaceSign"
    robot     = "shadowlite"

    print("\n" + "="*60)
    print("SHADOW HAND LITE — PEACESIGN INSPECTION")
    print("="*60)

    # Load agent yaml
    agent_cfg = load_hand_task_agent_cfg(task_name, robot, agent_cfg_entry_point)
    print(f"\n[OK] Loaded agent config: {agent_cfg_entry_point}")

    # Build env config
    env_cfg = PeaceSignCfg()
    env_cfg = update_env_cfg_minimal(env_cfg, agent_cfg)
    print(f"[OK] Built env config: {env_cfg.__class__.__name__}")

    # ------------------------------------------------------------------ #
    # 2. Joint names and ordering
    # ------------------------------------------------------------------ #
    print("\n--- JOINT ORDERING (policy order) ---")
    if hasattr(env_cfg, "joint_names"):
        for i, name in enumerate(env_cfg.joint_names):
            print(f"  [{i:02d}] {name}")
        print(f"\nTotal joints: {len(env_cfg.joint_names)}")
    else:
        print("  env_cfg has no 'joint_names' attribute — check PeaceSignCfg manually")

    # ------------------------------------------------------------------ #
    # 3. Action space
    # ------------------------------------------------------------------ #
    print("\n--- ACTION SPACE ---")
    if hasattr(env_cfg, "num_actions"):
        print(f"  num_actions: {env_cfg.num_actions}")
    else:
        print("  env_cfg has no 'num_actions' attribute")

    # ------------------------------------------------------------------ #
    # 4. Observation space keys and shapes
    # ------------------------------------------------------------------ #
    print("\n--- OBSERVATION SPACE (from agent cfg) ---")
    obs_cfg = agent_cfg.get("observations", {})
    obs_list = obs_cfg.get("obs_list", [])
    obs_stack = obs_cfg.get("obs_stack", 1)
    print(f"  obs_list:  {obs_list}")
    print(f"  obs_stack: {obs_stack}")

    # ------------------------------------------------------------------ #
    # 5. Try to spin up the env and inspect real spaces
    # ------------------------------------------------------------------ #
    print("\n--- LIVE ENVIRONMENT SPACES ---")
    print("  (attempting to create env — requires Isaac Lab / GPU)")
    try:
        import gymnasium as gym
        import argparse
        from common_utils import make_env

        # minimal fake args
        class FakeArgs:
            num_envs = 1
            device   = "cpu"
            video    = False
            task     = "PeaceSign_Shadowlite"
            gym_env_id = "PeaceSign_Shadowlite"

        from multimodal_rl.rl.writer import DummyWriter
        writer = DummyWriter()

        env = make_env(agent_cfg, env_cfg, writer, FakeArgs())

        print(f"\n  Observation space keys:")
        for k, v in env.observation_space["policy"].items():
            print(f"    '{k}': shape={v.shape}  dtype={v.dtype}")

        print(f"\n  Action space:")
        print(f"    shape={env.action_space.shape}  dtype={env.action_space.dtype}")

        total_obs = sum(
            int(np.prod(v.shape))
            for v in env.observation_space["policy"].values()
        )
        print(f"\n  Total observation dims: {total_obs}")
        print(f"  Total action dims:      {env.action_space.shape[-1]}")

        env.close()

    except Exception as e:
        print(f"  Could not create live env: {e}")
        print("  (this is fine — use the cfg values above instead)")

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    print("\n--- SUMMARY (use these in run_RL.py) ---")
    if hasattr(env_cfg, "num_actions"):
        print(f"  num_actions = {env_cfg.num_actions}")
    if hasattr(env_cfg, "joint_names"):
        print(f"  policy_joint_order = {list(env_cfg.joint_names)}")
    print("="*60 + "\n")


def update_env_cfg_minimal(env_cfg, agent_cfg):
    """Minimal version of update_env_cfg that doesn't need args_cli."""
    env_cfg.seed = agent_cfg.get("seed", 42)
    env_cfg.debug = agent_cfg.get("experiment", {}).get("debug", False)
    obs_cfg = agent_cfg.get("observations", {})
    env_cfg.obs_list  = obs_cfg.get("obs_list", getattr(env_cfg, "obs_list", []))
    env_cfg.obs_stack = obs_cfg.get("obs_stack", getattr(env_cfg, "obs_stack", 1))
    num_eval_envs = agent_cfg.get("trainer", {}).get("num_eval_envs", 0)
    env_cfg.num_eval_envs = num_eval_envs
    return env_cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Shadow Hand Lite PeaceSign env")
    parser.add_argument(
        "--agent",
        default="rl_only_pt",
        help="Agent config key to load (default: rl_only_pt). "
             "Options: default_cfg, rl_only_pt, rl_only_ptd, rl_only_ptg, "
             "tac_recon, full_recon, forward_dynamics, tac_dynamics",
    )
    args = parser.parse_args()
    inspect_peacesign_shadowlite(agent_cfg_entry_point=args.agent)