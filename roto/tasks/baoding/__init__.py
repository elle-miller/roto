# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central baoding task registration (all robots share the ``Baoding`` gym id)."""

import os

import gymnasium as gym

from . import agents
from .baoding import (
    BaodingAllegroCfg,
    BaodingAllegroEnv,
    BaodingCfg,
    BaodingOrcaCfg,
    BaodingOrcaEnv,
    BaodingShadowEnv,
    BaodingShadowLiteCfg,
    BaodingShadowLiteEnv,
    BaodingShadowLitePadTacCfg,
    BaodingShadowLitePadTacEnv,
    BaodingShadowLitePadTacBTCfg,
    BaodingShadowLitePadTacBTEnv,
    BaodingShadowLitePadTacBTSparseCfg,
    BaodingShadowLitePadTacBTStuck8Cfg,
    BaodingShadowLitePadTacBTLegacyCfg,
    BaodingShadowLitePadTacBTLegacyNoSlewCfg,
    BaodingShadowLitePadTacBTLegacyNoMassDRCfg,
    BaodingShadowLitePadTacBTLegacyNoTacCorruptCfg,
    BaodingShadowLitePadTacBTLegacyFrictionMassOnlyCfg,
)

_AGENTS_DIR = os.path.dirname(agents.__file__)

_SHADOW_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
}

# PadTac / PadTac+BT agent configs live alongside the classic shadowlite ones in
# agents/shadowlite/.
_SHADOWLITE_PADTAC_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt_padtac": "rl_only_pt_padtac.yaml",
    "rl_only_pt_padtac_bt": "rl_only_pt_padtac_bt.yaml",
    "rl_only_pt_padtac_bt_sweep": "rl_only_pt_padtac_bt_sweep.yaml",
    "forward_dynamics_padtac_bt": "forward_dynamics_padtac_bt.yaml",
}


def _variant_paths(robot_subdir: str, variant_files: dict[str, str]) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, robot_subdir)
    return {key: os.path.join(base, filename) for key, filename in variant_files.items()}


def baoding_make_env(cfg, render_mode: str | None = None, **kwargs):
    """Instantiate the correct env class from the config type (set from ``--robot`` in training scripts)."""
    reg_keys = (
        set(_SHADOW_VARIANT_FILES)
        | set(_SHADOWLITE_PADTAC_VARIANT_FILES)
        | {"env_cfg_entry_point"}
    )
    for k in reg_keys:
        kwargs.pop(k, None)
    if isinstance(cfg, BaodingOrcaCfg):
        return BaodingOrcaEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    if isinstance(cfg, BaodingAllegroCfg):
        return BaodingAllegroEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    # NOTE: check the BT subclass BEFORE its PadTac parent (isinstance would match both).
    if isinstance(cfg, BaodingShadowLitePadTacBTCfg):
        return BaodingShadowLitePadTacBTEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    if isinstance(cfg, BaodingShadowLitePadTacCfg):
        return BaodingShadowLitePadTacEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    if isinstance(cfg, BaodingShadowLiteCfg):
        return BaodingShadowLiteEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    return BaodingShadowEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="Baoding",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingCfg,
        **_variant_paths("shadow", _SHADOW_VARIANT_FILES),
    },
)

# Register robot-specific variants
gym.register(
    id="Baoding_Shadowlite",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLiteCfg,
        **_variant_paths("shadowlite", _SHADOW_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTac",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Sparse",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTSparseCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Stuck8",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTStuck8Cfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Legacy",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTLegacyCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Legacy_NoSlew",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTLegacyNoSlewCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Legacy_NoMassDR",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTLegacyNoMassDRCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Legacy_NoTac",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTLegacyNoTacCorruptCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

gym.register(
    id="Baoding_Shadowlite_PadTacBT_Legacy_FrictionMass",
    entry_point=baoding_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaodingShadowLitePadTacBTLegacyFrictionMassOnlyCfg,
        **_variant_paths("shadowlite", _SHADOWLITE_PADTAC_VARIANT_FILES),
    },
)

