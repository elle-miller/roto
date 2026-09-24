# archive_from_s2r

Files taken off the `s2r` branch while finalising it (2026-09-24). Kept here
so nothing is lost. Paths mirror their original location on `s2r` at tag
`pre-cleanup-s2r` (9b3bcc8).

- `genan/`, `scripts/fit_j1_gains.py`, `scripts/play_qcmd_residual.py`:
  position-residual actuator-net attempt; needs `roto.tasks.uan_shadowlite`.
- `scripts/ablation/`, `scripts/tactile_use/`, `scripts/run_ablation_*.sh`,
  `scripts/collect_ablation_grid.py`, `scripts/play_no_{corrupt,flip}.py`:
  ablation tooling.
- `run_RL.py`, `run_shadow.py`, `my_policy_node.py`, `rospy_rl.md`: old ROS
  policy nodes, superseded by `deploy/` on `s2r`.
- `roto/assets/shadow_lite_old/`: superseded Shadow Lite asset set.
- `*.db`, `scripts/baoding_recording*`, logs: Optuna sweeps and run outputs.
