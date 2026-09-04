#!/usr/bin/env bash
# Ablation grid for the two tactile-DR training runs, evaluated under PERFECT tactile.
#
# The two runs differ only in how their tactile DR corrupts the FSR pads:
#   0p9    BaodingShadowLitePadTacBTCfg        k~U{0..8} stuck, flip 0.1/0.1 scope=corrupted
#                                              (a stuck channel holds its forced value ~90%
#                                              of steps -- the "0.9" in the run's name)
#   0p25   BaodingShadowLitePadTacBTSparseCfg  k~U{0..6} fully stuck, flip 0.25/0.25
#                                              scope=all_fsr (the 12-k UNSELECTED pads)
#
# Both DRs are switched OFF here (--fsr_corrupt_max 0 --tactile_flip_prob 0) so every
# condition is measured against clean taxels. Both flags are required: on the 0p9 profile
# dropping only the corrupt draw RAISES (its dither is scoped to that draw), and on the
# 0p25 profile it would leave the 0.25 flip firing on all 12 pads -- dirtier, not cleaner.
#
# Each checkpoint is run twice: tactile active, and --zero_tactile (prop-only), which is
# the "with and without tactile" half of the comparison.
#
# Resumable: a condition whose log already contains "===== SUMMARY =====" is skipped.

set -u

ROTO=/home/ayush/icra/roto
PY=/media/storage/ayush/miniconda3/envs/s2r/bin/python
OUT=${OUT:-$ROTO/scripts/ablation_perfect_tactile}
GPU=${GPU:-1}
NUM_ENVS=${NUM_ENVS:-256}
EPISODES=${EPISODES:-3}
SEED=${SEED:-42}
# Real runs finish in 50-150s; anything past this is the startup hang, not slow work.
RUN_TIMEOUT=${RUN_TIMEOUT:-420}
ATTEMPTS=${ATTEMPTS:-3}

CKPT_0p9="$ROTO/scripts/logs/shadowlite_baoding/Baoding_rl_only_pt_padtac_bt_pad_tac_2_0p9_flip/2026-08-25_12-50-51/checkpoints/best_agent.pt"
CKPT_0p25="$ROTO/scripts/logs/shadowlite_baoding/Baoding_rl_only_pt_padtac_bt_sparse_corrupt_k6_flip0p25/2026-08-25_22-57-17/checkpoints/best_agent.pt"

# run_tag|robot|checkpoint
RUNS=(
  "0p9|shadowlite_padtac_bt|$CKPT_0p9"
  "0p25|shadowlite_padtac_bt_sparse|$CKPT_0p25"
)

# cond_tag|extra ablate_play_tac.py args
# Mirrors the 14 conditions in the existing ablation/ video sets.
#
# --no_ball conditions get --log_traj: with the balls parked there is no reward and no
# rotation counter movement, so their return/rotations/drop_rate are all-zero BY
# CONSTRUCTION and say nothing. The question a no-ball run actually answers -- does the
# hand still cycle when nothing is in it, i.e. open-loop, or does the motion collapse --
# lives in the joint trajectory, so it has to be written out to be compared at all.
CONDS=(
  "none|"
  "pos_zero|--ablate pos"
  "vel_zero|--ablate vel"
  "vel_zero_noball|--ablate vel --no_ball --log_traj TRAJ"
  "pos_error_zero|--ablate pos_error"
  "pos_error_zero_noball|--ablate pos_error --no_ball --log_traj TRAJ"
  "pos_error_freeze|--ablate pos_error --ablate_mode freeze"
  "prev_action_zero|--ablate prev_action"
  "prev_action_zero_noball|--ablate prev_action --no_ball --log_traj TRAJ"
  "none_noball|--no_ball --log_traj TRAJ"
  "none_mass45|--ball_mass_g 45"
  "none_mass55|--ball_mass_g 55"
  "none_mass70|--ball_mass_g 70"
  "none_mass100|--ball_mass_g 100"
)

# A baseline WITH the ball, logged the same way, so each no-ball trajectory has a
# with-ball counterpart to be compared against -- the comparison is the point.
CONDS+=("none_traj|--log_traj TRAJ")

mkdir -p "$OUT"
cd "$ROTO/scripts" || exit 1

total=$(( ${#RUNS[@]} * 2 * ${#CONDS[@]} ))
i=0
started=$(date +%s)

for run in "${RUNS[@]}"; do
  IFS='|' read -r tag robot ckpt <<< "$run"
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP RUN] checkpoint missing: $ckpt"
    continue
  fi
  for tacmode in active taczero; do
    tacflag=""
    [ "$tacmode" = "taczero" ] && tacflag="--zero_tactile"
    for cond in "${CONDS[@]}"; do
      IFS='|' read -r ctag cargs <<< "$cond"
      i=$((i + 1))
      log="$OUT/${tag}_${tacmode}_${ctag}.log"
      if grep -q "===== SUMMARY =====" "$log" 2>/dev/null; then
        echo "[$i/$total] skip (done): ${tag}/${tacmode}/${ctag}"
        continue
      fi
      echo "[$i/$total] $(date +%H:%M:%S) run: ${tag}/${tacmode}/${ctag}"
      # TRAJ is a placeholder in CONDS so each run gets its own uniquely-named npz.
      cargs=${cargs//TRAJ/$OUT/traj_${tag}_${tacmode}_${ctag}.npz}

      # Isaac Sim fails at startup maybe 1 run in 4, two ways, both ~10-20s in:
      #   * malloc(): invalid size (unsorted)  -> SIGABRT, exit 134
      #   * spins at "Starting the simulation" -> never returns, ~42 cores burnt
      # Neither is caused by anything this grid does and both clear on a retry, so
      # each condition gets ATTEMPTS tries. The timeout is what makes the hang
      # survivable: -k is REQUIRED, since Isaac Sim ignores plain SIGTERM (a bare
      # `timeout 500` on this same script left a process spinning for six days).
      # RUN_TIMEOUT is sized off real runtimes (50-150s), not the hang.
      rc=1
      for attempt in $(seq 1 "$ATTEMPTS"); do
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROTO PYTHONUNBUFFERED=1 \
          timeout -k 20 "$RUN_TIMEOUT" "$PY" -u ablate_play_tac.py \
            --task Baoding --robot "$robot" --agent_cfg rl_only_pt_padtac_bt \
            --checkpoint "$ckpt" \
            --no_video --num_envs "$NUM_ENVS" --episodes "$EPISODES" --seed "$SEED" --headless \
            --fsr_corrupt_max 0 --tactile_flip_prob 0 \
            $tacflag $cargs > "$log" 2>&1
        rc=$?
        # Treat a written SUMMARY as success even on a nonzero rc: Isaac Sim sometimes
        # faults during its own teardown, after every metric has already been printed.
        if [ $rc -eq 0 ] || grep -q "===== SUMMARY =====" "$log" 2>/dev/null; then
          rc=0
          break
        fi
        echo "    attempt $attempt/$ATTEMPTS failed (exit $rc), retrying"
        cp "$log" "$log.attempt$attempt" 2>/dev/null
      done

      if [ $rc -ne 0 ]; then
        echo "    -> FAILED after $ATTEMPTS attempts (see $log)"
      else
        grep -E "^mean_num_rotations|^mean_return|^drop_rate" "$log" | tr '\n' ' ' | sed 's/^/    -> /'
        echo
      fi
    done
  done
done

echo "grid finished in $(( ($(date +%s) - started) / 60 )) min -> $OUT"
touch "$OUT/_GRID_DONE"
