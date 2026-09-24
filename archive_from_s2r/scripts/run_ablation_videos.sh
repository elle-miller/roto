#!/usr/bin/env bash
# Video pass for the ablation grid: one full-episode clip per condition.
#
# Separate from run_ablation_grid.sh on purpose. That pass runs 256 envs with --no_video
# for statistics; a 256-env recording is an unreadable wall of hands. Video runs use 4
# envs so the motion is actually watchable, which makes them useless for metrics -- the
# two passes answer different questions and neither substitutes for the other.
#
# Same clean-tactile contract as the metrics pass: --fsr_corrupt_max 0 --tactile_flip_prob 0,
# so the with-tactile arm sees no corruption and no 0.9/0.25 dropout.
#
# Waits for the metrics grid to finish before starting (set WAIT=0 to skip), because two
# concurrent Isaac Sim instances on one GPU raise the startup-hang rate.
#
# Videos land in $OUT/<run>_<tacmode>/, tagged by condition. build_video_tag() does not
# encode which checkpoint produced a clip, so the per-run subfolder IS the provenance --
# without it the 0p9 and 0p25 clips differ only by timestamp.

set -u

ROTO=/home/ayush/icra/roto
PY=/media/storage/ayush/miniconda3/envs/s2r/bin/python
OUT=${OUT:-$ROTO/scripts/ablation_videos_perfect_tactile}
METRICS_DIR=${METRICS_DIR:-$ROTO/scripts/ablation_perfect_tactile}
GPU=${GPU:-1}
NUM_ENVS=${NUM_ENVS:-4}
EPISODES=${EPISODES:-1}
SEED=${SEED:-42}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}   # video render is slower than headless metrics
ATTEMPTS=${ATTEMPTS:-3}
WAIT=${WAIT:-1}

CKPT_0p9="$ROTO/scripts/logs/shadowlite_baoding/Baoding_rl_only_pt_padtac_bt_pad_tac_2_0p9_flip/2026-08-25_12-50-51/checkpoints/best_agent.pt"
CKPT_0p25="$ROTO/scripts/logs/shadowlite_baoding/Baoding_rl_only_pt_padtac_bt_sparse_corrupt_k6_flip0p25/2026-08-25_22-57-17/checkpoints/best_agent.pt"

RUNS=(
  "0p9|shadowlite_padtac_bt|$CKPT_0p9"
  "0p25|shadowlite_padtac_bt_sparse|$CKPT_0p25"
)

# Same 15 conditions as the metrics pass, minus --log_traj (the clip is the artefact here).
CONDS=(
  "none|"
  "pos_zero|--ablate pos"
  "vel_zero|--ablate vel"
  "vel_zero_noball|--ablate vel --no_ball"
  "pos_error_zero|--ablate pos_error"
  "pos_error_zero_noball|--ablate pos_error --no_ball"
  "pos_error_freeze|--ablate pos_error --ablate_mode freeze"
  "prev_action_zero|--ablate prev_action"
  "prev_action_zero_noball|--ablate prev_action --no_ball"
  "none_noball|--no_ball"
  "none_mass45|--ball_mass_g 45"
  "none_mass55|--ball_mass_g 55"
  "none_mass70|--ball_mass_g 70"
  "none_mass100|--ball_mass_g 100"
)

if [ "$WAIT" = "1" ]; then
  echo "waiting for metrics grid ($METRICS_DIR/_GRID_DONE) ..."
  while [ ! -f "$METRICS_DIR/_GRID_DONE" ]; do sleep 30; done
  echo "metrics grid done, starting video pass at $(date +%H:%M:%S)"
fi

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
    vdir="$OUT/${tag}_${tacmode}"
    mkdir -p "$vdir"
    for cond in "${CONDS[@]}"; do
      IFS='|' read -r ctag cargs <<< "$cond"
      i=$((i + 1))
      log="$vdir/${ctag}.log"
      # Resume on the artefact itself: a .mp4 whose name starts with this condition's
      # tag means the clip exists, so re-running would only add a duplicate.
      if ls "$vdir"/ablate-* 2>/dev/null | grep -q .; then
        if grep -q "Saved video" "$log" 2>/dev/null; then
          echo "[$i/$total] skip (have clip): ${tag}/${tacmode}/${ctag}"
          continue
        fi
      fi
      echo "[$i/$total] $(date +%H:%M:%S) video: ${tag}/${tacmode}/${ctag}"
      rc=1
      for attempt in $(seq 1 "$ATTEMPTS"); do
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROTO PYTHONUNBUFFERED=1 \
          timeout -k 20 "$RUN_TIMEOUT" "$PY" -u ablate_play_tac.py \
            --task Baoding --robot "$robot" --agent_cfg rl_only_pt_padtac_bt \
            --checkpoint "$ckpt" \
            --num_envs "$NUM_ENVS" --episodes "$EPISODES" --seed "$SEED" --headless \
            --fsr_corrupt_max 0 --tactile_flip_prob 0 --out_dir "$vdir" \
            $tacflag $cargs > "$log" 2>&1
        rc=$?
        if grep -q "Saved video" "$log" 2>/dev/null; then rc=0; break; fi
        echo "    attempt $attempt/$ATTEMPTS failed (exit $rc), retrying"
      done
      if [ $rc -ne 0 ]; then
        echo "    -> FAILED after $ATTEMPTS attempts (see $log)"
      else
        grep "Saved video" "$log" | tail -1 | sed 's/^/    -> /'
      fi
    done
  done
done

echo "video pass finished in $(( ($(date +%s) - started) / 60 )) min -> $OUT"
touch "$OUT/_VIDEOS_DONE"
