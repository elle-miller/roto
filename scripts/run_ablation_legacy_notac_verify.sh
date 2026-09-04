#!/usr/bin/env bash
# Ablation video pass for the two legacy re-training checkpoints: legacy_notac and
# legacy_verify. Same recipe as the earlier noslew3 / nomassdr passes, which follow
# run_ablation_videos.sh: 4 envs so the motion is watchable, one full episode per
# condition, the 14-condition block-masking / no-ball / ball-mass grid, run once with
# tactile active and once with --zero_tactile (the _TACZERO folder).
#
#   legacy_notac  -> robot shadowlite_padtac_bt_legacy_notac
#                    (BaodingShadowLitePadTacBTLegacyNoTacCorruptCfg: legacy profile,
#                     FSR corruption disabled -- perfect tactile)
#   legacy_verify -> robot shadowlite_padtac_bt_legacy
#                    (BaodingShadowLitePadTacBTLegacyCfg: the HW-validated profile)
#
# Clean-tactile contract from run_ablation_videos.sh kept: --fsr_corrupt_max 0
# --tactile_flip_prob 0, so the tactile-active arm sees no per-episode taxel
# corruption and no per-step dither.
#
# Clips land in scripts/ablation_<tag>/ (active) and scripts/ablation_<tag>_TACZERO/
# (--zero_tactile). build_video_tag() does not encode the checkpoint, so the folder
# name is the provenance.
#
# GPU 0 on purpose: shared with the still-running legacy_notac sweep.py. 4-env video
# runs fit in the free VRAM, but two Isaac Sim instances on one GPU raise the
# startup-hang rate -- hence ATTEMPTS retries per condition.

set -u

ROTO=/home/ayush/icra/roto
PY=/media/storage/ayush/miniconda3/envs/s2r/bin/python
GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-4}
EPISODES=${EPISODES:-1}
RUN_TIMEOUT=${RUN_TIMEOUT:-1200}
ATTEMPTS=${ATTEMPTS:-3}

# Snapshotted copies (the live training run renamed best_agent.pt -> best_agent_legacy_*.pt
# mid-pass on 2026-09-03, which broke the first attempt from cond 13 on; copy decouples us).
CKPT_NOTAC="$ROTO/scripts/checkpoint_snapshots/legacy_notac_ablation_20260903-123532.pt"
CKPT_VERIFY="$ROTO/scripts/checkpoint_snapshots/legacy_verify_ablation_20260903-123532.pt"

RUNS=(
  "notac|shadowlite_padtac_bt_legacy_notac|$CKPT_NOTAC"
  "verify|shadowlite_padtac_bt_legacy|$CKPT_VERIFY"
)

# ctag | filename-prefix build_video_tag() emits (up to and including "_seed") | extra args
# The prefix lets the resume check target this exact condition rather than any ablate-* file.
CONDS=(
  "none|ablate-none_zero_seed|"
  "pos_zero|ablate-pos_zero_seed|--ablate pos"
  "vel_zero|ablate-vel_zero_seed|--ablate vel"
  "vel_zero_noball|ablate-vel_zero_noball_seed|--ablate vel --no_ball"
  "pos_error_zero|ablate-pos_error_zero_seed|--ablate pos_error"
  "pos_error_zero_noball|ablate-pos_error_zero_noball_seed|--ablate pos_error --no_ball"
  "pos_error_freeze|ablate-pos_error_freeze_seed|--ablate pos_error --ablate_mode freeze"
  "prev_action_zero|ablate-prev_action_zero_seed|--ablate prev_action"
  "prev_action_zero_noball|ablate-prev_action_zero_noball_seed|--ablate prev_action --no_ball"
  "none_noball|ablate-none_zero_noball_seed|--no_ball"
  "none_mass45|ablate-none_zero_mass45g_seed|--ball_mass_g 45"
  "none_mass55|ablate-none_zero_mass55g_seed|--ball_mass_g 55"
  "none_mass70|ablate-none_zero_mass70g_seed|--ball_mass_g 70"
  "none_mass100|ablate-none_zero_mass100g_seed|--ball_mass_g 100"
)

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
    vsuffix=""
    prefsuffix="_seed"
    if [ "$tacmode" = "taczero" ]; then
      tacflag="--zero_tactile"
      vsuffix="_TACZERO"
      prefsuffix="_taczero_seed"
    fi
    vdir="$ROTO/scripts/ablation_${tag}${vsuffix}"
    mkdir -p "$vdir"
    for cond in "${CONDS[@]}"; do
      IFS='|' read -r ctag cpref cargs <<< "$cond"
      i=$((i + 1))
      # build_video_tag() emits <cpref-with-_taczero-inserted>seed... for the taczero pass
      match="${cpref%_seed}${prefsuffix}"
      log="$vdir/${ctag}.log"
      if ls "$vdir"/"${match}"* >/dev/null 2>&1; then
        echo "[$i/$total] skip (have clip): ${tag}/${tacmode}/${ctag}"
        continue
      fi
      echo "[$i/$total] $(date +%H:%M:%S) video: ${tag}/${tacmode}/${ctag}"
      rc=1
      for attempt in $(seq 1 "$ATTEMPTS"); do
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROTO PYTHONUNBUFFERED=1 \
          timeout -k 20 "$RUN_TIMEOUT" "$PY" -u ablate_play_tac.py \
            --task Baoding --robot "$robot" --agent_cfg rl_only_pt_padtac_bt \
            --checkpoint "$ckpt" \
            --num_envs "$NUM_ENVS" --episodes "$EPISODES" --headless \
            --fsr_corrupt_max 0 --tactile_flip_prob 0 --out_dir "$vdir" \
            --video_dir "./videos_ablleg_${tag}_${tacmode}/" \
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

echo "ablation video pass finished in $(( ($(date +%s) - started) / 60 )) min"
echo "  legacy_notac  -> $ROTO/scripts/ablation_notac{,_TACZERO}/"
echo "  legacy_verify -> $ROTO/scripts/ablation_verify{,_TACZERO}/"
touch "$ROTO/scripts/_ABLATION_LEGACY_NOTAC_VERIFY_DONE"
