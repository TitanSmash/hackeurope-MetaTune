#!/bin/bash
#SBATCH --job-name=metatune_sweep
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:3
#SBATCH --output=logs/metatune_sweep_%j.out
#SBATCH --error=logs/metatune_sweep_%j.err

# ==============================================================================
# Two-Stage + Rerun Hyperparameter Sweep (single launcher job)
# ==============================================================================
# Usage:
#   Stage 1 only:
#     sbatch -A <share_name> simulations/euler_scripts/run_hyperparam_sweep.sbatch
#   Stage 2 only:
#     sbatch -A <share_name> --export=STAGE=2,CONFIG=simulations/configs/lora_metatune_v1.yaml simulations/euler_scripts/run_hyperparam_sweep.sbatch
#   Reruns only:
#     sbatch -A <share_name> --export=STAGE=3,CONFIG=simulations/configs/lora_metatune_v1.yaml simulations/euler_scripts/run_hyperparam_sweep.sbatch
#   Full chain in one job (1 -> 2 -> 3):
#     sbatch -A <share_name> --export=STAGE=full,CONFIG=simulations/configs/lora_metatune_v1.yaml simulations/euler_scripts/run_hyperparam_sweep.sbatch
#
# Optional parallelism controls:
#   --export=JOBS_PER_GPU=2   # default 2; total parallel jobs = NUM_GPUS * JOBS_PER_GPU
#   --export=NUM_GPUS=3       # override detected GPU count
#   --export=CSV_SYNC_EVERY=20
# ==============================================================================

set -euo pipefail

log() {
  local msg="$1"
  local ts
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] $msg"
}

# Project root on Euler
PROJECT_ROOT=${PROJECT_ROOT:-"/cluster/home/$USER/hackeurope-MetaTune"}
cd "$PROJECT_ROOT"

# Optional conda setup (keep these env vars as defaults but overridable).
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
CONDA_ENV=${CONDA_ENV:-"mrag"}
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || true
fi

CONFIG=${CONFIG:-"simulations/configs/lora_metatune_v1.yaml"}
STAGE=${STAGE:-"1"}
FORCE_PREFETCH=${FORCE_PREFETCH:-"0"}
JOBS_PER_GPU=${JOBS_PER_GPU:-"2"}
CSV_SYNC_EVERY=${CSV_SYNC_EVERY:-"20"}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/cluster/scratch/$USER/hackeurope-MetaTune"}
export SIM_OUTPUT_ROOT=${SIM_OUTPUT_ROOT:-"$SCRATCH_ROOT/outputs"}
DATASETS_CACHE_ROOT=${DATASETS_CACHE_ROOT:-"/cluster/scratch/$USER/datasets"}
export SIM_HF_CACHE_DIR=${SIM_HF_CACHE_DIR:-"$DATASETS_CACHE_ROOT"}
export SIM_RAW_ROOT=${SIM_RAW_ROOT:-"$SIM_OUTPUT_ROOT/raw_datasets"}
export SIM_PROCESSED_ROOT=${SIM_PROCESSED_ROOT:-"$SIM_OUTPUT_ROOT/datasets"}
OUTPUT_ROOT="$SIM_OUTPUT_ROOT"
MANIFEST_ROOT="$OUTPUT_ROOT/manifests"
DATASET_INDEX="$SIM_PROCESSED_ROOT/index.json"

# HF cache defaults (shared persistent location on Euler).
export HF_HOME=${HF_HOME:-"$SCRATCH_ROOT/huggingface_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DATASETS_CACHE_ROOT"}
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Training jobs should stay offline on compute nodes.
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p logs
RUNNER_LOG_DIR="$OUTPUT_ROOT/history/runner_logs"
mkdir -p "$RUNNER_LOG_DIR" "$SIM_OUTPUT_ROOT" "$SIM_PROCESSED_ROOT" "$SIM_RAW_ROOT"

NUM_GPUS_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
NUM_GPUS=${NUM_GPUS:-3}
if [ -n "$NUM_GPUS_DETECTED" ] && [ "$NUM_GPUS_DETECTED" -gt 0 ] && [ "$NUM_GPUS" -gt "$NUM_GPUS_DETECTED" ]; then
  NUM_GPUS=$NUM_GPUS_DETECTED
fi
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
  NUM_GPUS=1
fi
if [ "$JOBS_PER_GPU" -le 0 ]; then
  JOBS_PER_GPU=1
fi
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))
if [ "$MAX_PARALLEL" -le 0 ]; then
  MAX_PARALLEL=1
fi

echo "=============================================="
echo "LoRA METATUNE SWEEP - STAGE ${STAGE}"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Config:      $CONFIG"
echo "Project root:$PROJECT_ROOT"
echo "Scratch:     $SCRATCH_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Raw root:    $SIM_RAW_ROOT"
echo "Proc root:   $SIM_PROCESSED_ROOT"
echo "SIM cache:   $SIM_HF_CACHE_DIR"
echo "HF home:     $HF_HOME"
echo "Datasets:    $HF_DATASETS_CACHE"
echo "Models:      $TRANSFORMERS_CACHE"
echo "GPUs:        $NUM_GPUS"
echo "Jobs/GPU:    $JOBS_PER_GPU"
echo "Parallel:    $MAX_PARALLEL"
echo "Started:     $(date)"
echo "=============================================="

if [ "$FORCE_PREFETCH" = "1" ]; then
  log "[prefetch] forced"
  python -m simulations.main --config "$CONFIG" prefetch --force
else
  if [ ! -f "$DATASET_INDEX" ]; then
    log "[prefetch] $DATASET_INDEX not found, running prefetch"
    python -m simulations.main --config "$CONFIG" prefetch
  else
    log "[prefetch] already available, skipping"
  fi
fi

run_stage() {
  local stage_cmd=$1
  local manifest_file=$2
  local stage_name
  stage_name=$(basename "$manifest_file" .jsonl)
  local stage_log="$RUNNER_LOG_DIR/${stage_name}_runner.log"
  local trial_log_dir="$RUNNER_LOG_DIR/${stage_name}_trials"
  mkdir -p "$trial_log_dir"

  log "[stage] building manifest via: ${stage_cmd}" | tee -a "$stage_log"
  python -m simulations.main --config "$CONFIG" "$stage_cmd"

  if [ ! -f "$manifest_file" ]; then
    log "[stage] manifest not found: $manifest_file" | tee -a "$stage_log"
    return 1
  fi

  local total
  total=$(wc -l < "$manifest_file" | tr -d ' ')
  if [ "$total" -le 0 ]; then
    log "[stage] no trials in manifest: $manifest_file" | tee -a "$stage_log"
    return 0
  fi

  log "[stage] running manifest: $manifest_file (trials=$total)" | tee -a "$stage_log"

  declare -A PID_TO_IDX
  declare -A PID_TO_GPU
  declare -A GPU_RUNNING
  local next_idx=0
  local running=0
  local completed=0
  local failed=0

  for ((g=0; g<NUM_GPUS; g++)); do
    GPU_RUNNING[$g]=0
  done

  while [ "$completed" -lt "$total" ]; do
    # Launch as many as possible across GPU slots.
    while [ "$next_idx" -lt "$total" ] && [ "$running" -lt "$MAX_PARALLEL" ]; do
      local selected_gpu=-1
      for ((g=0; g<NUM_GPUS; g++)); do
        local in_use=${GPU_RUNNING[$g]:-0}
        if [ "$in_use" -lt "$JOBS_PER_GPU" ]; then
          selected_gpu=$g
          break
        fi
      done
      if [ "$selected_gpu" -lt 0 ]; then
        break
      fi

      local idx="$next_idx"
      local out_log="$trial_log_dir/trial_${idx}.out"
      local err_log="$trial_log_dir/trial_${idx}.err"
      (
        export CUDA_VISIBLE_DEVICES="$selected_gpu"
        python -m simulations.main --config "$CONFIG" run-trial --manifest "$manifest_file" --index "$idx"
      ) >"$out_log" 2>"$err_log" &

      local pid=$!
      PID_TO_IDX[$pid]=$idx
      PID_TO_GPU[$pid]=$selected_gpu
      GPU_RUNNING[$selected_gpu]=$(( ${GPU_RUNNING[$selected_gpu]} + 1 ))
      running=$((running + 1))
      next_idx=$((next_idx + 1))

      log "[stage] launched idx=$idx gpu=$selected_gpu pid=$pid running=$running/$MAX_PARALLEL" | tee -a "$stage_log"
    done

    # Collect finished jobs.
    local collected=0
    for pid in "${!PID_TO_IDX[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        local idx=${PID_TO_IDX[$pid]}
        local gpu=${PID_TO_GPU[$pid]}
        if wait "$pid"; then
          log "[stage] finished idx=$idx gpu=$gpu status=ok" | tee -a "$stage_log"
        else
          failed=$((failed + 1))
          log "[stage] finished idx=$idx gpu=$gpu status=failed (see $trial_log_dir/trial_${idx}.err)" | tee -a "$stage_log"
        fi
        unset PID_TO_IDX[$pid]
        unset PID_TO_GPU[$pid]
        GPU_RUNNING[$gpu]=$(( ${GPU_RUNNING[$gpu]} - 1 ))
        running=$((running - 1))
        completed=$((completed + 1))
        collected=1

        local pct=$((100 * completed / total))
        log "[stage] progress ${completed}/${total} (${pct}%), failed=$failed, active=$running" | tee -a "$stage_log"

        if [ "$CSV_SYNC_EVERY" -gt 0 ] && [ $((completed % CSV_SYNC_EVERY)) -eq 0 ]; then
          python -m simulations.main --config "$CONFIG" export-csv >/dev/null 2>&1 || true
        fi
      fi
    done

    if [ "$collected" -eq 0 ]; then
      sleep 2
    fi
  done

  python -m simulations.main --config "$CONFIG" export-csv
  log "[stage] completed stage_name=$stage_name total=$total failed=$failed" | tee -a "$stage_log"
}

if [ "$STAGE" = "1" ]; then
  run_stage "build-stage1" "$MANIFEST_ROOT/stage1_all.jsonl"

elif [ "$STAGE" = "2" ]; then
  run_stage "build-stage2" "$MANIFEST_ROOT/stage2_all.jsonl"

elif [ "$STAGE" = "3" ]; then
  run_stage "build-reruns" "$MANIFEST_ROOT/rerun_all.jsonl"

elif [ "$STAGE" = "full" ]; then
  run_stage "build-stage1" "$MANIFEST_ROOT/stage1_all.jsonl"
  run_stage "build-stage2" "$MANIFEST_ROOT/stage2_all.jsonl"
  run_stage "build-reruns" "$MANIFEST_ROOT/rerun_all.jsonl"

else
  echo "ERROR: Invalid STAGE=$STAGE. Use 1,2,3,full."
  exit 1
fi

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo "Completed at: $(date)"
echo "=============================================="
