#!/bin/bash
#SBATCH --job-name=metatune_prep
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/metatune_prep_%j.out
#SBATCH --error=logs/metatune_prep_%j.err

# ==============================================================================
# Prepare datasets for nanoGPT-LoRA and export CSV manifests
# ==============================================================================
# Usage:
#   sbatch -A <share_name> simulations/euler_scripts/run_prepare_nanogpt_data.sbatch
# Optional overrides:
#   sbatch -A <share_name> --export=CONFIG=simulations/configs/lora_metatune_v1.yaml,FORCE_PREPARE=1 simulations/euler_scripts/run_prepare_nanogpt_data.sbatch
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/cluster/home/$USER/hackeurope-MetaTune"}
CONFIG=${CONFIG:-"simulations/configs/lora_metatune_v1.yaml"}
CONDA_ENV=${CONDA_ENV:-"mrag"}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/cluster/scratch/$USER/hackeurope-MetaTune"}
FORCE_PREFETCH=${FORCE_PREFETCH:-"0"}
FORCE_PREPARE=${FORCE_PREPARE:-"0"}
DATASETS_CACHE_ROOT=${DATASETS_CACHE_ROOT:-"/cluster/scratch/$USER/datasets"}

export SIM_OUTPUT_ROOT=${SIM_OUTPUT_ROOT:-"$SCRATCH_ROOT/outputs"}
export SIM_HF_CACHE_DIR=${SIM_HF_CACHE_DIR:-"$DATASETS_CACHE_ROOT"}
export SIM_RAW_ROOT=${SIM_RAW_ROOT:-"$SIM_OUTPUT_ROOT/raw_datasets"}
export SIM_PROCESSED_ROOT=${SIM_PROCESSED_ROOT:-"$SIM_OUTPUT_ROOT/datasets"}
export HF_HOME=${HF_HOME:-"$SCRATCH_ROOT/huggingface_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DATASETS_CACHE_ROOT"}
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$SIM_OUTPUT_ROOT" "$SIM_PROCESSED_ROOT" "$SIM_RAW_ROOT"
cd "$PROJECT_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || true
fi

DATASET_INDEX="$SIM_PROCESSED_ROOT/index.json"

echo "=============================================="
echo "PREPARE NANO-GPT DATASETS"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Project root:$PROJECT_ROOT"
echo "Config:      $CONFIG"
echo "Scratch:     $SCRATCH_ROOT"
echo "Output root: $SIM_OUTPUT_ROOT"
echo "Raw root:    $SIM_RAW_ROOT"
echo "Proc root:   $SIM_PROCESSED_ROOT"
echo "HF cache:    $SIM_HF_CACHE_DIR"
echo "Started:     $(date)"
echo "=============================================="

if [ "$FORCE_PREFETCH" = "1" ]; then
  python -m simulations.main --config "$CONFIG" prefetch --force
elif [ ! -f "$DATASET_INDEX" ]; then
  echo "[prefetch] $DATASET_INDEX not found, running prefetch"
  python -m simulations.main --config "$CONFIG" prefetch
else
  echo "[prefetch] existing prepared datasets found, skipping"
fi

if [ "$FORCE_PREPARE" = "1" ]; then
  python -m simulations.main --config "$CONFIG" prepare-nanogpt-data --force
else
  python -m simulations.main --config "$CONFIG" prepare-nanogpt-data
fi

python -m simulations.main --config "$CONFIG" export-csv

echo ""
echo "Done."
echo "nanoGPT dataset CSV: $SIM_OUTPUT_ROOT/history/nanogpt_datasets.csv"
echo "run history CSV:     $SIM_OUTPUT_ROOT/history/runs.csv"
echo "Completed at:        $(date)"
