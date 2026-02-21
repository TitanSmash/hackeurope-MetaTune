#!/bin/bash
# Download HF datasets/models on Euler login node (internet-enabled).
# Run this once before submitting offline compute jobs.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/cluster/home/$USER/hackeurope-MetaTune"}
CONFIG=${CONFIG:-"simulations/configs/lora_metatune_v1.yaml"}
CONDA_ENV=${CONDA_ENV:-"mrag"}
MODEL_ID=${MODEL_ID:-""}
SCRATCH_ROOT=${SCRATCH_ROOT:-"/cluster/scratch/$USER/hackeurope-MetaTune"}
export SIM_OUTPUT_ROOT=${SIM_OUTPUT_ROOT:-"$SCRATCH_ROOT/outputs"}
DATASETS_CACHE_ROOT=${DATASETS_CACHE_ROOT:-"/cluster/scratch/$USER/datasets"}
export SIM_HF_CACHE_DIR=${SIM_HF_CACHE_DIR:-"$DATASETS_CACHE_ROOT"}
export SIM_RAW_ROOT=${SIM_RAW_ROOT:-"$SIM_OUTPUT_ROOT/raw_datasets"}
export SIM_PROCESSED_ROOT=${SIM_PROCESSED_ROOT:-"$SIM_OUTPUT_ROOT/datasets"}

cd "$PROJECT_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || true
fi

export HF_HOME=${HF_HOME:-"$SCRATCH_ROOT/huggingface_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DATASETS_CACHE_ROOT"}
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$SIM_OUTPUT_ROOT" "$SIM_RAW_ROOT" "$SIM_PROCESSED_ROOT"



# Ensure online mode during cache warmup.
unset TRANSFORMERS_OFFLINE || true
unset HF_DATASETS_OFFLINE || true

echo "=============================================="
echo "HF CACHE PREFETCH"
echo "=============================================="
echo "Project:   $PROJECT_ROOT"
echo "Config:    $CONFIG"
echo "Scratch:   $SCRATCH_ROOT"
echo "Outputs:   $SIM_OUTPUT_ROOT"
echo "Raw root:  $SIM_RAW_ROOT"
echo "Proc root: $SIM_PROCESSED_ROOT"
echo "SIM cache: $SIM_HF_CACHE_DIR"
echo "HF_HOME:   $HF_HOME"
echo "Datasets:  $HF_DATASETS_CACHE"
echo "Models:    $TRANSFORMERS_CACHE"
echo "Started:   $(date)"
echo "=============================================="

python -m simulations.main --config "$CONFIG" prefetch
python -m simulations.main --config "$CONFIG" prepare-nanogpt-data

if [ -n "$MODEL_ID" ]; then
  python - << EOF
from transformers import AutoModel, AutoTokenizer
model_id = "${MODEL_ID}"
print(f"Downloading model assets for: {model_id}")
AutoTokenizer.from_pretrained(model_id)
AutoModel.from_pretrained(model_id)
print("Model + tokenizer cached.")
EOF
fi

echo ""
echo "Prefetch complete."
echo "For compute jobs, set:"
echo "  export TRANSFORMERS_OFFLINE=1"
echo "  export HF_DATASETS_OFFLINE=1"
