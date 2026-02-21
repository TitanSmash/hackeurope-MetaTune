#!/bin/bash
set -euo pipefail

# Optional overrides:
#   LOCAL_DIR=/path/to/repo REMOTE_USER=myuser REMOTE_DIR=~/hackeurope-MetaTune ./simulations/euler_scripts/copy_to_euler.sh
LOCAL_DIR="${LOCAL_DIR:-$(pwd)}"
REMOTE_USER="ssutanto"
REMOTE_HOST="${REMOTE_HOST:-euler.ethz.ch}"
REMOTE_DIR="${REMOTE_DIR:-~/hackeurope-MetaTune/}"

python simulations/euler_scripts/copy_to_euler.py \
  --local-dir "$LOCAL_DIR" \
  --remote-user "$REMOTE_USER" \
  --remote-host "$REMOTE_HOST" \
  --remote-dir "$REMOTE_DIR" \
  --exclude-output
