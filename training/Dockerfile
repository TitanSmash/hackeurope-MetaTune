# ---- Base: Red Hat UBI Python image (avoids Docker Hub pull limits) ----
FROM registry.access.redhat.com/ubi9/python-311:latest

# OpenShift runs containers with a random UID; avoid assuming user=1000, etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/opt/app-root \
    APP_HOME=/opt/app-root/src

WORKDIR ${APP_HOME}

# Upgrade pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

COPY . ${APP_HOME}

# ---- Python deps ----
# NOTE: torch install uses PyTorch's CUDA index URL for cu121 wheels.
RUN set -eux; \
    req_file=""; \
    for f in \
      "${APP_HOME}/trainer_requirements.txt" \
      "${APP_HOME}/training/trainer_requirements.txt" \
      "${APP_HOME}/requirements.txt"; do \
      if [ -s "$f" ]; then \
        req_file="$f"; \
        break; \
      fi; \
    done; \
    if [ -z "$req_file" ]; then \
      echo "No non-empty requirements file found." >&2; \
      exit 1; \
    fi; \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r "$req_file"

# ---- OpenShift runtime dirs ----
# UBI Python base is already OpenShift-friendly; just ensure runtime dirs exist.
RUN mkdir -p ${HOME} ${APP_HOME} /tmp

# Optional: default envs useful for HF + caching (you can override in the Job YAML)
ENV HF_HOME=/opt/app-root/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/app-root/.cache/huggingface/datasets \
    TORCH_HOME=/opt/app-root/.cache/torch

# Default command: prints GPU visibility; your actual Job will override command/args
CMD ["python3", "-c", "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"]
