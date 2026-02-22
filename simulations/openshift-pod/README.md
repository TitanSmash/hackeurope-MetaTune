# OpenShift LoRA Training App

This folder contains a standalone OpenShift system for LoRA fine-tuning with:
1. One always-on scheduler API pod.
2. One active GPU training `Job` pod at a time.
3. SQLite state on a shared PVC.
4. nanoGPT-LoRA training jobs with callback-based completion.

## Components

- `app/`: FastAPI scheduler + dispatcher + SQLite store + Kubernetes job launcher.
- `trainer/runner.py`: training pod entrypoint that runs nanoGPT-LoRA.
- `Dockerfile.scheduler`: scheduler image build.
- `Dockerfile.trainer`: trainer image build.
- `manifests/`: OpenShift resources (`ServiceAccount`, `Role`, `RoleBinding`, `PVC`, `Deployment`, `Service`, `Route`).

## Shared PVC layout

Mounted at `/data` in both scheduler and trainer pods:

- `/data/datasets/<dataset_rel_path>/train.bin`
- `/data/datasets/<dataset_rel_path>/val.bin`
- `/data/db/scheduler.sqlite`
- `/data/outputs/<run_id>/metrics.json`
- `/data/outputs/<run_id>/curve.jsonl`
- `/data/outputs/<run_id>/train.log`
- `/data/outputs/<run_id>/run_meta.json`

## Build images

Run from repository root:

```bash
docker build -f simulations/openshift-pod/Dockerfile.scheduler -t <scheduler-image> .
docker build -f simulations/openshift-pod/Dockerfile.trainer -t <trainer-image> .
```

Push both images to your OpenShift image registry and set `SCHEDULER_IMAGE` and `TRAINER_IMAGE` before applying manifests.

## PowerShell automation scripts

- End-to-end deploy + submit + monitor:
`simulations/openshift-pod/scripts/deploy_and_run.ps1`
- Submit/monitor only (after deployment):
`simulations/openshift-pod/scripts/submit_run.ps1`
- Upload `tiny_stories_copy` folder into PVC dataset path:
`simulations/openshift-pod/scripts/upload_tiny_stories_copy_to_pvc.ps1`
- Build images directly in OpenShift (no local docker push needed):
`simulations/openshift-pod/scripts/build_images_in_openshift.ps1`

Example:

```powershell
.\simulations\openshift-pod\scripts\deploy_and_run.ps1 `
  -Server "https://api.<cluster>:6443" `
  -Token "<api-key>" `
  -SchedulerImage "quay.io/<you>/metatune-scheduler:latest" `
  -TrainerImage "quay.io/<you>/metatune-trainer:latest" `
  -LocalTrainBin "C:\data\tinystories\train.bin" `
  -LocalValBin "C:\data\tinystories\val.bin" `
  -DatasetName "tinystories"
```

Upload local `tiny_stories_copy` into `/data/datasets/tinystories` on PVC:

```powershell
.\simulations\openshift-pod\scripts\upload_tiny_stories_copy_to_pvc.ps1 `
  -Namespace "hack-europe-team-i" `
  -SourceFolder "tiny_stories_copy" `
  -DatasetName "tinystories"
```

Build both images in OpenShift:

```powershell
.\simulations\openshift-pod\scripts\build_images_in_openshift.ps1
```

## Deploy

Follow `simulations/openshift-pod/manifests/README.md`.

## Scheduler API

### Create run

```bash
curl -X POST "https://<route-host>/api/v1/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "dataset_rel_path": "tinystories",
    "hp": {
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05,
      "learning_rate": 0.0003,
      "batch_size": 8
    },
    "token_budget": 25000000,
    "seq_len": 1024,
    "seed": 1
  }'
```

### Get run

```bash
curl "https://<route-host>/api/v1/runs/<run_id>"
```

### List runs

```bash
curl "https://<route-host>/api/v1/runs?limit=50"
```

## Notes

- Dispatch policy is fixed to one active training pod at a time.
- Callback endpoint is internal and unauthenticated: `/internal/runs/{run_id}/complete`.
- Existing `backend/*` APIs are not modified by this implementation.
