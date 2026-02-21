# LoRA Meta-Tuning Pipeline

## Components
- `ds_downloader.py`: downloads and caches raw Hugging Face datasets.
- `ds_loader.py`: normalizes data into unified JSONL train/val files and computes metafeatures.
- `training.py`: runs one trial (simulated by default, optional external backend).
- `optimizer.py`: coarse-grid generation + transfer-aware Bayesian suggestions.
- `slurm.py`: sbatch generation from manifests.
- `main.py`: CLI orchestrator.

## Typical Flow
```bash
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml prefetch
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml prepare-nanogpt-data
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml build-stage1
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml make-slurm --stage stage1
```

After stage-1 jobs complete:
```bash
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml build-stage2
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml make-slurm --stage stage2
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml build-reruns
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml make-slurm --stage rerun
python -m simulations.main --config simulations/configs/lora_metatune_v1.yaml export-csv
```

`export-csv` writes `outputs/history/runs.csv` by default.
`prepare-nanogpt-data` writes `outputs/history/nanogpt_datasets.csv` by default.
You can cap dataset size in YAML with `max_total_bytes` per dataset (e.g. `209715200` for ~200 MB).

## Euler Slurm
```bash
# Prep only (prefetch + nanoGPT train.bin/val.bin + CSV export)
sbatch -A <share_name> simulations/euler_scripts/run_prepare_nanogpt_data.sbatch

# Full sweep launcher
sbatch -A <share_name> simulations/euler_scripts/run_hyperparam_sweep.sbatch
```

## Integrating a Real NanoGPT-LoRA Trainer
Set:
- `training.mode: external`
- `training.external_command: "<your command template>"`

Command template variables:
- `{run_id}`, `{dataset_id}`, `{train_path}`, `{val_path}`
- `{token_budget}`, `{seq_len}`
- `{hp_json}`, `{metrics_path}`, `{curve_path}`

External command must write metrics JSON to `{metrics_path}` with:
- `val_loss`
- `best_val_loss` (optional)
- `gpu_hours` (optional)
- `peak_mem_mb` (optional)
