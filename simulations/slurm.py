from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from simulations.io_utils import count_jsonl_rows, ensure_dir


@dataclass
class SlurmConfig:
    account: str
    gpus: str = "rtx_4090:1"
    time: str = "04:00:00"
    array_concurrency: int = 16
    cpus_per_task: int = 4
    mem: str = "24G"
    partition: Optional[str] = None
    constraint: Optional[str] = None


def _sbatch_header(
    job_name: str,
    slurm: SlurmConfig,
    array_high: int,
    output_path: Path,
    error_path: Path,
) -> List[str]:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -A {slurm.account}",
        f"#SBATCH --gpus={slurm.gpus}",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --array=0-{array_high}%{slurm.array_concurrency}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH -p {slurm.partition}")
    if slurm.constraint:
        lines.append(f"#SBATCH --constraint={slurm.constraint}")
    return lines


def render_sbatch_script(
    *,
    job_name: str,
    manifest_path: Path,
    config_path: Path,
    slurm: SlurmConfig,
) -> str:
    trial_count = count_jsonl_rows(manifest_path)
    if trial_count <= 0:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    logs_dir = manifest_path.parent.parent / "slurm" / "logs"
    out_path = logs_dir / f"{job_name}_%A_%a.out"
    err_path = logs_dir / f"{job_name}_%A_%a.err"
    header = _sbatch_header(
        job_name=job_name,
        slurm=slurm,
        array_high=trial_count - 1,
        output_path=out_path,
        error_path=err_path,
    )
    lines = header + [
        "",
        "set -euo pipefail",
        f"mkdir -p \"{logs_dir}\"",
        "",
        "echo \"Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID starting on $(hostname)\"",
        f"python -m simulations.main run-trial --config \"{config_path}\" --manifest \"{manifest_path}\" --index \"$SLURM_ARRAY_TASK_ID\"",
    ]
    return "\n".join(lines) + "\n"


def write_sbatch_script(
    *,
    output_path: Path,
    job_name: str,
    manifest_path: Path,
    config_path: Path,
    slurm: SlurmConfig,
) -> Path:
    script = render_sbatch_script(
        job_name=job_name,
        manifest_path=manifest_path,
        config_path=config_path,
        slurm=slurm,
    )
    ensure_dir(output_path.parent)
    output_path.write_text(script, encoding="utf-8")
    return output_path


def write_submit_all_script(script_paths: List[Path], destination: Path) -> Path:
    ensure_dir(destination.parent)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for script in script_paths:
        lines.append(f"sbatch \"{script}\"")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
