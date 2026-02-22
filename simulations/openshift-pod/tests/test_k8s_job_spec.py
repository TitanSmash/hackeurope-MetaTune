from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppConfig

try:
    from app.k8s_jobs import build_training_job_object

    K8S_AVAILABLE = True
except ModuleNotFoundError:
    K8S_AVAILABLE = False


def _test_config(tmpdir: Path) -> AppConfig:
    env = {
        "NAMESPACE": "hack-europe-team-i",
        "DATA_ROOT": str(tmpdir / "data"),
        "DATASETS_ROOT": str(tmpdir / "data" / "datasets"),
        "OUTPUT_ROOT": str(tmpdir / "data" / "outputs"),
        "DB_PATH": str(tmpdir / "data" / "db" / "scheduler.sqlite"),
        "TRAINER_IMAGE": "example/trainer:latest",
        "TRAINER_COMMAND_JSON": "[\"python\", \"simulations/openshift-pod/trainer/runner.py\"]",
    }
    with patch.dict(os.environ, env, clear=False):
        return AppConfig.from_env()


@unittest.skipUnless(K8S_AVAILABLE, "kubernetes package is not installed")
class JobSpecTests(unittest.TestCase):
    def test_job_spec_contains_gpu_pvc_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(Path(tmp))
            run = {
                "run_id": "run_123",
                "model": "gpt2",
                "dataset_rel_path": "tinystories",
                "hp_json": json.dumps(
                    {
                        "lora_r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.1,
                        "learning_rate": 3e-4,
                        "batch_size": 8,
                    }
                ),
                "token_budget": 1000,
                "seq_len": 128,
                "seed": 1,
                "job_name": "custom-name",
            }
            job = build_training_job_object(run, config)
            container = job.spec.template.spec.containers[0]
            self.assertEqual(
                container.resources.requests[config.gpu_resource_name],
                str(config.gpu_count),
            )
            self.assertEqual(
                container.resources.limits[config.gpu_resource_name],
                str(config.gpu_count),
            )
            self.assertEqual(
                job.spec.template.spec.volumes[0].persistent_volume_claim.claim_name,
                config.shared_pvc_name,
            )

            env_names = {entry.name for entry in container.env}
            for required in {
                "RUN_ID",
                "MODEL",
                "DATASET_REL_PATH",
                "HP_JSON",
                "TOKEN_BUDGET",
                "SEQ_LEN",
                "SEED",
                "OUTPUT_ROOT",
                "DATASETS_ROOT",
                "SCHEDULER_CALLBACK_URL",
            }:
                self.assertIn(required, env_names)


if __name__ == "__main__":
    unittest.main()

