from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.config import AppConfig
    from app.main import create_app
    from fastapi.testclient import TestClient

    FASTAPI_TESTCLIENT_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_TESTCLIENT_AVAILABLE = False


@unittest.skipUnless(FASTAPI_TESTCLIENT_AVAILABLE, "fastapi test client unavailable")
class ApiTests(unittest.TestCase):
    def test_create_get_list_and_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dataset_dir = tmpdir / "data" / "datasets" / "tinystories"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "train.bin").write_bytes(b"\x00\x01")
            (dataset_dir / "val.bin").write_bytes(b"\x00\x01")

            env = {
                "NAMESPACE": "hack-europe-team-i",
                "DATA_ROOT": str(tmpdir / "data"),
                "DATASETS_ROOT": str(tmpdir / "data" / "datasets"),
                "OUTPUT_ROOT": str(tmpdir / "data" / "outputs"),
                "DB_PATH": str(tmpdir / "data" / "db" / "scheduler.sqlite"),
                "ENABLE_DISPATCHER": "0",
                "TRAINER_IMAGE": "example/trainer:latest",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_app(app_config=AppConfig.from_env())

            client = TestClient(app)

            create_payload = {
                "model": "gpt2",
                "dataset_rel_path": "tinystories",
                "hp": {
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "learning_rate": 3e-4,
                    "batch_size": 8,
                },
                "token_budget": 10_000,
                "seq_len": 256,
                "seed": 1,
            }
            create_resp = client.post("/api/v1/runs", json=create_payload)
            self.assertEqual(create_resp.status_code, 200)
            body = create_resp.json()
            self.assertEqual(body["status"], "QUEUED")
            run_id = body["run_id"]

            get_resp = client.get(f"/api/v1/runs/{run_id}")
            self.assertEqual(get_resp.status_code, 200)
            self.assertEqual(get_resp.json()["run_id"], run_id)

            list_resp = client.get("/api/v1/runs?limit=10")
            self.assertEqual(list_resp.status_code, 200)
            runs = list_resp.json()["runs"]
            self.assertTrue(any(row["run_id"] == run_id for row in runs))

            callback_resp = client.post(
                f"/internal/runs/{run_id}/complete",
                json={
                    "status": "SUCCEEDED",
                    "metrics": {"val_loss": 1.0},
                    "artifacts": {"output_dir": "/data/outputs/run"},
                },
            )
            self.assertEqual(callback_resp.status_code, 200)
            self.assertEqual(callback_resp.json()["status"], "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
