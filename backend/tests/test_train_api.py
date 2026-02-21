from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False

from train_config import DC_PROFILES
from train_orchestrator import TrainingOrchestrator
from train_store import TrainRunStore


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class TrainApiTests(unittest.TestCase):
    def setUp(self):
        os.environ["ENABLE_TRAIN_ORCHESTRATOR"] = "0"
        self.api_module = importlib.import_module("api")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = TrainRunStore(csv_path=Path(self.tmpdir.name) / "runs.csv")
        self.api_module.run_store = self.store
        self.api_module.train_orchestrator = TrainingOrchestrator(self.store)
        self.api_module.train_orchestrator.enabled = False
        self.client = TestClient(self.api_module.app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_create_get_and_list_run(self):
        dc_id = next(iter(DC_PROFILES))
        payload = {
            "hp_json": json.dumps(
                {"lora_r": 16, "learning_rate": 0.0003, "lora_dropout": 0.1}
            ),
            "dc_id": dc_id,
            "dataset_rel_path": "demo-ds",
        }
        create_resp = self.client.post("/api/train/runs", json=payload)
        self.assertEqual(create_resp.status_code, 200)
        body = create_resp.json()
        self.assertEqual(body["status"], "QUEUED")
        self.assertEqual(body["dc_id"], dc_id)
        self.assertIn("run_id", body)

        run_id = body["run_id"]
        get_resp = self.client.get(f"/api/train/runs/{run_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["run_id"], run_id)

        list_resp = self.client.get("/api/train/runs?limit=10")
        self.assertEqual(list_resp.status_code, 200)
        runs = list_resp.json()["runs"]
        self.assertTrue(any(r["run_id"] == run_id for r in runs))

    def test_create_run_rejects_invalid_hp_json(self):
        dc_id = next(iter(DC_PROFILES))
        payload = {
            "hp_json": "{bad-json",
            "dc_id": dc_id,
            "dataset_rel_path": "demo-ds",
        }
        create_resp = self.client.post("/api/train/runs", json=payload)
        self.assertEqual(create_resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
