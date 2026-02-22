from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from train_config import DC_PROFILES
from train_orchestrator import TrainingOrchestrator, parse_hp_json
from train_store import TrainRunStore


class ParseHpJsonTests(unittest.TestCase):
    def test_parse_hp_json_valid(self):
        hp = parse_hp_json(
            json.dumps(
                {"lora_r": 16, "learning_rate": 3e-4, "lora_dropout": 0.05},
            )
        )
        self.assertEqual(hp["lora_r"], 16.0)
        self.assertEqual(hp["learning_rate"], 3e-4)
        self.assertEqual(hp["lora_dropout"], 0.05)

    def test_parse_hp_json_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_hp_json("{not-json}")

    def test_parse_hp_json_missing_keys(self):
        with self.assertRaises(ValueError):
            parse_hp_json(json.dumps({"lora_r": 8}))

    def test_parse_hp_json_invalid_ranges(self):
        with self.assertRaises(ValueError):
            parse_hp_json(
                json.dumps(
                    {"lora_r": 0, "learning_rate": 3e-4, "lora_dropout": 0.1},
                )
            )

        with self.assertRaises(ValueError):
            parse_hp_json(
                json.dumps(
                    {"lora_r": 8, "learning_rate": -1e-4, "lora_dropout": 0.1},
                )
            )

        with self.assertRaises(ValueError):
            parse_hp_json(
                json.dumps(
                    {"lora_r": 8, "learning_rate": 1e-4, "lora_dropout": 1.1},
                )
            )


class QueuePolicyTests(unittest.TestCase):
    def test_orchestrator_runs_one_job_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "runs.csv"
            store = TrainRunStore(csv_path=csv_path)
            dc_id = next(iter(DC_PROFILES))
            hp_json = json.dumps(
                {"lora_r": 8, "learning_rate": 3e-4, "lora_dropout": 0.1}
            )

            first = store.create_run(
                hp_json=hp_json,
                dc_id=dc_id,
                namespace="hack-europe-team-i",
                dataset_rel_path="demo-set",
                job_name=None,
            )
            second = store.create_run(
                hp_json=hp_json,
                dc_id=dc_id,
                namespace="hack-europe-team-i",
                dataset_rel_path="demo-set",
                job_name=None,
            )

            states = {}

            def submit_job_fn(**kwargs):
                run_id = kwargs["run_id"]
                states[run_id] = "RUNNING"
                return f"k8s-{run_id}"

            def get_job_status_fn(namespace: str, job_name: str):
                run_id = job_name.replace("k8s-", "")
                state = states.get(run_id, "NOT_FOUND")
                if state == "RUNNING":
                    return {
                        "state": "RUNNING",
                        "reason": "",
                        "pod_name": "pod-1",
                        "completion_time": None,
                    }
                if state == "SUCCEEDED":
                    return {
                        "state": "SUCCEEDED",
                        "reason": "",
                        "pod_name": "pod-1",
                        "completion_time": "2026-02-21T00:00:00+00:00",
                    }
                return {
                    "state": "NOT_FOUND",
                    "reason": "missing",
                    "pod_name": None,
                    "completion_time": None,
                }

            orchestrator = TrainingOrchestrator(
                store=store,
                poll_interval_seconds=0.01,
                submit_job_fn=submit_job_fn,
                get_job_status_fn=get_job_status_fn,
            )

            orchestrator.tick()
            first_row = store.get_run(first["run_id"])
            second_row = store.get_run(second["run_id"])
            self.assertEqual(first_row["status"], "RUNNING")
            self.assertEqual(second_row["status"], "QUEUED")

            states[first["run_id"]] = "SUCCEEDED"
            orchestrator.tick()
            first_row = store.get_run(first["run_id"])
            self.assertEqual(first_row["status"], "SUCCEEDED")

            orchestrator.tick()
            second_row = store.get_run(second["run_id"])
            self.assertEqual(second_row["status"], "RUNNING")


if __name__ == "__main__":
    unittest.main()
