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
from app.dispatcher import Dispatcher
from app.store import RunStore


def _test_config(tmpdir: Path) -> AppConfig:
    env = {
        "NAMESPACE": "hack-europe-team-i",
        "DATA_ROOT": str(tmpdir / "data"),
        "DATASETS_ROOT": str(tmpdir / "data" / "datasets"),
        "OUTPUT_ROOT": str(tmpdir / "data" / "outputs"),
        "DB_PATH": str(tmpdir / "data" / "db" / "scheduler.sqlite"),
        "DISPATCH_INTERVAL_SECONDS": "0.1",
        "ENABLE_DISPATCHER": "0",
        "TRAINER_IMAGE": "test/image:latest",
    }
    with patch.dict(os.environ, env, clear=False):
        return AppConfig.from_env()


class DispatcherTests(unittest.TestCase):
    def test_one_active_run_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            config = _test_config(tmpdir)
            store = RunStore(config.db_path)
            hp_json = json.dumps(
                {
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "learning_rate": 3e-4,
                    "batch_size": 8,
                }
            )
            first = store.enqueue_run(
                namespace=config.namespace,
                model="gpt2",
                dataset_rel_path="tinystories",
                hp_json=hp_json,
                token_budget=1000,
                seq_len=128,
                seed=1,
                job_name="",
                output_dir=str(config.output_root / "pending"),
            )
            second = store.enqueue_run(
                namespace=config.namespace,
                model="gpt2",
                dataset_rel_path="tinystories",
                hp_json=hp_json,
                token_budget=1000,
                seq_len=128,
                seed=1,
                job_name="",
                output_dir=str(config.output_root / "pending"),
            )

            states: dict[str, str] = {}

            def submit_job_fn(*, run, app_config):
                states[str(run["run_id"])] = "RUNNING"
                return f"k8s-{run['run_id']}"

            def get_job_status_fn(*, namespace, job_name):
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
                        "completion_time": "2026-02-22T00:00:00+00:00",
                    }
                return {
                    "state": "NOT_FOUND",
                    "reason": "missing",
                    "pod_name": None,
                    "completion_time": None,
                }

            dispatcher = Dispatcher(
                app_config=config,
                store=store,
                submit_job_fn=submit_job_fn,
                get_job_status_fn=get_job_status_fn,
            )

            dispatcher.tick()
            first_row = store.get_run(str(first["run_id"]))
            second_row = store.get_run(str(second["run_id"]))
            self.assertEqual(first_row["status"], "RUNNING")
            self.assertEqual(second_row["status"], "QUEUED")

            first_output_dir = Path(str(first_row["output_dir"]))
            first_output_dir.mkdir(parents=True, exist_ok=True)
            (first_output_dir / "metrics.json").write_text(
                json.dumps({"val_loss": 1.23, "best_val_loss": 1.2}),
                encoding="utf-8",
            )
            states[str(first["run_id"])] = "SUCCEEDED"
            dispatcher.tick()
            first_row = store.get_run(str(first["run_id"]))
            self.assertEqual(first_row["status"], "SUCCEEDED")

            dispatcher.tick()
            second_row = store.get_run(str(second["run_id"]))
            self.assertEqual(second_row["status"], "RUNNING")
            store.close()


if __name__ == "__main__":
    unittest.main()

