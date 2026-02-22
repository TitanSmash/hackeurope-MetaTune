from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.store import RunStore


class StorePersistenceTests(unittest.TestCase):
    def test_sqlite_persists_runs_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "scheduler.sqlite"
            store = RunStore(db_path)
            row = store.enqueue_run(
                namespace="hack-europe-team-i",
                model="gpt2",
                dataset_rel_path="tinystories",
                hp_json=json.dumps(
                    {
                        "lora_r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.1,
                        "learning_rate": 1e-4,
                        "batch_size": 8,
                    }
                ),
                token_budget=1000,
                seq_len=128,
                seed=1,
                job_name="",
                output_dir="/data/outputs/pending",
            )
            run_id = str(row["run_id"])
            store.close()

            store_reloaded = RunStore(db_path)
            restored = store_reloaded.get_run(run_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored["run_id"], run_id)
            self.assertEqual(restored["status"], "QUEUED")
            store_reloaded.close()


if __name__ == "__main__":
    unittest.main()

