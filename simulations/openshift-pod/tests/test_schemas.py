from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pydantic import ValidationError
    from app.schemas import CreateRunRequest

    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:
    PYDANTIC_AVAILABLE = False


def _valid_payload() -> dict:
    return {
        "model": "gpt2",
        "dataset_rel_path": "tinystories",
        "hp": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 3e-4,
            "batch_size": 8,
        },
        "token_budget": 25_000_000,
        "seq_len": 1024,
        "seed": 1,
    }


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic is not installed")
class SchemaValidationTests(unittest.TestCase):
    def test_accepts_valid_payload(self) -> None:
        request = CreateRunRequest(**_valid_payload())
        self.assertEqual(request.model, "gpt2")
        self.assertEqual(request.dataset_rel_path, "tinystories")

    def test_rejects_unsupported_model(self) -> None:
        payload = _valid_payload()
        payload["model"] = "llama2"
        with self.assertRaises(ValidationError):
            CreateRunRequest(**payload)

    def test_rejects_absolute_dataset_path(self) -> None:
        payload = _valid_payload()
        payload["dataset_rel_path"] = "/abs/path"
        with self.assertRaises(ValidationError):
            CreateRunRequest(**payload)

    def test_rejects_dataset_path_traversal(self) -> None:
        payload = _valid_payload()
        payload["dataset_rel_path"] = "../secret"
        with self.assertRaises(ValidationError):
            CreateRunRequest(**payload)

    def test_rejects_negative_lora_r(self) -> None:
        payload = _valid_payload()
        payload["hp"]["lora_r"] = 0
        with self.assertRaises(ValidationError):
            CreateRunRequest(**payload)


if __name__ == "__main__":
    unittest.main()
