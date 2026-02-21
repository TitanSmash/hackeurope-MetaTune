from __future__ import annotations

import csv
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from train_config import RUNS_CSV_PATH

RUN_COLUMNS = [
    "run_id",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "dc_id",
    "namespace",
    "job_name",
    "k8s_job_name",
    "hp_json",
    "dataset_rel_path",
    "failure_reason",
]

ACTIVE_STATUSES = {"SUBMITTING", "RUNNING"}
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainRunStore:
    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = csv_path or RUNS_CSV_PATH
        self._lock = threading.Lock()
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
                writer.writeheader()

    def _read_all(self) -> list[dict[str, str]]:
        self._ensure_csv()
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                normalized = {col: (row.get(col, "") or "") for col in RUN_COLUMNS}
                rows.append(normalized)
            return rows

    def _write_all(self, rows: list[dict[str, str]]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".train_runs_", suffix=".csv", dir=str(self.csv_path.parent)
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({col: row.get(col, "") for col in RUN_COLUMNS})
            os.replace(tmp_path, self.csv_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def create_run(
        self,
        *,
        hp_json: str,
        dc_id: str,
        namespace: str,
        dataset_rel_path: str | None,
        job_name: str | None,
    ) -> dict[str, str]:
        with self._lock:
            rows = self._read_all()
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
            now = utc_now_iso()
            row = {
                "run_id": run_id,
                "status": "QUEUED",
                "created_at": now,
                "started_at": "",
                "finished_at": "",
                "dc_id": dc_id,
                "namespace": namespace,
                "job_name": job_name or "",
                "k8s_job_name": "",
                "hp_json": hp_json,
                "dataset_rel_path": dataset_rel_path or "",
                "failure_reason": "",
            }
            rows.append(row)
            self._write_all(rows)
            row["queue_position"] = str(self.queue_position(rows, run_id))
            return row

    @staticmethod
    def queue_position(rows: list[dict[str, str]], run_id: str) -> int:
        queued = [r for r in rows if r.get("status") == "QUEUED"]
        for idx, row in enumerate(queued, start=1):
            if row.get("run_id") == run_id:
                return idx
        return 0

    def get_run(self, run_id: str) -> dict[str, str] | None:
        with self._lock:
            rows = self._read_all()
            for row in rows:
                if row.get("run_id") == run_id:
                    return row
            return None

    def list_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, str]]:
        with self._lock:
            rows = self._read_all()
            if status:
                rows = [r for r in rows if r.get("status") == status]
            rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            return rows[: max(1, limit)]

    def update_run(self, run_id: str, **updates: Any) -> dict[str, str] | None:
        with self._lock:
            rows = self._read_all()
            updated = None
            for row in rows:
                if row.get("run_id") == run_id:
                    for key, value in updates.items():
                        if key in RUN_COLUMNS:
                            row[key] = "" if value is None else str(value)
                    updated = row
                    break
            if updated is not None:
                self._write_all(rows)
            return updated

    def get_active_run(self) -> dict[str, str] | None:
        with self._lock:
            rows = self._read_all()
            active_rows = [r for r in rows if r.get("status") in ACTIVE_STATUSES]
            if not active_rows:
                return None
            active_rows.sort(key=lambda r: r.get("created_at", ""))
            return active_rows[0]

    def get_next_queued_run(self) -> dict[str, str] | None:
        with self._lock:
            rows = self._read_all()
            queued = [r for r in rows if r.get("status") == "QUEUED"]
            if not queued:
                return None
            queued.sort(key=lambda r: r.get("created_at", ""))
            return queued[0]
