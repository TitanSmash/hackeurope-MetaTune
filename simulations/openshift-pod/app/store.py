from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.db import RUN_COLUMNS, open_database
from app.utils import generate_run_id, utc_now_iso


ACTIVE_STATUSES = {"SUBMITTING", "RUNNING"}
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.conn = open_database(db_path)
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {column: row[column] for column in RUN_COLUMNS}

    def enqueue_run(
        self,
        *,
        namespace: str,
        model: str,
        dataset_rel_path: str,
        hp_json: str,
        token_budget: int,
        seq_len: int,
        seed: int,
        job_name: str,
        output_dir: str,
    ) -> dict[str, Any]:
        run_id = generate_run_id()
        created_at = utc_now_iso()
        row = {
            "run_id": run_id,
            "status": "QUEUED",
            "created_at": created_at,
            "started_at": "",
            "finished_at": "",
            "namespace": namespace,
            "job_name": job_name,
            "k8s_job_name": "",
            "model": model,
            "dataset_rel_path": dataset_rel_path,
            "hp_json": hp_json,
            "token_budget": int(token_budget),
            "seq_len": int(seq_len),
            "seed": int(seed),
            "output_dir": output_dir,
            "metrics_json": "",
            "artifacts_json": "",
            "failure_reason": "",
            "callback_received_at": "",
        }
        with self._lock, self.conn:
            fields = ", ".join(RUN_COLUMNS)
            placeholders = ", ".join(["?"] * len(RUN_COLUMNS))
            values = [row[column] for column in RUN_COLUMNS]
            self.conn.execute(
                f"INSERT INTO runs ({fields}) VALUES ({placeholders})",
                values,
            )
            cursor = self.conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            )
            fetched = cursor.fetchone()
        return self._row_to_dict(fetched) or row

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cursor = self.conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            )
            return self._row_to_dict(cursor.fetchone())

    def list_runs(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                cursor = self.conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, max(1, int(limit))),
                )
            else:
                cursor = self.conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                )
            return [self._row_to_dict(row) for row in cursor.fetchall() if row is not None]

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any] | None:
        mutable_columns = [column for column in updates if column in RUN_COLUMNS and column != "run_id"]
        if not mutable_columns:
            return self.get_run(run_id)
        set_clause = ", ".join(f"{column} = ?" for column in mutable_columns)
        values = [updates[column] for column in mutable_columns]
        values.append(run_id)
        with self._lock, self.conn:
            self.conn.execute(
                f"UPDATE runs SET {set_clause} WHERE run_id = ?",
                values,
            )
            cursor = self.conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            )
            return self._row_to_dict(cursor.fetchone())

    def get_active_run(self) -> dict[str, Any] | None:
        with self._lock:
            placeholders = ", ".join(["?"] * len(ACTIVE_STATUSES))
            cursor = self.conn.execute(
                f"""
                SELECT * FROM runs
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                LIMIT 1
                """,
                tuple(ACTIVE_STATUSES),
            )
            return self._row_to_dict(cursor.fetchone())

    def get_next_queued_run(self) -> dict[str, Any] | None:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT * FROM runs
                WHERE status = 'QUEUED'
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            return self._row_to_dict(cursor.fetchone())

