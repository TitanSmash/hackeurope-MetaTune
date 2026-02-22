from __future__ import annotations

import sqlite3
from pathlib import Path


RUN_COLUMNS = [
    "run_id",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "namespace",
    "job_name",
    "k8s_job_name",
    "model",
    "dataset_rel_path",
    "hp_json",
    "token_budget",
    "seq_len",
    "seed",
    "output_dir",
    "metrics_json",
    "artifacts_json",
    "failure_reason",
    "callback_received_at",
]

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    namespace TEXT NOT NULL,
    job_name TEXT NOT NULL DEFAULT '',
    k8s_job_name TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    dataset_rel_path TEXT NOT NULL,
    hp_json TEXT NOT NULL,
    token_budget INTEGER NOT NULL,
    seq_len INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    output_dir TEXT NOT NULL DEFAULT '',
    metrics_json TEXT NOT NULL DEFAULT '',
    artifacts_json TEXT NOT NULL DEFAULT '',
    failure_reason TEXT NOT NULL DEFAULT '',
    callback_received_at TEXT NOT NULL DEFAULT ''
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);",
]


def open_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(CREATE_RUNS_TABLE_SQL)
        for stmt in CREATE_INDEXES_SQL:
            conn.execute(stmt)
    return conn

