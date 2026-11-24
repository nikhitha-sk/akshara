"""SQLite storage for request history and user corrections (the data flywheel).

Swap for PostgreSQL in Phase 5 by changing AKSHARA_DB / the connect() call.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from common.paths import DATA

DB_PATH = Path(os.getenv("AKSHARA_DB", DATA / "akshara.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    lang TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    model_version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    input_ref TEXT,
    model_output TEXT NOT NULL,
    user_fix TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_request(conn, tool: str, lang: str, inp: str, out: dict, version: str) -> int:
    with _lock:
        cur = conn.execute(
            "INSERT INTO requests(tool, lang, input, output, model_version) VALUES (?,?,?,?,?)",
            (tool, lang, inp, json.dumps(out, ensure_ascii=False), version),
        )
        conn.commit()
        return cur.lastrowid


def add_feedback(conn, tool: str, input_ref: str | None, model_output: str, user_fix: str) -> int:
    with _lock:
        cur = conn.execute(
            "INSERT INTO feedback(tool, input_ref, model_output, user_fix) VALUES (?,?,?,?)",
            (tool, input_ref, model_output, user_fix),
        )
        conn.commit()
        return cur.lastrowid


def history(conn, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT id, tool, lang, input, output, created_at FROM requests ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) | {"output": json.loads(r["output"])} for r in rows]
