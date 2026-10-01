"""SQLite persistence for immutable consensus versions.

Stored versions are append-only and never updated or deleted.  A
``BEGIN IMMEDIATE`` transaction plus a busy timeout makes
same-``request_id`` idempotency safe under concurrent writers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import contextlib
from typing import Optional, Tuple

from .consensus import build_consensus, payload_hash
from .schemas import ConsensusRequest, ConsensusResult

DEFAULT_DB_PATH = os.environ.get("UMI_DB_PATH", "/data/umi.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id   TEXT PRIMARY KEY,
    sample_id    TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    version      INTEGER NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    result_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    version      INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    result_json  TEXT NOT NULL
);
"""


class ConflictError(Exception):
    """Same request_id replayed with a different payload (HTTP 409)."""


class Database:
    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    def get_by_request(self, request_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def get_version(self, version: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM versions WHERE version = ?",
                (version,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def latest_version(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(version) AS v FROM versions").fetchone()
        return row["v"] if row and row["v"] is not None else None

    # ------------------------------------------------------------------
    def submit(self, req: ConsensusRequest, created_at: str) -> Tuple[dict, bool]:
        """Insert a new version. Returns (result, replayed).

        Raises ConflictError if the request_id was already stored with a
        different payload hash.
        """
        phash = payload_hash(req)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload_hash, result_json FROM requests WHERE request_id = ?",
                (req.request_id,),
            ).fetchone()
            if row is not None:
                conn.commit()
                if row["payload_hash"] != phash:
                    raise ConflictError(
                        f"request_id={req.request_id!r} already submitted with a "
                        "different payload"
                    )
                return json.loads(row["result_json"]), True

            result = build_consensus(req)
            cur = conn.execute(
                "INSERT INTO versions (request_id, created_at, result_json) "
                "VALUES (?, ?, '{}')",
                (req.request_id, created_at),
            )
            version = cur.lastrowid
            result["version"] = version
            result_blob = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            conn.execute(
                "UPDATE versions SET result_json = ? WHERE version = ?",
                (result_blob, version),
            )
            conn.execute(
                "INSERT INTO requests "
                "(request_id, sample_id, payload_hash, version, created_at, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    req.request_id,
                    req.sample_id,
                    phash,
                    version,
                    created_at,
                    result_blob,
                ),
            )
            conn.commit()
            return result, False
