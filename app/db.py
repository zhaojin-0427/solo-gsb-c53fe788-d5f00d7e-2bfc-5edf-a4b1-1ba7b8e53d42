"""SQLite persistence for immutable consensus versions.

Stored versions are append-only and never updated or deleted.  A
``BEGIN IMMEDIATE`` transaction plus a busy timeout makes
same-``request_id`` idempotency safe under concurrent writers.

A ``read_contents`` registry persists the content fingerprint of every
accepted ``read_id`` so the same id cannot silently reappear with
different locus/umi/sequence/qualities in a later request.  The registry
is checked and extended inside the same transaction that allocates the
version, so concurrent conflicting submissions cannot both succeed.

Databases created before the registry existed are migrated on open:
read_ids found in historical results are registered with an unknown
(NULL) fingerprint — the original content is not stored in old snapshots
and cannot be reconstructed, so any later reuse of those ids is rejected
rather than guessed.  Historical versions remain readable unchanged.
"""
from __future__ import annotations

import json
import os
import sqlite3
import contextlib
from typing import Dict, List, Optional, Tuple

from .consensus import build_consensus, payload_hash, read_content_hash
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

-- One row per read_id ever accepted.  content_hash is NULL for ids
-- backfilled from pre-registry databases (original content unknown).
CREATE TABLE IF NOT EXISTS read_contents (
    read_id       TEXT PRIMARY KEY,
    content_hash  TEXT,
    first_version INTEGER,
    created_at    TEXT NOT NULL
);
"""

# Schema/user version for idempotent migrations (PRAGMA user_version).
_USER_VERSION = 1

# SQLite caps bound variables per statement; keep IN (...) chunks well below.
_LOOKUP_CHUNK = 500


class ConflictError(Exception):
    """Same request_id replayed with a different payload (HTTP 409)."""


class ReadConflictError(Exception):
    """A read_id was already persisted with different/unknown content (HTTP 409)."""

    def __init__(self, conflicts: List[dict]) -> None:
        super().__init__("read_id content conflict")
        self.conflicts = conflicts


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
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("PRAGMA user_version").fetchone()[0] < _USER_VERSION:
                    self._backfill_read_contents(conn)
                    conn.execute(f"PRAGMA user_version = {_USER_VERSION}")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _backfill_read_contents(self, conn: sqlite3.Connection) -> None:
        """Register read_ids already present in stored versions as unknown.

        Old snapshots contain read_ids (in ``clusters[].read_ids``) but not
        their locus/umi/sequence/qualities, so the content cannot be
        reconstructed.  Rows are inserted with a NULL fingerprint: later
        submissions referencing these ids are rejected instead of trusting
        unverifiable content.  Existing versions/requests rows are untouched.
        """
        rows = conn.execute(
            "SELECT version, created_at, result_json FROM versions ORDER BY version"
        ).fetchall()
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except (TypeError, ValueError):
                continue
            read_ids = set()
            for cluster in result.get("clusters") or []:
                read_ids.update(cluster.get("read_ids") or [])
            for rid in sorted(read_ids):
                # Versions are scanned in ascending order, so the first
                # insert wins and records the earliest sighting.
                conn.execute(
                    "INSERT OR IGNORE INTO read_contents "
                    "(read_id, content_hash, first_version, created_at) "
                    "VALUES (?, NULL, ?, ?)",
                    (rid, row["version"], row["created_at"]),
                )

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

    def read_content_record(self, read_id: str) -> Optional[dict]:
        """Return the registry row for read_id (mainly for tests/ops)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT read_id, content_hash, first_version, created_at "
                "FROM read_contents WHERE read_id = ?",
                (read_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    @staticmethod
    def _payload_fingerprints(req: ConsensusRequest) -> Dict[str, str]:
        """Map read_id -> content fingerprint for the payload.

        Raises ReadConflictError if one read_id carries two different
        contents inside the same payload (the API layer normally rejects
        this first; kept here so direct callers stay safe too).
        """
        entries: Dict[str, str] = {}
        conflicts: List[dict] = []
        for r in req.reads:
            chash = read_content_hash(r)
            previous = entries.setdefault(r.read_id, chash)
            if previous != chash:
                conflicts.append(
                    {"read_id": r.read_id, "reason": "same read_id with different content"}
                )
        if conflicts:
            raise ReadConflictError(conflicts)
        return entries

    @staticmethod
    def _check_read_registry(
        conn: sqlite3.Connection, entries: Dict[str, str]
    ) -> None:
        """Compare payload read_ids against the persisted registry.

        Must be called while holding the write transaction so the check and
        the subsequent insert are atomic against concurrent submitters.
        """
        ids = list(entries)
        conflicts: List[dict] = []
        for i in range(0, len(ids), _LOOKUP_CHUNK):
            chunk = ids[i : i + _LOOKUP_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT read_id, content_hash FROM read_contents "
                f"WHERE read_id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            for row in rows:
                stored = row["content_hash"]
                if stored is None:
                    conflicts.append(
                        {
                            "read_id": row["read_id"],
                            "reason": "read_id exists in a version created before "
                            "content tracking; original content cannot be verified",
                        }
                    )
                elif stored != entries[row["read_id"]]:
                    conflicts.append(
                        {
                            "read_id": row["read_id"],
                            "reason": "read_id already recorded with different content",
                        }
                    )
        if conflicts:
            raise ReadConflictError(conflicts)

    def submit(self, req: ConsensusRequest, created_at: str) -> Tuple[dict, bool]:
        """Insert a new version. Returns (result, replayed).

        Raises ConflictError if the request_id was already stored with a
        different payload hash, or ReadConflictError if any read_id was
        already persisted with different (or unverifiable) content.
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
                # Idempotent replay: content was registered at first submit.
                return json.loads(row["result_json"]), True

            try:
                entries = self._payload_fingerprints(req)
                self._check_read_registry(conn, entries)

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
                for rid, chash in entries.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO read_contents "
                        "(read_id, content_hash, first_version, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (rid, chash, version, created_at),
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            return result, False
