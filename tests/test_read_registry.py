"""Cross-request read_id registry: conflicts, reuse, concurrency, migration."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.consensus import build_consensus, payload_hash
from app.db import Database, ReadConflictError
from app.schemas import ConsensusRequest, ReadIn

OLD_SCHEMA = """
CREATE TABLE requests (
    request_id   TEXT PRIMARY KEY,
    sample_id    TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    version      INTEGER NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    result_json  TEXT NOT NULL
);
CREATE TABLE versions (
    version      INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    result_json  TEXT NOT NULL
);
"""


def make_read(read_id, sequence="ACGT", locus="L1", umi="AAAAA", quals=None):
    return {
        "locus": locus,
        "read_id": read_id,
        "umi": umi,
        "sequence": sequence,
        "qualities": quals if quals is not None else [30] * len(sequence),
    }


def make_payload(request_id, reads, sample_id="s"):
    return {"request_id": request_id, "sample_id": sample_id, "reads": reads}


def trio(read_id, **kw):
    """Three reads that share one UMI so the request yields a consensus."""
    return [make_read(read_id, **kw), make_read(f"{read_id}-x", **kw), make_read(f"{read_id}-y", **kw)]


# ---------------------------------------------------------------------------
# cross-request conflicts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "changed_field",
    ["locus", "umi", "sequence", "qualities"],
)
def test_cross_request_readid_content_change_is_409(client: TestClient, changed_field):
    first = make_payload("req-a", trio("shared"))
    assert client.post("/consensus", json=first).status_code == 200

    changed = make_read("shared")
    changed[changed_field] = {"locus": "L2", "umi": "AAAAT", "sequence": "ACGG",
                              "qualities": [31, 30, 30, 30]}[changed_field]
    second = make_payload("req-b", [changed, make_read("n1"), make_read("n2")])

    resp = client.post("/consensus", json=second)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "read_id content conflict"
    assert detail["conflicts"][0]["read_id"] == "shared"

    # No new version, and the original snapshot is untouched.
    assert client.get("/healthz").json()["latest_version"] == 1
    assert client.get("/consensus/2").status_code == 404
    assert client.get("/consensus/1").json()["request_id"] == "req-a"


def test_cross_request_same_content_creates_new_version(client: TestClient):
    first = make_payload("req-a", trio("shared"))
    assert client.post("/consensus", json=first).status_code == 200

    # Different request_id, but read_id "shared" carries identical content.
    second = make_payload("req-b", trio("shared"))
    resp = client.post("/consensus", json=second)
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Idempotent-Replay"] == "false"
    assert resp.json()["version"] == 2
    assert client.get("/healthz").json()["latest_version"] == 2


def test_replay_still_returns_original_version(client: TestClient):
    payload = make_payload("req-a", trio("shared"))
    first = client.post("/consensus", json=payload)
    assert first.status_code == 200

    replay = client.post("/consensus", json=payload)
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert replay.json() == first.json()
    assert client.get("/healthz").json()["latest_version"] == 1


def test_same_request_id_different_payload_still_409(client: TestClient):
    payload = make_payload("req-a", trio("shared"))
    assert client.post("/consensus", json=payload).status_code == 200

    changed = make_payload("req-a", trio("shared", sequence="ACGG"))
    resp = client.post("/consensus", json=changed)
    assert resp.status_code == 409
    assert "different payload" in resp.json()["detail"]["error"]
    assert client.get("/healthz").json()["latest_version"] == 1


def test_conflicting_read_id_blocks_only_the_conflicting_request(client: TestClient):
    assert client.post(
        "/consensus", json=make_payload("req-a", trio("shared"))
    ).status_code == 200

    # Unrelated read_ids are unaffected by the registry entry for "shared".
    ok = client.post("/consensus", json=make_payload("req-b", trio("fresh")))
    assert ok.status_code == 200
    assert ok.json()["version"] == 2


# ---------------------------------------------------------------------------
# concurrency: only one content may win
# ---------------------------------------------------------------------------

def test_concurrent_conflicting_submits_only_one_lands(tmp_path):
    db = Database(str(tmp_path / "race.db"))

    def build(request_id, seq):
        return ConsensusRequest(
            request_id=request_id,
            sample_id="s",
            reads=[
                ReadIn(locus="L1", read_id="shared", umi="AAAAA", sequence=seq,
                       qualities=[30] * 4),
            ],
        )

    outcomes = []

    def attempt(req):
        try:
            db.submit(req, "2026-10-02T00:00:00Z")
            outcomes.append("ok")
        except ReadConflictError:
            outcomes.append("conflict")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(attempt, build("race-a", "ACGT")),
            pool.submit(attempt, build("race-b", "ACGG")),
        ]
        for f in futs:
            f.result()

    assert sorted(outcomes) == ["conflict", "ok"]
    assert db.latest_version() == 1
    # The losing content was never registered.
    winner = db.get_by_request("race-a") or db.get_by_request("race-b")
    assert winner is not None


def test_concurrent_identical_submits_both_serialize(tmp_path):
    db = Database(str(tmp_path / "race2.db"))

    def build(request_id):
        return ConsensusRequest(
            request_id=request_id,
            sample_id="s",
            reads=[
                ReadIn(locus="L1", read_id="shared", umi="AAAAA", sequence="ACGT",
                       qualities=[30] * 4),
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda rid: db.submit(build(rid), "2026-10-02T00:00:00Z"),
                ["same-a", "same-b"],
            )
        )
    # Same content under two request_ids: both commits succeed, two versions.
    assert [replayed for _, replayed in results] == [False, False]
    assert db.latest_version() == 2


# ---------------------------------------------------------------------------
# migration of pre-registry databases
# ---------------------------------------------------------------------------

def _build_legacy_db(path):
    """Create a database with the pre-registry schema and one stored version."""
    reads = [
        ReadIn(locus="L1", read_id=rid, umi="AAAAA", sequence="ACGT",
               qualities=[30] * 4)
        for rid in ("legacy-read", "legacy-2", "legacy-3")
    ]
    req = ConsensusRequest(request_id="legacy-req", sample_id="old-sample", reads=reads)
    result = build_consensus(req)
    result["version"] = 1
    blob = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO versions (request_id, created_at, result_json) VALUES (?, ?, ?)",
        ("legacy-req", "2026-01-01T00:00:00Z", blob),
    )
    conn.execute(
        "INSERT INTO requests "
        "(request_id, sample_id, payload_hash, version, created_at, result_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("legacy-req", "old-sample", payload_hash(req), 1, "2026-01-01T00:00:00Z", blob),
    )
    conn.commit()
    conn.close()
    return result


@pytest.fixture()
def legacy_client(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    snapshot = _build_legacy_db(str(db_path))
    monkeypatch.setenv("UMI_DB_PATH", str(db_path))
    from app.main import app

    with TestClient(app) as c:
        c.legacy_snapshot = snapshot
        c.legacy_db_path = str(db_path)
        yield c


def test_migration_keeps_old_versions_readable(legacy_client):
    got = legacy_client.get("/consensus/1")
    assert got.status_code == 200
    assert got.json() == legacy_client.legacy_snapshot


def test_migration_rejects_unverifiable_read_id_reuse(legacy_client):
    # Even byte-identical content cannot be verified against a NULL record.
    payload = make_payload("new-req", trio("legacy-read"))
    resp = legacy_client.post("/consensus", json=payload)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "read_id content conflict"
    assert detail["conflicts"][0]["read_id"] == "legacy-read"
    assert "cannot be verified" in detail["conflicts"][0]["reason"]

    # Nothing was written: no new version, old snapshot intact.
    assert legacy_client.get("/healthz").json()["latest_version"] == 1
    assert legacy_client.get("/consensus/1").json() == legacy_client.legacy_snapshot


def test_migration_does_not_fabricate_content_records(legacy_client):
    db = Database(legacy_client.legacy_db_path)
    record = db.read_content_record("legacy-read")
    assert record is not None
    assert record["content_hash"] is None  # unknown, not guessed
    assert record["first_version"] == 1

    # A rejected reuse attempt must not fill in the unknown content either.
    legacy_client.post("/consensus", json=make_payload("new-req", trio("legacy-read")))
    assert db.read_content_record("legacy-read")["content_hash"] is None


def test_migration_allows_fresh_read_ids_and_legacy_replay(legacy_client):
    # Brand-new read_ids still work on a migrated database.
    ok = legacy_client.post("/consensus", json=make_payload("new-req", trio("fresh")))
    assert ok.status_code == 200
    assert ok.json()["version"] == 2

    # Replaying the pre-migration request still returns the original version.
    replay = legacy_client.post(
        "/consensus",
        json=make_payload(
            "legacy-req",
            [make_read(rid) for rid in ("legacy-read", "legacy-2", "legacy-3")],
            sample_id="old-sample",
        ),
    )
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert replay.json() == legacy_client.legacy_snapshot
