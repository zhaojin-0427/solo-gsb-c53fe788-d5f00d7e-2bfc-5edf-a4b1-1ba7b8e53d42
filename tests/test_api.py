"""End-to-end API tests: idempotency, conflicts, immutability."""
from fastapi.testclient import TestClient

PAYLOAD = {
    "request_id": "req-1",
    "sample_id": "sample-A",
    "reads": [
        {
            "locus": "IGH-A",
            "read_id": f"r{i}",
            "umi": "AAAAA",
            "sequence": "ACGT",
            "qualities": [30, 30, 30, 30],
        }
        for i in range(4)
    ],
}


def test_post_creates_version_and_get_is_identical(client: TestClient):
    resp = client.post("/consensus", json=PAYLOAD)
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Idempotent-Replay"] == "false"
    body = resp.json()
    assert body["version"] == 1
    assert body["consensus"][0]["sequence"] == "ACGT"
    assert body["request_id"] == "req-1"

    got = client.get(f"/consensus/{body['version']}")
    assert got.status_code == 200
    assert got.json() == body


def test_same_request_same_payload_is_idempotent(client: TestClient):
    first = client.post("/consensus", json=PAYLOAD)
    assert first.status_code == 200

    # Reorder reads: still the same canonical payload.
    reordered = dict(PAYLOAD, reads=list(reversed(PAYLOAD["reads"])))
    second = client.post("/consensus", json=reordered)
    assert second.status_code == 200
    assert second.headers["X-Idempotent-Replay"] == "true"
    assert second.json() == first.json()

    assert client.get("/consensus/2").status_code == 404  # no version 2
    assert client.get("/healthz").json()["latest_version"] == 1


def test_same_request_different_payload_is_409(client: TestClient):
    assert client.post("/consensus", json=PAYLOAD).status_code == 200
    changed = {
        **PAYLOAD,
        "reads": [
            {**PAYLOAD["reads"][0], "sequence": "ACGG"},
            *PAYLOAD["reads"][1:],
        ],
    }
    resp = client.post("/consensus", json=changed)
    assert resp.status_code == 409
    assert "different payload" in resp.json()["detail"]["error"]


def test_readid_content_conflict_is_409(client: TestClient):
    payload = {
        "request_id": "req-2",
        "sample_id": "s",
        "reads": [
            {
                "locus": "L1",
                "read_id": "dup",
                "umi": "AAAAA",
                "sequence": "ACGT",
                "qualities": [30] * 4,
            },
            {
                "locus": "L1",
                "read_id": "dup",
                "umi": "AAAAT",
                "sequence": "ACGT",
                "qualities": [30] * 4,
            },
            {
                "locus": "L1",
                "read_id": "other",
                "umi": "AAAAA",
                "sequence": "ACGT",
                "qualities": [30] * 4,
            },
        ],
    }
    resp = client.post("/consensus", json=payload)
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflicts"][0]["read_id"] == "dup"


def test_identical_duplicate_rows_are_allowed(client: TestClient):
    read = {
        "locus": "L1",
        "read_id": "r1",
        "umi": "AAAAA",
        "sequence": "ACGT",
        "qualities": [30] * 4,
    }
    payload = {
        "request_id": "req-3",
        "sample_id": "s",
        "reads": [dict(read), dict(read)],
    }
    resp = client.post("/consensus", json=payload)
    assert resp.status_code == 200
    assert resp.json()["stats"]["duplicate_read_rows"] == 1


def test_validation_errors_are_422(client: TestClient):
    bad = {
        "request_id": "req-4",
        "sample_id": "s",
        "reads": [
            {
                "locus": "L1",
                "read_id": "r1",
                "umi": "AAAAA",
                "sequence": "ACGT",
                "qualities": [30, 30],  # length mismatch
            }
        ],
    }
    assert client.post("/consensus", json=bad).status_code == 422

    bad_q = {
        "request_id": "req-5",
        "sample_id": "s",
        "reads": [
            {
                "locus": "L1",
                "read_id": "r1",
                "umi": "AAAAA",
                "sequence": "ACGT",
                "qualities": [30, 50, 30, 30],  # out of range
            }
        ],
    }
    assert client.post("/consensus", json=bad_q).status_code == 422


def test_versions_are_immutable_and_independent(client: TestClient):
    v1 = client.post("/consensus", json=PAYLOAD).json()
    second_payload = {
        "request_id": "req-2",
        "sample_id": "sample-B",
        "reads": [
            {
                "locus": "IGH-B",
                "read_id": f"b{i}",
                "umi": "TTTTT",
                "sequence": "TTTT",
                "qualities": [40] * 4,
            }
            for i in range(3)
        ],
    }
    v2 = client.post("/consensus", json=second_payload).json()
    assert v2["version"] == 2

    # Re-reading v1 must still return the original snapshot.
    assert client.get("/consensus/1").json() == v1
    assert client.get("/consensus/2").json() == v2
    assert client.get("/consensus/999").status_code == 404
