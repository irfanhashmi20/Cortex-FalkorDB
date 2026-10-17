from fastapi.testclient import TestClient
import main as m


def test_reset_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    response = TestClient(m.app).post("/api/reset")

    assert response.status_code == 403


def test_reset_endpoint_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    response = TestClient(m.app).post(
        "/api/reset",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 403


def test_reset_endpoint_accepts_valid_key_without_real_reset(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    # Prevent any real database reseeding during this test.
    monkeypatch.setattr(m.gt, "get_db", lambda: object())
    monkeypatch.setattr(m.seed_graph, "seed_acme_corp", lambda *args: None)
    monkeypatch.setattr(m.seed_graph, "seed_beta_health", lambda *args: None)

    response = TestClient(m.app).post(
        "/api/reset",
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "reset_complete"


def test_graph_endpoint_rejects_invalid_tenant():
    response = TestClient(m.app).get(
        "/api/graph",
        params={"tenant_id": "../../malicious-tenant"},
    )

    assert response.status_code == 400
