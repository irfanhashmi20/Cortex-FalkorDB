from types import SimpleNamespace
from fastapi.testclient import TestClient
import main as m


def fake_graph(node_id, service_name):
    node = SimpleNamespace(
        id=node_id,
        labels=["Microservice"],
        properties={
            "name": service_name,
            "status": "healthy",
            "tier": 1,
            "ui_x": 100,
            "ui_y": 100,
        },
    )

    graph = SimpleNamespace()
    graph.query = lambda query: SimpleNamespace(
        result_set=[[node, None, None]]
    )
    return graph


def test_graph_api_selects_acme_tenant(monkeypatch):
    monkeypatch.setattr(
        m.gt,
        "get_graph",
        lambda tenant: fake_graph(101, "CheckoutService")
        if tenant == "acme_corp"
        else fake_graph(201, "PatientPortal"),
    )

    response = TestClient(m.app).get(
        "/api/graph",
        params={"tenant_id": "acme_corp"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "acme_corp"
    assert any(n["label"] == "CheckoutService" for n in data["nodes"])
    assert not any(n["label"] == "PatientPortal" for n in data["nodes"])


def test_graph_api_selects_beta_tenant(monkeypatch):
    monkeypatch.setattr(
        m.gt,
        "get_graph",
        lambda tenant: fake_graph(201, "PatientPortal")
        if tenant == "beta_health"
        else fake_graph(101, "CheckoutService"),
    )

    response = TestClient(m.app).get(
        "/api/graph",
        params={"tenant_id": "beta_health"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "beta_health"
    assert any(n["label"] == "PatientPortal" for n in data["nodes"])
    assert not any(n["label"] == "CheckoutService" for n in data["nodes"])


def test_graph_api_rejects_invalid_tenant():
    response = TestClient(m.app).get(
        "/api/graph",
        params={"tenant_id": "../../malicious-tenant"},
    )

    assert response.status_code == 400


def test_graph_api_uses_default_tenant(monkeypatch):
    requested_tenants = []

    def get_graph_spy(tenant):
        requested_tenants.append(tenant)
        return fake_graph(101, "CheckoutService")

    monkeypatch.setattr(m.gt, "get_graph", get_graph_spy)

    response = TestClient(m.app).get("/api/graph")

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "acme_corp"
    assert requested_tenants == ["acme_corp"]
