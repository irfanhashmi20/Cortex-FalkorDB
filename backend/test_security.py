import pytest
from starlette.requests import Request
from fastapi import HTTPException
import main as m


def make_request(api_key=None):
    headers = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode()))

    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/reset",
        "raw_path": b"/api/reset",
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    })


def test_admin_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    with pytest.raises(HTTPException) as exc:
        m._verify_admin_api_key(make_request())

    assert exc.value.status_code == 403


def test_admin_rejects_wrong_api_key(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    with pytest.raises(HTTPException) as exc:
        m._verify_admin_api_key(make_request("wrong-key"))

    assert exc.value.status_code == 403


def test_admin_accepts_correct_api_key(monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_API_KEY", "test-secret-key")

    result = m._verify_admin_api_key(make_request("test-secret-key"))

    assert result is None


def test_invalid_tenant_is_rejected():
    with pytest.raises(HTTPException) as exc:
        m._validate_tenant_or_400("../../malicious-tenant")

    assert exc.value.status_code == 400
