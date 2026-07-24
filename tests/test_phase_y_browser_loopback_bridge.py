from pathlib import Path

from fastapi.testclient import TestClient

from src.api.batches import BatchStore
import src.api.browser_bridge as bridge_module
from src.api.browser_bridge import BrowserBridgeSessionManager
from src.api.jobs import JobStore
from src.api.server import BRIDGE_SESSION_HEADER, create_app


VERCEL_ORIGIN = "https://safetrace-iota.vercel.app"
LOCAL_ORIGIN = "http://127.0.0.1:5173"


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("SAFETRACE_ALLOWED_FRONTEND_ORIGINS", f"{VERCEL_ORIGIN},{LOCAL_ORIGIN}")
    return TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))


def _bootstrap(client: TestClient, origin: str = VERCEL_ORIGIN) -> dict:
    response = client.post("/api/bridge/bootstrap", headers={"Origin": origin})
    assert response.status_code == 200
    return response.json()


def test_health_advertises_loopback_connector_contract(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/health", headers={"Origin": VERCEL_ORIGIN})
    assert response.status_code == 200
    assert response.json()["api"] == "safetrace-local"
    assert response.json()["bridgeSupported"] is True
    assert response.json()["connectorProtocolVersion"] == 1
    assert response.json()["runtimeProfile"]


def test_private_network_preflight_is_exact_origin_and_header_allowlisted(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.options(
        "/api/analyze",
        headers={
            "Origin": VERCEL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-safetrace-session",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == VERCEL_ORIGIN
    assert response.headers["access-control-allow-private-network"] == "true"
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "x-safetrace-session" in allowed_headers
    assert "*" not in allowed_headers


def test_arbitrary_browser_origin_cannot_read_or_bootstrap(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    headers = {"Origin": "https://attacker.example"}
    assert client.get("/api/health", headers=headers).status_code == 403
    assert client.post("/api/bridge/bootstrap", headers=headers).status_code == 403
    response = client.options(
        "/api/analyze",
        headers={
            **headers,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.status_code in {400, 403}
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-private-network" not in response.headers


def test_wildcard_and_null_allowlist_entries_are_never_authorized(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SAFETRACE_ALLOWED_FRONTEND_ORIGINS",
        f"{VERCEL_ORIGIN},*,null",
    )
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))
    assert client.post(
        "/api/bridge/bootstrap",
        headers={"Origin": VERCEL_ORIGIN},
    ).status_code == 200
    assert client.post(
        "/api/bridge/bootstrap",
        headers={"Origin": "https://attacker.example"},
    ).status_code == 403
    assert client.post(
        "/api/bridge/bootstrap",
        headers={"Origin": "null"},
    ).status_code == 403


def test_null_origin_and_simple_form_csrf_are_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    null_origin = client.post(
        "/api/system/vlm/settings",
        data={"selectedProfile": "rule_based", "enabled": "false"},
        headers={"Origin": "null"},
    )
    assert null_origin.status_code == 403
    malicious_form = client.post(
        "/api/system/vlm/settings",
        data={"selectedProfile": "rule_based", "enabled": "false"},
        headers={"Origin": "https://attacker.example", "Content-Type": "application/x-www-form-urlencoded"},
    )
    assert malicious_form.status_code == 403


def test_allowed_browser_requires_startup_scoped_session_for_api(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    payload = {"selectedProfile": "rule_based", "enabled": False}
    missing = client.post("/api/system/vlm/settings", json=payload, headers={"Origin": VERCEL_ORIGIN})
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "session_required"

    bridge = _bootstrap(client)
    accepted = client.post(
        "/api/system/vlm/settings",
        json=payload,
        headers={"Origin": VERCEL_ORIGIN, BRIDGE_SESSION_HEADER: bridge["sessionToken"]},
    )
    assert accepted.status_code == 200


def test_bridge_token_is_origin_bound_and_invalid_after_backend_restart(tmp_path, monkeypatch):
    first = _client(tmp_path / "first", monkeypatch)
    bridge = _bootstrap(first)
    origin_mismatch = first.get(
        "/api/system/status",
        headers={"Origin": LOCAL_ORIGIN, BRIDGE_SESSION_HEADER: bridge["sessionToken"]},
    )
    assert origin_mismatch.status_code == 401
    assert origin_mismatch.json()["detail"]["code"] == "origin_mismatch"

    second = _client(tmp_path / "second", monkeypatch)
    restarted = second.get(
        "/api/system/status",
        headers={"Origin": VERCEL_ORIGIN, BRIDGE_SESSION_HEADER: bridge["sessionToken"]},
    )
    assert restarted.status_code == 401
    assert restarted.json()["detail"]["code"] in {"invalid_session", "backend_restarted"}


def test_expired_session_is_rejected(monkeypatch):
    manager = BrowserBridgeSessionManager(ttl_seconds=60)
    token, _ = manager.issue(VERCEL_ORIGIN)
    monkeypatch.setattr(bridge_module.time, "time", lambda: 10**12)
    validation = manager.validate(token, VERCEL_ORIGIN)
    assert validation.valid is False
    assert validation.reason == "session_expired"


def test_no_origin_cli_contract_remains_compatible(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "rule_based", "enabled": False},
    )
    assert response.status_code == 200


def test_frontend_uses_one_fetch_implementation_and_loopback_address_space():
    frontend = Path("frontend-react/src")
    fetch_users = []
    for path in frontend.rglob("*"):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        if "fetch(" in path.read_text(encoding="utf-8"):
            fetch_users.append(path.as_posix())
    assert fetch_users == ["frontend-react/src/services/apiClient.ts"]
    client_source = (frontend / "services/apiClient.ts").read_text(encoding="utf-8")
    assert "targetAddressSpace = 'loopback'" in client_source
    assert "sessionStorage" in client_source
    assert "localStorage" not in client_source
    assert "non_loopback_api_base" in client_source
    assert "safetrace-session-expired" in client_source


def test_vercel_config_and_launcher_keep_backend_loopback_only():
    vercel = Path("frontend-react/vercel.json").read_text(encoding="utf-8")
    launcher = Path("START_SAFETRACE_VERCEL.bat").read_text(encoding="utf-8")
    assert "connect-src 'self' http://127.0.0.1:8000 http://localhost:8000" in vercel
    assert "https://safetrace-iota.vercel.app" in launcher
    assert "--host 127.0.0.1" in launcher
    assert "--host 0.0.0.0" not in launcher
    assert "SAFETRACE_ALLOWED_FRONTEND_ORIGINS" in launcher
    assert "SAFETRACE_CONNECTOR_SESSION_TTL_SECONDS" in launcher
