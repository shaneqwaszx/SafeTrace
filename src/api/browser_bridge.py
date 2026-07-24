"""Browser-to-loopback connector sessions for the hosted SafeTrace frontend."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any


CONNECTOR_PROTOCOL_VERSION = 1


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))


@dataclass(frozen=True)
class SessionValidation:
    valid: bool
    reason: str
    claims: dict[str, Any] | None = None


class BrowserBridgeSessionManager:
    """Issues HMAC-signed tokens that are invalidated whenever the backend restarts."""

    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self._secret = secrets.token_bytes(32)
        self.session_id = secrets.token_urlsafe(18)
        self.ttl_seconds = max(60, min(int(ttl_seconds), 86400))

    def issue(self, origin: str) -> tuple[str, dict[str, Any]]:
        now = int(time.time())
        claims = {
            "sid": self.session_id,
            "origin": origin,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "nonce": secrets.token_urlsafe(12),
            "protocol": CONNECTOR_PROTOCOL_VERSION,
        }
        payload = _encode(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        signature = _encode(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
        return f"{payload}.{signature}", claims

    def validate(self, token: str | None, origin: str) -> SessionValidation:
        if not token:
            return SessionValidation(False, "session_required")
        try:
            payload, signature = token.split(".", 1)
            expected = _encode(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                return SessionValidation(False, "invalid_session")
            claims = json.loads(_decode(payload))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return SessionValidation(False, "invalid_session")
        if claims.get("sid") != self.session_id:
            return SessionValidation(False, "backend_restarted")
        if claims.get("origin") != origin:
            return SessionValidation(False, "origin_mismatch")
        if claims.get("protocol") != CONNECTOR_PROTOCOL_VERSION:
            return SessionValidation(False, "protocol_mismatch")
        if int(claims.get("exp") or 0) <= int(time.time()):
            return SessionValidation(False, "session_expired")
        return SessionValidation(True, "ok", claims)
