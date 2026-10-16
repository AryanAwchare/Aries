"""Control-plane authentication.

Mutating/dangerous endpoints (manual order placement, runtime config, leader
roster, setup, kill switch, copy-trading, re-evaluation) require an API key
**when ``OPS_API_KEY`` is configured**. The key is accepted either as
``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.

When ``OPS_API_KEY`` is unset the endpoints stay open so the bundled
single-user local dashboard keeps working; the app logs a CRITICAL warning at
startup. Always set ``OPS_API_KEY`` before exposing the control plane beyond
localhost.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from ..config import Settings


def extract_key(header_key: str | None, authorization: str | None) -> str | None:
    if header_key and header_key.strip():
        return header_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


async def require_ops_auth(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    settings: Settings = request.app.state.settings
    expected = (settings.ops_api_key or "").strip()
    if not expected:
        return
    supplied = extract_key(x_api_key, authorization)
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid control-plane API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
