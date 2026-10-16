"""Credential redaction in logs (Myfxbook URLs must never leak secrets)."""
from __future__ import annotations

import logging

from app.logging import _RedactingFilter, scrub_secrets


def test_scrub_myfxbook_login_url():
    line = (
        "HTTP Request: GET https://www.myfxbook.com/api/login.json"
        "?email=dub%40gmail.com&password=Sup3rSecret HTTP/1.1 200 OK"
    )
    out = scrub_secrets(line)
    assert "Sup3rSecret" not in out
    assert "dub%40gmail.com" not in out
    assert "password=***" in out
    assert "email=***" in out


def test_scrub_session_token():
    out = scrub_secrets("?session=JG6Ww1uW%3D%3D&id=10515575")
    assert "JG6Ww1uW" not in out
    assert "session=***" in out
    # non-secret params are preserved
    assert "id=10515575" in out


def test_filter_rewrites_message():
    rec = logging.LogRecord(
        "httpx", logging.INFO, __file__, 1, "GET /x?password=Secret1", None, None
    )
    assert _RedactingFilter().filter(rec) is True
    assert "Secret1" not in rec.getMessage()


def test_filter_rewrites_tuple_and_dict_args():
    rec = logging.LogRecord(
        "httpx", logging.INFO, __file__, 1, "url=%s", ("?token=abc&keep=1",), None
    )
    _RedactingFilter().filter(rec)
    msg = rec.getMessage()
    assert "abc" not in msg.replace("token=***", "")
    assert "keep=1" in msg

    rec2 = logging.LogRecord("x", logging.INFO, __file__, 1, "url=%(url)s", None, None)
    rec2.args = {"url": "?password=sekret&keep=ok"}
    _RedactingFilter().filter(rec2)
    msg2 = rec2.getMessage()
    assert "sekret" not in msg2
    assert "password=***" in msg2
    assert "keep=ok" in msg2
