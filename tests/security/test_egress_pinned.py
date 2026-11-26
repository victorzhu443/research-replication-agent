"""P4 egress and the zip budget, exercised end to end and offline.

Each fetcher is driven through the REAL `secprim.safe_fetch` with an injected
resolver (every name is public) and an `httpx.MockTransport` that plays a hostile
server: a redirect off the allow-listed host, or an over-cap body. The fetcher must
refuse. These are the regressions for `security/mutations.yaml` P4-* and
P2-french-zip-unbudgeted, which the first mutation run showed nothing caught.
"""
from __future__ import annotations

import io
import random
import zipfile

import httpx
import pytest
from secprim import MalformedInput
from secprim.safe_fetch import UnsafeDestination, safe_fetch

from replicator.data.adapters import french
from replicator.spec import intake

PUBLIC = [(2, 1, 6, "", ("93.184.216.34", 0))]  # getaddrinfo shape; any public address


def _public_resolver(host, port):
    return PUBLIC


def _wire(monkeypatch, module, handler):
    """Make `module.safe_fetch` the real safe_fetch with a hostile transport and an
    offline resolver. The module's own keyword arguments (allow_hosts, max_bytes, ...)
    pass through untouched, so what is under test is the call site."""
    transport = httpx.MockTransport(handler)

    def wired(url, **kw):
        return safe_fetch(url, transport=transport, resolver=_public_resolver, **kw)

    monkeypatch.setattr(module, "safe_fetch", wired)


def _redirect_off_host(good_host: str, body: bytes = b"%PDF-1.4 evil"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Host", "").split(":")[0] == good_host:
            return httpx.Response(302, headers={"Location": "https://evil.example/anything"})
        return httpx.Response(200, content=body)
    return handler


# ---------------------------------------------------------------- intake

def test_fetch_arxiv_refuses_a_redirect_off_arxiv(tmp_path, monkeypatch):
    _wire(monkeypatch, intake, _redirect_off_host("arxiv.org"))
    with pytest.raises(UnsafeDestination):
        intake.fetch_arxiv("2401.00001", tmp_path)
    assert not list(tmp_path.glob("*.pdf")), "an off-list body was written to disk"


def test_fetch_html_refuses_a_body_over_the_cap(tmp_path, monkeypatch):
    big = b"<p>" + b"a" * (intake.MAX_HTML_BYTES + 1) + b"</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big)

    _wire(monkeypatch, intake, handler)
    with pytest.raises(MalformedInput):
        intake.fetch_html_as_text("https://papers.example/x.html", tmp_path, "x")
    assert not (tmp_path / "x.txt").exists()


def test_fetch_html_under_the_cap_still_works(tmp_path, monkeypatch):
    _wire(monkeypatch, intake, lambda r: httpx.Response(200, content=b"<h1>ok</h1>"))
    p = intake.fetch_html_as_text("https://papers.example/x.html", tmp_path, "x")
    assert p.read_text().strip() == "ok"


# ---------------------------------------------------------------- French data library

def test_french_download_refuses_a_redirect_off_the_library_host(monkeypatch):
    _wire(monkeypatch, french, _redirect_off_host("mba.tuck.dartmouth.edu", body=b"PK\x03\x04"))
    with pytest.raises(UnsafeDestination):
        french._download("ff3")


def _zip_with(member: bytes, name: str = "F-F_Research_Data_Factors.CSV") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(name, member)
    return buf.getvalue()


def test_french_download_refuses_a_member_over_the_declared_size_budget(monkeypatch):
    payload = random.Random(0).randbytes(5_000)  # incompressible: the size check, not the ratio check
    _wire(monkeypatch, french, lambda r: httpx.Response(200, content=_zip_with(payload)))
    monkeypatch.setattr(french, "MAX_MEMBER_BYTES", 1_000)
    with pytest.raises(MalformedInput):
        french._download("ff3")


def test_french_download_reads_a_member_inside_the_budget(monkeypatch):
    _wire(monkeypatch, french, lambda r: httpx.Response(200, content=_zip_with(b"hello,world\n")))
    assert french._download("ff3") == "hello,world\n"
