"""Intake: PDF -> per-page PNGs + per-page text. The PDF itself is the source of truth for the
spec call; the derived artifacts serve companion discovery, claim evidence, and the re-reader."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from secprim.safe_fetch import safe_fetch

MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_HTML_BYTES = 2 * 1024 * 1024  # a paper page is far under this


def _checked(r, url: str):
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} fetching {url}")
    return r


def fetch_arxiv(arxiv_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    # Pinned, egress-validated fetch: resolves once, connects to the validated address, follows
    # redirects hop by hop within the arxiv.org allow-list, and caps the body.
    r = _checked(safe_fetch(url, allow_hosts=["arxiv.org"], max_bytes=MAX_PDF_BYTES, timeout=120), url)
    p = dest / f"{arxiv_id.replace('/', '_')}.pdf"
    p.write_bytes(r.content)
    return p


# Linear-time stripping (RC-2). `<[^<>]+>` cannot straddle tags, so it is linear; script/style
# blocks are removed by a single forward scan (find opener, find its closer, jump past it) because
# any regex that must pair an opener with a closer re-scans to the end from every unterminated
# opener and goes quadratic on hostile markup.
_TAG = re.compile(r"<[^<>]+>")
_SS_OPEN = re.compile(r"<(script|style)\b", re.I)
_NBSP = re.compile(r"&nbsp;")
_HSPACE = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n[ \t\r\f\v]*\n+")


def _strip_script_style(html: str) -> str:
    out: list[str] = []
    low = html.lower()
    i = 0
    while True:
        m = _SS_OPEN.search(html, i)
        if m is None:
            out.append(html[i:])
            break
        out.append(html[i:m.start()])
        close = low.find(f"</{m.group(1).lower()}>", m.end())
        if close < 0:  # unterminated block: everything after it is script, drop it
            break
        i = close + len(m.group(1)) + 3
    return " ".join(out)


def html_to_text(html: str) -> str:
    """Strip tags to text. Untrusted input; the whole path is cost-swept in tests/security."""
    text = _TAG.sub(" ", _strip_script_style(html))
    text = _HSPACE.sub(" ", _NBSP.sub(" ", text))
    return _BLANKS.sub("\n\n", text)


def fetch_html_as_text(url: str, dest: Path, name: str) -> Path:
    """HTML-only papers: strip tags to text; no page images, so claim re-reads are skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    r = _checked(safe_fetch(url, max_bytes=MAX_HTML_BYTES, timeout=120, headers={"User-Agent": "Mozilla/5.0"}), url)
    p = dest / f"{name}.txt"
    p.write_text(html_to_text(r.text))
    return p


def render(pdf: Path, out: Path, dpi: int = 110) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    known = pdf.stem if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", pdf.stem) else None  # fetch_arxiv names files by id
    if pdf.suffix.lower() != ".pdf":
        full = pdf.read_text()
        (out / "paper.txt").write_text(full)
        return {"n_pages": 0, "pages": [], "companions": find_companions(full), "arxiv_version": _arxiv_version(full, known)}
    doc = pymupdf.open(pdf)
    pages = []
    text_all = []
    for i, page in enumerate(doc):
        png = out / f"p{i+1:03d}.png"
        if not png.exists():
            page.get_pixmap(dpi=dpi).save(png)
        txt = page.get_text()
        (out / f"p{i+1:03d}.txt").write_text(txt)
        pages.append({"page": i + 1, "png": str(png), "chars": len(txt)})
        text_all.append(txt)
    full = "\n".join(text_all)
    (out / "paper.txt").write_text(full)
    return {"n_pages": len(pages), "pages": pages, "companions": find_companions(full),
            "arxiv_version": _arxiv_version(full, known)}


_LINK = re.compile(r"https?://(?:github\.com|gitlab\.com|zenodo\.org|osf\.io|huggingface\.co)/[\w\-./]+")


def find_companions(text: str) -> list[str]:
    return sorted(set(m.rstrip(".)") for m in _LINK.findall(text)))


def _arxiv_version(text: str, known_id: str | None = None) -> str | None:
    """If the paper was fetched by arXiv id, only the version suffix is read from the text (the
    margin stamp 'arXiv:ID vN'); guessing the id from the body picks up citations."""
    if known_id:
        m = re.search(re.escape(known_id) + r"(v\d+)?", text)
        return known_id + (m.group(1) if (m and m.group(1)) else "")
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?\s*\[[\w.\-]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}", text)
    if not m:
        m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", text)
    return (m.group(1) + (m.group(2) or "")) if m else None
