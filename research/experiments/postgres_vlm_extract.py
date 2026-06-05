"""Re-extract postgresql-14-internals via a local VLM, page by page.

The PDF embeds subset fonts with no ToUnicode CMap for specific glyphs, so PyMuPDF
maps high-value tokens to Armenian/Cyrillic garbage ("PostgreSQL"->"PostgreԯԭԨ",
"WAL"->"ԳԝԨ", digits/versions -> noise). Only ~0.3% of chars, but exactly the search
terms. No text extractor can recover them (the mapping isn't in the file) — only
OCR/vision. We rasterise each garble-bearing page and have the local Qwen3.6 VLM
transcribe it; clean pages keep their (correct) native PyMuPDF text.

Headings come from the PDF's embedded TOC (clean), injected at page boundaries — same
reconstruction as ingest.convert_pdf_with_toc. Output overwrites the markdown so the
normal chunk step picks it up. Resumable: every page's text is cached to disk, so an
interrupted run continues where it left off.

Usage: python3 scripts/postgres_vlm_extract.py
"""

import base64
import json
import pathlib
import sys
import time

import fitz  # PyMuPDF
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from knowledge_base.ingest import _GARBLE_RE, clean_markdown, normalize_heading_levels, strip_front_back_matter

PDF = ROOT / "knowledge-base" / "databases" / "postgresql-14-internals.pdf"
OUT = ROOT / "data" / "markdown" / "postgresql-14-internals.md"
CACHE = ROOT / "data" / "postgres_vlm_cache.json"

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3.6-35b-a3b-mlx"
DPI = 150
PROMPT = (
    "Transcribe this book page to clean markdown. Output ONLY the transcribed text — "
    "no commentary, no ``` fences around the whole page. Preserve code, SQL, and command "
    "output verbatim inside fenced code blocks. Preserve inline values, version numbers, "
    "and identifiers exactly. Ignore page headers/footers and bare page numbers."
)


def _load_cache() -> dict[str, str]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _vlm_page(client: OpenAI, png: bytes) -> str:
    b64 = base64.b64encode(png).decode()
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=2000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 3:
                raise
            print(f"    VLM retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(3)
    return ""


def main() -> None:
    doc = fitz.open(str(PDF))
    toc = doc.get_toc()  # [[level, title, page(1-based)], ...]
    page_headings: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in toc:
        if title.strip():
            page_headings.setdefault(page - 1, []).append((level, title.strip()))

    cache = _load_cache()
    client = OpenAI(base_url=BASE_URL, api_key="lm-studio", timeout=240.0)

    n_vlm = n_native = n_cached = 0
    t0 = time.time()
    for i in range(doc.page_count):
        key = str(i)
        if key in cache:
            n_cached += 1
            continue
        native = doc.load_page(i).get_text("text")
        if not _GARBLE_RE.findall(native):
            cache[key] = native
            n_native += 1
            continue
        # garble-bearing page -> VLM transcription
        png = doc.load_page(i).get_pixmap(dpi=DPI).tobytes("png")
        cache[key] = _vlm_page(client, png)
        n_vlm += 1
        CACHE.write_text(json.dumps(cache))  # checkpoint after each (slow) VLM page
        done = n_vlm + n_native + n_cached
        rate = n_vlm / max(time.time() - t0, 1)
        print(f"  [{done}/{doc.page_count}] VLM page {i}  ({n_vlm} vlm, {rate*60:.1f}/min)", flush=True)

    CACHE.write_text(json.dumps(cache))
    print(f"\nExtraction done: {n_vlm} VLM, {n_native} native, {n_cached} cached", flush=True)

    # Reconstruct markdown with TOC headings injected at page boundaries.
    parts: list[str] = []
    for i in range(doc.page_count):
        for level, title in page_headings.get(i, []):
            parts.append("#" * min(level, 3) + " " + title)
        body = cache[str(i)]
        # neutralise body lines that would parse as markdown headings (VLM may emit #)
        body = "\n".join((" " + ln if ln[:1] == "#" else ln) for ln in body.split("\n"))
        parts.append(body)
    doc.close()

    text = "\n\n".join(parts)
    text = clean_markdown(text)
    text = strip_front_back_matter(text)
    text = normalize_heading_levels(text)
    OUT.write_text(text, encoding="utf-8")

    garble = len(_GARBLE_RE.findall(text)) / max(len(text), 1)
    print(f"Wrote {OUT}  ({len(text)//1024} KB, residual garble {garble*100:.3f}%)", flush=True)


if __name__ == "__main__":
    main()
