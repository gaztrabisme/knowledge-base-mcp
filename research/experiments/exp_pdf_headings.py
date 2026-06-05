"""
Experiment: compare 4 approaches to recover heading hierarchy from PDFs.

Approaches:
  1. PyMuPDF get_toc() two-pass merge  — embed TOC bookmarks as markdown headings
  2. pymupdf4llm TocHeaders            — TOC-guided heading detection
  3. pymupdf4llm IdentifyHeaders       — font-size-based heading detection (default)
  4. Docling                           — layout-model heading detection

Usage:
    python scripts/exp_pdf_headings.py [--out-dir /tmp/pdf-exp-out]

Outputs per approach × book:
  - Markdown file at <out_dir>/approach<N>/<book>.md
  - Summary printed to stdout with heading counts and timing
"""

import argparse
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent.parent
BOOKS = {
    "postgresql-14-internals": BASE / "knowledge-base/databases/postgresql-14-internals.pdf",
    "boneh-shoup-applied-cryptography": BASE / "knowledge-base/security/boneh-shoup-applied-cryptography.pdf",
    "designing-distributed-systems": BASE / "knowledge-base/distributed-systems/designing-distributed-systems.pdf",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_headings(text: str) -> dict:
    """Count H1/H2/H3/total markdown headings in a text string."""
    h1 = len(re.findall(r"^# .+", text, re.MULTILINE))
    h2 = len(re.findall(r"^## .+", text, re.MULTILINE))
    h3 = len(re.findall(r"^### .+", text, re.MULTILINE))
    return {"h1": h1, "h2": h2, "h3": h3, "total": h1 + h2 + h3}


def first_headings(text: str, n: int = 30) -> list[str]:
    """Return the first N heading lines from a markdown text."""
    headings = []
    for line in text.splitlines():
        if re.match(r"^#{1,3} .+", line):
            headings.append(line)
            if len(headings) >= n:
                break
    return headings


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def save_markdown(text: str, out_dir: Path, approach: str, book: str) -> Path:
    d = out_dir / approach
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{book}.md"
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Approach 1: PyMuPDF get_toc() two-pass merge
# ---------------------------------------------------------------------------

def approach1_toc_merge(pdf_path: Path, out_dir: Path, book: str) -> dict:
    """
    Two-pass strategy:
      Pass 1 — call doc.get_toc() to get [(level, title, page), ...]
      Pass 2 — extract full page text; at the start of each TOC-entry page,
               inject a markdown heading of the right depth.

    Caveats:
      - Heading is inserted at page *start*, not at the exact line where the
        chapter title appears.  This is inherently approximate but cheap.
      - If a PDF has no embedded bookmarks, get_toc() returns [] and the output
        has zero headings (the function reports this clearly).
    """
    import fitz  # PyMuPDF

    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()  # [(level, title, page_1indexed), ...]
    has_toc = bool(toc)

    # Build page → [(level, title), ...] lookup
    page_headings: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in toc:
        page_headings.setdefault(page, []).append((level, title.strip()))

    lines = []
    for pno in range(len(doc)):
        page_1 = pno + 1
        # Inject TOC headings before this page's text
        for level, title in page_headings.get(page_1, []):
            prefix = "#" * min(level, 3)  # cap at H3 to match chunker's H1/H2/H3 use
            lines.append(f"{prefix} {title}")
            lines.append("")

        page = doc.load_page(pno)
        text = page.get_text("text")
        if text.strip():
            lines.append(text.rstrip())
            lines.append("")

    doc.close()
    elapsed = time.time() - t0

    markdown = "\n".join(lines)
    path = save_markdown(markdown, out_dir, "approach1", book)
    counts = count_headings(markdown)

    return {
        "approach": "1_toc_merge",
        "book": book,
        "has_embedded_toc": has_toc,
        "toc_entries": len(toc),
        "counts": counts,
        "elapsed_s": round(elapsed, 2),
        "out_path": str(path),
        "sample_headings": first_headings(markdown, 30),
    }


# ---------------------------------------------------------------------------
# Approach 2: pymupdf4llm TocHeaders
# ---------------------------------------------------------------------------

def approach2_toc_headers(pdf_path: Path, out_dir: Path, book: str) -> dict:
    """
    Uses pymupdf4llm.to_markdown with TocHeaders, which matches TOC titles
    to individual text spans as each page is rendered.  Produces richer
    inline placement than the two-pass merge (heading appears on the exact
    span, not at page start).

    If get_toc() returns nothing, TocHeaders will silently produce no headings.
    """
    import fitz
    import pymupdf4llm
    from pymupdf4llm.helpers.pymupdf_rag import TocHeaders

    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()
    has_toc = bool(toc)

    hdr = TocHeaders(doc)
    markdown = pymupdf4llm.to_markdown(doc, hdr_info=hdr)
    doc.close()
    elapsed = time.time() - t0

    path = save_markdown(markdown, out_dir, "approach2", book)
    counts = count_headings(markdown)

    return {
        "approach": "2_toc_headers",
        "book": book,
        "has_embedded_toc": has_toc,
        "toc_entries": len(toc),
        "counts": counts,
        "elapsed_s": round(elapsed, 2),
        "out_path": str(path),
        "sample_headings": first_headings(markdown, 30),
    }


# ---------------------------------------------------------------------------
# Approach 3: pymupdf4llm IdentifyHeaders (font-size, default)
# ---------------------------------------------------------------------------

def approach3_identify_headers(pdf_path: Path, out_dir: Path, book: str) -> dict:
    """
    Default pymupdf4llm behaviour: scans all page text to find the modal font
    size (body text), then maps larger sizes to H1-H6.  Works even on PDFs
    with no TOC bookmarks.

    Risk: running headers / footers that happen to use a larger font (e.g.
    chapter titles repeated in headers) may be misdetected.
    """
    import fitz
    import pymupdf4llm
    from pymupdf4llm.helpers.pymupdf_rag import IdentifyHeaders

    t0 = time.time()
    doc = fitz.open(str(pdf_path))

    hdr = IdentifyHeaders(doc)
    markdown = pymupdf4llm.to_markdown(doc, hdr_info=hdr)
    doc.close()
    elapsed = time.time() - t0

    path = save_markdown(markdown, out_dir, "approach3", book)
    counts = count_headings(markdown)

    return {
        "approach": "3_identify_headers",
        "book": book,
        "counts": counts,
        "elapsed_s": round(elapsed, 2),
        "out_path": str(path),
        "sample_headings": first_headings(markdown, 30),
    }


# ---------------------------------------------------------------------------
# Approach 4: Docling
# ---------------------------------------------------------------------------

def approach4_docling(pdf_path: Path, out_dir: Path, book: str) -> dict:
    """
    IBM Docling uses a deep learning layout model (DocLayNet / RT-DETR-v2) to classify
    document regions (paragraph, heading, table, figure, …) and reconstructs
    a logical document hierarchy.  export_to_markdown() emits standard #/##/###
    headings that the chunker can consume directly.

    First call per process downloads ~400 MB of model weights to ~/.cache/docling.

    KNOWN FAILURE on Apple Silicon (MPS): docling's RT-DETR-v2 layout model calls
    torch.arange(..., dtype=torch.float64) which MPS does not support.
    PYTORCH_ENABLE_MPS_FALLBACK=1 does NOT help because the error is a hard dtype
    conversion, not a missing op.  Workaround: run on Linux/CUDA or set
    TORCH_DEVICE=cpu before import (not yet exposed as a docling API kwarg in
    current releases).  On the SSH GPU box (RTX 5080, Linux) this would work fine.
    """
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from docling.document_converter import DocumentConverter

    t0 = time.time()
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    elapsed = time.time() - t0

    path = save_markdown(markdown, out_dir, "approach4", book)
    counts = count_headings(markdown)

    return {
        "approach": "4_docling",
        "book": book,
        "counts": counts,
        "elapsed_s": round(elapsed, 2),
        "out_path": str(path),
        "sample_headings": first_headings(markdown, 30),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

APPROACH_LABELS = {
    "1_toc_merge": "Approach 1: get_toc() two-pass merge",
    "2_toc_headers": "Approach 2: pymupdf4llm TocHeaders",
    "3_identify_headers": "Approach 3: pymupdf4llm IdentifyHeaders",
    "4_docling": "Approach 4: Docling",
}

BOOK_SHORT = {
    "postgresql-14-internals": "postgresql",
    "boneh-shoup-applied-cryptography": "boneh-shoup",
    "designing-distributed-systems": "dist-sys",
}


def print_result(r: dict) -> None:
    label = APPROACH_LABELS.get(r["approach"], r["approach"])
    book = BOOK_SHORT.get(r["book"], r["book"])
    print_section(f"{label}  |  {book}")

    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return

    c = r["counts"]
    print(f"  Headings: H1={c['h1']}  H2={c['h2']}  H3={c['h3']}  total={c['total']}")
    print(f"  Elapsed: {r['elapsed_s']}s")
    if "has_embedded_toc" in r:
        print(f"  Embedded TOC: {r['has_embedded_toc']}  ({r.get('toc_entries', 0)} entries)")
    print(f"  Output: {r['out_path']}")
    print(f"\n  --- First {len(r['sample_headings'])} headings ---")
    for h in r["sample_headings"]:
        print(f"    {h}")


def print_summary_table(results: list[dict]) -> None:
    print_section("SUMMARY TABLE")
    header = f"{'Approach':<32} {'Book':<12} {'H1':>4} {'H2':>4} {'H3':>5} {'Total':>6} {'Time':>6}  Notes"
    print(header)
    print("-" * len(header))
    for r in results:
        label = r["approach"]
        book = BOOK_SHORT.get(r["book"], r["book"])
        if "error" in r:
            print(f"  {label:<30} {book:<12} {'ERROR':>31}   {r['error'][:50]}")
            continue
        c = r["counts"]
        toc_note = ""
        if "has_embedded_toc" in r:
            toc_note = f"TOC={'yes' if r['has_embedded_toc'] else 'NO'} ({r.get('toc_entries',0)} entries)"
        print(f"  {label:<30} {book:<12} {c['h1']:>4} {c['h2']:>4} {c['h3']:>5} {c['total']:>6} {r['elapsed_s']:>5}s  {toc_note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_approach(fn, pdf_path, out_dir, book, label):
    print(f"\n[Running] {label}  /  {book} ...", flush=True)
    try:
        return fn(pdf_path, out_dir, book)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"  FAILED: {exc}")
        print(tb[:600])
        return {"approach": label.split(":")[0].strip().lower().replace(" ", "_"),
                "book": book, "error": str(exc), "counts": {}, "elapsed_s": 0,
                "sample_headings": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/pdf-exp-out", type=Path)
    parser.add_argument(
        "--approaches", default="1,2,3,4",
        help="Comma-separated list of approach numbers to run (default: 1,2,3,4)"
    )
    parser.add_argument(
        "--books", default=None,
        help="Comma-separated book slugs to run (default: all 3)"
    )
    args = parser.parse_args()

    active_approaches = set(args.approaches.split(","))
    active_books = set(args.books.split(",")) if args.books else set(BOOKS.keys())

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    approach_fns = {
        "1": (approach1_toc_merge, "Approach 1: get_toc() two-pass merge", "1_toc_merge"),
        "2": (approach2_toc_headers, "Approach 2: pymupdf4llm TocHeaders", "2_toc_headers"),
        "3": (approach3_identify_headers, "Approach 3: pymupdf4llm IdentifyHeaders", "3_identify_headers"),
        "4": (approach4_docling, "Approach 4: Docling", "4_docling"),
    }

    all_results = []
    for book_slug, pdf_path in BOOKS.items():
        if book_slug not in active_books:
            continue
        for num in sorted(active_approaches):
            if num not in approach_fns:
                print(f"Unknown approach: {num}")
                continue
            fn, label, approach_key = approach_fns[num]
            result = run_approach(fn, pdf_path, out_dir, book_slug, label)
            # Ensure approach key is set correctly
            result["approach"] = approach_key
            all_results.append(result)
            print_result(result)

    print_summary_table(all_results)
    print("\nDone. Inspect markdown samples in:", out_dir)


if __name__ == "__main__":
    main()
