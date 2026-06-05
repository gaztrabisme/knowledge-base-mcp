"""EPUB → clean markdown pipeline using markitdown."""

import re
from pathlib import Path

from markitdown import MarkItDown

from .config import DOCS_DIR, MARKDOWN_DIR
from . import manifest

_converter = MarkItDown()


def convert_epub(epub_path: Path) -> str:
    result = _converter.convert(str(epub_path))
    return result.text_content


def clean_markdown(text: str) -> str:
    # Strip markitdown metadata header (Title/Authors/Publisher/etc.)
    text = re.sub(r"^\*\*(?:Title|Authors?|Language|Publisher|Date|Description|Identifier)\*\*:.*\n", "", text, flags=re.MULTILINE)

    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Strip backslash escapes that markitdown adds
    text = text.replace("\\-", "-")
    text = text.replace("\\*", "*")
    text = text.replace("\\_", "_")
    text = text.replace("\\.", ".")
    text = text.replace("\\[", "[")
    text = text.replace("\\]", "]")

    text = re.sub(r"</?span[^>]*>", "", text)
    text = re.sub(r"</?div[^>]*>", "", text)

    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def strip_front_back_matter(text: str) -> str:
    lines = text.split("\n")

    # A bare "Chapter N"/numeric heading marks the first real chapter — but only
    # if it appears early. Some books name real chapters descriptively and reserve
    # "Chapter N" headings for a back-matter answer key; trusting a late match would
    # discard the whole book as front matter. Guard with a position cap.
    numeric_cap = int(len(lines) * 0.4)
    content_start = None
    for i, line in enumerate(lines):
        if i > numeric_cap:
            break
        if not re.match(r"^#{1,2}\s+", line):
            continue
        heading_text = re.sub(r"^#{1,6}\s+", "", line).strip()
        if re.match(r"^(\d+|chapter\s+\d+)$", heading_text, re.IGNORECASE):
            content_start = i
            break

    if content_start is None:
        for i, line in enumerate(lines):
            if not re.match(r"^#{1,2}\s+", line):
                continue
            if not _is_front_matter_heading(line):
                content_start = i
                break

    if content_start is None:
        content_start = 0

    content_end = len(lines)
    for i in range(len(lines) - 1, content_start, -1):
        if re.match(r"^#{1,2}\s+", lines[i]) and _is_back_matter_heading(lines[i]):
            content_end = i
            break

    return "\n".join(lines[content_start:content_end]).strip() + "\n"


def _is_front_matter_heading(line: str) -> bool:
    heading = re.sub(r"^#{1,6}\s+", "", line).strip().lower()
    front_patterns = [
        "table of contents", "contents", "copyright", "dedication",
        "about the author", "about the reviewer", "about the technical",
        "preface", "acknowledgment", "foreword", "contributor",
        "join our community", "discord", "landmarks", "cover",
        "who this book is for", "what this book covers",
        "what you need for this book", "conventions used",
        "get in touch", "download the example", "packt.link",
        "to get the most out", "sections", "reviews", "leave a review",
        "making the most out", "unlock your book", "how to unlock",
        "running a jupyter", "conventions", "code in action",
        "why subscribe", "packtpub", "packt.com", "subscribe",
    ]
    return any(p in heading for p in front_patterns)


def _is_back_matter_heading(line: str) -> bool:
    heading = re.sub(r"^#{1,6}\s+", "", line).strip().lower()
    # Exact matches: the book index ("Index") is huge page-number noise, but
    # "Index Scans"/"Indexing" mid-book must not trigger the cut.
    if heading in ("index", "subject index", "author index"):
        return True
    back_patterns = [
        "other books you may enjoy", "packt page",
        "download a free pdf",
    ]
    return any(p in heading for p in back_patterns)


def normalize_heading_levels(text: str) -> str:
    lines = text.split("\n")
    heading_levels = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+", line)
        if m:
            heading_levels.append(len(m.group(1)))

    if not heading_levels:
        return text

    min_level = min(heading_levels)
    if min_level == 1:
        return text

    shift = min_level - 1
    result = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code:
            m = re.match(r"^(#{1,6})(\s+.*)", line)
            if m:
                new_level = max(1, len(m.group(1)) - shift)
                line = "#" * new_level + m.group(2)
        result.append(line)
    return "\n".join(result)


def ingest_epub(epub_path: Path, output_path: Path) -> Path:
    text = convert_epub(epub_path)
    text = clean_markdown(text)
    text = strip_front_back_matter(text)
    text = normalize_heading_levels(text)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def convert_pdf_with_toc(pdf_path: Path) -> str | None:
    """Reconstruct markdown headings from a PDF's embedded TOC bookmarks.

    Returns None if the PDF has no usable TOC (caller should fall back to
    markitdown). The body text never contributes headings — we inject only
    the TOC entries and neutralize any body line that looks like a heading,
    so code comments like `# init` can't masquerade as structure.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()  # [[level, title, page], ...]; page is 1-based
    if not toc:
        doc.close()
        return None

    page_headings: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in toc:
        title = title.strip()
        if not title:
            continue
        page_headings.setdefault(page - 1, []).append((level, title))

    parts = []
    for page_index in range(doc.page_count):
        for level, title in page_headings.get(page_index, []):
            parts.append("#" * min(level, 3) + " " + title)
        body = str(doc.load_page(page_index).get_text("text"))
        # Neutralize body lines that would parse as markdown headings.
        body = re.sub(r"(?m)^(#{1,6}\s)", r" \1", body)
        parts.append(body)

    doc.close()
    return "\n\n".join(parts)


# Glyphs that should never appear in English technical PDFs. Some PDFs embed
# subset fonts with no ToUnicode CMap, so PyMuPDF maps glyphs to garbage in the
# Armenian/Hebrew/extended-Cyrillic ranges (e.g. "PostgreSQL" → "PostgreԯԭԨ",
# "2022" → "ҁѿҁ҂"). When that happens we lose the body text, so we discard the
# TOC headings and fall back to markitdown, which extracts these PDFs cleanly.
_GARBLE_RE = re.compile(r"[԰-֏֐-׿Ҁ-ӿ]")
_GARBLE_RATIO_THRESHOLD = 0.002  # clean PDFs ~0%; the one bad book ran ~0.88%


def _garble_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_GARBLE_RE.findall(text)) / len(text)


def ingest_pdf(pdf_path: Path, output_path: Path) -> Path:
    text = convert_pdf_with_toc(pdf_path)
    if text is not None and _garble_ratio(text) > _GARBLE_RATIO_THRESHOLD:
        # CID-font subset with no ToUnicode map: PyMuPDF text is corrupt. Headings
        # come from the TOC (clean) but the body is unusable — markitdown wins here.
        print(f"  PyMuPDF text garbled ({_garble_ratio(text):.2%}); falling back to markitdown: {pdf_path.stem}")
        text = None
    if text is None:
        # markitdown fallback (no TOC bookmarks or garbled body) — yields no
        # headings, so front-matter stripping would be a no-op; skip it.
        text = convert_epub(pdf_path)
        text = clean_markdown(text)
        text = normalize_heading_levels(text)
    else:
        text = clean_markdown(text)
        text = strip_front_back_matter(text)
        text = normalize_heading_levels(text)
    output_path.write_text(text, encoding="utf-8")
    return output_path


_SUFFIXES = {".epub", ".pdf", ".html", ".htm", ".md", ".markdown"}


def _ingest_html(source: Path, output: Path) -> None:
    text = convert_epub(source)  # markitdown handles HTML too
    text = clean_markdown(text)
    text = normalize_heading_levels(text)
    output.write_text(text, encoding="utf-8")


def _ingest_markdown(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = normalize_heading_levels(text)
    output.write_text(text, encoding="utf-8")


def ingest_all() -> list[Path]:
    """Auto-discover every supported document under DOCS_DIR (recursively) and
    convert it to markdown. One file = one book. The book's domain is the first
    path part relative to DOCS_DIR when nested in a subdir, else "general".

    Resumable: an existing markdown output is reused (skip conversion) but the
    book is still registered. The manifest is written once at the end.
    """
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    discovered: list[tuple[str, str]] = []  # (slug, domain)

    for file in sorted(DOCS_DIR.rglob("*")):
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        if suffix not in _SUFFIXES:
            continue

        slug = file.stem
        rel = file.relative_to(DOCS_DIR)
        domain = rel.parts[0] if len(rel.parts) > 1 else "general"
        output = MARKDOWN_DIR / f"{slug}.md"

        if output.exists():
            print(f"  Skipping (exists): {slug}")
            results.append(output)
            discovered.append((slug, domain))
            continue

        if suffix == ".epub":
            print(f"  Converting: {slug}")
            ingest_epub(file, output)
            results.append(output)
        elif suffix == ".pdf":
            print(f"  Converting PDF: {slug}")
            try:
                ingest_pdf(file, output)
                results.append(output)
            except Exception as e:
                print(f"  ERROR converting {slug}: {e}")
                if output.exists():
                    output.unlink()  # remove partial output
                continue
        elif suffix in (".html", ".htm"):
            print(f"  Converting HTML: {slug}")
            try:
                _ingest_html(file, output)
                results.append(output)
            except Exception as e:
                print(f"  ERROR converting {slug}: {e}")
                if output.exists():
                    output.unlink()
                continue
        else:  # .md / .markdown
            print(f"  Copying markdown: {slug}")
            _ingest_markdown(file, output)
            results.append(output)

        discovered.append((slug, domain))

    # Register all discovered books in one manifest write (preserve existing entries).
    m = manifest.load()
    for slug, domain in discovered:
        if slug not in m["books"]:
            m["books"][slug] = {
                "title": manifest._titleize(slug),
                "domain": domain,
                "rerank_input": "child",
            }
    manifest.save(m)

    return results


if __name__ == "__main__":
    print("Ingesting EPUBs → markdown...")
    paths = ingest_all()
    print(f"Done. {len(paths)} files in {MARKDOWN_DIR}")
