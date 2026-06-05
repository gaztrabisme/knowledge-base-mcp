"""FastMCP server exposing semantic search over ML books."""

import json
import re
import subprocess

from fastmcp import FastMCP

from .config import CHUNKS_DIR, MARKDOWN_DIR
from . import manifest
from .search import get_client, hybrid_search

_CHUNK_ID_RE = re.compile(r"^(.+)-p(\d+)-c(\d+)$")

mcp = FastMCP(
    "Developer Knowledge Base",
    instructions="Semantic search over technical books: ML, databases, security, distributed systems",
)


@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    book: str | None = None,
    hyde_passage: str | None = None,
    extra_queries: list[str] | None = None,
    bm25_keywords: list[str] | None = None,
) -> list[dict]:
    """Search across all indexed books using hybrid semantic + keyword search.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 10).
        book: Optional book slug to filter results to a single book.
        hyde_passage: Optional hypothetical answer paragraph. When provided, this
            text is embedded instead of the query for the dense retrieval leg,
            bridging vocabulary gaps between questions and textbook prose. The
            BM25 keyword leg still uses the original query (or bm25_keywords).
        extra_queries: Optional list of alternative query phrasings to broaden
            recall. Each extra query adds its own dense + BM25 retrieval legs,
            and all candidates are fused before reranking.
        bm25_keywords: Optional list of exact terms for the BM25 keyword leg
            (e.g. ["MVCC", "xmin", "snapshot isolation"]). When provided, these
            terms drive lexical matching instead of the natural-language query,
            separating semantic intent (dense leg) from exact-term matching.
            Use when you know the precise vocabulary the source text would use.

    Returns:
        List of search results with score, book, chapter, section, content, and matching child chunk.
    """
    # ⚡2: reuse the long-lived client singleton (no per-call open/close).
    client = get_client()
    results = hybrid_search(
        query=query,
        client=client,
        top_k=top_k,
        book_filter=book,
        hyde_passage=hyde_passage,
        extra_queries=extra_queries,
        bm25_keywords=bm25_keywords,
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def search_book(
    book: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Search within a specific book.

    Args:
        book: Book slug (use list_books() to see available slugs).
        query: Natural language search query.
        top_k: Number of results to return (default 5).

    Returns:
        List of search results scoped to the specified book.
    """
    if not manifest.known(book):
        return [{"error": f"Unknown book: {book}. Use list_books() to see available books."}]

    # ⚡2: reuse the long-lived client singleton (no per-call open/close).
    client = get_client()
    results = hybrid_search(
        query=query,
        client=client,
        top_k=top_k,
        book_filter=book,
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def read_chunk(chunk_id: str, window: int = 1) -> dict:
    """Expand context around a search hit: read a chunk plus its neighbours.

    Pass the `chunk_id` from a search() / search_book() result to pull the exact
    chunk, its parent passage, and the adjacent sibling chunks — without
    re-searching or loading a whole chapter via get_chapter(). Useful for
    following a promising hit forward/backward through the surrounding text.

    Args:
        chunk_id: A chunk id from search results, e.g. "crypto-101-p0-c1".
        window: Number of sibling chunks to include on each side (default 1;
            window=0 returns only the chunk itself).

    Returns:
        Dict with the chunk text, its parent passage, neighbouring siblings
        (`previous`/`next`), heading_path, and `seen_chunk_ids` (every id
        returned) so repeated calls can skip already-seen chunks.
    """
    m = _CHUNK_ID_RE.match(chunk_id)
    if not m:
        return {"error": f"Malformed chunk_id (expected 'slug-pN-cM'): {chunk_id}"}
    slug, p_idx, _ = m.groups()
    parent_id = f"{slug}-p{p_idx}"

    chunk_path = CHUNKS_DIR / f"{slug}.json"
    if not chunk_path.exists():
        return {"error": f"Book not found: {slug}"}

    data = json.loads(chunk_path.read_text(encoding="utf-8"))
    siblings = [c for c in data.get("children", []) if c["parent_id"] == parent_id]
    ids = [c["child_id"] for c in siblings]
    if chunk_id not in ids:
        return {"error": f"Chunk not found: {chunk_id}"}
    idx = ids.index(chunk_id)

    lo = max(0, idx - window)
    hi = min(len(siblings), idx + window + 1)
    parent = next((p for p in data.get("parents", []) if p["parent_id"] == parent_id), None)
    target = siblings[idx]
    meta = target.get("metadata", {})

    def _slim(c: dict) -> dict:
        return {"chunk_id": c["child_id"], "text": c["text"]}

    return {
        "chunk_id": chunk_id,
        "book": meta.get("book_title", slug),
        "heading_path": meta.get("heading_path", ""),
        "text": target["text"],
        "parent_text": parent["text"] if parent else "",
        "previous": [_slim(c) for c in siblings[lo:idx]],
        "next": [_slim(c) for c in siblings[idx + 1:hi]],
        "seen_chunk_ids": ids[lo:hi],
    }


@mcp.tool()
def get_chapter(book: str, chapter: str) -> str:
    """Get the full text of a specific chapter from a book.

    Args:
        book: Book slug (use list_books() to see available slugs).
        chapter: Chapter heading or number to retrieve.

    Returns:
        Full markdown text of the chapter, or an error message.
    """
    md_path = MARKDOWN_DIR / f"{book}.md"
    if not md_path.exists():
        return f"Book not found: {book}"

    text = md_path.read_text(encoding="utf-8")
    chapter_lower = chapter.lower().strip()

    lines = text.split("\n")
    capturing = False
    chapter_lines: list[str] = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code

        if not in_code and re.match(r"^#\s+", line):
            heading = re.sub(r"^#\s+", "", line).strip().lower()
            if capturing:
                break
            if chapter_lower in heading or heading in chapter_lower:
                capturing = True

        if capturing:
            chapter_lines.append(line)

    if not chapter_lines:
        return f"Chapter '{chapter}' not found in {book}. Try get_chapter with a different chapter name."

    result = "\n".join(chapter_lines).strip()
    if len(result) > 50000:
        result = result[:50000] + "\n\n[... truncated at 50,000 characters ...]"
    return result


@mcp.tool()
def list_books() -> list[dict]:
    """List all indexed books with their slugs and chapter count.

    Returns:
        List of books with slug, title, and chapters.
    """
    result = []
    for slug, entry in manifest.books().items():
        md_path = MARKDOWN_DIR / f"{slug}.md"
        chapters = []
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8")
            for line in text.split("\n"):
                if re.match(r"^#\s+", line):
                    heading = re.sub(r"^#\s+", "", line).strip()
                    chapters.append(heading)

        result.append({
            "slug": slug,
            "title": entry["title"],
            "chapters": chapters,
            "chapter_count": len(chapters),
        })
    result.sort(key=lambda x: x["title"])
    return result


@mcp.tool()
def grep_books(
    pattern: str,
    book: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Exact-match keyword search across book markdown files using ripgrep (with Python fallback).

    Unlike semantic search, this finds exact text matches — useful for finding specific
    function names, error messages, code snippets, or terms that semantic search may miss.

    Args:
        pattern: Regex pattern to search for (case-insensitive).
        book: Optional book slug to restrict search to a single book.
        max_results: Maximum total results to return (default 20).

    Returns:
        List of matches with book slug, line_number, and context (matching line + 1 line of surrounding context).
    """
    if book is not None and not manifest.known(book):
        return [{"error": f"Unknown book: {book}. Use list_books() to see available books."}]

    # Build list of markdown files to search
    if book is not None:
        md_files = [MARKDOWN_DIR / f"{book}.md"]
    else:
        md_files = sorted(
            p for p in MARKDOWN_DIR.glob("*.md")
            if manifest.known(p.stem)
        )

    if not md_files:
        return []

    per_file_limit = max(3, max_results // len(md_files))

    try:
        return _grep_ripgrep(pattern, md_files, per_file_limit, max_results)
    except FileNotFoundError:
        return _grep_python(pattern, md_files, per_file_limit, max_results)


def _grep_ripgrep(
    pattern: str,
    md_files: list,
    per_file_limit: int,
    max_results: int,
) -> list[dict]:
    """Search using ripgrep for speed."""
    paths = [str(p) for p in md_files]
    cmd = [
        "rg",
        "--ignore-case",
        "--line-number",
        "--with-filename",
        "--context", "1",
        "--max-count", str(per_file_limit),
        pattern,
        *paths,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # rg returns 1 when no matches found, 2+ on error
    if proc.returncode > 1:
        return []

    results: list[dict] = []
    # Group consecutive lines by file:linenum blocks separated by "--"
    current_block_lines: list[str] = []
    current_book: str | None = None
    current_match_lineno: int | None = None

    def _flush_block():
        nonlocal current_book, current_match_lineno, current_block_lines
        if current_book is not None and current_match_lineno is not None:
            context = "\n".join(current_block_lines)[:500]
            results.append({
                "book": current_book,
                "line_number": current_match_lineno,
                "context": context,
            })
        current_block_lines = []
        current_book = None
        current_match_lineno = None

    for raw_line in proc.stdout.splitlines():
        if len(results) >= max_results:
            break

        # Block separator from --context
        if raw_line == "--":
            _flush_block()
            continue

        # Match lines: file:linenum:text  or context lines: file-linenum-text
        # When searching multiple files, rg prefixes with filename
        # Match line format: /path/to/file.md:123:matched text
        # Context line format: /path/to/file.md-122-context text
        match = re.match(r"^(.+?)([:-])(\d+)\2(.*)$", raw_line)
        if not match:
            continue

        filepath, separator, lineno_str, text = match.groups()
        slug = filepath.rsplit("/", 1)[-1].removesuffix(".md")
        lineno = int(lineno_str)

        if separator == ":":
            # This is a match line
            if current_book is not None:
                _flush_block()
            current_book = slug
            current_match_lineno = lineno
            current_block_lines.append(text)
        else:
            # Context line (separator is "-")
            current_block_lines.append(text)

    # Flush last block
    _flush_block()

    return results[:max_results]


def _grep_python(
    pattern: str,
    md_files: list,
    per_file_limit: int,
    max_results: int,
) -> list[dict]:
    """Pure Python fallback when ripgrep is not installed."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return [{"error": f"Invalid regex pattern: {pattern}"}]

    results: list[dict] = []

    for md_path in md_files:
        if len(results) >= max_results:
            break
        if not md_path.exists():
            continue

        lines = md_path.read_text(encoding="utf-8").splitlines()
        file_matches = 0
        slug = md_path.stem

        for i, line in enumerate(lines):
            if file_matches >= per_file_limit or len(results) >= max_results:
                break

            if compiled.search(line):
                # Build context: 1 line before + match + 1 line after
                context_lines = []
                if i > 0:
                    context_lines.append(lines[i - 1])
                context_lines.append(line)
                if i < len(lines) - 1:
                    context_lines.append(lines[i + 1])

                context = "\n".join(context_lines)[:500]
                results.append({
                    "book": slug,
                    "line_number": i + 1,  # 1-based
                    "context": context,
                })
                file_matches += 1

    return results[:max_results]


def main():
    mcp.run()


if __name__ == "__main__":
    main()
