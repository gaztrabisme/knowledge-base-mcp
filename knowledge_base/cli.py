"""`kb` command-line interface for the knowledge-base MCP server.

Subcommands: ingest, chunk, index, build, serve, search, books.

IMPORTANT: config.py reads KB_DOCS_DIR / KB_DATA_DIR / KB_DEVICE from the
environment at import time. So main() parses args FIRST, exports the env vars
from any provided flags, and only THEN do the subcommand handlers import the
heavy modules (lazily, inside each handler) — never at module top level.
"""

import argparse
import os
import sys


def _add_global_flags(p: argparse.ArgumentParser) -> None:
    """Flags accepted on every subcommand (applied to env before heavy imports)."""
    p.add_argument("--data-dir", help="Where the tool writes outputs (sets KB_DATA_DIR).")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                   help="Compute device for embedding/reranking (sets KB_DEVICE).")


def _apply_env(args: argparse.Namespace) -> None:
    """Translate parsed flags into the env vars config.py reads at import time."""
    if getattr(args, "docs_dir", None):
        os.environ["KB_DOCS_DIR"] = os.path.abspath(args.docs_dir)
    if getattr(args, "data_dir", None):
        os.environ["KB_DATA_DIR"] = os.path.abspath(args.data_dir)
    if getattr(args, "device", None):
        os.environ["KB_DEVICE"] = args.device


def cmd_ingest(args: argparse.Namespace) -> int:
    from knowledge_base import ingest
    paths = ingest.ingest_all()
    print(f"Ingested {len(paths)} markdown file(s).")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from knowledge_base import chunk
    data = chunk.chunk_all()
    parents = sum(len(d["parents"]) for d in data.values())
    children = sum(len(d["children"]) for d in data.values())
    print(f"Chunked {len(data)} book(s): {parents} parents, {children} children.")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from knowledge_base import index
    count = index.index_all()
    print(f"Indexed {count} vectors.")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from knowledge_base import chunk, index, ingest
    paths = ingest.ingest_all()
    print(f"  ingest: {len(paths)} markdown file(s)")
    data = chunk.chunk_all()
    parents = sum(len(d["parents"]) for d in data.values())
    children = sum(len(d["children"]) for d in data.values())
    print(f"  chunk:  {len(data)} book(s), {parents} parents, {children} children")
    count = index.index_all()
    print(f"  index:  {count} vectors")
    print("Build complete.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    # stdio MCP protocol: nothing may go to stdout before the server starts.
    from knowledge_base import server
    print("Starting knowledge-base MCP server (stdio)...", file=sys.stderr)
    server.main()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from knowledge_base.search import get_client, hybrid_search
    client = get_client()
    try:
        results = hybrid_search(
            query=args.query,
            client=client,
            top_k=args.top_k,
            book_filter=args.book,
            rerank=not args.no_rerank,
        )
    finally:
        client.close()

    if not results:
        print("No results found.")
        return 0

    print(f"\nQuery: {args.query}\nResults: {len(results)}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r.score:.4f}) ---")
        print(f"Book: {r.book_title}")
        print(f"Path: {r.heading_path}")
        print(f"Snippet: {r.child_match[:200]}...\n")
    return 0


def cmd_books(args: argparse.Namespace) -> int:
    from knowledge_base import manifest
    entries = manifest.books()
    if not entries:
        print("No books registered. Run `kb ingest` first.")
        return 0
    rows = [(slug, e.get("domain", ""), e.get("rerank_input", ""), e.get("title", ""))
            for slug, e in sorted(entries.items())]
    headers = ("SLUG", "DOMAIN", "RERANK", "TITLE")
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    print(f"\n{len(rows)} book(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="Local hybrid-search MCP server over your own documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Convert source documents under docs-dir to markdown.")
    p.add_argument("docs_dir", nargs="?", help="Source documents directory (sets KB_DOCS_DIR).")
    _add_global_flags(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chunk", help="Chunk ingested markdown into parent/child chunks.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("index", help="Embed and index chunks into Qdrant.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("build", help="Run ingest -> chunk -> index end to end.")
    p.add_argument("docs_dir", nargs="?", help="Source documents directory (sets KB_DOCS_DIR).")
    _add_global_flags(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("serve", help="Run the MCP server over stdio.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("search", help="Debug search from the terminal (not the MCP server).")
    p.add_argument("query", help="Search query.")
    p.add_argument("--book", help="Restrict to a single book slug.")
    p.add_argument("--top-k", type=int, default=5, help="Number of results (default 5).")
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("books", help="List registered books from the manifest.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_books)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Parse FIRST so flags can set env vars BEFORE any config-dependent import.
    args = build_parser().parse_args(argv)
    _apply_env(args)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # surface a clean error, non-zero exit
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
