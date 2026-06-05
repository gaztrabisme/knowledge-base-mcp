#!/usr/bin/env python3
"""Manual search testing against the indexed knowledge base."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_base.search import get_client, hybrid_search


def main():
    parser = argparse.ArgumentParser(description="Test search against the knowledge base")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--book", help="Filter to specific book slug")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--bm25-keywords",
        nargs="+",
        help="Exact terms for the BM25 sparse leg (separate from the semantic query)",
    )
    parser.add_argument(
        "--hyde-passage",
        help="Hypothetical answer paragraph to embed instead of the query (dense leg)",
    )
    args = parser.parse_args()

    client = get_client()
    try:
        results = hybrid_search(
            query=args.query,
            client=client,
            top_k=args.top_k,
            book_filter=args.book,
            rerank=not args.no_rerank,
            hyde_passage=args.hyde_passage,
            bm25_keywords=args.bm25_keywords,
        )
    finally:
        client.close()

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}\n")

    for i, r in enumerate(results):
        print(f"--- Result {i+1} (score: {r.score:.4f}) ---")
        print(f"Book: {r.book_title}")
        print(f"Path: {r.heading_path}")
        print(f"Child match: {r.child_match[:200]}...")
        print()


if __name__ == "__main__":
    main()
