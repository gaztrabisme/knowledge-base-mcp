#!/usr/bin/env python3
"""Run the full ingestion pipeline: EPUB → markdown → chunks → (optional) enrich → index."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_base.ingest import ingest_all
from knowledge_base.chunk import chunk_all
from knowledge_base.index import index_all


def main():
    parser = argparse.ArgumentParser(description="Run the knowledge base ingestion pipeline")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip contextual enrichment (requires LM Studio)")
    parser.add_argument("--enrich", action="store_true",
                        help="Run contextual enrichment via LM Studio")
    parser.add_argument("--step", choices=["ingest", "chunk", "enrich", "index"],
                        help="Run only a specific step")
    args = parser.parse_args()

    if args.step:
        if args.step == "ingest":
            print("Step 1: Ingesting EPUBs → markdown...")
            ingest_all()
        elif args.step == "chunk":
            print("Step 2: Chunking markdown...")
            chunk_all()
        elif args.step == "enrich":
            print("Step 3: Enriching chunks via LM Studio...")
            from knowledge_base.enrich import enrich_all
            enrich_all()
        elif args.step == "index":
            print("Step 4: Indexing into Qdrant...")
            index_all()
        return

    print("=" * 60)
    print("Knowledge Base Ingestion Pipeline")
    print("=" * 60)

    print("\nStep 1: Ingesting EPUBs → markdown...")
    paths = ingest_all()
    print(f"  → {len(paths)} files\n")

    print("Step 2: Chunking markdown → parent/child chunks...")
    all_data = chunk_all()
    total_parents = sum(len(d["parents"]) for d in all_data.values())
    total_children = sum(len(d["children"]) for d in all_data.values())
    print(f"  → {total_parents} parents, {total_children} children\n")

    if args.enrich:
        print("Step 3: Enriching chunks via LM Studio...")
        print("  (Make sure LM Studio is running at http://localhost:1234)")
        from knowledge_base.enrich import enrich_all
        enrich_all()
        print()
    else:
        print("Step 3: Skipping enrichment (use --enrich to enable)\n")

    print("Step 4: Indexing into Qdrant...")
    count = index_all()
    print(f"  → {count} vectors indexed\n")

    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Books: {len(all_data)}")
    print(f"  Parents: {total_parents}")
    print(f"  Children: {total_children}")
    print(f"  Vectors indexed: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
