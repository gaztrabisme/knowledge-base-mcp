#!/usr/bin/env python3
"""Calibrate adaptive score cutoff for Ettin-150m reranker.

Runs diverse queries, collects full score distributions (all 50 candidates scored),
and analyzes gap patterns to determine optimal cutoff parameters.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_base.search import get_client, get_ranker
from knowledge_base.index import embed_texts, text_to_sparse_vector
from knowledge_base.config import COLLECTION_NAME
from qdrant_client import models

QUERIES = [
    # Broad ML
    ("broad_ml", "gradient descent optimization techniques"),
    ("broad_ml_2", "feature engineering best practices for machine learning"),
    # Narrow DB
    ("narrow_db", "PostgreSQL MVCC transaction isolation levels"),
    ("narrow_db_2", "Redis pub/sub message patterns"),
    # Cross-domain security
    ("security", "TLS certificate validation and pinning"),
    ("security_2", "OWASP top 10 injection prevention"),
    # Cryptography
    ("crypto", "authenticated encryption with associated data AEAD"),
    # Distributed systems
    ("distributed", "consensus algorithm Raft leader election"),
    # Code-specific
    ("code", "PyTorch DataLoader num_workers multiprocessing"),
    # Niche single-book
    ("niche", "Kaggle competition ensembling stacking blending"),
    # Very broad
    ("very_broad", "how to build production machine learning systems"),
    # Nonsense (should return low scores)
    ("nonsense", "purple elephant quantum saxophone migration"),
    # Conceptual gap (vocabulary mismatch)
    ("conceptual", "making neural networks smaller for edge devices"),
    # Multi-hop
    ("multi_hop", "database indexing strategies for time series data"),
]


def run_full_scored_search(query: str, client, candidates: int = 50):
    """Run hybrid search and return ALL candidates with reranker scores."""
    query_vector = embed_texts([query], task="retrieval.query")[0]
    query_sparse = text_to_sparse_vector(query)

    fused = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_vector, using="", limit=candidates),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices, values=query_sparse.values
                ),
                using="bm25",
                limit=candidates,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=candidates,
        with_payload=True,
    ).points

    merged = [
        {"id": str(p.id), "score": p.score, "payload": p.payload}
        for p in fused
    ]

    reranker = get_ranker()
    documents = []
    for r in merged:
        p = r["payload"]
        text = p.get("child_text", "")
        ctx = p.get("context_text", "")
        documents.append(f"{ctx}\n{text}" if ctx else text)

    # Score ALL candidates (no top_k limit)
    rankings = reranker.rank(query, documents)

    scored = []
    for item in rankings:
        result = merged[int(item["corpus_id"])]
        scored.append({
            "score": float(item["score"]),
            "book": result["payload"].get("book_title", ""),
            "heading": result["payload"].get("heading_path", ""),
            "child_preview": result["payload"].get("child_text", "")[:100],
        })

    return scored


def analyze_gaps(scores: list[float]):
    """Analyze score gaps for cutoff detection."""
    if len(scores) < 2:
        return {}

    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]

    cutoff_points = {}
    for multiplier in [1.0, 1.5, 2.0, 2.5, 3.0]:
        threshold = multiplier * median_gap
        for i, gap in enumerate(gaps):
            if gap > threshold:
                cutoff_points[f"{multiplier}x"] = {
                    "position": i + 1,
                    "gap": round(gap, 4),
                    "score_before": round(scores[i], 4),
                    "score_after": round(scores[i + 1], 4),
                }
                break
        else:
            cutoff_points[f"{multiplier}x"] = {"position": len(scores), "gap": 0}

    return {
        "median_gap": round(median_gap, 4),
        "max_gap": round(max(gaps), 4),
        "min_gap": round(min(gaps), 4),
        "cutoff_points": cutoff_points,
    }


def main():
    client = get_client()

    print("Loading reranker (first call)...")
    get_ranker()
    print("Reranker loaded.\n")

    all_results = {}

    for label, query in QUERIES:
        print(f"[{label}] {query}")
        scored = run_full_scored_search(query, client)
        scores = [r["score"] for r in scored]

        gap_analysis = analyze_gaps(scores)

        print(f"  Candidates: {len(scored)}")
        print(f"  Score range: [{scores[-1]:.4f}, {scores[0]:.4f}]")
        print(f"  Median gap: {gap_analysis.get('median_gap', 'N/A')}")

        # Show where different multipliers would cut
        for mult, info in gap_analysis.get("cutoff_points", {}).items():
            print(f"  Cutoff @{mult}: position {info['position']}"
                  f" (score {info.get('score_before', 'N/A')} → {info.get('score_after', 'N/A')})")

        # Score floors
        for floor in [0.0, 0.01, 0.05, 0.1, 0.5]:
            above = sum(1 for s in scores if s >= floor)
            print(f"  Scores >= {floor}: {above}")

        # Top 5 results
        print(f"  Top 5:")
        for i, r in enumerate(scored[:5]):
            print(f"    {i+1}. [{r['score']:.4f}] {r['book'][:30]} | {r['heading'][:40]}")

        all_results[label] = {
            "query": query,
            "scores": [round(s, 4) for s in scores],
            "gap_analysis": gap_analysis,
            "top5": scored[:5],
        }
        print()

    # Summary statistics across all queries
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_medians = []
    all_cutoffs_1_5x = []
    all_score_ranges = []

    for label, data in all_results.items():
        scores = data["scores"]
        ga = data["gap_analysis"]
        all_score_ranges.append((scores[-1], scores[0]))
        if "median_gap" in ga:
            all_medians.append(ga["median_gap"])
        cp = ga.get("cutoff_points", {}).get("1.5x", {})
        if "position" in cp:
            all_cutoffs_1_5x.append(cp["position"])

    print(f"Median gap across queries: {sorted(all_medians)}")
    print(f"1.5x cutoff positions: {all_cutoffs_1_5x}")
    print(f"Score ranges: min_low={min(r[0] for r in all_score_ranges):.4f}, "
          f"max_high={max(r[1] for r in all_score_ranges):.4f}")

    # Save raw data
    out = Path("data/cutoff_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw data saved to {out}")

    client.close()


if __name__ == "__main__":
    main()
