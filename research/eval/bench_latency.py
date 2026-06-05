"""Search-path latency benchmark — before/after anchor for search.py changes.

Deterministic: takes the first N queries from a golden set and runs the REAL
full hybrid_search() pipeline (fusion + rerank + cutoff) one at a time, timing
wall-clock per query. Same query set in/out → comparable across runs (⚡2/⚡4).

Reports median / mean / p90 total, plus an isolated query-embed timing (the ⚡4
target: embed_texts on a single query string).

Usage:
  python3 scripts/bench_latency.py --label baseline --n 20
  python3 scripts/bench_latency.py --label after-singleton --n 20 --golden data/golden/golden_queries_hard.jsonl
"""

import argparse
import json
import statistics
import time

from knowledge_base.config import CHUNKS_DIR
from knowledge_base.index import embed_texts
from knowledge_base.search import get_client, hybrid_search

GOLDEN_PATH = CHUNKS_DIR.parent / "golden" / "golden_queries_hard.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--golden", default=str(GOLDEN_PATH))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.golden)][: args.n]
    queries = [r["query"] for r in rows]

    client = get_client()

    # warm up models + lock (exclude cold-start from the measurement)
    _ = hybrid_search(queries[0], client, top_k=10, rerank=True)
    _ = embed_texts([queries[0]], task="retrieval.query")

    totals, embeds = [], []
    for q in queries:
        t0 = time.perf_counter()
        _ = embed_texts([q], task="retrieval.query")
        embeds.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = hybrid_search(q, client, top_k=10, rerank=True)
        totals.append((time.perf_counter() - t0) * 1000)

    client.close()

    def stats(xs):
        xs = sorted(xs)
        p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        return {"median": round(statistics.median(xs), 1),
                "mean": round(statistics.mean(xs), 1),
                "p90": round(p90, 1)}

    print(f"===== latency [{args.label}]  n={len(queries)} (warmed) =====")
    print(f"  full hybrid_search (ms): {stats(totals)}")
    print(f"  query embed only  (ms): {stats(embeds)}")


if __name__ == "__main__":
    main()
