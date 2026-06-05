#!/usr/bin/env python3
"""Benchmark Ettin-150m reranker across backends: PyTorch, ONNX+CPU, ONNX+CoreML.

Results (2026-05-28, Apple M-series):
  PyTorch:  avg=0.169s for 50 docs, 9s cold load (once per server lifetime)
  ONNX:     BLOCKED — optimum requires transformers<4.58, ModernBERT needs >=5.2.0
  CoreML:   BLOCKED — same dependency conflict

Conclusion: PyTorch is fast enough (sub-200ms reranking). ONNX revisit when optimum
supports transformers 5.x. The earlier "5-8s" concern was cold model loading, not reranking.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "cross-encoder/ettin-reranker-150m-v1"

QUERY = "How does PostgreSQL handle MVCC and transaction isolation?"
DOCUMENTS = [
    "PostgreSQL uses Multi-Version Concurrency Control (MVCC) to manage concurrent access to data.",
    "Each transaction sees a snapshot of the database as of the start of the transaction.",
    "VACUUM is responsible for reclaiming storage occupied by dead tuples.",
    "B-tree indexes are the default index type in PostgreSQL.",
    "The Write-Ahead Log (WAL) ensures data integrity during crashes.",
    "Random forests combine multiple decision trees through bagging.",
    "Gradient boosting builds trees sequentially to correct errors.",
    "Neural networks learn hierarchical feature representations.",
    "Redis is an in-memory data structure store used as a database.",
    "OAuth 2.0 defines four grant types for different use cases.",
    "The CAP theorem states that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance.",
    "Raft consensus algorithm ensures leader election and log replication.",
    "Cross-entropy loss measures the difference between predicted and true distributions.",
    "Batch normalization normalizes layer inputs to stabilize training.",
    "Attention mechanisms allow models to focus on relevant input parts.",
    "PostgreSQL's query planner chooses between sequential and index scans.",
    "Hash indexes in PostgreSQL are only useful for equality comparisons.",
    "GIN indexes are designed for composite values like arrays and full-text search.",
    "PostgreSQL supports row-level security policies for fine-grained access control.",
    "The buffer pool cache stores frequently accessed pages in shared memory.",
    "Distributed hash tables partition data across multiple nodes.",
    "MapReduce processes large datasets in parallel across clusters.",
    "Paxos is a family of protocols for solving consensus in distributed systems.",
    "TLS 1.3 reduces handshake latency compared to TLS 1.2.",
    "AES-GCM provides authenticated encryption with associated data.",
    "HMAC uses a cryptographic hash function with a secret key.",
    "Feature engineering transforms raw data into informative features.",
    "Principal component analysis reduces dimensionality while preserving variance.",
    "K-means clustering partitions data into k groups by minimizing within-cluster variance.",
    "Support vector machines find the maximum-margin hyperplane for classification.",
    "PostgreSQL's TOAST mechanism stores large field values out-of-line.",
    "Logical replication in PostgreSQL allows selective table replication.",
    "Connection pooling with PgBouncer reduces connection overhead.",
    "Partitioning large tables improves query performance and maintenance.",
    "PostgreSQL's EXPLAIN ANALYZE shows actual execution times and row counts.",
    "The shared_buffers parameter controls the size of PostgreSQL's buffer cache.",
    "Autovacuum runs automatically to prevent transaction ID wraparound.",
    "PostgreSQL supports JSON and JSONB data types for semi-structured data.",
    "Window functions perform calculations across sets of rows related to the current row.",
    "Common Table Expressions (CTEs) improve readability of complex queries.",
    "Convolutional neural networks excel at spatial feature extraction.",
    "Recurrent neural networks process sequential data with hidden state.",
    "Transformers use self-attention to process sequences in parallel.",
    "BERT uses masked language modeling for bidirectional pretraining.",
    "LoRA adds low-rank adapter matrices for efficient fine-tuning.",
    "Quantization reduces model precision from fp32 to int8 for faster inference.",
    "Knowledge distillation transfers knowledge from a large teacher to a small student.",
    "Retrieval-augmented generation combines search with language model generation.",
    "Vector databases use approximate nearest neighbor search for similarity queries.",
    "BM25 is a probabilistic relevance ranking function based on term frequency.",
]


def benchmark_backend(name: str, backend: str | None, provider: str | None = None):
    from sentence_transformers import CrossEncoder

    print(f"\n{'='*60}")
    print(f"Backend: {name}")
    print(f"{'='*60}")

    kwargs = {"trust_remote_code": True}
    if backend:
        kwargs["backend"] = backend
    if provider:
        kwargs["model_kwargs"] = {"provider": provider}

    try:
        t0 = time.perf_counter()
        model = CrossEncoder(MODEL, **kwargs)
        load_time = time.perf_counter() - t0
        print(f"  Load time: {load_time:.2f}s")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    # Warmup
    try:
        model.rank(QUERY, DOCUMENTS[:5], top_k=3)
    except Exception as e:
        print(f"  FAILED on warmup: {e}")
        return None

    # Benchmark: rank all 50 docs, repeat 3 times
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        rankings = model.rank(QUERY, DOCUMENTS, top_k=10)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            top3 = [(r["corpus_id"], round(r["score"], 4)) for r in rankings[:3]]
            print(f"  Top 3: {top3}")

    avg = sum(times) / len(times)
    print(f"  Latencies (3 runs): {[f'{t:.3f}s' for t in times]}")
    print(f"  Average: {avg:.3f}s")
    return {"name": name, "avg": avg, "times": times, "load": load_time}


def main():
    results = []

    # PyTorch baseline
    r = benchmark_backend("PyTorch (CPU)", backend=None)
    if r:
        results.append(r)

    # ONNX + CPU
    r = benchmark_backend("ONNX (CPU)", backend="onnx")
    if r:
        results.append(r)

    # ONNX + CoreML
    r = benchmark_backend("ONNX (CoreML)", backend="onnx", provider="CoreMLExecutionProvider")
    if r:
        results.append(r)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        baseline = results[0]["avg"]
        for r in results:
            speedup = baseline / r["avg"] if r["avg"] > 0 else 0
            print(f"  {r['name']:20s}  avg={r['avg']:.3f}s  load={r['load']:.2f}s  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
