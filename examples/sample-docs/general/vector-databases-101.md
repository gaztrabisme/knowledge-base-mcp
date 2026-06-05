# Vector Databases 101

A vector database stores high-dimensional vectors — embeddings produced by a model — and
answers "which stored vectors are most similar to this query vector?" quickly. It is the
storage layer behind semantic search, recommendation, and retrieval-augmented generation.

## Why not a normal database

Relational and document databases excel at exact matches and range queries: find the row
where `id = 42`, or all orders after a date. Embeddings don't work that way. Two pieces of
text that *mean* the same thing map to vectors that are *close* in space but almost never
equal. The core operation is therefore nearest-neighbor search over hundreds or thousands
of dimensions, which ordinary B-tree indexes cannot accelerate.

## Distance metrics

Similarity is measured by a distance (or its inverse):

- **Cosine similarity** — the angle between two vectors, ignoring magnitude. The default
  for normalized text embeddings.
- **Dot product** — cosine scaled by magnitude; common when vectors are not normalized.
- **Euclidean (L2)** — straight-line distance. Used when magnitude carries meaning.

Pick the metric the embedding model was trained for; mixing them silently degrades results.

## Approximate nearest neighbor

Exact nearest-neighbor search means comparing the query against every stored vector — fine
for thousands of items, far too slow for millions. Vector databases use **approximate
nearest neighbor (ANN)** indexes that trade a little recall for a lot of speed:

- **HNSW** (Hierarchical Navigable Small World) — a layered graph you traverse greedily.
  Fast and high-recall; memory-hungry. The most common default.
- **IVF** (inverted file) — clusters vectors and only searches the nearest clusters.
  Compact, tunable via the number of clusters probed.
- **PQ** (product quantization) — compresses vectors into codes so far more fit in RAM,
  at some accuracy cost. Often combined with IVF.

The knobs (graph degree, clusters probed, quantization bits) all trade recall against
latency and memory. There is no universally correct setting — measure on your data.

## Hybrid search

Dense vectors capture meaning but miss exact tokens — a rare error code, a function name, a
product SKU. Keyword search (BM25) captures those but misses paraphrases. **Hybrid search**
runs both and fuses the results, so a query for "MVCC snapshot isolation" finds passages
that use those exact terms *and* passages that explain the concept in other words. Fusion
methods like Reciprocal Rank Fusion or Distribution-Based Score Fusion combine the two
ranked lists into one.

## Reranking

Retrieval is tuned for recall: pull a generous candidate set cheaply. A **reranker** — a
cross-encoder that reads the query and each candidate together — then reorders that set for
precision. It is far more accurate than the bi-encoder used for retrieval but too slow to
run over the whole corpus, so it only sees the top candidates. Retrieve wide, rerank narrow:
the pattern behind most modern search quality.
