# Architecture

> Reflects the **v6** live code (`knowledge_base/`). Last synced 2026-05-30.
> Enrichment is **not shipped** — full-index enrichment regressed the hard gate
> (NDCG 0.7907→0.7759) and was reverted. The corpus is index-now-enrich-never;
> `context_text` is present in the schema but empty. See [decisions.md](decisions.md).

## Pipeline Overview

```
Source files (EPUB · PDF · HTML · Markdown — 46 books)
    │
    ▼
[ingest.py] → clean markdown
    │   - EPUB/HTML: markitdown (heading hierarchy preserved)
    │   - PDF: PyMuPDF TOC reconstruction (headings recovered from TOC)
    │           markitdown fallback on CID-font garble (clean body, NO headings)
    │   - Front/back-matter stripping
    │
    ▼
[chunk.py] tiktoken heading-aware parent/child chunker
    │   - Split on ## (H2) → parents (≤1900 real tokens)
    │   - Parents split into children (≤250, min 75 tokens)
    │   - Small sections merged · oversized blocks hard-split · H2 section overlap
    │   - Token counts via tiktoken cl100k_base (≈ Jina v5 within ~2%)
    │
    ▼
[enrich.py] contextual enrichment — DISABLED (reverted, net-negative)
    │   - context_text stays empty; stage retained but unused
    │
    ▼
[index.py] dual-vector embedding → Qdrant
    │   - Embed input = heading_path + context_text + child_text  (joined "\n\n")
    │   - Dense: Jina v5-text-small (677M, 1024-dim, sentence-transformers)
    │   - Sparse: FastEmbed Qdrant/bm25 (tokenization + stopwords + IDF)
    │   - One Qdrant collection: developer_knowledge (48,009 vectors)
    │
    ▼
[search.py] hybrid retrieval + reranking
    │   - One Qdrant query_points with two prefetches (dense + bm25), top-50 each
    │   - DBSF fusion (distribution-based score fusion)
    │   - Ettin-150m cross-encoder rerank (ModernBERT, 151M)
    │   - Adaptive cutoff (score floor 4.0, min 3 results)
    │   - Parent deduplication → returns full parent_text
    │
    ▼
[server.py] FastMCP → Claude Code
        5 tools: search · search_book · get_chapter · list_books · grep_books
```

## Data Flow

```
knowledge-base/                          SOURCE (gitignored, ~1.9GB)
  applied-ml/*.epub
  databases/*.pdf · *.md
  security/*.pdf · *.md
  distributed-systems/*.pdf
  rust/*.epub
  production-rag-guide/*.md
         │
         ▼
data/markdown/*.md       ── intermediate (gitignored)
         │                   { one cleaned markdown file per book slug }
         ▼
data/chunks/*.json       ── intermediate (gitignored)
         │                   { parents: [...], children: [...] } per book
         ▼
data/qdrant/             ── index (gitignored)
                             collection: developer_knowledge, 48,009 vectors
```

## Search Pipeline Detail

Query arrives via MCP → `hybrid_search()`:

1. **Embed query** — dense vector from `hyde_passage` if provided, else `query`
   (Jina v5, `retrieval.query` task). Sparse BM25 vector from `bm25_keywords` if
   provided, else `query`.
2. **Multi-query (optional)** — each `extra_queries` entry adds its own dense +
   sparse prefetch pair, broadening recall before fusion.
3. **Qdrant prefetch** — all prefetches run in a single `query_points` call:
   - Dense leg: cosine similarity, top-50 (`candidates`)
   - Sparse leg: IDF-weighted BM25, top-50
4. **DBSF fusion** — Distribution-Based Score Fusion normalizes dense and BM25
   score distributions into a comparable range, then merges.
5. **Ettin-150m rerank** — cross-encoder scores each fused candidate against the
   query. Document text is **per-domain** (`_rerank_doc`, v6.2): `heading_path +
   context_text + child_text` for descriptive-heading books (ml, rust), and
   `context_text + child_text` (child-only) for generic/formal-heading books
   (security, databases, distributed) where the heading is orthogonal noise to a
   developer-symptom query. context_text is empty today. Returns reranked order.
6. **Adaptive cutoff** — keep results with score ≥ 4.0, capped at `top_k`; if
   fewer than 3 survive, fall back to the top 3.
7. **Parent dedup** — multiple children from one parent collapse to a single
   result returning the full `parent_text` section with the best child as
   `child_match`.

## Shared Models (singletons, lazy-loaded)

| Model | Role | Notes |
|-------|------|-------|
| Jina v5-text-small | Dense embeddings (1024-dim) | `index.py:get_model()` |
| FastEmbed Qdrant/bm25 | Sparse vectors | `index.py:get_sparse_model()` |
| Ettin-150m cross-encoder | Reranking | `search.py:get_ranker()` |

Models are cached module-level globals. **The Qdrant client is *not*** — `server.py`
opens and closes a fresh file-based client per request (see Known Costs).

## Key Design Choices

- **Parent/child chunking** — children are the indexed/matched unit (small,
  focused); parents are the returned unit (full section context). Precise
  retrieval, rich results.
- **File-based Qdrant** — no server process; index lives in `data/qdrant/`.
  Adequate for our scale (<100K vectors).
- **Agent-provided HyDE / multi-query / BM25 keywords** — Claude supplies the
  hypothetical passage, query expansions, and exact keywords as tool parameters.
  Zero local-LLM dependency, zero added latency.
- **Standalone remote scripts** — `remote_index.py` / `remote_enrich.py` are
  self-contained (no package imports), rsync-deployed to the RTX 5080 box for
  ~10× faster indexing. `remote_index.py` skips books already in the collection,
  so a full re-index requires deleting the remote collection first.

## Known Costs / Limitations

- **Qdrant reopened per query** — `server.py` does `get_client()` … `client.close()`
  on every call; the file-based open re-reads segment metadata each time. A
  long-lived singleton would remove this per-query overhead.
- **Reranker input is per-domain** (v6.2, resolves the old "heading-blind" note) —
  ⚡1 (v6.1) fed `heading_path` to the cross-encoder globally; the rank diagnostic
  showed that helps ml/rust but hurts security/db/distributed. `_rerank_doc()` now
  routes heading_path to ml/rust only (hard NDCG 0.7921→0.8021). Env
  `KB_RERANK_INPUT` overrides the policy for A/B.
- **postgresql-14-internals headings** — this PDF's subset fonts lack a ToUnicode
  CMap, so PyMuPDF garbled it and ingestion fell back to markitdown. Result: clean
  body text (~0.9% glyph garble, mostly digits) but **zero `#` heading markers**,
  so `get_chapter()` / `list_books()` are empty for it. The 29 chapter titles are
  still present in the body as bare `N Title` lines (e.g. `2 Isolation`) — a regex
  promotion pass can restore them without OCR.
- **Enrichment pending → cancelled** — v6 chunks carry `heading_path` +
  `child_text` but no `context_text`; the enrichment experiment was run and
  reverted as net-negative on retrieval quality.
```
