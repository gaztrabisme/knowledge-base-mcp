# RAG Frontier Survey — 2025–2026

> Research note, 2026-05-30. A scan of the cutting edge in RAG, mapped against
> **this** system (v6.2: hybrid dense+BM25 → DBSF → Ettin-150m rerank, parent/child
> chunking, agent-provided HyDE, no LLM in the query path). Purpose: separate genuine
> paradigm shifts from hype, and find what's actually actionable here.
> Sources are linked inline; this is a point-in-time snapshot, not a maintained doc.

## ⚠️ MEASURED OUTCOME (v6.3, 2026-05-30) — both Tier-1 levers REJECTED

The two actionable levers this survey proposed were built and measured against the golden
gates. **Both lost.** Full record in [decisions.md](decisions.md) → "Embedding Upgrade +
Late Chunking (v6.3)".

- **Embedding upgrade → rejected → inconclusive (v6.4) → PAIRED-RETESTED (v6.6).** Qwen3-Embedding-0.6B
  and EmbeddingGemma-300M scored below Jina v5 on the v6.3 single-label gate (0.790 / 0.770 vs 0.802).
  v6.6 re-indexed both on the 5080 and ran a **paired test on gate v2**: **Qwen3 Δ=−0.0084** (n.s., ρ=0.917,
  CI upper ≈ 0 → no upside), **Gemma Δ=−0.0107** (borderline, wilcox p=.043). **Both gaps shrank**
  (0.012→0.008, 0.033→0.011) — the single-label gate had *inflated* embedding deficits. Net: keep Jina
  v5 (no upside to either), but "significantly worse" is unproven and Gemma's "clear reject" is downgraded
  to "weakly worse." "MTEB rank ≠ corpus performance" still holds directionally.
- **Late chunking → rejected, now SIGNIFICANT (v6.6).** Net-negative every v6.3 measurement (Gemma
  −0.026 w/rerank; Jina v3 native −0.089) and it lowers the recall ceiling. v6.6 paired re-test (Gemma
  raw-late vs raw-naive) Δ=−0.0164, **wilcox p=0.045** — the rejection now carries a p-value. The ConTEB
  dilution mechanism (below) is real for this jargon corpus.
- **Jina v3 non-deployable** anyway: transformers conflict with both the Ettin reranker and
  production v5.

So the survey's read below was directionally right about the *frontier* but **over-optimistic
about the two levers** — the gate caught it. Keep the v6.2 stack. The text below is preserved
as the original pre-measurement analysis.

## TL;DR (original, pre-measurement)

The 2025–26 frontier does **not** say "rebuild." It says our architecture *is* the
validated 2026 production baseline, several of our rejected experiments were rejected
for reasons now in the published literature, and there is **one high-ROI move** — a
dense-embedder upgrade — that targets our one known weakness (the 8 never-retrieved
recall misses). A second, more novel lever (late chunking) is a separate experiment
because of a pooling constraint (below). *(Both measured negative — see the boxed
outcome above.)*

## The through-line of 2025–26 RAG

Field moved from one-shot *retrieve-then-generate* to *reasoning-driven, iterative,
self-correcting retrieval*. The load-bearing insight underneath the noise is a **cost
dichotomy**:

- **Per-query-LLM** (expensive): agentic loops, RL-trained search (Search-R1 family),
  GraphRAG global / DRIFT.
- **Index-time-LLM or no-LLM-in-query-path** (cheap at serve time): HippoRAG 2,
  LightRAG, learned sparse retrieval, classic hybrid+rerank.

2026 production consensus (stated near-identically across multiple independent guides):
**default to hybrid dense+BM25 + cross-encoder rerank, no LLM in the query path; route
adaptively; reserve per-query-LLM paradigms for multi-hop / ambiguous / high-stakes
queries.** That is a description of the system already built here.

## What the frontier VALIDATES about our stack

| Our decision | 2025–26 evidence that backs it |
|---|---|
| **Enrichment regressed → reverted** (DeepSeek context_text, hard gate 0.7907→0.7759) | **ConTEB / "Context is Gold"** ([arXiv 2505.24782](https://arxiv.org/abs/2505.24782), May 2025): contextual enrichment *regresses* on **extractive / technical-term corpora** — COVID-QA **−21 nDCG@10** — because queries match on rare jargon (`MVCC`, `WAL`, `PyO3`, `nonce`) and prepended LLM prose dilutes those exact-term signals (worst in BM25). Our corpus is the textbook failure case. We found empirically what they proved. |
| **HyDE / multi-query as agent-provided params** (no local LLM) | **FrugalRAG** ([arXiv 2507.07634](https://arxiv.org/pdf/2507.07634)): a plain ReAct agent matches RL-trained Search-R1 on multi-hop recall. Our consumer *is* a ReAct agent (Claude Code) driving the MCP tool. We already have "agentic RAG" at zero added server-side per-query LLM cost. |
| **GraphRAG skipped** (v4) | Graph structure earns its cost *only* on multi-hop / global-thematic queries; HippoRAG 2 ([arXiv 2502.14802](https://arxiv.org/abs/2502.14802)) is the cheap version if ever needed. For book-fact lookup, over-engineering. |
| **Hybrid file-based store, no "stuff it all in long context"** | **Context Rot** ([Chroma, Jul 2025](https://www.trychroma.com/research/context-rot), 18 frontier models): accuracy degrades with input length *even on trivial tasks, even below the window limit*; focused ~300-tok excerpts **beat** 113k-tok full-context on LongMemEval. RAG is not dead for static, latency-sensitive, local corpora. **LaRA** ([arXiv 2502.09977](https://arxiv.org/abs/2502.09977), ICML'25): "no silver bullet" — RAG robust to answer position, wins on dynamic/large/latency-sensitive. |

Four settled questions, now externally corroborated. Recorded so they don't get
re-litigated.

## Actionable shortlist for this system (ranked)

### Tier 1 — dense-embedder upgrade (the lever we already flagged)
Open problem = the **8 never-retrieved recall misses** (real embedding failures,
unreachable by HyDE or rerank). Concrete 2025 targets:

- **Qwen3-Embedding-0.6B** (Alibaba, Jun 2025, **Apache 2.0**, 1024-dim, MRL, 32K ctx) —
  top open MTEB at its tier, fits the 5080, small enough to query on the Mac.
  [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) ·
  [arXiv 2506.05176](https://arxiv.org/pdf/2506.05176)
- **EmbeddingGemma-300M** (Google, Sep 2025, mean pool, 768-dim, MRL) — best MTEB under
  500M, on-device-fast. [card](https://huggingface.co/google/embeddinggemma-300m)
- Larger (Qwen3-Embedding-4B, NV-Embed-v2, Jina v4 3.8B) are likely **too heavy for Mac
  query-time** — see the hard constraint below.

Our golden hard/easy gate harness is already built to measure exactly this.

### Tier 1b — late chunking (the "context without enrichment" play)
**Late chunking** ([Jina, arXiv 2409.04701](https://arxiv.org/pdf/2409.04701)): embed the
whole document/section first, then mean-pool token spans per chunk → chunk vectors carry
document context with **no per-chunk LLM call and no jargon dilution**. It sidesteps the
*exact* ConTEB failure mode that killed our enrichment — same goal (context-aware chunks),
opposite cost/mechanism. **This is the one to test precisely because enrichment failed.**

### Tier 2 — reranker A/B (optional, low priority)
Qwen3-Reranker-0.6B / mxbai-rerank-base-v2 (0.5B) are the new open SOTA cross-encoders.
But Ettin-150m is ModernBERT-family and we *just* tuned its per-domain input (v6.2).
Prior steer holds — "cutting edge, tune don't replace." A/B only if Tier 1 leaves headroom.

### Tier 3 — targeted fix, not a paradigm
**ColPali / visual retrieval** (renders PDF pages to images, embeds pixels — no font/CMap
dependency) is the clean answer to the **postgresql-14-internals CID-font** failure. But it
returns **page images, not text**, implying a VLM generation step that mismatches our
text-for-Claude design. Pragmatic collapse: run a modern **VLM/OCR pass once at ingest** to
recover that one book's headings. Not a ColPali index — a one-book re-OCR.

## The pooling fork (key technical constraint discovered)

**Late chunking and "best embedding model" do not combine in one model today.**

- Late chunking needs **mean pooling** — you split the contextualized token sequence along
  chunk boundaries and average each span. Only works if mean-over-tokens *is* the trained
  sentence representation.
- The strongest small open embedders are **LLM-decoder, last-token/EOS pooled**:
  **Qwen3-Embedding-0.6B** and **our current Jina v5-small** (built on Qwen3-0.6B-Base).
  Last-token pooling collapses a sequence to one token's state → cannot produce per-span
  contextual vectors → **no late chunking.** Jina staff confirmed v5 dropped late-chunking
  support for this exact reason
  ([discussion](https://huggingface.co/jinaai/jina-embeddings-v5-text-small/discussions/9)).

| Model | Params | Dim | Pooling | Late chunking | Mac-query feasible | License |
|---|---|---|---|---|---|---|
| Jina v5-small (current) | 677M | 1024 | last-token | ❌ | yes | CC BY-NC 4.0 |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | last-token | ❌ | yes | Apache 2.0 |
| Jina v3 | 570M | 1024 | **mean** | ✅ native flag | yes | CC BY-NC 4.0 |
| EmbeddingGemma-300M | 308M | 768 | **mean** | ✅ manual | yes (fastest) | Gemma terms |
| Jina v4 | 3.8B | 2048 | mean | ✅ native | borderline (heavy) | CC BY-NC 4.0 |
| voyage-context-3 | API | 2048↓ | baked-in | ✅ native | ❌ API-only | proprietary |

**Consequence:** the embedding upgrade and late chunking are **two separate experiments**.
A last-token SOTA embedder (Qwen3-0.6B) is the Tier-1 quality play; late chunking forces a
**mean-pooling** model (Jina v3 or EmbeddingGemma). voyage-context-3 "bakes in" context
without the fork but is API-only → violates the local/offline constraint.

## Hard constraint for any embedding change

**Index on the 5080, query on the Mac.** Every `search()` embeds the query on Apple Silicon
(CPU/MPS). So a candidate must be (a) the *same* model for index and query — vector spaces
must match — and (b) small/fast enough for sub-~300ms single-query embedding on the Mac.
This prunes 4–8B embedders from the query path even though they're fine for batch indexing.
Mac-feasible shortlist: Jina v5 (current), Qwen3-Embedding-0.6B, Jina v3, EmbeddingGemma-300M.

## Not for us (with one-line why)

- **RL-trained retrieval** (Search-R1, DeepRAG) — per-query LLM + RL training; ReAct agent matches it (FrugalRAG). We already have the agent.
- **Compilation-stage / Pinecone Nexus "context architecture"** (May 2026) — enterprise productization, self-reported unverified numbers, irrelevant to a local personal KB.
- **Late-interaction multi-vector store** (GTE-ModernColBERT + PLAID) — genuine revival, storage tax down to ~10×, but a real architectural change; same logic as the v4 TurboVec rejection at 48K scale. Phase-2-someday.

## Freshness caveat

Well-established (trust): Qwen3-Embedding, ColPali, Context Rot, ConTEB, late chunking,
HippoRAG 2, voyage-context-3. Very fresh / verify before betting: any Feb–May 2026 arXiv
IDs (A-RAG, Nemotron-ColEmbed, ColBERT-Zero, Pinecone Nexus numbers).
