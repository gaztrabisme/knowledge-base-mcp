# Developer Knowledge Base — Project Wiki

Last updated: 2026-05-31 (v6.7.1 GATE-RESTORE: regenerated the 12 queries dropped by the v6.7 strip
(2 keyword + 10 navigational, whose gold had been index pages) from the **clean** corpus — seeds now
valid by construction; added a junk-nav phrasing reject (`index for…`/`Index (part N)`) so the
artifact can't recur. Pooled+judged against the clean index (mean 3.0 positives/query, validated zero
index golds / zero dead positives). Gate **428 → 440**, archetype balance restored to **200/80/80/80**.
Re-baseline held: NDCG@10 **0.8218** (was 0.8212@428; +0.0006 noise — cosmetic restore, no verdict
moves), R@10 0.989 / ceiling 0.996 intact. Paired reference `pq-v67-clean-hard.jsonl` regenerated at
n=440. Script: `scripts/regen_dropped_queries.py`.)
PREVIOUS (v6.7 CORPUS-HYGIENE round: chasing the retest's one open gap (navigational
0.763) found it was **two artifacts, not weak retrieval** — (1) 12 gate queries whose gold was a book
back-of-book INDEX page (dropped; legit nav = **0.866**, R@10 1.0), and (2) **6,276 chunks (12.7%) of
back-of-book index pages** that survived v6 stripping across 15 books. Stripped them
(`strip_index_backmatter.py`, validated contiguous-tail; kept real "Index of Security Definitions" /
pandas "Index" content), re-indexed clean on the 5080 → corpus **48,056 → 41,780**. Paired clean-vs-old
on the 428-query cleaned gate: **NDCG +0.0075 [+0.0037,+0.0118], wilcox p=0.0001, SIGNIFICANT** (keyword
stratum +0.030; recall/ceiling unchanged) — the **first statistically significant POSITIVE retrieval
change in all of v6.x**, found by chasing a measurement artifact, not a retrieval lever. Clean index
PROMOTED to production (old → `data/qdrant.v66.bak`). Indexer default batch 128→48 (128 OOMs Jina on
15GB). Full record in [decisions.md](decisions.md).)
PREVIOUS (v6.6 RETEST round: **paired re-adjudication of all six v6.3 verdicts on
gate v2** — Qwen3/Gemma/late-chunk/enrichment re-indexed on the 5080 over LAN, HyDE + per-domain
reranker re-run locally, each paired vs the Jina-v5 baseline via `significance.py` (now reports ρ +
MDE@80%). **Nothing overturned — production config vindicated** (Jina v5, no enrichment, no late-chunk,
descriptive-heading rerank, HyDE selective). Late-chunk (wilcox p=.045) and enrichment (p=.024) are
now *statistically* significant rejections they never had. The two embedding "rejections" **shrank
under graded labels** (Qwen3 0.012→0.008, Gemma 0.033→0.011) — the single-label gate had inflated
them; both n.s. with CI upper bound ≈ 0 (no upside → keep Jina; "significantly worse" needs n≈1000+).
Per-domain reranker confirmed (heading-in-input beats child-only, MRR p=.0018). Fixed `remote_index.py`
batch starvation (4→`KB_INDEX_BATCH`); driver `gpu_sweep.sh`; full table in
[decisions.md](decisions.md) + [v66_retest_notes](../data/golden/eval_runs/v66_retest_notes.md).)
PREVIOUS (v6.5 gate-UPGRADE round: executed the v6.4 audit's fixes. **Multi-positive graded labels** (TREC pooling + deepseek judge, mean 2.8 positives/query; judge validated vs Claude — 100% within ±1, 0.95 precision) + **3 new query archetypes** (+240: keyword/error_symbol/navigational), main gate **200→440**. Result: the single-label gate was deflating **recall/MRR/ceiling, not NDCG** — baseline stratum R@10 0.930→**0.980**, MRR 0.761→**0.875**, R@50 ceiling 0.960→**0.995** (the "8 never-retrieved" was mostly a label artifact). New finding the old gate couldn't see: **navigational queries (0.763) are the real gap**; keyword/error retrieve fine (BM25 works). Discrimination floor ~2× better (paired MDE@ρ0.7 0.048→0.025). Graded NDCG + `--by-archetype` in eval. v2 sets canonical; originals archived `*_v1`. Prior v6.4: gate audited D+ — embedding rejection downgraded to inconclusive.)
PREVIOUS (v6.4 gate-audit round: **audited the golden gate itself — overall grade D+**. The gate cannot resolve the ~0.01-NDCG deltas it has been adjudicating: per-query NDCG σ=0.31, n=200 → paired MDE ≥0.0195 at 80% power *even at ρ=0.95*. **v6.3's embedding-upgrade "rejection" (0.802 vs 0.790, gap 0.012) is downgraded to INCONCLUSIVE — below the discrimination floor, never significance-tested.** Also: 44.4% of sampled queries have an unlabeled-but-relevant top-10 hit (single-positive labels deflate NDCG; the "8 never-retrieved" may be label artifacts), query realism LOW (100% long why/how, 0 keyword/error/navigational). Added significance harness (`significance.py`, `--per-query`) + report card [gate-audit.md](gate-audit.md). **The gate's late-chunking rejection stands** (those gaps −0.06 to −0.09 are above the floor). Prior v6.3: late chunking rejected (ConTEB dilution); Jina v3 non-deployable.)

## Quick Links

- [Architecture](architecture.md) — Pipeline design, data flow, component interactions
- [Decisions](decisions.md) — Key technical decisions and their rationale
- [Gotchas](gotchas.md) — Known issues, workarounds, failure patterns
- [Ops Runbook](ops-runbook.md) — Common operations, remote GPU workflows, troubleshooting
- [RAG Frontier Survey 2025–26](rag-frontier-2026.md) — Cutting-edge scan mapped to this system; embedding-upgrade + late-chunking levers; the pooling fork
- [Gate Audit Report Card](gate-audit.md) — **v6.4.** Rigorous audit of the golden eval gate (D+): statistical power, label completeness (44% incomplete), query realism, coverage. Significance harness. Why v6.3's embedding rejection is inconclusive.
- [DSPy Enrichment Plan](dspy-enrichment-plan.md) — **CLOSED.** Phase 1+2 done (DSPy lost the prompt gate); Phase 3 done — full-index enrichment regressed the hard gate (0.7907→0.7759) so it was **reverted**. Corpus stays no-enrichment. Post-mortem in [decisions.md](decisions.md).

## Project Summary

MCP server providing semantic search over 46 technical books to Claude Code. Pipeline: markitdown/PyMuPDF-TOC → tiktoken-sized chunker (with overlap) → enrichment → heading_path + Jina v5 + FastEmbed BM25 → Qdrant (DBSF fusion) → HyDE/multi-query → Ettin-150m reranking → adaptive cutoff → FastMCP.

## Current State

| Metric | Value |
|--------|-------|
| Version | v6.7.1 (index back-matter stripped + promoted; gate restored to 440) |
| Books indexed | 46 (19 ML + 5 DB + 7 security + 2 distributed + 13 Rust) |
| Total vectors | **41,780** (v6.7 stripped 6,276 back-of-book index chunks from 48,056) |
| Hard gate (NDCG@10) | **0.8218** (clean index, 440-query v2 gate, balance 200/80/80/80; +0.0075 sig. over pre-strip) · easy 0.9225 |
| Collection | `developer_knowledge` |
| MCP tools | 5 (search, search_book, get_chapter, list_books, grep_books) — `get_chapter`/`list_books` now work for postgres |
| Reranker | Ettin-150m (ModernBERT) — **per-domain input (v6.2): heading_path for ml/rust, child-only for security/db/distributed** (`descriptive-heading`, supersedes ⚡1's global heading) |
| Search latency | MCP search ≈3.2s (was ≈10.2s) after ⚡2 client singleton — lands on next server restart |
| Hard gate (NDCG@10) | **0.8021** (v6.1 0.7921) · easy 0.9379 (v6.1 0.9434, R@10 intact at 0.975) |
| Token counting | tiktoken cl100k_base (child max 250, parent max 1900) |
| PDF headings | PyMuPDF TOC reconstruction (markitdown fallback on CID-font garble) |
| Enriched chunks | **None (by decision).** Full-index enrichment was tested (problem-symptom prompt, 48K, DeepSeek) and REGRESSED the hard gate (0.7907→0.7759) → reverted. Corpus is index-now-enrich-never. Rejected enriched chunks preserved at `data/chunks_enriched_rejected/`. |
| Eval harness | `eval_retrieval.py` (+ `--hyde` mode, `KB_RERANK_INPUT` A/B) + 2× golden sets (`data/golden/`); hard-set gate NDCG 0.802 / R@10 0.930 |
| Enrichment LM | `deepseek-v4-flash` (thinking-off), replaces local Qwen |
| Downloaded (awaiting ingestion) | 7 advanced-ML PDFs + 118 OWASP cheat sheets + 2 scraped web books |
