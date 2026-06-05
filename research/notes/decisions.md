# Technical Decisions

## Pipeline

### markitdown over pandoc (v2)
**Context:** pandoc flattened EPUB heading hierarchy (all H1), producing tiny chunks (parent avg 295 tokens, child avg 35). markitdown preserves H1/H2/H3.
**Decision:** Switched to markitdown. Chunk quality jumped to parent avg 467, child avg 187 tokens.

### Custom chunker over Chonkie
**Context:** Evaluated `RecursiveChunker.from_recipe("markdown")`. It splits on heading boundaries but has no parent-child support.
**Decision:** Keep custom chunker. It does heading-aware splitting with parent-child linking, which is our core retrieval pattern.

### Single Qdrant collection over per-domain split
**Context:** Expanding from ML-only to 4 domains. Per-domain collections allow independent rebuild but complicate cross-domain search.
**Decision:** Single collection. Corpus is small (<20K vectors). Cross-domain results are valuable ("regularization" should hit both ML and security books). Domain filtering available via payload field if needed.

## Search Pipeline (v4)

### FastEmbed Qdrant/bm25 over custom regex tokenizer
**Context:** Custom tokenizer was `re.findall(r"[a-z0-9]+")` with hash-based indices. No stopwords, no stemming, hash collisions.
**Decision:** FastEmbed's `Qdrant/bm25` model. Proper tokenization, 29-language stopword lists, no collisions. Drop-in replacement for `text_to_sparse_vector()`.

### DBSF fusion over client-side RRF
**Context:** Had two separate `query_points` calls merged with custom RRF (k=60). Qdrant has built-in fusion via `prefetch` API.
**Decision:** Qdrant's DBSF fusion. Normalizes dense/sparse score distributions automatically. Single API call. Deleted `_rrf_fusion()` entirely.

### MiniLM-L-12 reranker over TinyBERT-L-2
**Context:** FlashRank default `Ranker()` loaded TinyBERT-L-2 (4MB, 2-layer) — weakest tier. Larger models (BGE 568M, Jina 278M) exceed 500ms latency on Apple Silicon.
**Decision:** `ms-marco-MiniLM-L-12-v2` (34MB, 12-layer). Best FlashRank English model. ~50-100ms for 50 candidates on CPU. One-line change.

### rank-T5-flan as future option
**Context:** As we expand beyond ML into security/databases, out-of-domain generalization matters more. `rank-T5-flan` (110MB) has better zero-shot performance.
**Decision:** Deferred. MiniLM-L-12 is good enough for now. Config makes it a one-line swap.

## Search Pipeline (v5)

### Ettin-150m over MiniLM-L-12 / Jina v3 / Qwen3-Reranker
**Context:** MiniLM-L-12 (34MB, FlashRank) was the v4 reranker. Researched modern alternatives: Jina v3 (278M, 0.5856 NDCG@10), Qwen3-Reranker-0.6B (0.5940), Ettin family (6 ModernBERT models, 17M-1B, Apache 2.0, distilled from mxbai-rerank-large-v2).
**Decision:** Ettin-150m (151M, 0.5994 NDCG@10). Beats Jina v3 at half the size. PyTorch at 0.17s/50-docs on Apple Silicon. sentence-transformers CrossEncoder replaces FlashRank (FlashRank has a fixed model list that can't load Ettin).

### Absolute score floor over gap-based cutoff
**Context:** Planned gap detection (1.5x median gap) for adaptive cutoff. Calibrated on 14 diverse queries with full 50-candidate score distributions.
**Decision:** Absolute floor (score >= 4.0) + min results (3). Gap detection triggers at position 1-2 for most queries because Ettin's smooth distributions put the largest gap between #1 and #2 — too aggressive. Score floor of 4.0 perfectly separates nonsense (max 3.68) from real queries (min top score 8.36).

### HyDE/multi-query as agent-provided parameters over local LLM
**Context:** HyDE traditionally requires a local LLM to generate hypothetical answers. Multi-query needs a model to reformulate queries.
**Decision:** Expose `hyde_passage` and `extra_queries` as MCP tool parameters. The calling agent (Claude) generates these directly — zero latency overhead, zero dependency on LM Studio or any inference server. BM25 leg still uses original query for keyword matching.

### ONNX acceleration deferred
**Context:** Ettin-150m (ModernBERT) on ONNX would give 2-3x speedup via `sentence-transformers[onnx]`.
**Decision:** Blocked. `optimum` requires `transformers<4.58`, ModernBERT needs `>=5.2.0`. Hard conflict, no compatible version exists (as of 2026-05-28). PyTorch at 0.17s is fast enough. Revisit when optimum supports transformers 5.x.

### Parent chunk overlap at H2 boundaries
**Context:** Cross-section answers (where the answer spans two consecutive H2 sections) get split across chunks, degrading retrieval for boundary content.
**Decision:** Copy last 2 sentences of each section into the start of the next section (within same chapter). +3.3% chunk count (19,103 → 19,728). PDF-sourced books with empty chapter metadata are skipped (can't determine section boundaries).

### Heading path in embeddings
**Context:** Section titles carry strong semantic signal ("MVCC and Transaction Isolation") but weren't included in the embedded text.
**Decision:** Prepend `heading_path` before `context_text` and `child_text` when constructing the embedding input. Both `index.py` and `remote_index.py` updated in sync.

## Pipeline (v6)

### tiktoken token counting + "Option 1" chunk sizing
**Context:** Chunk sizes were estimated by word count, which undercounted real tokens by ~1.65x — children drifted well past their nominal target. Calibration showed tiktoken `cl100k_base` matches the Jina v5 tokenizer within ~2%.
**Decision:** Count tokens with tiktoken `cl100k_base`. Re-tightened thresholds to **Option 1**: children to a true `CHILD_MAX_TOKENS=250` (sharper match units), `CHILD_MIN_TOKENS=75`; parents kept near their prior effective size at `PARENT_MAX_TOKENS=1900` to preserve returned context. Rationale: the **child is the match unit** (embedded/retrieved/reranked) so it should be tight and focused, while the **parent is the context unit** returned to the agent, so it stays large. Smaller children = more precise reranking without sacrificing the context the agent actually reads.

### PDF heading recovery via PyMuPDF TOC bookmarks
**Context:** v5's known limitation — markitdown extracts PDF body text but drops heading hierarchy, so PDF-sourced books had empty chapter/section metadata. Compared 4 approaches (markitdown-only, PyMuPDF TOC, Docling, heuristic font-size detection) across 3 books.
**Decision:** PyMuPDF `doc.get_toc()` two-pass reconstruction. Inject TOC bookmark headings at their page-start positions; neutralize any body line that already looks like a markdown heading (`^#{1,6}\s` → prefixed with a space) so code comments like `# init` can't masquerade as structure. Body text never contributes headings. Result: 100% TOC coverage on the production PDF set; postgres/dist-sys/owasp/crypto all gained clean chapter/section paths with zero code-comment false H1s. Docling rejected — crashes on macOS MPS backend.

### PostgreSQL Internals → markitdown fallback (CID-font garble guard)
**Context:** postgresql-14-internals embeds subset fonts with no ToUnicode CMap. PyMuPDF maps those glyphs into the Armenian/Hebrew/extended-Cyrillic ranges ("PostgreSQL" → "PostgreԯԭԨ", "2022" → "ҁѿҁ҂"), garbling ~0.3-0.9% of chars including the flagship keyword. v5/markitdown rendered this book cleanly but flat. It's the only book of 46 affected.
**Decision:** Auto-detect garble — if PyMuPDF text exceeds `_GARBLE_RATIO_THRESHOLD=0.002` suspicious glyphs, discard the TOC headings and fall back to markitdown (clean body, no headings). General, not a hardcoded book name, so any future CID-broken PDF self-heals. Trade-off accepted: clean searchable body + intact BM25 keyword match beats structured-but-garbled text, since the agent reads returned snippets directly. Chose clean text over headings for this one book.

### Front-matter strip: position guard on "Chapter N" heading
**Context:** `strip_front_back_matter` used a first-pass "bare Chapter N / numeric heading = first real chapter" heuristic. creativeprojectsforrustprogrammers names real chapters descriptively but reserves "# Chapter N" headings for a back-matter answer key — the heuristic matched that late appendix heading and discarded the entire 713KB book as front matter, leaving only the 23KB answer key (26 chunks).
**Decision:** Only trust a numeric/"Chapter N" heading if it appears within the first 40% of lines; otherwise fall through to the descriptive-heading fallback. Recovered the book to 873 children.

### Index back-matter narrowing + hard-split oversized chunks
**Context:** A book's "Index" back-matter (page-number noise) and monster paragraphs were producing children far over 250 tokens (16.6% over, one 37,249-token chunk — an "# Index" page).
**Decision:** (1) Narrow exact-match back-matter cut to `index`/`subject index`/`author index` only, so mid-book "Index Scans"/"Indexing" headings don't trigger it. (2) `_hard_split_block()` token-bounded splitting of oversized paragraphs — code blocks split by line and re-fenced, prose split by sentence with a word-level fallback for monster sentences. Result: max child 946, p99 255, 3.3% over 250.

### BM25 keyword parameter + metadata-enriched enrichment prompt
**Context:** Two smaller v6 search-quality items.
**Decision:** (1) Expose a BM25 keyword customization parameter on search. (2) Enrichment prompt now includes the chunk's metadata fields (book/chapter/section) so generated context is grounded in document structure.

### Rust corpus expansion (33 → 46 books)
**Context:** Acquired 13 Rust EPUBs (Humble Bundle Packt Rust bundle). Generalized the EPUB ingestion loop to scan all `SOURCE_DIRS` (added `rust/`) rather than only `applied-ml/`.
**Decision:** Folded all 13 into the v6 re-index. BOOKS entries added in sync to both `config.py` and `scripts/remote_index.py`. Corpus is now 46 books, 48,009 vectors.

### Index now, enrich later
**Context:** Re-chunking invalidates the v5 enrichment context_text, which forces re-enrichment of all 47.6K chunks — a 13-20h serial GPU pass that would block every structural v6 win.
**Decision:** Re-index immediately with `heading_path + child_text` (empty context_text), defer enrichment to a separate background pass (`remote_enrich.py`, then re-index enriched chunks). The embedding-text builder already drops empty context_text, so no code change was needed for the index-now path. Structural improvements (tiktoken sizing, PDF headings, Rust books, bug fixes) ship now; enrichment quality returns later without blocking.

## Enrichment / Eval Harness (v6.x — DSPy sprint)

### DeepSeek-v4-flash as the enrichment + DSPy LM (replaces local Qwen)
**Context:** The DSPy plan originally specced local Qwen3.6-27B via llama-server (free on owned GPU, but a ~13–20h serial enrichment pass for 48K chunks). DeepSeek V4 launched cheap with high concurrency.
**Decision:** Use `deepseek-v4-flash` (OpenAI-compatible, `base_url=https://api.deepseek.com`) for both DSPy optimization and the final enrichment. Pricing: input $0.14/M (cache miss), $0.0028/M (cache hit), output $0.28/M — the full 48K enrichment pass costs **under ~$15** and finishes in **~15–45 min** via async fan-out (concurrency cap is **2500** for flash, no RPM/TPM caps, 429 on overflow). This frees the GPU entirely and collapses the serial pass.
**Thinking control:** `extra_body={"thinking": {"type": "disabled"}}` — ⚠️ defaults to ENABLED, so non-thinking must be set explicitly. Task model (enrichment execution, many calls) runs thinking-OFF; MIPROv2 instruction-proposer (few calls) runs thinking-ON for better prompt candidates. `reasoning_effort` (high/max) only applies when thinking is on.
**Wiring:** DSPy→LiteLLM via the OpenAI passthrough (`dspy.LM("openai/deepseek-v4-flash", api_base=..., extra_body=...)`) rather than the native `deepseek/` provider, because the brand-new model ID may not be in LiteLLM's registry yet (would break cost-tracking / strip extra_body).
**Cache ordering:** order enrichment calls by parent so the stable prefix (instruction → metadata → parent_text) is prompt-cached across a parent's children — cache hit is 50× cheaper than miss.

### Extrinsic eval harness, scored at parent granularity
**Context:** Enrichment text is never read by anyone — it only shifts the child embedding. So the metric must be extrinsic (retrieval quality), and the golden set is `query → relevant-chunk` pairs, not `chunk → gold-summary`.
**Decision:** `scripts/eval_retrieval.py` runs the golden queries through the REAL `hybrid_search()` and scores **NDCG@10 / Recall@10 / MRR** plus a **Recall@50 ceiling** (fusion-only, no rerank). Matching is at **parent** granularity — the pipeline dedups by parent and the agent receives parent content, so parent-level is the meaningful unit. The ceiling-vs-delivered gap separates "never retrieved" (embedding/recall lever — enrichment's target) from "retrieved but reranked out" (reranker lever). Added `parent_id`/`child_id` to `SearchResult` (dataclass only; `to_dict` unchanged, so the MCP contract is untouched) so eval can match without re-querying internals.

### Vocab-gap hardening of the golden set (the harness's first real finding)
**Context:** The first golden set was LLM-generated FROM each chunk, so queries inherited the chunk's rare terms. Baseline came back near-ceiling — NDCG@10 0.931, Recall@10 0.975, **Recall@50 1.000**. That means retrieval already wins on chunk-vocabulary queries AND enrichment's lever (recall) is maxed, so the set literally **cannot measure enrichment**.
**Decision:** Build a second set (`golden_queries_hard.jsonl`) that paraphrases each query into developer problem/symptom framing, deliberately avoiding the chunk's signature terminology (`scripts/harden_golden_set.py`). Verified objectively: mean query→chunk content-word overlap dropped **52%** (0.649→0.313). The hard baseline exposes real headroom — **NDCG@10 0.791, Recall@10 0.925, Recall@50 0.960** — with the 15 misses splitting into **8 never-retrieved** (enrichment's target) + **7 reranked-out** (reranker's target).
**Use:** the **hard set is the gate** any enrichment/DSPy change must beat (NDCG 0.791 / MRR 0.747); the **easy set is a regression guard** (don't regress the already-solved case). Both baselines recorded under `data/golden/eval_runs/`.

### DSPy/MIPROv2 did NOT beat the hand-written prompt → ship a manual prompt (gate 2D)
**Context:** Phase 2 ran DSPy on a self-contained mini-retrieval metric: a **frozen distractor background** (~800 chunks, baseline-enriched, embedded once into an in-memory Qdrant with the production dense+BM25+DBSF+Ettin pipeline) plus a **freshly candidate-enriched positive** temp-inserted per query. This makes the global retrieval metric per-example so DSPy can average it, while embedding only one chunk per example. `scripts/dspy_enrich.py` (`--smoke`/`--build-bg`/`--bootstrap`/`--mipro`/`--manual`).
**Finding 1 — hard negatives are mandatory:** A flat random distractor pool scored a useless flat **1.0000** (the positive is the only on-topic chunk, wins trivially). Seeding the pool with each positive's same-book/same-chapter **siblings** (the real competitors) dropped the baseline off the ceiling and made the metric discriminate. Same "measure-the-wrong-thing" trap as the golden-set vocab bias, one level down.
**Finding 2 — DSPy lost:** On the hard pool (100-dev), MIPROv2 (`auto=light`, task-LM thinking-OFF, proposer thinking-ON) tied the incumbent exactly (0.9615 → 0.9615, **Δ +0.0000**). Two hand-written variants beat it: **problem-symptom 0.9678**, query-anticipation 0.9676, baseline 0.9615, entity-dense 0.9461 (the keyword-stuffing variant — worst). BootstrapFewShot also showed no durable gain.
**Decision:** Per the pre-registered gate 2D, **DSPy does not earn its place — ship a hand-written prompt.** The DSPy harness is kept as reusable infrastructure (re-runnable if the corpus/objective changes), but the v6.x enrichment prompt is manual.
**The meaningful signal is directional, not absolute:** the two **problem/query-vocabulary** framings both beat baseline and the keyword-stuffing one lost — 3/3 in the direction Phase-1 theory predicts (match the searcher's vocabulary, since that's how the golden set and real devs phrase queries). Each individual margin (~+0.006) is within the 100-query mini-harness noise, so the mini-harness can *rank* prompts but cannot *confirm* a sub-0.04 win. **The decisive test is the full 48K index** (Phase 3): enrich with the chosen problem-symptom framing, re-index, and require beating the no-enrichment hard baseline (NDCG 0.791) — else enrichment isn't worth shipping and we keep index-now-enrich-never.

### Phase 3 — enrichment LOST the full-index gate → REVERTED, ship no-enrichment (2026-05-30)
**Context:** Per the Phase-2 plan, all 48,009 chunks were enriched with the gate-winning **problem-symptom** prompt (run via the DSPy `Enricher`, Envelope A, on `deepseek-v4-flash` thinking-OFF; ~$13, two passes — the first stalled at 72% on an "Insufficient Balance" API error, resumed cleanly after top-up). Re-indexed on the RTX 5080 (fresh remote qdrant; `remote_index.py` builds `[heading_path, context_text, child_text]`). Then re-ran the eval harness against the no-enrichment baseline.
**Finding — enrichment made retrieval WORSE on the gate:** hard-set **NDCG 0.7907 → 0.7759 (−0.0148)**, **MRR 0.7468 → 0.7267 (−0.0201)**, Recall@10 unchanged (0.925), Recall@50 +0.005. Per-domain NDCG: ml +0.008 and distributed +0.044 (n=6, noise), but **databases −0.060, security −0.061, rust −0.018** — it regressed the largest-n domains. Critically, it *hurt the vocab-gap (hard) set* — the exact case enrichment was supposed to help most.
**Decision:** Per the pre-registered ship rule, **enrichment does not ship. Reverted to the no-enrichment index (NDCG 0.7907).** The repo stays **index-now-enrich-never** for this corpus. The enriched chunks are preserved at `data/chunks_enriched_rejected/` (gitignored) for any future revisit; canonical `data/chunks` were stripped back to `context_text=""` so disk ↔ shipped index stay consistent.
**Why it failed — the lesson:** the mini-harness (Phase 2) gave a **false positive**. An 800-chunk in-memory pool with a single freshly-enriched positive rewards any prompt that makes the positive *distinctive*; the full 48K index instead exposes the real cost — enriching **every** chunk with an LLM-written "problem/symptom" gloss adds correlated, model-voiced boilerplate that **blurs inter-chunk distinctions** and pulls the embedding toward the enrichment's vocabulary rather than the chunk's actual content. Heading-path + raw child text (already in the embed input since v5) carries the signal; the extra LLM sentence is net noise here. This is exactly why the full-index gate was reserved as decisive over the cheap proxy metric — and why the gate was pre-registered before seeing the result.

## Search Pipeline (v6.1 — search-path quality + PostgreSQL rescue, 2026-05-30)

Sprint after the enrichment revert closed that lever. An architecture review (diagram +
live-code read) surfaced five search-path hotspots; each was measured against the registered
golden gate (hard NDCG **0.7907**, easy **0.9311**) — nothing shipped that didn't clear it.
Search-path only, no corpus rebuild, no GPU.

### ⚡2 — long-lived Qdrant client singleton (the headline)
**Context:** `server.py` did `get_client()` … `client.close()` on every MCP search. Pre-flight
measured the real cost: opening a **48K-point file-based collection re-scans all segment
metadata ≈ 5.9 s**. So every search paid that twice (open+close) — reopen-per-call **~10.2 s**
vs reused client **~3.2 s** per query.
**Decision:** `search.py:get_client()` is now a module-level singleton; `server.py` drops the
per-call `close()`. **~3× latency cut, zero quality risk** (identical results — same client, same
data). The architecture review had filed this as a minor tidy-up; measurement made it the
single biggest win of the sprint. Realized when a server next restarts (running servers hold
old code). The file-based singleton holds the lock for the process lifetime — the server must
be down during any offline delete/index (already true).
**Latent follow-up:** the 5.9 s open is a local-mode symptom of >20K points; Qdrant-in-Docker
would make opens instant *and* speed queries — bigger than ⚡2 but more infra.

### ⚡4 — batch the query-variant embeds
**Decision:** `hybrid_search` embeds the primary query + all `extra_queries` in one
`embed_texts` batch instead of one call per variant. Zero risk on the gate/normal path (a
list of one → byte-identical). Batch-vs-single differs by ≤5e-3 only when `extra_queries` is
actually passed (attention masking under padding) — negligible, well below reranker score
gaps. Minor latency win, only on the rarely-used multi-query path.

### ⚡1 — reranker reads heading_path (SHIPPED, mixed per-domain)
**Context:** the dense embedding includes `heading_path`, but the Ettin cross-encoder reranked
**bare `child_text`** (`context_text` empty post-enrichment-revert) — blind to the section
breadcrumb. Fixed `_rerank` to build documents as `heading_path + context_text + child_text`,
mirroring `index.py`'s embed input.
**Result:** **hard NDCG 0.7887 → 0.7921 (+0.0034 over post-postgres, +0.0014 over the 0.7907
gate), MRR 0.7441 → 0.7504; easy 0.9311 → 0.9434 (+0.0123)**, easy databases perfect (1.0).
**Honest caveat — it's a mixed trade, not a uniform win.** On the hard set it helped where
headings are descriptive (rust 0.793→0.823) but *hurt* generic-heading domains (databases
0.848→0.809, security 0.773→0.742, +1 miss). Aggregate clears the gate on both sets, which is
the ship criterion. The dip suggests the `(part N)` suffixes in heading_paths are partly
fighting the reranker — cleaning that noise before reranking is a candidate follow-up.

### ⚡3 — candidate depth 50→75→100 (REJECTED, keep 50)
**Context:** hard fusion ceiling is 0.96–0.97 at k=50, so ~3–4% of positives never reach the
reranker. A cheap no-rerank headroom probe showed real headroom: ceiling @50=0.97, **@75=0.98**,
@100=0.98 (75→100 adds nothing).
**Finding:** the headroom does **not** convert. Full hard eval at candidates=75: **NDCG 0.7921 →
0.7778 (−0.0143)**, Recall@10 0.92→0.905. Feeding the cross-encoder 25 extra candidates adds
more distractors than true positives it surfaces — the reranker gets noisier and pushes
positives out of the top-10.
**Decision:** **Keep candidates=50.** Ceiling recall ≠ delivered NDCG; the reranker is the
binding constraint, not retrieval depth. Added a `--candidates` arg to `eval_retrieval.py` so
the sweep is re-runnable (default 50, pipeline unchanged).

### PostgreSQL 14 Internals — heading rescue + full re-chunk (the one CID-font book)
**Context:** postgres fell back to markitdown (CID-font garble guard, v6) → clean body but
**zero headings**, so its 1,113 chunks carried `heading_path=" (part N)"` / `chapter="0"`, and
`get_chapter`/`list_books` were empty for it — the corpus's worst-indexed book. Pre-flight
showed only **4 of 200** golden queries touch postgres and databases was already at ceiling 1.0,
so this is a **regression-guard + navigation fix, not a metric win**.
**Decisions:**
- **Full path over minimal.** Stripped the 682-line front matter (title page + two TOCs, pure
  navigation noise) and restored structure by promoting the 29 chapter titles — which exist in
  the body as bare `N Title` lines — to markdown headings (a regex pass located all 29 uniquely
  in book order; manual sign-off gate before editing). No OCR: zero content pages were lost, the
  garble is ~0.9% digit glyphs that don't affect search.
- **H1 chapters, not H1-book + H2-chapters.** H1+H2 gives a richer `heading_path`
  ("PostgreSQL 14 Internals > 2 Isolation") but `list_books`/`get_chapter` match `^#\s+` (H1
  only), so H2 chapters wouldn't surface — and widening that regex regresses `get_chapter` for
  every multi-H2 book. Chose **all-H1 chapters** (matches the working database-design book);
  `get_chapter`/`list_books` now work, heading_path = "2 Isolation". The retrieval difference is
  unmeasurable here (4 queries, domain at ceiling).
- **Chunker bug found:** restoring headings alone wasn't enough — `split_large_section` →
  `_force_split` set `heading_path = f"{heading} (part N)"`, but oversized headingless chapters
  had empty `heading`, yielding `" (part N)"`. With headings restored, paths are now informative
  (`"2 Isolation (part 3)"`). (The underlying force-split-clobbers-heading behavior is a known
  limitation for any heading-sparse oversized chapter; postgres now carries the chapter name.)
- **Golden re-anchor.** Re-chunking shifts every parent boundary, so the 4 postgres golden
  `positive_child_ids` (in both sets) dangled; `eval_retrieval.py` *silently skips* queries whose
  positives aren't in the chunk map (would drop databases n=13→9). Re-anchored all 4 by
  `source_text` match — one needed care (the answer passage straddled a chunk boundary; matched
  on the distinctive `cost=21.03..21.04` line + answer prose).
**Result:** re-chunked 135→**143 parents / 1,113→1,160 children**; index 48,009→**48,056**;
`get_chapter`/`list_books` restored (30 entries). **Retrieval-neutral:** identical 15 hard misses
(zero postgres misses in either run), Recall@10 and ceiling unchanged; NDCG −0.0020 from a pure
within-top-10 ordering shift on re-anchored databases positives. Verdict: guard satisfied in
substance — the nav/heading wins are banked, retrieval didn't move.
**Safety:** fresh pre-mutation index backup at `data/qdrant.v6-pre-postgres.bak` (gitignored).
The markdown fix lives in gitignored `data/markdown/` (survives pipeline re-runs since ingest
skips existing markdown, but is lost if regenerated from source — see ops-runbook for re-apply).

## Search Pipeline (v6.2 — measure-first optimization round, 2026-05-31)

### Fresh diagnostic re-anchored the miss taxonomy (8 / 8, not reranker-dominated)
**Context:** entered another optimization round. The v6.1 snapshot read "reranker is the binding
constraint" (from the ⚡3 candidate-depth rejection). A fresh full eval on both gates
(hard NDCG **0.7921** / easy **0.9434**, n=200/201) reclassified the **16 hard misses** as an
even **8 never-retrieved** (not in top-50 fusion — recall/embedding lever) + **8 reranked-out**
(retrieved @50, ranked below top-10 — reranker lever). The "reranker is binding" verdict was
specific to *candidate depth* (more candidates = more distractors); per-miss, the two levers are
**co-equal**. Security/crypto is the weakest domain (NDCG 0.742, `boneh-shoup` ×3) and bleeds on
both. Easy ceiling is a perfect **1.000** → when query vocab matches the chunk, recall is maxed,
so every easy miss is a reranker drop and the hard never-retrieved are *purely* vocab-gap.

### HyDE is NOT a corpus-wide lever — REJECTED both modes (the round's main finding)
**Context:** `eval_retrieval.py` tests the bare-query path, but production exposes `hyde_passage`
(Claude generates it). The 8 never-retrieved looked like textbook HyDE wins, so before paying for
an embedding swap we measured the HyDE-augmented path. Built `gen_hyde.py` (DeepSeek generates a
hypothetical answer from the **query alone** — `source_text` never shown, no gold leakage) +
a `--hyde` flag on the harness.
**Finding:** HyDE **lost** on the hard gate in both variants.

| Variant | NDCG@10 | Ceiling | Misses |
|---|---|---|---|
| Bare query (baseline) | **0.7921** | **0.960** | **16** |
| HyDE replace | 0.7753 | 0.950 | 20 |
| HyDE augment | 0.7809 | 0.955 | 19 |

**Mechanism — a clean domain split:** HyDE helps **ML** (broad conceptual; generated passage lands
near textbook prose — rescued the neural-style-transfer never-retrieved miss, ml ceiling
0.966→0.978) and **harms Rust** (precise borrow-checker/async semantics; generic passage drifts
to wrong chunks — broke 3 working rust queries clean out of top-50, rust ceiling 0.946→0.911).
Of the 8 never-retrieved, augment-HyDE reached only **2/8** at recall, delivered **1/8**.
**Why it loses structurally:** Jina v5's bare-query ceiling is already 0.96 — little recall
headroom to capture — while HyDE's drift cost is real and concentrated in precise-vocab domains.
**Decision:** HyDE does **not** ship as a default. It is a **selective, caller's-discretion** tool
(good on ML-conceptual queries, bad on Rust-precise) — exactly the per-query judgment an agent
should make, never corpus-wide. The 8 never-retrieved are confirmed **real recall failures**:
HyDE, the standard free fix, can't reach them, so the only path to them is an embedding upgrade.

### HyDE replace→augment: a production footgun fixed (search.py, SHIPPED)
**Context:** `hybrid_search` did HyDE by *replacement* — `dense_text = hyde_passage if hyde_passage
else query` — so a supplied passage **removed the original query vector from the prefetch entirely**.
That's the weak variant: when Claude passes a HyDE passage on a query that already retrieves well,
it could silently make results *worse* (the 5 broken queries under replace-mode).
**Decision:** changed to **augment** — the original query is always a dense leg (the anchor); a
`hyde_passage` adds a *second* dense leg via Qdrant multi-prefetch (same mechanism as
`extra_queries`). Strict improvement to the exposed param (0.7809 > 0.7753 when used); the
no-HyDE **gate path is byte-identical** (verified: `dense_texts=[query]` → same two prefetches).
Kept even though HyDE-default lost, because it de-risks the API Claude already calls.

### Reranker input: per-domain heading (`descriptive-heading`, SHIPPED — the round's win)
**Context:** the user pushed back on "swap the reranker" — Ettin-150m is current (2025 ModernBERT),
so *tune before replace*. The `diag_rerank.py` rank-position probe on the 8 reranked-out misses
showed the cross-encoder ranks the positives at 9–28 (close, not fundamentally disagreeing) and —
the smoking gun — **child-only input beat ⚡1's heading-aware input on 7 of 8 misses** (joy-of-crypto
positive 23→9, deep-rl 20→10). ⚡1 shipped heading_path *globally*; the probe showed it helps where
headings are descriptive and hurts where they're formal/generic.
**A/B over both gates (env `KB_RERANK_INPUT`, default now the winner):**

| reranker input | hard NDCG | easy NDCG | note |
|---|---|---|---|
| heading (⚡1 v6.1) | 0.7921 | 0.9434 | global heading_path |
| child (pre-⚡1) | 0.7887 | 0.9311 | loses both — heading carries vocab the easy set + ml/rust need |
| rust-heading | 0.7989 | 0.9333 | wins hard, regresses easy; leaves ml on child |
| **descriptive-heading** | **0.8021** | **0.9379** | **SHIPPED** — heading for {ml,rust}, child for {security,databases,distributed} |

**Decision:** ship `descriptive-heading`. **+0.0100 hard gate** (rescued 2 positives into top-10,
R@10 0.920→0.930) for **−0.0055 easy** — and the easy regression is benign: **easy R@10 unchanged
at 0.975, no positive lost, pure within-top-10 reorder**. It dominates every other variant
(strictly beats rust-heading on both axes). The mechanism: heading_path is a *vocabulary* signal —
it reinforces the match when the query shares the chunk's vocab (descriptive ml/rust titles, and
the whole vocab-matched easy set) and adds orthogonal noise when the heading is formal/generic
(crypto formalism, OWASP section numbers, redis ops) against a developer-symptom query.
**Implementation:** `search.py:_rerank_doc()` — `_GENERIC_HEADING_BOOKS` (14 security/db/dist slugs)
get child-only; everything else (ml + rust) keeps heading. No model swap, no re-index.
**Caveat (anti-overfit):** the per-domain hypothesis rests on the robust cells (ml n=89, rust n=56,
security n=36); databases n=13 and distributed n=6 are noisy and align directionally but don't
drive the decision. The easy guard (R@10 intact) is the check that this isn't hard-set overfitting.
**Supersedes ⚡1** (which is now the `heading` fallback mode, retained for A/B reproducibility).

## Embedding Upgrade + Late Chunking (v6.3 experiment — frontier levers, measured negative, 2026-05-30)

A "research the RAG frontier" session surfaced two Tier-1 levers from the 2025–26 literature: swap
Jina v5 for a higher-MTEB embedder, and adopt **late chunking** (context-aware chunk embeddings
without LLM enrichment). Both were measured against the hard/easy golden gates. **Both rejected.**
Full survey in [rag-frontier-2026.md](rag-frontier-2026.md).

**Hard constraint that shaped candidates:** index on the 5080 GPU, **query on the Mac**. The embedder
must run per-query on Apple Silicon *and* be the same model for index + query (vector spaces must
match). Prunes 4–8B embedders from the query path. Mac-feasible shortlist: Jina v5 (current),
Qwen3-Embedding-0.6B, Jina v3, EmbeddingGemma-300M.

### Stage 1 — embedding swap (naive chunking held fixed; dense model = only variable)

| Hard gate | NDCG@10 | R@10 | R@50 ceiling |
|---|---|---|---|
| **Jina v5-small (kept)** | **0.8021** | 0.930 | 0.96 |
| Qwen3-Embedding-0.6B | 0.7900 | 0.910 | 0.955 |
| EmbeddingGemma-300M | 0.7695 | 0.890 | 0.930 |

**Decision: KEEP Jina v5.** Both candidates lose on the hard gate AND neither lifts the R@50 ceiling
— a "better" embedder retrieves *no more* of the 8 never-retrieved positives, so the swap doesn't
touch the recall problem it was meant for. Easy gate is a three-way tie (~0.9394, R@10 0.980) —
vocab-matched queries are BM25-carried, so the easy set is insensitive to the dense model.
**Mechanism:** Jina v5 is *retrieval-specialized* (`…-text-small-retrieval`, query/passage LoRA) and
is itself built on a Qwen3-0.6B backbone — effectively "Qwen3 + heavy retrieval finetuning," which
beats raw Qwen3-Embedding and general-purpose Gemma on technical-retrieval queries. **MTEB rank ≠
corpus performance.** Retroactively validates the v6.x embedder choice.

> ⚠️ **v6.4 audit caveat — the Qwen3 half of this decision is unproven.** The Qwen3 gap is **0.012**,
> which the [gate audit](gate-audit.md) shows is *below the gate's discrimination floor* (paired MDE
> ≥0.0195 at 80% power even at ρ=0.95; never significance-tested). So Qwen3 is **not reliably worse —
> it's indistinguishable on this gate**, status INCONCLUSIVE pending a paired re-test. Gemma's 0.033
> gap is above the floor, so *that* rejection holds. "Keep Jina v5" still stands as the safe default
> (no evidence to switch), but "Qwen3 is worse" is not established.
>
> ✅ **v6.6 update — re-tested with a paired test on gate v2.** Qwen3 Δ=−0.0084 (n.s., CI upper ≈ 0,
> no upside); **Gemma's gap shrank 0.033→0.011** (the single-label gate had inflated it) and is now
> only *borderline* worse, not a clear reject. Decision unchanged in practice (keep Jina v5), but the
> table above overstates both gaps. See the v6.6 Retest Sweep section.

### Stage 2 — late chunking (the pooling fork)

Late chunking pools each child's token span from the **parent's** contextualized token embeddings —
context without an LLM call. It requires **mean pooling**: incompatible with the **last-token** pooling
of Jina v5 *and* Qwen3-Embedding (Jina staff confirmed v5 dropped late-chunk support for exactly this).
So "best embedder" and "late chunking" are a fork. Tested on the two mean-pool options — Gemma (manual
span-pool, raw-mean space) and Jina v3 (native task-LoRA late chunking). Mechanism unit-tested first:
100% span-location, cos(late, naive-isolated)=0.94 — no alignment bug.

| late − naive (same model/space) | NDCG@10 | R@50 ceiling |
|---|---|---|
| Gemma, with rerank (gate) | **−0.026** | −0.025 |
| Gemma, fusion-only | **−0.064** | −0.025 |
| Jina v3 **native**, fusion-only | **−0.089** | **−0.085** (0.955→0.870) |

**Decision: REJECT late chunking.** Net-negative every way measured — and it *lowers the R@50 ceiling*,
pushing the precise chunk out of the top-50 entirely (not just mis-ranking it). Jina's purpose-built
**native** late chunking hurts the *most* → it's the method meeting the corpus, not a sloppy
implementation. **Mechanism = ConTEB / "Context is Gold"** (arXiv 2505.24782): context-pooling
regresses on extractive/technical-term corpora (their COVID-QA −21 nDCG) because it dilutes the
rare-term signal that jargon queries (`MVCC`, `WAL`, `PyO3`, `nonce`) match on. Our corpus is that
regime; late chunking wins the *opposite* regime (pronoun/reference-heavy chunks needing neighbors).

### Jina v3 non-deployable (independent of quality)

v3 needs transformers **<5.0** (else `all_tied_weights_keys`); the Ettin reranker (ModernBERT) needs
transformers **≥5.x** (`TokenizersBackend`); production Jina v5 needs **≥5.x**. So v3 cannot share a
process with the reranker *or* with v5 — non-shippable in our pipeline regardless of retrieval quality.
(v3-naive fusion 0.6237 / ceiling 0.955 is ~v5-class as a pure retriever, but moot.) Building/eval'ing
v3 required an isolated transformers-4.49 env on both the 5080 and the Mac.

### Reusable infra retained (production-neutral, default-off)

The experiment left clean infrastructure that does **not** change shipped behavior (defaults = Jina v5 /
`developer_knowledge`): `embedder.py` (model registry + per-model encode adapter — prevents the Qwen3
silent-`task=`-ignored footgun, single source of truth for index + query encoding), env-overridable
`EMBEDDING_MODEL/EMBEDDING_DIM/QDRANT_DIR/COLLECTION`, parameterized `remote_index.py`, and
`remote_index_lc.py` (late-chunk / raw-mean indexer). The next embedding experiment is now a one-env-var
change. **Net: both 2025–26 frontier levers measured negative on this corpus — the well-engineered
fundamentals (hybrid + Ettin rerank + structural parent/child chunking) are at/near the ceiling, and
context-adding tricks are the wrong tool for a self-identifying jargon corpus.**

## Gate Audit (v6.4 — audit the evaluator itself, 2026-05-30)

Every prior decision in this file was gated on the golden set. So before trusting it for one more
round, we audited the gate against IR test-collection criteria. Scope: **diagnose + significance**
(not rebuild). Method: deterministic coverage/power stats (`audit_prep.py`) + a paired-significance
harness (`significance.py`, with per-query dumps now persisted via `eval_retrieval.py --per-query`) +
an 8-agent LLM relevance-judging workflow over a stratified 54-query sample. Full report card in
[gate-audit.md](gate-audit.md). **Overall grade: D+.**

**The decisive finding — the gate cannot resolve the deltas it has been adjudicating.** Per-query
NDCG σ = **0.3111** (single-positive NDCG is near-bimodal), n=200. The paired minimum-detectable
effect at 80% power is **≥0.0195 even at an implausible ρ=0.95**, and ~0.045–0.055 at realistic
ρ≈0.6–0.8. The gate has been used to decide gaps of ~0.01–0.012. So:

- **v6.3's embedding-upgrade rejection (0.802 vs 0.790, gap 0.012) is downgraded to INCONCLUSIVE.**
  It is below the floor and was never significance-tested. (Gemma's 0.033 and late-chunking's
  −0.06…−0.09 *are* above the floor — those rejections stand.)
  *→ Resolved in v6.6 (paired re-test): Qwen3 Δ=−0.008 still n.s. but no upside; the gap shrank under
  graded labels. See the v6.6 Retest Sweep section below.*
- **Rule going forward:** never accept/reject on an aggregate delta alone. Require a paired bootstrap
  + sign/Wilcoxon p<0.05 (`significance.py`). The harness and per-query persistence now exist.

**Other findings:** (1) **Label incompleteness** — 100% single-positive over 48k chunks; **44.4%** of
sampled queries retrieved an unlabeled-but-relevant chunk (mean 0.963/query), so NDCG/MRR are
*understated* and the 0.96 R@50 ceiling + "8 never-retrieved misses" may be **label artifacts, not
corpus gaps**. (2) **Query realism LOW** — 100% long why/how questions (median 27 words), **0/200**
short-keyword, **1/200** error-string, **0/200** navigational; the gate under-tests the BM25/sparse
leg and terse-input reranking that real Claude Code queries trigger. (3) **Power skew** —
distributed n=6, databases n=13 are too thin to gate per-domain (retroactively weakens confidence in
v6.2's per-domain reranker split). (4) **The gate's one clear strength:** the hardening step really
does suppress lexical leakage (query↔source overlap 0.65→0.31), so it's a *clean* test of dense
semantic matching — just of that one narrow construct.

**Decision:** the gate stays in service (it reliably catches large regressions and full-book gaps),
but its remit is narrowed: **trust it only for deltas above ~0.02 with a paired test; treat sub-0.02
verdicts as inconclusive.** Remediations (pooled multi-positive labels, new query archetypes, hard
negatives, n→500, paired-test rule) are recorded in the report card, **not yet executed** — this
round was diagnosis. Resource-acquisition thread (IR-eval theory into the KB) deferred pending these
findings, which now tell us exactly what to add.

## Gate v2 — multi-positive re-labeling + missing archetypes (v6.5, 2026-05-31)

The v6.4 audit graded the gate D+ and named the fixes. This round executed two of them (the third,
ingesting IR-eval resources, was deferred). **Production serving path unchanged throughout** — this
is all eval-harness work.

**1. Multi-positive graded labels (fixes label incompleteness).** TREC-style pooling: for each query,
union top-20 parents from 3 retrieval legs off the live index — dense-only, BM25-only, HyDE-dense
(`pool_and_judge.py`; plain fusion dropped as redundant with dense∪bm25). deepseek-v4-flash grades
each candidate 0/1/2 (one batched call/query); gold forced to 2. Result: **mean 2.8 positives/query**
(hard), 2.3 (easy), 3.4 (archetypes). The judge prompt needed a strict rewrite — the first pass
over-labeled at mean 12/query because every pooled candidate is topically related *by construction*;
adding "topical relatedness is NOT relevance, when in doubt choose 0" fixed it.

**2. Judge validated before trusting labels (Phase B gate — PASSED).** Claude blind-regraded 60
grade-stratified triples (`judge_validation_workflow.js`, 6 agents). **within-±1 agreement 100%**,
deepseek precision/recall on the relevant (≥1) decision **0.95 / 0.905**. Exact agreement 68% — the
disagreement is almost all the 1↔2 boundary (deepseek conservatively grades some perfects as partial),
which doesn't create false positives. Verdict: labels trustworthy, no re-pool.

**3. Three new query archetypes (fixes realism).** +240 queries (`gen_archetypes.py`), 80 each:
keyword (terse 2–6 tok), error_symbol (exact `backtick` API/error lookups, seeded from
`chunk_type=="code"`), navigational ("where is X defined"). Main gate **200 → 440**.

**4. Graded eval (`eval_retrieval.py`).** `ndcg_at_k` takes a parent→grade map, gain `2^g-1`, IDCG
over ALL positives (correct multi-positive ideal); reduces exactly to old binary when all grades=1
(unit-tested: 0.919721 both). Added `--by-archetype`.

**Measured (Jina v5 on gate v2):**

| baseline stratum (same 200 hard q) | NDCG@10 | R@10 | MRR | R@50 |
|---|---|---|---|---|
| v1 single-label | 0.8021 | 0.930 | 0.761 | 0.960 |
| v2 multi-positive | 0.801 | **0.980** | **0.875** | **0.995** |

**Audit prediction confirmed: incompleteness deflated recall/MRR/ceiling, NOT NDCG** (the system
already ranked the gold high; it was being penalized for *also* surfacing other valid chunks). The
0.96 R@50 ceiling and "8 never-retrieved misses" were **mostly label artifacts** — true ceiling 0.995.

Per-archetype (n=440, aggregate NDCG 0.7939): error_symbol **0.822**/R@10 1.0, keyword 0.778/R@10 1.0,
**navigational 0.763/R@10 0.925 ← the real gap**. Counter-intuitive and previously invisible: the
BM25 leg handles keyword/symbol queries *well*; **navigational (locating) intent is the weakest** —
the actionable finding the old 100%-why/how gate could never produce.

**Discrimination floor improved ~2×** (`power_analysis_v2.json`): per-query NDCG σ 0.311→**0.244**,
marginal CI ±0.043→**±0.023**, paired MDE@80% at ρ=0.7 0.048→**0.025**. The gate now reliably resolves
**~0.025-class** deltas (was ~0.048). It still does NOT reach 0.015 at realistic ρ — that needs
n→~1000+ or ρ≈0.9. So the v6.4 rule holds, relaxed: **require a paired p<0.05; trust deltas ≥~0.025.**

**Promotion:** v2 sets are now canonical (`golden_queries_hard.jsonl` = 440 baseline+archetypes,
`golden_queries.jsonl` = 201 easy, both graded multi-positive); originals archived as `*_v1.jsonl`.
New reference baselines: `v65-baseline-mainv2.json` / `v65-baseline-easyv2.json`.

**Still deferred:** hard negatives + multi-hop (audit defect #4), n→1000 for 0.015 resolution,
domain-balance (distributed n=7 / databases n=17 still thin), and the navigational weakness now
quantified — all candidates for a future round.

## Index back-matter strip — the navigational "gap" was junk, and removing it is a real win (v6.7, 2026-05-31)

The v6.6 retest left one open thread: the v6.5 gate flagged navigational queries (NDCG 0.763) as the
weakest stratum. Diagnosing it before "fixing retrieval" (measure before you cut) revealed **two
artifacts, neither of which is weak retrieval**:

1. **Gate contamination.** 12 of 440 hard queries had a book **back-of-book INDEX page** as their only
   gold — `where is the index for dot product` → an alphabetical page-ref list (`dot product [21]...`).
   They scored ~0.33 and dragged the stratum down. The **70 legit navigational queries score 0.866
   (R@10 1.0)** — *better* than every other stratum. Dropped the 12 (gate 440→**428**); 2 were
   keyword queries with the same defect.
2. **Corpus bloat.** **6,276 chunks (12.7% of the corpus)** were back-of-book index pages (heading
   `Index (part N)`) that survived v6's front/back-matter stripping, across **15 books**. The v6 strip
   missed them because their indexes use varied link formats (EPUB `idIndexMarker`, `dx1-` anchors,
   bare page numbers) — but all share the TOC-assigned `Index (part N)` heading, the reliable signal.

**The fix (`scripts/strip_index_backmatter.py`).** Strip children whose heading top is exactly
`Index (part N)`; drop orphaned parents. Validated: every flagged region is a contiguous page-tail;
real content that merely starts with "Index" is **kept** (joy-of-crypto's *Index of Security
Definitions* — a curated definition reference; pandascookbook's *Index* — the pandas Index object).
Corpus **48,056 → 41,780**. Golden reconciled (`clean_golden_after_strip.py`): 12 dead queries
dropped, 18 partial pruned, 145 dead hard-negatives pruned.

**The result — first significant POSITIVE retrieval change in all of v6.x.** Re-indexed clean on the
5080; paired clean-index vs old-index on the same 428-query cleaned gate:

| metric | old-index | clean-index | Δ | verdict |
|---|---|---|---|---|
| **NDCG@10** | 0.8137 | **0.8212** | **+0.0075** [+0.0037,+0.0118] | **SIGNIFICANT** (sign p=.002, wilcox p=.0001, ρ=0.981) |
| keyword stratum | 0.791 | **0.821** | +0.030 | index pages were matching keyword queries & displacing real hits |
| Recall@10 | 0.9883 | 0.9907 | +0.0023 | n.s. |
| R@50 ceiling | 0.9977 | 0.9977 | 0.0 | identical — no real content lost |

Mechanism: the strip didn't change recall (ceiling identical) — it improved **ranking** by removing
6,276 incoherent index-page distractors from the reranker's candidate pool. Easy set: 0.9225, R@10
1.0, ceiling 1.0 (a −0.009 NDCG vs v6.5 is near-saturation reorder, zero recall loss). **Promoted to
production** (`data/qdrant`, old → `data/qdrant.v66.bak`). Indexer default batch **128→48** (128
reliably OOMs Jina v5 on the 15 GB 5080). **The lesson: the apparent navigational weakness was a
measurement artifact — but chasing it honestly surfaced a genuine 12.7% corpus-bloat bug whose removal
is the only significant retrieval gain the v6.x tuning ever produced.**

## Retest Sweep — paired re-adjudication of all v6.3 verdicts on gate v2 (v6.6, 2026-05-31)

Every v6.3 verdict was a **bare aggregate comparison** — never a paired significance test. With the
v6.5 gate (440 graded multi-positive queries) and a paired harness (`significance.py`, now also
reporting measured ρ + MDE@80%), we re-ran **all six** candidates against the saved Jina-v5 baseline
(`pq-v65-mainv2.jsonl`, aggregate NDCG **0.7939**). GPU candidates re-indexed on the 5080 over LAN
(Tailscale was down); evals run on the box (qdrant stays local, only per-query JSONL returns). The
`remote_index.py` batch-size starvation (`batch_size=4`) is fixed — now `KB_INDEX_BATCH` (128 for
Jina; **dropped to 32–48 for Qwen3 / enriched-text after batch-128 OOMs**). **Serving path untouched.**

**Verdict table** (paired, n=440 except HyDE n=200; SIGNIFICANT = bootstrap CI excludes 0 AND
min(sign,wilcoxon) p<0.05):

| Candidate | old Δ (single-label) | new Δ (graded) | 95% CI | ρ | MDE@80% | verdict |
|---|---|---|---|---|---|---|
| HyDE (optional) | ~−0.011 | −0.0081 | [−0.026,+0.007] | 0.895 | 0.023 | n.s. — keep optional (no-op) |
| Qwen3-Embedding-0.6B | −0.012 | −0.0084 | [−0.019,**+0.001**] | 0.917 | 0.0135 | n.s. — no upside → keep Jina |
| EmbeddingGemma-300M | −0.033 | −0.0107 | [−0.023,**+0.001**] | 0.875 | 0.017 | borderline (wilcox .043) |
| Per-domain rerank (shipped) | +0.003 | +0.001 vs heading; **−0.024 MRR vs child** | — | 0.95 | 0.011 | CONFIRMED |
| Late chunking | −0.026…−0.089 | −0.0164 | [−0.029,−0.005] | 0.884 | 0.018 | **SIGNIFICANT reject** (wilcox .045) |
| Enrichment | −0.0148 | −0.0241 | [−0.045,−0.003] | 0.600 | 0.031 | **SIGNIFICANT reject** (wilcox .024) |

**Bottom line: nothing overturned — the production config is vindicated.** Every candidate is ≤ Jina
v5, so serving stays exactly as shipped (Jina v5, no enrichment, no late-chunk, `descriptive-heading`
rerank, HyDE selective). Three findings of substance:

1. **Two rejections gained the p-value they never had.** Late chunking (raw-late vs raw-naive, same
   Gemma space) and enrichment are now *statistically* significant rejections — v6.3 only had eyeballed
   deltas. Enrichment again cratered databases (0.86→**0.605**), reproducing the v6.3 per-domain signature.
2. **The two embedding "rejections" SHRANK under graded labels** — Qwen3 0.012→0.008, Gemma 0.033→0.011.
   The single-label gate was **inflating embedding-candidate deficits**, the same incompleteness artifact
   that deflated baseline recall (v6.5). Both now sit just below the floor with CI upper bounds ≈ 0:
   *no upside, slight-worse trend*. "Keep Jina" is well-supported; "significantly worse" would need
   n≈1000–1250 at these ρ. The high ρ (0.88–0.95) for embedder swaps was as predicted — it dropped
   Qwen3's MDE to 0.0135, nearly resolving it.
3. **The per-domain reranker decision is confirmed for the right reason** — `descriptive-heading` ties
   global `heading` (n.s.) but significantly beats child-only on MRR (−0.024, wilcox .0018): it's the
   *heading-in-rerank-input*, not the per-domain split per se, that earns its place.

Reusable infra: `scripts/gpu_sweep.sh` (unattended index→eval driver for candidates). Per-query dumps
+ running notes at `data/golden/eval_runs/{pq-v66-*.jsonl, v66_retest_notes.md}`. Closes task #19.

## Rejected Approaches

### Embedding upgrade (Qwen3-Embedding-0.6B / EmbeddingGemma-300M) — v6.3 → re-tested v6.6
**Why considered:** 2025–26 frontier; higher open-MTEB than Jina v5; the open recall lever (8 never-retrieved).
**Why rejected (v6.3):** both scored below the hard gate (0.790 / 0.770 vs 0.802); neither lifted the R@50 ceiling.
**v6.4 audit correction:** the **Qwen3** rejection (0.012 gap) was downgraded to INCONCLUSIVE (below the gate's discrimination floor, never significance-tested).
**v6.6 paired re-test (resolved):** re-indexed both on the 5080, paired vs Jina on gate v2. **Qwen3 Δ=−0.0084** [−0.019,+0.001], ρ=0.917 — n.s., but CI upper bound ≈ 0 → *no upside*. **Gemma Δ=−0.0107** [−0.023,+0.001], wilcox p=0.043 — borderline. **Both gaps shrank vs v6.3** (0.012→0.008, 0.033→0.011): the single-label gate had *inflated* them. Net: **keep Jina v5 — no upside to either; "significantly worse" unproven (needs n≈1000+).** Gemma's "clear reject" is downgraded to "weakly worse." See [v66_retest_notes](../data/golden/eval_runs/v66_retest_notes.md).

### Late chunking (manual mean-pool + Jina v3 native) — v6.3
**Why considered:** "contextual chunks without LLM enrichment" — the survey's most novel lever, and the one that should sidestep the enrichment failure.
**Why rejected:** net-negative every measurement (−0.026 to −0.089), and it lowers the recall ceiling. ConTEB mechanism: context-pooling dilutes the rare-term signal on extractive/jargon corpora. Native v3 hurt most. Full record above.
**v6.6 paired re-test (strengthened):** Gemma raw-late vs raw-naive (same space, n=440) Δ=**−0.0164** [−0.029,−0.005], **wilcox p=0.045 → now statistically SIGNIFICANT** (v6.3 only had eyeballed deltas). Reject stands, reinforced.

### Corpus-wide HyDE (query→hypothetical-passage on the dense leg)
**Why considered:** standard fix for vocab-gap recall; the 8 never-retrieved hard misses looked like textbook HyDE wins.
**Why rejected:** measured net-negative on the hard gate (NDCG 0.7921 → 0.7809 augment / 0.7753 replace). Domain-split mechanism — helps ML-conceptual, drifts on Rust-precise — and Jina v5's 0.96 bare-query ceiling leaves too little recall headroom to offset the drift. Kept as a **selective caller-side param** (augment-mode), not a default. Full record in the v6.2 section above.
**v6.6 paired re-test (confirmed no-op):** on gate v2 baseline-200, HyDE Δ=−0.0081 [−0.026,+0.007], ρ=0.895 — n.s.; only 37/200 queries even changed (BM25 leg uses the original query). Confirms "keep as optional, not a default, not a significant regression."

### GraphRAG / LightRAG
**Why considered:** Cross-book concept linking (e.g., "regularization" in 8 books).
**Why rejected:** Our structured books + enrichment + hybrid search already handles this. LightRAG needs another LLM pass (~11 hours) for graph construction. Marginal quality gain for our query patterns (single-concept lookups, not multi-hop reasoning over unstructured corporate docs).
**Lighter alternatives if needed later:** Concept tagging during enrichment pass, entity-to-chunk side index via spaCy NER, cross-book "see also" links via embedding similarity.

### TurboVec
**Why considered:** Rust ANN index, 8-16x vector compression, faster than FAISS on M-series.
**Why rejected:** Pure dense index — no BM25, no hybrid, no metadata filtering. We'd lose everything Qdrant gives us. Only relevant at 100K+ vectors for memory pressure.

### SPLADE++ learned sparse retrieval
**Why considered:** Term expansion handles vocabulary mismatch ("overfitting" → "regularization", "dropout").
**Why rejected for now:** 532MB model vs zero-cost BM25. Available via same FastEmbed interface if BM25 quality proves insufficient. Dense retrieval already handles semantic matching.

### Grep replacing BM25 (Anthropic "grep beats RAG")
**Why considered:** Anthropic's Claude Code team found grep outperformed RAG for codebases.
**Why rejected for primary search:** That finding is specific to codebases with literal identifiers. Our queries are conceptual ("how does batch normalization work"). Added `grep_books()` as a complement for exact-match lookups, not a replacement.
