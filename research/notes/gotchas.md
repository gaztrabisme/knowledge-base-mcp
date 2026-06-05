# Gotchas & Known Issues

## Golden-Set Vocabulary Bias (measure-the-wrong-thing trap)

**Problem:** LLM-generating a benchmark query FROM the chunk it should retrieve makes the query inherit the chunk's rare terms, so BM25/dense match trivially. The v6 eval baseline came back near-ceiling (NDCG 0.93, Recall@50 1.00) — which looks great but means the set can't measure recall-improving changes like enrichment (recall is already maxed).
**Fix:** Harden with a paraphrase pass (`scripts/harden_golden_set.py`) that reframes each query as a developer problem/symptom and avoids the chunk's signature terminology. Verify it worked by measuring query→chunk content-word overlap (dropped 52%: 0.649→0.313). The hard set then showed real headroom (NDCG 0.79, Recall@50 0.96).
**Watch for:** any benchmark where positives are generated from the answer text — check lexical overlap before trusting near-perfect scores.

## DeepSeek API Transient Timeouts Under Fan-Out

**Problem:** A batch of ~200 concurrent calls all returned `Request timed out` even though single calls worked seconds earlier — a brief API window the SDK's default 2 retries didn't ride out. (Not balance/auth: those raise different errors.)
**Fix:** Build the client with `AsyncOpenAI(..., timeout=60.0, max_retries=5)` and keep concurrency modest (25 was ample; the account cap is 2500). Diagnose with a single-call smoke test — if that works, it's transient, just retry the batch.
**Note:** thinking must be disabled explicitly — `extra_body={"thinking": {"type": "disabled"}}` — it defaults to ENABLED on deepseek-v4-flash.

## PDF Heading Loss (RESOLVED in v6)

**Problem:** markitdown extracts text from PDFs but loses heading hierarchy. No `#` markers in the output markdown.
**v6 fix:** `convert_pdf_with_toc()` reconstructs headings from PyMuPDF's embedded TOC bookmarks (`doc.get_toc()`), injecting them at page-start positions. PDF-sourced books now carry chapter/section/heading_path. See decisions.md "PDF heading recovery".
**Caveat:** Only works if the PDF has TOC bookmarks (100% of the production set does). PDFs without a TOC fall back to markitdown (headingless, as before).

## PDF CID-Font Garble

**Problem:** Some PDFs embed subset fonts with no ToUnicode CMap. PyMuPDF `get_text()` maps those glyphs to garbage in the Armenian/Hebrew/extended-Cyrillic Unicode ranges — e.g. "PostgreSQL" → "PostgreԯԭԨ", "2022" → "ҁѿҁ҂". Only postgresql-14-internals (of 46 books) is affected; markitdown extracts it cleanly.
**Fix:** `ingest_pdf()` measures suspicious-glyph ratio against `_GARBLE_RATIO_THRESHOLD` (0.002). Above it, discard TOC headings and fall back to markitdown (clean body, no headings). General — any future CID-broken PDF self-heals.
**Trade-off:** That one book loses heading metadata but keeps clean, keyword-searchable text. Chose clean text over headings.

## Front-Matter Strip: "Chapter N" Appendix Misfire

**Problem:** `strip_front_back_matter` treats the first bare "Chapter N"/numeric heading as the start of real content. A book whose real chapters are descriptively titled but whose answer-key appendix uses "# Chapter N" headings would match the *appendix* and discard the whole book as front matter (creativeprojectsforrustprogrammers collapsed 713KB → 23KB, 26 chunks).
**Fix:** Position guard — only trust a numeric/"Chapter N" heading within the first 40% of lines; otherwise fall through to the descriptive-heading fallback.
**Watch for:** Suspiciously small markdown output (`wc -c data/markdown/*.md | sort -n`) is the tell. A 4MB EPUB yielding 23KB markdown means stripping ate the content.

## Tailscale Unreachable After GPU Wake

**Problem:** After Wake-on-LAN, the GPU box responds on its LAN IP (<gpu-host-lan>) but the Tailscale IP (<gpu-host>) can time out for a while (tailscaled re-establishing).
**Fix:** When on the same LAN, just SSH the LAN IP. Verify with `ping`/`ssh -o ConnectTimeout=3` before expensive transfers.

## macOS Has No `timeout`

**Problem:** Validation/wrapper scripts using `timeout 600 python3 ...` fail on macOS with "command not found: timeout".
**Fix:** macOS ships `gtimeout` (via coreutils) or nothing. Drop the wrapper for local one-shot scripts, or use `gtimeout` if coreutils is installed.

## numpy float32 Serialization

**Problem:** sentence-transformers CrossEncoder returns numpy float32 scores. JSON serialization fails on numpy types.
**Fix:** `search.py` wraps all scores in `float()` — in `SearchResult.to_dict()` and `_rerank()`.
**Watch for:** Any new code path that puts numpy values into dicts returned via MCP.

## optimum / transformers Version Conflict

**Problem:** ONNX acceleration for Ettin-150m (ModernBERT) blocked. `optimum` requires `transformers<4.58`, ModernBERT needs `>=5.2.0`.
**Impact:** Cannot use `backend="onnx"` in sentence-transformers CrossEncoder for Ettin models.
**Workaround:** PyTorch at 0.17s/50-docs is fast enough. No ONNX needed.
**Future fix:** Revisit when `optimum` supports transformers 5.x.

## Remote Script Duplication

**Problem:** `scripts/remote_index.py` is self-contained (no package imports) for rsync deployment. It duplicates BOOKS dict and BM25/embedding logic from the main package.
**Impact:** When adding new books or changing indexing logic, both `knowledge_base/index.py` AND `scripts/remote_index.py` must be updated.
**Mitigation:** Both files now use the same FastEmbed BM25 approach, eliminating the tokenizer divergence. BOOKS dict still duplicated — update both when adding books.

## GPU OOM During Indexing

**Problem:** Jina v5 (677M params) can OOM on 16GB GPU if other processes are using VRAM, or with large batch sizes on long texts.
**Fix:** Check `nvidia-smi` before indexing. Kill stale processes. Use batch_size=4 for large books (e.g., Boneh-Shoup crypto at 2,761 chunks). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
**Pattern:** The first indexing run after boot works fine (fresh GPU memory). Subsequent runs may OOM if previous model wasn't freed.

## Qdrant "Already Indexed" Check

**Problem:** `index_book()` and `remote_index.py` check if a book is already indexed by scrolling for any point with matching `book` field. If you need to re-index a book (e.g., after re-chunking), you must delete its points first.
**Fix:** Delete points before re-indexing:
```python
client.delete(
    collection_name="developer_knowledge",
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[
            models.FieldCondition(key="book", match=models.MatchValue(value="book-slug"))
        ])
    ),
)
```

## Enrichment Buffering

**Problem:** `nohup` buffers stdout. Remote enrichment logs may be empty for 30-60s after launch.
**Fix:** Use `python3 -u` (unbuffered) in the nohup command. Check `ps aux | grep python` to confirm process is alive before assuming failure from empty logs.

## Qdrant File Lock

**Problem:** Qdrant file-based storage allows only one client at a time. If the MCP server has the index open and you try to run `test_search.py` simultaneously, one will fail.
**Fix:** The MCP server opens/closes the client per query (`get_client()` → use → `client.close()`). Scripts do the same. Don't leave a Python REPL with an open client.

## llama-server OOM During Enrichment

**Problem:** Qwen3.6-27B MTP on RTX 5080 (16GB) can get OOM-killed (exit 137) during long enrichment runs. The model uses ~14GB VRAM; sustained inference can push past limits.
**Fix:** Enrichment script saves every 50 chunks, so it's resumable. Restart the container (`docker compose --profile llama-cpp-s016 up -d`), kill stale enrichment process, relaunch — it skips already-enriched chunks.
**Watch for:** Check `docker ps` before assuming enrichment is running. Exit 137 = kernel OOM kill.
