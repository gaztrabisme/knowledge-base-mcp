# research/ — evaluation & experimentation apparatus

**You do not need anything in this directory to use the tool.** This is the author's
retrieval-quality lab: how the defaults (embedder, reranker, chunk sizes, fusion) were
chosen and validated.

> ⚠️ These scripts reflect a **specific 56-book corpus** and a **particular GPU
> workstation**. Several hardcode SSH hosts, local LLM endpoints, and a DeepSeek API
> key read from the environment. They are archival references, not turnkey tools —
> parameterize paths/endpoints before reuse.

## `eval/` — the retrieval gate

A golden-set benchmark (graded relevance labels) scored on NDCG@10 / Recall@10 / MRR,
with a paired significance harness so changes are accepted only when they beat noise.

- `eval_retrieval.py` — run the golden queries through the live pipeline, score metrics.
- `build_golden_set.py`, `harden_golden_set.py`, `gen_archetypes.py` — author the query set.
- `pool_and_judge.py`, `judge_agreement.py`, `judge_validate_prep.py` — TREC-style
  pooling + LLM relevance judging + judge validation.
- `significance.py` — bootstrap CI + sign/Wilcoxon paired tests.
- `bench_latency.py` — search-path latency.
- `scrub_golden.py` — strips verbatim copyrighted excerpts from the golden files.
- `golden/` — the published golden set. **Copyrighted `source_text` excerpts have been
  stripped** (queries, chunk-id labels, grades, and metrics are retained), so it is
  illustrative of the methodology but not reproducible without the original corpus.

## `experiments/` — one-off investigations

GPU sweeps, tokenizer/cutoff calibration, PDF-heading extraction comparisons, reranker
backend benchmarks, enrichment trials, and various diagnostics tied to the author's
corpus journey. Read them for rationale; don't expect them to run unmodified.
