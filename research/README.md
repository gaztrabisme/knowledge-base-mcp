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

## `notes/` — findings & decision log

The narrative behind the defaults — *why* each knob is set the way it is. This is the
most useful part to read if you want to understand or tune the pipeline:

- `decisions.md` — every technical decision with its context, the rejected alternatives,
  and post-mortems on reverted directions (enrichment, late-chunking, embedder swaps).
- `gate-audit.md` — a rigorous audit of the evaluation gate itself (statistical power,
  label completeness, query realism) — how the benchmark was validated before trusting it.
- `rag-frontier-2026.md` — a survey of current retrieval techniques mapped onto this system.
- `architecture.md` — pipeline design and data flow.
- `dspy-enrichment-plan.md` — the contextual-enrichment experiment (and why it was dropped).
- `gotchas.md` — failure patterns and watch-fors.
- `ops-runbook.md` — the author's remote-GPU indexing/enrichment workflow (host/IP
  identifiers scrubbed to `<gpu-host>` placeholders; adapt to your own box).
- `index.md` — catalog of the above.

These are historical lab notes for a specific 56-book corpus; figures and version numbers
refer to that corpus, not to anything you build with the tool.

## `experiments/` — one-off investigations

GPU sweeps, tokenizer/cutoff calibration, PDF-heading extraction comparisons, reranker
backend benchmarks, enrichment trials, and various diagnostics tied to the author's
corpus journey. Read them for rationale; don't expect them to run unmodified.
