# Audit Report Card — Retrieval "Golden Gate" (`data/golden/`)

> v6.4 audit round, 2026-05-30. Method: deterministic coverage/power stats
> (`scripts/audit_prep.py`) + a paired-significance harness (`scripts/significance.py`) +
> a parallel LLM relevance-judging workflow over a stratified 54-query sample
> (`scripts/gate_audit_workflow.js`, 8 agents). Scope was **diagnose + significance**, not
> rebuild — remediations below are recommendations, not yet executed.
>
> **Headline:** the gate is a clean instrument for ONE narrow construct but has been used to
> adjudicate sub-0.02 NDCG deltas it provably cannot resolve. **v6.3's embedding-upgrade
> rejection (0.802 vs 0.790) is below the gate's detection floor and is downgraded to
> _inconclusive, pending a paired re-test_** — see [decisions.md](decisions.md).

---

## ✅ Gate v2 (v6.5) — defects remediated (2026-05-31)

The audit below (D+) drove a rebuild. Two of the three top defects are now fixed; the report card
beneath is preserved as the diagnosis that motivated it.

**What changed:**
- **Multi-positive graded labels (fixes Label completeness D).** TREC-style pooling (3 legs:
  dense / BM25 / HyDE-dense off the live index) → deepseek graded each candidate 0/1/2 → **mean
  2.8 positives/query** (was 1). Gold forced to grade 2. `pool_and_judge.py`.
- **Judge validated (Phase B gate PASSED).** Claude blind-regraded 60 stratified triples:
  **100% within ±1**, deepseek precision/recall on the relevant decision **0.95 / 0.905**. Exact
  agreement 68% — disagreement is almost entirely the 1↔2 (partial vs perfect) boundary, a
  conservative direction that doesn't add false positives. `judge_agreement.json`.
- **3 new query archetypes (fixes Query realism D–).** +240 queries: 80 keyword (terse 2–6 tok),
  80 error_symbol (exact `backtick` API/error lookups), 80 navigational ("where is X defined").
  `gen_archetypes.py`. Main gate grew **200 → 440**; graded NDCG + `--by-archetype` added to
  `eval_retrieval.py`.

**Measured outcome (Jina v5 baseline on gate v2):**

| | v1 gate (single-label, n=200) | v2 gate baseline stratum (multi-pos, n=200) |
|---|---|---|
| NDCG@10 | 0.8021 | 0.801 (flat — system already ranked well) |
| Recall@10 | 0.930 | **0.980** |
| MRR | 0.761 | **0.875** |
| R@50 ceiling | 0.960 | **0.995** |

→ **The audit's core prediction confirmed: single-label incompleteness deflated recall/MRR/ceiling,
not NDCG.** The "8 never-retrieved misses" / 0.96 ceiling was **mostly a label artifact** — true
ceiling is 0.995.

**Per-archetype (n=440 main gate, aggregate NDCG 0.7939 / R@10 0.977 / R@50 0.989):**

| stratum | n | NDCG@10 | R@10 | note |
|---|---|---|---|---|
| baseline (long why/how) | 200 | 0.801 | 0.980 | — |
| error_symbol | 80 | 0.822 | 1.000 | BM25 nails exact symbols |
| keyword | 80 | 0.778 | 1.000 | terse lookups retrieve fine |
| **navigational** | 80 | **0.763** | **0.925** | **the real gap** — locating-intent is weakest |

→ Surprise the old monoculture gate could never surface: keyword/error queries retrieve **well**
(the BM25 leg does its job); **navigational** ("where is X / which chapter") is the genuine soft spot.

**Discrimination floor — improved ~2×** (`power_analysis_v2.json`): per-query NDCG σ **0.311→0.244**,
marginal CI half-width **±0.043→±0.023**, paired MDE@80% at ρ=0.7 **0.048→0.025**. Honest caveat:
the gate now reliably resolves **~0.025-class** deltas (not 0.015 — that needs ρ≈0.9 or n→~1000+).
Easy gate v2: R@10/R@50 both **1.000**, NDCG 0.921 (graded multi-positive is a stricter ranking test).

**Still open (deferred):** hard-negative curation + multi-hop queries (defect #4), n→1000 for ~0.015
resolution, and the per-domain power skew (distributed n=7, databases n=17 still thin though improved).

---

**Auditor stance:** skeptical IR-evaluation expert. **Subject:** the 200-query "hard" golden set
(`data/golden/golden_queries_hard.jsonl`) plus its 201-query "easy" twin, used as the pass/fail gate
for every retrieval change in this project (e.g., the v6.3 embedding-upgrade rejection).

**Verdict in one line:** The gate is a *well-built measuring instrument for one narrow construct*
(dense semantic match on long, paraphrased, single-answer why/how questions) that has been *misused as
a general-purpose adjudicator of sub-0.02 NDCG deltas it provably cannot resolve.* The v6.3 rejection at
a 0.012 gap was never significance-tested and is below the gate's detection floor at every plausible
correlation level.

---

## Grades by criterion

| Criterion | Grade | One-line justification (with the number) |
|---|---|---|
| Construct validity | **C–** | Measures *single-target dense recall on verbose paraphrased questions*, not "retrieval quality." 100% of 200 hard queries are full sentences (word len min **15**, median **27**); not one sub-7-word query exists, so the BM25/sparse leg of the hybrid index is structurally under-tested. |
| Label completeness | **D** | Every query has exactly **1** positive against a **48,009**-chunk corpus (`single_positive_pct=100`). The stratified sample finds **44.4%** (24/54) of queries have ≥1 *unlabeled* relevant chunk retrieved (mean **0.963** per query). NDCG@10/MRR are systematically *understated*. |
| Label correctness | **C** | No gold is outright wrong (`gold_no=0`), but **18.5%** (10/54) of golds are non-unique and **14.8%** (8/54) had a retrieved chunk judged *better than gold*. "Single correct answer" is a fiction for ~1 in 5 queries. |
| Query realism / external validity | **D–** | `realism_grade: low`. **88%** of queries are why/how, **50%** inject a first-person "my", and generated-from-answer leakage survives paraphrase (q[10], q[30]). Backticked identifiers dropped from **38→8** queries; exact-error lookups **1/200**; navigational **0/200**. Tests one synthetic archetype, not the agent workload. |
| Coverage / stratification | **C+** | Book coverage is excellent: **46/46** books represented, **0** missing, median **4** queries/book. But domain balance is badly skewed — ml(**89**)+rust(**56**) = **73%**; distributed(**6**) and databases(**13**) are too thin for stable per-domain deltas. |
| Statistical power / discrimination | **D** | Marginal NDCG 95% CI half-width is **±0.0434** (per-query σ=**0.3111**, n=200). The v6.3 gap of **0.012** is **3.6×** below that. `power_analysis.json` shows the *paired* MDE at 80% power is **0.0195 even at ρ=0.95** — `detects_0.012: false` at *every* correlation level tested (0.0→0.95). |

---

## Detailed findings

### 1. Construct validity — what does the gate actually measure? (C–)
The gate is internally coherent but narrow. All 200 hard queries are long natural-language questions (`query_word_len`: min 15 / median 27 / max 46), first word why/how 88% of the time, with a single self-contained gold chunk and **zero** hard negatives (`queries_with_hard_negatives=0`, confirmed in-file: every record's `hard_negative_child_ids` is `[]`). This makes the gate a **single-target recall test on the dense embedding path**. It cannot, by construction, measure: (a) precision against near-duplicate chunks, (b) all-relevant recall (multi-positive), (c) the BM25/sparse leg (which terse/keyword/error queries exercise), or (d) reranker behavior on short input. The paraphrase step does genuinely work — hard-set vocab overlap is **0.31** vs **0.65** easy — so the gate is a *clean* test of semantic matching. It is just a test of one construct, mislabeled as a test of the system.

### 2. Label completeness — the dominant weakness (D)
A 1-positive-per-query design over 48k chunks is the classic incomplete-judgments failure mode (à la pre-pooling TREC). The audit sample quantifies the damage directly: **44.4%** of queries (24/54) retrieved an *unlabeled relevant* chunk, mean **0.963** unlabeled-relevant per query. Consequences that the gate's headline numbers hide:
- **NDCG@10 (0.8021) and MRR (0.7608) are floors, not estimates.** Any run that surfaces a true-but-unlabeled chunk above the lone gold is *penalized for being correct*.
- **The R@50 "ceiling" of 0.96 and the "misses" are label artifacts, not necessarily coverage gaps.** Of the 14 baseline misses (e.g., the REINFORCE policy-gradient query mapped to `hands-onartificialintelligenceforiot`), an unknown fraction are cases where the *system found a better passage in a different book* and the single-gold rubric scored it as a miss. Until misses are re-judged with pooling, "8 never-retrieved misses" is an unproven claim about the corpus.

### 3. Label correctness (C)
No gold was judged outright wrong (`gold_no=0`), which is reassuring. But uniqueness — the premise the NDCG/MRR math depends on — fails for **18.5%** of sampled queries, and in **14.8%** a retrieved non-gold chunk was judged *strictly better* than the gold. Combined with #2, the rank-sensitive metrics (NDCG, MRR) are the *least* trustworthy outputs of this gate; the set-based recall@k is comparatively robust.

### 4. Query realism / external validity (D–)
The realism assessment is damning and corroborated by the artifacts. The miss list itself contains the smoking gun: *"Why does my reinforcement learning agent's update rule involve multiplying the log probability of each action by the total reward…"* — a REINFORCE *derivation* from a textbook, reframed as a personal bug. Real developers hitting that math do not phrase it as "my agent." The "debugging frame" (50% inject "my", 46 follow "Why does my X…?") is an LLM paraphrase artifact, and generated-from-answer leakage survives it (q[30]'s gold is a `partial_name` Tera variable the query is reverse-engineered from). The workload gap is the core external-validity problem: an MCP search tool called by Claude Code overwhelmingly receives **short keyword/jargon phrases** (`rocket project layout`, `argon2 iteration count` — **0/200**), **verbatim error/symbol lookups** (**1/200**, code tokens paraphrased out 38→8), and **navigational "where is X" queries** (**0/200**). The gate tests *none* of the query classes that exercise the sparse + rerank machinery production actually triggers.

### 5. Coverage / stratification (C+)
The bright spot. Book coverage is complete (46/46, median 4 queries/book, range 2–14), so a regression that wipes out one book will almost certainly surface. The weakness is domain skew: ml+rust = 73% of mass, while distributed (n=6) and databases (n=13) are statistically too thin — a per-domain NDCG estimate on n=6 has a CI roughly ±0.25, so a real regression isolated to distributed systems can hide entirely inside sampling noise. (Note: `per_domain` is empty `{}` in the saved baseline report, so per-domain regression detection is currently not even wired up.)

### 6. Statistical power / discrimination — the decisive failure (D)
This is where the gate's load-bearing use collapses. With per-query NDCG σ = 0.3111 and n=200, the **marginal** 95% CI is **0.8021 ± 0.0434**. The gate has been used to adjudicate gaps of ~0.01–0.012. Stating the nuance correctly:

- The 0.012 v6.3 gap is **~3.6× smaller** than the marginal NDCG CI half-width (0.0434). A marginal CI is the *wrong* test for two correlated systems, so this alone does not prove 0.012 is noise.
- A **paired** bootstrap cancels per-query variance and resolves smaller gaps — but `data/golden/audit/power_analysis.json` computes the paired MDE across ρ ∈ {0, 0.5, 0.7, 0.9, 0.95} and reports `detects_0.012: false` at **every** level. Even at ρ=0.95 the paired MDE at 80% power is **0.0195** — still 1.6× the observed gap. So *even the optimistic paired analysis says 0.012 is below the floor.*
- **The v6.3 rejection was never significance-tested at all.** The saved runs (`audit-baseline-hard.json`, `emb-qwen3-hard.json`) store only aggregates (0.8021 vs 0.7900) and a misses list — **no per-query NDCG vectors were persisted** (fixed this round via `--per-query`), so a paired test cannot be reconstructed from the old artifacts. Running it requires re-indexing the Qwen3 candidate, which is currently remote-gated (open task #19).

**Honest framing:** 0.012 is *not established as significant* — it is "pending paired test," and all available evidence (marginal CI, and the paired MDE table that fails to detect it at ρ up to 0.95) points to *not significant*. It is not a *proven* null, but the burden of proof was never met before the candidate was rejected.

---

## Can the gate distinguish the ~0.01-NDCG deltas it has been used to adjudicate?

**No — not as currently run.** At n=200 with σ=0.31, neither the marginal CI (±0.043) nor the paired MDE (≥0.0195 even at ρ=0.95) reaches down to 0.012. The v6.3 decision to reject the embedding upgrade on a 0.802-vs-0.790 gap was **statistically unsound**: the difference is indistinguishable from sampling noise under every analysis the repo itself contains, and the upgrade may have been wrongly discarded. The gate *can* reliably resolve large deltas (the gemma run at 0.7695 is ~0.033 below baseline — within reach of a paired test), but it has been routinely operated below its own discrimination floor.

---

## Prioritized remediation (highest impact first)

1. **Run the paired test before any further rejections; persist per-query scores.** *(Harness now exists: `--per-query` + `significance.py`.)* Re-index the Qwen3 candidate (remote-gated) and retroactively test the v6.3 gap. Rule: never reject on aggregate deltas alone; require p<0.05 paired.
2. **Pool judgments and re-label to kill the 1-positive bias.** Pool top-k from all candidate systems run to date, judge the pool (LLM-assisted then spot-checked), allow **multiple graded positives**. Directly fixes the 44.4% unlabeled-relevant rate, un-deflates NDCG/MRR, and re-classifies "misses" into true coverage gaps vs label artifacts.
3. **Add the missing query archetypes (rebalance the construct).** ≥3 new strata, ~40–60 each: (a) short keyword/jargon (2–6 tokens), (b) verbatim error-string / exact-API-symbol lookups, (c) navigational "where is X defined." These exercise the BM25/sparse leg and reranker-on-terse-input that 0/200 current queries touch. Report metrics *per stratum*.
4. **Add hard negatives and multi-hop/multi-positive queries.** Curate `hard_negative_child_ids` (near-duplicate chunks) so the gate measures precision, not just single-target recall.
5. **Fix power for thin domains.** Raise distributed (6) and databases (13) to ≥30 queries each, or mark per-domain deltas there as "underpowered — do not gate on." Wire up the empty `per_domain` block. To reliably detect a ~0.015 paired gap at 80% power (ρ≈0.7, σ_Δ≈0.24), grow total n to ≈**500+**.
6. **De-synthesize / de-leak the queries.** Replace generated-from-answer items with queries authored *without* sight of the gold chunk (ideally from real Claude Code session logs). Stop paraphrasing code tokens/error strings *out*.

---

## Bottom line

As an instrument, the gate is carefully constructed and the paraphrase step honestly suppresses lexical leakage. As a **decision gate for fine-grained retrieval changes, it is not fit for purpose**: it tests one synthetic query archetype, deflates its own metrics via single-label incompleteness (44% unlabeled-relevant), and has been used to adjudicate deltas ~2–4× below its statistical detection floor. The v6.3 rejection should be treated as *unproven, pending a paired re-test*, not as a settled result. Fix significance-testing and label completeness first (remediations 1–2); they are the cheapest and most load-bearing.

**Overall grade: D+** — a valid measurement of a narrow construct, currently operated outside its discrimination envelope on incomplete labels.

---

## Appendix A — paired minimum-detectable-effect table (`power_analysis.json`)

Per-query NDCG σ = 0.3111, n = 200. Paired MDE at 80% power, α=0.05 two-sided, by assumed cross-system per-query correlation ρ. σ_Δ = σ·√(2(1−ρ)); MDE = (z_{α/2}+z_β)·σ_Δ/√n.

| ρ (v5 ↔ candidate) | σ_Δ (NDCG) | MDE @ 80% power | detects v6.3's 0.012? |
|---|---|---|---|
| 0.00 | 0.4399 | 0.0871 | no |
| 0.50 | 0.3111 | 0.0616 | no |
| 0.70 | 0.2409 | 0.0477 | no |
| 0.90 | 0.1391 | 0.0276 | no |
| 0.95 | 0.0984 | 0.0195 | no |

Even at an implausibly high ρ=0.95, the gate cannot resolve a 0.012 gap. Realistic retriever pairs sit ρ≈0.6–0.8 → MDE ≈ 0.045–0.055.

## Appendix B — audit harness (reusable)

```bash
# 1. per-query scores + retrieved-candidate dump (for significance + label judging)
PYTHONPATH=. python3 scripts/eval_retrieval.py --label <run> \
  --golden data/golden/golden_queries_hard.jsonl \
  --per-query data/golden/eval_runs/pq-<run>.jsonl

# 2. discrimination floor (single file) OR paired significance (two files)
python3 scripts/significance.py data/golden/eval_runs/pq-baseline-hard.jsonl
python3 scripts/significance.py pq-baseline-hard.jsonl pq-<candidate>-hard.jsonl

# 3. deterministic coverage/power stats + judge sample
PYTHONPATH=. python3 scripts/audit_prep.py

# 4. label-quality + realism audit workflow (8 agents) -> report card
#    Workflow tool: scriptPath scripts/gate_audit_workflow.js
```
