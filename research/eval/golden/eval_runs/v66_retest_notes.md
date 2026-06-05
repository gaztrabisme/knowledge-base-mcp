# v6.6 Retest — Paired Re-adjudication on Gate v2 (running notes)

Baseline reference: `pq-v65-mainv2.jsonl` (Jina v5, 440 hard). Paired tests via `significance.py`
(now reports rho + MDE@80%). Gate metric = NDCG@10 at parent granularity, graded labels.

## Verdicts (accumulating)

### HyDE (local) — CONFIRMED n.s. (keep as optional param)
- Arms: no-HyDE baseline-200 (0.8014) vs HyDE-200 (0.7934). Baseline stratum only (HyDE passages
  exist for the 200 why/how queries; archetypes have none).
- Δ NDCG = **−0.0081**, CI [−0.0255, +0.0068] straddles 0 → **noise (n.s.)**. ρ=0.895, MDE@80%=0.0232.
- Only 37/200 queries changed (HyDE perturbs ranking weakly; BM25 leg uses original query).
- vs v6.3 (~−0.011 "net-negative, kept optional"): direction reproduced, magnitude below floor.
  **Stands: optional param, not a significant regression.**

### Qwen3-Embedding-0.6B (GPU re-index, n=440) — STILL-INCONCLUSIVE, but "keep Jina" now supported
- Re-indexed 48,056 vec on the 5080 (batch 32 after OOM at 128). Eval on the box. NDCG 0.7856.
- Paired vs Jina-v5 baseline (0.7939): Δ NDCG = **−0.0084**, CI **[−0.0186, +0.0005]**, ρ=**0.917**,
  MDE@80%=**0.0135**, sign p=0.128, wilcox p=0.118 → **noise (n.s.)**.
- The high ρ (as predicted) nearly resolved it: MDE fell to 0.0135 but the gap is smaller (−0.0084).
  KEY: CI upper bound ≈ 0 (+0.0005) → **no upside, slight-worse trend**. Recall/MRR/ceiling all n.s.-down.
- Per-archetype: nav 0.763 / kw 0.775 / err 0.820 — same profile as Jina, no stratum where Qwen3 wins.
- **Verdict:** "significantly different" UNPROVEN (would need n≈1250 at this ρ to resolve 0.008), but
  the practical decision **STAY ON JINA V5 is now well-supported** — zero evidence of benefit, weak
  evidence of slight harm. Upgrades v6.3's eyeballed-0.012 and v6.4's "inconclusive" to a rigorous
  bounded estimate. (Originally measured 0.012 worse on single-label; graded gives −0.008.)

### EmbeddingGemma-300M (GPU re-index, n=440) — DOWNGRADED from "clear reject" to "weakly worse"
- NDCG 0.7833. Paired vs Jina: Δ=**−0.0107**, CI **[−0.0228, +0.0005]**, ρ=0.875, MDE@80%=0.0166,
  wilcox p=**0.043** (sign p=0.056). CI includes 0 → **noise (n.s.)** by the AND-rule, but right at
  the edge (trending worse, wilcoxon alone would flag it).
- **The headline correction:** Gemma's gap **shrank 0.033 → 0.011** moving from single-label to graded
  labels. v6.3 "rejected, 0.033 above floor" was **inflated by single-label incompleteness** (same
  artifact that deflated baseline recall). Under proper labels Gemma is only marginally worse.
- **Verdict:** keep Jina (no upside, slight-worse trend), but v6.3's *clear* rejection is overstated;
  honest status is "weakly worse, not decisively separable."

### Per-domain reranker: heading (global) vs descriptive-heading (shipped) — n.s. WASH
- Δ=+0.0013, CI [−0.0059, +0.0087], ρ=0.949, MDE@80%=0.0105 → indistinguishable. The v6.2 split
  (+0.0034 on single-label) is NOT a real win under graded labels — but not a loss either.
- child-only (pre-⚡1) vs descriptive-heading: NDCG Δ=−0.0120 (CI [−0.0237,−0.0008] excludes 0, but
  sign/wilcox n.s. — symmetric per-query churn), **MRR Δ=−0.0242 SIGNIFICANT** (wilcox p=0.0018).
- **Verdict: shipped decision CONFIRMED.** Heading-in-reranker-input genuinely helps vs child-only
  (significant MRR gain); the per-domain `descriptive-heading` refinement is ≥ global `heading` (wash)
  and strictly better than child-only. Keep `descriptive-heading`.

### Late chunking (Gemma raw-late vs raw-naive, same space, n=440) — CONFIRMED reject, now SIGNIFICANT
- raw-naive 0.7724 → raw-late 0.7560. Late-chunk EFFECT: NDCG Δ=**−0.0164**, CI **[−0.0293, −0.0048]**
  (excludes 0), **wilcox p=0.045 → SIGNIFICANT**. MRR −0.010 (n.s.), recall/ceiling flat.
- v6.3 rejected on eyeballed −0.026/−0.064; now it carries a paired p<0.05. ConTEB mechanism
  (context-pooling dilutes rare-term signal on extractive/jargon corpora) holds.
- **Verdict: reject CONFIRMED and strengthened** (first time it's statistically significant).
- (Native Jina-v3 `jina-late` not re-run — Gemma raw-late is the representative mechanism test; v6.3
  found native v3 hurt most, so re-running it would only reinforce.)

### Enrichment — FIRST RUN INVALID (batch-128 OOM, indexed 21,215/48,056 = partial → ceiling 0.525).
- Re-run at batch 48 (enriched chunks carry longer context_text → Jina OOMs at 128): 48,009 vec,
  ceiling 0.986 (healthy). NDCG 0.7699. Paired vs Jina: Δ=**−0.0241**, CI **[−0.0453, −0.0025]**
  (excludes 0), **wilcox p=0.024 → SIGNIFICANT**. ρ=0.600 (enrichment perturbs embeddings more),
  MDE@80%=0.0305. databases cratered 0.86→**0.605**, security −, matching v6.3's per-domain pattern.
- **Verdict: reject CONFIRMED and strengthened** (v6.3 was −0.015 eyeballed; now p<0.05). Stay
  index-now-enrich-never.

## Summary — all six re-adjudicated (baseline Jina v5 = 0.7939, n=440)

| Candidate | old Δ (single-label) | new Δ (graded, paired) | ρ | verdict |
|---|---|---|---|---|
| HyDE (optional param) | ~−0.011 | −0.0081 (n=200) | 0.895 | n.s. — keep optional (no-op) |
| Qwen3-Embedding-0.6B | −0.012 | −0.0084 [−0.019,+0.001] | 0.917 | n.s. — INCONCLUSIVE, no upside → keep Jina |
| EmbeddingGemma-300M | −0.033 | −0.0107 [−0.023,+0.001] | 0.875 | borderline (wilcox .043) — DOWNGRADED from clear-reject |
| Per-domain rerank (shipped) | +0.003 vs heading | +0.001 vs heading; **−0.024 MRR vs child (sig)** | 0.95 | CONFIRMED (heading-in-input helps) |
| Late chunking | −0.026…−0.089 | −0.0164 [−0.029,−0.005] | 0.884 | **SIGNIFICANT reject** (wilcox .045) — strengthened |
| Enrichment | −0.0148 | −0.0241 [−0.045,−0.003] | 0.600 | **SIGNIFICANT reject** (wilcox .024) — strengthened |

**Bottom line: nothing overturned; production config vindicated.** Every candidate ≤ Jina v5 →
serving path unchanged (Jina v5, no enrichment, no late-chunk, descriptive-heading rerank, HyDE
optional). Two rejections (late-chunk, enrichment) now carry paired p<0.05 they never had. The two
embedding "rejections" SHRANK markedly under graded labels (Qwen3 0.012→0.008, Gemma 0.033→0.011) —
the single-label gate was **inflating embedding-candidate deficits**, the same incompleteness artifact
that deflated baseline recall. Embedding swaps are not worth it (no upside, ~0.01 worse at most), but
calling them "significantly worse" would need n≈1000–1250 at these ρ.

## v6.7 — index back-matter strip (the navigational thread → a real win)

The retest's one open gap (navigational 0.763) was diagnosed as **two artifacts, not weak retrieval**:
1. **Gate contamination:** 12 of 440 hard queries (10 nav + 2 keyword) had a book back-of-book INDEX
   page as their only gold ("where is the index for dot product"). Dropped them (gate 440→428). The
   64→70 legit nav queries score **0.866** (R@10 1.0) — better than every other stratum.
2. **Corpus bloat:** 6,276 chunks (12.7%) were back-of-book index pages (heading `Index (part N)`)
   that survived v6 stripping, across 15 books. Stripped via `strip_index_backmatter.py` (validated:
   contiguous tails; kept real content like joy-of-crypto's "Index of Security Definitions" and
   pandas' "Index" object section). Corpus 48,056 → **41,780**.

**Re-indexed clean on the 5080; paired clean-index vs old-index on the same 428-query cleaned gate:**

| metric | old-index | clean-index | Δ | verdict |
|---|---|---|---|---|
| NDCG@10 | 0.8137 | **0.8212** | **+0.0075** [+0.0037,+0.0118] | **SIGNIFICANT** (wilcox p=0.0001, ρ=0.981) |
| keyword stratum | 0.791 | **0.821** | +0.030 | index pages were displacing keyword hits |
| Recall@10 | 0.9883 | 0.9907 | +0.0023 | n.s. (recall not hurt) |
| R@50 ceiling | 0.9977 | 0.9977 | 0.0 | identical — no content lost |

**First statistically significant POSITIVE retrieval change in all of v6.x.** Removing index junk
improved *ranking* (fewer distractors in the reranker pool), didn't touch recall, and shrank the
index 12.7%. Found by chasing a measurement artifact, not a retrieval-architecture lever.

## v6.7.1 — gate restore (428 → 440)

The v6.7 strip dropped 12 hard-gate queries (2 keyword + 10 navigational) whose only gold was an
index page. Regenerated replacements from the **clean** corpus via `regen_dropped_queries.py`
(reuses `gen_archetypes` helpers; seed 1234 for a fresh shuffle, dedup vs all 440 prior queries, +a
`index for…`/`Index (part N)` nav-junk reject so the artifact can't recur). Pooled+judged on the
clean production index (`pool_and_judge.py`): mean **3.0 positives/query**, validated **0 index
golds / 0 dead positives**, every row anchored by a grade-2.

Gate **428 → 440**, archetype balance back to **200/80/80/80**. Re-baseline on the clean index:

| metric | 428 (v6.7) | 440 (v6.7.1) | note |
|---|---|---|---|
| NDCG@10 | 0.8212 | **0.8218** | +0.0006 — noise; cosmetic restore, no verdict moves |
| R@10 | 0.9907 | 0.9886 | flat |
| R@50 ceiling | 0.9977 | 0.9955 | flat |
| keyword (n) | 0.791 (78) | 0.824 (80) | 2 new kw scored high |
| navigational (n) | 0.866 (70) | 0.861 (80) | 10 new nav ≈ stratum |

Paired reference `pq-v67-clean-hard.jsonl` regenerated at n=440. The 440 gate (balanced, valid) is
canonical; production index unchanged.
