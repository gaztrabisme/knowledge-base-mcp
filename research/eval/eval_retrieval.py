"""Phase 1B — retrieval evaluation harness.

The tool-agnostic gate for every retrieval change (enrichment, reranker, chunking,
embeddings). Runs the golden query set through the REAL hybrid_search() pipeline and
scores ranking quality at PARENT granularity (the agent receives parent content, and
the pipeline dedups by parent, so parent-level is the meaningful unit).

Two views per query:
  - full pipeline  (rerank + adaptive cutoff, top_k=10) → NDCG@10 / Recall@10 / MRR
                     = what the agent actually gets.
  - retrieval ceiling (fusion only, no rerank, top_k=50) → Recall@50
                     = did we retrieve the positive at all? Separates "never found"
                       from "found but reranker/cutoff dropped it".

Usage:
  python3 scripts/eval_retrieval.py --label v6-no-enrichment
  python3 scripts/eval_retrieval.py --label v6-enriched   # after Phase 3, compare
"""

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

from knowledge_base.config import CHUNKS_DIR
from knowledge_base.search import get_client, hybrid_search

GOLDEN_PATH = CHUNKS_DIR.parent / "golden" / "golden_queries.jsonl"
RESULTS_DIR = CHUNKS_DIR.parent / "golden" / "eval_runs"
CEILING_K = 50


def child_to_parent_map() -> dict[str, str]:
    m = {}
    for f in glob.glob(str(CHUNKS_DIR / "*.json")):
        data = json.loads(open(f).read())
        for c in data["children"]:
            m[c["child_id"]] = c["parent_id"]
    return m


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_parents: list[str], grade_map: dict[str, int], k: int) -> float:
    """Graded NDCG@k. grade_map: parent_id -> relevance grade (>=1 for relevant).
    Gain = 2^grade - 1 (standard). IDCG over ALL relevant parents (the correct ideal for
    multi-positive). Reduces to the old binary NDCG when every grade is 1."""
    gains = [(2 ** grade_map.get(p, 0) - 1) for p in ranked_parents[:k]]
    ideal_grades = sorted(grade_map.values(), reverse=True)[:k]
    idcg = dcg([2 ** g - 1 for g in ideal_grades])
    return dcg(gains) / idcg if idcg > 0 else 0.0


def first_hit_rank(ranked_parents: list[str], positives: set[str]) -> int | None:
    for i, p in enumerate(ranked_parents):
        if p in positives:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label, e.g. v6-no-enrichment")
    ap.add_argument("--golden", default=str(GOLDEN_PATH), help="path to golden JSONL")
    ap.add_argument("--limit", type=int, default=0, help="eval only first N queries (debug)")
    ap.add_argument("--candidates", type=int, default=50, help="fusion candidate depth fed to reranker (⚡3 sweep)")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder; score fusion top-k directly. "
                                                             "Use to isolate retrieval-level effects (e.g. late chunking) "
                                                             "and to eval models whose env can't load the Ettin reranker (Jina v3).")
    ap.add_argument("--hyde", default="", help="path to hyde_<tag>.jsonl {query, hyde_passage}; "
                                               "routes the dense leg through the hypothetical passage "
                                               "(BM25 stays on the original query). Measures the "
                                               "agent-augmented production path, not the bare-query floor.")
    ap.add_argument("--by-archetype", action="store_true", help="also report metrics grouped by the "
                                                               "row's `archetype` field (keyword / error_symbol / "
                                                               "navigational / baseline), so a hybrid-leg regression "
                                                               "in one stratum can't hide in the aggregate.")
    ap.add_argument("--per-query", default="", help="also write one JSONL row per query: per-query "
                                                    "scores (for paired significance testing) + the "
                                                    "retrieved top-k candidate texts (for label-quality "
                                                    "auditing). Path, e.g. data/golden/eval_runs/pq-<label>.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.golden)]
    if args.limit:
        rows = rows[: args.limit]
    c2p = child_to_parent_map()

    hyde_map = {}
    if args.hyde:
        hyde_map = {json.loads(l)["query"]: json.loads(l)["hyde_passage"] for l in open(args.hyde)}
        missing = sum(1 for r in rows if r["query"] not in hyde_map)
        print(f"HyDE mode: {len(hyde_map)} passages loaded · {missing} queries without a passage (fall back to bare query)")

    client = get_client()
    def _empty():
        return {"ndcg": [], "recall10": [], "mrr": [], "recall_ceiling": []}
    per_domain = defaultdict(_empty)
    per_archetype = defaultdict(_empty)
    misses = []
    per_query_rows = []

    for n, r in enumerate(rows, 1):
        # grade_map: parent_id -> relevance grade (>=1). v2 sets carry relevance_grades (graded,
        # parent-keyed); v1 single-positive sets fall back to binary (grade 1 per positive parent).
        raw_grades = r.get("relevance_grades") or {}
        if raw_grades:
            grade_map = {p: int(g) for p, g in raw_grades.items() if int(g) >= 1}
        else:
            grade_map = {c2p[cid]: 1 for cid in r["positive_child_ids"] if cid in c2p}
        positives = set(grade_map)
        if not positives:
            continue
        dom = r["domain"]
        arch = r.get("archetype", "baseline")
        hyde = hyde_map.get(r["query"]) or None

        # full pipeline (what the agent gets); --no-rerank scores fusion top-10 directly
        results = hybrid_search(r["query"], client, top_k=10, candidates=args.candidates,
                                rerank=not args.no_rerank, hyde_passage=hyde)
        ranked = [res.parent_id for res in results]
        ndcg = ndcg_at_k(ranked, grade_map, 10)
        rank = first_hit_rank(ranked, positives)
        recall10 = 1.0 if rank else 0.0
        mrr = 1.0 / rank if rank else 0.0

        # retrieval ceiling (fusion only, no rerank/cutoff)
        ceil = hybrid_search(r["query"], client, top_k=CEILING_K, rerank=False, hyde_passage=hyde)
        ceil_hit = 1.0 if positives & {x.parent_id for x in ceil} else 0.0

        for grp, key in ((per_domain, dom), (per_archetype, arch)):
            grp[key]["ndcg"].append(ndcg)
            grp[key]["recall10"].append(recall10)
            grp[key]["mrr"].append(mrr)
            grp[key]["recall_ceiling"].append(ceil_hit)

        if args.per_query:
            # candidate dump from the full-pipeline results (what the agent would see):
            # enough text per retrieved parent for an LLM judge to rate relevance to the
            # query WITHOUT the single gold label, so we can estimate label incompleteness.
            cands = [{
                "rank": i + 1,
                "parent_id": res.parent_id,
                "is_labeled_positive": res.parent_id in positives,
                "heading_path": res.heading_path,
                "book": res.book,
                "snippet": (res.child_match or res.content or "")[:600],
            } for i, res in enumerate(results)]
            per_query_rows.append({
                "query": r["query"],
                "domain": dom,
                "book": r["book"],
                "heading_path": r.get("heading_path", ""),
                "positive_parent_ids": sorted(positives),
                "ndcg": round(ndcg, 6),
                "recall10": recall10,
                "mrr": round(mrr, 6),
                "rank": rank,
                "ceiling_hit": ceil_hit,
                "candidates": cands,
            })

        if not recall10:
            misses.append({"query": r["query"], "book": r["book"], "domain": dom,
                           "in_ceiling": bool(ceil_hit), "heading_path": r["heading_path"]})
        if n % 25 == 0:
            print(f"  {n}/{len(rows)}...")

    client.close()

    def agg(key):
        allv = [v for d in per_domain.values() for v in d[key]]
        return sum(allv) / len(allv) if allv else 0.0

    report = {"label": args.label, "n_queries": sum(len(d["ndcg"]) for d in per_domain.values()),
              "aggregate": {k: round(agg(k), 4) for k in ["ndcg", "recall10", "mrr", "recall_ceiling"]},
              "per_domain": {}}
    for dom, d in sorted(per_domain.items()):
        report["per_domain"][dom] = {
            "n": len(d["ndcg"]),
            **{k: round(sum(d[k]) / len(d[k]), 4) for k in ["ndcg", "recall10", "mrr", "recall_ceiling"]},
        }
    if len(per_archetype) > 1:
        report["per_archetype"] = {
            arch: {"n": len(d["ndcg"]),
                   **{k: round(sum(d[k]) / len(d[k]), 4) for k in ["ndcg", "recall10", "mrr", "recall_ceiling"]}}
            for arch, d in sorted(per_archetype.items())
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({"report": report, "misses": misses}, indent=2))

    if args.per_query:
        pq_path = Path(args.per_query)
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pq_path, "w") as fh:
            for row in per_query_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"  per-query dump ({len(per_query_rows)} rows) -> {pq_path}")

    print(f"\n===== {args.label}  (n={report['n_queries']}) =====")
    a = report["aggregate"]
    print(f"  NDCG@10={a['ndcg']:.4f}  Recall@10={a['recall10']:.4f}  "
          f"MRR={a['mrr']:.4f}  Recall@{CEILING_K}(ceiling)={a['recall_ceiling']:.4f}")
    print("  per-domain:")
    for dom, d in report["per_domain"].items():
        print(f"    {dom:12s} n={d['n']:3d}  NDCG@10={d['ndcg']:.3f}  R@10={d['recall10']:.3f}  "
              f"MRR={d['mrr']:.3f}  R@{CEILING_K}={d['recall_ceiling']:.3f}")
    if (args.by_archetype or len(per_archetype) > 1) and "per_archetype" in report:
        print("  per-archetype:")
        for arch, d in report["per_archetype"].items():
            print(f"    {arch:14s} n={d['n']:3d}  NDCG@10={d['ndcg']:.3f}  R@10={d['recall10']:.3f}  "
                  f"MRR={d['mrr']:.3f}  R@{CEILING_K}={d['recall_ceiling']:.3f}")
    gap = a["recall_ceiling"] - a["recall10"]
    print(f"\n  retrieval ceiling - delivered recall = {gap:.4f} "
          f"({'reranker/cutoff dropping hits' if gap > 0.05 else 'reranker keeping retrieved hits'})")
    print(f"  full report + {len(misses)} misses -> {out}")


if __name__ == "__main__":
    main()
