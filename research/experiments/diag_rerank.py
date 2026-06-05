"""Diagnose WHERE Ettin ranks the reranked-out positives.

For each hard miss that was retrieved into the top-50 fusion set but dropped before
top-10 (the "reranker lever" half), report:
  - fusion_rank   : best rank of any positive-parent child in the fused candidate list
  - rerank_rank   : best rank of any positive-parent child after Ettin reranks
  - pos_score     : Ettin score of that best positive child
  - cut10_score   : Ettin score of the 10th-ranked candidate (the bar to clear)
  - which doc text variants Ettin saw (child vs heading+child)

Gating question: if positives land at rerank_rank 11-15 with pos_score just under
cut10_score, tuning the rerank stage (granularity, heading-aware, score blend) can
reach them. If they land at rank ~40, the cross-encoder fundamentally disagrees and
tuning won't — that's a model/embedding problem, not a tuning one.

Usage:  python3 scripts/diag_rerank.py
"""

import glob
import json

from qdrant_client import models

from knowledge_base.config import CHUNKS_DIR, COLLECTION_NAME
from knowledge_base.index import embed_texts, text_to_sparse_vector
from knowledge_base.search import get_client, get_ranker

GOLDEN = CHUNKS_DIR.parent / "golden" / "golden_queries_hard.jsonl"
BASELINE = CHUNKS_DIR.parent / "golden" / "eval_runs" / "diag-v6.1-hard.json"
CAND = 50


def child_to_parent():
    m = {}
    for f in glob.glob(str(CHUNKS_DIR / "*.json")):
        for c in json.loads(open(f).read())["children"]:
            m[c["child_id"]] = c["parent_id"]
    return m


def fuse(query, client):
    dv = embed_texts([query], task="retrieval.query")[0]
    sp = text_to_sparse_vector(query)
    pts = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dv, using="", limit=CAND),
            models.Prefetch(query=models.SparseVector(indices=sp.indices, values=sp.values),
                            using="bm25", limit=CAND),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=CAND, with_payload=True,
    ).points
    return [{"id": str(p.id), "payload": p.payload} for p in pts]


def doc_text(p, heading=True):
    parts = [p.get("heading_path", ""), p.get("context_text", ""), p.get("child_text", "")] \
        if heading else [p.get("context_text", ""), p.get("child_text", "")]
    return "\n".join(x for x in parts if x)


def best_pos_rank(order, cand, positives, c2p):
    """order: list of corpus indices into cand (best first). Return (rank, idx) of first
    candidate whose child's parent is a positive, else (None, None)."""
    for rank, ci in enumerate(order, 1):
        pid = c2p.get(cand[ci]["payload"].get("child_id", ""))
        if pid in positives:
            return rank, ci
    return None, None


def main():
    rows = {json.loads(l)["query"]: json.loads(l) for l in open(GOLDEN)}
    base = json.load(open(BASELINE))
    reranked_out = [m["query"] for m in base["misses"] if m["in_ceiling"]]
    c2p = child_to_parent()
    client = get_client()
    ranker = get_ranker()

    print(f"{len(reranked_out)} reranked-out queries\n")
    print(f"{'dom':11s} {'fus':>4s} {'rr(H)':>6s} {'rr(C)':>6s} {'posS':>6s} {'cut10':>6s}  book / query")
    print("-" * 110)

    summary = []
    for q in reranked_out:
        r = rows[q]
        positives = {c2p[cid] for cid in r["positive_child_ids"] if cid in c2p}
        cand = fuse(q, client)
        fus_rank, _ = best_pos_rank(list(range(len(cand))), cand, positives, c2p)

        out = {}
        for tag, heading in [("H", True), ("C", False)]:
            docs = [doc_text(c["payload"], heading=heading) for c in cand]
            ranking = ranker.rank(q, docs)  # list of {corpus_id, score} best-first
            order = [int(it["corpus_id"]) for it in ranking]
            scores = {int(it["corpus_id"]): float(it["score"]) for it in ranking}
            rr, ci = best_pos_rank(order, cand, positives, c2p)
            cut10 = float(ranking[9]["score"]) if len(ranking) >= 10 else float("nan")
            pos_score = scores.get(ci, float("nan")) if ci is not None else float("nan")
            out[tag] = (rr, pos_score, cut10)

        rrH, posS_H, cut10_H = out["H"]
        rrC, _, _ = out["C"]
        print(f"{r['domain']:11s} {str(fus_rank):>4s} {str(rrH):>6s} {str(rrC):>6s} "
              f"{posS_H:>6.2f} {cut10_H:>6.2f}  {r['book'][:24]} | {q[:46]}")
        summary.append({"q": q, "domain": r["domain"], "book": r["book"],
                        "fusion_rank": fus_rank, "rerank_rank_heading": rrH,
                        "rerank_rank_child": rrC, "pos_score": posS_H, "cut10_score": cut10_H})

    client.close()
    out_path = CHUNKS_DIR.parent / "golden" / "eval_runs" / "diag-rerank-positions.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # quick aggregate read
    near = sum(1 for s in summary if s["rerank_rank_heading"] and s["rerank_rank_heading"] <= 15)
    far = sum(1 for s in summary if s["rerank_rank_heading"] and s["rerank_rank_heading"] > 25)
    helped = sum(1 for s in summary if s["rerank_rank_heading"] and s["rerank_rank_child"]
                 and s["rerank_rank_heading"] < s["rerank_rank_child"])
    hurt = sum(1 for s in summary if s["rerank_rank_heading"] and s["rerank_rank_child"]
               and s["rerank_rank_heading"] > s["rerank_rank_child"])
    print(f"\nrerank_rank(heading): <=15 (tunable) = {near} | >25 (hard) = {far}")
    print(f"heading vs child input: heading better = {helped} | child better = {hurt}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
