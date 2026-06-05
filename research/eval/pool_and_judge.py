"""Phase A (gate v2) — TREC-style pooling + graded relevance judging.

The golden set is 1-positive over 48k chunks; the v6.4 audit found 44.4% of queries have an
unlabeled-but-relevant top-10 hit, so NDCG/MRR are understated. This builds MULTIPLE graded
positives per query: pool candidates from several retrieval configs off the live index, then have
deepseek grade each (0/1/2) for relevance to the query. Gold seed is forced to grade 2.

Pool diversity (single embedding, so vary the retrieval mode) — top-K parents from each of:
  dense-only · BM25-only · fusion(no rerank) · fusion+HyDE (if --hyde given)
Union unique parents, keep one representative child per parent, + the gold parent. One batched
deepseek call grades all of a query's candidates at once (≈N_queries calls, not N×pool).

Output (originals untouched): <golden>_v2.jsonl with positive_child_ids=grades≥1,
relevance_grades (parent-keyed), hard_negative_child_ids=grade-0. Plus judge_pool.jsonl audit trail.

Usage:
  python3 scripts/pool_and_judge.py --golden data/golden/golden_queries_hard.jsonl \
      --hyde data/golden/hyde_hard.jsonl --out data/golden/golden_queries_hard_v2.jsonl
"""

import argparse
import asyncio
import glob
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import models

from knowledge_base.config import CHUNKS_DIR, COLLECTION_NAME
from knowledge_base.index import embed_texts, text_to_sparse_vector
from knowledge_base.search import get_client

MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}
POOL_K = 20


def _load_key() -> str:
    """DEEPSEEK_API_KEY from shell env or repo .env (same loader as gen_hyde.py)."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPSEEK_API_KEY not found (shell env or .env)")

JUDGE_PROMPT = """You are grading retrieval relevance for a developer knowledge base. Grade STRICTLY.

QUERY: {query}

IMPORTANT: every candidate below was retrieved for this query, so they are ALL at least
topically related. Topical relatedness is NOT relevance. Grade only on whether the passage
actually ANSWERS the specific question asked.

  2 = directly and specifically answers the query — a developer searching this would be satisfied
      this is the/an answer.
  1 = answers a genuine PART of the specific question, or is a near-duplicate of the real answer
      (same fact stated elsewhere). NOT "same subject area".
  0 = same topic but does NOT answer THIS specific question, OR only mentions it in passing,
      OR irrelevant.
When in doubt between 0 and 1, choose 0. Most candidates should be 0. Expect only a few 1s/2s.

Return ONLY a JSON array, one object per candidate, no prose:
[{{"i": 0, "grade": 2}}, {{"i": 1, "grade": 0}}, ...]

CANDIDATES:
{candidates}"""


def child_to_parent_map() -> dict[str, str]:
    m = {}
    for f in glob.glob(str(CHUNKS_DIR / "*.json")):
        for c in json.loads(open(f).read())["children"]:
            m[c["child_id"]] = c["parent_id"]
    return m


def _leg(client, *, dense=None, sparse=None):
    """Top-K from ONE leg (a dense vector or a BM25 sparse vector). Direct query_points —
    no fusion, no rerank, no redundant re-embed. Plain fusion is intentionally dropped: DBSF of
    dense∪bm25 surfaces no parent the two single legs don't already union in."""
    if dense is not None:
        pf = models.Prefetch(query=dense, using="", limit=POOL_K)
    else:
        pf = models.Prefetch(query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                             using="bm25", limit=POOL_K)
    pts = client.query_points(collection_name=COLLECTION_NAME, prefetch=[pf],
                              query=models.FusionQuery(fusion=models.Fusion.DBSF),
                              limit=POOL_K, with_payload=True).points
    return [(p.payload.get("parent_id", ""), p.payload.get("child_id", ""),
             p.payload.get("child_text", "")) for p in pts]


def build_pool(client, row, c2p, hyde_map):
    """Union of (parent_id, rep_child_id, snippet) across 3 legs (dense / BM25 / HyDE-dense),
    gold parent always included. All legs are direct single-leg queries off the live index."""
    q = row["query"]
    hyde = hyde_map.get(q)
    texts = [q] + ([hyde] if hyde else [])
    vecs = embed_texts(texts, task="retrieval.query")  # one embed call: query [+ hyde]
    sparse = text_to_sparse_vector(q)
    cands = _leg(client, dense=vecs[0]) + _leg(client, sparse=sparse)
    if hyde:
        cands += _leg(client, dense=vecs[1])

    by_parent = {}
    for pid, cid, snip in cands:
        if pid and pid not in by_parent:
            by_parent[pid] = (cid, snip)
    # gold parent forced in
    gold_cid = row["positive_child_ids"][0]
    gold_pid = c2p.get(gold_cid, "")
    if gold_pid:
        by_parent[gold_pid] = (gold_cid, row.get("source_text", "")[:600])
    return gold_pid, by_parent


def parse_grades(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for o in arr:
        if isinstance(o, dict) and "i" in o and "grade" in o:
            out[int(o["i"])] = max(0, min(2, int(o["grade"])))
    return out if out else None


async def judge_row(client_ai, sem, row, gold_pid, by_parent):
    parents = list(by_parent.keys())
    blocks = []
    for i, pid in enumerate(parents):
        snip = re.sub(r"\s+", " ", by_parent[pid][1])[:500]
        blocks.append(f"[{i}] {snip}")
    prompt = JUDGE_PROMPT.format(query=row["query"], candidates="\n\n".join(blocks))
    grades = None
    async with sem:
        for _ in range(2):
            try:
                r = await client_ai.chat.completions.create(
                    model=MODEL, temperature=0.0, extra_body=THINKING_OFF,
                    messages=[{"role": "user", "content": prompt}])
                grades = parse_grades(r.choices[0].message.content or "")
                if grades is not None:
                    break
            except Exception as e:
                print(f"  judge err: {e}")
    if grades is None:
        grades = {}
    # gold is always perfect, never downgraded
    gmap = {}
    for i, pid in enumerate(parents):
        gmap[pid] = 2 if pid == gold_pid else grades.get(i, 0)
    return row, parents, by_parent, gmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--hyde", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.golden)]
    if args.limit:
        rows = rows[: args.limit]
    hyde_map = {}
    if args.hyde and Path(args.hyde).exists():
        hyde_map = {json.loads(ln)["query"]: json.loads(ln)["hyde_passage"] for ln in open(args.hyde)}

    c2p = child_to_parent_map()
    qclient = get_client()
    print(f"Pooling {len(rows)} queries (4 legs, top-{POOL_K})...")
    pools = []
    for n, row in enumerate(rows, 1):
        gold_pid, by_parent = build_pool(qclient, row, c2p, hyde_map)
        pools.append((row, gold_pid, by_parent))
        if n % 25 == 0:
            print(f"  pooled {n}/{len(rows)}  (last pool size {len(by_parent)})")
    qclient.close()

    async def run():
        ai = AsyncOpenAI(api_key=_load_key(),
                         base_url="https://api.deepseek.com", timeout=90.0, max_retries=4)
        sem = asyncio.Semaphore(args.concurrency)
        return await asyncio.gather(*(judge_row(ai, sem, r, g, bp) for r, g, bp in pools))

    print("Judging (deepseek, batched per query)...")
    judged = asyncio.run(run())

    out_rows, audit, gained, total_pos, grade_hist = [], [], 0, 0, {0: 0, 1: 0, 2: 0}
    for row, parents, by_parent, gmap in judged:
        pos_children, neg_children, grade_map = [], [], {}
        for pid in parents:
            cid = by_parent[pid][0]
            g = gmap[pid]
            grade_hist[g] += 1
            if g >= 1:
                pos_children.append(cid)
                grade_map[pid] = g
            else:
                neg_children.append(cid)
        total_pos += len(pos_children)
        if len(pos_children) > 1:
            gained += 1
        nr = dict(row)
        nr["positive_child_ids"] = pos_children
        nr["relevance_grades"] = grade_map
        nr["hard_negative_child_ids"] = neg_children
        nr.setdefault("archetype", "baseline")
        out_rows.append(nr)
        audit.append({"query": row["query"], "gold": row["positive_child_ids"][0],
                      "graded": {by_parent[p][0]: gmap[p] for p in parents}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    audit_path = Path(args.out).parent / "judge_pool.jsonl"
    with open(audit_path, "a") as f:
        for a in audit:
            f.write(json.dumps(a) + "\n")

    n = len(out_rows)
    print(f"\nDone. {n} rows -> {args.out}")
    print(f"  mean positives/query: {total_pos / n:.2f}  ({gained}/{n} = {100*gained/n:.0f}% gained ≥1 new positive)")
    print(f"  grade histogram (all pooled candidates): {grade_hist}")
    print(f"  audit trail appended -> {audit_path}")


if __name__ == "__main__":
    main()
