"""Phase B prep — sample judged triples for the Claude spot-check of the deepseek judge.

Reads the judge_pool.jsonl audit trail (query -> {child_id: deepseek_grade}), samples a
grade-stratified set, joins each candidate's text from the corpus, and writes a BLIND validation
sample (deepseek's grade kept in a separate key the Claude judge is told to ignore).

Output: data/golden/audit/judge_validation_sample.jsonl  +  N batch files for the workflow.
Usage: python3 scripts/judge_validate_prep.py --n 60 --batches 6
"""

import argparse
import glob
import json
import random
from collections import defaultdict

from knowledge_base.config import CHUNKS_DIR

GOLDEN = CHUNKS_DIR.parent / "golden"
POOL = GOLDEN / "judge_pool.jsonl"
OUT = GOLDEN / "audit"
SEED = 42


def child_text_map():
    m = {}
    for f in glob.glob(str(CHUNKS_DIR / "*.json")):
        for c in json.loads(open(f).read())["children"]:
            m[c["child_id"]] = c.get("text", "")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--batches", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(ln) for ln in open(POOL)]
    txt = child_text_map()

    # collect (query, child_id, grade) triples, excluding the gold (always 2, not judged)
    by_grade = defaultdict(list)
    for r in rows:
        gold = r.get("gold")
        for cid, g in r["graded"].items():
            if cid == gold:
                continue
            if cid in txt and txt[cid].strip():
                by_grade[int(g)].append({"query": r["query"], "child_id": cid,
                                         "candidate_text": txt[cid][:700], "deepseek_grade": int(g)})

    rng = random.Random(SEED)
    per = max(1, args.n // 3)
    sample = []
    for g in (0, 1, 2):
        pool = by_grade.get(g, [])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    rng.shuffle(sample)
    for idx, s in enumerate(sample):
        s["idx"] = idx

    # master file keeps deepseek_grade for the join; batch files are BLIND (no grade leaked)
    (OUT / "judge_validation_sample.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in sample))
    batches = [sample[i::args.batches] for i in range(args.batches)]
    for i, b in enumerate(batches):
        blind = [{"idx": r["idx"], "query": r["query"], "candidate_text": r["candidate_text"]} for r in b]
        (OUT / f"jval_batch_{i}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in blind))

    dist = {g: len(by_grade.get(g, [])) for g in (0, 1, 2)}
    print(f"judge_pool triples by grade: {dist}")
    print(f"sampled {len(sample)} ({per}/grade) -> {args.batches} batches in {OUT}")


if __name__ == "__main__":
    main()
