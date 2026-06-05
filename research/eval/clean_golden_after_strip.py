"""Reconcile golden sets with the index-stripped corpus.

After strip_index_backmatter.py removed back-of-book index chunks, some golden positives/negatives
point at deleted chunks. For each row: drop dead child_ids from positive_child_ids and
hard_negative_child_ids; drop dead parents from relevance_grades; if a row has no surviving
positive, drop the row entirely. Operates on hard + easy sets in place (caller backs up via git).
"""
import json, glob, sys
from pathlib import Path

live_c, live_p = set(), set()
for f in glob.glob("data/chunks/*.json"):
    d = json.load(open(f))
    for ch in d["children"]:
        live_c.add(ch["child_id"]); live_p.add(ch["parent_id"])

def clean(path):
    rows = [json.loads(l) for l in open(path)]
    kept, dropped, pruned = [], 0, 0
    for r in rows:
        pos = [c for c in r.get("positive_child_ids", []) if c in live_c]
        if not pos:
            dropped += 1; continue
        if len(pos) != len(r.get("positive_child_ids", [])): pruned += 1
        r["positive_child_ids"] = pos
        if "relevance_grades" in r:
            r["relevance_grades"] = {p: g for p, g in r["relevance_grades"].items() if p in live_p}
        if "hard_negative_child_ids" in r:
            r["hard_negative_child_ids"] = [c for c in r["hard_negative_child_ids"] if c in live_c]
        kept.append(r)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{path}: {len(rows)} -> {len(kept)}  (dropped {dropped} dead, pruned {pruned} partial)")

for p in sys.argv[1:]:
    clean(p)
