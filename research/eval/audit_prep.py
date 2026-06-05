"""Prep for the gate audit: deterministic coverage/power stats + the LLM-judge sample.

Splits the work cleanly: this script does everything that is exact arithmetic
(no LLM needed), and emits a stratified sample for the workflow's relevance-judging
agents to assess label quality.

Outputs:
  data/golden/audit/coverage_stats.json   coverage, balance, length, easy/hard overlap
  data/golden/audit/sample_batch_{0..N}.jsonl  joined golden+retrieval rows for judges

Run after eval_retrieval.py --per-query has produced pq-baseline-hard.jsonl.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
HARD = GOLDEN / "golden_queries_hard.jsonl"
EASY = GOLDEN / "golden_queries.jsonl"
PQ = GOLDEN / "eval_runs" / "pq-baseline-hard.jsonl"
OUT = GOLDEN / "audit"
N_BATCHES = 6
SAMPLE_PER_DOMAIN = 12
SEED = 42

_STOP = set("the a an of to in for and or is are how do you i what when which with on at as be by "
            "from this that these those it its can does using use your my we our should would could".split())
_WORD = re.compile(r"[a-z0-9]+")


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def content_words(s):
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hard = load_jsonl(HARD)
    easy = load_jsonl(EASY)
    easy_by_pos = {tuple(r["positive_child_ids"]): r for r in easy}

    # ---- coverage / balance / length ----
    dom_counts = Counter(r["domain"] for r in hard)
    book_counts = Counter(r["book"] for r in hard)
    n_books_in_corpus = 46
    q_lens = [len(r["query"]) for r in hard]
    q_words = [len(r["query"].split()) for r in hard]
    multi_positive = sum(1 for r in hard if len(r["positive_child_ids"]) > 1)
    with_hard_neg = sum(1 for r in hard if r.get("hard_negative_child_ids"))

    # ---- easy vs hard lexical overlap (the vocab-gap claim) ----
    overlaps = []
    for r in hard:
        e = easy_by_pos.get(tuple(r["positive_child_ids"]))
        if not e:
            continue
        hw, ew = content_words(r["query"]), content_words(e["query"])
        src = content_words(r.get("source_text", ""))
        # overlap of the QUERY with the SOURCE chunk vocabulary (the leakage measure)
        if src:
            overlaps.append({
                "hard_src_overlap": len(hw & src) / max(1, len(hw)),
                "easy_src_overlap": len(ew & src) / max(1, len(ew)),
            })
    avg = lambda k: round(sum(o[k] for o in overlaps) / len(overlaps), 4) if overlaps else None

    stats = {
        "n_hard": len(hard), "n_easy": len(easy),
        "domains": dict(dom_counts),
        "domain_balance_min_max": [min(dom_counts.values()), max(dom_counts.values())],
        "books_represented": len(book_counts),
        "books_in_corpus": n_books_in_corpus,
        "books_missing_from_gate": n_books_in_corpus - len(book_counts),
        "book_query_counts_min_max_median": [
            min(book_counts.values()), max(book_counts.values()),
            sorted(book_counts.values())[len(book_counts) // 2],
        ],
        "query_char_len": {"min": min(q_lens), "median": sorted(q_lens)[len(q_lens) // 2], "max": max(q_lens)},
        "query_word_len": {"min": min(q_words), "median": sorted(q_words)[len(q_words) // 2], "max": max(q_words)},
        "queries_with_multiple_positives": multi_positive,
        "queries_with_hard_negatives": with_hard_neg,
        "single_positive_pct": round(100 * (len(hard) - multi_positive) / len(hard), 1),
        "avg_query_source_vocab_overlap": {"hard": avg("hard_src_overlap"), "easy": avg("easy_src_overlap")},
    }
    (OUT / "coverage_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # ---- build stratified judge sample joined with retrieval candidates ----
    if not PQ.exists():
        print(f"\n[warn] {PQ} missing — run eval_retrieval --per-query first; sample not built.")
        return
    pq_by_q = {r["query"]: r for r in load_jsonl(PQ)}
    by_dom = {}
    for r in hard:
        by_dom.setdefault(r["domain"], []).append(r)
    rng = random.Random(SEED)
    sample = []
    for dom, rows in sorted(by_dom.items()):
        rng.shuffle(rows)
        for r in rows[:SAMPLE_PER_DOMAIN]:
            pq = pq_by_q.get(r["query"])
            if not pq:
                continue
            e = easy_by_pos.get(tuple(r["positive_child_ids"]))
            sample.append({
                "query": r["query"],
                "easy_query": e["query"] if e else "",
                "domain": dom,
                "book": r["book"],
                "gold_heading_path": r.get("heading_path", ""),
                "gold_source_text": r.get("source_text", ""),
                "gold_parent_ids": pq["positive_parent_ids"],
                "retrieved_candidates": pq["candidates"],
            })
    rng.shuffle(sample)
    batches = [sample[i::N_BATCHES] for i in range(N_BATCHES)]
    for i, b in enumerate(batches):
        with open(OUT / f"sample_batch_{i}.jsonl", "w") as fh:
            for row in b:
                fh.write(json.dumps(row) + "\n")
    print(f"\nSample: {len(sample)} queries -> {N_BATCHES} batches "
          f"({[len(b) for b in batches]}) in {OUT}")


if __name__ == "__main__":
    main()
