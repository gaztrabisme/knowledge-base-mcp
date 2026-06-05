"""v6.7.1 — regenerate the 12 gate queries dropped by the v6.7 index-strip.

The strip removed 6,276 back-of-book index chunks; 12 hard-gate queries (2 keyword + 10
navigational) whose only gold was an index page went dead and were dropped (440->428). This
re-mints replacements from the CLEAN corpus so seeds are valid by construction (index pages no
longer exist to seed from). Two guards over the original gen_archetypes flow:

  - dedup against ALL prior queries (current 428 + the 12 dropped) -> guaranteed net-new
  - reject nav "index for X" / "Index (part N)" phrasings -> kills the junk pattern at the source

Output: data/golden/regen_archetypes.jsonl (gen_archetypes schema). Flows through pool_and_judge.py
next to gain graded multi-positives like the rest of the gate.

Usage: python3 scripts/regen_dropped_queries.py
"""

import asyncio
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # knowledge_base
sys.path.insert(0, str(ROOT / "scripts"))     # gen_archetypes (scripts/ is not a package)

from openai import AsyncOpenAI

from knowledge_base.config import CHUNKS_DIR
from gen_archetypes import book_domain_map, load_children, gen_one, _load_key

SEED = 1234  # fresh shuffle -> seed chunks disjoint from the SEED=42 originals
NEED = {"keyword": 2, "navigational": 10}
OVERSAMPLE = 12  # seeds per archetype before filters; plenty of headroom for 2/10
GOLDEN_DIR = CHUNKS_DIR.parent / "golden"
OUT = GOLDEN_DIR / "regen_archetypes.jsonl"

# nav-junk: a query that asks for the back-of-book index itself (the v6.7 artifact)
JUNK_NAV = re.compile(r"\bindex\b.*\b(for|of|section|part)\b|\bindex\s*\(part", re.I)


def existing_queries() -> set[str]:
    seen = set()
    for name in ("golden_queries_hard.jsonl", "golden_hard_v2_full.jsonl"):
        p = GOLDEN_DIR / name
        if p.exists():
            for ln in open(p):
                seen.add(json.loads(ln)["query"].strip().lower())
    return seen


def main():
    rng = random.Random(SEED)
    b2d = book_domain_map()
    prose, code = load_children()
    rng.shuffle(prose)
    rng.shuffle(code)
    exclude = existing_queries()
    print(f"Pools: {len(prose)} prose, {len(code)} code. Excluding {len(exclude)} prior queries.")

    # generous seed slices; keyword+nav both draw from prose (matches gen_archetypes)
    plan = {
        "keyword": prose[: NEED["keyword"] * OVERSAMPLE],
        "navigational": prose[NEED["keyword"] * OVERSAMPLE: (NEED["keyword"] + NEED["navigational"]) * OVERSAMPLE],
    }

    async def run():
        ai = AsyncOpenAI(api_key=_load_key(), base_url="https://api.deepseek.com",
                         timeout=90.0, max_retries=4)
        sem = asyncio.Semaphore(40)
        tasks = []
        for arch, seeds in plan.items():
            for c in seeds:
                dom = b2d.get(c["metadata"].get("book", ""), "ml")
                tasks.append(gen_one(ai, sem, arch, c, dom))
        return await asyncio.gather(*tasks)

    results = [r for r in asyncio.run(run()) if r]

    kept, counts, batch_seen = [], {k: 0 for k in NEED}, set()
    for r in results:
        a = r["archetype"]
        ql = r["query"].strip().lower()
        if counts[a] >= NEED[a] or ql in exclude or ql in batch_seen:
            continue
        if a == "navigational" and JUNK_NAV.search(r["query"]):
            print(f"  reject junk-nav: {r['query']!r}")
            continue
        counts[a] += 1
        batch_seen.add(ql)
        kept.append(r)

    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(kept)} queries -> {OUT}  counts={counts}")
    for r in kept:
        print(f"  [{r['archetype']}|{r['domain']}|{r['book']}] {r['query']}")
    if any(counts[a] < NEED[a] for a in NEED):
        print("\nWARNING: under target — raise OVERSAMPLE or SEED and re-run.")


if __name__ == "__main__":
    main()
