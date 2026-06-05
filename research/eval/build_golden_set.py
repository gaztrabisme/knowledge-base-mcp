"""Phase 1A — build the golden query set for the retrieval eval harness.

Reframe (see wiki/dspy-enrichment-plan.md): the metric is EXTRINSIC (retrieval
quality), so the golden set is `query -> relevant-chunk(s)` pairs, not
`chunk -> gold-summary`. We sample substantive child chunks stratified across
all 46 books, then ask deepseek-v4-flash (thinking OFF) to write a SPECIFIC
developer question each chunk uniquely answers. The source chunk is the
automatic positive label.

Output JSONL: {query, positive_child_ids, domain, book, heading_path, source_text}
`source_text` is carried for the human spot-check pass (discard vague queries,
add hard negatives), then can be dropped before eval.

Usage:
  export DEEPSEEK_API_KEY=...   # from .env
  python3 scripts/build_golden_set.py --target 200 --concurrency 50
"""

import argparse
import asyncio
import json
import os
import random
import re

import tiktoken
from openai import AsyncOpenAI

from knowledge_base.config import BOOKS, CHUNKS_DIR, RAG_GUIDE_DIR, SOURCE_DIRS
from knowledge_base.ingest import _is_back_matter_heading, _is_front_matter_heading

SEED = 42
MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}
GOLDEN_PATH = CHUNKS_DIR.parent / "golden" / "golden_queries.jsonl"

# Chunk eligibility: enough unique content to anchor a specific question, but
# not a giant catch-all. (token dist: p50=223, p90=247.)
MIN_TOK, MAX_TOK = 90, 250

_enc = tiktoken.get_encoding("cl100k_base")

GEN_PROMPT = """You are building a retrieval benchmark for a developer knowledge base.

Given ONE chunk of text from a technical book, write ONE realistic question a developer would type into a search box that THIS chunk specifically and completely answers.

Rules:
- The question must target the chunk's UNIQUE content (a specific technique, API, parameter, algorithm, definition, or trade-off). It should NOT be a generic question that dozens of other sections could also answer.
- Do NOT mention the book title, chapter, author, or "this section/chapter/text".
- Do NOT use phrases like "according to the passage". Phrase it as a standalone search query or question.
- Prefer the concrete vocabulary a practitioner would use.
- One sentence. Output ONLY the question, nothing else.

Book: {book_title}
Section: {heading_path}
Chunk:
{chunk_text}"""


def domain_map() -> dict[str, str]:
    label = {
        "applied-ml": "ml", "databases": "databases", "security": "security",
        "distributed-systems": "distributed", "rust": "rust",
    }
    out: dict[str, str] = {}
    for dom, d in SOURCE_DIRS.items():
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in {".epub", ".pdf", ".html"}:
                out[f.stem] = label[dom]
    out["production-rag-guide"] = "ml"  # lives in RAG_GUIDE_DIR
    missing = [b for b in BOOKS if b not in out]
    if missing:
        print(f"  WARN: no domain derived for {missing}; check {[*SOURCE_DIRS]} / {RAG_GUIDE_DIR.name}")
    return out


def _is_matter_heading(heading_path: str) -> bool:
    """True if any segment of the heading path is front/back matter (TOC, index,
    'why subscribe', preface, ...). Those chunks make junk or weak-positive queries."""
    for seg in heading_path.split(">"):
        seg = seg.strip()
        if not seg:
            continue
        line = "# " + seg
        if _is_front_matter_heading(line) or _is_back_matter_heading(line):
            return True
        low = seg.lower()
        if low.startswith("index (part") or "table of contents" in low:
            return True
    return False


def eligible_children(book: str) -> list[dict]:
    path = CHUNKS_DIR / f"{book}.json"
    data = json.loads(path.read_text())
    out = []
    for c in data["children"]:
        text = c["text"].strip()
        # skip heading-only / code-fence-only stubs
        if re.fullmatch(r"#{0,6}\s*[\w .,:-]*", text) and len(text) < 40:
            continue
        if _is_matter_heading(c["metadata"].get("heading_path", "")):
            continue
        n = len(_enc.encode(text))
        if MIN_TOK <= n <= MAX_TOK:
            out.append(c)
    return out


def allocate(target: int, counts: dict[str, int]) -> dict[str, int]:
    """2 per book floor + remainder proportional to substantive-chunk count."""
    books = list(counts)
    base = {b: min(2, counts[b]) for b in books}
    remaining = target - sum(base.values())
    total = sum(counts.values())
    if remaining > 0 and total > 0:
        for b in books:
            bonus = round(remaining * counts[b] / total)
            base[b] = min(counts[b], base[b] + bonus)
    return base


async def gen_query(client, sem, chunk, domain, book_title):
    md = chunk["metadata"]
    prompt = GEN_PROMPT.format(
        book_title=book_title,
        heading_path=md.get("heading_path", "") or "(none)",
        chunk_text=chunk["text"].strip()[:2000],
    )
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.3, extra_body=THINKING_OFF,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"  ERROR {chunk['child_id']}: {e}")
            return None
    q = (r.choices[0].message.content or "").strip().strip('"')
    if not q or len(q) < 10:
        return None
    return {
        "query": q,
        "positive_child_ids": [chunk["child_id"]],
        "hard_negative_child_ids": [],
        "domain": domain,
        "book": md["book"],
        "heading_path": md.get("heading_path", ""),
        "source_text": chunk["text"].strip(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()

    random.seed(SEED)
    doms = domain_map()

    pools = {b: eligible_children(b) for b in BOOKS}
    counts = {b: len(p) for b, p in pools.items()}
    alloc = allocate(args.target, counts)

    sampled = []
    for b, n in alloc.items():
        if n <= 0 or not pools[b]:
            continue
        sampled += [(c, doms.get(b, "unknown"), BOOKS[b]) for c in random.sample(pools[b], min(n, len(pools[b])))]
    random.shuffle(sampled)
    print(f"Sampling {len(sampled)} chunks across {sum(1 for n in alloc.values() if n>0)} books "
          f"(target {args.target}). Generating questions on {MODEL} (thinking off)...")

    key = os.environ["DEEPSEEK_API_KEY"]
    client = AsyncOpenAI(
        api_key=key, base_url="https://api.deepseek.com",
        timeout=60.0, max_retries=5,  # ride out transient API windows
    )
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(gen_query(client, sem, c, d, t) for c, d, t in sampled))
    rows = [r for r in results if r]

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    dist = Counter(r["domain"] for r in rows)
    print(f"\nWrote {len(rows)} queries -> {GOLDEN_PATH}")
    print(f"by domain: {dict(dist)}")
    print(f"dropped (errors/too-short): {len(sampled) - len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
