"""Phase 1A.2 — harden the golden set into vocabulary-gap queries.

The chunk-vocabulary baseline is near-ceiling because queries generated FROM a
chunk inherit its rare terms, so BM25/dense match trivially. That's the EASY
regime and can't measure enrichment (whose job is the vocab-gap regime). This
pass rewrites each query the way a developer would ask WITHOUT having read the
book — problem/symptom framing, synonyms, deliberately avoiding the chunk's
signature terminology — while keeping it uniquely answerable by the same chunk.

Also drops back-matter junk that slipped the matter filter (Packt "become an
author" etc.). Writes:
  data/golden/golden_queries_hard.jsonl   (paraphrased, vocab-gap)
and prints a lexical-overlap comparison (easy vs hard) to prove the gap is real.

Usage:
  export DEEPSEEK_API_KEY=...
  python3 scripts/harden_golden_set.py --concurrency 25
"""

import argparse
import asyncio
import json
import os
import re

from openai import AsyncOpenAI

from knowledge_base.config import CHUNKS_DIR

GOLDEN = CHUNKS_DIR.parent / "golden" / "golden_queries.jsonl"
HARD = CHUNKS_DIR.parent / "golden" / "golden_queries_hard.jsonl"
MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}

# Back-matter junk that the heading matter-filter missed (the query itself betrays it).
JUNK_RE = re.compile(r"\b(packt|become an author|subscribe|leave a review|this book|the author|publishing)\b", re.I)

_STOP = set("the a an of to in for and or is are how do you i what when which with on at as be by from "
            "this that these those it its can does using use your my we our".split())

PROMPT = """You are creating HARD test queries for a retrieval benchmark.

Given a developer's question and the EXACT source passage it came from, rewrite the question the way a real developer would ask it WITHOUT having read this book — describe their problem, symptom, or goal in everyday practitioner language.

Rules:
- AVOID the distinctive technical terms, API names, and exact phrasing from the source passage. Use synonyms, plain descriptions, or symptom/problem framing. (E.g. instead of "GIN posting list compression" → "why does my Postgres full-text index keep getting so big".)
- BUT it must still be UNIQUELY answered by THIS passage. Keep the specific intent; change only the vocabulary and framing. Do not make it so generic that unrelated topics would answer it.
- One or two sentences, phrased as a natural question or search query. Output ONLY the rewritten question.

Source passage:
{chunk}

Original question:
{query}"""


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def overlap(query: str, chunk: str) -> float:
    """Fraction of query content-words that also appear in the chunk."""
    q = content_words(query)
    return len(q & content_words(chunk)) / len(q) if q else 0.0


async def harden(client, sem, row):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.4, extra_body=THINKING_OFF,
                messages=[{"role": "user", "content": PROMPT.format(
                    chunk=row["source_text"][:2000], query=row["query"])}],
            )
        except Exception as e:
            print(f"  ERROR {row['book']}: {e}")
            return None
    q = (r.choices[0].message.content or "").strip().strip('"')
    if not q or len(q) < 10:
        return None
    out = dict(row)
    out["query"] = q
    out["original_query"] = row["query"]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(GOLDEN)]
    clean = [r for r in rows if not JUNK_RE.search(r["query"])]
    dropped = len(rows) - len(clean)
    print(f"Loaded {len(rows)}; dropped {dropped} junk; hardening {len(clean)} on {MODEL} (thinking off)...")

    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                         base_url="https://api.deepseek.com", timeout=60.0, max_retries=5)
    sem = asyncio.Semaphore(args.concurrency)
    res = await asyncio.gather(*(harden(client, sem, r) for r in clean))
    hard = [r for r in res if r]

    with HARD.open("w") as f:
        for r in hard:
            f.write(json.dumps(r) + "\n")

    # prove the gap: lexical overlap easy vs hard, against the same source chunks
    easy_ov = sum(overlap(r["original_query"], r["source_text"]) for r in hard) / len(hard)
    hard_ov = sum(overlap(r["query"], r["source_text"]) for r in hard) / len(hard)
    print(f"\nWrote {len(hard)} hard queries -> {HARD}")
    print(f"mean query→chunk content-word overlap:  easy={easy_ov:.3f}  hard={hard_ov:.3f}  "
          f"(↓{100*(easy_ov-hard_ov)/easy_ov:.0f}%)")
    print("\nexamples (original → hardened):")
    for r in hard[:6]:
        print(f"  - {r['original_query'][:70]}")
        print(f"    → {r['query'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
