"""Phase C (gate v2) — author the missing query archetypes.

The v6.4 audit found the gate is 100% long why/how questions: 0/200 short-keyword, 1/200
error-string, 0/200 navigational. Those are exactly the query classes that exercise the BM25/sparse
leg and reranker-on-terse-input that real Claude Code sessions emit. This authors ~80 each:

  keyword       : terse 2-6 token jargon lookups ("rocket project layout", "argon2 iterations")
  error_symbol  : exact API-symbol / error-string lookups, seeded from chunk_type=="code" chunks
  navigational  : "where is X defined" / "which chapter covers Y", from heading_path + book

Each row gets archetype + a seed positive (the source chunk). They then flow through
pool_and_judge.py to gain graded multi-positives like the rest of the set.

Output: data/golden/archetypes_all.jsonl (golden schema + `archetype`).
Usage:  python3 scripts/gen_archetypes.py --per 80 --concurrency 40
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI

from knowledge_base.config import CHUNKS_DIR

MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}
GOLDEN = CHUNKS_DIR.parent / "golden" / "golden_queries.jsonl"
OUT = CHUNKS_DIR.parent / "golden" / "archetypes_all.jsonl"
SEED = 42
MIN_TOK, MAX_TOK = 90, 250
_enc = tiktoken.get_encoding("cl100k_base")

KEYWORD_PROMPT = """Given a passage from a technical book, output the SHORT keyword search phrase a
developer would type to find this content — the way you'd actually search, not a sentence.

Rules: 2 to 6 words. NO question mark, NO full sentence, NO "how/why/what". Use the concrete
technical nouns / API names / concepts a practitioner would type. Lowercase unless a proper noun.
Output ONLY the phrase.

Section: {heading}
Passage:
{chunk}"""

ERROR_PROMPT = """Given a passage of technical/code content, output ONE short search a developer would
type when they hit this in practice — either an exact API/function/type symbol lookup, OR a short
error/symptom string. Keep real identifiers verbatim (keep backticks/case). 2-7 tokens, terse, no
full sentence. Output ONLY the search string.

Section: {heading}
Passage:
{chunk}"""

NAV_PROMPT = """A developer wants to NAVIGATE a knowledge base to the section covering a concept.
Given the section heading and a snippet, write ONE short navigational query like "where is X
defined", "which chapter covers Y", or "docs for Z". Keep it short and about LOCATING the topic,
not explaining it. Output ONLY the query.

Section: {heading}
Snippet:
{chunk}"""

PROMPTS = {"keyword": KEYWORD_PROMPT, "error_symbol": ERROR_PROMPT, "navigational": NAV_PROMPT}


def book_domain_map():
    m = {}
    for ln in open(GOLDEN):
        r = json.loads(ln)
        m[r["book"]] = r["domain"]
    return m


def load_children():
    """All eligible children with text in the token band, split into prose-pool and code-pool."""
    prose, code = [], []
    for f in glob.glob(str(CHUNKS_DIR / "*.json")):
        for c in json.loads(open(f).read())["children"]:
            txt = c.get("text", "")
            n = len(_enc.encode(txt))
            if not (MIN_TOK <= n <= MAX_TOK):
                continue
            meta = c.get("metadata", {})
            if not meta.get("heading_path"):
                continue
            (code if meta.get("chunk_type") == "code" else prose).append(c)
    return prose, code


async def gen_one(ai, sem, archetype, chunk, dom):
    meta = chunk["metadata"]
    prompt = PROMPTS[archetype].format(heading=meta.get("heading_path", ""), chunk=chunk["text"][:1500])
    async with sem:
        try:
            r = await ai.chat.completions.create(model=MODEL, temperature=0.4, extra_body=THINKING_OFF,
                                                 messages=[{"role": "user", "content": prompt}])
            q = re.sub(r"\s+", " ", (r.choices[0].message.content or "").strip().strip('"')).strip()
        except Exception as e:
            print(f"  err {chunk['child_id']}: {e}")
            return None
    if len(q) < 3 or len(q.split()) > 12:
        return None
    return {
        "query": q,
        "positive_child_ids": [chunk["child_id"]],
        "hard_negative_child_ids": [],
        "domain": dom,
        "book": meta.get("book", ""),
        "heading_path": meta.get("heading_path", ""),
        "source_text": chunk["text"],
        "archetype": archetype,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=80, help="queries per archetype")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()

    rng = random.Random(SEED)
    b2d = book_domain_map()
    prose, code = load_children()
    rng.shuffle(prose)
    rng.shuffle(code)
    print(f"Pools: {len(prose)} prose, {len(code)} code chunks eligible.")

    # oversample seeds (some generations get dropped by the length filter)
    plan = {
        "keyword": prose[: args.per * 2],
        "navigational": prose[args.per * 2: args.per * 4],
        "error_symbol": (code or prose)[: args.per * 2],
    }

    async def run():
        ai = AsyncOpenAI(api_key=_load_key(), base_url="https://api.deepseek.com",
                         timeout=90.0, max_retries=4)
        sem = asyncio.Semaphore(args.concurrency)
        tasks = []
        for arch, seeds in plan.items():
            for c in seeds:
                dom = b2d.get(c["metadata"].get("book", ""), "ml")
                tasks.append(gen_one(ai, sem, arch, c, dom))
        return await asyncio.gather(*tasks)

    results = [r for r in asyncio.run(run()) if r]
    # cap to --per each, keep order
    capped, counts = [], {k: 0 for k in PROMPTS}
    seen = set()
    for r in results:
        a = r["archetype"]
        if counts[a] >= args.per or r["query"].lower() in seen:
            continue
        counts[a] += 1
        seen.add(r["query"].lower())
        capped.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in capped:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(capped)} archetype queries -> {OUT}")
    print(f"  counts: {counts}")
    for a in PROMPTS:
        ex = [r["query"] for r in capped if r["archetype"] == a][:4]
        print(f"  {a}: {ex}")


def _load_key() -> str:
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


if __name__ == "__main__":
    main()
