"""Phase 3 — enrich all chunks via the DeepSeek API (async, concurrent, resumable).

Replaces the serial local-Qwen pass (remote_enrich.py, ~13-20h) with a concurrent
DeepSeek fan-out (~$15, ~30 min). Writes context_text into each data/chunks/*.json.

Design notes:
  - Resumable: children that already carry context_text are skipped; each file is
    saved after its batch, so an interrupted run resumes cleanly.
  - Prompt-cache friendly: within a file, children are processed in parent order so
    the long shared parent_text prefix hits DeepSeek's context cache (50x cheaper).
  - Thinking OFF (extra_body) — defaults ENABLED on deepseek-v4-flash.
  - Prompt: the proven CONTEXT_PROMPT by default. MIPROv2 (auto=light) did not beat
    it on the mini-harness, so baseline is the incumbent; pass --instruction to swap
    in a gate-winning variant (kept inside the same <document>/<chunk> structure so
    the cache prefix is preserved).

Usage:
  python3 scripts/deepseek_enrich_all.py                       # all books, baseline prompt
  python3 scripts/deepseek_enrich_all.py --book postgresql-14-internals
  python3 scripts/deepseek_enrich_all.py --concurrency 100 --limit 200   # dry-ish test
"""

import argparse
import asyncio
import json
import os
import time

from openai import AsyncOpenAI

from knowledge_base.config import CHUNKS_DIR
from knowledge_base.enrich import CONTEXT_PROMPT, _location_clause

MODEL = "deepseek-v4-flash"
API_BASE = "https://api.deepseek.com"
THINKING_OFF = {"thinking": {"type": "disabled"}}

# alternate template body for a custom (gate-winning) instruction; keeps parent_text
# first so the DeepSeek prompt-cache prefix still applies.
CUSTOM_TEMPLATE = """<document>
{parent_text}
</document>

The passage above is from the book "{book_title}"{location}.

<chunk>
{child_text}
</chunk>

{instruction}
Output only the context sentences."""


def _load_env() -> None:
    env = CHUNKS_DIR.parent.parent / ".env"
    if env.exists() and "DEEPSEEK_API_KEY" not in os.environ:
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def build_prompt(child: dict, parent_text: str, instruction: str | None) -> str:
    meta = child.get("metadata", {})
    book_title = meta.get("book_title") or meta.get("book", "")
    loc = _location_clause(meta)
    if instruction:
        return CUSTOM_TEMPLATE.format(
            parent_text=parent_text[:4000], child_text=child["text"][:2000],
            book_title=book_title, location=loc, instruction=instruction.strip(),
        )
    return CONTEXT_PROMPT.format(
        parent_text=parent_text[:4000], child_text=child["text"][:2000],
        book_title=book_title, location=loc,
    )


async def enrich_child(client, sem, child, prompt) -> str:
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.3, extra_body=THINKING_OFF,
                messages=[{"role": "user", "content": prompt}],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            meta = child.get("metadata", {})
            print(f"    error {child['child_id']}: {e}")
            return f"From {meta.get('book_title', '')}, {meta.get('heading_path', '')}."


async def enrich_file(client, sem, path, instruction, limit) -> tuple[int, int]:
    data = json.loads(path.read_text())
    parent_map = {p["parent_id"]: p["text"] for p in data["parents"]}
    # parent order → shared prefix cache hits
    todo = [c for c in data["children"] if not c.get("context_text")]
    todo.sort(key=lambda c: c["parent_id"])
    if limit:
        todo = todo[:limit]
    skipped = len(data["children"]) - len([c for c in data["children"] if not c.get("context_text")])
    if not todo:
        return 0, skipped

    prompts = [build_prompt(c, parent_map.get(c["parent_id"], ""), instruction) for c in todo]
    results = await asyncio.gather(*(enrich_child(client, sem, c, p) for c, p in zip(todo, prompts)))
    for c, ctx in zip(todo, results):
        c["context_text"] = ctx
    path.write_text(json.dumps(data, indent=2))
    return len(todo), skipped


async def main_async(book, concurrency, instruction, limit):
    _load_env()
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=API_BASE,
                         timeout=60.0, max_retries=5)
    sem = asyncio.Semaphore(concurrency)
    files = [CHUNKS_DIR / f"{book}.json"] if book else sorted(CHUNKS_DIR.glob("*.json"))

    total_enr, total_skip, t0 = 0, 0, time.monotonic()
    for i, f in enumerate(files, 1):
        if not f.exists():
            print(f"  missing: {f.name}");  continue
        enr, skip = await enrich_file(client, sem, f, instruction, limit)
        total_enr += enr; total_skip += skip
        print(f"  [{i}/{len(files)}] {f.stem}: +{enr} enriched, {skip} already done")
        if limit and total_enr >= limit:
            break
    dt = time.monotonic() - t0
    print(f"\nDone. {total_enr} enriched, {total_skip} skipped in {dt/60:.1f} min "
          f"({total_enr/dt:.1f}/s)" if dt else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default=None, help="single book slug (default: all)")
    ap.add_argument("--concurrency", type=int, default=80)
    ap.add_argument("--instruction", default=None,
                    help="custom enrichment instruction (gate winner); default = CONTEXT_PROMPT")
    ap.add_argument("--limit", type=int, default=0, help="cap enrichments (smoke test)")
    args = ap.parse_args()
    asyncio.run(main_async(args.book, args.concurrency, args.instruction, args.limit))


if __name__ == "__main__":
    main()
