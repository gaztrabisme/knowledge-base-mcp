"""Generate HyDE (hypothetical-answer) passages for a golden set — query-only.

Mirrors what Claude does in production: it sees ONLY the user's query and writes a
hypothetical answer passage, which the MCP `search` tool embeds as the dense query
(`hyde_passage`). We approximate Claude-the-caller with DeepSeek here so the eval
harness can measure the HyDE-augmented path, not just the bare-query floor.

INTEGRITY: the passage is generated from `query` ALONE. The golden set's `source_text`
(the gold chunk) is never shown to the model — leaking it would guarantee retrieval
and invalidate the measurement.

Output: data/golden/hyde_<tag>.jsonl, one {query, hyde_passage} per line. Resumable
(skips queries already present) and concurrent.

Usage:
  python3 scripts/gen_hyde.py --golden data/golden/golden_queries_hard.jsonl --tag hard
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}

PROMPT = (
    "You are answering a developer's technical question with a short passage as it "
    "would appear in an authoritative technical book. Write 3-5 sentences of confident, "
    "specific, factual prose that directly answers the question, using the correct "
    "domain terminology and concept names. Do NOT hedge, do NOT say you are unsure, do "
    "NOT ask questions, do NOT mention that this is hypothetical. Output only the passage.\n\n"
    "Question: {query}"
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--tag", required=True, help="output suffix, e.g. hard")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = OpenAI(api_key=_load_key(), base_url="https://api.deepseek.com")
    out_path = Path(args.golden).parent / f"hyde_{args.tag}.jsonl"

    queries = []
    seen = set()
    for line in open(args.golden):
        q = json.loads(line)["query"]
        if q not in seen:
            seen.add(q)
            queries.append(q)

    done = {}
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done[r["query"]] = r["hyde_passage"]
    todo = [q for q in queries if q not in done]
    print(f"{len(queries)} unique queries · {len(done)} cached · {len(todo)} to generate")

    def gen(q: str) -> tuple[str, str]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(query=q)}],
            temperature=0.0,
            extra_body=THINKING_OFF,
        )
        return q, (resp.choices[0].message.content or "").strip()

    n = 0
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(gen, q) for q in todo]
        for fut in as_completed(futures):
            try:
                q, passage = fut.result()
            except Exception as e:
                print(f"  ERR: {e}")
                continue
            fout.write(json.dumps({"query": q, "hyde_passage": passage}) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}...")

    print(f"done · {n} generated · -> {out_path}")


if __name__ == "__main__":
    main()
