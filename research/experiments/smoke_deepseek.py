"""Smoke test: deepseek-v4-flash with thinking DISABLED, via OpenAI-compatible API.

Validates (before we build the harness/enrichment around it):
  1. The model ID + base_url work with our key.
  2. thinking:disabled returns clean text with NO reasoning_content.
  3. A real enrichment-style call produces a usable 1-2 sentence context.
  4. Reports token usage so we can sanity-check cost projections.
"""

import os
import time

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
assert API_KEY, "set DEEPSEEK_API_KEY (see .env)"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}


def call(messages, label):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        extra_body=THINKING_OFF,
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    u = resp.usage
    print(f"\n=== {label} ({dt:.2f}s) ===")
    print(f"content: {msg.content!r}")
    print(f"reasoning_content present: {bool(reasoning)} (want False)")
    print(f"usage: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    # cached-prefix accounting (DeepSeek reports prompt_cache_hit/miss)
    details = getattr(u, "prompt_tokens_details", None)
    if details:
        print(f"cache details: {details}")
    return msg.content


# 1. Trivial sanity call
call([{"role": "user", "content": "Reply with exactly: OK"}], "sanity")

# 2. Real enrichment-style call (the actual downstream use case)
enrich_prompt = (
    "You write a 1-2 sentence context that places a text chunk within its book "
    "and section, resolving ambiguous references, to make the chunk more retrievable. "
    "Do not restate the chunk. Output only the context sentences.\n\n"
    "Book: PostgreSQL 14 Internals\n"
    "Section: MVCC > Snapshot Isolation\n"
    "Parent context: This chapter explains multiversion concurrency control. Each "
    "transaction sees a snapshot of the database as of its start.\n"
    "Chunk: The xmin and xmax system columns mark the visibility boundaries of each "
    "row version; a tuple is visible if xmin is committed and xmax is not."
)
call([{"role": "user", "content": enrich_prompt}], "enrichment-style")

print("\nSmoke test complete.")
