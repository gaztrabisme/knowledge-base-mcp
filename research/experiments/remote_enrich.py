#!/usr/bin/env python3
"""Standalone enrichment script for running on a remote machine with llama-server."""

import json
import sys
from pathlib import Path

from openai import OpenAI

CONTEXT_PROMPT = """<document>
{parent_text}
</document>

The passage above is from the book "{book_title}"{location}.

Here is the chunk we want to situate within it:
<chunk>
{child_text}
</chunk>

Write a short (1-2 sentence) context that makes this chunk findable by search: name the book and the specific topic or section it covers, and resolve any ambiguous references (e.g. "this method", "the algorithm above", pronouns) to the concrete entities they name. Do not repeat or summarize the chunk's content. Output only the context sentences."""

LLM_BASE_URL = "http://localhost:8081/v1"
LLM_MODEL = "local-model"


def _location_clause(metadata: dict) -> str:
    """Build a section descriptor from heading_path, skipping empty or bare-number placeholders."""
    heading = (metadata.get("heading_path") or "").strip()
    if heading and any(c.isalpha() for c in heading):
        return f', in the section "{heading}"'
    return ""


def main():
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    chunks_dir = work_dir / "chunks"

    if len(sys.argv) > 2:
        global LLM_BASE_URL
        LLM_BASE_URL = sys.argv[2]

    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

    # Verify server is reachable
    try:
        client.models.list()
        print(f"Connected to LLM at {LLM_BASE_URL}")
    except Exception as e:
        print(f"Cannot connect to LLM at {LLM_BASE_URL}: {e}")
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*.json"))
    total_enriched = 0

    for chunk_file in chunk_files:
        slug = chunk_file.stem
        print(f"Enriching: {slug}")

        with open(chunk_file) as f:
            data = json.load(f)

        parent_map = {p["parent_id"]: p["text"] for p in data["parents"]}
        enriched = 0
        skipped = 0

        for child in data["children"]:
            if child.get("context_text"):
                skipped += 1
                continue

            parent_text = parent_map.get(child["parent_id"], "")
            meta = child.get("metadata", {})
            prompt = CONTEXT_PROMPT.format(
                parent_text=parent_text,
                child_text=child["text"],
                book_title=meta.get("book_title") or meta.get("book", ""),
                location=_location_clause(meta),
            )

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    temperature=0.7,
                    top_p=0.8,
                )
                content = response.choices[0].message.content
                child["context_text"] = content.strip() if content else ""
            except Exception as e:
                book = meta.get("book_title", "")
                section = meta.get("heading_path", "")
                child["context_text"] = f"From {book}, {section}."
                print(f"  LLM error: {e}")

            enriched += 1
            if enriched % 50 == 0:
                print(f"  {enriched} enriched...")
                with open(chunk_file, "w") as f:
                    json.dump(data, f, indent=2)

        with open(chunk_file, "w") as f:
            json.dump(data, f, indent=2)

        total_enriched += enriched
        print(f"  → {enriched} enriched, {skipped} skipped")

    print(f"\nDone. {total_enriched} total chunks enriched.")


if __name__ == "__main__":
    main()
