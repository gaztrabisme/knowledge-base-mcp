"""Contextual enrichment via LM Studio (OpenAI-compatible API)."""

import json

from openai import OpenAI

from .config import CHUNKS_DIR, LM_STUDIO_BASE_URL, LM_STUDIO_MODEL

CONTEXT_PROMPT = """<document>
{parent_text}
</document>

The passage above is from the book "{book_title}"{location}.

Here is the chunk we want to situate within it:
<chunk>
{child_text}
</chunk>

Write a short (1-2 sentence) context that makes this chunk findable by search: name the book and the specific topic or section it covers, and resolve any ambiguous references (e.g. "this method", "the algorithm above", pronouns) to the concrete entities they name. Do not repeat or summarize the chunk's content. Output only the context sentences."""


def _location_clause(metadata: dict) -> str:
    """Build a section descriptor from heading_path, skipping empty or bare-number placeholders."""
    heading = (metadata.get("heading_path") or "").strip()
    if heading and any(c.isalpha() for c in heading):
        return f', in the section "{heading}"'
    return ""


def get_client() -> OpenAI:
    return OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")


def generate_context(client: OpenAI, parent_text: str, child_text: str, metadata: dict) -> str:
    book_title = metadata.get("book_title") or metadata.get("book", "")
    prompt = CONTEXT_PROMPT.format(
        parent_text=parent_text,
        child_text=child_text,
        book_title=book_title,
        location=_location_clause(metadata),
    )

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"    LM Studio error: {e}")
        book = metadata.get("book_title", "")
        section = metadata.get("heading_path", "")
        return f"From {book}, {section}."


def enrich_book(book_slug: str, client: OpenAI | None = None) -> dict:
    chunk_path = CHUNKS_DIR / f"{book_slug}.json"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunk_path}")

    with open(chunk_path) as f:
        data = json.load(f)

    if client is None:
        client = get_client()

    parent_map = {p["parent_id"]: p["text"] for p in data["parents"]}
    enriched = 0
    skipped = 0

    for child in data["children"]:
        if child.get("context_text"):
            skipped += 1
            continue

        parent_text = parent_map.get(child["parent_id"], "")
        context = generate_context(client, parent_text, child["text"], child.get("metadata", {}))
        child["context_text"] = context
        enriched += 1

        if enriched % 50 == 0:
            print(f"    Enriched {enriched} chunks...")
            with open(chunk_path, "w") as f:
                json.dump(data, f, indent=2)

    with open(chunk_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"    → {enriched} enriched, {skipped} already had context")
    return data


def enrich_all() -> None:
    client = get_client()
    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))

    for chunk_file in chunk_files:
        slug = chunk_file.stem
        print(f"  Enriching: {slug}")
        try:
            enrich_book(slug, client)
        except Exception as e:
            print(f"    Error enriching {slug}: {e}")


if __name__ == "__main__":
    print("Enriching chunks via LM Studio...")
    print("Make sure LM Studio is running at http://localhost:1234")
    enrich_all()
    print("Done.")
