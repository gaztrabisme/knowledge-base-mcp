"""Book manifest — the single source of truth for which books exist and their metadata.

Replaces the old hardcoded BOOKS dict. The manifest is a JSON file under KB_DATA_DIR:

    {
      "books": {
        "<slug>": {
          "title": "Human Readable Title",
          "domain": "rust",            # top-level subdir of KB_DOCS_DIR, or "general"
          "rerank_input": "child"      # "child" | "heading" (see search._rerank_doc)
        },
        ...
      }
    }

Slug = source filename stem (the existing convention). Ingestion auto-registers every
discovered document; a user may hand-edit titles / rerank_input afterwards and those
edits are preserved across re-ingests (register() never overwrites existing entries).

`rerank_input` controls what text the cross-encoder scores for a book:
  - "child"   — context + child text only (default; safest for formal/generic headings)
  - "heading" — heading_path + context + child (helps when section titles are
                descriptive, e.g. "Async > Futures > Pinning")
"""

import json
import re
from functools import lru_cache

from .config import MANIFEST_PATH

VALID_RERANK_INPUTS = ("child", "heading")
DEFAULT_RERANK_INPUT = "child"


def _titleize(slug: str) -> str:
    """Best-effort human title from a filename stem (used when none is supplied)."""
    words = re.split(r"[-_\s]+", slug.strip())
    return " ".join(w[:1].upper() + w[1:] if w else w for w in words).strip() or slug


def load() -> dict:
    """Read the manifest from disk. Returns {'books': {...}} (empty if absent/corrupt)."""
    if not MANIFEST_PATH.exists():
        return {"books": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"books": {}}
    if not isinstance(data, dict) or "books" not in data:
        return {"books": {}}
    return data


def save(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    books.cache_clear()


def register(slug: str, *, domain: str = "general", title: str | None = None,
             rerank_input: str | None = None) -> dict:
    """Idempotently add a book to the manifest. Existing entries are preserved
    (a user's hand-edited title/rerank_input survives re-ingest). Returns the entry.

    NOTE: callers that register many books in a loop should batch — load() once,
    mutate, save() once — rather than calling this per book. This helper is the
    single-book convenience path and saves on every call.
    """
    manifest = load()
    entry = manifest["books"].get(slug)
    if entry is None:
        entry = {
            "title": title or _titleize(slug),
            "domain": domain,
            "rerank_input": rerank_input if rerank_input in VALID_RERANK_INPUTS else DEFAULT_RERANK_INPUT,
        }
        manifest["books"][slug] = entry
        save(manifest)
    return entry


@lru_cache(maxsize=1)
def books() -> dict:
    """slug -> entry dict. Cached for the process lifetime; save() clears the cache.

    Query-time callers (the MCP server) read this on every tool call, so it must not
    hit disk each time. If you mutate the manifest out-of-band, call books.cache_clear().
    """
    return load().get("books", {})


def known(slug: str) -> bool:
    return slug in books()


def title_for(slug: str) -> str:
    entry = books().get(slug)
    return entry["title"] if entry else _titleize(slug)


def rerank_input_for(slug: str) -> str:
    entry = books().get(slug)
    if not entry:
        return DEFAULT_RERANK_INPUT
    return entry.get("rerank_input", DEFAULT_RERANK_INPUT)
