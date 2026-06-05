# Ingesting your own documents

## Corpus layout

Point `KB_DOCS_DIR` at a folder. The ingester walks it **recursively** and treats each
supported file as one "book":

```
my-docs/
  rust/                       ← top-level subdir name becomes the book's `domain`
    the-rust-book.pdf
    async-rust.epub
  databases/
    postgres-internals.md
  cheatsheet.md               ← a flat file → domain "general"
```

- **slug** = the filename stem (`the-rust-book.pdf` → `the-rust-book`).
- **domain** = the first subdirectory under `KB_DOCS_DIR`, or `general` for flat files.

Supported formats: `.epub`, `.pdf`, `.html`/`.htm`, `.md`/`.markdown`.

## Build

```bash
export KB_DATA_DIR=~/kb-data
export KB_DOCS_DIR=~/my-docs
kb build                 # ingest + chunk + index
# or step by step:
kb ingest                # → $KB_DATA_DIR/markdown/*.md  (+ manifest entries)
kb chunk                 # → $KB_DATA_DIR/chunks/*.json
kb index                 # → $KB_DATA_DIR/qdrant/
```

Re-running is **resumable** — files already converted/indexed are skipped, so adding a
few new documents and re-running `kb build` only processes the new ones.

## The manifest

Ingestion writes `$KB_DATA_DIR/manifest.json`, the source of truth for which books
exist and how they're treated:

```json
{
  "books": {
    "the-rust-book": { "title": "The Rust Book", "domain": "rust", "rerank_input": "child" }
  }
}
```

- `title` is auto-derived from the slug; **edit it by hand** for nicer display in
  `list_books`. Your edits survive re-ingest (existing entries are never overwritten).
- `rerank_input` controls what text the cross-encoder scores:
  - `child` (default) — context + chunk text only. Safest when section headings are
    formal/generic (e.g. "Section 4.2", "Chapter 7").
  - `heading` — prepend the heading path. Helps when headings are *descriptive*
    (e.g. "Async > Futures > Pinning"). Flip a book to this if its TOC is meaningful.
- Set `KB_RERANK_INPUT=child|heading` to globally override the per-book setting;
  default `manifest` honors each book's value.

See [`examples/gary-corpus/manifest.json`](../examples/gary-corpus/manifest.json) for a
filled-in example.

## Known limitation: CID-font PDFs

Some PDFs embed subset fonts with no ToUnicode map. Their text extracts as garbage
under PyMuPDF, so the ingester auto-falls back to markitdown — which yields clean body
text but **no headings**. Search and `grep_books` still work for those books;
`get_chapter`/heading-based features will be limited. This is rare.
