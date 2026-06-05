# Example manifest — the author's 56-book corpus

`manifest.json` here is a **worked example** of the manifest format, taken from the
author's personal library (ML, databases, security, distributed systems, Rust). The
books themselves are not included (copyright) — this is only the metadata.

It shows what auto-discovery produces plus hand-tuning:

- `slug` — the source filename stem (the key).
- `title` — human-readable; auto-derived from the slug, then edited by hand.
- `domain` — the top-level subdirectory the file lived in under `KB_DOCS_DIR`.
- `rerank_input` — `heading` or `child`. Here, ML and Rust books use `heading`
  (their section titles are descriptive, e.g. "Async > Futures > Pinning"), while
  databases/distributed/security use `child` (their headings are formal/generic
  section numbers that dilute the match). This split came from A/B evaluation on
  the author's golden set; for your own corpus, start with the default (`child`)
  and flip individual books to `heading` if their headings are descriptive.

To use it as a starting point: copy to your `KB_DATA_DIR/manifest.json`, then edit.
