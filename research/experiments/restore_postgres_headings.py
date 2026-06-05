"""Restore chapter headings to postgresql-14-internals markdown (v6.1).

postgresql-14-internals is the one CID-font book: PyMuPDF garbles its glyphs, so
ingestion falls back to markitdown — clean body text but ZERO markdown headings.
This leaves its chunks with `heading_path=" (part N)"` and breaks get_chapter/
list_books. The 29 chapter titles DO exist in the body as bare `N Title` lines.

This script strips the 682-line front matter (title page + two TOCs) and promotes
the book title + 29 chapter titles to H1 markdown headings. All-H1 (not H1-book +
H2-chapters) so server.py's H1-only get_chapter/list_books matchers surface them,
consistent with the database-design book. See wiki/decisions.md (v6.1).

IDEMPOTENT: re-running on already-fixed markdown is a no-op. Run this after any
re-ingest of postgres (ingest skips existing markdown, so a normal pipeline run
preserves the fix — this is only needed if the markdown is regenerated from source).

    python3 scripts/restore_postgres_headings.py
    # then: re-chunk + targeted re-index (see ops-runbook.md "PostgreSQL re-index")

Usage notes: after running, the 4 postgres golden queries' positive_child_ids must
be re-anchored to the new chunk ids (re-chunking shifts parent boundaries) — see
ops-runbook.md.
"""

from knowledge_base.config import MARKDOWN_DIR

CHAPTERS = [
    (1, "Introduction"), (2, "Isolation"), (3, "Pages and Tuples"), (4, "Snapshots"),
    (5, "Page Pruning and HOT Updates"), (6, "Vacuum and Autovacuum"), (7, "Freezing"),
    (8, "Rebuilding Tables and Indexes"), (9, "Buffer Cache"), (10, "Write-Ahead Log"),
    (11, "WAL Modes"), (12, "Relation-Level Locks"), (13, "Row-Level Locks"),
    (14, "Miscellaneous Locks"), (15, "Locks on Memory Structures"),
    (16, "Query Execution Stages"), (17, "Statistics"), (18, "Table Access Methods"),
    (19, "Index Access Methods"), (20, "Index Scans"), (21, "Nested Loop"),
    (22, "Hashing"), (23, "Sorting and Merging"), (24, "Hash"), (25, "B-tree"),
    (26, "GiST"), (27, "SP-GiST"), (28, "GIN"), (29, "BRIN"),
]
H1 = "# PostgreSQL 14 Internals"


def main() -> None:
    path = MARKDOWN_DIR / "postgresql-14-internals.md"
    lines = path.read_text(encoding="utf-8").split("\n")

    if lines and lines[0].strip() == H1:
        print("Already restored (H1 present) — no-op.")
        return

    # drop front matter: keep from the body 'About This Book' onward
    body = next(i for i, l in enumerate(lines) if l.strip() == "About This Book" and i > 500)
    if not 600 < body < 700:
        raise SystemExit(f"unexpected body start at line {body + 1}; aborting")
    content = [H1, ""] + lines[body:]

    # promote each chapter's bare 'N' / blank / 'Title' triplet to '# N Title'
    done = 0
    for num, title in CHAPTERS:
        matches = []
        for i, l in enumerate(content):
            if l.strip() == str(num):
                for j in range(i + 1, min(i + 4, len(content))):
                    if content[j].strip() == title:
                        matches.append((i, j))
                        break
        if len(matches) != 1:
            raise SystemExit(f"ch{num} {title!r}: expected 1 body match, found {len(matches)}")
        i, j = matches[0]
        content[i] = f"# {num} {title}"
        content[j] = ""
        done += 1

    path.write_text("\n".join(content), encoding="utf-8")
    print(f"Restored: dropped {body} front-matter lines, promoted 1 book + {done} chapter H1s.")


if __name__ == "__main__":
    main()
