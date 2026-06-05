#!/usr/bin/env python3
"""Scrub copyright-risky verbatim book text from the golden eval data.

The golden eval files embed VERBATIM copyrighted book excerpts in free-text
fields (``source_text``, ``snippet``, ``candidate_text``, ``heading_path``,
etc.). The eval METHODOLOGY -- queries, chunk-id labels, relevance grades, and
metrics -- must be preserved so the eval remains reproducible, but the book
prose must be removed before this repo is published.

This script walks ``research/eval/golden/`` recursively and, for every ``.json``
and ``.jsonl`` file, removes the risky free-text keys wherever they appear
(recursively through nested dicts and lists), preserving all id/label/metric
fields and the overall structure.

Stdlib only. Run from anywhere; the target directory is resolved relative to
this script's location and the script refuses to operate outside it.

Usage:
    python3 research/eval/scrub_golden.py           # scrub in place
    python3 research/eval/scrub_golden.py --dry-run  # report only, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Free-text keys that may carry verbatim copyrighted book text. Removed wherever
# they appear, at any nesting depth. NOTE: ``query``, ``original_query``, and
# ``hyde_passage`` are deliberately NOT here -- they are eval methodology
# (user/agent-authored prompts), not book prose, and must be preserved.
RISKY_KEYS = frozenset(
    {
        "source_text",
        "heading_path",
        "content",
        "child_match",
        "parent_text",
        "child_text",
        "context_text",
        "text",
        "passage",
        "document",
        "snippet",
        "chapter_text",
        "candidate_text",  # verbatim excerpt in audit/judge_validation_sample.jsonl
    }
)

# Directory we are allowed to touch (relative to this script: research/eval/golden).
GOLDEN_DIR = (Path(__file__).resolve().parent / "golden").resolve()


def scrub(obj, removed_counter: dict[str, int]):
    """Recursively return a copy of ``obj`` with RISKY_KEYS stripped.

    Robust to dicts nested in lists nested in dicts. ``removed_counter`` is
    mutated in place: maps removed key name -> instance count.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in RISKY_KEYS:
                removed_counter[key] = removed_counter.get(key, 0) + 1
                continue
            out[key] = scrub(value, removed_counter)
        return out
    if isinstance(obj, list):
        return [scrub(item, removed_counter) for item in obj]
    return obj


def process_jsonl(path: Path, dry_run: bool):
    """Return (modified: bool, removed_counter: dict). Preserves line order."""
    removed: dict[str, int] = {}
    out_lines: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                # Preserve blank lines verbatim.
                out_lines.append(raw.rstrip("\n"))
                continue
            obj = json.loads(stripped)
            before = dict(removed)
            scrubbed = scrub(obj, removed)
            if removed != before:
                changed = True
            out_lines.append(json.dumps(scrubbed, ensure_ascii=False))
    if changed and not dry_run:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed, removed


def process_json(path: Path, dry_run: bool):
    """Return (modified: bool, removed_counter: dict)."""
    removed: dict[str, int] = {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    scrubbed = scrub(obj, removed)
    changed = bool(removed)
    if changed and not dry_run:
        path.write_text(
            json.dumps(scrubbed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return changed, removed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files/keys would change without writing.",
    )
    args = parser.parse_args(argv)

    if not GOLDEN_DIR.is_dir():
        print(f"ERROR: golden dir not found: {GOLDEN_DIR}", file=sys.stderr)
        return 2

    # Guard: refuse to operate anywhere but research/eval/golden.
    if GOLDEN_DIR.name != "golden" or GOLDEN_DIR.parent.name != "eval":
        print(
            f"ERROR: refusing to run; target is not research/eval/golden: {GOLDEN_DIR}",
            file=sys.stderr,
        )
        return 2

    files = sorted(
        p
        for p in GOLDEN_DIR.rglob("*")
        if p.is_file() and p.suffix in (".json", ".jsonl")
    )

    processed = 0
    modified = 0
    total_removed = 0
    failures: list[str] = []
    global_keys: dict[str, int] = {}

    for path in files:
        rel = path.relative_to(GOLDEN_DIR.parent)
        try:
            if path.suffix == ".jsonl":
                changed, removed = process_jsonl(path, args.dry_run)
            else:
                changed, removed = process_json(path, args.dry_run)
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"{rel}: {exc}")
            continue

        processed += 1
        file_removed = sum(removed.values())
        total_removed += file_removed
        for k, n in removed.items():
            global_keys[k] = global_keys.get(k, 0) + n

        if changed:
            modified += 1
            verb = "WOULD MODIFY" if args.dry_run else "MODIFIED"
            detail = ", ".join(f"{k}x{n}" for k, n in sorted(removed.items()))
            print(f"{verb}: {rel}  [{detail}]")

    print()
    mode = "DRY RUN" if args.dry_run else "IN-PLACE"
    print(f"=== scrub_golden summary ({mode}) ===")
    print(f"files processed:       {processed}")
    print(f"files modified:        {modified}")
    print(f"key-instances removed: {total_removed}")
    if global_keys:
        print("by key:")
        for k, n in sorted(global_keys.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}: {n}")
    if failures:
        print()
        print(f"PARSE/IO FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
