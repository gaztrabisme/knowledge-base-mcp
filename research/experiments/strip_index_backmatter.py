"""Strip book back-of-book INDEX pages that survived v6 front/back-matter stripping.

Signal: a child whose heading_path top segment is exactly `Index (part N)` is a back-of-book
alphabetical index entry (validated across all 15 affected books — EPUB idIndexMarker links,
`dx1-` anchors, and bare page-number formats all share this TOC-assigned heading). This is the
reliable universal signal; it deliberately EXCLUDES real content that merely starts with "Index":
  - `Index of Security Definitions (part N)` — joy-of-cryptography's curated definition reference
  - bare `Index` — pandascookbook's section about the pandas Index object
Both are kept.

Removes flagged index children and any parent left with no surviving child (back-of-book index
parents are dropped as orphans). Writes cleaned chunk files in place; back up with --backup.

Usage:
  python3 scripts/strip_index_backmatter.py --dry-run
  python3 scripts/strip_index_backmatter.py --backup data/chunks.v66-pre-strip.bak
"""

import argparse
import json
import re
import shutil
from pathlib import Path

CHUNKS = Path("data/chunks")
INDEX_HEADING = re.compile(r"^Index \(part \d+\)$")


def is_index_child(ch: dict) -> bool:
    top = (ch.get("metadata", {}).get("heading_path", "") or "").split(">")[0].strip()
    return bool(INDEX_HEADING.match(top))


def page_of(cid: str) -> int:
    m = re.search(r"-p(\d+)-", cid)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()

    if args.backup and not args.dry_run:
        bdir = Path(args.backup)
        if not bdir.exists():
            shutil.copytree(CHUNKS, bdir)
            print(f"backed up {CHUNKS} -> {bdir}\n")
        else:
            print(f"(backup {bdir} already exists — reusing)\n")

    tot_c = tot_p = tot_books = 0
    for f in sorted(CHUNKS.glob("*.json")):
        data = json.loads(f.read_text())
        flagged = [c for c in data["children"] if is_index_child(c)]
        if not flagged:
            continue
        flagged_ids = {c["child_id"] for c in flagged}
        kept_children = [c for c in data["children"] if c["child_id"] not in flagged_ids]
        used_parents = {c["parent_id"] for c in kept_children}
        kept_parents = [p for p in data["parents"] if p["parent_id"] in used_parents]
        # safety: index chunks must be a contiguous tail (max kept page <= min flagged page, loosely)
        fp = [page_of(c["child_id"]) for c in flagged if page_of(c["child_id"]) >= 0]
        kp = [page_of(c["child_id"]) for c in kept_children if page_of(c["child_id"]) >= 0]
        tail_note = ""
        if fp and kp and min(fp) < max(kp) * 0.5:
            tail_note = "  !! WARN: flagged pages not in tail — REVIEW"
        dc, dp = len(flagged), len(data["parents"]) - len(kept_parents)
        tot_c += dc
        tot_p += dp
        tot_books += 1
        rng = f"{min(fp)}-{max(fp)}" if fp else "?"
        print(f"  {f.stem:52s}  -{dc:4d} ch  -{dp:3d} par  (index pp {rng}){tail_note}")
        if not args.dry_run:
            data["children"] = kept_children
            data["parents"] = kept_parents
            f.write_text(json.dumps(data))

    mode = "DRY-RUN" if args.dry_run else "STRIPPED"
    print(f"\n{mode}: {tot_c} index children + {tot_p} index parents across {tot_books} books")


if __name__ == "__main__":
    main()
