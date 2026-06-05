"""Concatenate the 118 OWASP Cheat Sheet .md files into one ingestable book.

The cheat sheets are individual markdown files; registering each as a "book" would
bloat config + list_books with 118 entries. Instead we merge them into a single
`security/owasp-cheat-sheets.md` with structure the heading-aware chunker expects:

  # OWASP Cheat Sheet Series        <- H1 = book/chapter
  ## <Sheet Title>                  <- H2 = one section per sheet
  ...sheet body, internal headings demoted to H3+ so they never read as sections...

Idempotent: rewrites the output each run. Regenerate after refreshing the sheets.

Usage: python3 scripts/build_owasp_book.py
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "knowledge-base" / "security" / "owasp-cheat-sheets"
OUT = SRC.parent / "owasp-cheat-sheets.md"


def _title(text: str, fallback: str) -> str:
    """First H1 in the sheet, else a cleaned filename."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return fallback.replace("_", " ").replace("-", " ").strip()


def _demote(text: str, by: int = 2) -> str:
    """Push every heading down `by` levels (cap H6) so the highest becomes H3 —
    keeps a sheet's own #/## from being parsed as book chapters/sections."""
    out, in_code = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code:
            m = re.match(r"^(#{1,6})(\s+.*)", line)
            if m:
                line = "#" * min(len(m.group(1)) + by, 6) + m.group(2)
        out.append(line)
    return "\n".join(out)


def _balance_fences(text: str) -> str:
    """Append a closing fence if a sheet has an odd number of ```-toggling lines.

    The chunker toggles code-mode on any line starting with ``` (chunk.py), so an
    inline ```...``` span on its own line, or a genuinely unclosed block, would
    desync and swallow the *next* sheet's heading. Re-balance per sheet to contain
    it — matches the chunker's line-level model exactly."""
    toggles = sum(1 for ln in text.split("\n") if ln.lstrip().startswith("```"))
    return text + "\n```" if toggles % 2 else text


def main() -> None:
    sheets = sorted(SRC.glob("*.md"))
    if not sheets:
        raise SystemExit(f"No sheets found under {SRC}")

    parts = ["# OWASP Cheat Sheet Series\n"]
    for f in sheets:
        raw = f.read_text(encoding="utf-8").strip()
        title = _title(raw, f.stem)
        # drop the sheet's own leading H1 (becomes the ## section header instead)
        body = re.sub(r"^#\s+.+?\n", "", raw, count=1)
        parts.append(f"## {title}\n\n{_balance_fences(_demote(body))}\n")

    OUT.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT}  ({len(sheets)} sheets, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
