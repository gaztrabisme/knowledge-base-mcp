"""Heading-aware parent/child chunker with code block preservation."""

import json
import re

import tiktoken

from .config import (
    CHUNKS_DIR, MARKDOWN_DIR,
    PARENT_MAX_TOKENS, CHILD_MAX_TOKENS, CHILD_MIN_TOKENS,
)
from . import manifest

PARENT_MIN_TOKENS = 300

_encoder = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    # cl100k_base tracks the Jina v5 tokenizer within ~2%; far closer than word count.
    return len(_encoder.encode(text, disallowed_special=()))


def extract_sections(markdown: str) -> list[dict]:
    """Split markdown into sections by ## headings, tracking heading hierarchy."""
    lines = markdown.split("\n")
    sections = []
    current_h1 = ""
    current_h2 = ""
    current_lines: list[str] = []
    current_heading_path = ""

    def flush():
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({
                    "text": text,
                    "chapter": current_h1,
                    "section": current_h2,
                    "heading_path": current_heading_path,
                })

    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            current_lines.append(line)
            continue

        if in_code:
            current_lines.append(line)
            continue

        h1_match = re.match(r"^#\s+(.+)", line)
        h2_match = re.match(r"^##\s+(.+)", line)

        if h1_match:
            flush()
            current_h1 = h1_match.group(1).strip()
            current_h2 = ""
            current_lines = [line]
            current_heading_path = current_h1
        elif h2_match:
            flush()
            current_h2 = h2_match.group(1).strip()
            current_lines = [line]
            current_heading_path = f"{current_h1} > {current_h2}" if current_h1 else current_h2
        else:
            current_lines.append(line)

    flush()
    return sections


def merge_small_sections(sections: list[dict]) -> list[dict]:
    """Merge consecutive small sections into larger parents."""
    if not sections:
        return sections

    merged = []
    accumulator = None

    for section in sections:
        tokens = estimate_tokens(section["text"])

        if accumulator is None:
            if tokens < PARENT_MIN_TOKENS:
                accumulator = section.copy()
            else:
                merged.append(section)
            continue

        acc_tokens = estimate_tokens(accumulator["text"])
        same_chapter = section["chapter"] == accumulator["chapter"]

        if same_chapter and acc_tokens + tokens <= PARENT_MAX_TOKENS:
            accumulator["text"] = accumulator["text"] + "\n\n" + section["text"]
            if section["section"]:
                accumulator["section"] = section["section"]
                accumulator["heading_path"] = section["heading_path"]
        else:
            merged.append(accumulator)
            if tokens < PARENT_MIN_TOKENS:
                accumulator = section.copy()
            else:
                merged.append(section)
                accumulator = None

    if accumulator:
        merged.append(accumulator)

    return merged


def split_large_section(section: dict) -> list[dict]:
    """If a section exceeds PARENT_MAX_TOKENS, split on ### headings."""
    if estimate_tokens(section["text"]) <= PARENT_MAX_TOKENS:
        return [section]

    lines = section["text"].split("\n")
    subsections = []
    current_lines: list[str] = []
    current_h3 = ""
    in_code = False

    def flush_sub():
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                path = section["heading_path"]
                if current_h3:
                    path = f"{path} > {current_h3}"
                subsections.append({
                    "text": text,
                    "chapter": section["chapter"],
                    "section": section["section"],
                    "heading_path": path,
                })

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            current_lines.append(line)
            continue
        if in_code:
            current_lines.append(line)
            continue

        h3_match = re.match(r"^###\s+(.+)", line)
        if h3_match:
            flush_sub()
            current_h3 = h3_match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush_sub()

    if not subsections:
        return [section]

    final = []
    for sub in subsections:
        if estimate_tokens(sub["text"]) > PARENT_MAX_TOKENS:
            final.extend(_force_split(sub))
        else:
            final.append(sub)
    return final


def _force_split(section: dict) -> list[dict]:
    """Last resort: split on paragraph boundaries when still too large."""
    paragraphs = _split_preserving_code(section["text"])
    parts = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = estimate_tokens(para)
        if current_tokens + pt > PARENT_MAX_TOKENS and current:
            parts.append("\n\n".join(current))
            current = [para]
            current_tokens = pt
        else:
            current.append(para)
            current_tokens += pt

    if current:
        parts.append("\n\n".join(current))

    return [
        {
            "text": part,
            "chapter": section["chapter"],
            "section": section["section"],
            "heading_path": f"{section['heading_path']} (part {i+1})",
        }
        for i, part in enumerate(parts)
    ]


def _split_preserving_code(text: str) -> list[str]:
    """Split text into paragraphs, keeping code blocks as single units."""
    blocks = []
    current: list[str] = []
    in_code = False

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if not in_code:
                if current:
                    para_text = "\n".join(current).strip()
                    if para_text:
                        blocks.extend(p for p in para_text.split("\n\n") if p.strip())
                    current = []
                in_code = True
                current = [line]
            else:
                current.append(line)
                blocks.append("\n".join(current))
                current = []
                in_code = False
        else:
            current.append(line)

    if current:
        para_text = "\n".join(current).strip()
        if para_text:
            blocks.extend(p for p in para_text.split("\n\n") if p.strip())

    return blocks


def _hard_split_block(block: str, max_tokens: int) -> list[str]:
    """Split a single oversized block (code or prose) into token-bounded pieces.

    Code blocks are split by lines and re-fenced so each piece stays valid
    markdown; prose is split by sentences, falling back to words for a single
    monster sentence. Children are match units, so aggressive splitting is fine
    — the parent still returns the whole block as context.
    """
    if block.strip().startswith("```"):
        lines = block.split("\n")
        fence = lines[0] if lines[0].strip().startswith("```") else "```"
        body = lines[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        pieces, cur, cur_tok = [], [], 0
        for ln in body:
            lt = estimate_tokens(ln) or 1
            if cur and cur_tok + lt > max_tokens:
                pieces.append(f"{fence}\n" + "\n".join(cur) + "\n```")
                cur, cur_tok = [ln], lt
            else:
                cur.append(ln)
                cur_tok += lt
        if cur:
            pieces.append(f"{fence}\n" + "\n".join(cur) + "\n```")
        return pieces or [block]

    pieces, cur, cur_tok = [], [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", block):
        st = estimate_tokens(sentence) or 1
        if st > max_tokens:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_tok = [], 0
            wcur, wtok = [], 0
            for word in sentence.split():
                wt = estimate_tokens(word) or 1
                if wcur and wtok + wt > max_tokens:
                    pieces.append(" ".join(wcur))
                    wcur, wtok = [word], wt
                else:
                    wcur.append(word)
                    wtok += wt
            if wcur:
                pieces.append(" ".join(wcur))
        elif cur and cur_tok + st > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tok = [sentence], st
        else:
            cur.append(sentence)
            cur_tok += st
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [block]


def create_children(parent_text: str, parent_id: str) -> list[dict]:
    """Split a parent chunk into children on paragraph boundaries."""
    raw_paragraphs = _split_preserving_code(parent_text)
    paragraphs = []
    for para in raw_paragraphs:
        if estimate_tokens(para) > CHILD_MAX_TOKENS:
            paragraphs.extend(_hard_split_block(para, CHILD_MAX_TOKENS))
        else:
            paragraphs.append(para)
    children = []
    current: list[str] = []
    current_tokens = 0

    def flush_child():
        if current:
            text = "\n\n".join(current).strip()
            if estimate_tokens(text) >= CHILD_MIN_TOKENS:
                children.append({
                    "child_id": f"{parent_id}-c{len(children)}",
                    "parent_id": parent_id,
                    "text": text,
                    "context_text": "",
                    "chunk_type": "code" if text.strip().startswith("```") else "prose",
                })

    for para in paragraphs:
        pt = estimate_tokens(para)

        if current_tokens + pt > CHILD_MAX_TOKENS and current:
            flush_child()
            current = [para]
            current_tokens = pt
        else:
            current.append(para)
            current_tokens += pt

    flush_child()

    if not children and parent_text.strip():
        children.append({
            "child_id": f"{parent_id}-c0",
            "parent_id": parent_id,
            "text": parent_text.strip(),
            "context_text": "",
            "chunk_type": "prose",
        })

    return children


def _extract_last_sentences(text: str, n: int = 2) -> str:
    """Extract last N sentences from text, skipping code blocks."""
    lines = text.split("\n")
    prose_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            prose_lines.append(line)

    prose = " ".join(prose_lines)
    sentences = re.split(r"(?<=[.!?])\s+", prose.strip())
    sentences = [s for s in sentences if len(s) > 20]
    if not sentences:
        return ""
    return " ".join(sentences[-n:])


def chunk_book(book_slug: str) -> dict:
    """Chunk a single book into parent/child structure."""
    md_path = MARKDOWN_DIR / f"{book_slug}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    sections = extract_sections(text)
    sections = merge_small_sections(sections)

    all_parts = []
    for section in sections:
        all_parts.extend(split_large_section(section))

    for i in range(1, len(all_parts)):
        if all_parts[i]["chapter"] == all_parts[i - 1]["chapter"] and all_parts[i]["chapter"]:
            overlap = _extract_last_sentences(all_parts[i - 1]["text"])
            if overlap:
                all_parts[i]["text"] = f"{overlap}\n\n{all_parts[i]['text']}"

    parents = []
    all_children = []
    chapter_num = 0

    for part in all_parts:
        parent_id = f"{book_slug}-p{len(parents)}"

        if part["chapter"] and part["chapter"] != (parents[-1]["metadata"]["chapter_title"] if parents else ""):
            chapter_num += 1

        parent = {
            "parent_id": parent_id,
            "text": part["text"],
            "metadata": {
                "book": book_slug,
                "book_title": manifest.title_for(book_slug),
                "chapter": str(chapter_num),
                "chapter_title": part["chapter"],
                "section": part["section"],
                "heading_path": part["heading_path"],
            },
        }
        parents.append(parent)

        children = create_children(part["text"], parent_id)
        for child in children:
            child["metadata"] = parent["metadata"].copy()
            child["metadata"]["chunk_type"] = child.pop("chunk_type")
        all_children.extend(children)

    return {"parents": parents, "children": all_children}


def chunk_all() -> dict[str, dict]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for md_file in sorted(MARKDOWN_DIR.glob("*.md")):
        slug = md_file.stem
        output = CHUNKS_DIR / f"{slug}.json"
        if output.exists():
            print(f"  Skipping (exists): {slug}")
            with open(output) as f:
                results[slug] = json.load(f)
            continue

        print(f"  Chunking: {slug}")
        data = chunk_book(slug)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        results[slug] = data
        print(f"    → {len(data['parents'])} parents, {len(data['children'])} children")

    return results


if __name__ == "__main__":
    print("Chunking markdown files...")
    all_data = chunk_all()
    total_parents = sum(len(d["parents"]) for d in all_data.values())
    total_children = sum(len(d["children"]) for d in all_data.values())
    print(f"Done. {total_parents} parents, {total_children} children across {len(all_data)} books")
