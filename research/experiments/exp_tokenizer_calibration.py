"""
Tokenizer Calibration Experiment
=================================
Compares three token-counting methods against the Jina v5 tokenizer (ground truth):
  1. word_count  — len(text.split())          [current method in chunk.py]
  2. jina_v5     — AutoTokenizer from jinaai/jina-embeddings-v5-text-small-retrieval
  3. tiktoken    — cl100k_base encoding

Samples ~300-500 child texts and ~200 parent texts, stratified across 8 books
spanning ML (prose), databases (technical), and security (code-heavy) domains.

Usage:
    python3 scripts/exp_tokenizer_calibration.py

Outputs a statistics table and threshold recalibration recommendations.
"""

import json
import os
import random
import statistics
import time
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"

# Stratified sample: books chosen to cover ML prose, DB technical, security/code
# Format: (book_slug, n_children, n_parents)
SAMPLE_PLAN = [
    # ML prose-heavy
    ("kagglebook_secondedition",                      60, 25),
    ("machinelearningwithpytorchandscikit-learn",     60, 25),
    ("deepreinforcementlearninghands-on_thirdedition",60, 25),
    # ML code-heavy / math
    ("hands-onmachinelearningwithcplusplus_secondedition", 50, 25),
    ("mathematicsofmachinelearning",                  50, 20),
    # Databases — technical prose + SQL
    ("boneh-shoup-applied-cryptography",              60, 25),
    ("postgresql-14-internals",                       50, 25),
    # Security — OWASP / web testing
    ("owasp-wstg-v4.2",                               50, 25),
]
# Approx totals: 440 children, 195 parents

JINA_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"
TIKTOKEN_ENCODING = "cl100k_base"

# Current chunk.py thresholds (in "word" units)
CURRENT = dict(
    PARENT_MAX_TOKENS=1200,
    PARENT_MIN_TOKENS=200,
    CHILD_MAX_TOKENS=250,
    CHILD_MIN_TOKENS=50,
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_chunks() -> tuple[list[str], list[str]]:
    """Return (child_texts, parent_texts) sampled per SAMPLE_PLAN."""
    children: list[str] = []
    parents: list[str] = []

    rng = random.Random(42)

    for slug, n_child, n_parent in SAMPLE_PLAN:
        path = CHUNKS_DIR / f"{slug}.json"
        if not path.exists():
            print(f"  [WARN] missing: {path.name}")
            continue

        data = json.loads(path.read_text())
        book_children = [
            c["text"] for c in data["children"]
            if c.get("text", "").strip()
        ]
        book_parents = [
            p["text"] for p in data["parents"]
            if p.get("text", "").strip()
        ]

        sampled_c = rng.sample(book_children, min(n_child, len(book_children)))
        sampled_p = rng.sample(book_parents, min(n_parent, len(book_parents)))

        children.extend(sampled_c)
        parents.extend(sampled_p)
        print(f"  {slug}: sampled {len(sampled_c)} children, {len(sampled_p)} parents")

    return children, parents


# ---------------------------------------------------------------------------
# Token counters
# ---------------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(text.split())


def make_jina_counter(model_id: str):
    print(f"\nLoading Jina tokenizer: {model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


def make_tiktoken_counter(encoding_name: str):
    enc = tiktoken.get_encoding(encoding_name)

    def count(text: str) -> int:
        return len(enc.encode(text))

    return count


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def ratio_stats(ratios: list[float]) -> dict:
    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)
    p95_idx = int(0.95 * n)
    return {
        "n": n,
        "mean": statistics.mean(ratios),
        "median": statistics.median(ratios),
        "p95": ratios_sorted[min(p95_idx, n - 1)],
        "stdev": statistics.stdev(ratios) if n > 1 else 0.0,
    }


def compute_ratios(
    texts: list[str],
    jina_fn,
    word_fn,
    tiktoken_fn,
) -> tuple[dict, dict]:
    """Return (word_vs_jina_stats, tiktoken_vs_jina_stats)."""
    word_ratios = []
    tiktoken_ratios = []

    for text in texts:
        j = jina_fn(text)
        w = word_fn(text)
        tk = tiktoken_fn(text)
        if w > 0:
            word_ratios.append(j / w)
        if tk > 0:
            tiktoken_ratios.append(j / tk)

    return ratio_stats(word_ratios), ratio_stats(tiktoken_ratios)


# ---------------------------------------------------------------------------
# Raw count stats (absolute token counts per method)
# ---------------------------------------------------------------------------
def raw_stats(texts: list[str], fn) -> dict:
    counts = [fn(t) for t in texts]
    return {
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "min": min(counts),
        "max": max(counts),
    }


# ---------------------------------------------------------------------------
# Threshold recalibration
# ---------------------------------------------------------------------------
def recalibrate(current_word_threshold: int, mean_ratio: float) -> int:
    """Convert a word-count threshold to real-token threshold using mean ratio."""
    return round(current_word_threshold * mean_ratio)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TOKENIZER CALIBRATION EXPERIMENT")
    print("=" * 70)

    # Sample
    print("\n[1/4] Sampling chunks...")
    child_texts, parent_texts = sample_chunks()
    print(f"\n  Total: {len(child_texts)} children, {len(parent_texts)} parents")

    # Load models
    print("\n[2/4] Loading tokenizers...")
    jina_count = make_jina_counter(JINA_MODEL_ID)
    tiktoken_count = make_tiktoken_counter(TIKTOKEN_ENCODING)

    # Measure children
    print("\n[3/4] Measuring children...")
    t0 = time.time()
    child_word_stats, child_tiktoken_stats = compute_ratios(
        child_texts, jina_count, word_count, tiktoken_count
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Measure parents
    print("\n[3b/4] Measuring parents...")
    t0 = time.time()
    parent_word_stats, parent_tiktoken_stats = compute_ratios(
        parent_texts, jina_count, word_count, tiktoken_count
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Raw absolute counts for context
    print("\n[4/4] Raw absolute token counts (Jina ground truth)...")
    child_jina_raw = raw_stats(child_texts, jina_count)
    parent_jina_raw = raw_stats(parent_texts, jina_count)
    child_word_raw = raw_stats(child_texts, word_count)
    parent_word_raw = raw_stats(parent_texts, word_count)
    child_tiktoken_raw = raw_stats(child_texts, tiktoken_count)
    parent_tiktoken_raw = raw_stats(parent_texts, tiktoken_count)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- RATIO TABLE: jina_tokens / counter_tokens ---")
    print(f"  (ratio > 1 means Jina sees MORE tokens than the counter)")
    print()
    hdr = f"{'Group':<12} {'Counter':<12} {'N':>6} {'Mean':>8} {'Median':>8} {'P95':>8} {'Stdev':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = [
        ("Children", "word_count",  child_word_stats),
        ("Children", "tiktoken",    child_tiktoken_stats),
        ("Parents",  "word_count",  parent_word_stats),
        ("Parents",  "tiktoken",    parent_tiktoken_stats),
    ]
    for group, counter, s in rows:
        print(
            f"  {group:<12} {counter:<12} {s['n']:>6} "
            f"{s['mean']:>8.4f} {s['median']:>8.4f} {s['p95']:>8.4f} {s['stdev']:>8.4f}"
        )

    print("\n--- ABSOLUTE TOKEN COUNTS (mean / median / min / max) ---")
    print()
    hdr2 = f"  {'Group':<12} {'Counter':<12} {'Mean':>8} {'Median':>8} {'Min':>6} {'Max':>8}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    abs_rows = [
        ("Children", "word_count",  child_word_raw),
        ("Children", "jina_v5",     child_jina_raw),
        ("Children", "tiktoken",    child_tiktoken_raw),
        ("Parents",  "word_count",  parent_word_raw),
        ("Parents",  "jina_v5",     parent_jina_raw),
        ("Parents",  "tiktoken",    parent_tiktoken_raw),
    ]
    for group, counter, s in abs_rows:
        print(
            f"  {group:<12} {counter:<12} {s['mean']:>8.1f} {s['median']:>8.1f} "
            f"{s['min']:>6} {s['max']:>8}"
        )

    # ---------------------------------------------------------------------------
    # Threshold recalibration
    # ---------------------------------------------------------------------------
    # Use child mean ratio for child thresholds, parent mean ratio for parent thresholds
    child_ratio = child_word_stats["mean"]
    parent_ratio = parent_word_stats["mean"]

    print("\n--- THRESHOLD RECALIBRATION (word → real Jina tokens) ---")
    print(f"  Using child mean ratio:  {child_ratio:.4f}")
    print(f"  Using parent mean ratio: {parent_ratio:.4f}")
    print()
    print(f"  {'Threshold':<25} {'Current (words)':>16} {'Recalibrated (Jina)':>20}")
    print("  " + "-" * 63)
    thresholds = [
        ("PARENT_MAX_TOKENS",  CURRENT["PARENT_MAX_TOKENS"],  parent_ratio),
        ("PARENT_MIN_TOKENS",  CURRENT["PARENT_MIN_TOKENS"],  parent_ratio),
        ("CHILD_MAX_TOKENS",   CURRENT["CHILD_MAX_TOKENS"],   child_ratio),
        ("CHILD_MIN_TOKENS",   CURRENT["CHILD_MIN_TOKENS"],   child_ratio),
    ]
    for name, current_val, ratio in thresholds:
        new_val = recalibrate(current_val, ratio)
        print(f"  {name:<25} {current_val:>16}  →  {new_val:>16}")

    # ---------------------------------------------------------------------------
    # Speed benchmark: Jina vs tiktoken
    # ---------------------------------------------------------------------------
    print("\n--- SPEED BENCHMARK (100 child texts) ---")
    sample100 = child_texts[:100]

    t0 = time.time()
    for t in sample100:
        word_count(t)
    t_word = time.time() - t0

    t0 = time.time()
    for t in sample100:
        tiktoken_count(t)
    t_tiktoken = time.time() - t0

    t0 = time.time()
    for t in sample100:
        jina_count(t)
    t_jina = time.time() - t0

    print(f"  word_count:    {t_word*1000:.2f}ms total  ({t_word/100*1000:.3f}ms/text)")
    print(f"  tiktoken:      {t_tiktoken*1000:.2f}ms total  ({t_tiktoken/100*1000:.3f}ms/text)")
    print(f"  jina_v5:       {t_jina*1000:.2f}ms total  ({t_jina/100*1000:.3f}ms/text)")
    print(f"  tiktoken speedup vs Jina: {t_jina/t_tiktoken:.1f}x")
    print(f"  word_count speedup vs Jina: {t_jina/t_word:.1f}x")

    print("\n" + "=" * 70)
    print("END OF EXPERIMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()
