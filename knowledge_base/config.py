"""Paths, model names, and tunables — all env-overridable.

Two roots drive every path. Both must work whether the package is run from a
clone (dev) or installed into site-packages via `pip`/`uvx` (no repo around it):

  KB_DATA_DIR  — where the tool WRITES its outputs (markdown, chunks, qdrant index,
                 manifest). Resolution: $KB_DATA_DIR → repo ./data (source checkout)
                 → OS per-user data dir.
  KB_DOCS_DIR  — where the user's SOURCE documents live (the corpus to ingest).
                 Resolution: $KB_DOCS_DIR → repo ./examples/sample-docs.

Books are auto-discovered from KB_DOCS_DIR and tracked in a manifest (manifest.py) —
there is no hardcoded book registry.
"""

import os
import sys
from pathlib import Path

from .embedder import model_dim

ROOT = Path(__file__).resolve().parent.parent


def _os_data_dir() -> Path:
    """Per-user OS data dir for pip/uvx installs (no repo to write into)."""
    plat: str = sys.platform  # annotated str defeats Pyright's single-platform narrowing
    if plat == "darwin":
        return Path.home() / "Library" / "Application Support" / "kb-mcp"
    if os.name == "nt":  # Windows
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "kb-mcp"
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "kb-mcp"


def _default_data_dir() -> Path:
    # A source checkout (has pyproject.toml next to the package) writes into ./data,
    # preserving the clone-and-run dev workflow. An installed copy uses the OS dir.
    if (ROOT / "pyproject.toml").exists():
        return ROOT / "data"
    return _os_data_dir()


DATA_DIR = Path(os.environ.get("KB_DATA_DIR", str(_default_data_dir())))
# User's source documents. Defaults to the bundled sample corpus for a first run.
DOCS_DIR = Path(os.environ.get("KB_DOCS_DIR", str(ROOT / "examples" / "sample-docs")))

MARKDOWN_DIR = DATA_DIR / "markdown"
CHUNKS_DIR = DATA_DIR / "chunks"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Experiment knobs (env-overridable so a candidate index can be built/eval'd into its
# own collection+dir without code edits). Default qdrant dir lives under DATA_DIR.
QDRANT_DIR = Path(os.environ.get("KB_QDRANT_DIR", str(DATA_DIR / "qdrant")))
COLLECTION_NAME = os.environ.get("KB_COLLECTION", "developer_knowledge")
EMBEDDING_MODEL = os.environ.get("KB_EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small-retrieval")
# Dim derives from the model registry unless explicitly overridden.
EMBEDDING_DIM = int(os.environ.get("KB_EMBEDDING_DIM", str(model_dim(EMBEDDING_MODEL))))

# Optional contextual-enrichment LLM (OpenAI-compatible). Off by default; the shipped
# pipeline is index-now-enrich-never. Endpoint/model are env-overridable.
LM_STUDIO_BASE_URL = os.environ.get("KB_ENRICH_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("KB_ENRICH_MODEL", "qwen3.6-35b-a3b-mlx")
BM25_MODEL = "Qdrant/bm25"
RERANKER_MODEL = os.environ.get("KB_RERANKER_MODEL", "cross-encoder/ettin-reranker-150m-v1")

RERANKER_SCORE_FLOOR = float(os.environ.get("KB_RERANKER_FLOOR", "4.0"))
RERANKER_MIN_RESULTS = int(os.environ.get("KB_RERANKER_MIN_RESULTS", "3"))

# Thresholds are real tokens (tiktoken cl100k_base ≈ Jina v5 tokenizer).
# Children tightened to a true 250 for sharper match units; parents kept near their
# prior effective size (~1900 real tokens) to preserve returned context.
PARENT_MAX_TOKENS = 1900
CHILD_MAX_TOKENS = 250
CHILD_MIN_TOKENS = 75

# Device for local embedding/reranking: auto|cuda|mps|cpu. `auto` prefers cuda, then
# Apple mps, then cpu. Indexing batch size (GPU memory-bound) is separately tunable.
DEVICE = os.environ.get("KB_DEVICE", "auto")
INDEX_BATCH = int(os.environ.get("KB_INDEX_BATCH", "48"))
