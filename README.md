# knowledge-base-mcp

**Local semantic search over your own documents, inside Claude Code.** An MCP server
that ingests your EPUB/PDF/HTML/Markdown, builds a hybrid (dense + keyword) search
index, and exposes it as tools your AI assistant can call mid-conversation.

Bring your own corpus — this ships **no** books. Point it at a folder, run one
command, and ask Claude questions grounded in *your* library.

```
docs/ (epub·pdf·html·md)
  → markitdown / PyMuPDF-TOC  → heading-aware parent/child chunker
  → Jina v5 dense + FastEmbed BM25 sparse  → Qdrant (DBSF fusion)
  → Ettin-150m cross-encoder rerank  → adaptive cutoff  → FastMCP tools
```

## Quickstart

**1. Install** (needs [`uv`](https://docs.astral.sh/uv/) on PATH):

```bash
# zero-install run straight from GitHub
uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb --help
```

**2. Build an index from your docs:**

```bash
export KB_DATA_DIR=~/kb-data        # where the index is written
uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb build ./my-docs
```

**3. Register it as an MCP server in Claude Code:**

```bash
claude mcp add -s user knowledge-base \
  -e KB_DATA_DIR=$KB_DATA_DIR \
  -- uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb serve
```

Run `/mcp` to confirm it's connected, then ask Claude something your docs cover — it
will call `search` and answer with citations.

## CLI

| Command | What it does |
|---------|--------------|
| `kb build [DOCS_DIR]` | Full pipeline: ingest → chunk → index |
| `kb ingest [DOCS_DIR]` | Convert source docs → clean markdown |
| `kb chunk` | Markdown → parent/child chunks |
| `kb index` | Chunks → Qdrant vector index |
| `kb serve` | Run the MCP server (stdio) |
| `kb search "<query>" [--book SLUG] [--top-k N] [--no-rerank]` | Debug search from the terminal |
| `kb books` | List indexed books (the manifest) |

Global flags: `--data-dir`, `--device {auto,cuda,mps,cpu}`. Everything is also
configurable via environment variables — see [`.env.example`](.env.example).

## MCP tools exposed to Claude

`search`, `search_book`, `get_chapter`, `list_books`, `grep_books`, `read_chunk`.

## ⚠️ Embedding-model license

The **default** embedding model `jinaai/jina-embeddings-v5-text-small-retrieval` is
**CC BY-NC 4.0 — non-commercial**. Fine for personal/research use. For commercial
use, set `KB_EMBEDDING_MODEL` to a permissive model (e.g. `BAAI/bge-small-en-v1.5`,
MIT) and rebuild the index. The reranker (Ettin-150m) is Apache-2.0. See [NOTICE](NOTICE).

## Bring your own documents — respect copyright

You are responsible for the copyright of anything you ingest. The `.gitignore`
excludes book formats and the built index so you don't accidentally republish
copyrighted material in a fork.

## More

- [docs/INSTALL.md](docs/INSTALL.md) — install paths, MCP registration, CPU/GPU, troubleshooting
- [docs/INGESTION.md](docs/INGESTION.md) — corpus layout, formats, the manifest, tuning
- [research/](research/) — the author's evaluation + experimentation apparatus (optional)

Code: Apache-2.0.
