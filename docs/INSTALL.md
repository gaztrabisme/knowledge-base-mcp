# Install & register

## Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (provides `uvx`): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- First run downloads ~1 GB of models from Hugging Face (Jina v5 embedder ~677M,
  Ettin-150m reranker, Qdrant/bm25). They're cached after that.

## Option A — run from GitHub with uvx (recommended)

No clone, no manual install. `uvx` resolves the package and its dependencies into an
isolated environment on demand:

```bash
uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb build ./my-docs
```

Register the server in Claude Code:

```bash
claude mcp add -s user knowledge-base \
  -e KB_DATA_DIR=/absolute/path/to/kb-data \
  -- uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb serve
```

Or hand-edit your MCP config (identical across Claude Code / Desktop / Cursor — see
[`examples/mcp-config/mcp.json`](../examples/mcp-config/mcp.json)):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/gaztrabisme/knowledge-base-mcp", "kb", "serve"],
      "env": { "KB_DATA_DIR": "/absolute/path/to/kb-data" }
    }
  }
}
```

## Option B — clone and install editable

```bash
git clone https://github.com/gaztrabisme/knowledge-base-mcp
cd knowledge-base-mcp
pip install -e .          # puts `kb` and `kb-server` on PATH
kb build ./my-docs
```

MCP config then just uses `"command": "kb", "args": ["serve"]`.

## CPU vs GPU

Set `KB_DEVICE`:

- `auto` (default) — CUDA if available, then Apple MPS, then CPU.
- `cuda` / `mps` / `cpu` — force one.

Indexing is the heavy step. On CPU it works but is slow on large corpora; a CUDA GPU
is much faster. If you hit GPU OOM at index time, lower `KB_INDEX_BATCH` (default 48).
Query-time search is light and fine on CPU.

## Verify

Inside Claude Code, run `/mcp` — you should see `knowledge-base: connected`. Then ask
a question your docs cover. From the terminal you can also smoke-test:

```bash
KB_DATA_DIR=/path/to/kb-data kb search "your query here" --top-k 5
```

## Troubleshooting

- **`spawn uvx ENOENT` / "command not found"** — `uv` isn't on PATH. Install it (above).
  On macOS, GUI apps (Claude Desktop) don't always inherit your shell PATH; use the
  absolute path to `uvx`, or launch from a terminal.
- **Server shows "failed" in `/mcp`** — check you used `uvx` (Python), not `npx`; confirm
  `KB_DATA_DIR` points at a directory where you actually ran `kb build`.
- **Empty results** — did the build finish? `kb books` should list your documents.
