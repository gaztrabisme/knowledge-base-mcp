# MCP server config

`mcp.json` shows two equivalent ways to register the server. The same JSON works in
Claude Code, Claude Desktop, and Cursor — only the file location differs per client.

- **`knowledge-base-uvx`** — zero-install: `uvx` fetches and runs the package straight
  from GitHub. Requires `uv` on your PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **`knowledge-base-dev`** — for a local clone where you ran `pip install -e .` (puts the
  `kb` command on PATH).

Set `KB_DATA_DIR` to the absolute path where you built your index (`kb build`).

## Register from the CLI (Claude Code)

```bash
claude mcp add -s user knowledge-base \
  -e KB_DATA_DIR=/absolute/path/to/kb-data \
  -- uvx --from git+https://github.com/gaztrabisme/knowledge-base-mcp kb serve
```

Then verify inside Claude Code with `/mcp` — it should show `knowledge-base: connected`.
