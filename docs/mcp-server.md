# ResearchGenie MCP server

`researchgenie-mcp` is a read-only [MCP](https://modelcontextprotocol.io) server over stdio, built on the official MCP Python SDK. It gives any MCP client ResearchGenie's runs, research library and DOI checker. Install ResearchGenie (`pip install -e .` from this repo) and the command is on your PATH.

## Tools

| Tool | What it returns |
|---|---|
| `list_runs(limit=20)` | Recent runs in the output folder: run id, question, state, stage, folder |
| `read_run_artifact(run_id, name)` | The text of one final file: `draft.md`, `quality_report.md`, `validation_report.md`, `references.md`, `evidence_table.md` or `paper.tex`. Any other name, or a path, is refused |
| `search_library(query, limit=10)` | Past runs and stored papers in the local research library that match |
| `summarize_run(run_id)` | Question, state, quality scores, verification counts, draft status, papers, gaps, final files |
| `search_literature(query, limit=10)` | OpenAlex search through ResearchGenie's scholarly gateway (pacing, keys, cache) |
| `verify_doi(doi)` | `verified`, `not_found` (possibly made up) or `unverifiable` (no answer right now) |

**Read-only by design.** No tool writes a run folder, the library or the config, and no tool starts a run. MCP request timeouts are short, and a run takes about 15 minutes, so runs go through the headless CLI (`researchgenie run --detach --json`) instead. The only thing written to disk is the scholarly gateway's response cache (`~/.researchgenie/cache.sqlite`), the same as for any search.

The server reads the OpenAlex key, NCBI key and contact email that `researchgenie config` saved. You don't need to put them in the client config.

## Claude Code

```
claude mcp add researchgenie -- researchgenie-mcp
```

## Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "researchgenie": {
      "command": "researchgenie-mcp"
    }
  }
}
```

Claude Desktop may not see your shell's PATH. If it can't start the server, use the full path (for example `C:\\Users\\you\\project\\.venv\\Scripts\\researchgenie-mcp.exe` or `/home/you/.venv/bin/researchgenie-mcp`).

## OpenClaw

Stop the local Gateway first: OpenClaw's docs say `add` and `--probe` need it stopped. Then:

```
openclaw mcp add researchgenie --command researchgenie-mcp
openclaw mcp doctor researchgenie --probe
```

The probe should report 6 tools. The same entry in OpenClaw's config, under `mcp.servers`:

```json5
{
  mcp: {
    servers: {
      researchgenie: { command: "researchgenie-mcp", args: [] },
    },
  },
}
```

Don't put keys in the `env` block as literal values. ResearchGenie reads them from its own config.

## Check it by hand

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) lists and calls the tools:

```
npx @modelcontextprotocol/inspector researchgenie-mcp
```
