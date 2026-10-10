# DataMind MCP server

Exposes the Text-to-SQL agent over the Model Context Protocol so any MCP client
(Claude Desktop, an IDE, your own agent) can ask questions about the Spider databases.

## Tools

| Tool | What it does |
|---|---|
| `query_database(question, db_id, max_rows=100, route="auto")` | Question -> guarded SQL -> rows, plus tokens, cost, latency and which model answered |
| `list_databases()` | Database ids the server can query |
| `describe_database(db_id)` | Tables, columns and `CREATE TABLE` DDL |
| `validate_sql(sql, db_id=None)` | Run the AST guardrails without executing anything |

Resources: `datamind://databases`, `datamind://schema/{db_id}`.

A query that fails (guard rejection, SQLite error, timeout) comes back as a normal result
with `status` other than `success` and an `error` message. Tool errors are reserved for bad
arguments, unknown databases and server-side limits.

## Run it

Build the Spider sample first (`python -m eval.loader`), then:

```bash
# Claude backend (needs ANTHROPIC_API_KEY)
PYTHONPATH=src python -m mcp_server --backend claude

# Local fine-tuned model on vLLM, or local-first routing with Claude fallback.
# --gpu-hourly-usd is mandatory so cost reporting is explicit (0 to ignore GPU cost).
PYTHONPATH=src python -m mcp_server --backend routed --vllm-url http://localhost:8000/v1 \
    --gpu-hourly-usd 0.526 --min-confidence -0.15

# HTTP instead of stdio (loopback by default)
PYTHONPATH=src python -m mcp_server --backend claude --transport http --port 8765
```

Inspect it interactively: `npx @modelcontextprotocol/inspector python -m mcp_server --backend claude`
(with `PYTHONPATH` set to the repo's `src` directory).

## Claude Desktop

Add to `claude_desktop_config.json`. Claude Desktop does not start in your repo, so use
absolute paths (on Windows, escape backslashes):

```json
{
  "mcpServers": {
    "datamind": {
      "command": "python",
      "args": ["-m", "mcp_server", "--backend", "claude",
               "--db-dir", "C:\\path\\to\\datamind-router\\data\\spider_sample\\databases"],
      "env": {
        "PYTHONPATH": "C:\\path\\to\\datamind-router\\src",
        "ANTHROPIC_API_KEY": "your key"
      }
    }
  }
}
```

Use the Python from the environment where you installed `requirements.txt` (give its full
path as `command` if it is a virtualenv).

## Safety and cost

- **Read-only:** the AST guard allows one SELECT-style statement; the executor then opens
  SQLite read-only with `query_only` and an authorizer that denies writes, ATTACH, PRAGMA and
  `sqlite_*` reads.
- **Spend:** every `query_database` call uses model tokens. `--max-concurrent` (default 4)
  and `--request-timeout` (default 120s) bound it.
- **No authentication on HTTP.** Keep it on loopback, or put it behind something that
  authenticates. The server warns if you bind elsewhere.
- **Results are data:** rows come from the database. A client model should treat them as
  content, not instructions (a poisoned cell is a prompt-injection vector).
- Under stdio the protocol uses stdout, so the server logs to stderr only.
