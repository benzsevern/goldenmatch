---
layout: default
title: MCP Server
nav_order: 18
---

# MCP Server

GoldenMatch provides an MCP (Model Context Protocol) server for integration with Claude Desktop and other MCP-compatible AI assistants.

---

## Remote Server (no install required)

GoldenMatch is available as a hosted remote MCP server on Smithery. Connect from Claude Desktop, Claude Code, or any MCP client without installing anything locally.

Add to your `claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "goldenmatch": {
            "url": "https://goldenmatch-mcp-production.up.railway.app/mcp/"
        }
    }
}
```

Or browse on Smithery: [https://smithery.ai/servers/benzsevern/goldenmatch](https://smithery.ai/servers/benzsevern/goldenmatch)

---

## Local Server

### Start the server

```bash
pip install goldenmatch[mcp]
goldenmatch mcp-serve --file customers.csv --config config.yaml
# Or with HTTP transport:
goldenmatch mcp-serve --file customers.csv --transport http --port 8200
```

Or add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "goldenmatch": {
            "command": "goldenmatch",
            "args": ["mcp-serve", "--file", "customers.csv"]
        }
    }
}
```

With a config file:

```json
{
    "mcpServers": {
        "goldenmatch": {
            "command": "goldenmatch",
            "args": ["mcp-serve", "--file", "customers.csv", "--config", "config.yaml"]
        }
    }
}
```

---

## Tools

The MCP server exposes the following tools to Claude Desktop:

### get_stats

Get dataset statistics: record count, cluster count, match rate, cluster sizes.

```
"How many duplicates did you find?"
```

### find_duplicates

Search for duplicate records matching a query.

```
"Find duplicates of John Smith"
```

### get_cluster

Get details of a specific cluster including all member records and pair scores.

```
"Show me cluster 42"
```

### match_record

Match a new record against the loaded dataset.

```
"Does this record match anything: name=Jane Doe, email=jane@example.com?"
```

### explain_match

Explain why two specific records were matched or not matched.

```
"Why were records 42 and 108 matched?"
```

### unmerge_record

Remove a record from its cluster. Re-clusters the remaining members.

```
"Remove record 215 from its cluster, it was a false match"
```

### suggest_config

Analyze bad merges, identify which fields caused false matches, and suggest threshold/weight changes.

```
"Cluster 5 has a bad merge. What should I change?"
```

### list_domains

List all available domain packs (built-in and custom).

```
"What domain packs are available?"
```

### create_domain

Create a new custom domain pack.

```
"Create a domain pack for automotive parts with fields: part_number, manufacturer, fitment"
```

### test_domain

Test a domain pack against sample data to verify extraction quality.

```
"Test the electronics domain pack on my data"
```

### pprl_auto_config

Analyze data and recommend optimal PPRL parameters.

```
"What PPRL settings should I use for this dataset?"
```

### pprl_link

Run privacy-preserving record linkage.

```
"Link hospital_a.csv and hospital_b.csv using PPRL"
```

---

## How it works

When the MCP server starts, it:

1. Loads data from the specified file(s)
2. Loads or auto-generates a matching configuration
3. Runs the full matching pipeline
4. Caches results in memory
5. Exposes tools via the MCP protocol over stdio

The server uses the `MatchEngine` from `goldenmatch.tui.engine` (which has no Textual dependency) for matching operations.

---

## Example conversation

With Claude Desktop connected to the GoldenMatch MCP server:

**You:** "How many duplicates are in my customer data?"

**Claude:** Uses `get_stats` tool. "Your dataset has 5,000 records with 847 duplicate clusters. The match rate is 12%, meaning about 600 records are duplicates."

**You:** "Show me the biggest cluster"

**Claude:** Uses `get_cluster` tool. "Cluster 14 has 8 members. Here are the records: [table of members with field values and pairwise scores]"

**You:** "Record 215 doesn't belong there, remove it"

**Claude:** Uses `unmerge_record` tool. "Done. Record 215 has been removed from cluster 14. The remaining 7 records have been re-clustered."

**You:** "The name matching is too aggressive. What should I change?"

**Claude:** Uses `suggest_config` tool. "The jaro_winkler threshold for first_name is 0.80, which is catching partial matches like 'John' and 'Jonathan'. I suggest raising it to 0.88 and adding a last_name exact match as a secondary check."

---

## Auto-config mode

If no config file is specified, the MCP server auto-configures matching rules:

```bash
goldenmatch mcp-serve --file customers.csv
```

Auto-config detects column types, assigns appropriate scorers, and picks a blocking strategy. The `suggest_config` tool can then help refine the auto-generated config based on observed results.
