# GoldenMatch

**Entity resolution toolkit** -- deduplicate records, match across sources, and maintain golden records.

<div class="grid cards" markdown>

- :material-lightning-bolt: **Fast**

    7,800 records/sec fuzzy matching on a laptop. Exact dedup of 1M records in 8 seconds.

- :material-target: **Accurate**

    97.2% F1 on structured data. 72.2% F1 on product matching with LLM boost.

- :material-lock: **Private**

    PPRL (Privacy-Preserving Record Linkage) with bloom filters and multi-party protocols.

- :material-database: **SQL Native**

    Use from PostgreSQL (`CREATE EXTENSION`) or DuckDB (`pip install goldenmatch-duckdb`).

</div>

## Install

```bash
pip install goldenmatch
```

## 30-Second Demo

```python
import goldenmatch as gm

result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85})
print(f"{result.total_clusters} clusters, {result.match_rate:.0%} match rate")
result.golden.write_csv("golden_records.csv")
```

## Use From SQL

=== "PostgreSQL"

    ```sql
    CREATE EXTENSION goldenmatch_pg;
    SELECT * FROM goldenmatch.goldenmatch_dedupe_pairs(
        'customers', '{"exact": ["email"]}'
    );
    ```

=== "DuckDB"

    ```python
    import duckdb, goldenmatch_duckdb
    con = duckdb.connect()
    goldenmatch_duckdb.register(con)
    con.sql("SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler')")
    ```

## Interfaces

| Interface | Install | Best For |
|-----------|---------|----------|
| **Python API** | `pip install goldenmatch` | Notebooks, scripts, AI agents |
| **CLI** | Same package, 21 commands | Terminal workflows |
| **TUI** | `goldenmatch tui` | Interactive exploration |
| **REST API** | `goldenmatch serve` | Microservices |
| **PostgreSQL** | [Pre-built binaries](https://github.com/benzsevern/goldenmatch-extensions/releases) | Production databases |
| **DuckDB** | `pip install goldenmatch-duckdb` | Analytics |
| **MCP** | `goldenmatch mcp-serve` | AI assistants |

## Links

- [PyPI](https://pypi.org/project/goldenmatch/) -- `pip install goldenmatch`
- [GitHub](https://github.com/benzsevern/goldenmatch) -- Source code
- [SQL Extensions](https://github.com/benzsevern/goldenmatch-extensions) -- PostgreSQL + DuckDB
