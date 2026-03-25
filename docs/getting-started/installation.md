# Installation

## Python Package

```bash
pip install goldenmatch
```

### Optional Dependencies

```bash
pip install goldenmatch[ray]        # Distributed processing (10M+ records)
pip install goldenmatch[postgres]    # PostgreSQL integration
pip install goldenmatch[duckdb]      # DuckDB out-of-core backend
pip install goldenmatch[snowflake]   # Snowflake connector
pip install goldenmatch[bigquery]    # BigQuery connector
```

## SQL Extensions

### PostgreSQL

=== "apt (Debian/Ubuntu)"

    ```bash
    curl -LO https://github.com/benzsevern/goldenmatch-extensions/releases/latest/download/postgresql-16-goldenmatch_0.3.0_amd64.deb
    sudo dpkg -i postgresql-16-goldenmatch_0.3.0_amd64.deb
    pip install goldenmatch>=1.1.0
    ```

=== "rpm (RHEL/CentOS)"

    ```bash
    curl -LO https://github.com/benzsevern/goldenmatch-extensions/releases/latest/download/postgresql-16-goldenmatch-0.3.0-1.x86_64.rpm
    sudo rpm -i postgresql-16-goldenmatch-0.3.0-1.x86_64.rpm
    pip install goldenmatch>=1.1.0
    ```

=== "Docker"

    ```bash
    docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres benzsevern/goldenmatch-pg:latest
    ```

=== "Install Script"

    ```bash
    pip install goldenmatch>=1.1.0
    curl -sSL https://raw.githubusercontent.com/benzsevern/goldenmatch-extensions/main/install.sh | bash
    ```

Then in PostgreSQL:

```sql
CREATE EXTENSION goldenmatch_pg;
```

### DuckDB

```bash
pip install goldenmatch-duckdb
```

## Verify Installation

```python
import goldenmatch as gm
print(gm.__version__)
# 1.1.0
```
