# dbt Integration

GoldenMatch can be integrated into dbt pipelines as a transformation step. Three approaches, from simplest to most robust.

## Approach 1: Post-hook (Simplest)

Run GoldenMatch after a model materializes:

```yaml
# models/staging/stg_customers.yml
models:
  - name: stg_customers
    config:
      post_hook:
        - "goldenmatch dedupe {{ this }} --config dbt_goldenmatch.yaml --output-all --output-dir ./goldenmatch_output"
```

This calls the CLI directly after the model builds. Works with any warehouse that materializes to a table accessible from the CLI host.

## Approach 2: Jinja Macro

Create a reusable macro for deduplication:

```sql
-- macros/goldenmatch.sql
{% macro goldenmatch_dedupe(model_name, config_path='dbt_goldenmatch.yaml', output_dir='./goldenmatch_output') %}
  {% set command %}
    goldenmatch dedupe {{ ref(model_name) }} --config {{ config_path }} --output-all --output-dir {{ output_dir }}
  {% endset %}
  {{ log("Running GoldenMatch on " ~ model_name, info=True) }}
  {% do run_query("SELECT 1") %}
  -- GoldenMatch runs externally; results land in output_dir
{% endmacro %}
```

Call from `dbt_project.yml`:

```yaml
# dbt_project.yml
on-run-end:
  - "goldenmatch dedupe ./target/stg_customers.csv --config dbt_goldenmatch.yaml --output-all"
```

Or use `dbt run-operation`:

```bash
dbt run-operation goldenmatch_dedupe --args '{"model_name": "stg_customers"}'
```

## Approach 3: Postgres Sync (Recommended)

The most robust approach uses GoldenMatch's incremental database sync. This is ideal when your warehouse is Postgres or you have a Postgres staging layer.

### Setup

```bash
pip install goldenmatch[postgres]
```

### Workflow

```bash
# 1. dbt builds the staging table
dbt run --select stg_customers

# 2. GoldenMatch deduplicates against the table
goldenmatch sync \
  --table stg_customers \
  --connection-string "$DATABASE_URL" \
  --config dbt_goldenmatch.yaml

# 3. Results are in gm_golden_records table
```

### Incremental

After the first full scan, subsequent runs only process new records:

```bash
# Only matches records added since last run
goldenmatch sync --table stg_customers --connection-string "$DATABASE_URL"
```

### Automate with dbt

```yaml
# dbt_project.yml
on-run-end:
  - "goldenmatch sync --table stg_customers --connection-string $DATABASE_URL --config dbt_goldenmatch.yaml"
```

Or schedule with `goldenmatch watch`:

```bash
# Continuously polls for new records every 30 seconds
goldenmatch watch --table stg_customers --connection-string "$DATABASE_URL" --interval 30
```

## Example Config

Save as `dbt_goldenmatch.yaml` alongside your `dbt_project.yml`:

```yaml
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: fuzzy_name_zip
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        scorer: jaro_winkler
        weight: 0.4
        transforms: [lowercase, strip]
      - field: last_name
        scorer: jaro_winkler
        weight: 0.4
        transforms: [lowercase, strip]
      - field: zip
        scorer: exact
        weight: 0.2

blocking:
  strategy: static
  keys:
    - fields: [zip]
  auto_select: true

standardization:
  rules:
    email: [email]
    first_name: [strip, name_proper]
    last_name: [strip, name_proper]
    zip: [zip5]

golden_rules:
  default_strategy: most_complete

output:
  directory: ./goldenmatch_output
  format: csv
```

## Snowflake / BigQuery

GoldenMatch doesn't connect to Snowflake or BigQuery directly, but you can bridge via:

### Snowflake via Postgres FDW

```sql
-- In Postgres
CREATE EXTENSION postgres_fdw;
CREATE SERVER snowflake_server FOREIGN DATA WRAPPER postgres_fdw
  OPTIONS (host 'your-account.snowflakecomputing.com', dbname 'mydb');

CREATE FOREIGN TABLE stg_customers_fdw (
  id INT, first_name TEXT, last_name TEXT, email TEXT, zip TEXT
) SERVER snowflake_server OPTIONS (table_name 'STG_CUSTOMERS');
```

Then run GoldenMatch against the FDW table:

```bash
goldenmatch sync --table stg_customers_fdw --connection-string "$POSTGRES_URL"
```

### BigQuery via Export

```bash
# Export from BigQuery
bq extract --destination_format=CSV 'project:dataset.stg_customers' gs://bucket/customers.csv
gsutil cp gs://bucket/customers.csv ./customers.csv

# Deduplicate
goldenmatch dedupe customers.csv --config dbt_goldenmatch.yaml --output-all

# Load results back
bq load --source_format=CSV 'project:dataset.golden_customers' ./goldenmatch_output/golden.csv
```

Or use GoldenMatch's native GCS support:

```bash
goldenmatch dedupe gs://bucket/customers.csv --config dbt_goldenmatch.yaml --output-all
```

## Metadata Tables

When using Postgres sync, GoldenMatch creates these tables automatically:

| Table | Purpose |
|-------|---------|
| `gm_state` | Processing state, watermarks |
| `gm_clusters` | Persistent cluster membership |
| `gm_golden_records` | Versioned golden records |
| `gm_embeddings` | Cached embeddings for ANN |
| `gm_match_log` | Audit trail of all match decisions |

These can be referenced in downstream dbt models:

```sql
-- models/marts/dim_customers.sql
SELECT
  g.cluster_id,
  g.record_data->>'first_name' AS first_name,
  g.record_data->>'last_name' AS last_name,
  g.record_data->>'email' AS email,
  g.version,
  g.created_at
FROM gm_golden_records g
WHERE g.is_current = true
```

## Tips

- Store `dbt_goldenmatch.yaml` in your dbt project root alongside `dbt_project.yml`
- Use `goldenmatch init` to generate the config interactively
- Run `goldenmatch profile` on your staging table first to understand data quality
- For incremental dbt models, pair with `goldenmatch sync` for efficient incremental matching
- Use `--dashboard` flag to generate an HTML report for data quality review in CI
