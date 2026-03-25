---
layout: default
title: Pipeline
nav_order: 8
---

# Pipeline

GoldenMatch runs a 10-step pipeline from raw files to golden records. Each step is a separate module in `goldenmatch/core/`.

```
Files/DB -> Ingest -> Column Map -> Auto Fix -> Validate -> Standardize
         -> Matchkeys -> Block -> Score -> Cluster -> Golden -> Output
```

---

## Pipeline steps

### 1. Ingest

Load data from CSV, Excel, Parquet, or a Polars DataFrame.

```python
from goldenmatch.core.ingest import load_file, load_files

lf = load_file("customers.csv")          # Returns LazyFrame
df = load_file("data.parquet").collect()  # Collect to DataFrame
df = load_files([("a.csv", "source_a"), ("b.csv", "source_b")])
```

Supported formats: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.json`. Cloud paths (`s3://`, `gs://`, `az://`) are handled by `cloud_ingest`.

Each record gets an `__row_id__` (int64) and `__source__` column.

### 2. Column Map

Map columns between different schemas when matching across sources.

```python
gm.auto_map_columns(df_a, df_b)
# {'full_name': 'first_name', 'postal_code': 'zip'}
```

Column maps can be specified in the config or auto-detected. File specs support a third element: `(path, source_name, column_map)`.

### 3. Auto Fix

Automatic data cleaning before validation.

```python
df, fixes = gm.auto_fix_dataframe(df)
# fixes: [{"column": "phone", "fix": "stripped non-digits", "rows": 142}]
```

Fixes include: encoding normalization, whitespace cleanup, type coercion (Polars infers zip/phone as Int64 -- auto-fix converts to string).

### 4. Validate

Apply validation rules and quarantine bad records.

```yaml
validation:
  auto_fix: true
  rules:
    - column: email
      rule_type: regex
      params: { pattern: "^.+@.+\\..+$" }
      action: flag
```

Actions: `flag` (keep but mark), `null` (set value to null), `quarantine` (remove from matching).

### 5. Standardize

Apply per-column standardization transforms.

```yaml
standardization:
  rules:
    email: [email]
    first_name: [name_proper, strip]
    phone: [phone]
    zip: [zip5]
```

Standardizers have a native Polars fast path (`_NATIVE_STANDARDIZERS`) that avoids Python UDFs for common transforms.

### 6. Matchkeys

Compute matchkey columns by applying field transforms.

```python
df = gm.compute_matchkeys(df, matchkeys)
# Adds __mk_exact_email__, __mk_fuzzy_name__, etc.
```

Internal columns are prefixed with `__mk_*__`. Matchkey transforms also have a native Polars fast path (`_try_native_chain`).

### 7. Block

Reduce the comparison space by grouping records that share a blocking key.

```python
blocks = gm.build_blocks(df, blocking_config)
# Returns list of DataFrames, one per block
```

Blocking key choice dominates fuzzy performance -- coarse keys create huge blocks. Use `auto_select: true` to let GoldenMatch pick the best key by histogram analysis.

Dynamic block splitting automatically handles oversized blocks by splitting on the highest-cardinality column.

### 8. Score

Compare record pairs within each block.

**Exact matching** uses Polars self-join (not Python loops):

```python
pairs = gm.find_exact_matches(df, fields)
```

**Fuzzy matching** uses `rapidfuzz.process.cdist` for vectorized NxN scoring:

```python
pairs = gm.find_fuzzy_matches(block_df, matchkey, exclude_pairs=set())
```

**Parallel scoring**: blocks are scored concurrently via `ThreadPoolExecutor`. RapidFuzz's `cdist` releases the GIL, so threads give real parallelism. For 2 or fewer blocks, threading overhead is skipped.

**Intra-field early termination**: after each expensive field, the scorer breaks early if no pair can reach the threshold.

**Backend selection**: `_get_block_scorer(config)` returns `score_blocks_parallel` (threads) or `score_blocks_ray` (Ray distributed) based on `config.backend`.

### 9. Cluster

Group matched pairs into clusters via iterative Union-Find.

```python
clusters = gm.build_clusters(scored_pairs)
# Returns dict[int, dict] with keys: members, size, pair_scores, confidence, bottleneck_pair
```

**Confidence scoring**: `confidence = 0.4 * min_edge + 0.3 * avg_edge + 0.3 * connectivity`. The `bottleneck_pair` identifies the weakest link in each cluster.

**Incremental updates**:

```python
gm.add_to_cluster(record_id, matches, clusters)   # Join or merge clusters
gm.unmerge_record(record_id, clusters)             # Remove and re-cluster
gm.unmerge_cluster(cluster_id, clusters)           # Shatter to singletons
```

### 10. Golden

Merge each cluster into one canonical record.

```python
golden = gm.build_golden_record(cluster, df, golden_rules)
```

Five merge strategies: `most_complete`, `majority_vote`, `source_priority`, `most_recent`, `first_non_null`. Strategies can be set per-field.

### Output

Write results to files or database.

```python
gm.write_output(result, config)
```

Outputs: golden records, duplicates, unique records, lineage JSON, HTML report, dashboard.

Lineage is auto-generated when the pipeline writes output. Each merge decision is saved with per-field score breakdown.

---

## Pipeline entry points

| Entry Point | Description |
|-------------|-------------|
| `gm.dedupe(*files)` | High-level file-based dedupe |
| `gm.dedupe_df(df)` | DataFrame-based dedupe (no file I/O) |
| `gm.match(target, reference)` | File-based list matching |
| `gm.match_df(target_df, ref_df)` | DataFrame-based list matching |
| `run_dedupe(file_specs, config)` | Low-level pipeline |
| `run_match(target_spec, ref_specs, config)` | Low-level pipeline |

The `_run_dedupe_pipeline()` and `_run_match_pipeline()` internal functions are shared by both file-based and DataFrame-based entry points.

---

## Domain extraction (optional step)

Between standardize and matchkeys, domain extraction auto-detects product subdomains and extracts structured fields:

```python
rulebooks = gm.discover_rulebooks()
enhanced_df, low_conf = gm.extract_with_rulebook(df, "title", rulebooks["electronics"])
```

Electronics extraction: brand, model, SKU, color, specs. Software extraction: name, version, edition, platform.
