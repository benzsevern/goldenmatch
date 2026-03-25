# Pipeline Overview

GoldenMatch processes data through a 10-step pipeline:

```
ingest -> column_map -> auto_fix -> validate -> standardize
    -> matchkeys -> block -> score -> cluster -> golden -> output
```

## Steps

### 1. Ingest
Loads CSV, Excel, Parquet, or accepts a Polars DataFrame directly via `dedupe_df()`.

### 2. Column Map
Maps source columns to a common schema (e.g., `first_name` -> `name`).

### 3. Auto-Fix
Automatically cleans common data issues (leading/trailing whitespace, encoding errors).

### 4. Validate
Applies validation rules. Records failing validation go to quarantine.

### 5. Standardize
Applies standardization rules (lowercase, strip, phonetic encoding). Native Polars fast path for common transforms.

### 6. Matchkeys
Computes match keys from field transforms. Three types:

- **Exact**: Binary match (email, SSN)
- **Weighted**: Fuzzy scoring with field weights (name, address)
- **Probabilistic**: Fellegi-Sunter with EM-trained parameters

### 7. Block
Groups records into candidate blocks to avoid N^2 comparisons. Strategies:

- **Static**: User-defined blocking keys
- **Learned**: Auto-discovers optimal predicates from data
- **ANN**: Approximate nearest neighbor (embedding-based)

### 8. Score
Scores candidate pairs within each block:

- **Fuzzy**: `rapidfuzz.cdist` for vectorized NxN scoring
- **Probabilistic**: Log-likelihood match weights
- **LLM**: GPT/Claude for borderline pairs (optional)

Blocks scored in parallel via `ThreadPoolExecutor` (or Ray for 10M+ records).

### 9. Cluster
Builds entity clusters from scored pairs using iterative Union-Find. Each cluster includes:

- `members`: Record IDs
- `confidence`: Weighted score (min_edge, avg_edge, connectivity)
- `pair_scores`: Score for each pair
- `bottleneck_pair`: Weakest link

### 10. Golden Record
Merges each cluster into a single canonical record using configurable strategies:

- `most_complete`: Prefer non-null values
- `longest`: Prefer longest string
- `majority`: Majority vote
- `firstmost`: First non-null

## Configuration

All steps are configured via YAML or Python kwargs:

```yaml
matchkeys:
  - name: email_exact
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: name_fuzzy
    type: weighted
    threshold: 0.85
    fields:
      - field: name
        scorer: jaro_winkler
        weight: 0.85

blocking:
  keys:
    - fields: [zip]

golden_rules:
  default_strategy: most_complete
```
