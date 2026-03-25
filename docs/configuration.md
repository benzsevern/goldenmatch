---
layout: default
title: Configuration
nav_order: 7
---

# Configuration

GoldenMatch uses YAML config files with Pydantic validation. Every section is optional -- GoldenMatch auto-configures what you leave out.

---

## Full YAML reference

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
    rerank: true
    rerank_band: 0.1
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

  - name: probabilistic_fs
    type: probabilistic
    em_iterations: 20
    convergence_threshold: 0.001
    fields:
      - field: first_name
        scorer: jaro_winkler
        levels: 3
        partial_threshold: 0.8
      - field: last_name
        scorer: jaro_winkler
        levels: 2
      - field: zip
        scorer: exact
        levels: 2

  - name: semantic
    type: weighted
    threshold: 0.80
    fields:
      - columns: [title, authors, venue]
        scorer: record_embedding
        weight: 1.0
        column_weights: {title: 2.0, authors: 1.0, venue: 0.5}

blocking:
  strategy: adaptive
  auto_select: true
  auto_suggest: true
  max_block_size: 5000
  skip_oversized: false
  keys:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]
  # Sorted neighborhood
  window_size: 20
  sort_key:
    - column: last_name
      transforms: [lowercase, soundex]
  # Multi-pass
  passes:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]
  union_mode: true
  # ANN blocking
  ann_column: description
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
  # Learned blocking
  learned_sample_size: 5000
  learned_min_recall: 0.95
  learned_min_reduction: 0.90
  learned_predicate_depth: 2
  learned_cache_path: .goldenmatch/learned_blocking.pkl
  # Canopy
  canopy:
    fields: [name, address]
    loose_threshold: 0.3
    tight_threshold: 0.7
    max_canopy_size: 500

golden_rules:
  default_strategy: most_complete
  max_cluster_size: 100
  field_rules:
    email:
      strategy: majority_vote
    first_name:
      strategy: source_priority
      source_priority: [crm, marketing]
    updated_at:
      strategy: most_recent
      date_column: updated_at

standardization:
  rules:
    email: [email]
    first_name: [name_proper, strip]
    last_name: [name_proper, strip]
    phone: [phone]
    zip: [zip5]
    address: [address, strip]
    state: [state]

validation:
  auto_fix: true
  rules:
    - column: email
      rule_type: regex
      params: {pattern: "^.+@.+\\..+$"}
      action: flag
    - column: zip
      rule_type: min_length
      params: {length: 5}
      action: null
    - column: name
      rule_type: not_null
      action: quarantine

domain:
  enabled: true
  pack: electronics

llm_scorer:
  enabled: true
  provider: openai
  model: gpt-4o-mini
  auto_threshold: 0.95
  candidate_lo: 0.75
  candidate_hi: 0.95
  batch_size: 20
  mode: pairwise           # or "cluster" for in-context LLM clustering
  cluster_max_size: 100
  cluster_min_size: 5
  budget:
    max_cost_usd: 0.05
    max_calls: 100
    warn_at_pct: 80

output:
  directory: ./output
  format: csv
  run_name: dedupe_run_001

backend: null              # null (Polars), "ray", or "duckdb"
```

---

## Matchkeys

Three matchkey types:

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `exact` | Binary match on transformed values | `field`, optional `transforms` |
| `weighted` | Weighted average of field scores | `field`, `scorer`, `weight`, `threshold` |
| `probabilistic` | Fellegi-Sunter log-likelihood ratios | `field`, `scorer`, optional `levels` |

### Transforms

Applied to field values before scoring.

| Transform | Description |
|-----------|-------------|
| `lowercase` | Convert to lowercase |
| `uppercase` | Convert to uppercase |
| `strip` | Remove leading/trailing whitespace |
| `strip_all` | Remove all whitespace |
| `soundex` | Soundex phonetic encoding |
| `metaphone` | Metaphone phonetic encoding |
| `digits_only` | Keep only digits |
| `alpha_only` | Keep only letters |
| `normalize_whitespace` | Collapse multiple spaces |
| `token_sort` | Sort tokens alphabetically |
| `first_token` | First whitespace-delimited token |
| `last_token` | Last whitespace-delimited token |
| `substring:start:end` | Substring extraction |
| `qgram:n` | Q-gram tokenization |
| `bloom_filter` or `bloom_filter:ngram:k:size` | Bloom filter (for PPRL) |

### Scorers

| Scorer | Description | Best For |
|--------|-------------|----------|
| `exact` | Binary 0/1 match | Email, phone, ID |
| `jaro_winkler` | Edit distance with prefix bonus | Names |
| `levenshtein` | Normalized Levenshtein distance | General strings |
| `token_sort` | Order-invariant token matching | Names, addresses |
| `soundex_match` | Phonetic match | Names |
| `ensemble` | max(jaro_winkler, token_sort, soundex) | Names with reordering |
| `embedding` | Cosine similarity of embeddings | Semantic matching |
| `record_embedding` | Concatenated multi-field embeddings | Cross-field semantic |
| `dice` | Dice coefficient on bloom filters | PPRL |
| `jaccard` | Jaccard similarity on bloom filters | PPRL |

### Cross-encoder reranking

Add `rerank: true` to a weighted matchkey to re-score borderline pairs with a cross-encoder model:

```yaml
matchkeys:
  - name: fuzzy_name
    type: weighted
    threshold: 0.85
    rerank: true
    rerank_band: 0.1       # pairs within threshold +/- 0.1 get reranked
    rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Blocking

| Strategy | Description |
|----------|-------------|
| `static` | Group by blocking key (default) |
| `adaptive` | Static + recursive sub-blocking for oversized blocks |
| `sorted_neighborhood` | Sliding window over sorted records |
| `multi_pass` | Union of blocks from multiple passes |
| `ann` | ANN via FAISS on embeddings |
| `ann_pairs` | Direct-pair ANN scoring (50--100x faster than `ann`) |
| `canopy` | TF-IDF canopy clustering |
| `learned` | Data-driven predicate selection |

Set `auto_select: true` to auto-pick the best blocking key by histogram analysis. Set `auto_suggest: true` to get blocking suggestions when no keys are specified.

---

## Golden rules

Five merge strategies for building canonical records:

| Strategy | Description |
|----------|-------------|
| `most_complete` | Pick value with fewest nulls |
| `majority_vote` | Most common value across cluster members |
| `source_priority` | Prefer values from specified sources (requires `source_priority` list) |
| `most_recent` | Latest value by date (requires `date_column`) |
| `first_non_null` | First non-null value encountered |

Set a default strategy and override per field:

```yaml
golden_rules:
  default_strategy: most_complete
  field_rules:
    email: { strategy: majority_vote }
    name: { strategy: source_priority, source_priority: [crm, erp] }
```

---

## Standardization

Map column names to standardizer functions:

```yaml
standardization:
  rules:
    email: [email]
    phone: [phone]
    zip: [zip5]
    first_name: [name_proper, strip]
    address: [address, strip]
    state: [state]
```

| Standardizer | Description |
|--------------|-------------|
| `email` | Lowercase, strip, validate format |
| `name_proper` | Title case |
| `name_upper` | Uppercase |
| `name_lower` | Lowercase |
| `phone` | Strip non-digits, normalize format |
| `zip5` | First 5 digits |
| `address` | Normalize abbreviations (St->Street, etc.) |
| `state` | Normalize state abbreviations |
| `strip` | Remove leading/trailing whitespace |
| `trim_whitespace` | Collapse multiple spaces |

---

## Validation

```yaml
validation:
  auto_fix: true
  rules:
    - column: email
      rule_type: regex
      params: { pattern: "^.+@.+\\..+$" }
      action: flag
    - column: name
      rule_type: not_null
      action: quarantine
    - column: zip
      rule_type: min_length
      params: { length: 5 }
      action: null
```

Rule types: `regex`, `min_length`, `max_length`, `not_null`, `in_set`, `format`.
Actions: `flag` (mark but keep), `null` (set to null), `quarantine` (remove from matching).

---

## Settings persistence

- **Global**: `~/.goldenmatch/settings.yaml` -- output mode, default model, API keys
- **Project**: `.goldenmatch.yaml` -- column mappings, thresholds, blocking config

Settings tuned in the TUI can be saved to the project file. Next run picks them up automatically.

---

## Programmatic config

```python
import goldenmatch as gm

config = gm.GoldenMatchConfig(
    matchkeys=[
        gm.MatchkeyConfig(name="exact_email", type="exact",
            fields=[gm.MatchkeyField(field="email", transforms=["lowercase"])]),
        gm.MatchkeyConfig(name="fuzzy_name", type="weighted", threshold=0.85,
            fields=[
                gm.MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
                gm.MatchkeyField(field="zip", scorer="exact", weight=0.3),
            ]),
    ],
    blocking=gm.BlockingConfig(strategy="learned"),
    llm_scorer=gm.LLMScorerConfig(enabled=True, mode="cluster"),
    backend="ray",
)

result = gm.dedupe("data.csv", config=config)
```

Or auto-generate from data:

```python
config = gm.auto_configure([("data.csv", "source")])
```
