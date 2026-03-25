---
layout: default
title: LLM Integration
nav_order: 13
---

# LLM Integration

GoldenMatch uses LLMs (GPT-4o-mini, Claude) to score borderline pairs that fuzzy matching alone cannot resolve. Two modes: pairwise scoring and in-context block clustering.

---

## Quick start

```python
import goldenmatch as gm

# Enable LLM scoring via convenience API
result = gm.dedupe("products.csv", fuzzy={"title": 0.80}, llm_scorer=True)
```

```bash
# CLI
goldenmatch dedupe products.csv --config config.yaml --llm-scorer
```

Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable.

---

## Pairwise scoring

The default mode. Sends individual borderline pairs to the LLM for match/no-match decisions.

```yaml
llm_scorer:
  enabled: true
  mode: pairwise
  provider: openai          # auto-detected from env vars if omitted
  model: gpt-4o-mini        # cheapest option, default
  auto_threshold: 0.95      # auto-accept pairs above this (no LLM call)
  candidate_lo: 0.75        # lower bound of LLM scoring range
  candidate_hi: 0.95        # upper bound (same as auto_threshold)
  batch_size: 20
```

How it works:
1. Fuzzy scoring produces pairs with scores in [0, 1]
2. Pairs above `auto_threshold` (0.95) are auto-accepted -- no LLM call
3. Pairs in `[candidate_lo, candidate_hi]` (0.75--0.95) are sent to the LLM
4. Pairs below `candidate_lo` (0.75) are rejected
5. The LLM returns a match probability that overrides the fuzzy score

```python
import goldenmatch as gm

scored = gm.llm_score_pairs(borderline_pairs, df, llm_config)
```

---

## Cluster mode

Send entire blocks of borderline records to the LLM for in-context clustering. More efficient than pairwise for large candidate sets.

```yaml
llm_scorer:
  enabled: true
  mode: cluster
  cluster_max_size: 100     # max records per LLM cluster block
  cluster_min_size: 5       # below this, fall back to pairwise
```

How it works:
1. Build connected components from borderline pairs
2. Send each component (block) to the LLM as a clustering task
3. LLM returns cluster assignments
4. Synthesize pair_scores from cluster confidence for compatibility with Union-Find, unmerge, and lineage

```python
import goldenmatch as gm

scored = gm.llm_cluster_pairs(borderline_pairs, df, llm_config)
```

Graceful degradation: cluster mode falls back to pairwise if a block is too small, then stops if the budget is exhausted.

---

## Budget tracking

Control LLM spending with `BudgetConfig`:

```yaml
llm_scorer:
  enabled: true
  budget:
    max_cost_usd: 0.05         # hard cost cap
    max_calls: 100             # max API calls
    warn_at_pct: 80            # warn at 80% of budget
    escalation_model: gpt-4o   # escalate to better model for hard pairs
    escalation_band: [0.80, 0.90]
    escalation_budget_pct: 20  # reserve 20% of budget for escalation
```

```python
import goldenmatch as gm

tracker = gm.BudgetTracker(max_cost_usd=0.05, max_calls=100)
# tracker.record_call(input_tokens, output_tokens, model)
# tracker.remaining_budget
# tracker.total_cost
# tracker.is_exhausted
```

The `BudgetTracker` class tracks token usage, cost, and enforces limits. When the budget runs out, scoring stops gracefully -- pairs are kept at their fuzzy scores.

Budget summary is available in `EngineStats.llm_cost` after a pipeline run.

---

## Model tiering

Automatic escalation sends harder pairs to a better (more expensive) model:

1. **Tier 1**: GPT-4o-mini for most pairs (cheapest)
2. **Tier 2**: GPT-4o for pairs in the escalation band (0.80--0.90)

The escalation budget percentage (default 20%) reserves a portion of the total budget for tier-2 calls.

---

## LLM boost

A separate feature from the LLM scorer. LLM boost fine-tunes an embedding model using LLM-generated labels:

```bash
goldenmatch dedupe products.csv --llm-boost
```

Tiered auto-escalation:
1. **Level 1** -- zero-shot (free, instant)
2. **Level 2** -- bi-encoder fine-tuning (~$0.20, ~2 min CPU)
3. **Level 3** -- Ditto-style cross-encoder with data augmentation (~$0.50, ~5 min CPU)

Active sampling selects the most informative pairs for labeling, reducing cost by ~45%.

LLM boost is most valuable for product matching with local models (MiniLM). For structured data, fuzzy matching alone achieves 97%+ F1.

---

## LLM feature extraction

Extract structured fields from unstructured text using the LLM. O(N) preprocessing, not O(N^2) pair scoring.

```python
import goldenmatch as gm

enhanced_df = gm.llm_extract_features(df, column="description", budget=tracker)
```

---

## Provider configuration

GoldenMatch auto-detects the provider from environment variables:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI (GPT-4o-mini, GPT-4o) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |

Both providers return `(text, input_tokens, output_tokens)` tuples for budget tracking.

---

## Cost benchmarks

| Dataset | Strategy | LLM Cost | F1 |
|---------|----------|----------|-----|
| Abt-Buy (electronics) | Domain + emb + LLM | $0.04 | 72.2% |
| Amazon-Google (software) | emb + ANN + LLM | $0.02 | 45.3% |
| Abt-Buy (Vertex AI + LLM) | Embeddings + GPT-4o-mini | $0.74 | 81.7% |
| Typical 5K dataset | LLM scorer (borderline only) | ~$0.05 | varies |

The LLM scorer sends only borderline pairs (typically 1--5% of all comparisons), keeping costs low. Budget cap of $0.05 covers most datasets.

---

## Python API summary

| Function | Description |
|----------|-------------|
| `gm.llm_score_pairs(pairs, df, config)` | Pairwise LLM scoring |
| `gm.llm_cluster_pairs(pairs, df, config)` | In-context block clustering |
| `gm.BudgetTracker(max_cost_usd, max_calls)` | Track and limit LLM spending |
| `gm.llm_label_pairs(pairs, df)` | Generate LLM-labeled training pairs |
| `gm.llm_extract_features(df, column)` | LLM-based feature extraction |
