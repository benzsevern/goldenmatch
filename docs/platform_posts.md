# Platform Posts for GoldenMatch v0.3.0

---

## Hacker News

**Title:** Show HN: GoldenMatch -- Entity resolution with LLM scoring, 97% F1, no Spark

**Body:**

GoldenMatch is an open-source entity resolution toolkit in Python. You give it messy data with duplicates, it finds and merges them.

    pip install goldenmatch
    goldenmatch dedupe customers.csv

I've been working on this for a while and just shipped v0.3.0. Here's what I think is interesting technically:

**The LLM scoring approach works better than I expected.** On the Abt-Buy product matching benchmark, embedding+ANN alone gets 44.5% F1 (35% precision -- lots of false positives). Adding GPT-4o-mini as a judge on the borderline pairs (score 0.75-0.95) pushes it to 66.3% F1 with 95.4% precision. Cost: $0.04 for 1,757 candidate pairs. The key insight is that LLMs are better at rejecting false positives than finding true positives -- so you use cheap fuzzy/embedding matching for recall and the LLM for precision.

**Fellegi-Sunter EM is harder than it looks.** I implemented the classic probabilistic model with EM training. The continuous Gaussian extension (Winkler 2006) collapses when the match rate in blocks is <1% -- both mixture components converge to the same distribution. The fix that works: estimate u-probabilities from random pairs (which are overwhelmingly non-matches), fix them, and only train m-probabilities via EM. This is what Splink does. Discrete comparison vectors with 3 levels are more stable than continuous scores, which is why every production system uses them.

**Benchmarks (Leipzig datasets):**

- DBLP-ACM (bibliographic, 4,910 records): 97.2% F1 with weighted fuzzy, 1.2s
- Abt-Buy (products, 2,173 records): 66.3% F1 with embedding+LLM, $0.04
- Scale: 7,823 rec/s at 100K records, OOMs at ~1M without DuckDB backend

**What it has that Splink/Dedupe.io/Zingg don't (combined):**

- LLM scorer with budget controls and graceful degradation
- Zero-config mode (auto-detects column types, picks scorers and blocking)
- Plugin architecture for custom scorers/transforms/connectors
- Streaming/CDC mode for incremental matching
- Multi-table graph ER with evidence propagation across relationships
- Privacy-preserving matching via bloom filters
- TUI with active learning
- Connectors to Snowflake, Databricks, BigQuery, HubSpot, Salesforce

**What it doesn't do as well:**

- Scale beyond ~500K records in-memory (DuckDB backend is new and unproven)
- Fellegi-Sunter recall is 58% vs 97% for weighted scoring (it's opt-in)
- No distributed compute -- single machine only

Python 3.11+, Polars, RapidFuzz, 792 tests, MIT license.

Code: https://github.com/benzsevern/goldenmatch
PyPI: https://pypi.org/project/goldenmatch/

---

## dev.to

**Title:** How to Deduplicate 100,000 Records in 13 Seconds with Python

**Tags:** python, datascience, opensource, tutorial

**Body:**

You have a CSV with duplicate records. Maybe it's customer data exported from two CRMs, a product catalog merged from multiple vendors, or academic papers from different databases. You need to find the duplicates, decide which to merge, and produce a clean dataset.

Here's how to do it in one command:

```bash
pip install goldenmatch
goldenmatch dedupe your_data.csv
```

That's the zero-config path. GoldenMatch auto-detects your column types (name, email, phone, zip, address), picks appropriate matching algorithms, chooses a blocking strategy, and launches an interactive TUI where you review the results.

But let's go deeper. I'll walk through what happens under the hood and how to tune it for better results.

### What happens when you run `goldenmatch dedupe`

**1. Column Classification**

GoldenMatch profiles your data and classifies each column:

| Detected Type | Scorer | Why |
|--------------|--------|-----|
| Name | Ensemble (best of Jaro-Winkler, token sort, soundex) | Handles misspellings, nicknames, word order |
| Email | Exact (after normalization) | Emails are structured identifiers |
| Phone | Exact (digits only) | Strip formatting, compare digits |
| Zip | Exact | High-cardinality blocking key |
| Address | Token sort | Word order varies ("123 Main St" vs "Main Street 123") |
| Free text | Record embedding | Semantic similarity via sentence-transformers |

**2. Blocking**

Comparing every record against every other record is O(n^2). For 100,000 records, that's 5 billion comparisons. Blocking reduces this to manageable chunks by grouping records that share a key (same zip code, same first 3 characters of name, same Soundex code).

GoldenMatch has 8 blocking strategies. The most interesting new one is **learned blocking** -- it samples your data, scores pairs, and automatically discovers which predicates give the best recall/reduction tradeoff:

```yaml
blocking:
  strategy: learned
  learned_sample_size: 5000
  learned_min_recall: 0.95
```

**3. Scoring**

Within each block, every pair is scored using vectorized NxN comparison via `rapidfuzz.process.cdist`. This releases the GIL, so blocks are scored in parallel via a thread pool.

For hard cases (product matching), you can add LLM scoring:

```yaml
llm_scorer:
  enabled: true
  model: gpt-4o-mini
  budget:
    max_cost_usd: 0.10
```

This sends borderline pairs (score 0.75-0.95) to GPT-4o-mini for a yes/no decision. On the Abt-Buy product benchmark, this boosts precision from 35% to 95% for $0.04.

**4. Clustering**

Scored pairs are clustered using iterative Union-Find with path compression. Each cluster gets a confidence score (weighted combination of minimum edge, average edge, and connectivity) and a bottleneck pair (the weakest link).

**5. Golden Records**

For each cluster, GoldenMatch creates a golden record using one of 5 merge strategies: most_complete, majority_vote, source_priority, most_recent, or first_non_null.

### Performance

| Records | Time | Throughput |
|---------|------|-----------|
| 1,000 | 0.15s | 6,667 rec/s |
| 10,000 | 1.67s | 5,975 rec/s |
| 100,000 | 12.78s | 7,823 rec/s |

Bottleneck is fuzzy scoring (49% of pipeline time), followed by golden record generation (30%).

### The config file

For full control, use a YAML config:

```yaml
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: fuzzy_name_address
    type: weighted
    threshold: 0.85
    fields:
      - field: name
        scorer: ensemble
        weight: 1.0
        transforms: [lowercase, strip]
      - field: zip
        scorer: exact
        weight: 0.5
      - field: phone
        scorer: exact
        weight: 0.3
        transforms: [digits_only]

blocking:
  keys:
    - fields: [zip]
  strategy: adaptive
  max_block_size: 500

golden_rules:
  default_strategy: most_complete
```

### Try it

```bash
pip install goldenmatch
goldenmatch dedupe your_data.csv --output-all --output-dir results/
```

GitHub: https://github.com/benzsevern/goldenmatch
PyPI: https://pypi.org/project/goldenmatch/

792 tests, MIT license. Contributions welcome.

---

## Medium / Towards Data Science

**Title:** Using GPT-4o-mini as an Entity Resolution Judge: 95% Precision for $0.04

**Subtitle:** How a three-tier LLM scoring approach transforms product matching accuracy without breaking the bank

**Body:**

Entity resolution -- finding records that refer to the same real-world entity -- is one of the oldest problems in data management. And for structured data (names, addresses, phone numbers), we've largely solved it. Modern fuzzy matching algorithms like Jaro-Winkler and token sort, combined with good blocking strategies, routinely achieve 95%+ F1 scores.

Product data is a different story.

### The Product Matching Problem

Consider matching these two records:

> **Source A:** "Sony Cyber-shot DSC-T77 Silver"
> **Source B:** "Sony - Cyber-shot 10.1-Megapixel Digital Camera - Silver"

A human instantly recognizes these as the same product. But for an algorithm:
- Token sort ratio: 0.72 (too low to be confident)
- Jaro-Winkler: 0.81 (borderline)
- Embedding cosine similarity: 0.83 (helpful but noisy)

The fundamental issue: product names encode different information across sources. One includes the model number, the other includes megapixels. They describe the same thing using different words.

Traditional approaches hit a wall here:
- **Fuzzy matching alone:** 37% F1 on the Abt-Buy benchmark
- **Embedding + ANN:** 44.5% F1 -- better recall, but precision drops to 35%
- **Cross-encoder reranking:** 42% F1 -- marginal improvement
- **Fine-tuned classifier:** 58.7% F1 -- needs labeled data you don't have

### The Three-Tier LLM Approach

What if we let the fuzzy/embedding matching handle the easy cases and only call an LLM for the hard ones?

In [GoldenMatch](https://github.com/benzsevern/goldenmatch) v0.3.0, I implemented a three-tier scoring system:

**Tier 1: Auto-accept (score >= 0.95)**
These are near-identical records. No LLM needed. Score = 1.0.

**Tier 2: LLM judge (score 0.75-0.95)**
These are the borderline pairs where fuzzy/embedding matching isn't confident. Send them to GPT-4o-mini with a simple prompt: "Are these the same entity? YES or NO."

**Tier 3: Auto-reject (score < 0.75)**
Too different to be matches. Keep original score. The pipeline filters them out.

### Results on Abt-Buy

The [Abt-Buy dataset](https://dbs.uni-leipzig.de/research/projects/object-matching/benchmark-datasets-for-entity-resolution) has 1,081 products from Abt.com and 1,092 from Buy.com, with 1,097 known matching pairs.

Using embedding + ANN blocking (record_embedding with all-MiniLM-L6-v2, ann_pairs strategy, top-20 candidates):

| Strategy | Precision | Recall | F1 | Cost |
|----------|-----------|--------|-----|------|
| Embedding + ANN baseline | 35.5% | 59.4% | 44.5% | $0.00 |
| + LLM scorer (GPT-4o-mini) | **95.4%** | 50.9% | **66.3%** | **$0.04** |

The numbers:
- **1,836 candidate pairs** from embedding blocking
- **79 auto-accepted** (score >= 0.95)
- **1,757 sent to LLM** in 88 batches of 20
- **Total cost: $0.0364** (less than 4 cents)
- **Precision jumped from 35.5% to 95.4%** -- the LLM correctly rejected 1,184 false positives

### Why the LLM Works Here

The LLM isn't finding matches that fuzzy matching missed. Recall actually drops slightly (59.4% to 50.9%) because the LLM rejects some true matches it's uncertain about. What the LLM excels at is **rejecting false positives.**

When an embedding says "Sony Cyber-shot DSC-T77" and "Sony Cyber-shot DSC-T700" are 87% similar, a fuzzy matcher accepts both. The LLM reads the model numbers and knows T77 != T700.

This is the right division of labor:
- **Embeddings for recall:** Cast a wide net, capture semantically similar records
- **LLM for precision:** Apply reasoning to reject the ones that look similar but aren't the same

### Budget Controls

Real-world datasets can have millions of candidate pairs. Without controls, LLM scoring costs could be unpredictable.

GoldenMatch's budget system:

```yaml
llm_scorer:
  enabled: true
  model: gpt-4o-mini
  budget:
    max_cost_usd: 5.00
    max_calls: 500
    warn_at_pct: 80
```

The `BudgetTracker` estimates token costs before each batch, records actual usage from API responses, and stops when the budget is exhausted. Remaining candidate pairs keep their fuzzy scores -- graceful degradation, not a crash.

For the Abt-Buy dataset, a budget of $0.05 covers everything. For a 100K record product catalog, you'd likely need $1-5 depending on the number of candidate pairs.

### The Bigger Picture

This approach works because LLMs are becoming cheap enough to use as a component in data pipelines, not just as chat interfaces. GPT-4o-mini at $0.15 per million input tokens means scoring 1,700 product pairs costs less than a cup of coffee.

The key architectural insight: don't replace your matching pipeline with an LLM -- use the LLM as a precision filter on top of cheap statistical matching. You get the recall of embeddings and the precision of human judgment, at a cost that scales linearly with the number of borderline pairs.

GoldenMatch is open source (MIT) and available on PyPI:

```bash
pip install goldenmatch
```

GitHub: https://github.com/benzsevern/goldenmatch

### Benchmarks Summary

| Dataset | Type | Best Strategy | F1 |
|---------|------|--------------|-----|
| DBLP-ACM | Bibliographic | Weighted fuzzy | 97.2% |
| DBLP-Scholar | Bibliographic (large) | Multi-pass fuzzy | 74.7% |
| Abt-Buy | Product | Embedding + LLM | 66.3% |
| Amazon-Google | Product | Embedding + ANN | 40.5% |

The structured data story (97.2% F1) is already strong. The product data story is improving -- and the LLM scoring approach is the most promising direction I've seen for closing that gap further.

---

## Lobsters

**Title:** GoldenMatch: Entity resolution with Fellegi-Sunter EM, LLM scoring, and learned blocking

**Tags:** python, data, ml, release

**Body:**

I've been building [GoldenMatch](https://github.com/benzsevern/goldenmatch), an entity resolution toolkit in Python. v0.3.0 shipped today with some technically interesting features:

**Fellegi-Sunter with Splink-style EM training.** The classic probabilistic record linkage model, but with a fix for the well-known EM collapse problem: estimate u-probabilities from random pairs (which are overwhelmingly non-matches), fix them, and only train m-probabilities via EM. Discrete comparison vectors with 3 levels -- the continuous Gaussian extension (Winkler 2006) collapses when match rates are <1%. Result: 98.8% precision, 57.6% recall on DBLP-ACM.

**LLM-as-precision-filter.** Send borderline pairs (embedding score 0.75-0.95) to GPT-4o-mini for yes/no decisions. On Abt-Buy product matching: precision jumps from 35% to 95%, F1 from 44.5% to 66.3%, cost $0.04. Budget controller tracks tokens and degrades gracefully.

**Learned blocking.** Instead of hand-picking blocking keys, sample the data, score pairs with static blocking, then evaluate candidate predicates by recall vs reduction ratio. Auto-discovers rules like `title:first_5 AND year:exact`. Matches hand-tuned blocking at 96.9% F1.

**Also:** Plugin architecture via entry points, connectors (Snowflake, Databricks, BigQuery, HubSpot, Salesforce), DuckDB backend, streaming/CDC, multi-table graph ER with evidence propagation, PPRL via bloom filters, TUI with active learning.

Built on Polars + RapidFuzz. 7,823 rec/s at 100K records. 792 tests, MIT.

The honest limitations: OOMs at ~1M records without DuckDB, F-S recall is lower than weighted scoring, product matching without LLM is mediocre.

`pip install goldenmatch` / https://github.com/benzsevern/goldenmatch
