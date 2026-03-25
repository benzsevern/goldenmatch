# Reddit Launch Posts for GoldenMatch v0.3.0

---

## r/dataengineering

**Title:** I built an open-source entity resolution tool that hits 97% F1 on structured data and 7,800 rec/s -- no Spark needed

**Body:**

I've been building [GoldenMatch](https://github.com/benzsevern/goldenmatch), an entity resolution / record deduplication toolkit in Python. Just shipped v0.3.0 with some features I haven't seen in other open-source ER tools, so wanted to share.

**The pitch:** `pip install goldenmatch` gives you fuzzy matching, embedding matching, LLM-powered scoring, Fellegi-Sunter probabilistic matching, privacy-preserving matching (bloom filters), streaming/CDC mode, and connectors to Snowflake/Databricks/BigQuery/HubSpot/Salesforce. No Spark, no JVM, no cluster needed.

**Benchmarks (Leipzig datasets):**

| Dataset | Strategy | Precision | Recall | F1 | Time |
|---------|----------|-----------|--------|-----|------|
| DBLP-ACM (bibliographic) | Weighted fuzzy | 97.2% | 97.1% | 97.2% | 1.2s |
| Abt-Buy (product) | Embedding + ANN | 35.5% | 59.4% | 44.5% | 17.8s |
| Abt-Buy (product) | Embedding + ANN + LLM | **95.4%** | 50.9% | **66.3%** | + $0.04 |

**Scale:** 7,823 rec/s at 100K records on a laptop. Pipeline is Polars-native with parallel block scoring via thread pool (rapidfuzz releases the GIL).

**Architecture:**

```
ingest -> auto_fix -> validate -> standardize -> matchkeys -> block -> score -> cluster -> golden -> output
```

- 8 blocking strategies (static, adaptive, sorted_neighborhood, multi_pass, ANN, ann_pairs, canopy, learned)
- 10+ scorers (exact, jaro_winkler, levenshtein, token_sort, soundex, embedding, ensemble, dice/jaccard for PPRL)
- Fellegi-Sunter EM with Splink-style training (fix u from random pairs, train only m)
- Learned blocking auto-discovers predicates from a sample run
- Plugin architecture for custom scorers/transforms/connectors via entry points

**What makes it different from Splink/Dedupe/Zingg:**

- No Spark dependency (Polars handles 100K records in 13s on a laptop)
- LLM scorer that boosts product matching precision from 35% to 95% for $0.04
- Zero-config mode: `goldenmatch dedupe file.csv` auto-detects everything
- TUI with active learning, live threshold tuning, and setup wizard
- Streaming/CDC mode for incremental matching
- Multi-table graph ER with cross-relationship evidence propagation

**Limitations I'll be upfront about:**

- OOMs at ~1M records without DuckDB backend (which exists but is new)
- Fellegi-Sunter is opt-in and lower recall than weighted scoring (73% vs 97% F1)
- Product matching without LLM is mediocre (44% F1)

PyPI: `pip install goldenmatch`
GitHub: https://github.com/benzsevern/goldenmatch
792 tests, MIT license.

Happy to answer questions about the architecture or benchmarks.

---

## r/datascience

**Title:** Entity resolution that actually works on product data: embedding + LLM scoring goes from 44% to 66% F1 for $0.04

**Body:**

I've been working on the record matching / entity resolution problem and wanted to share results from [GoldenMatch](https://github.com/benzsevern/goldenmatch) v0.3.0.

**The problem:** Matching product records across sources is hard. "Sony Cyber-shot DSC-T77" vs "Sony - Cyber-shot 10.1-Megapixel Digital Camera" -- fuzzy string matching fails, embeddings help but create false positives.

**The approach:** Three-tier LLM scoring with budget controls.

1. Score > 0.95: auto-accept (clearly same product)
2. Score 0.75-0.95: send to GPT-4o-mini for yes/no decision
3. Score < 0.75: auto-reject

**Results on Abt-Buy benchmark (1,081 vs 1,092 products):**

| Strategy | Precision | Recall | F1 | Cost |
|----------|-----------|--------|-----|------|
| Embedding + ANN (baseline) | 35.5% | 59.4% | 44.5% | $0 |
| + LLM scorer (GPT-4o-mini) | **95.4%** | 50.9% | **66.3%** | **$0.04** |

The LLM takes precision from 35% to 95% by correctly rejecting false positives that embeddings can't distinguish. 88 API calls, 1,757 candidate pairs evaluated, total cost $0.04.

**Budget controls:** You set `max_cost_usd: 0.10` in config and the system tracks token usage in real-time. When budget is exhausted, remaining pairs keep their fuzzy scores -- graceful degradation, no surprises.

**On structured data (names, addresses):**

97.2% F1 on DBLP-ACM with pure fuzzy matching. No embeddings or LLM needed. Jaro-Winkler + token sort + multi-pass blocking handles it.

Also implemented:
- Fellegi-Sunter probabilistic model with EM training (Splink-style: fix u from random pairs)
- Learned blocking that auto-discovers predicates (96.9% F1 matching hand-tuned blocking)
- Privacy-preserving matching via bloom filter transforms + Dice/Jaccard scoring

It's all open source: `pip install goldenmatch`

GitHub: https://github.com/benzsevern/goldenmatch

Would love feedback from anyone who's worked on ER/dedup at scale.

---

## r/Python

**Title:** goldenmatch -- deduplicate your data with one command, now with LLM scoring and 97% accuracy

**Body:**

```bash
pip install goldenmatch
goldenmatch dedupe customers.csv
```

That's it. It auto-detects your column types (name, email, phone, zip, address), picks scorers, chooses blocking strategy, and launches a TUI for review.

[GoldenMatch](https://github.com/benzsevern/goldenmatch) v0.3.0 just shipped. It's an entity resolution toolkit built on Polars and RapidFuzz.

**What's new in v0.3.0:**

- **LLM scorer with budget controls** -- GPT-4o-mini scores borderline pairs, boosting product matching from 44% to 66% F1 for $0.04. Set `max_cost_usd: 0.10` and it stops when budget is exhausted
- **Fellegi-Sunter probabilistic matching** -- EM-trained parameters with Splink-style training
- **Plugin architecture** -- extend with custom scorers, transforms, connectors via entry points
- **Learned blocking** -- auto-discovers blocking predicates from a sample run
- **5 enterprise connectors** -- Snowflake, Databricks, BigQuery, HubSpot, Salesforce (optional deps)
- **DuckDB backend** -- out-of-core processing for large datasets
- **Natural language explainability** -- "Matched because names are phonetically identical, zip codes match exactly"
- **Streaming/CDC mode** -- incremental record matching
- **Multi-table graph ER** -- match across entity types with evidence propagation

**Speed:** 7,800 records/second at 100K records on a laptop.

**Accuracy:** 97.2% F1 on bibliographic data, 66.3% F1 on product matching (with LLM).

**Tech stack:** Python 3.11+, Polars, RapidFuzz, Pydantic, FAISS (optional), sentence-transformers (optional).

**792 tests,** MIT license, works on Windows/Mac/Linux.

```bash
# With config
goldenmatch dedupe customers.csv --config config.yaml --output-all

# Match mode
goldenmatch match targets.csv --against reference.csv --config config.yaml

# Database sync
goldenmatch sync --table customers --connection-string "$DB" --config config.yaml
```

PyPI: https://pypi.org/project/goldenmatch/
GitHub: https://github.com/benzsevern/goldenmatch

---

## r/selfhosted

**Title:** Open-source alternative to Dedupe.io -- deduplicate records from your terminal with a single command

**Body:**

If you've ever paid for Dedupe.io or spent hours manually deduplicating CSVs, this might save you some pain.

[GoldenMatch](https://github.com/benzsevern/goldenmatch) is a self-hosted entity resolution tool. Install it with pip, run it from your terminal, no cloud service needed.

```bash
pip install goldenmatch
goldenmatch dedupe customers.csv
```

It auto-detects your columns, picks matching strategies, and launches an interactive TUI where you can review matches, tune thresholds, and export results.

**What it does:**
- Finds duplicate records across one or more files
- Matches records between a target and reference file
- Creates golden records (best value from each duplicate group)
- Connects to Snowflake, Databricks, BigQuery, HubSpot, Salesforce
- Syncs with PostgreSQL for continuous deduplication
- Runs as a daemon with health endpoint for production use

**Optional LLM integration:** For hard cases (product matching), it can send borderline pairs to GPT-4o-mini for $0.04 per run. Set a budget cap and it stops automatically.

**Privacy option:** Bloom filter transforms let you match on encrypted PII without exposing the raw data.

97.2% accuracy on name/address dedup. 7,800 records per second. 792 tests. MIT license.

No account needed, no data leaves your machine (unless you opt into LLM scoring), no subscription.

GitHub: https://github.com/benzsevern/goldenmatch
PyPI: `pip install goldenmatch`

---

## r/opensource

**Title:** GoldenMatch v0.3.0 -- entity resolution toolkit with LLM scoring, Fellegi-Sunter, plugin system, and 5 enterprise connectors

**Body:**

Sharing [GoldenMatch](https://github.com/benzsevern/goldenmatch), an open-source entity resolution / record deduplication toolkit I've been building.

v0.3.0 just shipped with a big feature set:

- Zero-config CLI + interactive TUI
- 10+ scoring methods (fuzzy, phonetic, embedding, PPRL)
- 8 blocking strategies including learned (auto-discovers predicates)
- LLM scorer with budget controls ($0.04 to score 1,700 product pairs)
- Fellegi-Sunter probabilistic model with EM training
- Plugin architecture (custom scorers/transforms/connectors via entry points)
- Connectors: Snowflake, Databricks, BigQuery, HubSpot, Salesforce
- DuckDB backend, streaming/CDC, multi-table graph ER
- Natural language explainability, lineage tracking
- REST API + MCP server for Claude Desktop
- Privacy-preserving matching via bloom filters

**Benchmarked on Leipzig datasets:** 97.2% F1 on structured data, 66.3% F1 on product matching (with LLM).

Built with Python 3.11+, Polars, RapidFuzz, Pydantic. 792 tests, MIT license.

GitHub: https://github.com/benzsevern/goldenmatch
PyPI: `pip install goldenmatch`

Contributions welcome -- the plugin system makes it easy to add custom scorers or connectors.

---

## r/CRM

**Title:** Free tool to deduplicate your CRM contacts -- connects to HubSpot and Salesforce directly

**Body:**

Duplicate contacts in your CRM waste time and mess up reporting. I built an open-source tool that connects to HubSpot or Salesforce, finds duplicates, and creates clean golden records.

```bash
pip install goldenmatch
goldenmatch dedupe contacts.csv
```

Or connect directly to your CRM:

```yaml
# config.yaml
input:
  files:
    - connector: hubspot
      source_name: contacts
      credentials_env: HUBSPOT_API_KEY
      object_type: contacts
      properties: [firstname, lastname, email, phone, city, zip]
```

**What it catches:**
- "John Smith" vs "Jon Smith" (fuzzy name matching)
- "555-1234" vs "5551234" (phone normalization)
- "john@company.com" vs "JOHN@Company.COM" (email normalization)
- Partial address matches, zip code matching, phonetic matching

**97.2% accuracy** on name/address deduplication. Interactive TUI lets you review matches before committing.

For harder cases (like matching product SKUs or company names that differ across systems), it can send borderline pairs to GPT-4o-mini for $0.04 per batch.

Free, open source, MIT license. Your data stays on your machine unless you opt into LLM scoring.

GitHub: https://github.com/benzsevern/goldenmatch

---

## r/salesforce

**Title:** Open-source duplicate contact detection for Salesforce -- connects via API, finds fuzzy matches, creates golden records

**Body:**

Built an open-source tool that connects to Salesforce, pulls contacts/leads/accounts, finds duplicates using fuzzy matching, and creates golden records with the best data from each duplicate group.

```bash
pip install goldenmatch[salesforce]
```

Configure your Salesforce credentials:
```bash
export SALESFORCE_USER=admin@company.com
export SALESFORCE_PASSWORD=yourpass
export SALESFORCE_KEY=security_token
```

Then run:
```bash
goldenmatch dedupe --connector salesforce --object Contact --fields Name,Email,Phone,MailingPostalCode
```

It handles:
- Fuzzy name matching (catches misspellings, nicknames)
- Phone/email normalization
- Phonetic matching (John/Jon, Smith/Smyth)
- Address standardization
- Duplicate groups with confidence scores

97.2% accuracy on name/address data. Free, open source, runs on your machine.

GitHub: https://github.com/benzsevern/goldenmatch

---

## r/HubSpot

**Title:** Free tool to clean up duplicate contacts in HubSpot

**Body:**

If you have duplicate contacts in HubSpot, here's an open-source tool that connects via the HubSpot API, finds duplicates using fuzzy matching, and shows you what to merge.

```bash
pip install goldenmatch
```

Set your API key:
```bash
export HUBSPOT_API_KEY=your-key-here
```

Run:
```bash
goldenmatch dedupe --connector hubspot --object contacts --properties firstname,lastname,email,phone,city,zip
```

It launches an interactive interface where you can review matches, adjust the matching threshold, and export results.

Catches things like:
- "John Smith" vs "Jon Smith"
- Different email formats for the same person
- Phone number formatting differences
- Partial address matches

97.2% accuracy. Free, open source, MIT license. Your data is processed locally.

GitHub: https://github.com/benzsevern/goldenmatch

---

## r/analytics

**Title:** Stop manually deduplicating in Excel -- this CLI tool does it automatically with 97% accuracy

**Body:**

If you've ever spent hours doing VLOOKUP + manual review to find duplicate records, this tool automates it.

```bash
pip install goldenmatch
goldenmatch dedupe your_data.csv
```

It auto-detects your column types (names, emails, phones, addresses, zip codes), applies fuzzy matching, and opens an interactive interface where you can:

- See all duplicate groups with confidence scores
- Adjust the matching threshold in real time
- Review borderline matches with per-field explanations
- Export clean data, duplicate list, or golden records

**97.2% accuracy** on name/address data. Handles misspellings, formatting differences, missing fields, and partial matches.

Works with CSV, Excel, Parquet. Also connects to Snowflake, BigQuery, HubSpot, Salesforce.

No coding required beyond the pip install. Free, open source.

GitHub: https://github.com/benzsevern/goldenmatch

---

## r/datasets

**Title:** Open-source tool for deduplicating datasets -- 97% F1 on structured data, supports PPRL for sensitive data

**Body:**

Working with messy data that has duplicates? [GoldenMatch](https://github.com/benzsevern/goldenmatch) is an entity resolution toolkit that finds and merges duplicate records.

**v0.3.0 benchmarks (Leipzig datasets):**

| Dataset | Strategy | F1 |
|---------|----------|-----|
| DBLP-ACM (4,910 records) | Weighted fuzzy | 97.2% |
| Abt-Buy (2,173 products) | Embedding + ANN + LLM | 66.3% |

**Privacy-preserving matching:** If your data contains PII, GoldenMatch supports bloom filter transforms with Dice/Jaccard scoring. Match on encrypted data without exposing raw values.

**Features:**
- Zero-config: `goldenmatch dedupe data.csv` auto-detects everything
- 10+ scoring methods, 8 blocking strategies
- Fellegi-Sunter probabilistic model with EM training
- LLM scoring for hard cases (product matching) -- $0.04 per run
- Lineage tracking: every merge decision is traceable
- Reads CSV, Excel, Parquet, connects to Snowflake/BigQuery/etc.

`pip install goldenmatch` -- 792 tests, MIT license.

GitHub: https://github.com/benzsevern/goldenmatch

---

## r/MachineLearning

**Title:** [P] Fellegi-Sunter + LLM hybrid for entity resolution: 97% F1 on structured data, 66% on products ($0.04 LLM cost)

**Body:**

Sharing [GoldenMatch](https://github.com/benzsevern/goldenmatch), an open-source entity resolution toolkit with a few approaches I found interesting from an ML perspective.

**Three scoring paradigms in one pipeline:**

1. **Weighted fuzzy scoring** (Jaro-Winkler, token sort, ensemble) -- 97.2% F1 on DBLP-ACM. The workhorse for structured data.

2. **Fellegi-Sunter probabilistic model** with EM-trained parameters. Implemented Splink-style: u-probabilities estimated from random pairs (fixed), m-probabilities trained via EM on blocked pairs. Discrete comparison vectors with 2-N levels. Result: 98.8% precision but 57.6% recall (72.8% F1) -- very conservative, useful when false positives are expensive.

   Attempted continuous Gaussian EM (Winkler 2006 extension) but it collapses when the match rate in blocks is <1%. The discrete approach is more stable, which aligns with what Splink, fastLink, and R's RecordLinkage package all do.

3. **LLM-as-scorer** -- Three-tier: auto-accept (>0.95), send borderline (0.75-0.95) to GPT-4o-mini, auto-reject (<0.75). On Abt-Buy product matching with embedding+ANN blocking:

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Embedding baseline | 35.5% | 59.4% | 44.5% |
| + LLM | 95.4% | 50.9% | 66.3% |

88 API calls, $0.04 total. Budget controller tracks tokens and gracefully degrades when cap is reached.

**Other ML components:**
- Learned blocking: predicate selection via recall/reduction evaluation on a sample run. Automatically discovers blocking rules like `title:first_5 AND year:exact`
- Active learning in the TUI: label pairs, train LogisticRegression, re-score
- PPRL via bloom filter transforms (CLK) + Dice/Jaccard scoring

**Limitations:**
- F-S continuous EM needs constrained optimization to prevent component collapse at low match rates
- Product matching recall capped at ~50% by ANN candidate retrieval (if a match isn't in top-20, LLM never sees it)
- No distributed compute -- scales to ~100K records in-memory on a single machine

`pip install goldenmatch` -- Python 3.11+, Polars, RapidFuzz, 792 tests, MIT.

GitHub: https://github.com/benzsevern/goldenmatch

Would appreciate feedback on the F-S implementation from anyone with record linkage experience.
