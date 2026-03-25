# Using GPT-4o-mini as an Entity Resolution Judge: 95% Precision for $0.04

Entity resolution -- finding records that refer to the same real-world entity -- is one of the oldest problems in data management. And for structured data (names, addresses, phone numbers), we've largely solved it. Modern fuzzy matching algorithms routinely achieve 95%+ accuracy.

Product data is a different story.

---

THE PRODUCT MATCHING PROBLEM

Consider matching these two records:

Source A: "Sony Cyber-shot DSC-T77 Silver"
Source B: "Sony - Cyber-shot 10.1-Megapixel Digital Camera - Silver"

A human instantly recognizes these as the same product. But for an algorithm, the token sort ratio is 0.72, Jaro-Winkler gives 0.81, and embedding cosine similarity is 0.83. All borderline. None confident.

The fundamental issue: product names encode different information across sources. One includes the model number, the other includes megapixels. They describe the same thing using different words.

Traditional approaches hit a wall:

Fuzzy matching alone gets 37% F1 on the Abt-Buy benchmark.
Embedding + ANN retrieval gets 44.5% F1 -- better recall, but precision drops to 35%.
Cross-encoder reranking gets 42% F1 -- marginal improvement.
Fine-tuned classifier gets 58.7% F1 -- but needs labeled data you don't have.

---

THE THREE-TIER LLM APPROACH

What if we let fuzzy/embedding matching handle the easy cases and only call an LLM for the hard ones?

In GoldenMatch v0.3.0, I implemented a three-tier scoring system:

Tier 1 -- Auto-accept (score >= 0.95). These are near-identical records. No LLM needed. Score set to 1.0.

Tier 2 -- LLM judge (score 0.75-0.95). These are the borderline pairs where fuzzy/embedding matching isn't confident. Send them to GPT-4o-mini with a simple prompt: "Are these the same entity? YES or NO."

Tier 3 -- Auto-reject (score < 0.75). Too different to be matches. Keep original score. The pipeline filters them out.

---

THE RESULTS

The Abt-Buy dataset has 1,081 products from Abt.com and 1,092 from Buy.com, with 1,097 known matching pairs.

Using embedding + ANN blocking (sentence-transformers all-MiniLM-L6-v2, top-20 candidates per record):

Without LLM: 35.5% precision, 59.4% recall, 44.5% F1.

With LLM (GPT-4o-mini): 95.4% precision, 50.9% recall, 66.3% F1. Cost: $0.04.

The breakdown:
- 1,836 candidate pairs from embedding blocking
- 79 auto-accepted (score >= 0.95)
- 1,757 sent to LLM in 88 batches of 20
- Total cost: $0.0364 (less than 4 cents)
- Precision jumped from 35.5% to 95.4%

---

WHY THE LLM WORKS HERE

The LLM isn't finding matches that fuzzy matching missed. Recall actually drops slightly (59.4% to 50.9%) because the LLM rejects some true matches it's uncertain about.

What the LLM excels at is rejecting false positives.

When an embedding says "Sony Cyber-shot DSC-T77" and "Sony Cyber-shot DSC-T700" are 87% similar, a fuzzy matcher accepts both. The LLM reads the model numbers and knows T77 is not T700.

This is the right division of labor:

Embeddings for recall -- cast a wide net, capture semantically similar records.
LLM for precision -- apply reasoning to reject the ones that look similar but aren't the same.

---

BUDGET CONTROLS

Real-world datasets can have millions of candidate pairs. Without controls, LLM scoring costs could be unpredictable.

GoldenMatch's budget system lets you set a hard cap:

    llm_scorer:
      enabled: true
      model: gpt-4o-mini
      budget:
        max_cost_usd: 5.00
        max_calls: 500

The BudgetTracker estimates token costs before each batch, records actual usage from API responses, and stops when the budget is exhausted. Remaining candidate pairs keep their fuzzy scores -- graceful degradation, not a crash.

For the Abt-Buy dataset, a budget of $0.05 covers everything. For a 100K record product catalog, you'd likely need $1-5 depending on the number of candidate pairs.

You can also configure model tiering -- use GPT-4o-mini for most pairs and escalate to GPT-4o only for the hardest cases (score 0.80-0.90), with a cap on what percentage of the budget goes to the expensive model.

---

STRUCTURED DATA IS ALREADY SOLVED

On the DBLP-ACM bibliographic benchmark (4,910 records, 2,224 known matches), pure fuzzy matching with Jaro-Winkler + token sort + multi-pass blocking achieves 97.2% F1. No embeddings needed. No LLM needed. Cost: $0.

For names, addresses, and other structured fields, the combination of good scorers + good blocking is enough. The LLM approach shines specifically for product data, company names, and other domains where the same entity gets described in fundamentally different ways across sources.

---

THE BIGGER PICTURE

This approach works because LLMs are becoming cheap enough to use as a component in data pipelines, not just as chat interfaces. GPT-4o-mini at $0.15 per million input tokens means scoring 1,700 product pairs costs less than a cup of coffee.

The key architectural insight: don't replace your matching pipeline with an LLM. Use the LLM as a precision filter on top of cheap statistical matching. You get the recall of embeddings and the precision of human judgment, at a cost that scales linearly with the number of borderline pairs.

---

GoldenMatch is open source (MIT license) and available on PyPI:

    pip install goldenmatch

GitHub: github.com/benzsevern/goldenmatch

It includes the LLM scorer, Fellegi-Sunter probabilistic matching, learned blocking, plugin architecture, connectors to Snowflake/Databricks/BigQuery/HubSpot/Salesforce, streaming mode, and multi-table graph entity resolution. 792 tests passing.

---

FULL BENCHMARK RESULTS

DBLP-ACM (bibliographic, 4,910 records): Weighted fuzzy -- 97.2% F1, 1.2 seconds
DBLP-Scholar (bibliographic, 66,879 records): Multi-pass fuzzy -- 74.7% F1, 109 seconds
Abt-Buy (product, 2,173 records): Embedding + LLM -- 66.3% F1, $0.04
Amazon-Google (product, 4,589 records): Embedding + ANN -- 40.5% F1

The structured data story (97.2% F1) is already strong. The product data story is improving -- and the LLM scoring approach is the most promising direction I've seen for closing that gap further.
