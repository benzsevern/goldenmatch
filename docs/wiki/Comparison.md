# Comparison with Other Tools

How GoldenMatch compares with other entity resolution tools.

## Feature Comparison

| Feature | **GoldenMatch** | **dedupe** | **Splink** | **Zingg** | **Ditto** |
|---|---|---|---|---|---|
| Language | Python | Python | Python (Spark) | Java (Spark) | Python (PyTorch) |
| Training required | No (optional) | Yes (active learning) | Yes (labels) | Yes (labels) | Yes (1000+ labels) |
| Zero-config mode | Yes | No | No | No | No |
| Interactive TUI | Yes (gold-themed) | No | No | No | No |
| Setup wizard | Yes | No | No | No | No |
| REST API | Yes | Cloud only (paid) | No | No | No |
| MCP Server | Yes (Claude Desktop) | No | No | No | No |
| Database sync | Postgres (incremental) | No | No | No | No |
| Live stream mode | Yes (`watch`) | No | No | No | No |
| GPU required | No (Vertex AI) | No | No | Yes (Spark) | Yes |
| Match explainer | Yes (per-field) | No | Yes | No | No |
| HTML report | Yes | No | Yes | No | No |
| Cluster graph | Yes (interactive) | No | No | No | No |
| Golden record merge | 5 strategies | No | No | No | No |
| License | MIT | MIT | MIT | AGPL | MIT |
| Status | Active | Active | Active | Stale | Academic |

## Accuracy Comparison (Leipzig Benchmarks)

| Dataset | **GoldenMatch** | **dedupe** | **Splink** | **Zingg** | **Ditto** |
|---|---|---|---|---|---|
| DBLP-ACM | **97.4%** | ~96% | ~95% | ~96% | **99.0%** |
| Abt-Buy | **84.7%** | ~75% | ~70% | ~80% | **89.3%** |
| Amazon-Google | **58.6%** | ~50% | ~45% | ~55% | **70.0%** |

GoldenMatch uses Vertex AI's `text-embedding-004` for these results. Without Vertex AI (local CPU-only), results are lower but still competitive.

## Where GoldenMatch Wins

### Ease of Use
No other tool goes from `pip install` to results as quickly:
```bash
pip install goldenmatch
goldenmatch dedupe customers.csv
```
Auto-detects columns, picks scorers, shows results in a TUI. No config file, no training, no labels.

### No GPU Required
Vertex AI provides state-of-the-art embeddings via API. Other tools (Ditto, Zingg) require local GPU hardware or Spark clusters.

### Production Features
- **Database sync** with incremental matching, persistent clusters, golden record versioning
- **REST API** for real-time matching
- **MCP Server** for Claude Desktop integration
- **Live stream mode** for continuous monitoring
- **Match explainer** shows exactly why records matched

### Interactive Experience
Gold-themed TUI with keyboard shortcuts, live threshold tuning, split-view results, and setup wizard. No other deduplication tool has an interactive interface.

## Where Others Win

### Ditto — Higher Accuracy
Ditto achieves 89.3% on Abt-Buy vs GoldenMatch's 84.7%. Ditto uses a fine-tuned DistilBERT model with 1000+ hand-labeled training pairs and data augmentation. If you have the labels and a GPU, Ditto wins on raw accuracy.

### Splink — Better at Scale
Splink is built on Spark and handles billions of records across distributed clusters. GoldenMatch's current scale ceiling is ~10M records per Postgres table. For truly massive datasets, Splink is the right choice.

### dedupe — Active Learning
dedupe's active learning loop is sophisticated — it picks the most informative pairs for you to label, learning from each answer. GoldenMatch's LLM boost simulates this with an LLM instead of a human, but dedupe's approach is more mature.

### Zingg — Spark Ecosystem
Zingg integrates with Hadoop/data lake ecosystems via Spark. If your data lives in HDFS or Delta Lake, Zingg connects natively.

## When to Use What

| Situation | Best Tool |
|---|---|
| Quick dedupe, no config | **GoldenMatch** |
| Best accuracy, have GPU + labels | **Ditto** |
| Billions of records, Spark cluster | **Splink** |
| Active learning with human labels | **dedupe** |
| Hadoop / data lake ecosystem | **Zingg** |
| Production API for real-time matching | **GoldenMatch** |
| Database sync with golden records | **GoldenMatch** |
| Claude Desktop integration | **GoldenMatch** |
