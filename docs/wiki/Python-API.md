# Python API Reference

```python
import goldenmatch as gm
```

## High-Level API

| Function | Description |
|----------|-------------|
| `gm.dedupe(*files, exact=[], fuzzy={}, ...)` | Deduplicate files, returns DedupeResult |
| `gm.match(target, reference, ...)` | Match across files, returns MatchResult |
| `gm.pprl_link(file_a, file_b, fields=[], ...)` | Privacy-preserving linkage |
| `gm.evaluate(*files, config, ground_truth)` | Measure P/R/F1 against ground truth |
| `gm.load_config(path)` | Load YAML config file |

## Result Types

| Class | Key Attributes |
|-------|---------------|
| `gm.DedupeResult` | `.golden`, `.clusters`, `.dupes`, `.unique`, `.match_rate`, `.to_csv()` |
| `gm.MatchResult` | `.matched`, `.unmatched`, `.to_csv()` |
| `gm.EvalResult` | `.precision`, `.recall`, `.f1`, `.tp`, `.fp`, `.fn` |

## Configuration

| Class | Purpose |
|-------|---------|
| `gm.GoldenMatchConfig` | Top-level config (matchkeys, blocking, llm, domain, backend) |
| `gm.MatchkeyConfig` | Match rule (exact, weighted, probabilistic) |
| `gm.MatchkeyField` | Single field config (scorer, weight, transforms) |
| `gm.BlockingConfig` | Blocking strategy (static, adaptive, learned, ann, etc.) |
| `gm.BlockingKeyConfig` | Single blocking key |
| `gm.LLMScorerConfig` | LLM scoring (pairwise or cluster mode) |
| `gm.BudgetConfig` | LLM budget controls |
| `gm.DomainConfig` | Domain extraction settings |
| `gm.GoldenRulesConfig` | Golden record merge strategy |
| `gm.OutputConfig` | Output format and directory |

## Pipeline Functions

| Function | Description |
|----------|-------------|
| `gm.run_dedupe(files, config)` | Full dedupe pipeline |
| `gm.run_match(target, refs, config)` | Full match pipeline |
| `gm.find_exact_matches(lf, mk)` | Polars self-join exact matching |
| `gm.find_fuzzy_matches(df, mk)` | NxN vectorized fuzzy scoring |
| `gm.score_pair(row_a, row_b, fields)` | Score a single pair |
| `gm.score_blocks_parallel(blocks, mk, matched)` | Thread-parallel block scoring |
| `gm.rerank_top_pairs(pairs, df, mk)` | Cross-encoder reranking |
| `gm.build_blocks(lf, config)` | Generate blocks from data |
| `gm.build_clusters(pairs, all_ids)` | Union-Find clustering |
| `gm.build_golden_record(cluster, df, rules)` | Merge cluster into golden record |
| `gm.compute_matchkeys(lf, matchkeys)` | Compute matchkey columns |
| `gm.apply_standardization(lf, rules)` | Apply field standardization |
| `gm.load_file(path)` | Load CSV/Excel/Parquet as LazyFrame |

## Streaming

| Function | Description |
|----------|-------------|
| `gm.match_one(record, df, mk)` | Match single record against dataset |
| `gm.StreamProcessor(df, config)` | Incremental matching processor |
| `gm.run_stream(df, config, source_fn)` | Continuous stream matching loop |

## Evaluation

| Function | Description |
|----------|-------------|
| `gm.evaluate_pairs(predicted, ground_truth)` | Evaluate pair list vs ground truth set |
| `gm.evaluate_clusters(clusters, ground_truth)` | Evaluate clusters vs ground truth |
| `gm.load_ground_truth_csv(path)` | Load ground truth pairs from CSV |

## Cluster Operations

| Function | Description |
|----------|-------------|
| `gm.add_to_cluster(record_id, matches, clusters)` | Incremental cluster update |
| `gm.unmerge_record(record_id, clusters)` | Remove record, re-cluster remaining |
| `gm.unmerge_cluster(cluster_id, clusters)` | Shatter to singletons |
| `gm.compute_cluster_confidence(pair_scores, size)` | Confidence + bottleneck pair |

## LLM Features

| Function | Description |
|----------|-------------|
| `gm.llm_score_pairs(pairs, df, config=...)` | Pairwise LLM yes/no scoring |
| `gm.llm_cluster_pairs(pairs, df, config=...)` | Block-level LLM clustering |
| `gm.llm_label_pairs(pairs, df)` | LLM labels pairs for training |
| `gm.llm_extract_features(df, row_ids, ...)` | LLM feature extraction |
| `gm.boost_accuracy(pairs, df, mk)` | Active learning + retrain |
| `gm.BudgetTracker(config)` | Track LLM cost and enforce limits |

## PPRL (Privacy-Preserving)

| Function | Description |
|----------|-------------|
| `gm.pprl_auto_config(df)` | Auto-detect optimal PPRL parameters |
| `gm.run_pprl(df_a, df_b, config)` | Run full PPRL pipeline |
| `gm.compute_bloom_filters(df, fields, config)` | Compute CLK bloom filters |
| `gm.link_trusted_third_party(party_a, party_b, config)` | TTP linkage mode |
| `gm.link_smc(party_a, party_b, config)` | SMC linkage mode |
| `gm.auto_configure_pprl(df)` | Full auto-config with field profiling |
| `gm.profile_for_pprl(df)` | Profile columns for PPRL suitability |
| `gm.PPRLConfig(fields, threshold, ...)` | PPRL configuration |
| `gm.PartyData(party_id, bloom_filters, ...)` | Encrypted party data |

## Domain Extraction

| Function | Description |
|----------|-------------|
| `gm.discover_rulebooks()` | List all domain packs (7 built-in) |
| `gm.load_rulebook(path)` | Load YAML domain rulebook |
| `gm.save_rulebook(rulebook, path)` | Save rulebook to YAML |
| `gm.match_domain(columns, rulebooks)` | Auto-detect best domain |
| `gm.extract_with_rulebook(df, col, rb)` | Extract features with rulebook |
| `gm.DomainRulebook(name, signals, ...)` | Domain rulebook dataclass |

## Probabilistic (Fellegi-Sunter)

| Function | Description |
|----------|-------------|
| `gm.train_em(df, mk, n_sample_pairs=...)` | EM-train m/u probabilities |
| `gm.score_probabilistic(block_df, mk, em)` | Score pairs with trained model |

## Learned Blocking

| Function | Description |
|----------|-------------|
| `gm.learn_blocking_rules(df, pairs, ...)` | Auto-discover blocking predicates |
| `gm.apply_learned_blocks(lf, rules)` | Apply learned rules to data |

## Data Quality

| Function | Description |
|----------|-------------|
| `gm.auto_fix_dataframe(df)` | Fix nulls, whitespace, encoding |
| `gm.validate_dataframe(df, rules)` | Validate constraints |
| `gm.detect_anomalies(df)` | Flag fake emails, placeholder data |
| `gm.profile_dataframe(df)` | Profile column types and quality |
| `gm.auto_map_columns(df_a, df_b)` | Auto-map schemas across sources |

## Auto-Configuration

| Function | Description |
|----------|-------------|
| `gm.auto_configure(files)` | Generate full config from data profiling |
| `gm.suggest_threshold(scores)` | Otsu's method threshold estimation |

## Explainability

| Function | Description |
|----------|-------------|
| `gm.explain_pair(row_a, row_b, mk)` | NL explanation for a pair |
| `gm.explain_cluster(cluster, df)` | NL summary for a cluster |
| `gm.build_lineage(pairs, df)` | Build per-field lineage |
| `gm.save_lineage(lineage, path)` | Save lineage to JSON |

## Graph ER

| Function | Description |
|----------|-------------|
| `gm.run_graph_er(entities, relationships)` | Multi-table ER with evidence propagation |

## Output

| Function | Description |
|----------|-------------|
| `gm.write_output(df, path, format)` | Write CSV/Parquet |
| `gm.generate_dedupe_report(result)` | Summary report |
| `gm.generate_diff(before, after)` | Before/after diff |
| `gm.rollback_run(run_id)` | Undo a previous run |
