# Python API Reference

```python
import goldenmatch as gm
```

## Core Functions

::: goldenmatch.dedupe
::: goldenmatch.dedupe_df
::: goldenmatch.match
::: goldenmatch.match_df
::: goldenmatch.score_strings
::: goldenmatch.score_pair_df
::: goldenmatch.explain_pair_df
::: goldenmatch.pprl_link
::: goldenmatch.evaluate
::: goldenmatch.load_config

## Result Types

::: goldenmatch.DedupeResult
::: goldenmatch.MatchResult

## Config Classes

::: goldenmatch.GoldenMatchConfig
::: goldenmatch.MatchkeyConfig
::: goldenmatch.MatchkeyField
::: goldenmatch.BlockingConfig
::: goldenmatch.LLMScorerConfig

## Pipeline Functions

::: goldenmatch.run_dedupe
::: goldenmatch.build_clusters
::: goldenmatch.build_blocks
::: goldenmatch.load_file

## Scoring

::: goldenmatch.score_pair

## PPRL

::: goldenmatch.run_pprl
::: goldenmatch.pprl_auto_config
::: goldenmatch.compute_bloom_filters

## Streaming

::: goldenmatch.match_one
::: goldenmatch.StreamProcessor

## Advanced

::: goldenmatch.train_em
::: goldenmatch.learn_blocking_rules
::: goldenmatch.llm_score_pairs
::: goldenmatch.llm_cluster_pairs
::: goldenmatch.BudgetTracker
::: goldenmatch.evaluate_pairs
::: goldenmatch.explain_pair
::: goldenmatch.build_lineage
::: goldenmatch.profile_dataframe
::: goldenmatch.unmerge_record
