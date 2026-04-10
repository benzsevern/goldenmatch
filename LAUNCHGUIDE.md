# GoldenMatch

## Tagline
Find duplicate records in 30 seconds. Zero-config entity resolution, 97.2% F1 out of the box.

## Description
GoldenMatch finds duplicate records with zero configuration. No rules to write, no models to train — point it at a CSV and get deduplicated results in 30 seconds. Achieves 97.2% F1 on DBLP-ACM out of the box. Supports single-source deduplication, cross-source matching, golden record creation, and privacy-preserving record linkage (PPRL) with bloom filters. 30 MCP tools give AI assistants full control over the entity resolution workflow — from analysis and configuration through matching, review, and export. DQBench ER score: 95.30.

## Setup Requirements
No environment variables required. Works out of the box with local CSV files.

## Category
Data & Analytics

## Use Cases
Record deduplication, Entity resolution, Customer matching, Data merging, Golden record creation, Privacy-preserving matching, Cross-source linking, Master data management

## Features
- Zero-config deduplication — auto-detects columns, picks scorers, and runs without training data
- 97.2% F1 on DBLP-ACM benchmark out of the box
- Cross-source matching between two datasets
- Golden record creation — merged canonical records from duplicate clusters
- Privacy-preserving record linkage (PPRL) with bloom filters — match across organizations without sharing raw data (92.4% F1)
- AI agent tools — analyze data, auto-configure, explain matches in natural language
- Review queue for borderline pairs — approve/reject with reasoning
- Strategy comparison — test different matching approaches on your data
- Domain-specific matching rules (people, companies, products)
- Cluster management — inspect, shatter, unmerge, suggest config improvements
- Real-time single-record matching against loaded datasets
- 30 MCP tools covering the full ER workflow
- Integrates with GoldenCheck (scan quality) and GoldenFlow (transform)

## Getting Started
- "Find duplicates in my customer list"
- "Match these two supplier files and show me the overlaps"
- "Explain why these two records were matched"
- "Create golden records from the duplicate clusters"
- "Set up privacy-preserving matching for cross-org deduplication"
- Tool: agent_deduplicate — Run full ER pipeline with confidence gating and reasoning
- Tool: agent_match_sources — Match two files with intelligent strategy selection
- Tool: agent_explain_pair — Natural language explanation for why two records match
- Tool: analyze_data — Profile data, detect domain, recommend ER strategy
- Tool: auto_configure — Generate optimal matching config from data analysis
- Tool: get_golden_record — Get the merged canonical record for a duplicate cluster
- Tool: pprl_link — Privacy-preserving record linkage using bloom filters
- Tool: agent_review_queue — Get borderline pairs awaiting human approval

## Tags
entity-resolution, deduplication, record-matching, golden-record, data-quality, pprl, privacy, bloom-filters, fuzzy-matching, master-data, csv, zero-config, mcp, ai-tools, record-linkage

## Documentation URL
https://benzsevern.github.io/goldenmatch/

## Health Check URL
https://goldenmatch-mcp-production.up.railway.app/mcp/
