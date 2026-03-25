---
layout: default
title: PPRL
nav_order: 12
---

# Privacy-Preserving Record Linkage (PPRL)

Match records across organizations without sharing raw data. GoldenMatch uses bloom filter encoding with HMAC salting and supports both trusted third party (TTP) and secure multi-party computation (SMC) protocols.

---

## Quick start

```python
import goldenmatch as gm

# Auto-configured: profiles data, picks fields and threshold
result = gm.pprl_link("hospital_a.csv", "hospital_b.csv")
print(f"Found {result['match_count']} matches across {len(result['clusters'])} clusters")

# Manual configuration
result = gm.pprl_link(
    "party_a.csv", "party_b.csv",
    fields=["first_name", "last_name", "dob", "zip"],
    threshold=0.85,
    security_level="high",
)
```

CLI:

```bash
goldenmatch pprl link party_a.csv party_b.csv --security-level high
goldenmatch pprl link a.csv b.csv --fields first_name last_name dob zip --threshold 0.85
```

---

## How it works

```
Party A data -> Bloom filter encoding -> Encoded vectors
Party B data -> Bloom filter encoding -> Encoded vectors
                                              |
                                     Dice/Jaccard similarity
                                              |
                                        Matched pairs
```

1. Each field value is converted to character n-grams (e.g., bigrams)
2. N-grams are hashed with multiple hash functions into a bloom filter bit array
3. HMAC salting ensures the same value produces different encodings with different keys
4. Encoded vectors are compared using Dice or Jaccard similarity
5. Pairs above the threshold are matched

---

## Bloom filter parameters

| Parameter | Description | Standard | High | Paranoid |
|-----------|-------------|----------|------|----------|
| `ngram_size` | Character n-gram size | 2 | 2 | 3 |
| `hash_functions` | Number of hash functions (k) | 20 | 30 | 40 |
| `bloom_filter_size` | Bit array length | 512 | 1024 | 2048 |

Larger bloom filters and more hash functions increase privacy at the cost of matching precision.

---

## Security levels

### Standard

Basic bloom filter encoding. Suitable for internal use across trusted departments.

```python
result = gm.pprl_link("a.csv", "b.csv", security_level="standard")
```

### High (default)

HMAC salting with per-field keys. Prevents frequency analysis attacks.

```python
result = gm.pprl_link("a.csv", "b.csv", security_level="high")
```

### Paranoid

HMAC salting + balanced padding. Padding equalizes bloom filter density to prevent inference from bit population counts.

```python
result = gm.pprl_link("a.csv", "b.csv", security_level="paranoid")
```

---

## Protocols

### Trusted Third Party (TTP)

Both parties send encoded vectors to a trusted intermediary who performs the matching.

```python
from goldenmatch.pprl.protocol import PPRLConfig, link_trusted_third_party

config = PPRLConfig(fields=["name", "dob", "zip"], threshold=0.85)
result = link_trusted_third_party(party_a_data, party_b_data, config)
```

### Secure Multi-Party Computation (SMC)

No trusted intermediary required. Parties exchange encrypted similarity computations.

```python
from goldenmatch.pprl.protocol import link_smc

result = link_smc(party_a_data, party_b_data, config)
```

---

## Auto-configuration

`pprl_auto_config` profiles your data and selects optimal fields, bloom filter parameters, and threshold.

```python
import goldenmatch as gm

config = gm.pprl_auto_config(df)
print(config.recommended_fields)     # ['first_name', 'last_name', 'zip_code', 'birth_year']
print(config.recommended_config)     # PPRLConfig with optimal parameters
```

Auto-config heuristics:
- Penalizes near-unique fields (IDs) -- they leak information
- Penalizes long fields (>15 chars) -- more bits needed
- Penalizes high-null fields -- reduce match quality
- Limits to 4 fields (beats 6 in benchmarks)
- Minimum threshold 0.85

CLI:

```bash
goldenmatch pprl auto-config data.csv
```

---

## PPRLConfig

```python
gm.PPRLConfig(
    fields: list[str],
    threshold: float = 0.85,
    security_level: str = "high",
    ngram_size: int = 2,
    hash_functions: int = 30,
    bloom_filter_size: int = 1024,
    protocol: str = "trusted_third_party",
)
```

---

## Low-level API

```python
from goldenmatch.pprl.protocol import run_pprl, compute_bloom_filters, PartyData, LinkageResult

# Compute bloom filters manually
bf_a = compute_bloom_filters(df_a, fields, config)
bf_b = compute_bloom_filters(df_b, fields, config)

# Run matching
result: LinkageResult = run_pprl(df_a, df_b, config)
print(result.clusters)
print(result.match_count)
print(result.total_comparisons)
```

Vectorized similarity uses numpy matrix multiply (`mat_a @ mat_b.T`) for bloom filter Dice -- 13x faster than per-pair Python loops.

---

## PPRL in YAML config

Use the `bloom_filter` transform and `dice`/`jaccard` scorer:

```yaml
matchkeys:
  - name: pprl_match
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        transforms: [lowercase, strip, "bloom_filter:2:30:1024"]
        scorer: dice
        weight: 0.3
      - field: last_name
        transforms: [lowercase, strip, "bloom_filter:2:30:1024"]
        scorer: dice
        weight: 0.4
      - field: zip
        transforms: ["bloom_filter:2:30:1024"]
        scorer: dice
        weight: 0.3
```

---

## Benchmarks

### FEBRL4 (5K vs 5K synthetic person records)

| Strategy | Precision | Recall | F1 | Privacy |
|----------|-----------|--------|-----|---------|
| Normal fuzzy (baseline) | 56.5% | 74.6% | 64.3% | None |
| PPRL manual tuning | 98.2% | 82.6% | 89.8% | Per-field HMAC |
| **PPRL auto-config** | **99.7%** | **86.1%** | **92.4%** | Per-field HMAC |
| PPRL paranoid | 98.9% | 76.0% | 86.0% | HMAC + balanced |

### NCVR (North Carolina Voter Registration)

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| **PPRL auto-config** | **64.0%** | **93.8%** | **76.1%** |

Auto-configuration beats manual tuning on both datasets. Zero-config PPRL profiles your data and picks optimal parameters automatically.

---

## MCP tools

The MCP server exposes PPRL tools for Claude Desktop:

| Tool | Description |
|------|-------------|
| `pprl_auto_config` | Analyze data and recommend PPRL parameters |
| `pprl_link` | Run privacy-preserving linkage |
