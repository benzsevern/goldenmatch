# Scoring & Survivorship Quality Upgrade

**Date:** 2026-04-04
**Status:** Approved
**Scope:** Cluster quality handling, advanced survivorship with data quality weighting, field-level provenance in lineage

---

## Problem

GoldenMatch produces golden records but lacks transparency and quality signals at critical stages:

- Oversized clusters are detected but not acted on
- All source records are treated equally regardless of data quality
- Golden record field provenance (which source, which strategy) is not tracked
- Cluster confidence doesn't reflect structural weakness

The design summary requires GoldenMatch to be autonomous — human review is exception handling, not part of the pipeline. These changes make the engine self-correcting and self-documenting.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pair-level confidence | Score IS confidence | Adding a separate derived number creates redundancy |
| Cluster remediation | Split + downgrade | Autonomous (no quarantine/human review) |
| GoldenCheck quality weighting | Optional-but-rewarded | Required would be a breaking change |
| Field provenance location | Lineage JSON sidecar | Follows existing pattern, avoids bloating output DataFrame |

---

## 1. Pair-Level Confidence

The existing match score (0.0-1.0) is the pair confidence. No new field.

**Changes:** Documentation only. Update CLAUDE.md and `MatchResult` docstring to state that the score IS the confidence.

---

## 2. Cluster Quality: Split + Downgrade

### Auto-Split

When a cluster exceeds `max_cluster_size` (default 100):

1. Build the minimum spanning tree (MST) of the cluster's pair graph using stored `pair_scores`
2. Remove the weakest MST edge (guaranteed to disconnect the tree into two components)
3. Re-run Union-Find on remaining MST edges to produce two sub-clusters
4. Re-compute confidence on each sub-cluster
5. If either sub-cluster is still oversized, recurse

Using MST guarantees disconnection — removing the weakest edge from an arbitrary graph may not split it if the graph is well-connected.

**File:** `goldenmatch/core/cluster.py`
**Functions:** Extend `build_clusters()` and `add_to_cluster()` (which currently hardcodes `max_cluster_size=100` — must accept the configurable threshold)

### Confidence Downgrade

After clustering, identify clusters where the weakest edge is more than `weak_cluster_threshold` (default 0.3) below the cluster's average edge weight. Downgrade their `cluster_confidence` by multiplying by 0.7.

This doesn't change cluster membership. It signals "this cluster held together but barely."

### Cluster Quality Field

Add `cluster_quality` string field to each cluster dict:

- `"strong"` — no issues detected
- `"weak"` — confidence downgraded due to edge gap
- `"split"` — was oversized, automatically split (takes precedence over `"weak"` — a split sub-cluster that also has a large edge gap is labeled `"split"`, not `"weak"`)

**File:** `goldenmatch/core/cluster.py`

### Testing

- Test oversized cluster is split into two valid sub-clusters
- Test recursive splitting for very large clusters
- Test confidence downgrade fires when edge gap exceeds threshold
- Test strong clusters are untouched
- Test `cluster_quality` field values

---

## 3. Advanced Survivorship with Data Quality Weighting

### Quality Scores Parameter

Add optional `quality_scores: dict[tuple[int, str], float] | None` parameter to `build_golden_record()`. Maps `(row_id, column_name)` to a 0.0-1.0 quality score.

`build_golden_record()` extracts row IDs from `cluster_df["__row_id__"]` and passes per-value quality weights to each `merge_field()` call. `merge_field()` signature gains an optional `quality_weights: list[float] | None` parameter aligned 1:1 with the `values` list.

When `quality_scores` is None, all strategies behave identically to current behavior. No breaking change.

**File:** `goldenmatch/core/golden.py`

### Strategy Enhancements

| Strategy | Current | With Quality Scores |
|----------|---------|-------------------|
| `most_complete` | Longest value wins; ties get 0.7 confidence | Ties broken by quality score; confidence = `base * quality_weight` |
| `majority_vote` | Raw count | Votes weighted by source quality |
| `source_priority` | Static ordered list | User's source ordering preserved; quality score used as tiebreaker within same priority level and as confidence modifier |
| `most_recent` | Newest timestamp; confidence 1.0 or 0.5 | Recency wins; confidence scales by source quality |
| `first_non_null` | First non-null; fixed 0.6 confidence | Pick from highest-quality source first |

### Quality Score Source

If GoldenCheck ran (via `quality.py` integration), per-column quality metrics (null rate, format consistency, pattern violations) are normalized to 0-1 per `(row_id, column)`. This normalization happens in `pipeline.py` after the GoldenCheck step, before golden record generation.

If GoldenCheck did not run, `quality_scores` is None. No degradation.

### Confidence Computation

Replace hardcoded confidence values with data-driven ones. For example, `most_complete` confidence becomes `base_confidence * mean(quality_weights_for_candidates)` instead of a flat 0.7.

### Testing

- Test `build_golden_record(quality_scores=None)` produces identical DataFrame values, dtypes, and `__golden_confidence__` (to 4 decimal places) as v1.3.2
- Test quality tiebreaking for each of the 5 strategies
- Test confidence values change with quality scores present
- Test with partial quality scores (some fields have scores, others don't)
- Test `build_golden_record_with_provenance()` returns `GoldenRecordResult` with `.df` matching `build_golden_record()` output

---

## 4. Field-Level Provenance in Lineage

### Provenance via opt-in parameter (no breaking change)

`build_golden_record()` keeps its current return type (dict). A new optional `provenance: bool = False` parameter controls whether provenance is collected. When `provenance=True`, a second return value is added via an overloaded helper: `build_golden_record_with_provenance()` returns `GoldenRecordResult`. Only `pipeline.py` calls the provenance variant — all other call sites (`engine.py`, `sync.py`, `reconcile.py`, `chunked.py`, `bench_1m.py`, `run_scale_curve.py`) continue calling the existing function unchanged.

Provenance dataclasses:

```python
@dataclass
class FieldProvenance:
    value: Any
    source_row_id: int
    strategy: str
    confidence: float
    candidates: list[dict]  # [{row_id, value, quality}]

@dataclass
class ClusterProvenance:
    cluster_id: int
    cluster_quality: str
    cluster_confidence: float
    fields: dict[str, FieldProvenance]

@dataclass
class GoldenRecordResult:
    df: pl.DataFrame  # Golden records (unchanged format)
    provenance: list[ClusterProvenance]
```

**File:** `goldenmatch/core/golden.py`

### Lineage Extension

`build_lineage()` includes a `golden_records` section when golden records were generated:

```json
{
  "pairs": [...],
  "golden_records": [
    {
      "cluster_id": 0,
      "cluster_quality": "strong",
      "cluster_confidence": 0.92,
      "fields": {
        "email": {
          "value": "john@example.com",
          "source_row_id": 42,
          "strategy": "majority_vote",
          "confidence": 0.85,
          "candidates": [
            {"row_id": 42, "value": "john@example.com", "quality": 0.95},
            {"row_id": 87, "value": "john@example.com", "quality": 0.80}
          ]
        }
      }
    }
  ]
}
```

**File:** `goldenmatch/core/lineage.py`

### Streaming Support

Extend `save_lineage_streaming()` to append golden record provenance incrementally.

**File:** `goldenmatch/core/lineage.py`

### Testing

- Test provenance structure matches schema for every field
- Test every field in golden record has a corresponding provenance entry
- Test streaming lineage includes golden records
- Test lineage without golden records (match pipeline, not dedupe) still works

---

## 5. Integration Points

### Pipeline Wiring

Two touch points in `pipeline.py`:

1. **After GoldenCheck** (if it ran): Extract per-column quality metrics from the GoldenCheck profile, normalize to 0-1 per `(row_id, column)`, store as `quality_scores` dict
2. **After clustering, before golden record generation:** Pass `cluster_quality` from split+downgrade into golden record generation and lineage

**File:** `goldenmatch/core/pipeline.py`

### Config Changes

Add to `GoldenRulesConfig` in `schemas.py`:

```python
auto_split: bool = True          # Automatic oversized cluster splitting
quality_weighting: bool = True   # Use GoldenCheck quality in survivorship
weak_cluster_threshold: float = 0.3  # Edge gap threshold for downgrade
```

**File:** `goldenmatch/config/schemas.py`

### No New Dependencies

Everything builds on existing modules. GoldenCheck remains an optional dependency.

### Backward Compatibility

All new behavior is additive:

- `build_golden_record()` signature and return type unchanged — new `quality_scores` parameter defaults to None
- `build_golden_record_with_provenance()` is a new function, not a replacement — only called from `pipeline.py`
- `build_clusters()` auto-splits oversized clusters (new behavior), but clusters that were already under `max_cluster_size` are untouched
- Default config with no GoldenCheck produces identical golden record values and confidence scores to v1.3.2
- "Identical" means: `GoldenRecordResult.df` has the same column values and Polars dtypes, and `__golden_confidence__` values match to 4 decimal places

---

## Files Modified

| File | Changes |
|------|---------|
| `core/cluster.py` | Auto-split via MST, confidence downgrade, `cluster_quality` field, `add_to_cluster()` accepts configurable `max_cluster_size` |
| `core/golden.py` | `quality_scores` parameter on `build_golden_record()`, enhanced merge strategies, new `build_golden_record_with_provenance()` function, provenance dataclasses |
| `core/lineage.py` | `golden_records` section, streaming provenance |
| `core/pipeline.py` | Quality score extraction from GoldenCheck, call `build_golden_record_with_provenance()`, pass cluster_quality to lineage |
| `core/streaming.py` | Pass configurable `max_cluster_size` to `add_to_cluster()` |
| `config/schemas.py` | `auto_split`, `quality_weighting`, `weak_cluster_threshold` config fields |
| `CLAUDE.md` | Document pair confidence = score, new config fields |

**Not modified** (no breaking changes to their call signatures): `tui/engine.py`, `db/sync.py`, `db/reconcile.py`, `core/chunked.py`, `__init__.py` — these continue calling `build_golden_record()` with `quality_scores=None` (default).

## Files Created

None. All changes extend existing modules.

## Estimated Test Count

~25 new tests across `test_cluster.py`, `test_golden.py`, `test_lineage.py`, and `test_pipeline_integration.py`.
