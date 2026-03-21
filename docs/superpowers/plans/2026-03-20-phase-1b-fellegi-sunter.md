# Phase 1B: Fellegi-Sunter Probabilistic Model — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fellegi-Sunter probabilistic matching as a third matchkey type (`probabilistic`) with EM-trained parameters and automatic threshold estimation.

**Architecture:** New `probabilistic.py` module handles EM training and scoring. Integrates as a third branch in the pipeline's matchkey loop (alongside `exact` and `weighted`). Comparison vectors reuse existing field scorers. Match weights are log-likelihood ratios normalized to 0-1 for compatibility with the existing pair format.

**Tech Stack:** Python 3.12, Polars, NumPy, existing scorer infrastructure.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `goldenmatch/core/probabilistic.py` (CREATE) | EM training, comparison vectors, match-weight scoring, threshold estimation |
| `goldenmatch/config/schemas.py` (MODIFY) | Extend MatchkeyConfig type Literal, add EM fields, update validator |
| `goldenmatch/core/pipeline.py` (MODIFY) | Add probabilistic branch in matchkey loop |
| `goldenmatch/core/scorer.py` (MODIFY) | Add `score_pair_probabilistic` for single-pair use by match_one |
| `goldenmatch/tui/engine.py` (MODIFY) | Add probabilistic branch in engine pipeline |
| `tests/test_probabilistic.py` (CREATE) | Unit + integration tests |

---

### Task 1: Schema Changes

- [ ] Write failing tests for probabilistic matchkey config
- [ ] Extend MatchkeyConfig type Literal to include "probabilistic"
- [ ] Add EM fields to MatchkeyConfig (em_iterations, convergence_threshold, link_threshold, review_threshold)
- [ ] Add comparison level fields to MatchkeyField (levels, partial_threshold)
- [ ] Update _validate_weighted to handle probabilistic type
- [ ] Run tests, commit

### Task 2: EM Algorithm Core

- [ ] Write failing tests for comparison vector generation
- [ ] Write failing tests for EM convergence
- [ ] Implement comparison_vector() using existing score_field
- [ ] Implement train_em() with Expectation-Maximization
- [ ] Implement compute_thresholds() for link/review boundaries
- [ ] Run tests, commit

### Task 3: Probabilistic Scoring

- [ ] Write failing tests for score_probabilistic on block DataFrames
- [ ] Implement score_probabilistic() returning standard pair format
- [ ] Implement normalize_match_weight() for 0-1 scaling
- [ ] Run tests, commit

### Task 4: Pipeline Integration

- [ ] Write failing integration test for full pipeline with probabilistic matchkey
- [ ] Add third branch in pipeline.py matchkey loop
- [ ] Add probabilistic branch in engine.py
- [ ] Run full test suite, commit

### Task 5: Single-Pair Scoring for match_one

- [ ] Write test for score_pair_probabilistic
- [ ] Implement in scorer.py for match_one compatibility
- [ ] Run full test suite, commit
