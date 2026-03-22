# Domain-Aware Feature Extraction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan.

**Goal:** Shift LLM usage from O(N^2) pair comparison to O(N) preprocessing. Extract structured features (brand, model, specs) from noisy product titles before matching, using heuristics for the easy 95% and LLM for the hard 5%.

**Architecture:** New preprocessing step between auto_fix and matchkeys. Domain detection extends autoconfig. Feature extraction is a new transform type. LLM validation reuses BudgetTracker.

```
ingest -> auto_fix -> domain_detect -> feature_extract -> [llm_validate] -> matchkeys -> block -> score -> ...
```

---

## File Structure

| File | Responsibility |
|------|---------------|
| `goldenmatch/core/domain.py` (CREATE) | Domain detection, feature extraction rules, extraction confidence scoring |
| `goldenmatch/core/llm_extract.py` (CREATE) | LLM-based feature extraction for low-confidence records |
| `goldenmatch/config/schemas.py` (MODIFY) | Add DomainConfig to GoldenMatchConfig |
| `goldenmatch/core/pipeline.py` (MODIFY) | Insert domain extraction step |
| `goldenmatch/core/autoconfig.py` (MODIFY) | Auto-detect domain mode |
| `tests/test_domain.py` (CREATE) | Tests |

---

### Task 1: Domain Detection + Feature Extraction Rules

**Files:** Create `goldenmatch/core/domain.py`

- Domain profiles: product, person, bibliographic, company
- Per-domain extraction rules (regex patterns for model numbers, SKUs, brands, specs)
- Confidence scoring per extraction
- Output: new derived columns added to DataFrame

### Task 2: LLM Validation for Low-Confidence Extractions

**Files:** Create `goldenmatch/core/llm_extract.py`

- Route bottom N% by confidence to LLM
- Prompt: "Extract brand, model, and key specs from: [title]"
- Reuse BudgetTracker for cost controls
- Merge LLM extractions back into DataFrame

### Task 3: Schema + Pipeline Integration

**Files:** Modify schemas.py, pipeline.py, autoconfig.py

- DomainConfig in GoldenMatchConfig
- Pipeline inserts extraction between auto_fix and matchkeys
- Autoconfig detects domain and enables extraction automatically

### Task 4: Matching on Extracted Features

- Auto-generate exact matchkeys on extracted columns (_brand, _model, _sku)
- Weight extracted features heavily in weighted matchkeys
- Combine with existing fuzzy scoring on original fields
