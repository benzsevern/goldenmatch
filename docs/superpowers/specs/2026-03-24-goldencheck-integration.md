# GoldenCheck Integration — Design Spec

## Goal

When GoldenCheck is installed, GoldenMatch uses it for enhanced data quality scanning and fixing before matching. Zero config required — just `pip install goldenmatch[quality]`.

## How It Works

**Before (autofix only):**
```
data.csv → autofix (7 steps) → validate → standardize → match
```

**After (with GoldenCheck):**
```
data.csv → GoldenCheck scan + fix → autofix (remaining) → validate → standardize → match
```

GoldenCheck runs first (encoding, Unicode, smart quotes, digits-in-names, format detection), then the existing autofix handles GoldenMatch-specific cleaning (null normalization, BOM, empty rows). They complement each other.

## Config

```yaml
# goldenmatch config
quality:
  enabled: true        # default: true if goldencheck installed, false otherwise
  mode: "announced"    # "silent" | "announced" | "disabled"
  fix_mode: "safe"     # "safe" | "moderate" | "none"
  domain: null         # "healthcare" | "finance" | "ecommerce" | null
```

Default behavior: if `goldencheck` is importable and `quality.mode` is not `"disabled"`, run it. Users who don't have GoldenCheck installed see no change.

## User Experience

```
$ goldenmatch dedupe patients.csv

GoldenCheck: scanning data quality... 5 issues found, 3 auto-fixed (whitespace, encoding, smart quotes)
Configuring match pipeline...
...
```

One line of output (in `announced` mode), then normal GoldenMatch flow. `silent` mode suppresses the line. `disabled` skips entirely.

## Implementation

### New file: `goldenmatch/core/quality.py`

```python
def run_quality_check(
    df: pl.DataFrame,
    config: QualityConfig | None = None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Run GoldenCheck scan + fix if available. Returns (df, fixes_applied).

    Falls back gracefully if goldencheck is not installed.
    """
```

**Logic:**
1. Try `from goldencheck import scan_file` — if ImportError, return `(df, [])`
2. Write df to temp CSV, run `scan_file()` on it
3. Apply `apply_fixes(df, findings, mode=config.fix_mode)`
4. Build fixes list matching autofix format: `{"fix": ..., "column": ..., "rows_affected": ...}`
5. If `config.mode == "announced"`, print one-line summary
6. Return `(fixed_df, fixes)`

### Config addition: `schemas.py`

```python
class QualityConfig(BaseModel):
    enabled: bool = True
    mode: str = "announced"  # silent | announced | disabled
    fix_mode: str = "safe"   # safe | moderate | none
    domain: str | None = None
```

Add to `GoldenMatchConfig`:
```python
quality: QualityConfig | None = None
```

### Pipeline integration: `pipeline.py`

In `_run_dedupe_pipeline()`, before the existing autofix call (line ~197):

```python
# Step 1.4 — GoldenCheck quality scan (if available)
if config.quality is None or config.quality.mode != "disabled":
    from goldenmatch.core.quality import run_quality_check
    df, gc_fixes = run_quality_check(df, config.quality)
    all_fixes.extend(gc_fixes)
```

The existing autofix (step 1.5a) still runs after — it handles GoldenMatch-specific cleaning that GoldenCheck doesn't cover (null string normalization, empty row dropping).

### Optional dependency: `pyproject.toml`

```toml
[project.optional-dependencies]
quality = ["goldencheck>=0.5.0"]
all = ["goldenmatch[embeddings,llm,postgres,mcp,quality]"]
```

## What GoldenCheck Catches That Autofix Doesn't

| Issue | autofix.py | GoldenCheck |
|-------|------------|-------------|
| BOM characters | Yes | Yes |
| Whitespace trim | Yes | Yes |
| Null normalization | Yes | No (different approach) |
| Empty rows/cols | Yes | No |
| Control chars | Yes | Yes |
| Unicode NFC normalization | No | Yes |
| Smart quotes → straight | No | Yes |
| Zero-width characters | No | Yes |
| Encoding detection (Latin-1) | No | Yes |
| Format validation (email/phone) | No | Yes (findings, not fixes) |
| Type inference warnings | No | Yes |
| Confidence scoring | No | Yes |
| Domain-specific types | No | Yes (--domain) |

## Testing

- Unit test: `run_quality_check` with goldencheck installed → returns fixed df
- Unit test: `run_quality_check` without goldencheck → returns original df unchanged
- Integration test: `run_dedupe` with quality config → GoldenCheck runs before autofix
- Config test: `mode="disabled"` skips GoldenCheck entirely

## Non-Goals

- GoldenCheck does not replace autofix — they run in sequence
- No GoldenCheck findings in the TUI (future enhancement)
- No goldencheck.yml generation from GoldenMatch (use `goldencheck init` separately)
- No blocking strategy recommendations (future)
