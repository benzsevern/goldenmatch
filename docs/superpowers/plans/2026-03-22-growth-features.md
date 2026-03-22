# GoldenMatch Growth Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 6 features that grow GoldenMatch's reach: domain packs, evaluation CLI, GitHub Actions try-it workflow, Codespaces devcontainer, incremental watch mode CLI, and dbt integration.

**Architecture:** Each feature is independently shippable. Domain packs and eval CLI are pure additions. GitHub Actions and Codespaces are infrastructure files. Incremental mode wraps existing StreamProcessor + watch_daemon. dbt integration is a separate package using DuckDB backend.

**Tech Stack:** Python 3.12, Polars, Typer CLI, GitHub Actions, devcontainers, dbt-core

---

## File Structure

### New Files
- `goldenmatch/domains/healthcare.yaml` — Healthcare domain rulebook
- `goldenmatch/domains/financial.yaml` — Financial domain rulebook
- `goldenmatch/domains/real_estate.yaml` — Real estate domain rulebook
- `goldenmatch/domains/people.yaml` — People/person domain rulebook
- `goldenmatch/domains/retail.yaml` — Retail/CPG domain rulebook
- `goldenmatch/core/evaluate.py` — Evaluation engine (precision/recall/F1 from ground truth)
- `goldenmatch/cli/evaluate.py` — CLI command for `goldenmatch evaluate`
- `tests/test_evaluate.py` — Tests for evaluation engine + CLI
- `tests/test_domain_packs.py` — Tests for all domain packs load and extract correctly
- `.github/workflows/try-it.yml` — GitHub Actions "Try It" workflow
- `.devcontainer/devcontainer.json` — Codespaces config
- `.devcontainer/post-create.sh` — Post-create setup script
- `goldenmatch/cli/incremental.py` — CLI command for `goldenmatch incremental`
- `tests/test_cli_incremental.py` — Tests for incremental CLI
- `dbt-goldenmatch/` — Separate package (out of scope for main test suite)

### Modified Files
- `goldenmatch/cli/main.py` — Register `evaluate` and `incremental` commands

---

## Task 1: Pre-built Domain Packs

**Files:**
- Create: `goldenmatch/domains/healthcare.yaml`
- Create: `goldenmatch/domains/financial.yaml`
- Create: `goldenmatch/domains/real_estate.yaml`
- Create: `goldenmatch/domains/people.yaml`
- Create: `goldenmatch/domains/retail.yaml`
- Create: `tests/test_domain_packs.py`

- [ ] **Step 1: Write tests for domain pack loading and extraction**

```python
"""Tests for pre-built domain packs."""
from __future__ import annotations

import pytest
from goldenmatch.core.domain_registry import discover_rulebooks, load_rulebook
from pathlib import Path

DOMAINS_DIR = Path(__file__).parent.parent / "goldenmatch" / "domains"

EXPECTED_PACKS = ["electronics", "software", "healthcare", "financial", "real_estate", "people", "retail"]


class TestDomainPacksDiscovery:
    def test_all_packs_discovered(self):
        rulebooks = discover_rulebooks()
        for name in EXPECTED_PACKS:
            assert name in rulebooks, f"Missing domain pack: {name}"

    def test_all_yamls_exist(self):
        for name in EXPECTED_PACKS:
            path = DOMAINS_DIR / f"{name}.yaml"
            assert path.exists(), f"Missing YAML: {path}"


class TestHealthcarePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "healthcare.yaml")
        rb.compile()
        return rb

    def test_ndc_extraction(self, rb):
        result = rb.extract("Medtronic Catheter NDC 12345-6789-01 sterile")
        assert result["identifiers"].get("ndc")
        assert result["brand"] == "Medtronic"

    def test_signals(self, rb):
        assert "ndc" in rb.signals or "patient" in rb.signals


class TestFinancialPack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "financial.yaml")
        rb.compile()
        return rb

    def test_cusip_extraction(self, rb):
        result = rb.extract("Goldman Sachs Bond CUSIP: 38141G104")
        assert result["identifiers"].get("cusip")

    def test_lei_extraction(self, rb):
        result = rb.extract("Entity LEI: 5493001KJTIIGC8Y1R12")
        assert result["identifiers"].get("lei")


class TestRealEstatePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "real_estate.yaml")
        rb.compile()
        return rb

    def test_sqft_extraction(self, rb):
        result = rb.extract("3 bed 2 bath 1500 sqft ranch home")
        assert result["attributes"].get("sqft")

    def test_zip_extraction(self, rb):
        result = rb.extract("123 Main St Springfield IL 62704")
        assert result["identifiers"].get("zip")


class TestPeoplePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "people.yaml")
        rb.compile()
        return rb

    def test_ssn_extraction(self, rb):
        result = rb.extract("John Smith SSN 123-45-6789")
        assert result["identifiers"].get("ssn")

    def test_dob_extraction(self, rb):
        result = rb.extract("Jane Doe DOB 01/15/1990")
        assert result["identifiers"].get("dob")


class TestRetailPack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "retail.yaml")
        rb.compile()
        return rb

    def test_upc_extraction(self, rb):
        result = rb.extract("Tide Detergent UPC 037000127864 64oz")
        assert result["identifiers"].get("upc")

    def test_brand(self, rb):
        result = rb.extract("Procter & Gamble Tide Original")
        assert result["brand"] in ("Procter & Gamble", "Tide")
```

Run: `pytest tests/test_domain_packs.py -v`
Expected: FAIL — healthcare.yaml etc. don't exist yet

- [ ] **Step 2: Create healthcare.yaml**

```yaml
name: healthcare
signals: ["ndc", "patient", "diagnosis", "icd", "cpt", "npi", "fda", "implant", "sterile", "dosage", "mg", "ml", "prescription"]
identifier_patterns:
  ndc: '\b(\d{5}-\d{4}-\d{2})\b'
  npi: '(?:NPI\s*:?\s*)\b(\d{10})\b'
  icd10: '\b([A-Z]\d{2}(?:\.\d{1,4})?)\b'
  cpt: '(?:CPT\s*:?\s*)\b(\d{5})\b'
brand_patterns:
  - Medtronic
  - Abbott
  - Johnson & Johnson
  - Baxter
  - Boston Scientific
  - Stryker
  - Becton Dickinson
  - Zimmer Biomet
  - Edwards Lifesciences
  - Cardinal Health
  - McKesson
  - AmerisourceBergen
  - Pfizer
  - Merck
  - Roche
  - Novartis
  - AstraZeneca
  - Bristol-Myers Squibb
  - Eli Lilly
  - Amgen
attribute_patterns:
  dosage_mg: '(\d+(?:\.\d+)?)\s*mg\b'
  dosage_ml: '(\d+(?:\.\d+)?)\s*ml\b'
  gauge: '(\d+)\s*(?:ga|gauge)\b'
  length_mm: '(\d+(?:\.\d+)?)\s*mm\b'
  count: '(\d+)\s*(?:count|ct|pack|pk)\b'
stop_words:
  - the
  - a
  - an
  - for
  - and
  - with
  - sterile
  - disposable
  - single
  - use
  - each
  - per
  - box
  - case
  - unit
  - dose
  - tablet
  - capsule
  - injection
  - solution
  - suspension
normalization:
  lowercase: true
  strip_punctuation: true
```

- [ ] **Step 3: Create financial.yaml**

```yaml
name: financial
signals: ["cusip", "isin", "lei", "ticker", "sedol", "figi", "fund", "bond", "equity", "portfolio", "aum", "nav"]
identifier_patterns:
  cusip: '(?:CUSIP\s*:?\s*)\b([A-Z0-9]{9})\b'
  isin: '(?:ISIN\s*:?\s*)\b([A-Z]{2}[A-Z0-9]{10})\b'
  lei: '(?:LEI\s*:?\s*)\b([A-Z0-9]{20})\b'
  sedol: '(?:SEDOL\s*:?\s*)\b([A-Z0-9]{7})\b'
  ticker: '(?:ticker|symbol|stock)\s*:?\s*\b([A-Z]{1,5})\b'
  figi: '\b(BBG[A-Z0-9]{9})\b'
brand_patterns:
  - Goldman Sachs
  - JPMorgan
  - Morgan Stanley
  - BlackRock
  - Vanguard
  - Fidelity
  - Charles Schwab
  - State Street
  - BNY Mellon
  - Citigroup
  - Bank of America
  - Wells Fargo
  - Deutsche Bank
  - UBS
  - Credit Suisse
  - Barclays
  - HSBC
  - BNP Paribas
  - Nomura
  - Bridgewater
attribute_patterns:
  currency: '\b(USD|EUR|GBP|JPY|CHF|CAD|AUD)\b'
  maturity_year: '\b(20[2-9]\d)\b'
  coupon_rate: '(\d+(?:\.\d+)?)\s*%'
  shares: '(\d{1,3}(?:,\d{3})*)\s*(?:shares|sh)\b'
stop_words:
  - the
  - a
  - an
  - inc
  - llc
  - corp
  - ltd
  - plc
  - sa
  - ag
  - co
  - group
  - holdings
  - capital
  - management
  - fund
  - trust
  - class
  - series
normalization:
  uppercase: true
  strip_punctuation: true
```

- [ ] **Step 4: Create real_estate.yaml**

```yaml
name: real_estate
signals: ["address", "parcel", "mls", "sqft", "acre", "bed", "bath", "lot", "zoning", "apn", "deed", "listing"]
identifier_patterns:
  zip: '\b(\d{5}(?:-\d{4})?)\b'
  apn: '\b(\d{3}-\d{3}-\d{3}(?:-\d{3})?)\b'
  mls: '\b(?:MLS\s*#?\s*)(\w{6,12})\b'
brand_patterns:
  - Keller Williams
  - RE/MAX
  - Coldwell Banker
  - Century 21
  - Berkshire Hathaway
  - Sotheby's
  - Compass
  - eXp Realty
  - Redfin
  - Zillow
attribute_patterns:
  sqft: '(\d{1,3}(?:,\d{3})*)\s*(?:sq\s*ft|sqft|sf)\b'
  acres: '(\d+(?:\.\d+)?)\s*(?:acre|ac)\b'
  bedrooms: '(\d+)\s*(?:bed|br|bedroom)\b'
  bathrooms: '(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)\b'
  year_built: '\b((?:19|20)\d{2})\b'
  price: '\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
stop_words:
  - the
  - a
  - an
  - for
  - and
  - with
  - in
  - at
  - on
  - street
  - st
  - avenue
  - ave
  - road
  - rd
  - drive
  - dr
  - lane
  - ln
  - boulevard
  - blvd
  - suite
  - ste
  - unit
  - apt
  - apartment
normalization:
  uppercase: true
  strip_punctuation: false
```

- [ ] **Step 5: Create people.yaml**

```yaml
name: people
signals: ["name", "first_name", "last_name", "dob", "ssn", "email", "phone", "address", "gender", "age"]
identifier_patterns:
  ssn: '\b(\d{3}-\d{2}-\d{4})\b'
  dob: '\b(\d{1,2}/\d{1,2}/\d{2,4})\b'
  phone: '\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'
  email: '\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
  drivers_license: '\b([A-Z]\d{7,12})\b'
brand_patterns: []
attribute_patterns:
  age: '\b(\d{1,3})\s*(?:years?\s*old|yo|y\.o\.)\b'
  zip: '\b(\d{5}(?:-\d{4})?)\b'
stop_words:
  - mr
  - mrs
  - ms
  - dr
  - jr
  - sr
  - ii
  - iii
  - iv
normalization:
  lowercase: true
  strip_punctuation: true
```

- [ ] **Step 6: Create retail.yaml**

```yaml
name: retail
signals: ["upc", "ean", "gtin", "sku", "asin", "brand", "price", "size", "flavor", "color", "pack", "oz", "lb"]
identifier_patterns:
  upc: '\b(\d{12})\b'
  ean: '\b(\d{13})\b'
  gtin: '\b(\d{14})\b'
  asin: '\b([A-Z0-9]{10})\b'
  sku: '\b([A-Z]{2,}\d{4,})\b'
brand_patterns:
  - Procter & Gamble
  - Unilever
  - Nestle
  - PepsiCo
  - Coca-Cola
  - Kraft Heinz
  - General Mills
  - Kellogg
  - Mars
  - Colgate-Palmolive
  - Johnson & Johnson
  - Henkel
  - Reckitt
  - Church & Dwight
  - Tide
  - Clorox
  - SC Johnson
  - Dial
  - Arm & Hammer
  - Bounty
attribute_patterns:
  oz: '(\d+(?:\.\d+)?)\s*(?:fl\s*)?oz\b'
  lb: '(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound)\b'
  count: '(\d+)\s*(?:count|ct|pk|pack)\b'
  size: '\b(small|medium|large|xl|xxl|s|m|l)\b'
stop_words:
  - the
  - a
  - an
  - for
  - and
  - with
  - new
  - improved
  - original
  - classic
  - value
  - pack
  - size
  - family
normalization:
  lowercase: true
  strip_punctuation: true
```

- [ ] **Step 7: Run tests to verify all packs load and extract**

Run: `pytest tests/test_domain_packs.py -v`
Expected: PASS — all packs discovered, all extraction tests pass

- [ ] **Step 8: Commit**

```bash
git add goldenmatch/domains/*.yaml tests/test_domain_packs.py
git commit -m "feat: add 5 pre-built domain packs (healthcare, financial, real_estate, people, retail)"
```

---

## Task 2: Evaluation CLI

**Files:**
- Create: `goldenmatch/core/evaluate.py`
- Create: `goldenmatch/cli/evaluate.py`
- Create: `tests/test_evaluate.py`
- Modify: `goldenmatch/cli/main.py` — register evaluate command

- [ ] **Step 1: Write tests for the evaluation engine**

```python
"""Tests for evaluation engine."""
from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from goldenmatch.core.evaluate import evaluate_clusters, evaluate_pairs, EvalResult


class TestEvaluatePairs:
    def test_perfect_pairs(self):
        """All predicted pairs are in ground truth."""
        predicted = [(1, 2, 0.9), (3, 4, 0.85)]
        ground_truth = {(1, 2), (3, 4)}
        result = evaluate_pairs(predicted, ground_truth)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_partial_match(self):
        predicted = [(1, 2, 0.9), (5, 6, 0.8)]  # (5,6) is FP
        ground_truth = {(1, 2), (3, 4)}  # (3,4) is FN
        result = evaluate_pairs(predicted, ground_truth)
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 1
        assert result.precision == 0.5
        assert result.recall == 0.5

    def test_empty_predicted(self):
        result = evaluate_pairs([], {(1, 2)})
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.fn == 1

    def test_symmetric_pairs(self):
        """(1,2) should match (2,1) in ground truth."""
        predicted = [(2, 1, 0.9)]
        ground_truth = {(1, 2)}
        result = evaluate_pairs(predicted, ground_truth)
        assert result.tp == 1

    def test_empty_ground_truth(self):
        result = evaluate_pairs([(1, 2, 0.9)], set())
        assert result.precision == 0.0
        assert result.recall == 0.0


class TestEvaluateClusters:
    def test_cluster_to_pairs(self):
        """Clusters with >1 member generate pairs for evaluation."""
        clusters = {
            1: {"members": [1, 2, 3], "size": 3},
            2: {"members": [4], "size": 1},
        }
        ground_truth = {(1, 2), (1, 3), (2, 3)}
        result = evaluate_clusters(clusters, ground_truth)
        assert result.tp == 3
        assert result.precision == 1.0
        assert result.recall == 1.0


class TestEvalResult:
    def test_summary_dict(self):
        result = EvalResult(tp=8, fp=2, fn=1)
        d = result.summary()
        assert d["precision"] == pytest.approx(0.8)
        assert d["recall"] == pytest.approx(8/9)
        assert "f1" in d
```

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — evaluate module doesn't exist

- [ ] **Step 2: Implement the evaluation engine**

Create `goldenmatch/core/evaluate.py`:

```python
"""Evaluation engine — precision, recall, F1 from ground truth pairs."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass
class EvalResult:
    """Evaluation metrics container."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "predicted_pairs": self.tp + self.fp,
            "ground_truth_pairs": self.tp + self.fn,
        }


def evaluate_pairs(
    predicted: list[tuple[int, int, float]],
    ground_truth: set[tuple],
) -> EvalResult:
    """Evaluate predicted pairs against ground truth.

    Ground truth pairs are matched symmetrically: (a,b) matches (b,a).
    """
    # Normalize ground truth to canonical form (min, max)
    gt_canonical = set()
    for pair in ground_truth:
        a, b = pair[0], pair[1]
        gt_canonical.add((min(a, b), max(a, b)))

    tp = fp = 0
    seen = set()
    for a, b, _score in predicted:
        canon = (min(a, b), max(a, b))
        if canon in seen:
            continue
        seen.add(canon)
        if canon in gt_canonical:
            tp += 1
        else:
            fp += 1
    fn = len(gt_canonical) - tp
    return EvalResult(tp=tp, fp=fp, fn=fn)


def evaluate_clusters(
    clusters: dict[int, dict],
    ground_truth: set[tuple],
) -> EvalResult:
    """Evaluate clusters by expanding to pairwise comparisons."""
    predicted = []
    for cid, info in clusters.items():
        members = info.get("members", [])
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            predicted.append((a, b, 1.0))
    return evaluate_pairs(predicted, ground_truth)


def load_ground_truth_csv(path: str, col_a: str = "id_a", col_b: str = "id_b") -> set[tuple]:
    """Load ground truth pairs from CSV.

    Supports both ID-based (integer) and string-based pair columns.
    """
    import polars as pl
    df = pl.read_csv(path)
    if col_a not in df.columns or col_b not in df.columns:
        # Try common alternative column names
        for alt_a, alt_b in [("idA", "idB"), ("id1", "id2"), ("left_id", "right_id")]:
            if alt_a in df.columns and alt_b in df.columns:
                col_a, col_b = alt_a, alt_b
                break
        else:
            raise ValueError(
                f"Ground truth CSV must have columns '{col_a}' and '{col_b}'. "
                f"Found: {df.columns}"
            )
    pairs = set()
    for row in df.select(col_a, col_b).to_dicts():
        a, b = row[col_a], row[col_b]
        # Try integer conversion (row IDs are ints in GoldenMatch)
        try:
            a = int(a)
        except (ValueError, TypeError):
            a = str(a).strip()
        try:
            b = int(b)
        except (ValueError, TypeError):
            b = str(b).strip()
        pairs.add((a, b))
    return pairs
```

- [ ] **Step 3: Run evaluation engine tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 4: Write tests for the evaluate CLI command**

Add to `tests/test_evaluate.py`:

```python
from goldenmatch.cli.main import app

runner = CliRunner()


class TestEvaluateCLI:
    @pytest.fixture
    def sample_data(self, tmp_path):
        # Create input CSV
        data_path = tmp_path / "data.csv"
        pl.DataFrame({
            "first_name": ["John", "john", "Jane", "Bob"],
            "last_name": ["Smith", "Smith", "Doe", "Jones"],
            "email": ["j@x.com", "j@x.com", "jane@t.com", "bob@t.com"],
        }).write_csv(data_path)

        # Create ground truth CSV
        gt_path = tmp_path / "ground_truth.csv"
        pl.DataFrame({"id_a": [0], "id_b": [1]}).write_csv(gt_path)

        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent("""\
            matchkeys:
              - name: exact_email
                type: exact
                fields:
                  - field: email
                    transforms: [lowercase, strip]
        """))
        return data_path, gt_path, config_path

    def test_evaluate_basic(self, sample_data):
        data_path, gt_path, config_path = sample_data
        result = runner.invoke(app, [
            "evaluate",
            str(data_path),
            "--config", str(config_path),
            "--ground-truth", str(gt_path),
        ])
        assert result.exit_code == 0
        assert "Precision" in result.stdout or "precision" in result.stdout.lower()
        assert "Recall" in result.stdout or "recall" in result.stdout.lower()

    def test_evaluate_missing_gt(self, sample_data, tmp_path):
        data_path, _, config_path = sample_data
        result = runner.invoke(app, [
            "evaluate",
            str(data_path),
            "--config", str(config_path),
            "--ground-truth", str(tmp_path / "does_not_exist.csv"),
        ])
        assert result.exit_code != 0
```

- [ ] **Step 5: Implement the evaluate CLI command**

Create `goldenmatch/cli/evaluate.py`:

```python
"""CLI evaluate command for GoldenMatch."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps
from goldenmatch.config.loader import load_config
from goldenmatch.core.evaluate import evaluate_clusters, load_ground_truth_csv

console = Console()
err_console = Console(stderr=True)


def evaluate_cmd(
    files: list[str] = typer.Argument(..., help="Input files (path or path:source_name)"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    ground_truth: Path = typer.Option(..., "--ground-truth", "--gt", help="Ground truth CSV path"),
    col_a: str = typer.Option("id_a", "--col-a", help="Ground truth column A"),
    col_b: str = typer.Option("id_b", "--col-b", help="Ground truth column B"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Override match threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
) -> None:
    """Evaluate matching quality against ground truth pairs."""
    from goldenmatch.core.pipeline import run_dedupe

    if not ground_truth.exists():
        err_console.print(f"[red]Ground truth file not found: {ground_truth}[/red]")
        raise typer.Exit(1)

    cfg = load_config(str(config))

    # Override threshold if specified
    if threshold is not None:
        for mk in cfg.get_matchkeys():
            if mk.threshold is not None:
                mk.threshold = threshold

    parsed = [_parse_file_source(f) for f in files]
    file_specs = _resolve_column_maps(parsed, cfg)

    gt_pairs = load_ground_truth_csv(str(ground_truth), col_a, col_b)

    console.print(f"[bold]Evaluating with {len(gt_pairs)} ground truth pairs...[/bold]\n")

    result = run_dedupe(file_specs, cfg)
    clusters = result["clusters"]

    eval_result = evaluate_clusters(clusters, gt_pairs)

    # Display results
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    summary = eval_result.summary()
    table.add_row("Precision", f"{summary['precision']:.1%}")
    table.add_row("Recall", f"{summary['recall']:.1%}")
    table.add_row("F1 Score", f"{summary['f1']:.1%}")
    table.add_row("True Positives", str(summary["tp"]))
    table.add_row("False Positives", str(summary["fp"]))
    table.add_row("False Negatives", str(summary["fn"]))
    table.add_row("Predicted Pairs", str(summary["predicted_pairs"]))
    table.add_row("Ground Truth Pairs", str(summary["ground_truth_pairs"]))

    console.print(table)

    if output:
        import json
        output.write_text(json.dumps(summary, indent=2))
        console.print(f"\n[green]Results saved to {output}[/green]")
```

- [ ] **Step 6: Register evaluate command in main.py**

In `goldenmatch/cli/main.py`, add import and registration:

```python
from goldenmatch.cli.evaluate import evaluate_cmd
# ... in the app setup section:
app.command("evaluate")(evaluate_cmd)
```

- [ ] **Step 7: Run all evaluate tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `pytest --tb=short`
Expected: 831+ tests pass, no regressions

- [ ] **Step 9: Commit**

```bash
git add goldenmatch/core/evaluate.py goldenmatch/cli/evaluate.py tests/test_evaluate.py goldenmatch/cli/main.py
git commit -m "feat: add goldenmatch evaluate CLI command with precision/recall/F1 reporting"
```

---

## Task 3: GitHub Actions "Try It" Workflow

**Files:**
- Create: `.github/workflows/try-it.yml`
- Create: `.github/ISSUE_TEMPLATE/try-goldenmatch.yml`

- [ ] **Step 1: Create the workflow file**

`.github/workflows/try-it.yml`:

```yaml
name: "Try GoldenMatch"

on:
  workflow_dispatch:
    inputs:
      csv_url:
        description: "URL to a CSV file (raw GitHub URL, public URL, or gist)"
        required: true
        type: string
      config:
        description: "Match config (simple|fuzzy|exact) or raw YAML"
        required: false
        default: "simple"
        type: string
      name_columns:
        description: "Comma-separated name columns (e.g. first_name,last_name)"
        required: false
        type: string
      email_column:
        description: "Email column name"
        required: false
        type: string

jobs:
  dedupe:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install GoldenMatch
        run: pip install goldenmatch

      - name: Download CSV
        run: |
          curl -fsSL "${{ github.event.inputs.csv_url }}" -o input.csv
          echo "=== First 5 rows ==="
          head -6 input.csv

      - name: Generate config
        run: |
          python -c "
          import sys, yaml

          preset = '${{ github.event.inputs.config }}'
          name_cols = '${{ github.event.inputs.name_columns }}'.strip()
          email_col = '${{ github.event.inputs.email_column }}'.strip()

          if preset in ('simple', 'exact', 'fuzzy'):
              matchkeys = []
              if email_col:
                  matchkeys.append({
                      'name': 'exact_email',
                      'type': 'exact',
                      'fields': [{'field': email_col, 'transforms': ['lowercase', 'strip']}],
                  })
              if name_cols:
                  fields = []
                  for col in name_cols.split(','):
                      col = col.strip()
                      fields.append({
                          'field': col,
                          'transforms': ['lowercase', 'strip'],
                          'scorer': 'jaro_winkler',
                          'weight': 1.0,
                      })
                  matchkeys.append({
                      'name': 'fuzzy_name',
                      'type': 'weighted',
                      'threshold': 0.85,
                      'fields': fields,
                  })
              if not matchkeys:
                  # Auto-detect: use all string columns
                  import polars as pl
                  df = pl.read_csv('input.csv', n_rows=5)
                  str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8][:3]
                  for col in str_cols:
                      matchkeys.append({
                          'name': f'exact_{col}',
                          'type': 'exact',
                          'fields': [{'field': col, 'transforms': ['lowercase', 'strip']}],
                      })
              config = {'matchkeys': matchkeys}
          else:
              config = yaml.safe_load(preset)

          with open('config.yaml', 'w') as f:
              yaml.dump(config, f, default_flow_style=False)
          print('Generated config:')
          print(yaml.dump(config, default_flow_style=False))
          "

      - name: Run GoldenMatch
        run: goldenmatch dedupe input.csv --config config.yaml --output results.csv

      - name: Show results summary
        run: |
          python -c "
          import polars as pl
          df = pl.read_csv('results.csv')
          total = df.height
          dupes = df.filter(pl.col('__classification__') == 'dupe').height if '__classification__' in df.columns else 0
          clusters = df['__cluster_id__'].n_unique() if '__cluster_id__' in df.columns else 0
          print(f'Total records: {total}')
          print(f'Clusters found: {clusters}')
          print(f'Duplicate records: {dupes}')
          print(f'Match rate: {dupes/total:.1%}' if total else 'N/A')
          print()
          print('First 20 rows:')
          print(df.head(20))
          "

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: goldenmatch-results
          path: results.csv
          retention-days: 30
```

- [ ] **Step 2: Create issue template for easy access**

`.github/ISSUE_TEMPLATE/try-goldenmatch.yml`:

```yaml
name: "Try GoldenMatch on my data"
description: "Submit a CSV URL to test GoldenMatch deduplication"
title: "[Try It] "
labels: ["try-it"]
body:
  - type: markdown
    attributes:
      value: |
        ## Try GoldenMatch
        Want to try GoldenMatch on your data without installing anything?

        **Option 1 (recommended):** Use the [Try GoldenMatch workflow](../../actions/workflows/try-it.yml) directly.

        **Option 2:** Fill out this form and a maintainer will trigger the workflow for you.
  - type: input
    id: csv_url
    attributes:
      label: CSV URL
      description: "Public URL to your CSV file (GitHub raw URL, gist, etc.)"
      placeholder: "https://raw.githubusercontent.com/..."
    validations:
      required: true
  - type: input
    id: columns
    attributes:
      label: Name columns
      description: "Comma-separated columns to match on (e.g. first_name,last_name)"
      placeholder: "first_name,last_name"
  - type: input
    id: email
    attributes:
      label: Email column
      description: "Email column name (optional)"
      placeholder: "email"
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/try-it.yml .github/ISSUE_TEMPLATE/try-goldenmatch.yml
git commit -m "feat: add GitHub Actions 'Try It' workflow for zero-install demo"
```

---

## Task 4: GitHub Codespaces Setup

**Files:**
- Create: `.devcontainer/devcontainer.json`
- Create: `.devcontainer/post-create.sh`

- [ ] **Step 1: Create devcontainer.json**

```json
{
  "name": "GoldenMatch Dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "postCreateCommand": "bash .devcontainer/post-create.sh",
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests"],
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff"
      },
      "extensions": [
        "ms-python.python",
        "charliermarsh.ruff",
        "ms-toolsai.jupyter"
      ]
    }
  },
  "forwardPorts": [8000],
  "portsAttributes": {
    "8000": {
      "label": "GoldenMatch API",
      "onAutoForward": "notify"
    }
  }
}
```

- [ ] **Step 2: Create post-create.sh**

```bash
#!/usr/bin/env bash
set -e

echo "=== Installing GoldenMatch ==="
pip install -e ".[dev]"

echo "=== Verifying installation ==="
goldenmatch --help

echo "=== Running quick test ==="
pytest tests/test_config.py -q --tb=short

echo ""
echo "============================================"
echo "  GoldenMatch dev environment ready!"
echo ""
echo "  Quick start:"
echo "    goldenmatch demo          # Run built-in demo"
echo "    goldenmatch dedupe --help # See dedupe options"
echo "    pytest --tb=short         # Run test suite"
echo "============================================"
```

- [ ] **Step 3: Commit**

```bash
git add .devcontainer/
git commit -m "feat: add GitHub Codespaces devcontainer for one-click dev environment"
```

---

## Task 5: Incremental / Append-Only Mode

**Files:**
- Create: `goldenmatch/cli/incremental.py`
- Create: `tests/test_cli_incremental.py`
- Modify: `goldenmatch/cli/main.py` — register incremental command

This wraps the existing `StreamProcessor` and `match_one` into a proper CLI experience for file-based incremental matching (without requiring Postgres).

- [ ] **Step 1: Write tests for incremental CLI**

```python
"""Tests for incremental CLI command."""
from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from goldenmatch.cli.main import app

runner = CliRunner()


@pytest.fixture
def base_csv(tmp_path) -> Path:
    path = tmp_path / "base.csv"
    pl.DataFrame({
        "first_name": ["John", "Jane", "Bob"],
        "last_name": ["Smith", "Doe", "Jones"],
        "email": ["john@ex.com", "jane@t.com", "bob@t.com"],
    }).write_csv(path)
    return path


@pytest.fixture
def new_csv(tmp_path) -> Path:
    path = tmp_path / "new.csv"
    pl.DataFrame({
        "first_name": ["john", "Alice"],
        "last_name": ["Smith", "Brown"],
        "email": ["john@ex.com", "alice@t.com"],
    }).write_csv(path)
    return path


@pytest.fixture
def simple_config(tmp_path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent("""\
        matchkeys:
          - name: exact_email
            type: exact
            fields:
              - field: email
                transforms: [lowercase, strip]
    """))
    return path


class TestIncrementalCLI:
    def test_basic_incremental(self, base_csv, new_csv, simple_config, tmp_path):
        output = tmp_path / "output.csv"
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(new_csv),
            "--config", str(simple_config),
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert df.height > 0

    def test_incremental_shows_stats(self, base_csv, new_csv, simple_config):
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(new_csv),
            "--config", str(simple_config),
        ])
        assert result.exit_code == 0
        out = result.stdout.lower()
        assert "matched" in out or "new" in out or "processed" in out

    def test_incremental_missing_new(self, base_csv, simple_config, tmp_path):
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(tmp_path / "does_not_exist.csv"),
            "--config", str(simple_config),
        ])
        assert result.exit_code != 0
```

Run: `pytest tests/test_cli_incremental.py -v`
Expected: FAIL — incremental command doesn't exist

- [ ] **Step 2: Implement incremental CLI command**

Create `goldenmatch/cli/incremental.py`:

```python
"""CLI incremental command for GoldenMatch."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.config.loader import load_config

console = Console()
err_console = Console(stderr=True)


def incremental_cmd(
    base_file: str = typer.Argument(..., help="Base dataset file path"),
    new_records: Path = typer.Option(..., "--new-records", "-n", help="New records CSV to match"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Override threshold"),
) -> None:
    """Match new records against an existing base dataset incrementally."""
    import polars as pl
    from goldenmatch.core.ingest import load_file
    from goldenmatch.core.autofix import auto_fix_dataframe
    from goldenmatch.core.standardize import apply_standardization
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.match_one import match_one
    from goldenmatch.core.cluster import build_clusters

    if not new_records.exists():
        err_console.print(f"[red]New records file not found: {new_records}[/red]")
        raise typer.Exit(1)

    cfg = load_config(str(config))
    matchkeys = cfg.get_matchkeys()

    if threshold is not None:
        for mk in matchkeys:
            if mk.threshold is not None:
                mk.threshold = threshold

    # Load base dataset
    console.print("[bold]Loading base dataset...[/bold]")
    base_lf = load_file(base_file)
    base_df = base_lf.collect()
    base_df = base_df.with_row_index("__row_id__").with_columns(
        pl.col("__row_id__").cast(pl.Int64),
        pl.lit("base").alias("__source__"),
    )
    base_df, _ = auto_fix_dataframe(base_df)

    # Load new records
    console.print("[bold]Loading new records...[/bold]")
    new_lf = load_file(str(new_records))
    new_df = new_lf.collect()
    base_max_id = base_df["__row_id__"].max() + 1 if base_df.height > 0 else 0
    new_df = new_df.with_row_index("__row_id__").with_columns(
        (pl.col("__row_id__").cast(pl.Int64) + base_max_id).alias("__row_id__"),
        pl.lit("new").alias("__source__"),
    )
    new_df, _ = auto_fix_dataframe(new_df)

    # Standardize and compute matchkeys on combined data
    combined = pl.concat([base_df, new_df], how="diagonal")
    lf = combined.lazy()
    if cfg.standardization:
        lf = apply_standardization(lf, cfg.standardization)
    for mk in matchkeys:
        lf = compute_matchkeys(lf, [mk])
    combined = lf.collect()

    # Match each new record against the base
    console.print(f"[bold]Matching {new_df.height} new records against {base_df.height} base records...[/bold]")
    t0 = time.perf_counter()

    all_matches = []
    matched_count = 0
    new_entity_count = 0

    new_ids = set(range(base_max_id, base_max_id + new_df.height))

    # Pre-index rows for O(1) lookup instead of O(N) filter per record
    row_index = {}
    for row in combined.to_dicts():
        row_index[row["__row_id__"]] = row

    for new_id in sorted(new_ids):
        row = row_index.get(new_id)
        if not row:
            continue
        # Collect matches across all matchkeys for this record
        record_matches = {}
        for mk in matchkeys:
            matches = match_one(row, combined, mk)
            for rid, score in matches:
                if rid not in new_ids:
                    # Keep best score per base record
                    if rid not in record_matches or score > record_matches[rid]:
                        record_matches[rid] = score
        if record_matches:
            matched_count += 1
            for rid, score in record_matches.items():
                all_matches.append((new_id, rid, score))
        else:
            new_entity_count += 1

    elapsed = time.perf_counter() - t0

    # Build results
    table = Table(title="Incremental Match Results")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("New records processed", str(new_df.height))
    table.add_row("Matched to base", str(matched_count))
    table.add_row("New entities", str(new_entity_count))
    table.add_row("Total match pairs", str(len(all_matches)))
    table.add_row("Time", f"{elapsed:.2f}s")
    console.print(table)

    if output and all_matches:
        rows = []
        combined_dicts = {r["__row_id__"]: r for r in combined.to_dicts()}
        for new_id, base_id, score in all_matches:
            rows.append({
                "new_row_id": new_id,
                "base_row_id": base_id,
                "score": round(score, 4),
            })
        result_df = pl.DataFrame(rows)
        result_df.write_csv(output)
        console.print(f"\n[green]Results saved to {output}[/green]")
    elif output:
        console.print("\n[yellow]No matches found - no output written[/yellow]")
```

- [ ] **Step 3: Register incremental command in main.py**

In `goldenmatch/cli/main.py`, add:

```python
from goldenmatch.cli.incremental import incremental_cmd
# ... registration:
app.command("incremental")(incremental_cmd)
```

- [ ] **Step 4: Run incremental tests**

Run: `pytest tests/test_cli_incremental.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass, no regressions

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/cli/incremental.py tests/test_cli_incremental.py goldenmatch/cli/main.py
git commit -m "feat: add goldenmatch incremental CLI for append-only matching"
```

---

## Task 6: dbt Integration (Separate Package)

**Files:**
- Create: `dbt-goldenmatch/` directory with package structure
- This is a separate pip-installable package

- [ ] **Step 1: Create package structure**

```
dbt-goldenmatch/
  pyproject.toml
  dbt_goldenmatch/
    __init__.py
    materialize.py
  README.md
```

- [ ] **Step 2: Write dbt-goldenmatch/pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dbt-goldenmatch"
version = "0.1.0"
description = "dbt integration for GoldenMatch entity resolution"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "Ben Severn", email = "benzsevern@gmail.com" }]
dependencies = [
    "goldenmatch>=0.3.0",
    "dbt-core>=1.7",
    "duckdb>=0.9",
]

[project.entry-points."goldenmatch.plugins"]
dbt = "dbt_goldenmatch:DbtPlugin"
```

- [ ] **Step 3: Write dbt_goldenmatch/__init__.py**

```python
"""dbt integration for GoldenMatch entity resolution."""
__version__ = "0.1.0"
```

- [ ] **Step 4: Write dbt_goldenmatch/materialize.py**

```python
"""GoldenMatch materialization for dbt.

Usage in dbt model:
    {{ config(materialized='goldenmatch_dedupe', match_config='match.yaml') }}
    SELECT * FROM {{ ref('raw_customers') }}
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from goldenmatch.config.loader import load_config
from goldenmatch.core.pipeline import run_dedupe


def run_goldenmatch_dedupe(
    input_table: str,
    config_path: str,
    output_table: str,
    database: str = ":memory:",
) -> dict:
    """Run GoldenMatch dedupe on a DuckDB table and write results back.

    Args:
        input_table: Source table name in DuckDB
        config_path: Path to GoldenMatch YAML config
        output_table: Destination table name
        database: DuckDB database path

    Returns:
        Summary dict with record counts and match rate
    """
    conn = duckdb.connect(database)

    # Read input
    df = conn.execute(f"SELECT * FROM {input_table}").pl()

    # Write to temp CSV for GoldenMatch ingest
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name
        df.write_csv(tmp_path)

    cfg = load_config(config_path)
    result = run_dedupe([(tmp_path, "source")], cfg)

    # Write results to DuckDB
    output_df = result.get("golden") or result.get("output")
    if output_df is not None:
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        conn.execute(f"CREATE TABLE {output_table} AS SELECT * FROM output_df")

    Path(tmp_path).unlink(missing_ok=True)

    stats = result.get("stats", {})
    conn.close()
    return {
        "input_rows": df.height,
        "output_rows": output_df.height if output_df is not None else 0,
        "clusters": stats.get("total_clusters", 0),
    }
```

- [ ] **Step 5: Write README.md**

```markdown
# dbt-goldenmatch

dbt integration for [GoldenMatch](https://github.com/benzsevern/goldenmatch) entity resolution.

## Installation

```bash
pip install dbt-goldenmatch
```

## Usage

Run GoldenMatch deduplication on a DuckDB table:

```python
from dbt_goldenmatch.materialize import run_goldenmatch_dedupe

result = run_goldenmatch_dedupe(
    input_table="raw_customers",
    config_path="match.yaml",
    output_table="deduped_customers",
    database="warehouse.duckdb",
)
print(f"Deduped {result['input_rows']} -> {result['clusters']} clusters")
```

## Status

Early stage - API may change. Full dbt materialization plugin coming soon.
```

- [ ] **Step 6: Commit**

```bash
git add dbt-goldenmatch/
git commit -m "feat: add dbt-goldenmatch package for DuckDB-based entity resolution"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
pytest --tb=short
```
Expected: All 831+ tests pass plus new tests for domain packs, evaluate, and incremental CLI.

- [ ] **Verify domain packs**

```bash
python -c "from goldenmatch.core.domain_registry import discover_rulebooks; print(list(discover_rulebooks().keys()))"
```
Expected: `['electronics', 'software', 'healthcare', 'financial', 'real_estate', 'people', 'retail']`

- [ ] **Test evaluate CLI**

```bash
goldenmatch evaluate --help
```
Expected: Shows help with --config, --ground-truth, --col-a, --col-b options

- [ ] **Test incremental CLI**

```bash
goldenmatch incremental --help
```
Expected: Shows help with --new-records, --config, --output options
