---
layout: default
title: Domain Packs
nav_order: 11
---

# Domain Packs

GoldenMatch includes 7 built-in YAML rulebooks that extract structured fields from unstructured product descriptions and other domain-specific text.

---

## Built-in packs

| Pack | Domain | Extracted Fields |
|------|--------|-----------------|
| `electronics` | Consumer electronics | brand, model, SKU, color, specs |
| `software` | Software products | name, version, edition, platform |
| `healthcare` | Medical records | NPI, CPT codes, drug names, dosages |
| `financial` | Financial instruments | CUSIP, LEI, ticker, account numbers |
| `real_estate` | Property listings | address, MLS number, lot size, year built |
| `people` | Person records | name parts, phone, email, SSN pattern |
| `retail` | General retail | brand, SKU, UPC, size, color |

---

## Using domain packs

### Auto-detection

```python
import goldenmatch as gm

rulebooks = gm.discover_rulebooks()  # Returns all 7 packs
print(list(rulebooks.keys()))
# ['electronics', 'software', 'healthcare', 'financial', 'real_estate', 'people', 'retail']
```

### Extract fields

```python
import goldenmatch as gm

rulebooks = gm.discover_rulebooks()
enhanced_df, low_confidence = gm.extract_with_rulebook(df, "title", rulebooks["electronics"])

# enhanced_df has new columns: __brand__, __model__, __sku__, etc.
# low_confidence contains records where extraction confidence was low
```

### Auto-detect domain

```python
domain = gm.match_domain(df, "description")
# Returns "electronics", "software", etc., or None
```

---

## YAML config

Enable domain extraction in your config file:

```yaml
domain:
  enabled: true
  pack: electronics
```

Or let GoldenMatch auto-detect:

```yaml
domain:
  enabled: true
```

---

## Electronics pack

Extracts brand, model number, SKU, color, and technical specs from product titles.

```
"Samsung Galaxy S24 Ultra 256GB Titanium Black SM-S928B"
  -> brand: Samsung
  -> model: Galaxy S24 Ultra
  -> sku: SM-S928B
  -> color: Titanium Black
  -> specs: 256GB
```

Model normalization strips hyphens, region suffixes, and color suffixes for better matching.

---

## Software pack

Extracts name, version, edition, and platform.

```
"Microsoft Office 365 Professional Plus - Windows"
  -> name: Microsoft Office
  -> version: 365
  -> edition: Professional Plus
  -> platform: Windows
```

---

## Healthcare pack

Extracts medical identifiers with contextual prefix requirements (e.g., `NPI:`, `CPT:`) to avoid false positives on generic numbers.

```
"Provider NPI:1234567890, CPT:99213 Office Visit"
  -> npi: 1234567890
  -> cpt_code: 99213
```

---

## Financial pack

Extracts financial identifiers (CUSIP, LEI, ticker). Contextual prefixes required.

```
"Bond CUSIP:037833AK6, Issuer LEI:HWUPKR0MPOU8FGXBT394"
  -> cusip: 037833AK6
  -> lei: HWUPKR0MPOU8FGXBT394
```

---

## Custom domain packs

Create your own YAML rulebook and place it in one of the search paths:

| Path | Scope |
|------|-------|
| `.goldenmatch/domains/` | Project-local |
| `~/.goldenmatch/domains/` | Global (user) |
| `goldenmatch/domains/` | Built-in (read-only) |

### Rulebook YAML format

```yaml
# .goldenmatch/domains/my_domain.yaml
name: my_domain
description: Custom domain for matching widgets
signals:
  - pattern: "widget"
    weight: 1.0
  - pattern: "part_?number"
    weight: 0.8
extractors:
  - name: part_number
    pattern: "PN[:-]?\\s*(\\w{6,12})"
    group: 1
  - name: manufacturer
    pattern: "(Acme|Globex|Initech)"
    group: 1
normalizers:
  part_number:
    strip_chars: "-"
    uppercase: true
```

### Create via Python

```python
import goldenmatch as gm

gm.save_rulebook("my_domain", rulebook)
loaded = gm.load_rulebook("my_domain")
```

### Create via MCP

The MCP server provides tools for domain management:

| Tool | Description |
|------|-------------|
| `list_domains` | List all available domain packs |
| `create_domain` | Create a new custom domain pack |
| `test_domain` | Test a domain pack against sample data |

---

## Domain extraction in the pipeline

Domain extraction runs between the standardize and matchkeys steps. It adds extracted fields as new columns (prefixed with `__`) that can be used in matchkeys:

```yaml
matchkeys:
  - name: product_match
    type: weighted
    threshold: 0.85
    fields:
      - field: __brand__
        scorer: exact
        weight: 0.3
      - field: __model__
        scorer: jaro_winkler
        weight: 0.5
      - field: title
        scorer: token_sort
        weight: 0.2
```

---

## Benchmarks

Domain extraction significantly improves product matching:

| Dataset | Without Domain | With Domain | Improvement |
|---------|---------------|-------------|-------------|
| Abt-Buy (electronics) | 44.5% F1 | 72.2% F1 | +27.7pp |
| Amazon-Google (software) | 45.3% F1 | 42.1% F1 | -3.2pp |

Domain extraction helps datasets with structured identifiers (brand, model, SKU) but can hurt datasets with unstructured descriptions. For software matching, clean embedding + ANN pipelines perform better.
