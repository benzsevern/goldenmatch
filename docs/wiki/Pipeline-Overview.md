# Pipeline Overview

GoldenMatch processes records through a sequential pipeline:

```
Ingest → Auto-Fix → Standardize → Matchkeys → Block → Score → Cluster → Golden → Output
```

## Stage Details

### 1. Ingest

Load data from CSV, Excel, Parquet, or Postgres. Multiple files can be combined with source labels.

```bash
goldenmatch dedupe file1.csv:source_a file2.csv:source_b
```

Smart ingestion auto-detects encoding, delimiters, headers, and junk rows.

### 2. Auto-Fix

Automatic data quality fixes:
- Strip BOM characters
- Drop empty rows and null-only columns
- Trim whitespace
- Normalize null representations
- Collapse multiple whitespace
- Remove non-printable characters

### 3. Standardize

Per-column standardization rules:

```yaml
standardization:
  email: [email]           # normalize email format
  phone: [phone]           # normalize phone numbers
  name: [name_proper]      # proper case names
  zip: [zip5]              # 5-digit zip codes
```

### 4. Matchkeys

Define what constitutes a match:

- **Exact matchkeys**: Records with identical transformed values are duplicates
- **Weighted matchkeys**: Multiple fields scored independently, combined with weights

### 5. Block

Reduce the comparison space. Instead of comparing every pair (O(n²)), blocking groups records by shared attributes and only compares within groups.

See [Blocking Strategies](Blocking-Strategies.md) for details.

### 6. Score

Compare record pairs within blocks using the configured scorer. Each pair gets a similarity score between 0.0 and 1.0.

See [Matchkeys & Scoring](Matchkeys-and-Scoring.md) for details.

### 7. Cluster

Group matched pairs into clusters using iterative Union-Find. If A matches B and B matches C, they form one cluster {A, B, C}.

### 8. Golden Record

Merge each cluster into a single canonical record using configured strategies.

See [Golden Records](Golden-Records.md) for details.

### 9. Output

Write results as CSV, Parquet, or to database tables.
