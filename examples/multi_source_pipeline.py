#!/usr/bin/env python
"""Multi-source deduplication pipeline.

Combines CRM, marketing, and vendor data into a unified golden record set.
Shows: multi-file dedupe, cross-source matching, golden record merging,
cluster inspection, and explainability.
"""
import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# CRM data
crm = pl.DataFrame({
    "first_name": ["John", "Jane", "Bob"],
    "last_name": ["Smith", "Doe", "Jones"],
    "email": ["john.smith@company.com", "jane.doe@company.com", "bob.jones@work.com"],
    "phone": ["555-1234", "555-5678", "555-9999"],
})

# Marketing data (same people, different formatting)
marketing = pl.DataFrame({
    "first_name": ["JOHN", "Janet", "Robert"],
    "last_name": ["SMITH", "Doe", "Jones"],
    "email": ["john.smith@company.com", "jane.doe@company.com", "bob@personal.com"],
    "phone": ["5551234", "", "555-9999"],
})

# Vendor data (partial overlap)
vendor = pl.DataFrame({
    "first_name": ["J.", "Alice", "Bob"],
    "last_name": ["Smith", "Brown", "Jones"],
    "email": ["john.smith@company.com", "alice@other.com", "bob.jones@work.com"],
    "phone": ["", "555-0000", "555-9999"],
})

tmp = Path(tempfile.mkdtemp())
crm_path = tmp / "crm.csv"
mkt_path = tmp / "marketing.csv"
vendor_path = tmp / "vendor.csv"
crm.write_csv(crm_path)
marketing.write_csv(mkt_path)
vendor.write_csv(vendor_path)

print("Input sources:")
print(f"  CRM: {crm.height} records")
print(f"  Marketing: {marketing.height} records")
print(f"  Vendor: {vendor.height} records")

# Multi-source dedupe
result = gm.dedupe(
    str(crm_path), str(mkt_path), str(vendor_path),
    exact=["email"],
    fuzzy={"first_name": 0.4, "last_name": 0.4, "phone": 0.2},
    blocking=["last_name"],
)

print(f"\nResult: {result}")

# Inspect clusters
print("\nClusters:")
for cid, cinfo in result.clusters.items():
    members = cinfo["members"]
    confidence = cinfo["confidence"]
    bottleneck = cinfo.get("bottleneck_pair")
    print(f"  Cluster {cid}: {len(members)} records, confidence={confidence:.2f}")

    # Show pair scores within cluster
    for (a, b), score in cinfo.get("pair_scores", {}).items():
        print(f"    {a} <-> {b}: {score:.3f}")

# Golden records
if result.golden is not None:
    print(f"\nGolden records ({result.golden.height}):")
    display_cols = [c for c in result.golden.columns if not c.startswith("__")][:5]
    print(result.golden.select(display_cols).to_pandas().to_string())

# Export everything
output_dir = tmp / "output"
output_dir.mkdir()
result.to_csv(str(output_dir / "results.csv"), which="all")
print(f"\nExported to {output_dir}")
