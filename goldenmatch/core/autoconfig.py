"""Auto-configuration engine for GoldenMatch zero-config mode."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    GoldenRulesConfig,
    MatchkeyConfig,
    MatchkeyField,
    OutputConfig,
)
from goldenmatch.core.profiler import _guess_type

logger = logging.getLogger(__name__)

# ── Column name heuristics ─────────────────────────────────────────────────

_NAME_PATTERNS = re.compile(
    r"(^name$|first.?name|last.?name|full.?name|fname|lname|surname|given.?name)",
    re.IGNORECASE,
)
_EMAIL_PATTERNS = re.compile(r"(email|e.?mail|email.?addr)", re.IGNORECASE)
_PHONE_PATTERNS = re.compile(r"(phone|tel|mobile|fax|cell)", re.IGNORECASE)
_ZIP_PATTERNS = re.compile(r"(zip|postal|postcode|zip.?code)", re.IGNORECASE)
_ADDRESS_PATTERNS = re.compile(r"(address|street|addr|line.?1|line.?2)", re.IGNORECASE)
_GEO_PATTERNS = re.compile(r"(^city$|^state$|^country$|province|region)", re.IGNORECASE)
_ID_PATTERNS = re.compile(r"(^id$|^key$|^code$|^sku$|_id$|_key$)", re.IGNORECASE)


@dataclass
class ColumnProfile:
    """Profile of a single column for auto-configuration."""

    name: str
    dtype: str
    col_type: str  # email, name, phone, zip, address, geo, identifier, description, numeric, date, string
    confidence: float  # 0.0 to 1.0
    sample_values: list[str] = field(default_factory=list)


def _classify_by_name(col_name: str) -> str | None:
    """Phase 1: classify column by name pattern matching."""
    if _EMAIL_PATTERNS.search(col_name):
        return "email"
    if _PHONE_PATTERNS.search(col_name):
        return "phone"
    if _ZIP_PATTERNS.search(col_name):
        return "zip"
    if _NAME_PATTERNS.search(col_name):
        return "name"
    if _ADDRESS_PATTERNS.search(col_name):
        return "address"
    if _GEO_PATTERNS.search(col_name):
        return "geo"
    if _ID_PATTERNS.search(col_name):
        return "identifier"
    return None


def _classify_by_data(values: list[str]) -> tuple[str, float]:
    """Phase 2: classify column by data profiling. Returns (type, confidence)."""
    if not values:
        return "string", 0.0

    data_type = _guess_type(values)

    # Map profiler types to our types
    type_map = {
        "email": "email",
        "phone": "phone",
        "zip": "zip",
        "state": "geo",
        "numeric": "numeric",
        "name": "name",
        "address": "address",
        "date": "date",
        "text": "string",
    }

    col_type = type_map.get(data_type, "string")

    # Check for description (long freetext)
    if col_type == "string":
        avg_len = sum(len(v) for v in values) / len(values) if values else 0
        if avg_len > 50:
            col_type = "description"

    # Confidence based on how strongly data matches the type
    confidence = 0.7 if col_type != "string" else 0.3
    return col_type, confidence


def profile_columns(df: pl.DataFrame, sample_size: int = 1000) -> list[ColumnProfile]:
    """Classify columns by type using name heuristics + data profiling.

    Samples randomly to avoid bias from header-adjacent rows.
    """
    # Sample randomly
    if df.height > sample_size:
        sample = df.sample(sample_size, seed=42)
    else:
        sample = df

    profiles = []
    for col_name in df.columns:
        # Skip internal columns
        if col_name.startswith("__"):
            continue

        dtype = str(df[col_name].dtype)

        # Get non-null string values for profiling
        values = [
            str(v) for v in sample[col_name].drop_nulls().to_list()
            if v is not None and str(v).strip()
        ]

        # Phase 1: name heuristics
        name_type = _classify_by_name(col_name)

        # Phase 2: data profiling
        data_type, data_confidence = _classify_by_data(values)

        # Combine: Phase 2 wins when it contradicts Phase 1
        if name_type and data_type != "string":
            # Both have opinions — Phase 2 wins if types differ
            if name_type == data_type:
                col_type = name_type
                confidence = min(data_confidence + 0.2, 1.0)
            else:
                col_type = data_type
                confidence = data_confidence
        elif name_type:
            col_type = name_type
            confidence = 0.6
        else:
            col_type = data_type
            confidence = data_confidence

        profiles.append(ColumnProfile(
            name=col_name,
            dtype=dtype,
            col_type=col_type,
            confidence=confidence,
            sample_values=values[:5],
        ))

    return profiles


# ── Scorer and matchkey generation ─────────────────────────────────────────

_SCORER_MAP = {
    "email": ("exact", 1.0, ["lowercase", "strip"]),
    "phone": ("exact", 0.8, ["digits_only"]),
    "zip": ("exact", 0.5, ["strip"]),
    "name": ("ensemble", 1.0, ["lowercase", "strip"]),
    "address": ("token_sort", 0.8, ["lowercase", "strip"]),
    "identifier": ("exact", 1.0, ["strip"]),
    "geo": ("exact", 0.3, ["lowercase", "strip"]),
    "string": ("token_sort", 0.5, ["lowercase", "strip"]),
}


def _adaptive_threshold(fields: list[MatchkeyField]) -> float:
    """Compute threshold based on field types in the matchkey."""
    exact_scorers = {"exact"}
    fuzzy_scorers = {"jaro_winkler", "levenshtein", "token_sort", "ensemble", "soundex_match"}
    embedding_scorers = {"embedding", "record_embedding"}

    scorers = {f.scorer for f in fields if f.scorer}

    if scorers <= exact_scorers:
        return 0.95
    if scorers & embedding_scorers:
        return 0.70
    if len(fields) == 1:
        return 0.85
    return 0.80


def build_matchkeys(profiles: list[ColumnProfile]) -> list[MatchkeyConfig]:
    """Generate matchkeys from column profiles."""
    # Separate exact and fuzzy columns
    exact_fields = []
    fuzzy_fields = []
    description_columns = []

    for p in profiles:
        if p.col_type in ("numeric", "date", "identifier"):
            continue  # skip non-matchable columns

        if p.col_type == "description":
            description_columns.append(p)
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, weight, transforms = scorer_info
        mf = MatchkeyField(
            field=p.name,
            scorer=scorer,
            weight=weight,
            transforms=transforms,
        )

        if scorer == "exact":
            exact_fields.append(mf)
        else:
            fuzzy_fields.append(mf)

    matchkeys = []

    # Exact matchkey from exact fields
    if exact_fields:
        for f in exact_fields:
            matchkeys.append(MatchkeyConfig(
                name=f"exact_{f.field}",
                type="exact",
                fields=[MatchkeyField(
                    field=f.field,
                    transforms=f.transforms,
                )],
            ))

    # Weighted matchkey from fuzzy fields
    all_weighted = list(fuzzy_fields)

    # Add description columns as record_embedding
    if description_columns:
        all_weighted.append(MatchkeyField(
            scorer="record_embedding",
            columns=[p.name for p in description_columns],
            weight=1.0,
            model=None,  # auto-selected later
        ))

    if all_weighted:
        threshold = _adaptive_threshold(all_weighted)
        matchkeys.append(MatchkeyConfig(
            name="fuzzy_match",
            type="weighted",
            threshold=threshold,
            fields=all_weighted,
        ))

    # Fallback: if nothing was generated, use all string columns with token_sort
    if not matchkeys:
        string_cols = [p for p in profiles if p.dtype.startswith("String") or p.dtype.startswith("Utf8")]
        if string_cols:
            fields = [
                MatchkeyField(
                    field=p.name,
                    scorer="token_sort",
                    weight=1.0,
                    transforms=["lowercase", "strip"],
                )
                for p in string_cols[:3]  # limit to first 3
            ]
            matchkeys.append(MatchkeyConfig(
                name="fallback_fuzzy",
                type="weighted",
                threshold=0.80,
                fields=fields,
            ))

    return matchkeys


# ── Blocking generation ────────────────────────────────────────────────────

def build_blocking(profiles: list[ColumnProfile], df: pl.DataFrame) -> BlockingConfig:
    """Generate blocking config from column profiles."""
    exact_cols = [p for p in profiles if p.col_type in ("email", "phone", "zip", "identifier")]
    name_cols = [p for p in profiles if p.col_type == "name"]
    text_cols = [p for p in profiles if p.col_type in ("description", "string", "address")]

    # Best case: block on highest-cardinality exact column
    if exact_cols:
        # Sort by cardinality (descending)
        best = max(exact_cols, key=lambda p: df[p.name].n_unique())
        transforms = ["lowercase", "strip"] if best.col_type == "email" else ["strip"]
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best.name], transforms=transforms)],
        )

    # Name columns: use multi-pass with soundex + substring
    if name_cols:
        best_name = name_cols[0].name
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "token_sort", "substring:0:8"]),
            ],
            max_block_size=500,
        )

    # Last resort: canopy on best text column
    if text_cols:
        from goldenmatch.config.schemas import CanopyConfig
        best_text = text_cols[0].name
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best_text], transforms=["lowercase", "substring:0:5"])],
            strategy="canopy",
            canopy=CanopyConfig(
                fields=[best_text],
                loose_threshold=0.3,
                tight_threshold=0.7,
            ),
        )

    # Absolute fallback
    first_string = next(
        (p for p in profiles if not p.col_type == "numeric"),
        profiles[0] if profiles else None,
    )
    if first_string:
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[first_string.name], transforms=["lowercase", "substring:0:5"])],
        )

    return BlockingConfig(keys=[BlockingKeyConfig(fields=[profiles[0].name])])


# ── Model selection ────────────────────────────────────────────────────────

def select_model(row_count: int, has_embedding_columns: bool, threshold: int = 50000) -> str | None:
    """Select embedding model. Returns None if no embedding columns needed."""
    if not has_embedding_columns:
        return None
    if row_count < threshold:
        return "gte-base-en-v1.5"
    return "all-MiniLM-L6-v2"


# ── Main entry point ──────────────────────────────────────────────────────

def auto_configure_df(df: pl.DataFrame) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from a DataFrame.

    Profiles columns by name heuristics and data sampling, then builds
    matchkeys, blocking, and golden rules automatically.

    Args:
        df: Polars DataFrame to auto-configure for.

    Returns:
        A fully populated GoldenMatchConfig ready for pipeline execution.
    """
    total_rows = df.height

    logger.info("Auto-configuring %d rows, %d columns", total_rows, len(df.columns))

    # Profile columns
    profiles = profile_columns(df)

    logger.info(
        "Detected column types: %s",
        {p.name: p.col_type for p in profiles},
    )

    # Build matchkeys
    matchkeys = build_matchkeys(profiles)

    # Check if embeddings are needed
    has_embeddings = any(
        f.scorer in ("embedding", "record_embedding")
        for mk in matchkeys
        for f in mk.fields
    )

    # Select model and apply to embedding fields
    model = select_model(total_rows, has_embeddings)
    if model:
        for mk in matchkeys:
            for f in mk.fields:
                if f.scorer in ("embedding", "record_embedding") and not f.model:
                    f.model = model

    # Build blocking (required for weighted/probabilistic matchkeys)
    has_fuzzy = any(mk.type in ("weighted", "probabilistic") for mk in matchkeys)
    blocking = build_blocking(profiles, df) if has_fuzzy else None

    # Build config
    config = GoldenMatchConfig(
        matchkeys=matchkeys,
        blocking=blocking,
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
    )

    return config


def auto_configure(files: list[tuple[str, str]]) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from input files.

    Args:
        files: List of (path, source_name) tuples.

    Returns:
        A fully populated GoldenMatchConfig ready for pipeline execution.
    """
    # Load and combine files
    dfs = []
    for path, source_name in files:
        p = Path(path)
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pl.read_excel(p, engine="openpyxl")
        elif p.suffix.lower() == ".parquet":
            df = pl.read_parquet(p)
        else:
            df = pl.read_csv(p, encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
        dfs.append(df)

    combined = pl.concat(dfs, how="diagonal") if len(dfs) > 1 else dfs[0]
    return auto_configure_df(combined)


def build_probabilistic_matchkeys(profiles: list[ColumnProfile]) -> list[MatchkeyConfig]:
    """Generate Fellegi-Sunter probabilistic matchkeys from column profiles.

    Produces a single probabilistic matchkey using all matchable columns
    with appropriate comparison levels and partial thresholds.
    """
    fields = []
    for p in profiles:
        if p.col_type in ("numeric", "date", "identifier", "description"):
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, _weight, transforms = scorer_info

        # Determine comparison levels based on scorer type
        if scorer == "exact":
            levels = 2
            partial_threshold = 0.9
        else:
            levels = 3
            partial_threshold = 0.8

        fields.append(MatchkeyField(
            field=p.name,
            scorer=scorer,
            transforms=transforms,
            levels=levels,
            partial_threshold=partial_threshold,
        ))

    if not fields:
        return []

    return [MatchkeyConfig(
        name="probabilistic_auto",
        type="probabilistic",
        fields=fields,
    )]
