"""Auto-configuration engine for GoldenMatch zero-config mode."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from goldenmatch._polars_lazy import pl

if TYPE_CHECKING:
    from goldenmatch.config.schemas import StandardizationConfig
    from goldenmatch.core.autoconfig_memory import AutoConfigMemory

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    BudgetConfig,
    GoldenMatchConfig,
    GoldenRulesConfig,
    LLMScorerConfig,
    MatchkeyConfig,
    MatchkeyField,
    MemoryConfig,
    NegativeEvidenceField,
    OutputConfig,
)
from goldenmatch.core.autoconfig_discriminative import (
    should_demote_attribute_field,
    should_veto_exact,
)
from goldenmatch.core.blocking_union_core import (
    assemble_union,
    finalize_union,
    union_via_core_enabled,
)
from goldenmatch.core.complexity_profile import DataProfile
from goldenmatch.core.profile_emitter import _emitter_stack, current_emitter
from goldenmatch.core.profiler import _guess_type

logger = logging.getLogger(__name__)

# One-time guard for the arrow-native autoconfig coercion-fallback warning
# (GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE default-on): a sub-feature that still
# assumes a polars target (match/throughput) coerces the arrow input to polars.
# Warn once per feature per process instead of once per auto_configure_df call.
_ARROW_NATIVE_COERCED_WARNED: set[str] = set()


def _warn_arrow_native_coerced_once(feature: str) -> None:
    if feature in _ARROW_NATIVE_COERCED_WARNED:
        return
    _ARROW_NATIVE_COERCED_WARNED.add(feature)
    logger.warning(
        "auto_configure_df: %s is not arrow-native yet; coercing the arrow input "
        "to polars for this run (config-equivalent, imports polars). The "
        "zero-config dedupe path stays arrow-native. Set "
        "GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE=0 to always coerce.",
        feature,
    )


# Refdata-aware matchkey-field refinement. Hoisted to module-top so the
# per-iteration loop in build_matchkeys / build_probabilistic_matchkeys
# doesn't pay the `from X import Y` lookup cost or re-run the try/except
# block per column. Falls back to a no-op identity when the refdata
# package is unimportable OR its top-level import raises for any reason
# (corrupt bundled data file, transitively-broken pack, etc.) — the
# autoconfig path must never fail because a refdata pack is unhealthy.
try:
    from goldenmatch.refdata.autoconfig_hooks import (
        refine_matchkey_field as _refdata_refine_matchkey_field,
    )
except Exception as _refdata_import_exc:  # noqa: BLE001 — see comment above
    logger.debug(
        "refdata.autoconfig_hooks unavailable (%s); refinements disabled.",
        _refdata_import_exc,
    )

    def _refdata_refine_matchkey_field(
        column_name: str,
        scorer: str,
        transforms: list[str],
        col_type: str | None = None,
    ) -> tuple[str, list[str]]:
        return scorer, transforms


def _emit_data_profile(df: pl.DataFrame) -> None:
    """Emit DataProfile from the input DataFrame. No-op when null emitter."""
    if not _emitter_stack.get():
        return
    user_cols = [c for c in df.columns if not c.startswith("__")]
    n_rows = df.height
    # W3c: shared seam-routed body with the controller's twin (mirror retired).
    from goldenmatch.core._profile_helpers import data_profile_column_stats

    (
        column_types,
        cardinality_ratio,
        null_rate,
        value_length_p50,
        value_length_p99,
    ) = data_profile_column_stats(df, user_cols)
    current_emitter().set_data(DataProfile(
        n_rows=n_rows,
        n_cols=len(user_cols),
        column_types=column_types,  # pyright: ignore[reportArgumentType]  # runtime values match ColumnType literal set

        cardinality_ratio=cardinality_ratio,
        null_rate=null_rate,
        value_length_p50=value_length_p50,
        value_length_p99=value_length_p99,
    ))

# ── Column name heuristics ─────────────────────────────────────────────────

_NAME_PATTERNS = re.compile(
    r"(^name$|first.?name|last.?name|full.?name|fname|lname|surname|given.?name)",
    re.IGNORECASE,
)
_EMAIL_PATTERNS = re.compile(r"(email|e.?mail|email.?addr)", re.IGNORECASE)
_PHONE_PATTERNS = re.compile(r"(phone|tel|mobile|fax|cell)", re.IGNORECASE)
_ZIP_PATTERNS = re.compile(r"(zip|postal|postcode|zip.?code)", re.IGNORECASE)
_PRICE_PATTERNS = re.compile(r"(price|cost|amount|revenue|salary|fee|charge|total|balance)", re.IGNORECASE)
_ADDRESS_PATTERNS = re.compile(r"(address|street|addr|line.?1|line.?2)", re.IGNORECASE)
_GEO_PATTERNS = re.compile(r"((?<![a-z])city|^state$|state.?cd|^country$|province|region|(?<![a-z])county)", re.IGNORECASE)
_DATE_PATTERNS = re.compile(r"(date|_dt$|_date$|registr|created|updated|birth.?d|dob)", re.IGNORECASE)
_YEAR_PATTERNS = re.compile(r"(^|_)(year|yr)(_|$)", re.IGNORECASE)
_ID_PATTERNS = re.compile(
    r"^(?i:id|key|code|sku)$"                      # whole-name matches
    r"|_(?i:id|key)$"                              # *_id / *_key suffix
    r"|(?<=[a-zA-Z])(?:ID|Id)$"                    # CamelCase ID suffix (e.g. recordID)
    r"|(?i:^uuid$|^guid$|_uuid$|_guid$)"           # uuid/guid bounded
    r"|(?i:^uuid_|^guid_)"                         # uuid_*/guid_* prefix (e.g. guid_col)
    r"|^(?i:account_no|account_num)$"              # whole-name account identifiers
    r"|_(?i:ref|ref_num|reg_num|account_no|account_num|account)$"  # targeted ID-suffixes
)


@dataclass
class ColumnProfile:
    """Profile of a single column for auto-configuration."""

    name: str
    dtype: str
    col_type: str  # email, name, phone, zip, address, geo, identifier, description, numeric, date, string
    confidence: float  # 0.0 to 1.0
    sample_values: list[str] = field(default_factory=list)
    null_rate: float = 0.0  # fraction of nulls (0-1)
    cardinality_ratio: float = 0.0  # unique values / total rows (0-1), OVER THE SAMPLE
    full_cardinality_ratio: float | None = None
    # EXACT unique/total over the FULL frame, not the ~20K profiling sample. Only
    # populated for columns the sample calls perfectly unique (see
    # `profile_columns`); None everywhere else, including hand-built profiles.
    # Read it through `_is_perfect_surrogate`, never directly -- the surrogate-key
    # decision is the one place the sampled `cardinality_ratio` is not a usable
    # estimator, because a fixed-size sample's distinct-FRACTION climbs toward
    # 1.0 as the frame grows.
    avg_len: float = 0.0  # average string length
    n_distinct: int | None = None
    # Count of distinct non-blank values IN THE PROFILING SAMPLE. Distinct from
    # `cardinality_ratio`, which is that count divided by the row total and so
    # cannot answer "is this column CONSTANT?" -- a single-valued column reads
    # 1/n, which shrinks with the frame and is indistinguishable from a merely
    # low-cardinality one.
    #
    # This is a CANDIDATE signal only, never a verdict: the sample is 1,000 rows,
    # so a column that is 99.99% one value reads 1 here. Read `full_n_distinct`
    # to decide anything. ``None`` means the profile was built without value
    # access (hand-built profiles in tests, the #2639/#2644 idiom).
    full_n_distinct: int | None = None
    # EXACT distinct count over the FULL frame, computed only for columns the
    # sample flags as apparently-constant (`n_distinct <= 1`). Same shape as
    # `full_cardinality_ratio` and for the same reason: a sampled statistic
    # compared against an absolute cut makes the verdict flip with SCALE rather
    # than with the data, which the scale-invariant-correctness commitment
    # forbids. A value at frequency 1e-4 is missed by a 1,000-row sample ~90% of
    # the time, so without this a 2,000-row frame keeps a field that a 5M-row
    # frame drops. None everywhere else, including hand-built profiles.
    date_parse_rate: float | None = None
    # For col_type=="date": fraction of the profiled non-null sample that parses
    # as a well-formed date (the same parser `date_diff` uses). A per-column
    # reliability signal for the FS domain-comparator decision -- see
    # `build_probabilistic_matchkeys`. None for non-date columns or when the
    # profile was built without value access (e.g. hand-built profiles in tests).
    numeric_parse_rate: float | None = None
    # For col_type=="numeric": fraction of the sample that parses as a finite
    # float (the `numeric_diff` domain). Same reliability role as date_parse_rate.
    coord_parse_rate: float | None = None
    # For a coordinate-shaped column: fraction of the sample that parses as a
    # valid "lat,long" pair (the `geo_haversine` domain). None otherwise.


def _is_perfect_surrogate(p: ColumnProfile) -> bool:
    """True when every record carries its own value for this column -- a row PK,
    which asserts no shared identity and emits zero pairs under an exact match.

    THE single authority for the "#876 surrogate guard" question; four call sites
    used to ask it independently of the sampled ratio. Prefers the EXACT
    full-frame ratio, falling back to the sampled one only when the exact value
    was never computed (hand-built profiles in tests, callers with no full frame),
    which keeps the pre-existing verdict for those.

    Why the sampled ratio cannot answer this on its own: `cardinality_ratio` is a
    distinct-FRACTION over a fixed-size sample, so it rises toward 1.0 as the
    frame grows for ANY column whose distinct count grows with n -- which is every
    real identifier. The threshold is absolute, so the verdict flips with SCALE
    rather than with the data.
    """
    if p.full_cardinality_ratio is not None:
        return p.full_cardinality_ratio >= 1.0
    return (p.cardinality_ratio or 0.0) >= 1.0


@dataclass
class AutoConfigDecisions:
    """Fields marked 'not read yet' are wired up in subsequent tasks; they must not be read before then.

    Captures the *choices* auto_configure_df makes from profiled data.

    Separating decisions from the GoldenMatchConfig enables future iterative
    tuning: a future loop can nudge these decisions without re-profiling and
    then rebuild the config via `_rebuild_from_decisions`.

    Populated only by auto_configure_df; not persisted to YAML.
    """

    blocking_strategy: str
    blocking_keys: list[BlockingKeyConfig]
    blocking_passes: list[BlockingKeyConfig]
    matchkeys: list[MatchkeyConfig]
    threshold: float                # TODO(autoconfig-verify): consumed by postflight threshold nudge — not read yet
    domain_mode: str | None         # populated from detected DomainProfile.name; not read yet
    llm_enabled: bool               # TODO(autoconfig-verify): preflight Check 5 input — not read yet
    allow_remote_assets: bool       # TODO(autoconfig-verify): preflight Check 5 input — not read yet


def _classify_by_name(col_name: str) -> str | None:
    """Phase 1: classify column by name pattern matching.

    Order matters: ID and price before phone/zip to prevent data profiling
    from overriding name-based classification (e.g., 7-digit IDs as phones,
    5-digit prices as zips).
    """
    if _DATE_PATTERNS.search(col_name):
        return "date"
    if _YEAR_PATTERNS.search(col_name):
        return "year"
    if _EMAIL_PATTERNS.search(col_name):
        return "email"
    if _ID_PATTERNS.search(col_name):
        return "identifier"
    if _PRICE_PATTERNS.search(col_name):
        return "numeric"
    if _ZIP_PATTERNS.search(col_name):
        return "zip"
    if _GEO_PATTERNS.search(col_name):
        return "geo"
    if _ADDRESS_PATTERNS.search(col_name):
        return "address"
    if _PHONE_PATTERNS.search(col_name):
        return "phone"
    if _NAME_PATTERNS.search(col_name):
        return "name"
    return None


def _classify_by_data(values: list[str]) -> tuple[str, float]:
    """Phase 2: classify column by data profiling. Returns (type, confidence)."""
    if not values:
        return "string", 0.0

    data_type = _guess_type(values)

    # Cardinality guard: near-unique numeric-looking columns (phone/zip
    # lookalikes) are almost certainly identifiers. Scoping to numeric-shaped
    # types avoids reclassifying long text columns (titles, descriptions,
    # distinct names) as identifiers. Require a non-trivial sample (>=10)
    # so a handful of genuinely-unique zip/phone rows don't trip the guard.
    if data_type in ("phone", "zip", "numeric") and len(values) >= 10:
        cardinality_ratio = len(set(values)) / len(values)
        # S2a (spec 2026-06-22): identifier floor max(0.95, 1 - 1/sqrt(n)). At
        # scale the floor RISES above the old fixed 0.95 (a 10k-row 0.95-card
        # column is a high-entropy name, not an ID), and never drops below 0.95
        # so small-n behavior is unchanged (a looser small-n floor reclassified
        # moderately-unique phone/numeric columns and broke established matchkey
        # behavior). math.sqrt is correctly-rounded IEEE 754 -> bit-identical to
        # Rust's .sqrt() (do NOT use n ** 0.5 -- pow isn't correctly-rounded).
        floor = max(0.95, 1.0 - 1.0 / math.sqrt(len(values)))
        if cardinality_ratio >= floor:
            return "identifier", 0.9

    # Year detection: 4-digit integers in 1900..2100. Cheap blocking signal
    # for bibliographic / birth-year data (not full dates).
    def _is_year(v: str) -> bool:
        """True if v looks like a 4-digit year in 1900-2100, tolerating
        float-promoted integer columns (e.g. '1999.0')."""
        v = v.strip()
        try:
            n = int(float(v))
        except (ValueError, TypeError, OverflowError):
            # OverflowError: float('1e500') -> inf; int(inf) raises OverflowError.
            return False
        if not (1900 <= n <= 2100):
            return False
        # Round-trip check: stringified int must match v (with optional '.0').
        # Prevents '1.999e3' / '2001abc' from sneaking through.
        return str(n) == v.replace(".0", "").strip()

    if values and all(_is_year(v) for v in values):
        return "year", 0.9

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

    # Multi-value name detection: comma/semicolon-delimited text with
    # substantive length is almost always a co-author / multi-name field.
    # Catches this before the generic description branch so it gets routed
    # to token_sort rather than the embedding pathway.
    if col_type == "string":
        rows_with_delim = sum(1 for v in values if "," in v or ";" in v)
        delim_ratio = rows_with_delim / max(len(values), 1)
        if rows_with_delim > 0:
            avg_delims_in_delim_rows = sum(
                (v.count(",") + v.count(";")) for v in values if ("," in v or ";" in v)
            ) / rows_with_delim
        else:
            avg_delims_in_delim_rows = 0
        avg_len = sum(len(v) for v in values) / max(len(values), 1)
        if avg_len > 30 and delim_ratio >= 0.7 and avg_delims_in_delim_rows >= 2:
            return "multi_name", 0.7

    # Check for description (long freetext)
    if col_type == "string":
        avg_len = sum(len(v) for v in values) / len(values) if values else 0
        if avg_len > 50:
            col_type = "description"

    # Confidence based on how strongly data matches the type
    confidence = 0.7 if col_type != "string" else 0.3
    return col_type, confidence


# ── Arrow→polars dtype-spelling contract (autoconfig arrow-port PR-3) ──────────
# The native ``autoconfig_classify_columns`` kernel is fed ``str(dtype)`` and its
# golden vectors + the downstream string-column check
# (``p.dtype.startswith("String"/"Utf8")`` further down) expect the POLARS dtype
# spelling ("Float64"/"Int64"/"String"/"Boolean"). Polars renders those directly;
# an ``ArrowFrame``'s ``str(dtype)`` spells the SAME types differently
# ("double"/"int64"/"large_string"/"bool"). We keep the Rust kernel's vocabulary
# and map arrow→polars-spelling at this ONE Python boundary so the kernel sees its
# own vocabulary regardless of backend -- no Rust change, no golden-vector regen.
# A polars spelling passed in is returned UNCHANGED (identity), so the polars path
# is byte-for-byte what it was before the port.


def _polars_scalar_spelling(build: Any, fallback: str) -> str:
    """``str()`` of a polars dtype, matched exactly so the arrow path's classify
    input equals the polars path's. Falls back when polars is unavailable
    (pure-arrow install); the fallback is still accepted downstream."""
    try:
        from goldenmatch._polars_lazy import pl as _pl  # noqa: PLC0415

        return str(build(_pl))
    except Exception:  # noqa: BLE001 -- polars absent / API drift: use the fallback
        return fallback


# LAZY: computing these at MODULE level touches ``pl.Utf8`` / ``pl.Datetime`` on
# the _polars_lazy proxy, which eagerly imports polars at ``import goldenmatch``
# time -- a zero-polars-eviction gate violation (test_lazy_import_gate). Compute
# + cache on first USE instead (the arrow classify path, never at import).
_POLARS_SCALAR_SPELLING_CACHE: dict[str, str] = {}


def _polars_string_spelling() -> str:
    """``str(pl.Utf8)`` ("String" on modern polars, "Utf8" older), cached."""
    cached = _POLARS_SCALAR_SPELLING_CACHE.get("string")
    if cached is None:
        cached = _polars_scalar_spelling(lambda p: p.Utf8, "Utf8")
        _POLARS_SCALAR_SPELLING_CACHE["string"] = cached
    return cached


def _polars_datetime_spelling() -> str:
    """``str(pl.Datetime("us"))`` ("Datetime(time_unit='us', ...)"), cached."""
    cached = _POLARS_SCALAR_SPELLING_CACHE.get("datetime")
    if cached is None:
        cached = _polars_scalar_spelling(lambda p: p.Datetime("us"), "Datetime")
        _POLARS_SCALAR_SPELLING_CACHE["datetime"] = cached
    return cached

# Exact arrow-spelling → polars-spelling. Keys are arrow's ``str(type)`` (all
# lowercase); polars spellings are TitleCase, so they are never keys and pass
# through ``.get`` unchanged (the identity guarantee).
_ARROW_TO_POLARS_DTYPE: dict[str, str] = {
    # floats
    "double": "Float64",
    "float": "Float32",
    "halffloat": "Float32",
    # signed ints
    "int64": "Int64",
    "int32": "Int32",
    "int16": "Int16",
    "int8": "Int8",
    # unsigned ints
    "uint64": "UInt64",
    "uint32": "UInt32",
    "uint16": "UInt16",
    "uint8": "UInt8",
    # bool
    "bool": "Boolean",
    # date (day / ms variants both render as polars Date)
    "date32[day]": "Date",
    "date64[ms]": "Date",
    # null
    "null": "Null",
}


def _arrow_to_polars_dtype_spelling(raw: object) -> str:
    """Turn a concrete column dtype string into the polars spelling the native
    ``autoconfig_classify_columns`` kernel expects.

    Identity for polars spellings (they are never arrow keys), translation for
    arrow spellings, and ``str(raw)`` passthrough for any dtype not covered
    (documented fallback -- keeps the pre-port ``str(dtype)`` behavior for exotic
    types).
    """
    s = str(raw)
    mapped = _ARROW_TO_POLARS_DTYPE.get(s)
    if mapped is not None:
        return mapped
    # string family (arrow: string / large_string / utf8 / large_utf8)
    if s in ("string", "large_string", "utf8", "large_utf8"):
        return _polars_string_spelling()
    # timestamp[..] / date[..] families are unit-parametrized (not single keys)
    if s.startswith("timestamp"):
        return _polars_datetime_spelling()
    if s.startswith("date"):
        return "Date"
    # Unknown (incl. every polars spelling) -> unchanged.
    return s


def _concrete_dtype_spelling(frame: Any, col_name: str) -> str:
    """Read ``frame``'s concrete column dtype (native spelling per backend) and
    map it to the polars spelling for the classify kernel.

    - ``PolarsFrame`` -> its own ``str(dtype)`` (the map is identity, so this is
      byte-identical to the pre-port ``str(df[col].dtype)``).
    - ``ArrowFrame`` -> the pyarrow field type, mapped arrow→polars.

    NOTE: the seam ``Column`` Protocol exposes no concrete-dtype accessor (only the
    coarse ``semantic_dtype()``), so this reaches ``frame.native``. A clean seam op
    (e.g. ``Column.dtype_str()``) would drop the ``.native`` branch -- flagged for a
    PR-1-style follow-up.
    """
    from goldenmatch.core.frame import ArrowFrame  # noqa: PLC0415

    if isinstance(frame, ArrowFrame):
        raw = str(frame.native.schema.field(col_name).type)
    else:
        raw = str(frame.native[col_name].dtype)
    return _arrow_to_polars_dtype_spelling(raw)


@dataclass(frozen=True)
class ExactStats:
    """Column statistics measured over the FULL population, not a sample.

    Supplied by a caller that can see the whole dataset when `profile_columns`
    cannot -- the Spark path, which hands it a 20,000-row driver sample.

    ``n_distinct`` and ``avg_len`` are ``None`` when the measurement was not
    taken. Absent is NOT zero: a consumer must keep the sampled value rather
    than read a missing measurement as a measured one.
    """

    n_rows: int
    n_non_null: int
    n_distinct: int | None = None
    avg_len: float | None = None


#: Cluster-computed statistics for the frame currently being profiled.
#:
#: `profile_columns` runs an exact confirm pass over the frame it is HANDED and
#: stores it as `full_n_distinct`, documented as "the EXACT full-frame count".
#: On the Spark path that label is false: the frame is a 20,000-row sample of a
#: possibly 500M-row table, and the drop-constant rule reads the result as
#: though it described the population.
#:
#: A ContextVar rather than a parameter because `profile_columns` has four call
#: sites inside the controller and threading an optional argument through all of
#: them would touch far more surface than the behaviour warrants. Same idiom as
#: `_LAST_CONTROLLER_RUN` above and the profile-emitter stack.
_EXACT_COLUMN_STATS: ContextVar = ContextVar("_EXACT_COLUMN_STATS", default=None)


@contextmanager
def exact_column_stats_applied(stats: dict) -> Any:
    """Apply cluster-measured column statistics for the duration of the block.

    Scoped and reset on exit including on exception, so one run's population
    statistics can never leak onto the next caller's frame.
    """
    token = _EXACT_COLUMN_STATS.set(stats)
    try:
        yield
    finally:
        _EXACT_COLUMN_STATS.reset(token)


def _apply_exact_stats(profiles: list) -> None:
    """Overwrite the DISTRIBUTION statistics from the cluster's measurement.

    Four fields move: `null_rate`, `avg_len`, `n_distinct` / `full_n_distinct`
    and the `cardinality_ratio` derived from it. Those are the ones compared
    against ABSOLUTE thresholds, so a sample measures them wrong in a way that
    worsens with scale -- #876 (a flat 0.28 reading 0.72 -> 1.00 across
    100K..5M) and #2687 (a 99.99%-constant column read as constant).

    `col_type`, `confidence` and `sample_values` are never touched: they answer
    "what kind of column is this", which a sample answers correctly.
    """
    stats = _EXACT_COLUMN_STATS.get()
    if not stats:
        return
    for p in profiles:
        st = stats.get(p.name)
        if st is None:
            continue
        if st.n_rows > 0:
            p.null_rate = (st.n_rows - st.n_non_null) / st.n_rows
        if st.avg_len is not None:
            p.avg_len = float(st.avg_len)
        if st.n_distinct is not None:
            p.n_distinct = int(st.n_distinct)
            # `full_n_distinct` is what the drop-constant rule reads, and it is
            # the field whose "full-frame" contract the Spark path breaks.
            p.full_n_distinct = int(st.n_distinct)
            if st.n_rows > 0:
                p.cardinality_ratio = st.n_distinct / st.n_rows


def profile_columns(
    df: pl.DataFrame, sample_size: int = 1000, max_columns: int = 40,
    llm_provider: str | None = None,
) -> list[ColumnProfile]:
    """Classify columns by type using name heuristics + data profiling.

    Samples randomly to avoid bias from header-adjacent rows.
    Wide datasets (>max_columns) are trimmed: columns matching known patterns
    (name, email, phone, zip, address) are prioritized, then remaining columns
    fill up to the cap.
    """
    # Route through the Frame seam so profile_columns runs on a polars frame
    # (today) OR an ArrowFrame (post-PR-6) -- `to_frame` is idempotent and accepts
    # pl.DataFrame / pa.Table / Frame. (W3c: polars rows unchanged; sample is exact
    # delegation, shuffle=False default matches the old call.)
    from goldenmatch.core.frame import to_frame

    frame = to_frame(df)

    if frame.height > sample_size:
        sample_frame = frame.sample(sample_size, seed=42)
    else:
        sample_frame = frame

    # For wide datasets, prioritize columns likely useful for matching
    columns = [c for c in frame.columns if not c.startswith("__")]
    if len(columns) > max_columns:
        # Phase 1: keep columns matching known patterns
        priority = []
        rest = []
        for col_name in columns:
            if _classify_by_name(col_name) is not None:
                priority.append(col_name)
            else:
                rest.append(col_name)
        # Fill remaining slots from unmatched columns
        remaining_slots = max(0, max_columns - len(priority))
        columns = priority + rest[:remaining_slots]
        logger.info(
            "Wide dataset (%d columns), auto-configure limited to %d columns "
            "(%d pattern-matched, %d additional)",
            len(frame.columns), len(columns), len(priority), remaining_slots,
        )

    # Collect per-column stats using Polars (shared by both the native and
    # pure-Python paths — the native path delegates ONLY the classify step).
    col_stats: list[tuple[str, str, list[str], float, float, float]] = []
    # RAW non-null sample values per column (blanks/whitespace RETAINED, unlike
    # the classify `values` below which drops them). The date reliability signal
    # is computed from this so a blank -- a non-null, unparseable value -- lowers
    # the parse rate instead of being silently excluded (Copilot #2337).
    nonnull_values_by_name: dict[str, list[str]] = {}
    for col_name in columns:
        if col_name.startswith("__"):
            continue

        # Concrete column dtype in the POLARS spelling the native classify kernel
        # expects, regardless of backend: identity on a PolarsFrame (byte-identical
        # to the old `str(df[col].dtype)`), arrow→polars-mapped on an ArrowFrame.
        dtype = _concrete_dtype_spelling(frame, col_name)

        col_series = sample_frame.column(col_name)
        total_rows = len(col_series)
        null_count = col_series.null_count()
        null_rate = null_count / total_rows if total_rows > 0 else 0.0

        nonnull_values = [str(v) for v in col_series.drop_nulls().to_list()]
        nonnull_values_by_name[col_name] = nonnull_values
        values = [v for v in nonnull_values if v.strip()]

        cardinality_ratio = len(set(values)) / total_rows if total_rows > 0 else 0.0
        avg_len = sum(len(v) for v in values) / len(values) if values else 0.0

        col_stats.append((col_name, dtype, values, null_rate, cardinality_ratio, avg_len))

    from goldenmatch.core._native_loader import native_enabled, native_module  # noqa: PLC0415

    _use_native = native_enabled("autoconfig")
    _nm = native_module() if _use_native else None
    _native_classify_available = (
        _use_native and _nm is not None and hasattr(_nm, "autoconfig_classify_columns")
    )

    if _native_classify_available:
        # Native batch path: send all column stats to Rust, reconstruct profiles.
        from goldenmatch.core.autoconfig_native import (  # noqa: PLC0415
            column_profiles_from_json,
            column_stats_to_json,
        )
        stats_dicts = [
            {
                "name": col_name,
                "dtype": dtype,
                "sample_values": values,  # FULL non-null list
                "null_rate": null_rate,
                "cardinality_ratio": cardinality_ratio,
                "avg_len": avg_len,
            }
            for col_name, dtype, values, null_rate, cardinality_ratio, avg_len in col_stats
        ]
        names_to_sample_values = {
            col_name: values
            for col_name, _dtype, values, _nr, _cr, _al in col_stats
        }
        native_json = _nm.autoconfig_classify_columns(  # type: ignore[union-attr]
            column_stats_to_json(stats_dicts)
        )
        profiles: list[ColumnProfile] = column_profiles_from_json(
            native_json, names_to_sample_values
        )
        # `n_distinct` is stamped here rather than carried through the Rust
        # classify contract: it is a property of the sample the caller already
        # holds, not a classification result, so widening the JSON schema for it
        # would couple the kernel to a field it has no opinion about. Stamping
        # both branches from the same `values` keeps native and fallback
        # byte-identical on this field, which the parity rules require.
        for _p in profiles:
            _vals = names_to_sample_values.get(_p.name)
            if _vals is not None:
                _p.n_distinct = len({v for v in _vals if v.strip()})
    else:
        # Pure-Python classification path (unchanged).
        profiles = []
        for col_name, dtype, values, null_rate, cardinality_ratio, avg_len in col_stats:
            # Phase 1: name heuristics
            name_type = _classify_by_name(col_name)

            # Phase 2: data profiling
            data_type, data_confidence = _classify_by_data(values)

            # Combine: name heuristics are authoritative for structural types
            # (date, geo, zip) because data profiling frequently misclassifies them
            # (e.g., ISO dates look like phone numbers, city names look like person
            # names, and ZIP+4 codes like '10001-3904' look like phone numbers).
            # A zip misclassified as phone would back an exact matchkey and
            # over-merge same-address records, so trust the name.
            # For other types, Phase 2 (data) wins when it contradicts Phase 1 (name).
            _name_authoritative = {"date", "geo", "identifier", "numeric", "year", "zip"}
            if name_type and name_type in _name_authoritative:
                # Name pattern is authoritative for date/geo — trust it
                col_type = name_type
                confidence = 0.9
            elif name_type and data_type != "string":
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
                null_rate=null_rate,
                cardinality_ratio=cardinality_ratio,
                avg_len=avg_len,
                n_distinct=len(set(values)),
            ))

    # LLM correction pass for ambiguous columns (runs AFTER native classify,
    # on the same filtered set the Python path uses — unchanged).
    if llm_provider and profiles:
        profiles = _llm_classify_columns(profiles, llm_provider)

    # EXACT full-frame cardinality for APPARENT surrogate keys (both classify
    # paths converge here, so the native path is covered too).
    #
    # `cardinality_ratio` above is a distinct-FRACTION over the ~20K profiling
    # sample, and a fixed-size sample of a GROWING frame drives that fraction to
    # 1.0 for any column whose distinct count grows with n. Measured on the QIS
    # realistic shape, an `email` column whose true ratio is a flat 0.28 reads
    # 0.72 / 0.94 / 0.97 / 0.98 / 1.00 at 100K / 500K / 1M / 2M / 5M. Every
    # "#876 surrogate guard" compares that number to an absolute 1.0, so at 5M
    # zero-config discarded its ONLY exact identity column as a "perfectly-unique
    # surrogate key", fell back to fuzzy-only matchkeys and name-based blocking,
    # and produced a 185M-pair run that took 66 GB and killed the CI runner. The
    # verdict flipped with SCALE, not with the data -- the exact thing the
    # scale-invariant-correctness commitment forbids.
    #
    # Restricting the exact count to sample-ratio-1.0 columns is SOUND, not a
    # cost heuristic: if every value in the full column is distinct then so is
    # every subset of it, hence full == 1.0 implies sample == 1.0. Anything below
    # 1.0 in the sample cannot be a perfect surrogate and is skipped. (Nulls make
    # the sampled ratio smaller, so a null-bearing column skips the check and
    # keeps its previous verdict -- conservative in the safe direction, since the
    # fallback can only RETAIN a column.) In practice this is one or two columns;
    # measured 44-307 ms each at 5M rows against a ~160s auto-config.
    _frame_cols = set(frame.columns)
    for p in profiles:
        if p.cardinality_ratio >= 1.0 and p.name in _frame_cols and frame.height > 0:
            p.full_cardinality_ratio = frame.column(p.name).n_unique() / frame.height

    # EXACT full-frame distinct count for APPARENTLY-CONSTANT columns (#2668).
    # The mirror image of the pass above, and forbidden by the same commitment:
    # `n_distinct` is a 1,000-row sample statistic, so a column that is 99.99%
    # one value reads 1 and `_drop_constant_scored_fields` would drop its field
    # from the matchkey. A value at frequency 1e-4 is missed by that sample ~90%
    # of the time, so the verdict would flip with scale -- a small frame samples
    # the column fully and keeps the field, a large one reads it constant and
    # drops it.
    #
    # The damage runs the same direction as the bug this rule exists to fix. A
    # field that agrees on nearly every pair and disagrees only on the rare ones
    # is carrying its whole signal in exactly those rare pairs; dropping it
    # removes the penalty where it discriminates and the rare cross-entity pair
    # merges.
    #
    # Restricting the exact count to sample-flagged columns is sound rather than
    # a cost heuristic: a column with >1 distinct value in the sample has >1 in
    # the full frame, so it cannot be constant and is skipped. In practice this
    # is a handful of columns, one `n_unique()` pass each.
    for p in profiles:
        if (p.n_distinct is not None and p.n_distinct <= 1
                and p.name in _frame_cols and frame.height > 0):
            p.full_n_distinct = int(frame.column(p.name).n_unique())

    # The CLUSTER's measurement wins over anything derived from this frame --
    # on the Spark path this frame is a 20k sample and the confirm above is
    # exact over the sample, not the population.
    _apply_exact_stats(profiles)

    # Per-column date reliability signal (both classify paths converge here, so
    # the native path is covered too). For every `date` column, record the
    # fraction of its profiled sample that parses as a well-formed date -- the
    # FS domain-comparator decision reads this to keep magnitude-aware `date_diff`
    # off messy/low-parse date columns (where it over-penalizes true matches with
    # noisy dates) while using it on clean ones. Computed over the SAME ~1000-row
    # sample `values` (not the 5 stored `sample_values`, which are too few).
    _date_profiles = [p for p in profiles if p.col_type == "date"]
    _numeric_profiles = [p for p in profiles if p.col_type == "numeric"]
    _coord_profiles = [p for p in profiles if _looks_like_latlong(p)]
    if _date_profiles or _numeric_profiles or _coord_profiles:
        from goldenmatch.core.scorer import (  # noqa: PLC0415
            _parse_date_ordinal,
            _parse_float,
            _parse_latlong,
        )

        def _rate(name: str, parse: Callable[[str], object | None]) -> float:
            # Rate over RAW non-null values (blanks retained): a blank is a
            # non-null unparseable value that must LOWER the rate, not be filtered
            # out (else a mostly-blank messy column looks reliable). Explicit 0.0
            # when there are no non-null samples -- never None on the auto-config
            # path, so the reliability gate can't read an empty column as reliable.
            vals = nonnull_values_by_name.get(name, [])
            if not vals:
                return 0.0
            return sum(1 for v in vals if parse(v) is not None) / len(vals)

        for p in _date_profiles:
            p.date_parse_rate = _rate(p.name, _parse_date_ordinal)
        for p in _numeric_profiles:
            p.numeric_parse_rate = _rate(p.name, _parse_float)
        for p in _coord_profiles:
            p.coord_parse_rate = _rate(p.name, _parse_latlong)

    return profiles


def _llm_classify_columns(
    profiles: list[ColumnProfile], provider: str,
) -> list[ColumnProfile]:
    """Use LLM to correct ambiguous column classifications and rank match fields.

    Only sends columns with low confidence or generic types (string, numeric).
    High-confidence classifications (date, geo, email, identifier) are trusted.
    """
    import json as _json
    import urllib.error

    # Filter to ambiguous profiles
    high_confidence_types = {"date", "geo", "email", "identifier"}
    ambiguous = [
        p for p in profiles
        if p.confidence < 0.8 or p.col_type in ("string", "numeric")
        if p.col_type not in high_confidence_types
    ]

    if not ambiguous:
        return profiles

    # Build prompt
    col_lines = []
    for p in ambiguous:
        samples = ", ".join(p.sample_values[:5]) if p.sample_values else "no samples"
        col_lines.append(f'  "{p.name}": [{samples}]')

    all_col_names = [p.name for p in profiles if p.col_type not in high_confidence_types]

    prompt = (
        "You are classifying database columns for entity matching/deduplication.\n\n"
        "For each column below, provide:\n"
        '1. "type": one of: identifier, name, description, numeric, date, geo, '
        "email, phone, zip, address, price, string\n"
        '2. "match_rank": rank the top 5 columns most useful for entity matching '
        "(1=most useful). Only rank columns that would help identify duplicate records.\n\n"
        "Columns with sample values:\n"
        + "\n".join(col_lines)
        + "\n\nAll columns available for ranking: " + ", ".join(all_col_names)
        + '\n\nRespond in JSON: {"classifications": {"col_name": "type", ...}, '
        '"match_ranking": ["col1", "col2", "col3", "col4", "col5"]}'
    )

    try:
        raw = _call_llm_for_blocking(prompt, provider)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as e:
        logger.warning("LLM column classification failed: %s. Using heuristics only.", e)
        return profiles

    # Parse response
    try:
        # Extract JSON from response (may be wrapped in markdown)
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = _json.loads(text)
    except (ValueError, IndexError) as e:
        logger.warning(
            "LLM column classification returned unparseable response (error: %s). "
            "Raw response (first 200 chars): %.200s", e, raw,
        )
        return profiles

    # Normalize type aliases
    _type_aliases = {
        "id": "identifier", "ids": "identifier",
        "desc": "description", "text": "string",
        "location": "geo", "city": "geo", "state": "geo",
        "postal": "zip", "postcode": "zip",
        "cost": "numeric", "price": "numeric", "amount": "numeric",
        "tel": "phone", "telephone": "phone",
    }
    valid_types = {
        "identifier", "name", "description", "numeric", "date", "geo",
        "email", "phone", "zip", "address", "string",
    }

    # Apply type corrections
    classifications = data.get("classifications", {})
    profile_by_name = {p.name: p for p in profiles}
    for col_name, llm_type in classifications.items():
        if col_name not in profile_by_name:
            continue
        if not isinstance(llm_type, str):
            continue
        p = profile_by_name[col_name]
        # Only correct ambiguous columns
        if p.col_type in high_confidence_types:
            continue
        normalized = _type_aliases.get(llm_type.lower(), llm_type.lower())
        if normalized in valid_types:
            logger.info("LLM reclassified '%s': %s -> %s", col_name, p.col_type, normalized)
            p.col_type = normalized
            p.confidence = 0.85

    # Apply match ranking (stored as metadata for build_matchkeys to use)
    match_ranking = data.get("match_ranking", [])
    if match_ranking:
        # Store ranking as a special attribute on profiles
        for rank, col_name in enumerate(match_ranking[:5]):
            if col_name in profile_by_name:
                # Use a high utility boost so LLM-ranked fields sort first
                p = profile_by_name[col_name]
                p.cardinality_ratio = max(p.cardinality_ratio, 0.9 - rank * 0.1)
                p.avg_len = max(p.avg_len, 40 - rank * 5)
        logger.info("LLM match ranking: %s", match_ranking[:5])

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

# ── Fellegi-Sunter auto-config v2 (comparison-set + blocking curation) ──
# Scoped to the PROBABILISTIC path only (auto_configure_probabilistic_df); the
# weighted/DQbench path is untouched. Default ON; GOLDENMATCH_FS_AUTOCONFIG_V2=0
# restores the legacy field set. See build_probabilistic_matchkeys docstring +
# the historical_50k PII-parity audit for rationale.
_PROB_FUZZY_CARD_FLOOR = 0.01

_ATOMIC_GIVEN_NAMES = frozenset({
    "first_name", "firstname", "first", "given_name", "givenname", "given",
    "forename", "fname",
})
_ATOMIC_FAMILY_NAMES = frozenset({
    "surname", "last_name", "lastname", "last", "family_name", "familyname",
    "family", "lname",
})
# Person-name composites that duplicate the atomic given/family signal. Dropped
# only when BOTH an atomic given and family field are present (so a dataset with
# just `full_name` keeps it).
_COMPOSITE_NAME_FIELDS = frozenset({
    "full_name", "fullname", "name", "first_and_surname", "firstandsurname",
    "first_and_last", "name_full", "complete_name", "whole_name",
    "display_name", "displayname", "first_last", "given_and_surname",
})


def _norm_colname(name: str) -> str:
    """Normalize a column name for atomic/composite matching."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _fs_autoconfig_v2_enabled() -> bool:
    """FS auto-config v2 comparison-set + blocking curation.

    **Default ON (2026-06-09).** ``GOLDENMATCH_FS_AUTOCONFIG_V2=0`` (or
    ``false``/``off``/``no``/``disabled``) restores the legacy field set. Flipped
    on after the curated set beat Splink on every measurable dataset and the
    DBLP-ACM mega-match was fixed (lever #4) — no remaining measured regression.

    MEASURED (scripts/bench_er_headtohead, GM probabilistic vs Splink, one
    evaluator) — v2 beats Splink on every PII set AND unbreaks bibliographic:
      historical_50k F1 0.624 -> 0.779 (Splink 0.757)
      febrl3         F1 0.983 -> 0.991 (Splink 0.965)
      synthetic      F1 0.972 -> 0.998 (Splink 0.996)
      dblp_acm       F1 0.003 -> 0.879 (auto-config was a venue-only mega-match)
    The `venue` low-card-floor worry did NOT materialize (venue card 0.010 >
    floor 0.01). The panel-v1-v2 lane in bench-probabilistic.yml is the standing
    v1-vs-v2 regression check.
    """
    return os.environ.get("GOLDENMATCH_FS_AUTOCONFIG_V2", "1").lower() not in (
        "0", "false", "off", "no", "disabled",
    )


def _fs_domain_comparators_enabled() -> bool:
    """FS domain comparators (spec 2026-07-23-fs-domain-comparators-design.md).

    **Default OFF.** When ``GOLDENMATCH_FS_DOMAIN_COMPARATORS`` is truthy, FS
    auto-config admits ``date`` columns with the magnitude-aware ``date_diff``
    scorer instead of ``levenshtein`` (edit-distance can't see that a 1-year DOB
    gap is a weak partial). Composes with ``_fs_autoconfig_v2_enabled`` (v2 is
    the path that admits dates at all). Default-off is byte-identical to today;
    flip only after the accuracy panel + qis_gate scale-neutrality prove it, per
    the ``GOLDENMATCH_FS_AUTOCONFIG_V2`` flag precedent. Phase 1 = ``date_diff``;
    Phase 2 also admits ``numeric`` columns as ``numeric_diff`` and single
    combined ``lat,long`` columns as ``geo_haversine`` (see
    ``build_probabilistic_matchkeys``).
    """
    return os.environ.get("GOLDENMATCH_FS_DOMAIN_COMPARATORS", "0").lower() in (
        "1", "true", "on", "yes", "enabled",
    )


# FS domain-comparator per-column DECISION (spec 2026-08-01-fs-lever-enablement,
# Phase 2). The global GOLDENMATCH_FS_DOMAIN_COMPARATORS flip is a net LOSS on
# the panel because magnitude-aware `date_diff` HELPS clean date columns but
# HURTS messy ones -- a true match whose recorded dates differ by months/years
# (noisy/historical DOBs) drops from a near-match (levenshtein ~0.90) to a weak
# partial (date_diff 0.60), costing recall. So `date_diff` is emitted per-column
# only when the date column is RELIABLE, keeping `levenshtein` when it is not.
# The reliability proxy is the column's date PARSE RATE (fraction of the profiled
# sample that parses as a well-formed date): a low parse rate marks a messy date
# column where magnitude banding over-penalizes. Calibrated against the corpora
# (like `_pick_missing_semantics`), NOT derived: measured ab_lever ΔF1 for
# date_diff is febrl3 +0.0014 (parse 0.993) / ncvr flat (1.000) vs historical_50k
# -0.0130 (parse 0.926); 0.95 sits inside the separating band so the messy
# genealogy DOB keeps levenshtein while the clean columns get date_diff.
_DATE_DIFF_MIN_PARSE_RATE = 0.95


def _date_column_reliable_for_diff(p: ColumnProfile) -> bool:
    """Should a `date` column use the magnitude-aware `date_diff` scorer?

    True when the column's `date_parse_rate` clears `_DATE_DIFF_MIN_PARSE_RATE`.
    A missing signal (``None`` -- e.g. a hand-built profile in tests, or a
    profile constructed without value access) is treated as reliable so the
    flag-on behavior is unchanged for callers that don't carry the rate; the
    real auto-config path always populates it for date columns.
    """
    return p.date_parse_rate is None or p.date_parse_rate >= _DATE_DIFF_MIN_PARSE_RATE


# The numeric_diff / geo_haversine half of the per-column domain-comparator
# decision. Unlike date_diff (a MEASURED panel regression, −0.0130 on
# historical_50k), this is a DEFENSIVE consistency gate, NOT a measured-F1 fix:
# the panel carries no lat/long column and only one clean numeric column, so
# there's no panel regression to calibrate against. Both scorers already
# exact-string fall back per-value on unparseable input, so the gate can only
# withhold a column that mostly DOESN'T parse as its domain (don't magnitude-band
# a "numeric"/"coord" column that isn't one) -- the dispersed-but-parseable
# failure mode (the date-gate's known limit) is out of a parse-rate signal's
# reach here too. The unreliable-else is the conservative pre-flag v2 default
# (the column simply isn't admitted as that comparator). Same 0.95 "parses
# cleanly" bar as the date cut (spec wording), not independently calibrated.
_NUMERIC_GEO_MIN_PARSE_RATE = 0.95


def _numeric_column_reliable_for_diff(p: ColumnProfile) -> bool:
    """Should a `numeric` column be admitted as a `numeric_diff` field? True when
    `numeric_parse_rate` clears the bar; ``None`` (no signal) is reliable-by
    default so hand-built profiles keep the flag-on behavior."""
    return p.numeric_parse_rate is None or p.numeric_parse_rate >= _NUMERIC_GEO_MIN_PARSE_RATE


def _coord_column_reliable_for_haversine(p: ColumnProfile) -> bool:
    """Should a coordinate-shaped column be admitted as a `geo_haversine` field?
    True when `coord_parse_rate` clears the bar; ``None`` is reliable-by-default."""
    return p.coord_parse_rate is None or p.coord_parse_rate >= _NUMERIC_GEO_MIN_PARSE_RATE


# Discrete-categorical col_types where a match on a COMMON value ("Smith", a
# frequent occupation/city/postcode) is far weaker evidence of identity than a
# match on a rare one -- the regime Winkler term-frequency (TF) adjustment is
# built for. `string` (occupation-like), `geo` (city/birthplace), and `zip`
# (postcode) carry NO existing frequency-awareness (contrast `name`, which the
# refdata hook routes to `name_freq_weighted_jw` whose per-value downweight
# already fires); `name` is included so a plain (unrefined) name field is still
# covered, and the TF bump self-neutralizes (log2(collision/fv) -> 0) where a
# refined scorer has already flattened the top-band agreement. email/phone/
# identifier are excluded: near-unique identity values where TF is inert anyway.
_TF_ELIGIBLE_COLTYPES = frozenset({"name", "string", "geo", "zip"})

# Above this cardinality a categorical's values are near-unique, so the TF bump
# is ~0; skipping saves the per-value frequency table build with no behavior
# change (TF self-neutralizes there regardless).
_TF_CARD_CEILING = 0.9


def _fs_tf_adjustment_enabled() -> bool:
    """FS term-frequency (Winkler) adjustment on skewed categorical fields.

    **Default OFF.** When ``GOLDENMATCH_FS_TF_ADJUSTMENT`` is truthy, FS
    auto-config sets ``MatchkeyField.tf_adjustment=True`` on discrete-categorical
    comparison fields (``_TF_ELIGIBLE_COLTYPES`` in a skewed cardinality band), so
    an exact agreement on a RARE value out-weights one on a common value
    (``+log2(Sum(freq^2)/freq(value))``, clamped +/-10 bits; frequencies built at
    EM-train time by ``_build_tf_tables``). The Winkler machinery already exists
    (``probabilistic.py``) and is native-accelerated (``FS_SUPPORTS_TF_ADJUSTMENT``
    in the kernel), but auto-config never enabled it -- so no zero-config FS run
    used it. It targets the OVER-MERGE regime (historical_50k: GM precision 0.72
    vs Splink 0.97) that skewed common-value agreement drives; the prior
    "no headroom" measurement was on DBLP-ACM (near-unique titles) and Febrl4
    (precision already ~1.0), neither of which is this regime. Default-off is
    byte-identical (no field gets the flag); flip only after the accuracy panel +
    ``qis_gate`` scale-neutrality prove it, per the domain-comparators/v2
    precedent. Composes with ``_fs_autoconfig_v2_enabled`` (the path that builds
    the categorical comparison set).
    """
    return os.environ.get("GOLDENMATCH_FS_TF_ADJUSTMENT", "0").lower() in (
        "1", "true", "on", "yes", "enabled",
    )


def _tf_adjustment_for(profile: ColumnProfile) -> bool:
    """Whether a categorical comparison field earns TF adjustment, given the
    ``GOLDENMATCH_FS_TF_ADJUSTMENT`` flag. Off by default -> always False ->
    byte-identical. Scoped to skewed-value discrete categoricals; see
    ``_TF_ELIGIBLE_COLTYPES`` / ``_fs_tf_adjustment_enabled``."""
    return (
        _fs_tf_adjustment_enabled()
        and profile.col_type in _TF_ELIGIBLE_COLTYPES
        and _PROB_FUZZY_CARD_FLOOR <= profile.cardinality_ratio < _TF_CARD_CEILING
    )


# Name column types that carry honorific/title tokens ("Sir", "Baronet") worth
# stripping before FS scores them. Person-name columns classify to "name"; a
# composite first+last field classifies to "multi_name".
_STRIP_HONORIFIC_COLTYPES = frozenset({"name", "multi_name"})


def _fs_strip_honorifics_enabled() -> bool:
    """FS honorific-token stripping on name comparison fields.

    **Default OFF.** When ``GOLDENMATCH_FS_STRIP_HONORIFICS`` is truthy, FS
    auto-config appends the ``strip_honorifics`` transform to name-typed
    comparison fields, so a title/rank token leaked into a name field ("Sir",
    "Baronet", "Bt.") stops carrying match weight. Targets the OVER-MERGE regime
    that TF down-weighting could NOT reach on historical_50k (byte-neutral there:
    the name coltype already routes through ``name_freq_weighted_jw``, so TF
    self-neutralizes; the honorifics are the residual it doesn't cut). The
    honorific tokens are a curated title/rank set (``strip_honorifics`` in
    ``utils/transforms.py``); regnal numerals are kept (the A/B showed keeping
    them recovers recall at no precision cost). Default-off is byte-identical (no
    field gets the transform); flip only after the accuracy panel proves it, per
    the domain-comparators/TF precedent.

    Spike A/B (historical_50k, panel-honorific lane): F1 0.7520 -> 0.7628
    (+0.0108), precision +0.0245, recall -0.0046; GM overtakes Splink (0.7571).
    """
    return os.environ.get("GOLDENMATCH_FS_STRIP_HONORIFICS", "0").lower() in (
        "1", "true", "on", "yes", "enabled",
    )


def _strip_honorifics_for(profile: ColumnProfile) -> bool:
    """Whether a comparison field earns the ``strip_honorifics`` transform, given
    the ``GOLDENMATCH_FS_STRIP_HONORIFICS`` flag. Off by default -> always False
    -> byte-identical. Scoped to name-typed columns; see
    ``_STRIP_HONORIFIC_COLTYPES`` / ``_fs_strip_honorifics_enabled``."""
    return (
        _fs_strip_honorifics_enabled()
        and profile.col_type in _STRIP_HONORIFIC_COLTYPES
    )


# Coverage / cardinality bands for the "unused orthogonal field" blocking rule
# (`_diversify_unused_orthogonal_blocking`). A field earns an additive single-field
# blocking pass only when it is WELL-POPULATED (null rate under the ceiling) AND
# sits in a MODERATE cardinality band -- neither near-constant (a gender-like field
# at ~0.002 makes two giant blocks) nor near-unique (a surrogate key makes
# singletons with no co-blocking value). Field-agnostic BY DESIGN: the win on
# historical_50k is `birth_place` (an orthogonal anchor for the ~22% of true pairs
# whose corrupted names never co-block), but the rule NEVER names it -- selection is
# by data shape so it generalizes to whatever orthogonal field a dataset carries.
_ORTHO_BLOCK_NULL_CEILING = 0.4
_ORTHO_BLOCK_CARD_FLOOR = 0.02
_ORTHO_BLOCK_CARD_CEILING = 0.9
# A candidate anchor is admitted only if it is ORTHOGONAL to the passes that
# already block -- measured as the fraction of its same-key row pairs that are
# ALREADY co-blocked by some existing pass. A HIGH overlap means the field
# re-blocks a signal the passes already cover (the classic case: an ATOMIC name
# field -- first_name/surname -- when the name COMPOSITES already block; those
# atomic fields pass the shape filter but their same-value pairs are ~0.8-0.94
# already-co-blocked). Adding them widens the same-name candidate set the FS
# scorer can't reject, over-merging on error-heavy data (historical_50k precision
# 0.91->0.75). A genuinely orthogonal anchor (birth_place: overlap ~0.29) brings
# NEW co-blocking and is the intended win. 0.5 cleanly separates the two regimes.
_ORTHO_BLOCK_OVERLAP_CEILING = 0.5
# Bounded head sample + pair cap for the overlap measurement (config-time, once).
_ORTHO_OVERLAP_SAMPLE = 4000
_ORTHO_OVERLAP_PAIR_CAP = 400_000


# Col_types that mark a NAME field for the person-shape gate. A person identity
# table pairs name fields with a strong temporal anchor (date of birth); that
# anchor is exactly what keeps the enlarged orthogonal blocks SAFE -- within a
# birth_place/dob block, name+dob still discriminate, so the topic-bucket
# over-merge that sinks bibliographic/product data (dblp_scholar/amazon_google:
# no `name`, no `date` -- title=description, authors=string, year=year) can't
# happen. Composite first+last fields classify `multi_name`.
_PERSON_NAME_COLTYPES = frozenset({"name", "multi_name"})


def _dataset_is_person_shaped(profiles: list[ColumnProfile]) -> bool:
    """True when the profile set looks like a PERSON identity table -- a name field
    AND a date field both present.

    This is the AUTO-enable condition for orthogonal-anchor blocking. Person data is
    where the lever generalises (historical_50k / febrl / synthetic_person all carry
    name+date and win out-of-panel); bibliographic/product data carries neither a
    ``name`` nor a ``date`` col_type and is where the lever regresses, so the ``name
    AND date`` signature cleanly separates the two validated regimes. Conservative by
    design -- a false negative merely forgoes the gain, a false positive risks the
    dblp_scholar-style precision collapse, so the gate errs toward NOT firing.
    """
    types = {p.col_type for p in profiles}
    return bool(types & _PERSON_NAME_COLTYPES) and "date" in types


def _fs_orthogonal_blocking_mode() -> str:
    """Resolve ``GOLDENMATCH_FS_ORTHOGONAL_BLOCKING`` to ``auto`` | ``on`` | ``off``.

    - ``auto`` (**default**): fire ONLY on person-shaped data
      (``_dataset_is_person_shaped``) -- the dataset-type gate.
    - ``on`` (``1``/``true``/``yes``/``enabled``): force everywhere (the pre-gate
      behavior; useful to force it onto a person dataset a classifier misjudged).
    - ``off`` (``0``/``false``/``no``/``disabled``): never fire (byte-identical to
      the pre-lever field set).

    Mirrors the ``GOLDENMATCH_NATIVE=auto/0/1`` tri-state idiom.
    """
    v = os.environ.get("GOLDENMATCH_FS_ORTHOGONAL_BLOCKING", "auto").lower()
    if v in ("1", "true", "on", "yes", "enabled"):
        return "on"
    if v in ("0", "false", "off", "no", "disabled"):
        return "off"
    return "auto"


def _fs_atomic_name_blocking_mode() -> str:
    """Resolve ``GOLDENMATCH_FS_ATOMIC_NAME_BLOCKING`` to ``on`` | ``auto`` | ``off``.

    Governs the atomic-name-soundex blocking lever
    (``_add_atomic_name_soundex_blocking``): when the blocking already keys on a
    COMPOSITE name via soundex (``full_name``/``first_and_surname``) but has no
    ATOMIC single-name soundex pass, a corrupted first name breaks the whole
    composite key even when the surname is intact -- so the same-surname
    duplicates are never co-blocked (historical_50k: 49% of the missed true pairs
    share a surname-soundex the blocking doesn't use).

    - ``off`` (**default -- KNOWN-NEGATIVE, do not flip**): never fire
      (byte-identical to the pre-lever config). **The canonical
      ``bench_er_headtohead`` panel measures this lever as a REGRESSION on its
      only target: historical_50k pairwise F1 -0.0148 (B3 -0.0078), robustly
      (stable across hash seeds + repeated runs).** It DOES raise candidate /
      blocking recall +6pp (0.886 -> 0.946) -- the blocking mechanism works --
      but the corrupted-name candidates it generates fall BELOW the FS threshold
      (threshold_loss +7pp) and the EM shift from the added passes degrades the
      operating point, so precision AND recall drop. (An earlier +0.0067 B3
      figure was measured against a stale pipeline state and did NOT reproduce
      on current main.) febrl3/febrl4 are no-ops. Kept gated for reference /
      future salvage, NOT because it works. SOUNDEX vs STRIP note: atomic-name
      STRIP over-merges even harder (historical_50k B3 precision -0.05); soundex
      is the less-bad variant, but "less bad" is still net-negative here.
    - ``auto`` (``auto``): fire ONLY on person-shaped data
      (``_dataset_is_person_shaped``) -- the regime where names are the primary
      blocking signal.
    - ``on`` (``1``/``true``/``yes``/``enabled``): force everywhere.

    Mirrors the ``GOLDENMATCH_FS_ORTHOGONAL_BLOCKING`` tri-state, but defaults OFF
    (not ``auto``) because the full ER panel hasn't blessed it yet.
    """
    v = os.environ.get("GOLDENMATCH_FS_ATOMIC_NAME_BLOCKING", "off").lower()
    if v in ("1", "true", "on", "yes", "enabled"):
        return "on"
    if v == "auto":
        return "auto"
    return "off"


# Fraction of a column's non-null sample values that must parse as valid
# coordinates for it to be admitted as a single-field geo_haversine column.
_LATLONG_SAMPLE_FLOOR = 0.8


def _has_geographic_evidence(pairs: list[tuple[float, float]]) -> bool:
    """Whether parsed ``(lat, lon)`` pairs carry POSITIVE evidence of being
    coordinates, rather than merely two comma-separated numbers that happen to
    fall in valid coordinate ranges.

    Parseability alone is a much weaker test than it looks: ``"25,35"`` is a
    perfectly valid coordinate (lat 25, lon 35) AND a perfectly ordinary packed
    age interval. So are salary bands, size ranges, version pairs and score
    spans. #2443 hit exactly this -- auto-config emitted ``geo_haversine`` for an
    ``age_range`` column on a real ~21k-row corpus, producing plausible-looking
    distances that took a field-by-field diff against a hand-built matchkey to
    notice. Nothing failed; the config was well-formed and simply wrong.

    Any ONE of these is enough, and each is something an integer-interval column
    essentially never exhibits:

    * a fractional component -- real coordinates are decimal degrees; whole
      degrees are ~111 km apart, a granularity at which haversine similarity is
      meaningless anyway, so declining those is correct rather than a miss;
    * ``|lon| > 90`` -- outside latitude's valid band, so unambiguously a
      longitude and not a second small magnitude;
    * a negative component -- southern or western hemisphere.

    Deliberately data-driven rather than name-driven. A name veto
    (``*_range``/``*_span``) was the other option in #2443 and would also have
    caught this case, but it fails on any locale or convention we did not
    anticipate, whereas "the numbers do not look geographic" holds regardless of
    what the column is called.
    """
    for lat, lon in pairs:
        if lat % 1 or lon % 1:
            return True
        if abs(lon) > 90.0:
            return True
        if lat < 0 or lon < 0:
            return True
    return False


def _looks_like_latlong(profile: ColumnProfile) -> bool:
    """Whether a column holds combined ``"lat,long"`` coordinate strings, judged
    by profiling ``sample_values`` through the same parser the scorer uses. A
    strong majority (>= _LATLONG_SAMPLE_FLOOR) of the non-null sample must parse
    to valid coordinates AND the parsed values must carry positive geographic
    evidence (see ``_has_geographic_evidence``) -- the parse-rate floor alone
    admits any column of two small comma-separated numbers, which is how #2443
    scored an age-interval column with ``geo_haversine``. Only consulted under
    the FS domain-comparators flag, so it never runs (and can't shift behavior)
    by default. Separate lat/long columns are the deferred cross-field
    comparator."""
    from goldenmatch.core.scorer import _parse_latlong

    sample = [v for v in profile.sample_values if v is not None and str(v).strip()]
    if len(sample) < 5:  # too little evidence to claim a coordinate column
        return False
    parsed = [_parse_latlong(str(v)) for v in sample]
    ok = [p for p in parsed if p is not None]
    if len(ok) / len(sample) < _LATLONG_SAMPLE_FLOOR:
        return False
    return _has_geographic_evidence(ok)


# Domain-extracted column scorer mapping.
# These columns are added by extract_features() and start with __.
_DOMAIN_SCORER_MAP = {
    # Electronics
    "__brand__": ("exact", 0.8, ["lowercase", "strip"]),
    "__model__": ("exact", 1.0, ["strip"]),
    "__model_norm__": ("exact", 1.0, []),
    "__color__": ("exact", 0.2, ["lowercase"]),
    "__specs__": ("token_sort", 0.3, ["strip"]),
    # Software
    "__sw_name__": ("token_sort", 1.0, ["lowercase", "strip"]),
    "__sw_version__": ("exact", 0.5, ["strip"]),
    "__sw_edition__": ("exact", 0.3, ["lowercase"]),
    "__sw_platform__": ("exact", 0.3, ["lowercase"]),
    "__sw_part_num__": ("exact", 1.0, ["strip"]),
    # Bibliographic
    "__title_key__": ("exact", 0.8, ["lowercase"]),
}

# A domain-extracted `exact` field becomes a STANDALONE exact matchkey -- "same
# value implies same entity", on its own, with nothing else consulted -- only if
# it is a genuine identifier. `_DOMAIN_SCORER_MAP` already states that: the weight
# is the author's own confidence, 1.0 for identifiers (`__model_norm__`,
# `__sw_part_num__`) and lower for partial signals (`__color__` 0.2,
# `__title_key__` 0.8). That weight used to be read and then DISCARDED when the
# standalone matchkey was built, so every exact entry made a full-strength
# identity claim regardless.
#
# Measured on DBLP-ACM (#2526): `__title_key__` is the first significant WORD of
# the title, so `domain_exact_title_key` asserted "two papers whose titles start
# with the same word are the same paper" -- 33,563 pairs against 2,224 in ground
# truth, clusters up to 96, precision 0.068. It cleared the old `< 0.01`
# cardinality floor 29x over (ratio 0.29), because that floor only rejects
# NEAR-CONSTANT columns; an identity claim needs cardinality near 1.0.
_DOMAIN_IDENTITY_WEIGHT = 1.0
# Belt-and-braces on top of the declared weight: even a 1.0-weighted extractor
# cannot be an identity claim on data where the column is coarse in practice. A
# real identifier is near-unique; this rejects the degenerate case where an
# extractor collapses a column on some dataset it was not tuned for.
_DOMAIN_IDENTITY_MIN_CARDINALITY = 0.5


# Transforms that only normalise (case, surrounding whitespace) and so preserve
# equality: if two values land in the same block under these, the values
# themselves still compare equal. Everything else -- soundex, substring:N:M,
# metaphone, token_sort -- is LOSSY: "Sony" and "Sonny" share a soundex block
# while being different strings, so the raw field still discriminates inside the
# block and must not be treated as constant. The list is deliberately a
# whitelist: an unrecognised transform is assumed lossy.
_EQUALITY_PRESERVING_TRANSFORMS = frozenset({
    "lowercase", "uppercase", "lower", "upper", "strip", "trim",
})


def _always_agreeing_blocking_fields(blocking: BlockingConfig | None) -> set[str]:
    """Fields on which EVERY candidate pair is guaranteed to agree.

    Two conditions, both necessary:

    * **Every pass must key on the field.** A blocking key is constant inside its
      own block, but under ``multi_pass`` a pair produced by pass 2 can disagree
      on pass 1's key -- so the field still carries signal. Hence the INTERSECTION
      across passes, not the union.
    * **Every transform on that key must preserve equality.** Sharing a block
      under a LOSSY transform does not mean sharing a value (measured: abt_buy
      keys all three passes on ``manufacturer``, but via ``soundex`` and
      ``substring:0:5``, so ``manufacturer`` genuinely discriminates within a
      block and dropping it would be wrong)."""
    if blocking is None:
        return set()
    keys = list(getattr(blocking, "passes", None) or [])
    if not keys:
        keys = list(getattr(blocking, "keys", None) or [])
    if not keys:
        return set()
    groups = []
    for k in keys:
        if not set(k.transforms or []).issubset(_EQUALITY_PRESERVING_TRANSFORMS):
            return set()  # a lossy pass -> no field is guaranteed constant
        groups.append(set(k.fields))
    return set.intersection(*groups)


def _drop_uninformative_blocking_fields(
    matchkeys: list[MatchkeyConfig], blocking: BlockingConfig | None,
) -> None:
    """Drop scored fields that every candidate pair agrees on by construction.

    Such a field contributes its full weight to every pair as a constant OFFSET
    with zero discriminative power -- it cannot separate a match from a non-match,
    it only shifts the whole score distribution up and flattens the threshold's
    effect. Measured on DBLP-ACM (#2526), where blocking is static on
    ``__title_key__`` and the same column was also scored at weight 0.8: it
    supplied **18.6%** of the weighted score to 100% of candidates while
    distinguishing none of them.

    The matchkey's ``threshold`` is rescaled by the surviving weight share so the
    decision boundary lands in the same place on the fields that actually
    discriminate. Without that, removing an 18.6% constant makes the SAME nominal
    threshold materially stricter and recall drops -- measured: dropping the field
    at an unchanged 0.70 scored F1 0.6855 vs 0.6894 before, a small REGRESSION,
    while the same field set at its rescaled boundary scored 0.7029.

    Mutates ``matchkeys`` in place. A matchkey is left untouched if the rule would
    empty it, since a matchkey with no fields cannot score anything."""
    dead = _always_agreeing_blocking_fields(blocking)
    if not dead:
        return
    for mk in matchkeys:
        if getattr(mk, "type", None) != "weighted":
            continue
        fields = list(getattr(mk, "fields", None) or [])
        keep = [f for f in fields if f.field not in dead]
        if not keep or len(keep) == len(fields):
            continue
        # `fields` came from an already-validated MatchkeyConfig, and this loop
        # already skipped anything not type=="weighted" -- weight is guaranteed
        # non-None here (MatchkeyConfig._validate_weighted enforces it). Using
        # the schema's own accessor instead of `or 0` means a validator-bypass
        # bug (mutation post-construction) crashes loudly here instead of
        # silently zeroing a field's weight and mis-scaling the threshold.
        total = sum(f.fuzzy_weight for f in fields)
        kept = sum(f.fuzzy_weight for f in keep)
        dropped = [f.field for f in fields if f.field in dead]
        if total > 0 and kept > 0 and mk.threshold is not None:
            # Same absolute score bar, expressed over the surviving weight.
            rescaled = (float(mk.threshold) * total - (total - kept)) / kept
            mk.threshold = round(min(max(rescaled, 0.05), 0.99), 4)
        mk.fields = keep
        logger.info(
            "auto-config: dropped %s from matchkey %r -- blocking guarantees every "
            "candidate pair agrees on %s, so it was a constant %.1f%% of the score "
            "with no discriminative power; threshold rescaled to %.4f",
            ", ".join(dropped), mk.name, "them" if len(dropped) > 1 else "it",
            100.0 * (total - kept) / total if total else 0.0, mk.threshold,
        )


def _drop_constant_scored_fields(
    matchkeys: list[MatchkeyConfig], profiles: list[ColumnProfile],
) -> None:
    """Drop scored fields that are CONSTANT in the data (#2668).

    Sibling of `_drop_uninformative_blocking_fields`, for the other way a scored
    field can carry no information. There the field varies in the data but
    blocking guarantees agreement WITHIN a block; here it does not vary at all,
    so no pair can ever disagree on it. Either way it is a constant offset with
    zero discriminative power.

    **It does not rescale the threshold, and that difference is the whole point.**
    A constant field at equal weight does not merely shift scores; it HALVES the
    effective bar on the field that actually discriminates. Measured on the
    goldengraph TINY fixture, whose `type` column is `'concept'` on every row:

        score = (name + 1.0) / 2      threshold 0.8  =>  the real bar on `name` is 0.6
        ensemble("Beta Corp", "Delta LLC") = 0.7037  =>  0.8519, a false merge

    Rescaling preserves the same absolute bar, so it would preserve that merge
    exactly -- it is a no-op on every decision. The sibling rule rescales because
    a blocking-agreeing field is genuinely informative elsewhere and its
    calibrated contribution should survive; a data-constant column is informative
    NOWHERE, so the honest config is the one auto-config would have built had the
    useless column not been in the frame. It picks 0.8 on `name` alone there, and
    that config makes no cross-entity merge on this fixture.

    Reads `full_n_distinct`, the EXACT full-frame count, never the sampled
    `n_distinct`. The sample is 1,000 rows, so a column that is 99.99% one value
    reads 1 there; acting on that would drop a field on a large frame and keep it
    on a small one -- the scale-flipping verdict `full_cardinality_ratio` exists
    to prevent, and it would fail in the same direction as the bug this rule
    fixes, since a field that disagrees only on rare pairs carries its whole
    signal in exactly those pairs.

    A field whose profile carries no `full_n_distinct` is LEFT ALONE: "cannot
    answer" must not become "assumed constant". Same for a rule that would empty
    a matchkey, since a matchkey with no fields cannot score anything.
    """
    const = {
        p.name for p in profiles
        if p.full_n_distinct is not None and p.full_n_distinct <= 1
    }
    if not const:
        return
    for mk in matchkeys:
        if getattr(mk, "type", None) != "weighted":
            continue
        fields = list(getattr(mk, "fields", None) or [])
        keep = [f for f in fields if f.field not in const]
        if not keep or len(keep) == len(fields):
            continue
        dropped = [f.field for f in fields if f.field in const]
        mk.fields = keep
        logger.info(
            "auto-config: dropped %s from matchkey %r -- %s constant in the data, "
            "so every pair agreed on %s by construction and %s only relaxed the "
            "effective bar on the fields that do discriminate. Threshold left at "
            "%s, which is now the bar on those fields.",
            ", ".join(dropped), mk.name,
            "they are" if len(dropped) > 1 else "it is",
            "them" if len(dropped) > 1 else "it",
            "they" if len(dropped) > 1 else "it",
            mk.threshold,
        )


def _is_identity_claim(col: str, weight: float, cardinality_ratio: float) -> bool:
    """Whether a domain-extracted ``exact`` field may stand alone as its own
    exact matchkey, versus being folded into the weighted matchkey at its
    declared weight. See `_DOMAIN_IDENTITY_WEIGHT` for why the weight decides."""
    if weight < _DOMAIN_IDENTITY_WEIGHT:
        return False
    return cardinality_ratio >= _DOMAIN_IDENTITY_MIN_CARDINALITY


# Free-text col_types where data-entry corruption is common and `token_sort`
# (word-order-robust but character-noise-fragile) underperforms. Surfaced by the
# NCVR regression: res_street_address scored with token_sort gave F1 0.871;
# jaro_winkler recovered 0.981. See #662.
_NOISE_PRONE_FUZZY_TYPES = frozenset({"address", "string"})

# Benchmark-confirmed upgrade target (#662). The sweep tied jaro_winkler with
# ensemble on NCVR-high F1; jaro_winkler is chosen as the cheaper of the tie
# (single-pass char similarity vs the multi-scorer ensemble) and is the
# NCVR-validated lever.
_NOISE_AWARE_TARGET_SCORER = "jaro_winkler"


def _noise_aware_target_scorer() -> str:
    """The scorer token_sort upgrades to on noise-prone col_types. Benchmark-only
    env override (GOLDENMATCH_NOISE_AWARE_TARGET) lets the harness sweep
    jaro_winkler/ensemble without code edits; unset -> the committed constant."""
    return os.environ.get("GOLDENMATCH_NOISE_AWARE_TARGET") or _NOISE_AWARE_TARGET_SCORER


def _noise_aware_scorers_enabled() -> bool:
    """Whether to upgrade noise-fragile token_sort assignments on free-text
    col_types. Default ON (#662: jaro_winkler gave +0.48pp NCVR-high F1,
    precision-driven, no Febrl3 regression; #528 CI gate guards clean precision).
    Kill-switch: GOLDENMATCH_NOISE_AWARE_SCORERS=0 (or "false"/"disabled",
    case-insensitive)."""
    return os.environ.get("GOLDENMATCH_NOISE_AWARE_SCORERS", "1").lower() not in (
        "0", "false", "disabled",
    )


def _tf_name_weighting_enabled() -> bool:
    """Whether auto-config populates ``MatchkeyField.tf_freqs`` for the
    data-driven name-frequency downweight on ``name_freq_weighted_jw`` fields
    (#1207 PR2a). Default ON. Without a populated table the scorer falls back to
    the static US-Census surname path, so this is the seam that makes the
    per-dataset downweight actually fire. Kill-switch:
    GOLDENMATCH_TF_NAME_WEIGHTING=0 (or false/no/off, case-insensitive)."""
    return os.environ.get("GOLDENMATCH_TF_NAME_WEIGHTING", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _route_to_probabilistic_enabled() -> bool:
    """Auto-route a probabilistic-shaped dataset (no strong-identity exact matchkey
    + multiple weak fuzzy fields) to the Fellegi-Sunter path instead of the default
    exact+weighted matchkeys. Default ON (2026-07-17) -- the dual-strategy corpus
    proof came back green: routing lifts the default-strategy F1 on every routed
    dataset (historical_50k 0.34->0.77, ncvr_synthetic 0.96->0.99, febrl3 flat) with
    NO regression on the one det-wins anchor (anchor_person_match holds ~0.99, kept on
    exact matching by its surviving email key -- see _is_probabilistic_shape). Only
    no-strong-id + >=2-fuzzy-field shapes route; strong-id (email/phone/identifier)
    shapes stay on exact matching. Kill-switch:
    GOLDENMATCH_AUTOCONFIG_ROUTE_PROBABILISTIC=0 (or false/no/off/disabled).

    An in-process override (`deterministic_routing()` / `_ROUTE_PROBABILISTIC_OVERRIDE`)
    takes precedence over the env var -- the streaming/incremental orchestrator uses it
    to stay deterministic because it cannot execute a routed F-S config per-block."""
    override = _ROUTE_PROBABILISTIC_OVERRIDE.get()
    if override is not None:
        return bool(override)
    return os.environ.get("GOLDENMATCH_AUTOCONFIG_ROUTE_PROBABILISTIC", "1").strip().lower() not in (
        "0", "false", "no", "off", "disabled",
    )


# Fellegi-Sunter EM needs enough rows to estimate stable m/u probabilities (u from
# random non-matching pairs, m from within-block pairs). Below this floor the EM is
# data-starved — it commits a RED, low-confidence config, and its discrete-level
# comparison (partial_threshold 0.8) drops fuzzy-close variants to the "disagree"
# level, under-merging surfaces that the robust weighted fuzzy path catches. Measured
# on the 150-row goldengraph KG concept universe: FS unified only 8/44 multi-surface
# concepts vs the weighted path's 19/44 (splitting sets like ['Jaro-Winkler distance',
# 'Jaro Winkler','jaro_winkler']). The floor is conservative — every dataset that
# validated the FS-default win sits far above it (dblp_acm ~2600, febrl3 ~5000,
# historical_50k 50k), so routing there is unchanged; only genuinely small corpora
# (KG entity sets, tiny demos) fall back to the weighted path. Override with
# GOLDENMATCH_FS_ROUTE_MIN_ROWS (0 disables the floor -> route purely by shape).
_FS_ROUTE_MIN_ROWS_DEFAULT = 500


def _fs_route_min_rows() -> int:
    """Row floor below which a probabilistic-shaped dataset stays on the weighted
    path (FS EM is data-starved at small N). Env-overridable; 0 disables the floor."""
    raw = os.environ.get("GOLDENMATCH_FS_ROUTE_MIN_ROWS")
    if raw is None:
        return _FS_ROUTE_MIN_ROWS_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _FS_ROUTE_MIN_ROWS_DEFAULT


# col_type given to a domain-extracted, exact-scored derived column (e.g.
# __title_key__, __model_norm__ -- see the injection at the "Add domain columns
# to blocking candidate profiles" site below). These are high-cardinality,
# exact-comparable, no-domain-cap columns functionally like email/identifier/
# phone for BLOCKING purposes, but are not real email addresses -- #2876:
# tagging them col_type="email" routed them through _detect_standardization_
# config's email branch, which built a StandardizationConfig rule applying the
# std_email standardizer (nulls any value without "@") to a derived key that
# never contains one.
_EXACT_DERIVED_COL_TYPE = "exact_derived"

# Exact matchkeys on these col_types are a strong identity claim — when one
# SURVIVES into the config, exact matching carries the dedup and the probabilistic
# path tends to lose (verified on anchor_person_match: clean-email exact beats FS).
# Broader than just "identifier": the dual-strategy harness showed email/phone exact
# matchkeys are equally strong. `zip` is NOT here (a blocking signal, not identity).
# `exact_derived` (#2876) is included for the same reason: a domain-extracted
# exact-scored column is exactly this kind of strong identity signal for
# blocking purposes, even though it is not itself an "identifier" column.
_STRONG_EXACT_TYPES = ("identifier", "email", "phone", _EXACT_DERIVED_COL_TYPE)


def _is_probabilistic_shape(
    matchkeys: list[MatchkeyConfig], profiles: list[ColumnProfile]
) -> bool:
    """Probabilistic shape = no SURVIVING exact matchkey backed by a strong-identity
    column (identifier/email/phone) + >=2 DISTINCT fuzzy (weighted) COLUMNS for EM to
    weight. Keys on the EMITTED matchkeys (not raw profiles), so a ceiling-excluded id
    column (a perfectly-unique surrogate) correctly counts as 'no surviving strong key'.

    The count is over DISTINCT REAL input COLUMNS, not matchkey-field entries: a single
    free-text/description column gets *two* fuzzy fields -- a `token_sort` on the column
    plus a whole-record `record_embedding` on the synthetic `__record__` field (the
    "route long text to fuzzy alongside embedding" rule). That is ONE semantic field
    re-scored, not two independent ones. Fellegi-Sunter EM assumes conditional
    independence ACROSS FIELDS, so scoring one column twice does not make an FS-shaped
    dataset -- and routing a single-text-column corpus to FS measurably splits obvious
    near-duplicates (it belongs on the LSH + fuzzy deterministic path). We therefore
    drop synthetic `__`-prefixed fields (e.g. `__record__`) from the count, so a real
    multi-field person dataset (first_name + last_name) still routes while a single-column
    corpus stays put."""
    col_type = {p.name: p.col_type for p in profiles}
    # f.field is str | None (embedding fields have None); drop None for the lookup.
    exact_fields = [f.field for mk in matchkeys if mk.type == "exact" for f in mk.fields if f.field]
    has_strong_id = any(col_type.get(fld) in _STRONG_EXACT_TYPES for fld in exact_fields)
    fuzzy_columns = {
        f.field
        for mk in matchkeys if mk.type == "weighted"
        for f in mk.fields
        if f.field and not f.field.startswith("__")
    }
    return (not has_strong_id) and len(fuzzy_columns) >= 2


def _noise_aware_scorer(col_type: str, scorer: str) -> str:
    """Upgrade a noise-fragile token_sort to the noise-aware target scorer for
    free-text col_types prone to character-level corruption. No-op unless the gate
    is on AND the assignment is exactly token_sort on a noise-prone col_type — so
    a non-default scorer chosen upstream (refdata/embedding/ensemble/qgram) is
    never overridden."""
    if (
        _noise_aware_scorers_enabled()
        and col_type in _NOISE_PRONE_FUZZY_TYPES
        and scorer == "token_sort"
    ):
        return _noise_aware_target_scorer()
    return scorer


def _is_short_code(p: ColumnProfile) -> bool:
    """#491: detect short alphanumeric *code* columns (SKU, part-no, plan-code).

    These collapse under the generic string scorers (``token_sort`` token-bag
    similarity is meaningless on a 6-char opaque code), so they get routed to
    the char-n-gram ``qgram`` scorer instead. Deliberately conservative — a
    real name/word column must NOT match, or we'd silently swap proven name
    scoring for qgram. The letter+digit-mix requirement is the main guard:
    English names/words are letters-only, so they fail it.
    """
    if not (3 <= p.avg_len <= 12):
        return False
    if p.cardinality_ratio < 0.3:
        return False
    # Only generic string-ish columns are candidates; named identity types
    # (name/email/phone/zip/geo/date/numeric) keep their tuned scorers.
    if p.col_type not in ("string", "identifier"):
        return False

    samples = [s for s in (p.sample_values or []) if s]
    if not samples:
        return False

    # Code-like signal: the value carries BOTH a letter and a digit
    # (e.g. ``A1B2C3``, ``SKU-9921``). This is the discriminating test that
    # excludes pure-alpha names/words and pure-numeric IDs (already handled
    # as identifiers/numeric upstream). Require a clear majority to look
    # code-like before overriding.
    def _is_code_like(s: str) -> bool:
        has_alpha = any(c.isalpha() for c in s)
        has_digit = any(c.isdigit() for c in s)
        return has_alpha and has_digit

    code_like = sum(1 for s in samples if _is_code_like(s))
    return code_like >= max(1, (len(samples) + 1) // 2)


def _adaptive_threshold(fields: list[MatchkeyField]) -> float:
    """Compute threshold based on field types in the matchkey."""
    exact_scorers = {"exact"}
    embedding_scorers = {"embedding", "record_embedding"}

    scorers = {f.scorer for f in fields if f.scorer}

    if scorers <= exact_scorers:
        return 0.95
    if scorers & embedding_scorers:
        return 0.70
    if len(fields) == 1:
        return 0.85
    return 0.80


# #858: distinguish real person-name columns from "name"-typed SHARED ATTRIBUTE
# columns (company, job_title, department, ...). The profiler's name heuristic
# classifies an organization/role column as col_type="name", which then enters
# the weighted matchkey as a full-weight person-name fuzzy field. Under
# multi-source that collapses distinct people who share an employer / title (the
# crm_multisource_realistic over-merge: bare F1 0.13 -> 0.77 once company +
# job_title are demoted). The org/attribute check runs FIRST so a name-bearing
# org column (e.g. "company_name") is still recognized as a shared attribute.
_ORG_ATTR_COLUMN_RE = re.compile(
    r"(?i)(?:^|[^a-z])("
    r"company|employer|organi[sz]ation|org|firm|business|corp|"
    r"job|title|role|position|occupation|department|dept|division|team|group|"
    r"category|segment|industry|sector|brand|product"
    r")(?:[^a-z]|$)"
)
_PERSON_NAME_COLUMN_RE = re.compile(
    r"(?i)(?:^|[^a-z])("
    r"first|last|given|family|sur|surname|middle|maiden|forename|fore|"
    r"nick|nickname|preferred|legal|full|display|fname|lname|mname|name|"
    r"person|contact|individual|party|customer|client|member|patient|applicant"
    r")(?:[^a-z]|$)"
)


def _is_person_name_column(column: str) -> bool:
    """True when a ``col_type='name'`` column is plausibly a PERSON name (so it
    earns a positive weighted-matchkey feature), False when it's a shared
    organization/role attribute (company, job_title, department). The org check
    wins, so ``company_name`` reads as a shared attribute, not a person name."""
    if _ORG_ATTR_COLUMN_RE.search(column):
        return False
    return bool(_PERSON_NAME_COLUMN_RE.search(column))


def _exact_matchkey_floor_py(col_type: str) -> float:
    """Pure-Python oracle for the S3 per-type exact-matchkey cardinality floor.

    phone 0.30 (legitimately shared -> permissive); everything else, INCLUDING
    email, keeps the historical 0.50 default. A shared email (cardinality 0.5) is
    a genuine identity signal this codebase keeps as an exact matchkey, and the
    matchkey-guard tests pin email's floor to 0.50; the spec's initial email=0.70
    demoted those and was corrected. Mirrors the Rust `exact_matchkey_floor`
    kernel (golden-vector parity). Closes the TODO at issue #715.
    """
    if col_type == "phone":
        return 0.30
    return 0.50


def exact_matchkey_floor(col_type: str) -> float:
    """S3 per-type exact-matchkey floor. Dispatches to the shared native core
    when enabled (byte-identical to the pure-Python oracle), else pure Python."""
    from goldenmatch.core._native_loader import (  # noqa: PLC0415
        native_enabled,
        native_module,
    )

    if native_enabled("autoconfig"):
        _nm = native_module()
        if hasattr(_nm, "autoconfig_exact_matchkey_floor"):
            import json  # noqa: PLC0415

            out = _nm.autoconfig_exact_matchkey_floor(
                json.dumps({"col_type": col_type})
            )
            return float(json.loads(out)["floor"])
    return _exact_matchkey_floor_py(col_type)


def build_matchkeys(
    profiles: list[ColumnProfile], df: pl.DataFrame | None = None,
    *, multi_source: bool = False,
) -> list[MatchkeyConfig]:
    """Generate matchkeys from column profiles."""
    # Separate exact and fuzzy columns
    exact_fields = []
    fuzzy_fields = []
    description_columns = []

    # Track why each exact-eligible column was skipped, so the aggregate
    # warning below can explain *which* columns were lost and *why* instead
    # of just a count. This is the difference between a notebook user
    # noticing their config silently degraded and not.
    skipped_exact: list[tuple[str, str]] = []  # (column, reason)

    # Person-name columns are the co-agreement anchor for the attribute-demotion
    # check in the loop below: a workplace/locality attribute is one whose
    # shared-value records do NOT share a person NAME. Built once (name-typed AND
    # person-name only, so an org/company "name" field is never in the anchor).
    _person_name_basket: list[tuple[str, bool]] = [
        (p.name, True)
        for p in profiles
        if getattr(p, "col_type", None) in ("name", "multi_name")
        and _is_person_name_column(p.name)
    ]

    for p in profiles:
        # identifier columns ARE matchable: a real shared identifier
        # (NPI/SSN/MRN) backs an exact matchkey, gated below by the
        # cardinality band. Per-record surrogate keys (card==1.0) are
        # excluded by the upper bound. See #715.
        if p.col_type in ("numeric", "date", "year"):
            continue  # year is blocking-only

        if p.col_type == "description":
            fuzzy_fields.append(MatchkeyField(
                field=p.name,
                scorer="token_sort",
                weight=1.5,  # higher weight ensures survival past max_fuzzy_fields truncation
                transforms=["lowercase", "strip"],
            ))
            description_columns.append(p)
            continue

        if p.col_type == "multi_name":
            fuzzy_fields.append(MatchkeyField(
                field=p.name,
                scorer="token_sort",
                weight=1.0,
                transforms=["lowercase", "strip"],
            ))
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, weight, transforms = scorer_info

        # Refdata hook: swap scorer / prepend transforms when the column
        # name signals a refdata-handled shape (last_name, first_name,
        # company name, address). No-op when goldenmatch.refdata isn't
        # imported or the relevant pack's data file is missing — the
        # module-top fallback at line ~38 wires a pass-through stub for
        # that case. Lift numbers per refdata pack are in CHANGELOG
        # entries for slices #2-#5. ``p.col_type`` gates each refinement
        # on the profiled data shape so a column literally named
        # ``last_name`` but holding non-name data isn't silently swapped.
        scorer, transforms = _refdata_refine_matchkey_field(
            p.name, scorer, transforms, p.col_type,
        )

        # #491: short opaque codes (SKU, part-no) score badly under the
        # generic string scorers — token_sort/ensemble assume word-ish text.
        # Route them to the char-n-gram qgram scorer instead. Gated on
        # _is_short_code (letter+digit mix, short, high-cardinality) so real
        # names/words/identifiers keep their tuned scorers untouched.
        if scorer in ("token_sort", "ensemble") and _is_short_code(p):
            scorer = "qgram"

        # #662: noise-aware refinement (opt-in, default OFF). Upgrade token_sort
        # -> jaro_winkler/ensemble on corruption-prone free-text col_types. Runs
        # AFTER the qgram short-code guard so a code-like string keeps qgram (the
        # guard already rewrote it; this no-ops on non-token_sort). See
        # _noise_aware_scorer.
        scorer = _noise_aware_scorer(p.col_type, scorer)

        # Group/list-attribute demotion for EXACT matchkeys (single-source
        # counterpart to the #858 multi-source demotion; gated, default OFF).
        # A shared group/list/facility value -- a clinic ``phone`` line, a
        # mailing-list / campaign ``identifier`` (tl_id), a facility NPI -- as an
        # EXACT matchkey force-merges every DIFFERENT person sharing it into one
        # mega-cluster (the DERM over-merge: exact_phone / exact_tl_id).
        #
        # Scoped to EXACT uses ONLY, deliberately. The same attribute as a
        # WEIGHTED fuzzy contributor is NOT demoted: it is a soft signal, not a
        # force-merge, and on corruption-heavy data (febrl3, whose synthetic
        # addresses also collide across people) it is LOAD-BEARING -- names are
        # too corrupted to carry identity alone, so the weighted address field is
        # needed to match true duplicates. Demoting the weighted use there
        # regressed febrl3 F1 0.99->0.86 in the accuracy sweep; restricting to
        # exact keeps that recall while still killing the hard force-merges.
        #
        # Verdict is data-measured (group-size-aware co-agreement on the person
        # name; see should_demote_attribute_field) and a no-op where the value is
        # genuinely identity-correlated (a personal cell / personal id whose
        # shared-value records DO co-agree on name is kept). Column stays a
        # blocking candidate.
        if scorer == "exact" and should_demote_attribute_field(
            df, p.name, p.col_type, _person_name_basket,
            is_person_name=_is_person_name_column(p.name),
        ):
            reason = (
                "group-attribute: the different people sharing its value do not "
                "co-agree on the person name -- a shared group/list/facility "
                "value, not an identity claim"
            )
            logger.info(
                "Demoting exact matchkey '%s' to blocking-only (%s). "
                "Column remains a blocking candidate.",
                p.name, reason,
            )
            skipped_exact.append((p.name, reason))
            continue

        # #858: when multi-source, phone is a blocking signal, not an identity
        # claim -- an exact match on phone collapses distinct people who share a
        # work / household phone. Demote to blocking-only (this continue skips
        # both exact_fields and fuzzy_fields; phone remains a build_blocking
        # candidate, which is unchanged).
        if multi_source and scorer == "exact" and p.col_type == "phone":
            skipped_exact.append((
                p.name,
                "multi-source: phone is a blocking signal, not an identity claim",
            ))
            continue

        # Geo and zip are blocking signals, NOT identity claims. An exact
        # matchkey on a city column asserts "two records sharing a city are
        # the same entity", which collapses every record per city into one
        # mega-cluster. These columns still drive blocking via build_blocking;
        # they just cannot back matchkeys themselves.
        if scorer == "exact" and p.col_type in ("zip", "geo"):
            reason = f"col_type={p.col_type} is a blocking signal, not an identity claim"
            logger.warning(
                "Skipping exact matchkey for '%s' (%s). "
                "Column remains a blocking candidate.",
                p.name, reason,
            )
            skipped_exact.append((p.name, reason))
            continue

        # Exact matchkeys assert identity equivalence, so the backing column
        # must be plausibly unique. S3 (spec 2026-06-22, issue #715): the floor
        # is per-type via the shared exact_matchkey_floor kernel -- emails are
        # near-unique (0.70), phones are legitimately shared (0.30), everything
        # else keeps the historical 0.50. This catches low-cardinality numeric
        # columns misclassified by upstream transforms (e.g. a 4-digit year
        # reshaped into an ISO date looking phone-shaped) that would collapse
        # every row sharing that value into one mega-cluster.
        _exact_floor = exact_matchkey_floor(p.col_type)
        if scorer == "exact" and p.cardinality_ratio > 0 and p.cardinality_ratio < _exact_floor:
            reason = (
                f"cardinality_ratio={p.cardinality_ratio:.4f} < {_exact_floor:.2f} "
                f"(col_type={p.col_type}) -- lacks identifier-level uniqueness"
            )
            logger.warning(
                "Skipping exact matchkey for '%s' (%s). "
                "Exact match would create spurious mega-clusters.",
                p.name, reason,
            )
            skipped_exact.append((p.name, reason))
            continue

        # Exact matchkeys are a Polars hash self-join (find_exact_matches),
        # not a nested loop, and do not pass through fuzzy blocking. Their
        # cost is the number of emitted equal-pairs, bounded by cardinality:
        # a high-cardinality column emits few pairs and is both cheap and
        # mega-cluster-safe. So there is NO row-count guard here (the old
        # df.height > 10000 guard mismodeled the cost and orphaned real
        # identifiers -- see #715). The mega-cluster risk is the OPPOSITE
        # shape (low cardinality), already caught by the >= 0.5 gate above.
        #
        # Upper bound: a perfectly-unique column (card == 1.0) is a
        # per-record surrogate key (e.g. a row PK). It is never shared, so an
        # exact match emits zero pairs and asserts no real identity. Exclude
        # it for config hygiene.
        if scorer == "exact" and _is_perfect_surrogate(p):
            _measured = (
                f"full-frame cardinality_ratio={p.full_cardinality_ratio:.4f}"
                if p.full_cardinality_ratio is not None
                else f"sampled cardinality_ratio={p.cardinality_ratio:.4f}"
            )
            reason = (
                f"{_measured} >= 1.0 "
                f"-- perfectly-unique surrogate key, no shared identity to match"
            )
            logger.info(
                "Skipping exact matchkey for '%s' (%s).", p.name, reason,
            )
            skipped_exact.append((p.name, reason))
            continue

        # #1351: discriminative-power veto. A column that clears the cardinality
        # gates can still be a shared LOCALITY attribute (e.g. a zip mis-promoted
        # to "identifier") rather than an identity key. Veto its exact matchkey
        # when records sharing its value don't co-agree on other identity fields.
        # Fail-safe keep (df is None / thin support / empty basket) is handled
        # inside should_veto_exact, so near-unique identity keys are unaffected.
        if scorer == "exact" and should_veto_exact(df, p.name, profiles):
            reason = "discriminative-power veto: shared-value records do not co-agree on identity fields"
            logger.warning("Skipping exact matchkey for '%s' (%s).", p.name, reason)
            skipped_exact.append((p.name, reason))
            continue

        # #858: under multi-source, a low-cardinality "name"-typed field whose
        # column is NOT a person name (company, job_title, department, ...) is a
        # shared workplace/categorical attribute, NOT an identity claim. As a
        # full-weight positive feature in the weighted matchkey it collapses
        # distinct people who share an employer / title -- the dominant driver of
        # the multi-source over-merge (crm_multisource_realistic: bare F1 0.13;
        # demoting company + job_title -> 0.77). Demote to blocking-only, exactly
        # like the phone demotion above (real person-name fields and
        # high-cardinality fields are kept). Gated on the multi_source detection,
        # so single-source autoconfig is byte-identical.
        if (
            multi_source
            and scorer != "exact"
            and p.col_type == "name"
            and 0 < p.cardinality_ratio < 0.5
            and not _is_person_name_column(p.name)
        ):
            logger.info(
                "Demoting weighted field '%s' to blocking-only "
                "(multi-source: low-cardinality non-person-name shared attribute, "
                "cardinality_ratio=%.4f). Avoids collapsing distinct people who "
                "share this value; column remains a blocking candidate.",
                p.name, p.cardinality_ratio,
            )
            continue

        mf = MatchkeyField(
            field=p.name,
            scorer=scorer,
            weight=weight,
            transforms=transforms,
        )

        # #1207 PR2a: arm the data-driven name-frequency downweight. The
        # name_freq_weighted_jw scorer applies a per-dataset common-value
        # downweight ONLY when handed a frequency table; without this it falls
        # back to the static US-Census surname path. Build the table from the
        # SAME (df, transforms) the scorer sees at score time: find_fuzzy_matches
        # scores apply_transforms(value, field.transforms), and these transforms
        # include `lowercase`, which neutralizes the `name_proper` standardizer
        # auto-config also emits (title-case only) -- so raw-df-plus-transforms
        # keys align byte-for-byte with the post-standardization scored values.
        # Skip when the table has no signal (<2 distinct values) or df is absent.
        if (
            scorer == "name_freq_weighted_jw"
            and df is not None
            and _tf_name_weighting_enabled()
        ):
            from goldenmatch.core.tf_tables import value_frequencies  # noqa: PLC0415

            _tf = value_frequencies(df, p.name, transforms)
            if len(_tf) >= 2:
                mf.tf_freqs = _tf

        if scorer == "exact":
            exact_fields.append(mf)
        else:
            fuzzy_fields.append(mf)

    # Aggregate warning: if every exact-eligible column was filtered out,
    # explain which ones and why. This is the load-bearing surface that tells
    # a notebook user their auto-config silently degraded to fuzzy-only.
    _exact_eligible = [
        p for p in profiles
        if p.col_type not in ("numeric", "date", "description")
        and _SCORER_MAP.get(p.col_type, (None,))[0] == "exact"
    ]
    if _exact_eligible and not exact_fields:
        if skipped_exact:
            detail = "; ".join(f"{col} ({why})" for col, why in skipped_exact)
        else:
            # All exact-eligible columns were filtered before reaching the
            # named skip paths above — e.g. by a source-overlap check, a
            # dropped profile, or a scorer_info lookup miss. Surface this
            # shape loudly so a future refactor that starts dropping columns
            # silently can be noticed.
            eligible_names = ", ".join(p.name for p in _exact_eligible)
            detail = (
                f"no per-column reason captured — eligible columns "
                f"({eligible_names}) were filtered before reaching the "
                f"exact-matchkey skip paths"
            )
        logger.warning(
            "All %d exact-eligible columns were excluded by auto-config guards "
            "(%s). Falling back to fuzzy-only matchkeys — if any of these "
            "columns actually are identifiers, provide an explicit config.",
            len(_exact_eligible), detail,
        )

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

    # Composite "strong identity" matchkey (NCVR recall fix).
    # Individual name/year columns rarely clear the cardinality>=0.5 exact gate
    # (and year is blocking-only), so person data degrades to a single weighted
    # matchkey that averages every field — one corrupted field (e.g. an
    # abbreviated address) then sinks an otherwise-clean true pair. Their
    # COMBINATION is highly unique, so an exact matchkey on name+dob recovers
    # those pairs WITHOUT loosening any fuzzy scorer (keeps DQbench precision).
    # Matchkeys are OR'd, so this only adds candidate pairs.
    _name_fields = sorted(
        [p for p in profiles if p.col_type == "name" and p.null_rate < 0.3],
        key=lambda p: p.cardinality_ratio, reverse=True,
    )[:2]
    _date_fields = [
        p for p in profiles
        if p.col_type in ("year", "date") and p.null_rate < 0.3
    ]
    # Gate on a DOB/year anchor: a name+date combination is a real identity
    # signal, but names ALONE collide heavily (measured on DQbench tier3:
    # first+last keys pile up to max-group-13 / 85% of rows in multi-record
    # groups, vs name+DOB on NCVR at max-group-4). Requiring a date anchor
    # keeps the composite from manufacturing false merges on name-collision
    # -heavy / adversarial data that has no DOB to disambiguate.
    if len(_name_fields) >= 1 and len(_date_fields) >= 1:
        composite_fields = [
            MatchkeyField(field=p.name, transforms=["lowercase", "strip"])
            for p in (_name_fields + _date_fields)
        ]
        matchkeys.append(MatchkeyConfig(
            name="exact_identity",
            type="exact",
            fields=composite_fields,
        ))
        # Phonetic sibling (#491 heuristic path): same name+DOB composite but
        # with a `soundex` transform on the name fields, so phonetically-equal
        # spellings (Smith/Smyth, Catherine/Katherine) that the EXACT composite
        # misses still match — provided the DOB anchor agrees. name-only soundex
        # keys collide far too much to be an identity claim, but soundex(name)+DOB
        # stays specific (adversarial data without a DOB gets neither composite).
        # OR'd in, so it only adds candidate pairs.
        #
        # SCALE GATE (#510): require a SPECIFIC anchor (enough distinct values),
        # not just a date-typed one. soundex collapses the name's cardinality to a
        # bounded code space, so anchoring on a low-cardinality YEAR (~65-130
        # distinct values) leaves soundex(name)+year non-specific. The spurious
        # cross-cluster matches it manufactures grow like n_rows^2 / selectivity,
        # so on a year anchor it stays invisible on a small sample but DEGRADES
        # PRECISION and explodes soundex block sizes (-> OOM) as the dataset grows
        # (#510 audit: phonetic+year precision fell 0.91->0.82 over 1K->1M and
        # OOM'd at 25M). A full DOB stays specific. The name classifier labels a
        # year column "date" too (the column name matches the date pattern), so
        # col_type alone can't tell year from DOB -- gate on the anchor's distinct
        # count (via the sample df; cardinality_ratio fallback when df is None).
        # The EXACT-name composite above is unaffected (real names stay specific);
        # year-anchored data still gets exact_identity + the fuzzy matchkey, just
        # not the unscalable soundex identity claim.
        def _anchor_specific(p: ColumnProfile) -> bool:
            if df is not None and p.name in df.columns:
                return int(df[p.name].n_unique()) >= 150
            return p.cardinality_ratio >= 0.2  # sample year ~0.05, DOB ~0.5+
        _phonetic_anchor = [p for p in _date_fields if _anchor_specific(p)]
        if _phonetic_anchor:
            phonetic_fields = [
                MatchkeyField(field=p.name, transforms=["lowercase", "strip", "soundex"])
                for p in _name_fields
            ] + [
                MatchkeyField(field=p.name, transforms=["lowercase", "strip"])
                for p in _phonetic_anchor
            ]
            matchkeys.append(MatchkeyConfig(
                name="phonetic_identity",
                type="exact",
                fields=phonetic_fields,
            ))

    # Dual-composite: also emit a name+DOB composite keyed on person-name-PATTERN
    # columns (given_name/surname/first/last) when they exist and differ from the
    # cardinality pick above. The data classifier sometimes mislabels address
    # columns as col_type="name" (Febrl3: address_1/address_2 outrank the real
    # names by cardinality), so the cardinality composite can key on addresses.
    # Different datasets corrupt different fields (Febrl3 mangles names, NCVR
    # mangles addresses), so anchoring on BOTH field-sets means a true pair clean
    # on EITHER matches. Same DOB gate; OR'd, so it only adds candidate pairs.
    _pattern_name_fields = sorted(
        [p for p in profiles if p.null_rate < 0.3 and _classify_by_name(p.name) == "name"],
        key=lambda p: p.cardinality_ratio, reverse=True,
    )[:2]
    if (_date_fields and _pattern_name_fields
            and {p.name for p in _pattern_name_fields} != {p.name for p in _name_fields}):
        matchkeys.append(MatchkeyConfig(
            name="exact_identity_name",
            type="exact",
            fields=[
                MatchkeyField(field=p.name, transforms=["lowercase", "strip"])
                for p in (_pattern_name_fields + _date_fields)
            ],
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

    # Limit fuzzy fields to prevent OOM on wide datasets
    # Rank by match utility: cardinality * completeness * string length
    max_fuzzy_fields = 5
    if len(all_weighted) > max_fuzzy_fields:
        profile_lookup = {p.name: p for p in profiles}

        def _field_utility(f: MatchkeyField) -> float:
            if not f.field or f.field not in profile_lookup:
                return f.weight or 0.0
            p = profile_lookup[f.field]
            return p.cardinality_ratio * (1 - p.null_rate) * min(p.avg_len / 20, 1.0)

        all_weighted.sort(key=_field_utility, reverse=True)
        dropped = [f.field for f in all_weighted[max_fuzzy_fields:] if f.field]
        all_weighted = all_weighted[:max_fuzzy_fields]
        logger.info(
            "Truncated fuzzy fields from %d to %d. Dropped: %s",
            len(all_weighted) + len(dropped), max_fuzzy_fields, dropped,
        )

    # Confidence-gated weighting: when a profile's classification confidence
    # is low (<0.5), cap the weight at 0.3 so noisy/ambiguous columns can't
    # dominate a weighted matchkey. Profile lookup is by column name.
    # Ordering note: this cap runs AFTER field-utility truncation above.
    # _field_utility uses f.weight only as a fallback when a profile is
    # missing; more importantly, we don't want the cap to skew the utility
    # ranking used by the truncation step. Reorder at your peril.
    _profile_lookup = {p.name: p for p in profiles}
    for f in all_weighted:
        if f.field is None:
            continue
        prof = _profile_lookup.get(f.field)
        if prof is not None and prof.confidence < 0.5:
            if (f.weight or 0) > 0.3:
                f.weight = 0.3

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

    _promote_facility_fullname_ne(matchkeys, skipped_exact, _person_name_basket)
    return matchkeys


_GIVEN_NAME_RE = re.compile(r"(first[_ ]?name|given[_ ]?name|fname|^first$|forename)", re.I)
_FAMILY_NAME_RE = re.compile(r"(last[_ ]?name|surname|lname|family[_ ]?name)", re.I)
_PERSON_FULLNAME_COL = "__gm_person_fullname__"


def _facility_name_ne_enabled() -> bool:
    return os.environ.get("GOLDENMATCH_FACILITY_NAME_NE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _promote_facility_fullname_ne(
    matchkeys: list[MatchkeyConfig],
    skipped_exact: list[tuple[str, str]],
    person_name_basket: list[tuple[str, bool]],
) -> None:
    """Opt-in (``GOLDENMATCH_FACILITY_NAME_NE=1``): on company/location-mode
    data, add the person FULL NAME as negative evidence on weighted matchkeys.

    Rationale: when a workplace attribute (phone/address/company) was demoted
    because the records sharing it do NOT co-agree on the person name
    (``should_demote_attribute_field``), the weighted matchkey still carries
    address/company (deliberately -- load-bearing for corruption-heavy recall),
    so two distinct colleagues at one clinic can still fuse. A full-name NE
    penalises a merge when the whole name is clearly different.

    Full name, not the given name: ``jaro_winkler`` on a given name alone can't
    separate colleagues (jw(mark,laura)=0.63) from nicknames (jw(robert,bob)=0.50).
    ``token_sort`` on the concatenated name separates cleanly -- colleagues
    0.48-0.60, every duplicate flavour (nickname 0.76 / truncation 0.71 / typo
    0.92) at/above 0.65 -- and is corruption-safe (febrl3's typo'd names stay
    high, so its true duplicates are never penalised). The full name is a
    synthesized column (``NegativeEvidenceField.derive_from``) materialized by
    ``precompute_matchkey_transforms``.
    """
    if not _facility_name_ne_enabled():
        return
    facility_mode = any(
        "group-attribute" in reason or "facility" in reason
        for _col, reason in skipped_exact
    )
    if not facility_mode:
        return
    name_cols = [c for c, _flag in person_name_basket]
    if len(name_cols) < 2:
        return
    family = next((c for c in name_cols if _FAMILY_NAME_RE.search(c)), None)
    given = next((c for c in name_cols if _GIVEN_NAME_RE.search(c)), None)
    sources = [given, family] if (given and family) else name_cols[:2]
    for mk in matchkeys:
        if mk.type != "weighted":
            continue
        if any(n.field == _PERSON_FULLNAME_COL for n in (mk.negative_evidence or [])):
            continue
        ne = list(mk.negative_evidence or [])
        ne.append(NegativeEvidenceField(
            field=_PERSON_FULLNAME_COL,
            transforms=["lowercase", "strip"],
            scorer="token_sort",
            threshold=0.65,
            penalty=0.5,
            derive_from=sources,
        ))
        mk.negative_evidence = ne
        logger.info(
            "auto-config[facility]: promoted full-name negative evidence "
            "(token_sort on %s) on weighted matchkey '%s' -- clearly-different "
            "colleagues at a shared facility should not fuse.",
            sources, mk.name,
        )


# ── Strong-identifier blocking-union (#1207) ──────────────────────────────
# Coverage is restored by the OR across passes, so a per-id pass is admitted on
# a minimal non-null population floor (NOT a null ceiling) — that keeps high-null
# phone/zip passes that each block only the rows that *have* that id. Scale-safety
# is enforced by the caller via _gate_passes, so it is NOT checked here. Strong-id
# col_types reuse the module-level `_STRONG_EXACT_TYPES` (identifier/email/phone).
_BLOCKING_UNION_COVERAGE_TARGET = 0.95
_UNION_PASS_MIN_NONNULL = 0.02  # a pass must block more than a trivial handful


def _union_coverage(df: pl.DataFrame, pass_field_lists: list[list[str]]) -> float:
    """Fraction of rows non-null on at least one pass's fields (OR across passes).
    A multi-field pass requires ALL its fields non-null (it can't block a row
    missing any component)."""
    # W3d: the seam's coverage_ratio op is this exact fold (edge cases
    # pinned by its fixtures: missing column, empty lists, NaN non-null).
    from goldenmatch.core.frame import to_frame

    return to_frame(df).coverage_ratio(pass_field_lists)


def _build_strong_identifier_union(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    *,
    n_rows_full: int | None = None,
) -> BlockingConfig | None:
    """Emit a multi_pass UNION of one pass per strong id + name+geo, or None.

    Returns None unless >=1 strong-id pass is present AND >=2 distinct passes
    survive AND their OR-coverage clears _BLOCKING_UNION_COVERAGE_TARGET. The
    >=1 strong-id requirement keeps this from emitting a name-only "strong-id
    union" (a name-only shape is left to the existing name-fallback path).
    Caller (build_blocking) is responsible for invoking this only on the
    fall-through (no single key passed the strict 0.20 ceiling).

    `n_rows_full` is reserved for call-site signature parity; scale-safety is
    enforced by the caller via `_gate_passes`, not here."""
    def _nonnull(col: str) -> float:
        from goldenmatch.core.frame import to_frame

        frame = to_frame(df)
        return 1.0 - (frame.column(col).null_count() / frame.height) if frame.height else 0.0

    candidate_passes: list[list[str]] = []

    # one pass per strong-identifier field, above the non-null population floor.
    # No scale-safety check here — _gate_passes at the call site enforces #715.
    strong_id_passes = 0
    for p in profiles:
        if p.col_type in _STRONG_EXACT_TYPES and p.name in df.columns:
            if _nonnull(p.name) < _UNION_PASS_MIN_NONNULL:
                continue
            # #876 surrogate guard: a perfect-surrogate id makes singleton blocks
            # (0 pairs) — exclude. NOTE: do NOT apply blocking_max_ratio here; the
            # union exists precisely to use near-unique-but-repeating ids the
            # single-key gate rejects — which is also why this must ask
            # `_is_perfect_surrogate` (exact, full-frame) rather than the sampled
            # ratio, whose drift toward 1.0 at scale rejects exactly those ids.
            if _is_perfect_surrogate(p):
                continue
            candidate_passes.append([p.name])
            strong_id_passes += 1

    # require >=1 strong-id pass; a name-only shape belongs to the name fallback.
    if strong_id_passes < 1:
        return None

    # name+geo passes for rows missing every strong id
    name_cols_local = [p for p in profiles if _classify_by_name(p.name) == "name"]
    first = next((p.name for p in name_cols_local if "first" in p.name.lower()), None)
    last = next((p.name for p in name_cols_local if "last" in p.name.lower()
                 or "surname" in p.name.lower()), None)
    geo = next((p.name for p in profiles if p.col_type in ("zip", "geo")), None)
    if first and last:
        candidate_passes.append([first, last])
    if last and geo:
        candidate_passes.append([last, geo])

    if len(candidate_passes) < 2:
        return None
    if _union_coverage(df, candidate_passes) < _BLOCKING_UNION_COVERAGE_TARGET:
        return None

    def _transforms_for(fields: list[str]) -> list[str]:
        prof = next((p for p in profiles if p.name == fields[0]), None)
        return (
            ["lowercase", "strip"]
            if prof and prof.col_type in ("email", _EXACT_DERIVED_COL_TYPE)
            else ["strip"]
        )

    passes = [BlockingKeyConfig(fields=f, transforms=_transforms_for(f))
              for f in candidate_passes]
    return BlockingConfig(
        keys=[passes[0]],
        strategy="multi_pass",
        passes=passes,
        skip_oversized=True,
    )


def _is_strong_identifier_union(
    cfg: BlockingConfig, profiles: list[ColumnProfile]
) -> bool:
    """True if ``cfg`` is a #1207 strong-identifier UNION.

    Shape: a ``multi_pass`` config with >=2 passes, at least one of which is a
    single strong-id field (identifier/email/phone). Both union build paths
    (the Rust core union and ``_build_strong_identifier_union``) emit exactly
    this shape, and no other autoconfig blocking path emits a single-field
    strong-id pass inside a multi_pass config (compound/name passes are
    multi-field; quality-aware/semantic passes are soundex/substring/simhash).
    Used by the >=50K learned-blocking gate to avoid clobbering the union.
    """
    if cfg.strategy != "multi_pass" or not cfg.passes or len(cfg.passes) < 2:
        return False
    col_types = {p.name: p.col_type for p in profiles}
    return any(
        len(p.fields) == 1 and col_types.get(p.fields[0]) in _STRONG_EXACT_TYPES
        for p in cfg.passes
    )


# ── Compound blocking helpers ─────────────────────────────────────────────


def _build_compound_blocking(
    profiles: list[ColumnProfile],
    df: pl.DataFrame | Any,  # #1852: also accepts pa.Table (routed via to_frame)
    max_safe_block: int,
    max_null_rate: float,
) -> BlockingConfig | None:
    """Try to build compound blocking keys when single columns are all oversized.

    Uses greedy refinement: pick the best single column, then find the second
    column that reduces max block size the most. Generates multi-pass compound
    keys for recall.

    Returns None if no compound pair brings blocks below max_safe_block.
    """
    # #1852: route every column/group op through the backend-neutral Frame
    # seam so this helper works on BOTH a polars ``pl.DataFrame`` and an Arrow
    # ``pa.Table`` (the latter is the default input since the arrow-native
    # autoconfig path, `GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE=1`). The raw
    # `df[col].null_count()` / `df.height` / `df.group_by(...)` polars idioms
    # AttributeError'd on a `pa.Table`.
    from goldenmatch.core.frame import to_frame

    frame = to_frame(df)

    def _null_rate(col_name: str) -> float:
        return frame.column(col_name).null_count() / frame.height if frame.height > 0 else 0.0

    def _max_block_size(col_name: str) -> int:
        """Largest group size when blocking on this column (W3d: seam)."""
        return int(frame.group_len([col_name]).column("len").max() or 0)

    def _nonnull_ratio(col_name: str) -> float:
        """Distinct/non-null ratio -- the TRUE per-record uniqueness, not the
        null-deflated ColumnProfile.cardinality_ratio. A near-1.0 value means a
        surrogate-key-like column (npi/phone/email) whose only big "block" is
        its null bucket -- useless as a blocking component."""
        nn = frame.column(col_name).drop_nulls()
        n = len(nn)
        return (nn.n_unique() / n) if n > 0 else 1.0

    # #715: judge compound COMPONENTS by whether they BOUND block size (i.e.
    # actually group records) and aren't surrogate keys -- NOT by col_type.
    # A sparse zip5 reclassifies `numeric -> identifier` and (at ~50% null)
    # exceeds the single-key null ceiling, so the old col_type/null filters
    # doubly excluded it, leaving only oversized name columns. As a compound
    # COMPONENT, a high-null column is fine: the multi_pass config's other
    # passes cover the null rows. So:
    #   - keep `numeric`/`date` excluded;
    #   - admit `identifier`, `zip`, and the high-cardinality `email`/`phone`
    #     types ONLY when the column genuinely GROUPS records: non-singleton
    #     blocks (`_max_block_size > 1`), `cardinality_ratio < 1.0`, and a
    #     non-null distinct ratio below the blocking gate (rejects surrogate keys
    #     like npi/phone/email whose non-null values are ~unique per record and
    #     whose only large block is the null bucket);
    #   - relax the single-key null ceiling to 0.6 for the component role so a
    #     ~50%-null zip5 qualifies.
    # `zip` is admitted here (not just `identifier`) so the choice doesn't depend
    # on whether the sampling artifact promoted the column to `identifier`: the
    # MEASURED guards decide. A moderate-cardinality zip (true distinct ratio
    # ~0.3) that pairs with a name to bound the block is a strong compound
    # component regardless of its type label; a near-unique surrogate zip is
    # still rejected by the non-null-ratio guard. This decouples blocking-key
    # selection from the over-eager identifier classification (S2a corrected the
    # latter; this aligns blocking with the comment's stated intent: judge by
    # whether a component GROUPS records, NOT by col_type).
    # NOTE: `_nonnull_ratio` / `_max_block_size` are computed on `df` directly.
    # On the v0 non-distributed path `df` IS the full dataset (controller passes
    # the full frame to `_initial_config`), so these are exact. If a SAMPLED df
    # is ever fed here (distributed path), the unprojected non-null ratio is
    # sample-inflated and would wrongly reject a mid-cardinality column -- such a
    # caller must Chao1-project the ratio (see scale_cardinality_ratio_to_full_population).
    # Tracked as a distributed-path follow-up; out of scope for #715 (single-node).
    from goldenmatch.core.blocking_candidates import _blocking_max_ratio
    _grouping_ratio_max = _blocking_max_ratio()
    _component_null_ceiling = max(max_null_rate, 0.6)
    _high_card_types = ("identifier", "zip", "email", "phone", _EXACT_DERIVED_COL_TYPE)

    def _is_admissible(p: ColumnProfile) -> bool:
        # numeric/date used to be rejected outright. That excluded exactly the
        # kind of column that makes the best compound COMPONENT: on the real
        # DBLP-ACM data `year` profiles as numeric with unique_rate 0.0038 (10
        # distinct over 4910 rows, zero nulls), and adding it to `__title_key__`
        # cuts candidate pairs 33,563 -> 5,749 at IDENTICAL recall (0.9717),
        # lifting the precision ceiling 0.064 -> 0.376 (#2633, measured).
        #
        # The blanket veto also contradicted the reasoning below, which admits
        # `zip` on the opposite principle -- judge a component by whether it
        # GROUPS records, not by col_type -- and then applies measured guards.
        # numeric/date now take that same route: they must group
        # (max_block > 1), not be a surrogate, and clear the grouping ratio,
        # which still rejects a numeric ID or a continuous price.
        if p.col_type in ("numeric", "date"):
            if not (
                _max_block_size(p.name) > 1
                and not _is_perfect_surrogate(p)
                and _nonnull_ratio(p.name) < _grouping_ratio_max
            ):
                return False
            return _null_rate(p.name) <= _component_null_ceiling
        if _check_source_overlap(df, p.name) <= 0.0:
            return False
        if p.col_type in _high_card_types:
            # Surrogate-key / near-unique guard: must actually group records.
            if not (
                _max_block_size(p.name) > 1
                and not _is_perfect_surrogate(p)
                and _nonnull_ratio(p.name) < _grouping_ratio_max
            ):
                return False
            # High-null is OK for a compound component (other passes cover nulls).
            return _null_rate(p.name) <= _component_null_ceiling
        # Low-cardinality types (name/string/geo/...): keep the stricter ceiling.
        return _null_rate(p.name) <= max_null_rate

    candidates = [p for p in profiles if _is_admissible(p)]
    if len(candidates) < 2:
        return None

    # Sort by cardinality descending — best single column first
    candidates.sort(key=lambda p: frame.column(p.name).n_unique(), reverse=True)
    best = candidates[0]

    # Test compound pairs: best + each other candidate (up to 5)
    pair_results: list[tuple[ColumnProfile, int]] = []
    for other in candidates[1:6]:
        try:
            max_block = int(frame.group_len([best.name, other.name]).column("len").max() or 0)
            pair_results.append((other, max_block))
            logger.debug(
                "Compound pair [%s, %s]: max_block=%d",
                best.name, other.name, max_block,
            )
        except Exception:
            continue

    if not pair_results:
        return None

    # Sort by max block ascending — smallest (safest) first
    pair_results.sort(key=lambda x: x[1])
    winner, winner_block = pair_results[0]

    if winner_block > max_safe_block:
        logger.info(
            "Best compound pair [%s, %s] still produces blocks of %d (> %d). "
            "No compound key is safe enough.",
            best.name, winner.name, winner_block, max_safe_block,
        )
        return None

    logger.info(
        "Compound blocking: [%s, %s] -> max_block=%d",
        best.name, winner.name, winner_block,
    )

    # Build multi-pass config for recall
    passes = [
        # Pass 1: winning compound pair
        BlockingKeyConfig(fields=[best.name, winner.name], transforms=["lowercase", "strip"]),
    ]

    # Pass 2: runner-up compound pair (if different and safe)
    if len(pair_results) > 1:
        runner_up, runner_up_block = pair_results[1]
        if runner_up_block <= max_safe_block and runner_up.name != winner.name:
            passes.append(
                BlockingKeyConfig(fields=[best.name, runner_up.name], transforms=["lowercase", "strip"]),
            )

    # Pass 3: recall-focused single-column soundex (relies on skip_oversized)
    passes.append(
        BlockingKeyConfig(fields=[best.name], transforms=["lowercase", "soundex"]),
    )

    return BlockingConfig(
        keys=[passes[0]],
        strategy="multi_pass",
        passes=passes,
        max_block_size=max_safe_block,
        skip_oversized=True,
    )


def _call_llm_for_blocking(prompt: str, provider: str) -> str:
    """Call LLM API for blocking key suggestion. Returns raw response text.

    Uses stdlib urllib (same pattern as llm_scorer.py) — no external deps.
    """
    import json as _json
    import os
    import urllib.request

    _MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}
    model = os.environ.get("GOLDENMATCH_LLM_MODEL", _MODELS.get(provider, ""))

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        body = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        body = _json.dumps({
            "model": model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["content"][0]["text"]

    raise ValueError(f"Unknown provider: {provider}")


def _llm_suggest_blocking_keys(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    provider: str,
    max_safe_block: int,
) -> BlockingConfig | None:
    """Ask LLM to suggest compound blocking keys, then validate.

    Returns a validated BlockingConfig or None if suggestions are invalid.
    """
    # #1852: coerce to the Frame seam once so the cardinality/block stats run
    # polars-free on a ``pa.Table`` (arrow-native lane). The raw ``df[...]`` /
    # ``df.group_by(...)`` idioms below AttributeError'd on arrow; the first one
    # (``df[p.name].n_unique()``) is unguarded, so an arrow+LLM-blocking run
    # crashed outright rather than degrading.
    from goldenmatch.core.frame import to_frame as _tf

    frame = _tf(df)
    frame_height = frame.height

    # Build prompt with cardinality stats (all non-numeric columns, including date)
    col_stats = []
    for p in profiles:
        if p.col_type == "numeric":
            continue
        n_unique = frame.column(p.name).n_unique()
        max_block = frame.group_len([p.name]).column("len").max()
        col_stats.append(
            f"  {p.name}: type={p.col_type}, {n_unique:,} unique / {frame_height:,} rows, "
            f"max_block={max_block:,}"
        )

    prompt = (
        "You are a data deduplication expert. Given these column profiles with cardinality stats:\n"
        + "\n".join(col_stats)
        + f"\n\nDataset: {frame_height:,} rows. Max safe block size: {max_safe_block:,}.\n"
        "Suggest 2-3 multi-pass compound blocking key combinations.\n"
        "Each pass: 2 columns that together keep max block under the safe limit.\n"
        "Prioritize recall — different passes should cover different match scenarios "
        "(e.g., same model different location vs same model different year).\n\n"
        'Return JSON: {"passes": [{"fields": ["col_a", "col_b"], "reason": "..."}, ...]}'
    )

    try:
        raw = _call_llm_for_blocking(prompt, provider)
    except Exception as e:
        logger.warning("LLM blocking key suggestion failed: %s", e)
        return None

    # Parse JSON
    import json as _json
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = _json.loads(text)
    except (ValueError, KeyError) as e:
        logger.warning("LLM returned invalid JSON for blocking keys: %s", e)
        return None

    suggested_passes = data.get("passes", [])
    if not suggested_passes:
        logger.warning("LLM returned empty passes list")
        return None

    # Validate each suggestion
    valid_columns = set(frame.columns)
    validated_passes: list[BlockingKeyConfig] = []

    for suggestion in suggested_passes:
        fields = suggestion.get("fields", [])
        reason = suggestion.get("reason", "")

        if not all(f in valid_columns for f in fields):
            bad = [f for f in fields if f not in valid_columns]
            logger.info("LLM suggestion rejected — unknown columns: %s", bad)
            continue

        try:
            max_block = int(frame.group_len(fields).column("len").max() or 0)  # #1852: seam
        except Exception:
            logger.info("LLM suggestion rejected — group_by failed for %s", fields)
            continue

        if max_block > max_safe_block:
            logger.info(
                "LLM suggestion [%s] rejected — max_block=%d > %d. Reason: %s",
                fields, max_block, max_safe_block, reason,
            )
            continue

        logger.info(
            "LLM suggestion accepted: [%s] -> max_block=%d. Reason: %s",
            fields, max_block, reason,
        )
        validated_passes.append(
            BlockingKeyConfig(fields=fields, transforms=["lowercase", "strip"])
        )

    if not validated_passes:
        logger.info("All LLM blocking key suggestions were rejected")
        return None

    return BlockingConfig(
        keys=[validated_passes[0]],
        strategy="multi_pass",
        passes=validated_passes,
        max_block_size=max_safe_block,
        skip_oversized=True,
    )


# ── Cross-source overlap ──────────────────────────────────────────────────


def _check_source_overlap(
    df: pl.DataFrame, col: str, partition_col: str = "__source__"
) -> float:
    """Compute value overlap ratio for a column across source partitions.

    Returns |intersection| / |union| of unique values per partition.
    Returns 1.0 if ``partition_col`` is absent or has only one partition
    (no check needed). ``partition_col`` defaults to the internal
    ``__source__`` so existing callers are unchanged; #858 passes a detected
    user source column when there is no ``__source__``.
    """
    from goldenmatch.core.frame import to_frame

    frame = to_frame(df)
    if partition_col not in frame.columns:
        return 1.0

    sources = frame.column(partition_col).unique().to_list()
    if len(sources) < 2:
        return 1.0

    value_sets = []
    for src in sources:
        vals = set(
            frame.filter_eq(partition_col, src)
            .column(col)
            .drop_nulls()
            .cast_str()
            .to_list()
        )
        value_sets.append(vals)

    intersection = value_sets[0]
    union = value_sets[0]
    for vs in value_sets[1:]:
        intersection = intersection & vs
        union = union | vs

    if not union:
        return 1.0

    return len(intersection) / len(union)


# ── #858: multi-source over-merge guard ──────────────────────────────────────

def _source_disjoint(df: pl.DataFrame, col: str, partition_col: str) -> bool:
    """True iff no value of ``col`` appears under 2+ distinct partitions.

    The "fully source-specific" signal #858 uses to identify provenance /
    per-source surrogate columns (the source label, per-source ids). Unlike
    ``_check_source_overlap`` (all-partition intersection, which is 0 even for a
    real identifier shared between only SOME of >2 sources), this is a
    max-pairwise test: a genuine shared identifier (the same person appearing in
    two sources with the same value) is NOT disjoint, so it is kept.
    """
    from goldenmatch.core.frame import to_frame

    _sd_frame = to_frame(df)  # arrow-port: seam columns (pa.Table.columns != names)
    if partition_col not in _sd_frame.columns:
        return False
    # W3d: group_nunique pins the either-col-null row drop this site's
    # frame-level drop_nulls performed.
    per_value = _sd_frame.group_nunique(col, partition_col)
    if per_value.height == 0:
        return False
    max_parts = cast("int | None", per_value.column("n_unique").max())
    return max_parts is None or max_parts <= 1


# Bounded source-indicator name patterns (case-insensitive). NOT a general
# provenance regex (name-regex was rejected as the primary mechanism, spec §1);
# this only LOCATES a user source partition when there is no ``__source__``.
# Deliberately excludes business-attribute-ish names (channel / system / crm)
# that are plausibly real low-card match signal, not data origin -- they were a
# false-positive vector flagged in spec review.
_SOURCE_NAME_RE = re.compile(
    r"(?i)^(source|origin|src|data_source|record_source|lead_source)$|_source$"
)


def _detect_source_partition(
    df: pl.DataFrame, profiles: list[ColumnProfile]
) -> str | None:
    """Return the column that partitions records by origin, or ``None``.

    ``None`` => single-source / match mode / killed => the whole #858 feature is
    a no-op (the regression firewall). See the design spec §0-§1.
    """
    if not _multisource_autoconfig_enabled():
        return None
    if _AUTOCONFIG_MATCH_MODE.get():
        return None
    from goldenmatch.core.frame import to_frame

    _sp_frame = to_frame(df)  # arrow-port: seam columns/height/n_unique
    # 1) internal __source__ with >= 2 distinct values (multi-file dedupe).
    if (
        "__source__" in _sp_frame.columns
        and _sp_frame.column("__source__").n_unique() >= 2
    ):
        return "__source__"
    # 2) a name-pattern user column: low cardinality + >= 2 distinct + a
    #    disjoint-id co-signature (>= 1 other column 0-overlap vs it). The
    #    co-signature is the real discriminator; the cardinality cap (absolute
    #    floor of 20, else 10% of rows) excludes free-text "source"-ish fields
    #    while staying robust on small frames.
    other_cols = [p.name for p in profiles if not p.name.startswith("__")]
    card_cap = max(20, int(0.1 * _sp_frame.height))
    for p in profiles:
        if p.name.startswith("__") or not _SOURCE_NAME_RE.search(p.name):
            continue
        n_unique = _sp_frame.column(p.name).n_unique()
        if n_unique < 2 or n_unique > card_cap:
            continue
        has_cosig = any(
            other != p.name and _source_disjoint(df, other, p.name)
            for other in other_cols
        )
        if has_cosig:
            return p.name
    return None


def _source_correlated_exclusions(
    df: pl.DataFrame,
    profiles: list[ColumnProfile],
    partition: str | None,
) -> set[str]:
    """Columns to exclude from ALL match features when multi-source (spec §3).

    = {the user source-indicator column} U {columns 0-overlap across sources}.
    Computed on the FULL frame so ``== 0.0`` is exact. Empty when no partition.
    ``__source__`` is never added (``profile_columns`` already skips dunder
    columns, so it never reaches the matchkeys).
    """
    if partition is None:
        return set()
    exclude: set[str] = set()
    if not partition.startswith("__"):
        exclude.add(partition)
    for p in profiles:
        col = p.name
        if col.startswith("__") or col == partition:
            continue
        if _source_disjoint(df, col, partition):
            exclude.add(col)
    return exclude


# ── Blocking generation ────────────────────────────────────────────────────

def _make_quality_column_profile(
    autoconfig_profile: ColumnProfile,
    n_rows: int,
) -> Any:
    """Bridge ``autoconfig.ColumnProfile`` -> ``quality_exclusions.ColumnProfile``.

    The two dataclasses share ``cardinality_ratio`` + ``null_rate`` +
    ``dtype`` but autoconfig's version doesn't carry ``distinct_count``
    or ``mean_string_length``. Project them from the available fields.

    Used to feed ``classify_column_role`` / ``find_composite_blocking_keys``
    from inside ``build_blocking`` without a cross-module refactor.
    """
    from goldenmatch.core.quality_exclusions import (
        ColumnProfile as _QualityColumnProfile,
    )
    distinct = max(int(autoconfig_profile.cardinality_ratio * max(n_rows, 1)), 1)
    return _QualityColumnProfile(
        cardinality_ratio=autoconfig_profile.cardinality_ratio,
        null_rate=autoconfig_profile.null_rate,
        distinct_count=distinct,
        dtype=autoconfig_profile.dtype,
        mean_string_length=(
            autoconfig_profile.avg_len if autoconfig_profile.avg_len > 0 else None
        ),
    )


def _pick_date_blocking_col(
    profiles: list[ColumnProfile],
    null_rate_fn: Callable[[str], float],
    *,
    max_null_rate: float = 0.20,
) -> str | None:
    """#438: pick a date column suitable as a blocking pass.

    Returns the first profile whose ``col_type == "date"`` (or
    name-classified as a date by `_classify_by_name`) and whose null
    rate is <= `max_null_rate`. Date-only blocking catches pairs that
    share birthdates but disagree on geo+name fields, which is the
    dominant recall-loss case on synthetic-typo data (Febrl3).

    Returns None when no date column qualifies.
    """
    for p in profiles:
        is_date_by_type = getattr(p, "col_type", None) == "date"
        is_date_by_name = _classify_by_name(p.name) == "date"
        if not (is_date_by_type or is_date_by_name):
            continue
        try:
            if null_rate_fn(p.name) > max_null_rate:
                continue
        except Exception:
            continue
        return p.name
    return None


def _degenerate_blocking_config(max_safe_block: int) -> BlockingConfig:
    """Empty-keys blocking config: the #715 degenerate signal.

    Emitted when every candidate blocking key/pass projects oversized at
    full N. Empty ``keys`` is exactly what the controller's #417 degenerate
    guard inspects (``not blocking.keys`` at ``>= REFUSE_AT_N``), so it
    refuses loudly rather than shipping a candidate-pair bomb. ``auto_suggest``
    keeps the model valid (the validator bypasses the keys-required check for
    auto_suggest) and matches the established empty-keys pattern used by the
    CLI / preview paths.

    Note: the #417 guard refuses on empty keys only when the committed profile
    is RED (``_no_blocking_keys AND _profile_red``); a GREEN/YELLOW profile
    with empty keys would not refuse on that branch (unconditional RED refusal
    is handled separately by the ``allow_red_config`` work).
    """
    return BlockingConfig(
        keys=[],
        auto_suggest=True,
        max_block_size=max_safe_block,
        skip_oversized=True,
    )


# --- Quality-aware blocking (GoldenCheck -> GoldenMatch door #1) -------------
# Spec: docs/design/2026-06-07-quality-aware-blocking-design.md
#
# Edit-distance value variants that survive lowercase/strip ("Californa" vs
# "California") shard true duplicates into different exact blocks -> recall lost
# before scoring. GoldenCheck's per-column fuzziness (via core.quality.
# blocking_risk) lets us ADD a fuzzy-tolerant pass for an affected blocking key
# so the variants co-block. Purely additive (the original key is retained), so
# recall can only rise; precision is unchanged (scoring still decides matches).

# A column is "fuzzy enough to matter" at >= 2% variant rows.
_BLOCKING_RISK_NORMALIZE = 0.02
# Transforms already tolerant of edit-distance variants (don't double up).
_FUZZY_TOLERANT_TRANSFORMS = ("soundex", "metaphone")


def _quality_aware_blocking_enabled() -> bool:
    """v1 is OFF by default (spec §7) until the Febrl/DBLP-ACM/NCVR sweep proves
    recall-up / no-precision-regression. Opt in with
    ``GOLDENMATCH_QUALITY_AWARE_BLOCKING=1`` (the #662 kill-switch pattern)."""
    val = os.environ.get("GOLDENMATCH_QUALITY_AWARE_BLOCKING")
    return val is not None and val.lower() in ("1", "true", "yes", "on")


def _cost_aware_blocking_enabled() -> bool:
    """Cost-aware primary-key selection (#2021). Default OFF (byte-identical) until
    the ``bench_er_headtohead`` panel + QIS scale-invariance prove F1-neutrality; opt
    in with ``GOLDENMATCH_BLOCKING_COST_AWARE=1`` (the #662 kill-switch pattern).

    When on, the exact-blocking "best case" branch refuses to commit a small-fixed-
    domain key (``year``/``date`` col_type) as the SOLE primary blocking key when a
    better-bounded fallback exists (a name key or a bounded compound). Such a key's
    block grows ∝N (fixed tiny domain), so it explodes candidate pairs at scale
    (birth_year: 65 distinct → ~7.7B pairs at 1M); it belongs as a recall PASS (its
    #438 role), not the primary.

    DOMAIN-ROUTED (``_is_bibliographic_dataset``): the demotion is SKIPPED on
    bibliographic data, where ``year`` is a publication year -- a legitimately strong
    same-year blocking signal (DBLP-ACM). Only person/other shapes get demoted. Spec:
    ``docs/superpowers/specs/2026-07-30-blocking-primary-key-cost-aware-design.md``."""
    val = os.environ.get("GOLDENMATCH_BLOCKING_COST_AWARE")
    return val is not None and val.lower() in ("1", "true", "yes", "on")


def _is_bibliographic_dataset(profiles: list[ColumnProfile]) -> bool:
    """True when the dataset's columns read as BIBLIOGRAPHIC (title/authors/venue/
    journal/doi/isbn/year), via ``core.domain.detect_domain`` on the column names.

    This routes the cost-aware year/date demotion (#2021): on bibliographic data a
    ``year`` column is the PUBLICATION year -- a strong shared-identity blocking
    signal (every true match is same-year, the DBLP-ACM shape), so blocking on it is
    correct and must NOT be demoted. On person/other data a ``year`` is a coarse
    ``birth_year`` whose block grows ∝N and explodes at scale, so it IS demoted. The
    person shape (first_name/last_name/email/birth) scores ``person`` here, not
    bibliographic, so demotion still applies to it. Fail-open to False (apply the
    demotion) on any detection error."""
    try:
        from goldenmatch.core.domain import detect_domain
        cols = [p.name for p in profiles if not p.name.startswith("__")]
        return detect_domain(cols).name == "bibliographic"
    except Exception:  # noqa: BLE001 -- detection is best-effort; default to demoting
        return False


def _has_fuzzy_tolerant_transform(transforms: list[str]) -> bool:
    return any(
        t in _FUZZY_TOLERANT_TRANSFORMS or t.startswith("substring:") for t in transforms
    )


def _fuzzy_pass_transforms(col_type: str) -> list[str]:
    # Phonetic for names (catches early-character typos like Jon/John); a prefix
    # block for everything else (catches typos after a shared prefix and reuses
    # an existing transform).
    if col_type == "name":
        return ["lowercase", "soundex"]
    return ["lowercase", "strip", "substring:0:6"]


def apply_quality_aware_blocking(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    *,
    enabled: bool | None = None,
) -> BlockingConfig | None:
    """Augment a blocking config with fuzzy-tolerant passes for columns
    GoldenCheck flags as edit-distance-fuzzy. Fail-open + additive: returns the
    config unchanged when disabled, goldencheck is absent, the data is clean, or
    the strategy isn't one we can safely extend (only ``static`` / ``multi_pass``
    -- the common auto-config outputs)."""
    if blocking is None:
        return blocking
    if enabled is None:
        enabled = _quality_aware_blocking_enabled()
    if not enabled or blocking.strategy not in ("static", "multi_pass"):
        return blocking

    from goldenmatch.core.quality import blocking_risk

    risk = blocking_risk(df)
    if not risk:
        return blocking

    col_type = {p.name: p.col_type for p in profiles}
    existing = list(blocking.passes or []) + list(blocking.keys or [])
    if not existing:
        return blocking

    # Fields already covered by a fuzzy-tolerant transform need no extra pass.
    already_tolerant = {
        f for kc in existing if _has_fuzzy_tolerant_transform(kc.transforms) for f in kc.fields
    }

    new_passes: list[BlockingKeyConfig] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for kc in existing:
        for fld in kc.fields:
            if fld in already_tolerant or risk.get(fld, 0.0) < _BLOCKING_RISK_NORMALIZE:
                continue
            transforms = _fuzzy_pass_transforms(col_type.get(fld, "string"))
            sig = (fld, tuple(transforms))
            if sig in seen:
                continue
            seen.add(sig)
            new_passes.append(BlockingKeyConfig(fields=[fld], transforms=transforms))

    if not new_passes:
        return blocking

    # Convert to an explicit multi_pass union: original keys/passes become passes
    # (so they survive `auto_select`, which otherwise picks a single key), plus
    # the fuzzy-tolerant passes. Every other config field is preserved.
    return blocking.model_copy(update={
        "strategy": "multi_pass",
        "passes": existing + new_passes,
        "keys": [],
    })


# ── Auto-enabled semantic blocking (door: GOLDENMATCH_AUTO_SEMANTIC_BLOCKING) ──
#
# #1090: when the data is text-heavy (a long free-text column the lexical /
# structured blocking keys under-cover) AND a semantic embedder is reachable,
# route blocking to SimHash over embeddings -- ANN-style candidate generation
# that co-blocks records that mean the same thing but share little surface text.
# Honest fallback: when no embedder is reachable this is a NO-OP (the lexical /
# structured scheme stands), never a silent degrade to a broken semantic path.
# DEFAULT ON (#1090): it fires for text-heavy data WHEN an embedder is reachable;
# absent an embedder the no-op fallback means a user without the in-house model /
# a configured provider still sees byte-identical output. Disable with
# GOLDENMATCH_AUTO_SEMANTIC_BLOCKING=0; tune recall with
# GOLDENMATCH_SEMANTIC_BLOCKING_THRESHOLD (the SimHash band/row split is derived
# from it). The SimHash band-hashing itself runs on the native Rust sketch kernel
# (the source of truth) by default -- see core/sketch.py + _native_loader.

# A column must average at least this many characters to count as "text-heavy"
# (free-text / description-like, where embeddings beat lexical keys). Short
# identity fields (name / email / zip) never qualify.
_SEMANTIC_TEXT_MIN_AVG_LEN = 40.0

# Strategies that ARE already a semantic / near-dup candidate generator -- never
# override these (they were chosen deliberately upstream).
_ALREADY_SEMANTIC_STRATEGIES = frozenset({"simhash", "ann", "ann_pairs", "lsh"})


@dataclass
class SemanticBlockingDecision:
    """Why auto semantic blocking did (not) fire -- surfaced for telemetry."""

    enabled: bool
    column: str | None
    reason: str
    embeddings_available: bool


def _text_heavy_columns(profiles: list[ColumnProfile]) -> list[ColumnProfile]:
    """Free-text columns long enough that semantic blocking helps."""
    return [
        p
        for p in profiles
        if p.col_type in ("description", "string", "multi_name")
        and p.avg_len >= _SEMANTIC_TEXT_MIN_AVG_LEN
    ]


def _auto_semantic_blocking_enabled() -> bool:
    """Default ON (#1090): text-heavy data routes to SimHash-over-embeddings
    blocking unless explicitly disabled via ``GOLDENMATCH_AUTO_SEMANTIC_BLOCKING``
    in ``{0, false, no, off, disabled}``.

    Default-on is safe because the decision still no-ops whenever no embedder is
    reachable (``decide_semantic_blocking`` -> ``embeddings_unavailable``), so a
    user without the in-house model or a configured provider sees byte-identical
    auto-config output -- the flip only fires when an embedder genuinely exists
    AND the data is text-heavy, which is exactly when semantic blocking helps.
    """
    val = os.environ.get("GOLDENMATCH_AUTO_SEMANTIC_BLOCKING", "").strip().lower()
    return val not in ("0", "false", "no", "off", "disabled")


# Default SimHash recall threshold for auto-enabled semantic blocking (#1090).
# A LOWER threshold -> more bands -> more candidate pairs -> higher recall (more
# permissive co-blocking of records that mean the same thing); HIGHER -> fewer,
# tighter candidates. 0.6 is the recall-leaning default for free-text near-dup
# blocking. The (bands, rows) split is derived from it by SimHashKeyConfig +
# sketch.optimal_bands -- so this is the user-facing recall knob #1090 exposes,
# replacing the old hardcoded num_bands.
_SEMANTIC_BLOCKING_THRESHOLD = 0.6


def _semantic_blocking_threshold() -> float:
    """Recall threshold (in ``(0, 1)``) for auto semantic blocking.

    Env override ``GOLDENMATCH_SEMANTIC_BLOCKING_THRESHOLD``; an absent, malformed
    or out-of-range value falls back to :data:`_SEMANTIC_BLOCKING_THRESHOLD`.
    """
    raw = os.environ.get("GOLDENMATCH_SEMANTIC_BLOCKING_THRESHOLD")
    if raw is None:
        return _SEMANTIC_BLOCKING_THRESHOLD
    try:
        val = float(raw)
    except ValueError:
        return _SEMANTIC_BLOCKING_THRESHOLD
    return val if 0.0 < val < 1.0 else _SEMANTIC_BLOCKING_THRESHOLD


def decide_semantic_blocking(
    profiles: list[ColumnProfile],
    config: GoldenMatchConfig | None = None,
    *,
    enabled: bool | None = None,
) -> SemanticBlockingDecision:
    """Decide whether to auto-enable semantic (ANN / SimHash) blocking.

    Returns a decision carrying an honest ``reason``. ``enabled`` is the door
    gate (defaults to the ``GOLDENMATCH_AUTO_SEMANTIC_BLOCKING`` env flag); the
    embedder-availability check is the honest fallback -- text-heavy data with no
    reachable embedder yields ``enabled=False, reason="embeddings_unavailable"``
    rather than committing a semantic plan that can't run.
    """
    if enabled is None:
        enabled = _auto_semantic_blocking_enabled()
    available = _embedder_available(config)
    if not enabled:
        return SemanticBlockingDecision(False, None, "disabled", available)
    text_cols = _text_heavy_columns(profiles)
    if not text_cols:
        return SemanticBlockingDecision(False, None, "not_text_heavy", available)
    col = max(text_cols, key=lambda p: p.avg_len).name
    if not available:
        return SemanticBlockingDecision(False, col, "embeddings_unavailable", False)
    return SemanticBlockingDecision(True, col, "text_heavy_with_embeddings", True)


def apply_auto_semantic_blocking(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
    df: pl.DataFrame | None = None,
    config: GoldenMatchConfig | None = None,
    *,
    enabled: bool | None = None,
) -> BlockingConfig | None:
    """Route blocking to SimHash over embeddings when the data is text-heavy and
    an embedder is reachable; otherwise return ``blocking`` unchanged.

    Default OFF (``GOLDENMATCH_AUTO_SEMANTIC_BLOCKING``) -- a no-op then, so the
    auto-config output is byte-identical. Never overrides a scheme that is
    already a semantic / near-dup generator (simhash / ann / lsh).
    """
    decision = decide_semantic_blocking(profiles, config, enabled=enabled)
    if not decision.enabled or decision.column is None:
        return blocking
    if blocking is not None and blocking.strategy in _ALREADY_SEMANTIC_STRATEGIES:
        return blocking
    from goldenmatch.config.schemas import SimHashKeyConfig

    return BlockingConfig(
        strategy="simhash",
        simhash=SimHashKeyConfig(
            column=decision.column,
            num_planes=256,
            threshold=_semantic_blocking_threshold(),
            seed=0,
        ),
    )


# ── #876: scale-invariant blocking helpers ────────────────────────────────────
# These are module-level (not closures) so they're importable by tests and by
# any caller that wants to reason about the budget without instantiating a full
# build_blocking run.

def _blocking_pairs_per_row_budget() -> int:
    """K: max candidate pairs per row a blocking option may project at full N.

    Constant (does NOT scale with N) — keeps the total pair count linear.  The
    scale-invariance knob; kept separate from ``max_safe_block`` (per-block OOM)
    so #715's scorer-matrix protection is untouched.

    Default: 50 (projected avg block ≤ ~101 rows).
    Override: ``GOLDENMATCH_BLOCKING_PAIRS_PER_ROW`` env var.
    """
    import os
    try:
        return max(1, int(os.environ.get("GOLDENMATCH_BLOCKING_PAIRS_PER_ROW", "50")))
    except ValueError:
        return 50


def _project_pairs_per_row(proj_block: int) -> int:
    """Pairs/row a (near-)uniform key contributes at full N.

    A block of B rows makes C(B, 2) pairs spread over B rows → (B-1)/2 per row.
    Uses the projected MAX block size (conservative on skew — over-counts rather
    than under-counts, so the budget gate stays safe).
    """
    return max(0, (int(proj_block) - 1) // 2)


# #876: a blocking key's block size at full N depends on whether its cardinality
# is BOUNDED (a closed domain → block grows ∝ N) or UNBOUNDED (an identifier-like
# column whose cardinality grows with N → block stays ~constant). A sample can't
# tell the two apart statistically (both look the same in a small complete
# sample, and a sparse sample of an unbounded key is all-singletons → Chao1
# undefined). So we use the column's SEMANTIC TYPE: a `zip` caps at ~100K
# (5-digit), a `year` at ~300, etc.; names / emails / strings are unbounded.
# This is the same shape of fix as the phonetic-anchor gate (#510): type-aware,
# not statistical. Types NOT listed are treated as unbounded (block constant) —
# the conservative-for-recall default; only KNOWN-bounded domains grow.
_BLOCKING_DOMAIN_CAP: dict[str, int] = {
    "zip": 100_000,   # 5-digit US zip
    "year": 300,      # ~1800-2100
    "month": 12,
    "boolean": 2,
}


def _typed_projected_block(
    col_types: dict[str, str], fields: list[str], sample_block: int, full_n: int,
    sample_n: int, sample_distinct: int,
) -> int:
    """Project a blocking key's full-N max block size.

    Three regimes, chosen so the projection is right for BOTH a genuinely
    high-cardinality key (block stays constant) AND a concentrated / null-heavy
    key (block grows ∝N) — the two failure modes a type-only rule can't tell apart:

    - **All-BOUNDED components** (every field a known-bounded domain — zip / year /
      month / boolean in ``_BLOCKING_DOMAIN_CAP``) → the domain saturates at the
      *product* of the per-column caps, so the block grows only as
      ``ceil(full_n / domain)``. The leniency that lets ``[zip, birth_year]``
      (joint domain ~30M) stay near-constant where a ∝N projector over-rejects it.
    - **Unbounded component, but the key is NEAR-UNIQUE on the sample** (joint
      distinct ratio ≥ the blocking-max-ratio, i.e. it spreads rows thin — a strong
      identifier like ``member_id``, or a unique-per-cluster name): cardinality
      grows ∝N, so the block stays at its (small) sample size. Constant.
    - **Unbounded component AND concentrated** (joint distinct ratio < the ratio —
      a 50-surname ``last_name``, a 50%-null ``zip5`` whose null bucket is a hot
      partition): the hot/null bucket's MAX block grows ∝N → use the **legacy ∝N
      projection** (``project_max_block_size``). This is the validated #715 OOM
      guard. (#876: a type-only rule returned the constant ``sample_block`` for ANY
      unbounded component, masking a real #715 regression — `[zip5, last_name]`
      emitted oversized; the cardinality split fixes it without re-breaking the
      #491 ``member_id``-wins-over-ANN case that ∝N-for-all over-rejected.)

    ``sample_block`` / ``sample_distinct`` are the max group size and the distinct
    group count on the (sub)sample of ``sample_n`` rows (one group_by yields both).
    """
    from goldenmatch.core.blocking_candidates import (
        _blocking_max_ratio,
        project_max_block_size,
    )
    domain = 1
    all_bounded = True
    for f in fields:
        cap = _BLOCKING_DOMAIN_CAP.get(col_types.get(f, ""))
        if cap is None:
            all_bounded = False
            break
        domain *= cap
    if all_bounded and domain > 0:
        return max(int(sample_block), -(-int(full_n) // domain))  # ceil(full_n/domain)
    # Unbounded component: near-unique ⇒ constant; concentrated ⇒ ∝N.
    joint_ratio = (sample_distinct / sample_n) if sample_n > 0 else 1.0
    if joint_ratio >= _blocking_max_ratio():
        return int(sample_block)  # spreads rows thin → block stays ~constant
    return project_max_block_size(int(sample_block), int(sample_n), int(full_n))


def _scale_safe_bounded_compound(
    candidates: list[ColumnProfile],
    is_scale_safe: Callable[[list[str]], bool],
) -> BlockingKeyConfig | None:
    """AND bounded-domain exact keys into the smallest scale-safe compound.

    #876: called when no SINGLE exact key is scale-safe. Considers only the
    BOUNDED candidates (those with a ``_BLOCKING_DOMAIN_CAP`` entry — zip / year /
    month / boolean); an unbounded key that's scale-safe alone was already kept as
    ``safe_exact`` (an unbounded key whose sample block already exceeds the OOM
    guard isn't bounded-compoundable here either — the name path handles it).
    Adds bounded keys most-selective-first
    (largest domain cap leads, so the compound is dominated by the discriminating
    key) until the running compound passes ``is_scale_safe``. Returns the compound
    ``BlockingKeyConfig`` or None if even the full bounded set doesn't bound the
    block (caller then falls through to the name path).

    Numeric-ish bounded keys: ``["lowercase", "strip"]`` is a safe transform
    chain (lowercase is a no-op on digit strings, normalizes a boolean's case).
    """
    bounded = [p for p in candidates
               if _BLOCKING_DOMAIN_CAP.get(p.col_type) is not None]
    if len(bounded) < 2:
        return None  # need >=2 bounded keys to form a joint-domain compound
    # Largest domain cap first → the most-selective bounded key leads.
    bounded.sort(key=lambda p: _BLOCKING_DOMAIN_CAP[p.col_type], reverse=True)
    fields: list[str] = []
    for p in bounded:
        fields.append(p.name)
        if len(fields) >= 2 and is_scale_safe(fields):
            return BlockingKeyConfig(fields=list(fields), transforms=["lowercase", "strip"])
    return None


def _is_text_corpus(profiles: list[ColumnProfile]) -> bool:
    """True when the data reads as a free-text corpus, not structured records.

    #1082 Phase A: a text corpus is a dataset whose identity signal lives in a
    long free-text (``description``) column with NO usable structured blocking
    key. A high-cardinality ``name``/``multi_name`` column (cardinality_ratio
    >= 0.1) is a usable structured blocking key, so its presence means this is
    structured data that happens to carry a free-text field — NOT a text
    corpus — and the normal exact/name blocking paths handle it. A low-
    cardinality name (a label, a category) can't block usefully, so a
    description-bearing dataset still reads as a corpus.
    """
    has_description = any(p.col_type == "description" for p in profiles)
    if not has_description:
        return False
    has_blockable_name = any(
        p.col_type in ("name", "multi_name") and p.cardinality_ratio >= 0.1
        for p in profiles
    )
    return not has_blockable_name


def _embedder_available(config: GoldenMatchConfig | None = None) -> bool:
    """True when a semantic embedder is reachable for text-corpus blocking.

    Gates the semantic (SimHash/embedding) branch of ``_text_corpus_blocking``:
    True when the in-house embedding model is importable OR the caller configured
    an embedding provider. #1082 Phase A routes both the lexical and the
    embedder-available case to lexical LSH; Phase B fills in the semantic branch.
    """
    from goldenmatch.core.embedder import inhouse_embedding_available
    if inhouse_embedding_available():
        return True
    return bool(getattr(getattr(config, "embedding", None), "provider", None))


#: Average character length at or below which a free-text column is "short
#: free text" (a product title, a headline) rather than a document, and blocks
#: better with `token` than with `lsh`. See `_text_corpus_blocking` for the
#: measurement and for why this is a calibration rather than a derived value.
_SHORT_FREE_TEXT_MAX_LEN = 120.0


def _best_sketch_column(profiles: list[ColumnProfile]) -> str | None:
    """The text column a blocking sketch should be built on.

    Chosen by IDENTITY (cardinality), not by LENGTH. The previous rule was
    ``max(avg_len)`` -- the longest text column -- which is backwards: the
    longest free-text field is the most verbose and the least identifying,
    because it shares boilerplate with every other row.

    Measured on Abt-Buy, whose two text columns BOTH profile as
    ``col_type=description`` so neither is treated as a structured key:

        name         avg_len= 54.3  cardinality=0.994   <- identity lives here
        description  avg_len=178.5  cardinality=0.771   <- max(avg_len) picked this

    Ground-truth blocking recall through the shipped blocker, on the plan
    auto-config actually committed versus the alternatives:

        lsh on 'description' (committed)   0.0009
        lsh on 'name'                      0.1139
        token on 'name' (max_df 0.02)      0.9198

    Amazon-Google escaped this only because it carries a `manufacturer` column
    profiling as ``col_type=name`` with cardinality 0.162, which trips
    ``_is_text_corpus``'s blockable-name test and routes it away from this
    branch entirely. That is luck, not design.

    Length is kept as the TIE-BREAKER: with nothing to choose on identity, a
    longer column carries more shingles for a sketch to work with, which is the
    grain of truth the old rule was reaching for.
    """
    if not profiles:
        return None
    # NO upper ceiling on cardinality, deliberately. The first version of this
    # capped it at 0.99 on the reasoning that a per-row-unique column "blocks
    # nothing, every value is its own block" -- which is true of an EXACT key
    # and false of a sketch. LSH and token blocking group by shared
    # shingles/tokens, not by equality, so a near-unique column is the IDEAL
    # sketch target: Abt-Buy's `name` is cardinality 0.994 and reaches 0.9198
    # blocking recall under token blocking. The cap excluded exactly the column
    # this function exists to pick.
    return max(
        profiles, key=lambda p: (round(p.cardinality_ratio, 3), p.avg_len)
    ).name


def _require_sketch_column(profiles: list[ColumnProfile]) -> str:
    """`_best_sketch_column` for callers that already guarantee a candidate.

    The sketch-config builders all run behind a check that a text column exists
    (`_is_text_corpus`, or `sketchable_text_cols` returning non-empty), and the
    code they replaced -- `max(profiles, key=...)` -- raised on an empty list.
    Keeping that contract here means those call sites stay `str`-typed instead
    of threading an `Optional` they can never actually receive.
    """
    col = _best_sketch_column(profiles)
    if col is None:
        raise ValueError(
            "no candidate text column for a blocking sketch; the caller is "
            "expected to check for one before building a sketch config"
        )
    return col


def _auto_build_lsh_config(profiles: list[ColumnProfile]) -> BlockingConfig:
    """Build a MinHash/LSH blocking config over the longest description column.

    #1082 Phase A: word-shingle MinHash/LSH (k=2, 128 perms, Jaccard
    threshold 0.5) on the description column with the largest average length —
    the field most likely to carry the near-duplicate lexical signal.
    """
    from goldenmatch.config.schemas import LSHKeyConfig
    descs = [p for p in profiles if p.col_type == "description"]
    col = _require_sketch_column(descs)
    return BlockingConfig(strategy="lsh", lsh=LSHKeyConfig(
        column=col, mode="word", k=2, num_perms=128, threshold=0.5, seed=0))


def _text_corpus_blocking(
    profiles: list[ColumnProfile], df: pl.DataFrame | None = None,
    config: GoldenMatchConfig | None = None,
) -> BlockingConfig:
    """Pick the text-corpus blocking strategy.

    Semantic (SimHash over embeddings) when an embedder is reachable, else
    lexical (MinHash/LSH). SimHash buckets cosine-near embeddings, catching
    near-duplicates that share meaning but little surface text — the lexical
    fallback only catches shared shingles. See #1082.
    """
    descs = [p for p in profiles if p.col_type == "description"]
    col = _require_sketch_column(descs)

    # SHORT free text blocks with `token`, documents with `lsh`. `blocking.mdx`
    # already prescribed exactly this -- "use `lsh` for near-duplicate
    # documents, `token` for short free text where only a few tokens agree" --
    # and the text-corpus branch committed LSH for everything it reached.
    #
    # Ground-truth blocking recall on Abt-Buy through the shipped blocker:
    #
    #   lsh   on 'name'        (thr 0.5)      0.1139    5,082 comparisons
    #   lsh   on 'name'        (thr 0.2)      0.5123   64,466
    #   token on 'name'        (max_df 0.02)  0.9198   64,056
    #   lsh   on 'description' (thr 0.5)      0.0009   13,071
    #   token on 'description' (max_df 0.02)  0.3801  137,720
    #
    # At equal cost -- ~64k comparisons -- token nearly DOUBLES LSH's recall on
    # the same column. The bar is read off the CHOSEN column (see
    # `_best_sketch_column`), never off the longest one: reading it off the
    # longest would send this shape back to LSH through the back door.
    #
    # `_SHORT_FREE_TEXT_MAX_LEN` is calibrated on the datasets available here
    # and separates them with room to spare -- Abt-Buy `name` 54.3 and
    # Amazon-Google `title` 51.1 on one side, Abt-Buy `description` 178.5 and
    # Amazon-Google `description` 551.2 on the other. Stated as a limitation
    # rather than presented as derived: a genuine document corpus (FineWeb-style,
    # thousands of characters) is far above it, which is the case LSH was built
    # for and keeps.
    # `0 <` deliberately: `avg_len` defaults to 0 on a hand-built ColumnProfile,
    # which means UNKNOWN, not "short". Routing unknown-length text to `token`
    # would be a silent behaviour change for every caller that constructs
    # profiles directly rather than measuring them, so an unmeasured column
    # falls through to the prior lsh/simhash behaviour.
    _chosen = next((p for p in descs if p.name == col), None)
    if _chosen is not None and 0 < _chosen.avg_len <= _SHORT_FREE_TEXT_MAX_LEN:
        from goldenmatch.config.schemas import BlockingConfig, TokenBlockingConfig

        return BlockingConfig(
            strategy="token",
            token=TokenBlockingConfig(column=col, min_token_length=3,
                                      max_df_ratio=0.02),
        )

    if _embedder_available(config):
        from goldenmatch.config.schemas import BlockingConfig, SimHashKeyConfig

        return BlockingConfig(
            strategy="simhash",
            simhash=SimHashKeyConfig(column=col, num_planes=256, num_bands=32, seed=0),
        )
    return _auto_build_lsh_config(profiles)


# ─────────────────────────────────────────────────────────────────────────────


# Column-DATA types that are structured (not free text) and can't be sketched.
_NON_SKETCHABLE_COL_TYPES = frozenset({"numeric", "date", "year", "zip", "phone", "geo"})


def sketchable_text_cols(profiles: list[ColumnProfile]) -> list[ColumnProfile]:
    """Columns the throughput tier can sketch on.

    Prefers the semantic text types (description / string / name / multi_name).
    But heterogeneous corpus text (web crawl) is routinely MIS-classified by the
    semantic col_type heuristics -- a long FineWeb/C4 document that embeds street
    names / emails / unique ids lands as "address" / "email" / "identifier", not
    "description" (measured on real FineWeb: a ~3k-char doc column classified
    "address"). So when no semantic text column exists, fall back to any column
    holding substantial free text -- keyed on ``avg_len``, NOT the semantic label
    -- excluding only columns whose DATA is structured. A short identifier
    (``doc_id``) is filtered by the length floor; a long mis-labelled text column
    is not.

    Shared by ``_throughput_blocking`` and ``auto_configure_df``'s early
    validation so the two cannot diverge.
    """
    semantic = [
        p for p in profiles
        if p.col_type in ("description", "string", "name", "multi_name")
    ]
    if semantic:
        return semantic
    return [
        p for p in profiles
        if p.col_type not in _NON_SKETCHABLE_COL_TYPES and p.avg_len >= 50
    ]


def _throughput_blocking(
    profiles: list[ColumnProfile], config: GoldenMatchConfig | None = None
) -> BlockingConfig:
    """Force sketch-then-verify (LSH or SimHash) blocking on the best text column.

    Accepts any text column type (description, string, name, multi_name).
    Routes to ``_text_corpus_blocking`` when description columns are present;
    otherwise builds LSH/SimHash directly on the longest text column.

    Raises ``ThroughputNotApplicableError`` when no text column is found --
    the throughput tier cannot operate without a sketch target.
    """
    from goldenmatch.core.throughput_verify import ThroughputNotApplicableError
    text_cols = sketchable_text_cols(profiles)
    if not text_cols:
        raise ThroughputNotApplicableError(
            "throughput tier requires a text column to sketch on; none found "
            f"(columns: {[(p.name, p.col_type, round(p.avg_len)) for p in profiles]})"
        )
    if any(p.col_type == "description" for p in profiles):
        return _text_corpus_blocking(profiles, None, config)
    if _embedder_available(config):
        from goldenmatch.config.schemas import SimHashKeyConfig
        col = _require_sketch_column(text_cols)
        return BlockingConfig(
            strategy="simhash",
            simhash=SimHashKeyConfig(column=col, num_planes=256, num_bands=32, seed=0),
        )
    from goldenmatch.config.schemas import LSHKeyConfig
    col = _require_sketch_column(text_cols)
    return BlockingConfig(strategy="lsh", lsh=LSHKeyConfig(
        column=col, mode="word", k=2, num_perms=128, threshold=0.5, seed=0))


def _compute_max_safe_block(height: int, native_scoring: bool) -> int:
    """The largest per-block size auto-config will accept for a blocking key.

    Scale-proportional (``height // 40``) with a 1000 floor and a scorer-aware
    ceiling. Rationale:

    - **Why proportional, not a fixed cap.** The old ``max(1000, height // 200)``
      pinned at 1000 for every dataset <= 200K rows. On census-Zipfian surnames the
      top surname-soundex block grows ~``0.013 * height`` and crosses 1000 between
      ~70K-90K rows; there the strong-identity surname pass was rejected as
      "oversized", which silently promoted surname into Fellegi-Sunter SCORING
      (blocking fields are excluded from scoring) where its weight over-merged
      same-block distinct people -- person F1 collapsed 0.97 -> 0.34 at 100K.
      ``height // 40`` (= ``0.025 * height``) grows faster than the surname block,
      so the pass is kept (validated: F1 back to ~0.96 at 100K/200K, at no wall
      cost -- the over-merged baseline is actually slower).

    - **Why the 1000 floor.** Below 40K rows ``height // 40 < 1000``, so the cap
      stays exactly 1000 -- byte-identical to the old behavior on the small
      datasets the accuracy gates use (DQbench/Febrl are unaffected; a 5000-row
      block on a 5K dataset would be the whole dataset in one degenerate block).

    - **Why the ceiling is scorer-aware.** The cap's original purpose was bounding
      the NUMPY ensemble scorer's NxN block matrix (float32 10K-block ~= 400 MB;
      50K OOMs). The native FS / bucket scorer scores PER-PAIR with no matrix, so
      memory is O(N) not O(N^2) and that basis doesn't bind -- when native
      block-scoring is active the ceiling lifts to 50K (wall/pair budget, not
      memory, then governs via the #715 projected-pair guard). Pure-numpy keeps
      the conservative 10K matrix ceiling.
    """
    ceiling = 50_000 if native_scoring else 10_000
    return max(1000, min(ceiling, height // 40))


# Learned-blocking training sample. `learned_sample_size` used to be
# `min(total_rows // 4, 5000)` -- pinned at 5,000 rows from 50K rows upward, no
# matter how big the frame got.
#
# The learner trains on the TRUE PAIRS it finds inside that sample, and the
# number of pairs whose BOTH members land in a fixed-size sample decays as
# `s^2 / n`. Measured on the QIS realistic shape, ground-truth pairs contained
# in a 5,000-row sample:
#
#     n = 500K -> 86     n = 1M -> 47     n = 2M -> 25     n = 4M -> 14
#
# i.e. the training signal HALVES every time the data doubles. In CI at 5M the
# learner saw 13 true pairs (against 56 at 1M), learned rules too coarse to
# separate anything, and emitted 160 blocks of ~31K rows instead of 791 blocks
# of ~1.3K -- roughly 78 BILLION candidate pairs against 0.6B at 1M. Five times
# the data, ~124x the pair work; the 5M rung ran 93 minutes without finishing.
#
# Holding the signal constant means growing the sample as sqrt(n): pairs scale
# as `s^2/n`, so `s ∝ sqrt(n)` keeps `s^2/n` flat. Verified against the same
# measurement -- an 11,180-row sample at 4M recovers 61 pairs, against 14 at
# 5,000.
#
# ANCHORED so this is a pure extension of today's behaviour: below the anchor
# the floor wins, reproducing `min(total_rows // 4, 5000)` exactly, so every
# frame under 1M keeps its current sample byte-for-byte and only >1M changes.
# The ceiling bounds training cost (the sample run is superlinear in `s`); past
# roughly 100M rows the signal starts decaying again, which is a known limit of
# this shape of fix rather than something it silently papers over.
_LEARNED_SAMPLE_FLOOR = 5_000          # the historical fixed value
_LEARNED_SAMPLE_ANCHOR_ROWS = 1_000_000  # frame size at which the floor is exactly right
_LEARNED_SAMPLE_CEILING = 50_000       # bound on training cost


def _learned_sample_size(total_rows: int) -> int:
    """Training-sample size for learned blocking, grown as sqrt(rows).

    Keeps the number of true pairs the learner trains on roughly CONSTANT with
    scale instead of letting it decay as 1/n. Returns the pre-existing
    ``min(total_rows // 4, 5000)`` for any frame at or below the anchor.
    """
    scaled = _LEARNED_SAMPLE_FLOOR * math.sqrt(total_rows / _LEARNED_SAMPLE_ANCHOR_ROWS)
    target = min(max(_LEARNED_SAMPLE_FLOOR, int(scaled)), _LEARNED_SAMPLE_CEILING)
    # `total_rows // 4` is the original held-out-data guard: never train on more
    # than a quarter of the frame, so predicates generalize instead of memorizing.
    return min(total_rows // 4, target)


def _learned_block_cap(total_rows: int, current_cap: int, native_scoring: bool) -> int:
    """Runtime oversized-DROP cap for learned blocking.

    Learned blocking runs with ``skip_oversized=True``, and
    ``apply_learned_blocks`` DROPS any block bigger than ``max_block_size``
    outright (``continue``) -- the whole block, and every true pair in it.
    So that cap must never sit BELOW the budget auto-config used to *select* the
    blocking key (``_compute_max_safe_block``): otherwise the selector accepts a
    key on the promise "blocks up to max_safe_block are fine" and the runtime
    then silently throws those very blocks away. The loss is recall-only (a drop
    never invents a merge), which is why it hides -- precision stays 1.0.

    That gap is scale-dependent, so it reads as a scale-invariance regression:
    a key whose blocks sit under the cap at 500K crosses it at 1M and is
    discarded. #1784 widened the gap further (its native ceiling puts
    max_safe_block at 25K on a 1M frame, against the default 5000 cap), which
    collapsed 1M zero-config recall to 0.82 while <=500K stayed at 1.0 (#1837).

    Raise-only: never tighten below the configured cap, so datasets under ~200K
    rows (where ``height // 40`` < 5000) keep their existing cap byte-for-byte.
    """
    return max(current_cap, _compute_max_safe_block(total_rows, native_scoring))


def defer_free_text_blocking_to_analyzer(blocking: Any, df: Any) -> Any:
    """Hand FREE-TEXT blocking to the analyzer instead of committing a prefix key.

    `build_blocking` only ever emits static/compound keys, so on free text it
    commits a prefix -- and `block_analyzer.free_text_columns` says the problem
    plainly: "a prefix key on a product title is a near-useless block."

    Measured on Amazon-Google. Auto-config commits `multi_pass` on
    `['title','manufacturer']`, blocking recall **0.0408**. The token strategy
    the analyzer can propose reaches **0.953** (94,938 candidate pairs, df<=50),
    taking achievable F1 from 0.094 to 0.404. The scorer is not the limit: 95%
    of true pairs already score above the 95th percentile of negatives.

    **Token blocking was never broken -- it was unreachable.** Two locks:
    `auto_suggest` defaults to False so `_run_auto_suggest` returns immediately,
    and its token branch sits behind `if not config.blocking.keys`, which
    auto-config has already populated. `build_token_blocks` never logged because
    nothing ever set `strategy="token"` -- misread as a broken integration
    (`block_analyzer.py:140`). Invoked directly it works.

    This clears BOTH locks for the one case where the static key is known bad:
    `auto_suggest=True` with no keys, a state `BlockingConfig` already models
    ("auto_suggest discovers keys at runtime", schemas.py:1283).

    Deliberately narrow:

    * only when a key field is free text by the MEASURED token-count bar
      (`_FREE_TEXT_MIN_MEAN_TOKENS` = 5.0; names and addresses sit at 2-4,
      product titles at 8-30), so name/address blocking is untouched. ANY rather
      than EVERY field, because compounding a free-text prefix with a short one
      does not rescue it -- `['title','manufacturer']` measures 0.0408;
    * never when a non-default strategy was chosen (canopy, ann, lsh, learned
      were picked for reasons this does not model);
    * fail-open -- any error leaves the config untouched.

    DBLP-ACM is unaffected: its committed key is `__title_key__`, a derived
    single-token feature, so the bar is not met and its 0.9820-recall key stands.
    """
    if blocking is None or not getattr(blocking, "keys", None):
        return blocking
    if getattr(blocking, "auto_suggest", False):
        return blocking
    if getattr(blocking, "strategy", None) not in (None, "static", "multi_pass"):
        return blocking
    try:
        from goldenmatch.core.block_analyzer import free_text_columns

        fields = [f for k in blocking.keys for f in (k.fields or [])]
        if not fields:
            return blocking
        free = set(free_text_columns(df, fields))
        if not free:
            return blocking
        logger.info(
            "Blocking: key field(s) %s are free text; deferring to the block "
            "analyzer (auto_suggest) instead of committing a prefix key -- that "
            "measures 0.0408 blocking recall on Amazon-Google against 0.953 for "
            "the token strategy the analyzer can reach. See #2717.",
            sorted(free),
        )
        return blocking.model_copy(update={"keys": [], "auto_suggest": True})
    except Exception:  # noqa: BLE001 -- advisory; never break config generation
        logger.debug("free-text blocking deferral skipped", exc_info=True)
        return blocking


def build_blocking(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    llm_provider: str | None = None,
    *,
    n_rows_full: int | None = None,
) -> BlockingConfig:
    """Generate blocking config from column profiles.

    Args:
        profiles: per-column profiles from ``profile_columns``.
        df: the (typically sample) frame.
        llm_provider: optional LLM provider for blocking suggestions.
        n_rows_full: full-population row count for the input data.
            When set AND larger than ``df.height``, drives Chao1 sample-
            size correction in the blocking-candidate gate (#410). When
            None, defaults to ``df.height`` (backward-compat for direct
            callers like tests that pass a full-table df).
    """
    # Arrow-port: coerce to the Frame seam once (build_blocking never reassigns
    # df) so column/height/group_by ops run polars-free on a pa.Table. Idempotent
    # for the polars backend -> byte-identical.
    from goldenmatch.core.frame import to_frame as _tf

    _bf = _tf(df)
    _bf_height = _bf.height

    # Filter out high-null columns (>20% null) — they create oversized null blocks
    # that cause O(N^2) comparison explosions
    max_null_rate = 0.20

    def _null_rate(col_name: str) -> float:
        return _bf.column(col_name).null_count() / _bf_height if _bf_height > 0 else 0.0

    # Tier 2 (autoconfig-tier1-tier2): null-aware v0 blocking selection.
    # Columns with null_rate > NULL_RATE_CEILING are skipped as blocking keys:
    # records with null values in the blocking field can't appear in any block,
    # structurally capping recall regardless of how good the scoring is.
    # NOTE: max_null_rate serves as NULL_RATE_CEILING here (value = 0.20).
    NULL_RATE_CEILING = max_null_rate  # 0.20 — explicit alias for Tier 2 intent

    # #408: cardinality-based blocking-candidate gate. The original
    # `cardinality_ratio < 0.95` let near-unique columns (NPI, federal IDs,
    # any column with >86% unique per record) through, producing singleton
    # blocks. The new gate (default 0.5) filters those out; env-overridable
    # via GOLDENMATCH_BLOCKING_MAX_RATIO. NPI keeps being picked as a
    # matchkey upstream — only its blocking role is denied.
    from goldenmatch.core.blocking_candidates import (
        _blocking_max_ratio,
        scale_cardinality_ratio_to_full_population,
    )
    blocking_max_ratio = _blocking_max_ratio()
    # #410: project sample cardinality to full-population. Auto-config
    # profiles run on a small sample; at sample N=1000, real
    # mid-cardinality columns (zip) look near-unique. Chao1 correction
    # uses the sample's distinct count + sample_n vs full_n to project.
    effective_n_full = n_rows_full if n_rows_full is not None else _bf_height
    sample_n = max(_bf_height, 1)

    # #1082: ANN is no longer AUTO-selected for description columns. A free-
    # text corpus (description column, no blockable structured key) is routed
    # to MinHash/LSH near-duplicate blocking via ``_is_text_corpus`` /
    # ``_text_corpus_blocking`` below (after the exact-key path). Explicit
    # ``ann``/``ann_pairs`` configs still work, and blocker.py's ANN sub-block
    # fallback inside oversized blocks is unchanged. The old #491 auto-ANN
    # path (ANN_MIN_ROWS gate + embedding-column detection here) was removed.

    def _projected_ratio(p: ColumnProfile) -> float:
        """Sample-corrected cardinality_ratio for the #408/#410 blocking gate.

        #876: TYPE-AWARE projection (same fix as ``_typed_projected_block`` for
        block size). ``scale_cardinality_ratio_to_full_population`` is a Chao1
        unseen-species estimator — it assumes a CLOSED domain and projects the
        cardinality ratio DOWN as N grows (more rows fill in the fixed domain →
        proportionally fewer distinct). That's valid only for a BOUNDED-domain
        key (zip/year/month/boolean). For an UNBOUNDED key (email/name/
        identifier/string) the domain grows WITH N, so a near-unique key stays
        near-unique at scale; Chao1 wrongly drives its ratio toward 0 (email
        0.56 → ~0.00 at 200M), letting a near-surrogate slip past the
        ``blocking_max_ratio`` gate and get picked as the SOLE blocking key —
        which blocks into near-singletons and tanks recall (#876: QIS email
        blocking recall 0.39). Keep an unbounded key at its sample ratio so the
        gate sees it for what it is. BOUNDED keys still Chao1-project (zip's true
        ratio really does fall as the 100K domain saturates)."""
        if effective_n_full <= sample_n:
            return p.cardinality_ratio
        if _BLOCKING_DOMAIN_CAP.get(p.col_type) is None:
            return p.cardinality_ratio  # unbounded domain: ratio does not fall with N
        sample_distinct = max(int(p.cardinality_ratio * sample_n), 1)
        return scale_cardinality_ratio_to_full_population(
            sample_distinct=sample_distinct,
            sample_n_rows=sample_n,
            full_n_rows=effective_n_full,
        )

    exact_cols = [
        p for p in profiles
        if p.col_type in ("email", "phone", "zip", "identifier", "year", _EXACT_DERIVED_COL_TYPE)
        and _null_rate(p.name) <= NULL_RATE_CEILING  # Tier 2: skip high-null candidates
        and _projected_ratio(p) <= blocking_max_ratio
        and _check_source_overlap(df, p.name) > 0.0
    ]
    # Log columns rejected by the cardinality gate (#408 / #410).
    for p in profiles:
        if p.col_type not in ("email", "phone", "zip", "identifier", "year", _EXACT_DERIVED_COL_TYPE):
            continue
        projected = _projected_ratio(p)
        if projected > blocking_max_ratio and projected < 1.0:
            logger.info(
                "Blocking candidate rejected: %r (projected_cardinality=%.3f > %.2f, "
                "sample_n=%d, full_n=%d); would produce near-singleton blocks. "
                "Kept for matchkey consideration. See #408/#410.",
                p.name, projected, blocking_max_ratio, sample_n, effective_n_full,
            )
    # Log skipped columns (cross-source overlap).
    for p in profiles:
        if (p.col_type in ("email", "phone", "zip", "identifier", "year", _EXACT_DERIVED_COL_TYPE)
                and _null_rate(p.name) <= max_null_rate
                and _projected_ratio(p) <= blocking_max_ratio
                and _check_source_overlap(df, p.name) == 0.0):
            sources = _bf.column("__source__").unique().to_list() if "__source__" in _bf.columns else []
            logger.warning(
                "Blocking key '%s' has 0%% overlap between sources %s -- skipping",
                p.name, ", ".join(str(s) for s in sources),
            )
    name_cols = [
        p for p in profiles
        if p.col_type == "name"
        and _check_source_overlap(df, p.name) > 0.0
    ]
    text_cols = [p for p in profiles if p.col_type in ("description", "string", "address")]

    # Auto-config's "is this blocking key safe?" threshold. Scales with
    # total_rows so the autoconfig's tolerance for block size grows with
    # the dataset.
    #
    # History: was 1000 historically, a margin set against float64
    # ensemble scorers OOM-ing on big block matrices. PR #173's float32
    # scoring brings a 5K block matrix down to ~100 MB. The fixed 1000
    # caused issue #199 at 2M+: the (state, name) compound block at 2M is
    # ~1.9K rows, "unsafe" under the old threshold, so autoconfig fell
    # through to a single-column soundex fallback. Soundex collapses
    # different surnames into one code, producing ~50K-sized blocks at
    # full scale that the pipeline filtered at max_block_size=5000 —
    # leaving no candidate pairs and ~99% singleton output.
    #
    # Auto-config's "is this blocking key safe?" ceiling. See _compute_max_safe_block:
    # scale-proportional (height // 40) with a 1000 floor and a scorer-aware cap.
    # Fixes the >=~90K over-merge (the census-Zipfian surname-soundex block crosses
    # the old fixed 1000 between 70K-90K rows, so the strong-identity surname pass
    # was DROPPED as "oversized" -> surname silently entered FS scoring -> person F1
    # collapsed 0.97 -> 0.34 at 100K); keeps small data (< 40K) at exactly 1000 so
    # DQbench/Febrl are byte-unchanged.
    from goldenmatch.core._native_loader import native_enabled as _native_enabled

    max_safe_block = _compute_max_safe_block(
        int(_bf_height), _native_enabled("block_scoring")
    )

    # #715: gate every emitted blocking key/pass by its PROJECTED full-N max
    # block size. build_blocking runs on a sample (or, in the v0 path, the
    # full df) and the emitted single-column soundex(name) passes had no
    # block-size guard. On a sparse-zip5 healthcare shape, zip5 reclassifies
    # to `identifier` and drops out of the compound, leaving single-name
    # passes whose max block projects to ~50K rows at 1M -> ~39.6M candidate
    # pairs -> an 18-min run. #876 keeps the legacy ∝N projector for keys with an
    # unbounded component (hot-bucket-safe) and adds a type-aware domain-product
    # path for ALL-BOUNDED-domain compounds (see _typed_projected_block).

    _col_types = {p.name: p.col_type for p in profiles}

    def _sample_block_and_distinct(fields: list[str]) -> tuple[int, int]:
        """(max group size, distinct group count) on the sample — one group_by.

        The distinct count drives the #876 cardinality split in
        ``_typed_projected_block`` (near-unique key ⇒ constant block; concentrated
        ⇒ ∝N). On error, return ``(effective_n_full, 1)`` so the key is treated as
        maximally oversized AND concentrated (dropped) — fail-safe.
        """
        try:
            g = _bf.group_len(fields)
            mb = int(g.column("len").max() or 0)  # "len" is int64 at runtime
            return mb, int(g.height)
        except Exception:  # pragma: no cover -- defensive
            return effective_n_full, 1

    def _projected_block(fields: list[str]) -> int:
        sample_mb, sample_distinct = _sample_block_and_distinct(fields)
        # #876: ALL-BOUNDED-domain key (zip×year) grows only as full_n/domain (the
        # leniency that lets [zip, birth_year] pass); an unbounded component is
        # constant if near-unique on the sample, else legacy ∝N (hot-bucket-safe).
        return _typed_projected_block(
            _col_types, fields, sample_mb, effective_n_full, sample_n, sample_distinct)

    def _is_scale_safe(fields: list[str]) -> bool:
        # #876: a key is scale-safe iff its candidate-pair count stays LINEAR in
        # N. Regimes from the projection (see _typed_projected_block):
        #   - ALL-BOUNDED-domain key (typed block grows ∝ N/domain, e.g. zip×year):
        #     pairs grow ∝ N^2/domain -> must satisfy the constant pairs-per-row
        #     budget K (block <= ~2K+1). Rejects sole-zip at 100M; admits the
        #     [zip, birth_year] compound (domain ~30M -> tiny block).
        #   - NEAR-UNIQUE unbounded key (member_id): constant small block -> linear
        #     pairs -> safe at any sample block size (per-block OOM guard aside).
        #   - CONCENTRATED unbounded key (50-surname last_name, 50%-null zip5): the
        #     legacy ∝N projection applies and a hot/null bucket projects oversized
        #     -> rejected here (the #715 guard).
        sb, sd = _sample_block_and_distinct(fields)
        pb = _typed_projected_block(
            _col_types, fields, sb, effective_n_full, sample_n, sd)
        if pb > max_safe_block:
            return False  # per-block OOM guard (#715), also kills huge-N bounded keys
        if pb > sb:  # block grows with N (bounded-cardinality / concentrated key)
            return _project_pairs_per_row(pb) <= _blocking_pairs_per_row_budget()
        return True

    def _pass_is_bounded(key: BlockingKeyConfig) -> bool:
        return _is_scale_safe(key.fields)

    def _gate_passes(
        primary: BlockingKeyConfig,
        passes: list[BlockingKeyConfig],
    ) -> tuple[BlockingKeyConfig | None, list[BlockingKeyConfig]]:
        """Drop oversized passes; pick a bounded primary key (#715).

        Returns ``(primary_or_None, surviving_passes)``. The chosen primary is
        the original primary if it is bounded, else the first bounded pass.
        When NOTHING is bounded, the primary is ``None`` (caller emits an
        empty/degenerate config so the controller refuses rather than
        shipping a candidate-pair bomb).
        """
        bounded = [p for p in passes if _pass_is_bounded(p)]
        dropped = [p for p in passes if not _pass_is_bounded(p)]
        if dropped:
            logger.info(
                "Dropping %d oversized blocking pass(es) by projected full-N "
                "block size (> %d, full_n=%d): %s. See #715.",
                len(dropped), max_safe_block, effective_n_full,
                [(p.fields, p.transforms) for p in dropped],
            )
        if _pass_is_bounded(primary):
            return primary, bounded
        if bounded:
            logger.info(
                "Primary blocking key %s projects oversized (> %d); promoting "
                "first bounded pass %s to primary. See #715.",
                primary.fields, max_safe_block, bounded[0].fields,
            )
            return bounded[0], bounded
        logger.warning(
            "All name-fallback blocking keys/passes project oversized at "
            "full_n=%d (> %d max_safe_block); emitting empty (degenerate) "
            "blocking config so the controller refuses. See #715.",
            effective_n_full, max_safe_block,
        )
        return None, []

    # Best case: block on highest-cardinality exact column (with low null rate + safe block size)
    if exact_cols:
        # Pre-filter: only evaluate top 5 by cardinality to avoid expensive group_by on all columns
        exact_cols_sorted = sorted(exact_cols, key=lambda p: _bf.column(p.name).n_unique(), reverse=True)
        # #876: drop a unique-per-row SURROGATE (id) before picking the best
        # exact key.  A surrogate creates singleton blocks (block size 1 → 0
        # candidate pairs → finds nothing) and was the degenerate fallback. Gate
        # ONLY on cardinality_ratio >= 1.0 (mirrors the exact-matchkey surrogate
        # guard at ~line 813). Do NOT also drop on projected_block<=1: a unique
        # email in a tiny sample (card_ratio < 1.0) is a legitimate exact
        # blocking key (it repeats in real data / match mode) — dropping it
        # regressed test_blocks_on_exact_column.
        exact_cols_sorted = [
            p for p in exact_cols_sorted
            if not _is_perfect_surrogate(p)
        ]
        candidates = exact_cols_sorted[:5]
        # #876: keep only scale-safe exact keys (the linear/pairs gate applies to
        # GROWING bounded keys only — see _is_scale_safe). At 100M a `zip` key
        # (bounded ~100K domain) projects a block of ~1000 that GROWS with N, so
        # it's rejected; an unbounded `email`/`name` key with a constant block is
        # kept.
        safe_exact = [p for p in candidates if _is_scale_safe([p.name])]
        # Cost-aware (#2021): a small-fixed-domain exact key (year/date col_type) is
        # "scale-safe" only at moderate N -- its block grows ∝N (fixed tiny domain),
        # so as the SOLE primary it explodes candidate pairs at scale (birth_year: 65
        # distinct → ~7.7B pairs at 1M, the issue #2021 runner-preemption). Demote it
        # from the primary slot when a better-bounded fallback exists (a name key or a
        # bounded compound); it stays a recall PASS via _pick_date_blocking_col (#438).
        # If it is the ONLY option, keep it rather than refuse outright.
        #
        # DOMAIN-ROUTED: skip the demotion on BIBLIOGRAPHIC data, where a `year`
        # column is the PUBLICATION year -- a strong shared-identity blocking signal
        # (every true match is same-year: the DBLP-ACM shape), so year-blocking is
        # correct there and must be preserved. Only person/other data (birth_year =
        # coarse ∝N block) gets demoted. Detection is by column names via
        # `core.domain.detect_domain` (the person shape scores `person`, not biblio).
        if (_cost_aware_blocking_enabled() and safe_exact
                and not _is_bibliographic_dataset(profiles)):
            non_growing = [p for p in safe_exact if p.col_type not in ("year", "date")]
            if non_growing:
                safe_exact = non_growing
            elif name_cols or _scale_safe_bounded_compound(candidates, _is_scale_safe):
                safe_exact = []  # fall through to the bounded-compound / name branches
        if safe_exact:
            best = max(safe_exact, key=lambda p: _bf.column(p.name).n_unique())
            transforms = (
                ["lowercase", "strip"]
                if best.col_type in ("email", _EXACT_DERIVED_COL_TYPE)
                else ["strip"]
            )
            # #2633: `best` wins by raw cardinality alone, which on bibliographic
            # data (DBLP-ACM: title-derived key ~300+ distinct) always beats a
            # `year`/`date` exact candidate (~10-65 distinct) even though `year`
            # is free selectivity there -- every true match shares its
            # publication year, the same domain-routed trust the demotion-skip
            # above already relies on. AND-ing `year` onto `best` is a strict
            # refinement (a compound record must share BOTH values), so it can
            # only keep-or-shrink `best`'s blocks, never lose a match `best`
            # alone would have kept together -- on THIS domain. Scoped to
            # `_is_bibliographic_dataset` rather than "any 2 safe_exact
            # candidates" because that guarantee is what the domain routing
            # vouches for; it is not free in general (an arbitrary second exact
            # column could disagree on real duplicates on other data shapes).
            if _is_bibliographic_dataset(profiles):
                year_candidates = [
                    p for p in safe_exact
                    if p.name != best.name and p.col_type in ("year", "date")
                ]
                if year_candidates:
                    refine = min(
                        year_candidates,
                        key=lambda p: _sample_block_and_distinct([best.name, p.name])[0],
                    )
                    best_block, _ = _sample_block_and_distinct([best.name])
                    compound_block, _ = _sample_block_and_distinct([best.name, refine.name])
                    if compound_block < best_block:
                        return BlockingConfig(
                            keys=[BlockingKeyConfig(
                                fields=[best.name, refine.name], transforms=transforms,
                            )],
                        )
            return BlockingConfig(
                keys=[BlockingKeyConfig(fields=[best.name], transforms=transforms)],
            )
        # #876: no SINGLE exact key is scale-safe. Before dropping these exact
        # discriminators for the name path, try to AND the BOUNDED ones into a
        # compound whose JOINT domain bounds the block under the pairs budget.
        # Two bounded keys that each explode alone (zip ∝N/100K, year ∝N/300) have
        # a joint domain of ~100K·300 = 30M, so their compound's block stays
        # ~constant and the pair count linear. The compound preserves what a sole
        # `zip` gives but can't scale: it co-locates a cluster's variants (both
        # components stable within a cluster) AND separates near-duplicate clusters
        # that share names but differ on the bounded keys (the QIS twin shape;
        # blocking on names alone collides those twins → over-merge). This is the
        # bounded-compound of the #876 design, instantiated as a refinement of the
        # rejected bounded keys rather than a greedy highest-cardinality pick.
        bounded_compound = _scale_safe_bounded_compound(candidates, _is_scale_safe)
        if bounded_compound is not None:
            logger.info(
                "Scale-safe bounded compound blocking: %s (no single exact key "
                "is scale-safe at full_n=%d; ANDed bounded exact keys to bound "
                "the block). See #876.",
                bounded_compound.fields, effective_n_full,
            )
            return BlockingConfig(keys=[bounded_compound], strategy="static")
        # All exact columns create oversized blocks — fall through
        logger.warning(
            "Exact blocking columns all produce oversized blocks (>%d), "
            "falling through to name-based blocking",
            max_safe_block,
        )

    # #1082: text-corpus fallback. We only reach here when no bounded exact
    # blocking key was found (the exact path above returns whenever it had a
    # usable safe_exact key). When the data reads as a free-text corpus (a
    # description column and no blockable structured key), route to MinHash/LSH
    # near-duplicate blocking rather than the name/compound fallback below.
    #
    # NOTE: this REPLACES the prior #491 ANN auto-selection for description
    # columns. ANN is no longer auto-picked here -- a text corpus now gets
    # lexical LSH blocking; explicit ann/ann_pairs configs still work, and the
    # ANN sub-block fallback inside oversized blocks (blocker.py) is unchanged.
    if _is_text_corpus(profiles):
        text_blk = _text_corpus_blocking(profiles, df)
        logger.info(
            "Auto-selecting %s blocking (text corpus): no bounded exact "
            "blocking key and no blockable structured column; routing the "
            "description corpus to near-duplicate blocking. See #1082.",
            text_blk.strategy,
        )
        return text_blk

    # #1207: per-identifier blocking-union. We reach here only when no single
    # exact key passed the strict NULL_RATE_CEILING (0.20) gate — exactly the
    # null-sparse multi-source shape where the compound fallback would build a
    # single-strong-id compound ([last_name, npi]) that caps recall at the id's
    # population. Prefer a UNION of one pass per strong id + name+geo, whose
    # OR-coverage restores the population the single key drops. Emitted before
    # the compound fallback so it wins; falls through unchanged when it can't
    # reach the coverage target or <2 passes survive.
    def _id_pass_scale_safe_nonnull(field: str) -> bool:
        """#1207: scale-safety for a strong-id SINGLETON pass, measured on the
        NON-NULL subframe.

        The runtime static blocker drops null block keys, so a high-null id's
        null bucket (which dominates its raw max block) never actually forms.
        The default ``_pass_is_bounded`` gate measures block size via a
        null-INCLUSIVE ``group_by`` — that null bucket falsely rejects exactly
        the null-sparse strong ids this union exists to add. We still apply the
        SAME full-N projection + ``max_safe_block`` guard (just over the
        non-null rows), so a bounded id (zip) or a mistyped low-cardinality
        ``identifier`` with a genuinely large non-null block is still dropped.
        """
        # #1852: route through the ``_bf`` Frame seam so this runs polars-free on
        # a ``pa.Table`` (the default under GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE=1).
        # The old ``df.filter(pl.col(...))`` idioms AttributeError'd on arrow and
        # the bare ``except`` swallowed it into ``return False`` -- every strong-id
        # pass was then silently rejected, so the #1207 union collapsed to the
        # name-only fallback on the arrow lane (the "silent zero/degraded blocking"
        # of #1852). ``filter_valid_key`` drops exactly the null + sentinel keys
        # the runtime static blocker drops (what this non-null projection means),
        # so the measured block matches what actually forms.
        sub = _bf.filter_valid_key(field)
        if sub.height == 0:
            return False
        g = sub.group_len([field])
        sb = int(g.column("len").max() or 0)
        sd = int(g.height)
        # Full-N projection unchanged: sample_n / effective_n_full stay the FULL
        # row counts; only the measured block/distinct exclude the null bucket.
        pb = _typed_projected_block(_col_types, [field], sb, effective_n_full, sample_n, sd)
        if pb > max_safe_block:
            return False
        if pb > sb:  # block grows with N (bounded-domain / concentrated key)
            return _project_pairs_per_row(pb) <= _blocking_pairs_per_row_budget()
        return True

    if union_via_core_enabled():
        # #1317 increment 3: route the #1207 union DECISION through the shared
        # native ``autoconfig-core`` kernel (byte-identical to the pure-Python
        # path below; see ``blocking_union_core``). The host still MEASURES the
        # two row-level signals (OR-coverage + per-pass scale-safety); the core
        # ASSEMBLES the candidate passes and applies the coverage/survivor gates.
        _union_cols = [
            {
                "name": p.name,
                "col_type": p.col_type,
                "null_rate": _null_rate(p.name),
                "cardinality_ratio": float(p.cardinality_ratio or 0.0),
            }
            for p in profiles
        ]
        _cand = assemble_union(_union_cols)
        if _cand is not None:
            _cov = _union_coverage(df, [list(p["fields"]) for p in _cand])
            _survives = [
                _id_pass_scale_safe_nonnull(p["fields"][0])
                if p["is_strong_id"]
                else _pass_is_bounded(
                    BlockingKeyConfig(fields=list(p["fields"]), transforms=list(p["transforms"]))
                )
                for p in _cand
            ]
            _out = finalize_union(_cand, _cov, _survives, max_safe_block)
            if _out is not None:
                logger.info(
                    "Auto-selecting strong-identifier blocking UNION via the shared "
                    "native core (%d passes) on null-sparse data. See #1207/#1317.",
                    len(_out["passes"]),
                )
                return BlockingConfig(
                    keys=[
                        BlockingKeyConfig(
                            fields=list(k["fields"]), transforms=list(k["transforms"])
                        )
                        for k in _out["keys"]
                    ],
                    strategy="multi_pass",
                    passes=[
                        BlockingKeyConfig(
                            fields=list(p["fields"]), transforms=list(p["transforms"])
                        )
                        for p in _out["passes"]
                    ],
                    max_block_size=int(_out["max_block_size"]),
                    skip_oversized=True,
                )
        # The core declined the union -> fall through to the name fallback below.
    else:
        union_cfg = _build_strong_identifier_union(profiles, df, n_rows_full=n_rows_full)
        if union_cfg is not None:
            id_passes: list[BlockingKeyConfig] = []
            other_passes: list[BlockingKeyConfig] = []
            for p in union_cfg.passes or []:
                if len(p.fields) == 1 and _col_types.get(p.fields[0]) in _STRONG_EXACT_TYPES:
                    id_passes.append(p)
                else:
                    other_passes.append(p)
            # Strong-id singletons: gate on the NON-NULL projected block (the static
            # blocker drops null keys). Name/geo passes: standard #715/#876 gate.
            surviving_ids = [p for p in id_passes if _id_pass_scale_safe_nonnull(p.fields[0])]
            surviving_other = [p for p in other_passes if _pass_is_bounded(p)]
            survivors = surviving_ids + surviving_other
            if surviving_ids and len(survivors) >= 2:
                primary = survivors[0]
                logger.info(
                    "Auto-selecting strong-identifier blocking UNION (%d passes, "
                    "%d strong-id) on null-sparse data: no single exact key cleared "
                    "the 0.20 null ceiling; union OR-coverage restores the dropped "
                    "population. Strong-id passes gated on non-null block size "
                    "(null keys are dropped at runtime). See #1207.",
                    len(survivors), len(surviving_ids),
                )
                return BlockingConfig(
                    keys=[primary],
                    strategy="multi_pass",
                    passes=survivors,
                    max_block_size=max_safe_block,
                    skip_oversized=True,
                )

    # ── Check if name-based fallback would also be oversized ──
    # #715: the name path picks ONE primary name column (pattern_names[0], else
    # name_cols[0]) and blocks on it. If THAT primary is oversized, single-name
    # blocking is degraded (it gets demoted to soundex/secondary or dropped) --
    # so we should try a bounded compound first, even if some OTHER name column
    # happens to be bounded on its own. Gating on "is the name path's primary
    # oversized" (by the projected full-N size, consistent with _gate_passes)
    # rather than "is EVERY name col oversized" lets the sparse-zip shape reach
    # zip5+last_name instead of degrading to a bare last_name block.
    def _name_path_primary() -> str | None:
        if not name_cols:
            return None
        pattern_names = [p for p in name_cols if _classify_by_name(p.name) == "name"]
        return (pattern_names[0] if pattern_names else name_cols[0]).name

    _primary_name = _name_path_primary()
    _all_single_oversized = True
    if _primary_name is not None:
        try:
            if _projected_block([_primary_name]) <= max_safe_block:
                _all_single_oversized = False
        except Exception:
            pass

    if _all_single_oversized and (name_cols or text_cols):
        # All single columns produce oversized blocks — try compound blocking
        if llm_provider:
            llm_config = _llm_suggest_blocking_keys(profiles, df, llm_provider, max_safe_block)
            if llm_config is not None:
                logger.info("Using LLM-suggested compound blocking keys")
                return llm_config
            logger.info("LLM suggestions invalid or unavailable — trying greedy compound")

        compound_config = _build_compound_blocking(profiles, df, max_safe_block, max_null_rate)
        if compound_config is not None:
            # #715: project the compound's keys/passes to full N before emitting.
            # _build_compound_blocking selects on SAMPLE block size, but a
            # high-null component (e.g. a ~50%-null zip5) hides a large null
            # bucket that scales linearly: zip5+last_name is ~323/block on a
            # 30K sample but projects to ~10K at 1M. Gate it through the same
            # projected guard as the name path; if nothing survives, fall
            # through to single-column fallbacks rather than ship a bomb.
            c_primary = (compound_config.keys or [None])[0]
            # resolved_keys() dispatches on strategy (multi_pass -> passes-first,
            # else keys-first) instead of always preferring `passes` -- the F2/
            # 1c843c8a5 shape: a schema-valid config carrying BOTH fields, where
            # the wrong unconditional preference silently used the wrong set.
            c_passes = compound_config.resolved_keys()
            if c_primary is not None:
                gated_primary, gated_passes = _gate_passes(c_primary, c_passes)
                if gated_primary is not None:
                    return BlockingConfig(
                        keys=[gated_primary],
                        strategy="multi_pass",
                        passes=gated_passes,
                        max_block_size=max_safe_block,
                        skip_oversized=True,
                    )
            logger.info(
                "Compound blocking config projects oversized at full_n=%d -- "
                "falling through to single-column fallbacks. See #715.",
                effective_n_full,
            )
        else:
            logger.info("Compound blocking failed — falling through to single-column fallbacks")

    # Name columns: use multi-pass with soundex + substring
    # Prefer columns matched by name pattern (person names) over data-profiled names
    if name_cols:
        pattern_names = [p for p in name_cols if _classify_by_name(p.name) == "name"]
        best_name = (pattern_names[0] if pattern_names else name_cols[0]).name

        # Check for geo columns to compound with name — prevents cross-region
        # false positives (e.g., same hospital name in different states)
        geo_cols = [
            p for p in profiles
            if p.col_type == "geo"
            and _null_rate(p.name) <= max_null_rate
        ]
        best_geo = None
        if geo_cols:
            # Pick the geo column that reduces max block size the most
            geo_results: list[tuple[ColumnProfile, int]] = []
            for g in geo_cols:
                # #1852: ``_bf`` seam (was ``df.group_by(...)``, which
                # AttributeError'd on a ``pa.Table`` -> the ``except: continue``
                # silently skipped every geo column, so name+geo compounding never
                # formed on the arrow lane).
                max_block = _bf.group_len([g.name, best_name]).column("len").max()
                if max_block is not None:
                    geo_results.append((g, int(max_block)))
            if geo_results:
                geo_results.sort(key=lambda x: x[1])
                candidate, candidate_block = geo_results[0]
                if candidate_block <= max_safe_block:
                    best_geo = candidate.name
                    logger.info(
                        "Geo-compound blocking: [%s, %s] -> max_block=%d",
                        best_geo, best_name, candidate_block,
                    )

        # #435: when multiple name columns exist (e.g. Febrl3 has both
        # given_name + surname), add a soundex pass for the SECOND one.
        # Without this, ~24pp of recall is lost on synthetic-typo data
        # where given_name was corrupted but surname stayed intact.
        secondary_name = None
        if len(name_cols) >= 2:
            secondary_candidates = [
                p for p in name_cols if p.name != best_name
            ]
            if secondary_candidates:
                secondary_name = secondary_candidates[0].name

        # #438: when a date column exists with low null rate, add it as
        # an extra blocking pass. Catches pairs that share DOB but
        # disagree on geo+name (typo'd name vs intact DOB) -- this is
        # the dominant recall-loss case on Febrl3-shape synthetic data.
        # Recall: 0.7229 -> 0.95+ on Febrl3 in local measurement.
        date_block_col = _pick_date_blocking_col(profiles, _null_rate)

        # #438: extra surname-substring pass complements the soundex
        # pass on the secondary name. Soundex collapses typos but is
        # noisy on short names; substring catches first-N-letter
        # corruption patterns ("Smith" -> "Smiht").
        secondary_name_substring = (
            secondary_name if secondary_name else None
        )

        if best_geo:
            extra_passes = []
            if secondary_name:
                extra_passes.append(BlockingKeyConfig(
                    fields=[secondary_name], transforms=["lowercase", "soundex"],
                ))
            if secondary_name_substring:
                extra_passes.append(BlockingKeyConfig(
                    fields=[secondary_name_substring],
                    transforms=["lowercase", "substring:0:5"],
                ))
            if date_block_col:
                extra_passes.append(BlockingKeyConfig(
                    fields=[date_block_col],
                    transforms=["lowercase", "strip"],
                ))
            geo_primary = BlockingKeyConfig(
                fields=[best_geo, best_name], transforms=["lowercase", "strip"]
            )
            geo_passes = [
                BlockingKeyConfig(fields=[best_geo, best_name], transforms=["lowercase", "strip"]),
                BlockingKeyConfig(fields=[best_geo, best_name], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"]),
                *extra_passes,
            ]
            primary, gated_passes = _gate_passes(geo_primary, geo_passes)
            if primary is None:
                return _degenerate_blocking_config(max_safe_block)
            return BlockingConfig(
                keys=[primary],
                strategy="multi_pass",
                passes=gated_passes,
                max_block_size=max_safe_block,
                skip_oversized=True,
            )

        extra_passes = []
        if secondary_name:
            extra_passes.append(BlockingKeyConfig(
                fields=[secondary_name], transforms=["lowercase", "soundex"],
            ))
        if secondary_name_substring:
            extra_passes.append(BlockingKeyConfig(
                fields=[secondary_name_substring],
                transforms=["lowercase", "substring:0:5"],
            ))
        if date_block_col:
            extra_passes.append(BlockingKeyConfig(
                fields=[date_block_col],
                transforms=["lowercase", "strip"],
            ))
        name_primary = BlockingKeyConfig(
            fields=[best_name], transforms=["lowercase", "soundex"]
        )
        name_passes = [
            BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "token_sort", "substring:0:8"]),
            *extra_passes,
        ]
        primary, gated_passes = _gate_passes(name_primary, name_passes)
        if primary is None:
            return _degenerate_blocking_config(max_safe_block)
        return BlockingConfig(
            keys=[primary],
            strategy="multi_pass",
            passes=gated_passes,
            max_block_size=max_safe_block,
            skip_oversized=True,
        )

    # #410: composite-blocking fallback. After exact_cols / name_cols
    # produce no winner, try a 2-column composite from mid-cardinality
    # blocking candidates (zip + last_name on healthcare data). This
    # beats the text_cols canopy / first_string fallback for any
    # structured dataset.
    import dataclasses as _dc

    from goldenmatch.core.blocking_candidates import (
        classify_column_role,
        find_composite_blocking_keys,
    )
    column_roles = []
    for p in profiles:
        if _check_source_overlap(df, p.name) <= 0.0:
            continue
        qcp = _make_quality_column_profile(p, effective_n_full)
        role = classify_column_role(
            qcp,
            sample_n_rows=sample_n,
            full_n_rows=effective_n_full,
        )
        column_roles.append(_dc.replace(role, name=p.name))

    composite = find_composite_blocking_keys(df, column_roles)
    if composite is not None:
        try:
            from goldenmatch.core.frame import to_frame

            joint_card = int(to_frame(df).joint_n_unique(list(composite)))
        except Exception:  # pragma: no cover -- defensive
            joint_card = 1
        avg_block = max(effective_n_full // max(joint_card, 1), 1)
        logger.info(
            "Composite blocking: %s -> ~%d rows/block (joint cardinality %d). See #410.",
            composite, avg_block, joint_card,
        )
        return BlockingConfig(
            keys=[BlockingKeyConfig(
                fields=list(composite), transforms=["lowercase"],
            )],
            max_block_size=max_safe_block,
            skip_oversized=True,
        )

    # Last resort: canopy on best text column
    if text_cols:
        from goldenmatch.config.schemas import CanopyConfig
        best_text = text_cols[0].name
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best_text], transforms=["lowercase", "substring:0:5"])],
            strategy="canopy",
            canopy=CanopyConfig(fields=[best_text], loose_threshold=0.3, tight_threshold=0.7),
            skip_oversized=True,
        )

    # Absolute fallback
    first_string = next(
        (p for p in profiles if p.col_type != "numeric"),
        profiles[0] if profiles else None,
    )
    if first_string:
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[first_string.name], transforms=["lowercase", "substring:0:5"])],
            skip_oversized=True,
        )

    return BlockingConfig(keys=[BlockingKeyConfig(fields=[profiles[0].name])], skip_oversized=True)


def _maybe_promote_blocking_to_adaptive(
    blocking: BlockingConfig | None,
    n_rows: int,
    *,
    threshold: int = 1_000_000,
) -> BlockingConfig | None:
    """Promote ``strategy="static"`` to ``"adaptive"`` for large datasets.

    Adaptive blocking recursively sub-partitions oversized blocks via
    ``core/blocker.py::_sub_block`` and falls back to
    ``_auto_split_block`` when ``sub_block_keys`` aren't configured
    (zero-config path). Without this promotion, ``build_blocking``
    always emits ``strategy="static"`` (or ``"multi_pass"`` / ``"canopy"``
    for specific shape paths). At full scale, oversized buckets — e.g.
    the Smith surname block at 5M+ scaling to ~57K rows — are scored
    as one giant block instead of being recursively sub-partitioned.

    Triggers when:

    - ``n_rows >= threshold`` (default 1M).
    - Current strategy is ``"static"`` only. ``multi_pass``, ``canopy``,
      ``ann``, ``learned``, ``sorted_neighborhood`` each have their own
      escape hatches; don't second-guess them.

    This is the eager half of the adaptive-blocking work. The
    controller-rule counterpart (``rule_blocking_adaptive_on_p99_outlier``)
    fires reactively when the measured block-size distribution
    shows a heavy P99 tail — useful when the eager threshold isn't
    reached but the data is pathologically skewed at smaller N.
    """
    if blocking is None or n_rows < threshold:
        return blocking
    if blocking.strategy != "static":
        return blocking
    return blocking.model_copy(update={"strategy": "adaptive"})


def _maybe_prune_blocking_passes(
    blocking: BlockingConfig | None,
    df: Any,  # pl.DataFrame | pl.LazyFrame | pa.Table (the arrow lane)
) -> BlockingConfig | None:
    """Opt-in weak-positive-aware pruning of multi-pass blocking passes.

    Default OFF. Enable via ``GOLDENMATCH_BLOCKING_PRUNE_PASSES=1``. The floor
    is ``GOLDENMATCH_BLOCKING_PASS_MIN_WEAKPOS`` (default 1 = drop only
    fully-redundant / all-noise passes, recall-safe). Higher floors trade a
    little recall for fewer candidates while protecting high-precision passes
    (the selector ranks by likely-match yield, NOT raw new-pair count, so a
    sparse-but-precise pass like exact date-of-birth survives).

    Runs on the auto-config sample (``df``); the surviving passes are applied to
    the full dataset. No-op unless the strategy is ``multi_pass`` with >1 pass.
    """
    if blocking is None or blocking.strategy != "multi_pass":
        return blocking
    if os.environ.get("GOLDENMATCH_BLOCKING_PRUNE_PASSES", "").strip().lower() not in (
        "1", "true", "yes", "on", "enabled",
    ):
        return blocking
    passes = blocking.passes or []
    if len(passes) <= 1:
        return blocking
    try:
        floor = int(os.environ.get("GOLDENMATCH_BLOCKING_PASS_MIN_WEAKPOS", "1"))
    except ValueError:
        floor = 1
    try:
        from goldenmatch.core.blocking_pass_selection import select_passes
        from goldenmatch.core.frame import is_polars_lazyframe

        # select_passes is polars-native (with_row_index / group_by), but the FS
        # routed / arrow lane passes `df` as a pyarrow Table (or a LazyFrame).
        # Coerce first so pruning actually RUNS -- without this the arrow lane
        # threw AttributeError, was swallowed below into "keep all passes", and
        # this whole opt-in was a silent no-op for every arrow-lane FS caller.
        prune_df = df
        if is_polars_lazyframe(prune_df):
            prune_df = prune_df.collect()
        elif not isinstance(prune_df, pl.DataFrame):
            # pl.from_arrow on a Table always yields a DataFrame (Series only for
            # an Array/ChunkedArray input, which this never is).
            prune_df = cast("pl.DataFrame", pl.from_arrow(prune_df))

        result = select_passes(prune_df, list(passes), min_marginal_weak_positive=floor)
    except Exception:
        logger.warning("blocking pass selection failed; keeping all passes", exc_info=True)
        return blocking
    if not result.dropped or not result.kept:
        return blocking
    logger.info(
        "blocking pass selection: kept %d/%d passes (dropped %s)",
        len(result.kept), len(passes), [list(p.fields) for p in result.dropped],
    )
    return blocking.model_copy(update={"passes": result.kept})


# ── Model selection ────────────────────────────────────────────────────────

def select_model(row_count: int, has_embedding_columns: bool, threshold: int = 50000) -> str | None:
    """Select embedding model. Returns None if no embedding columns needed."""
    if not has_embedding_columns:
        return None
    if row_count < threshold:
        return "gte-base-en-v1.5"
    return "all-MiniLM-L6-v2"


# ── Main entry point ──────────────────────────────────────────────────────

# ContextVar populated by auto_configure_df after each successful controller
# run. Holds (ComplexityProfile, RunHistory) so PostflightReport can pick up
# the profile + history without threading them through every call site.
_LAST_CONTROLLER_RUN: ContextVar = ContextVar("_LAST_CONTROLLER_RUN", default=None)

# ContextVar populated by auto_configure_df with the GoldenCheck exclusion
# list (#404). PostflightReport reads this to render the "Auto-config
# exclusions" section so users see the audit trail without grep-ing logs.
_LAST_AUTOCONFIG_EXCLUSIONS: ContextVar = ContextVar(
    "_LAST_AUTOCONFIG_EXCLUSIONS", default=None,
)

# ContextVar set by dedupe_df / match_df with the kwarg-supplied
# exclude_columns list (#unified-exclusions). Layered ADDITIVELY with
# config.exclude_columns and detector-derived exclusions inside
# auto_configure_df. Cleared after each call via context-bound semantics
# (callers do `.set(...)` then rely on the ContextVar value living only
# for the duration of their own frame).
_RUNTIME_EXCLUDE_COLUMNS: ContextVar = ContextVar(
    "_RUNTIME_EXCLUDE_COLUMNS", default=None,
)

# #858: set by the match pipeline (`_run_match_pipeline`) and `match_df` around
# their `auto_configure_df` call so the multi-source guard is suppressed in
# match mode (cross-source linking is the goal there). Dedupe paths never set it.
_AUTOCONFIG_MATCH_MODE: ContextVar = ContextVar(
    "_AUTOCONFIG_MATCH_MODE", default=False,
)


@contextlib.contextmanager
def _match_mode_autoconfig():  # pyright: ignore[reportUnusedFunction]  # used cross-module (pipeline / _api) via local imports
    """Suppress the #858 multi-source guard for the duration (match mode)."""
    token = _AUTOCONFIG_MATCH_MODE.set(True)
    try:
        yield
    finally:
        _AUTOCONFIG_MATCH_MODE.reset(token)


# Set by execution paths that CANNOT run the Fellegi-Sunter routed config --
# the streaming / incremental orchestrator (`goldenmatch sync` / `db.watch`) scores
# per-block and cannot train a global EM model, so a routed probabilistic config
# raises NotImplementedError in `_score_block_streaming`. These paths wrap their
# `auto_configure(_df)` call in `deterministic_routing()` so zero-config streaming
# stays on the deterministic weighted path. `None` = honor the env default.
_ROUTE_PROBABILISTIC_OVERRIDE: ContextVar = ContextVar(
    "_ROUTE_PROBABILISTIC_OVERRIDE", default=None,
)


@contextlib.contextmanager
def deterministic_routing():
    """Force auto-config OFF the probabilistic (Fellegi-Sunter) route for the
    duration -- for execution paths (streaming / incremental) that can't train
    per-partition EM. Zero-config on these paths stays on the weighted path."""
    token = _ROUTE_PROBABILISTIC_OVERRIDE.set(False)
    try:
        yield
    finally:
        _ROUTE_PROBABILISTIC_OVERRIDE.reset(token)


def _multisource_autoconfig_enabled() -> bool:
    """#858 multi-source zero-config guard. Default ON; disable with
    ``GOLDENMATCH_MULTISOURCE_AUTOCONFIG=0`` (or false/disabled/off/no)."""
    return os.environ.get(
        "GOLDENMATCH_MULTISOURCE_AUTOCONFIG", "1"
    ).strip().lower() not in {"0", "false", "disabled", "off", "no"}


def _env_force_exclude() -> list[str]:
    raw = os.environ.get("GOLDENMATCH_AUTOCONFIG_FORCE_EXCLUDE", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def _env_force_include() -> list[str]:
    raw = os.environ.get("GOLDENMATCH_AUTOCONFIG_FORCE_INCLUDE", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def _resolve_effective_exclusion_overrides(
    config: Any | None = None,
) -> tuple[list[str], list[str]]:
    """Combine every exclusion source into (force_exclude, force_include).

    Sources, layered additively for force_exclude:
      1. ``config.exclude_columns`` -- new top-level field (this PR).
      2. ``_RUNTIME_EXCLUDE_COLUMNS`` ContextVar -- kwarg from
         ``dedupe_df`` / ``match_df``.
      3. ``config.quality.autoconfig_force_exclude`` -- #404 sub-field.
      4. ``GOLDENMATCH_AUTOCONFIG_FORCE_EXCLUDE`` env var -- #404 V1 path.

    Sources for force_include (RESCUE -- beats every opt-out path):
      a. ``config.quality.autoconfig_force_include`` -- #404 sub-field.
      b. ``GOLDENMATCH_AUTOCONFIG_FORCE_INCLUDE`` env var -- #404 V1 path.

    Order in the final set is irrelevant (it's a union); ordering here
    just makes the log line predictable.
    """
    fe: list[str] = []
    fi: list[str] = []
    if config is not None:
        # New top-level field.
        cfg_excl = getattr(config, "exclude_columns", None) or []
        fe.extend(cfg_excl)
        # Legacy #404 sub-field on QualityConfig (still supported).
        qc = getattr(config, "quality", None)
        if qc is not None:
            fe.extend(getattr(qc, "autoconfig_force_exclude", None) or [])
            fi.extend(getattr(qc, "autoconfig_force_include", None) or [])
    runtime_excl = _RUNTIME_EXCLUDE_COLUMNS.get()
    if runtime_excl:
        fe.extend(runtime_excl)
    fe.extend(_env_force_exclude())
    fi.extend(_env_force_include())
    # De-dupe while preserving first-occurrence order for log readability.
    fe = list(dict.fromkeys(fe))
    fi = list(dict.fromkeys(fi))
    return fe, fi



# Tier 4: cross-run memory.  Set GOLDENMATCH_AUTOCONFIG_MEMORY=0 (or "false"
# or "disabled") to opt out (useful in CI or when disk I/O should be minimal).
_AUTOCONFIG_MEMORY_DISABLED: bool = (
    os.environ.get("GOLDENMATCH_AUTOCONFIG_MEMORY", "1").lower()
    in ("0", "false", "disabled")
)

# Tier 3: LLM fallback policy.  Set GOLDENMATCH_AUTOCONFIG_LLM=1 (or "true"
# or "enabled") to enable LLM-assisted config proposals when the heuristic
# rule table is exhausted but the profile is still RED/YELLOW.  Requires
# OPENAI_API_KEY.  Default OFF to avoid unexpected API spend.
_AUTOCONFIG_LLM_ENABLED: bool = (
    os.environ.get("GOLDENMATCH_AUTOCONFIG_LLM", "0").lower()
    in ("1", "true", "enabled")
)

# Module-level default memory singleton (uses ~/.goldenmatch/autoconfig_memory.db).
# Lazily initialised on first use to avoid creating the directory at import time.
_DEFAULT_MEMORY: AutoConfigMemory | None = None


def _autoconfig_memory_disabled() -> bool:
    """Whether cross-run memory is off, read at CALL time.

    ``_AUTOCONFIG_MEMORY_DISABLED`` is evaluated once at import, so
    ``GOLDENMATCH_AUTOCONFIG_MEMORY=0`` only ever worked when it was already in
    the environment before ``goldenmatch.core.autoconfig`` was imported. Setting
    it from Python afterwards -- which is what the docstring's "useful in CI"
    advice reads like, and what six test modules do via raw ``os.environ`` --
    silently did nothing, leaving the shared ``~/.goldenmatch/autoconfig_memory.db``
    live:

        after import,        disabled = False
        after setting env=0, disabled = False

    Checking the constant FIRST keeps every existing caller intact, including the
    tests that ``monkeypatch.setattr`` it to True; the env read only ever adds a
    reason to disable, so this can turn memory off but never on.
    """
    if _AUTOCONFIG_MEMORY_DISABLED:
        return True
    return os.environ.get("GOLDENMATCH_AUTOCONFIG_MEMORY", "1").strip().lower() in (
        "0", "false", "disabled",
    )


def _get_default_memory() -> AutoConfigMemory | None:
    """Return the default AutoConfigMemory, or None when disabled via env var."""
    global _DEFAULT_MEMORY
    if _autoconfig_memory_disabled():
        return None
    if _DEFAULT_MEMORY is None:
        try:
            from goldenmatch.core.autoconfig_memory import AutoConfigMemory
            _DEFAULT_MEMORY = AutoConfigMemory()
        except Exception as exc:
            logger.warning(
                "auto-config: could not initialise default memory store: %s", exc
            )
            return None
    return _DEFAULT_MEMORY


def _field_group_detection_enabled(config: Any) -> bool:
    """Return True when field-group detection is switched on (default OFF).

    Two opt-in paths:
    - ``golden_rules.field_group_detection = True`` on the incoming config
    - env var ``GOLDENMATCH_FIELD_GROUP_SURVIVORSHIP`` set to 1/true/yes/on
    """
    gr = getattr(config, "golden_rules", None)
    if gr is not None and getattr(gr, "field_group_detection", False):
        return True
    return os.environ.get("GOLDENMATCH_FIELD_GROUP_SURVIVORSHIP", "").lower() in (
        "1", "true", "yes", "on"
    )


def _maybe_active_domain_pack(config: Any):
    """Return a goldencheck-types DomainPack if one is clearly in play, else None.

    v1: best-effort -- return None (detection falls back to the heuristic).
    Kept minimal on purpose; infermap-fed detection is opportunistic.
    """
    return None


def _maybe_detect_field_groups(df: Any, config: Any) -> None:
    """Gated, fail-open hook: when enabled, detect field groups and write them
    onto ``config.golden_rules.field_groups``.

    Default OFF; explicit groups always kept. infermap is optional (its import
    lives inside ``build_field_groups``). Any exception leaves the config
    untouched so auto-config is never broken by optional detection.
    """
    enabled = _field_group_detection_enabled(config)
    explicit = []
    gr = getattr(config, "golden_rules", None)
    if gr is not None:
        explicit = list(getattr(gr, "field_groups", []) or [])
    if not enabled and not explicit:
        # Nothing to do; leave config byte-identical.
        return
    # Field-group detection is column-name-based and needs a Polars frame.
    # On the distributed path `df` is a Ray Dataset; skip silently (collecting
    # to detect would be wrong here) rather than letting build_field_groups
    # fail-open with a misleading warning.
    try:
        from goldenmatch.distributed import is_ray_dataset
        if is_ray_dataset(df):
            return
    except Exception:
        pass  # distributed utils not importable -> proceed; outer try/except still guards
    try:
        from goldenmatch.core.survivorship.groups import build_field_groups
        pack = _maybe_active_domain_pack(config)
        detected = build_field_groups(df, pack=pack, explicit=explicit, enabled=enabled)
    except Exception as exc:
        # Fail-open: never break auto-config over optional detection.
        logger.debug("field-group detection hook skipped: %s", exc)
        return
    if not detected:
        return
    # Ensure golden_rules exists so we can attach field_groups.
    if gr is None:
        from goldenmatch.config.schemas import GoldenRulesConfig as _GRC
        gr = _GRC(default_strategy="most_complete")
        config.golden_rules = gr
    gr.field_groups = detected


def auto_configure_df(
    df: Any,  # pl.DataFrame | pl.LazyFrame | pa.Table | Frame | ray.data.Dataset
    llm_provider: str | None = None,
    domain_config: Any = None,
    llm_auto: bool = False,
    strict: bool = False,
    allow_remote_assets: bool = False,
    *,
    reference: Any = None,  # pl.DataFrame | pl.LazyFrame | pa.Table | Frame | None
    _skip_finalize: bool = False,
    confidence_required: bool = True,
    allow_red_config: bool = False,
    planning_effort: str | None = None,
    n_rows_full: int | None = None,
    throughput: Any | None = None,
    fused_match_allowed: bool = False,
) -> GoldenMatchConfig:
    """Public auto-configuration entry point (controller-backed).

    Runs an iterative refit loop:
      1. Compute v0 config via the legacy heuristic (or retrieve from memory)
      2. Run blocking → score → cluster on a stratified sample under profile_capture
      3. Read the ComplexityProfile, ask the policy for a refit
      4. Loop until green/converged/budget
      5. Run the full pipeline once and return the committed config

    The committed config is what's returned. Profile + history are stashed
    on the ``_LAST_CONTROLLER_RUN`` ContextVar so the calling pipeline can
    surface them via PostflightReport.

    Cross-run memory (Tier 4): when a previous successful run used the same
    data shape, the cached config is returned as the starting point, bypassing
    the legacy heuristic. Set ``GOLDENMATCH_AUTOCONFIG_MEMORY=0`` to disable.

    Args:
        df: target DataFrame (or LazyFrame, which will be collected).
        reference: optional reference DataFrame for cross-source ``match_df``
            mode. When None, dedupe mode (single-source).
        llm_provider, domain_config, llm_auto, strict, allow_remote_assets:
            unchanged from the legacy signature; threaded into v0 only.

    Raises:
        TypeError: when ``df`` or ``reference`` is not a polars DataFrame/LazyFrame.
        ConfigValidationError: from the controller, when input is unworkable
            (empty, all-null, etc.).
    """
    # Boundary: accept pl.DataFrame | pl.LazyFrame | pa.Table | dict-of-arrays |
    # Frame | ray.data.Dataset. PR-3..6b landed the autoconfig-LAYER arrow seam
    # (arrow->polars dtype map, controller gates, sampling, blocking-size
    # measurement, v0 heuristic). The zero-config SCORING lane is NOT fully
    # arrow-ported yet, though: the eager indicators' empty/column guards
    # (indicators.py), the GoldenCheck exclusion detectors, and the legacy
    # `build_blocks(combined_lf)` scoring spine (pipeline.py:2284) still assume
    # polars -- a bare pa.Table there raises (e.g. `pa.Table.is_empty`) and the
    # controller degrades to a RED v0 fallback. So a non-polars input is coerced
    # to polars HERE (byte-identical to the same data passed as a pl.DataFrame --
    # no silent RED fallback / regression). Making this a true `.native` arrow
    # pass-through (Polars-free zero-config, tripwire green, [polars] stopgap
    # dropped) is the pipeline-spine follow-up; the seam routing above is its
    # foundation. (ray / lazy / polars inputs are handled by the block below.)
    from goldenmatch.core.frame import (
        is_polars_dataframe,
        is_polars_lazyframe,
        to_frame,
    )
    from goldenmatch.distributed._utils import is_ray_dataset as _is_ray_boundary

    # GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE default-ON (2026-07-14): a non-polars
    # input (pa.Table / Frame / dict-of-arrays) stays NATIVE through the
    # controller -- the zero-config config-GENERATION lane is arrow-native
    # (indicators guards + exclusion / discriminative-veto / source-partition
    # detectors seam-ported, sample scoring on the arrow-native bucket scorer;
    # parity-gated in test_autoconfig_arrow_native_parity.py). `=0` restores the
    # legacy arrow->polars boundary coercion.
    #
    # GUARDED FALLBACK: two sub-features are NOT arrow-native yet and still assume
    # a polars target -- cross-source match (`reference`, coerced to polars below
    # and whose match-scoring path is unproven on arrow) and the throughput tier
    # (its early text-column validation only fires for a polars df). For those we
    # coerce to polars (legacy behaviour) and WARN once, so a decline is visible
    # rather than a silent divergence.
    _arrow_native_ac = os.environ.get(
        "GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE", "1"
    ).lower() in ("1", "true", "yes")
    if _arrow_native_ac and throughput is not None:
        _warn_arrow_native_coerced_once("throughput tier")
        _arrow_native_ac = False
    if not (
        _is_ray_boundary(df)
        or is_polars_lazyframe(df)
        or is_polars_dataframe(df)
    ):
        _boundary_native = to_frame(df).native
        if is_polars_dataframe(_boundary_native) or _arrow_native_ac:
            df = _boundary_native
        else:
            import polars as _pl_boundary

            df = cast("pl.DataFrame", _pl_boundary.from_arrow(_boundary_native))

    # Throughput tier (#1083): early validation -- check text column exists BEFORE
    # the expensive controller run to give a clean ThroughputNotApplicableError.
    if throughput is not None:
        import polars as _pl_tp

        from goldenmatch.core.throughput_verify import ThroughputNotApplicableError
        if isinstance(df, _pl_tp.DataFrame):
            _early_profiles = profile_columns(df)
            # Same predicate as _throughput_blocking (shared helper) so the early
            # gate and the blocking builder cannot diverge — heterogeneous corpus
            # text mis-classified as "address"/"identifier" must pass here too.
            if not sketchable_text_cols(_early_profiles):
                raise ThroughputNotApplicableError(
                    "throughput tier requires a text column to sketch on; none found "
                    f"(columns: {[(p.name, p.col_type, round(p.avg_len)) for p in _early_profiles]})"
                )

    # Coerce + validate input types.
    # Phase 2: also accept ray.data.Dataset on the distributed path.
    from goldenmatch.distributed._utils import is_ray_dataset as _is_ray_dataset
    if _is_ray_dataset(df):
        # Dataset stays as-is; controller.run() handles it natively.
        _n_rows_for_budget: int = df.count()  # type: ignore[union-attr]
    elif is_polars_lazyframe(df):
        df = cast("pl.LazyFrame", df).collect()
        _n_rows_for_budget = df.height
    elif is_polars_dataframe(df):
        _n_rows_for_budget = df.height
    else:
        # Arrow-native boundary (pa.Table) -- the zero-config polars-free lane.
        # .height via the Frame seam; the controller consumes the arrow native.
        _n_rows_for_budget = to_frame(df).height
    if reference is not None:
        import pyarrow as _pa_ref

        if isinstance(reference, _pa_ref.Table):
            pass  # arrow reference flows natively (cross-source arrow lane)
        elif is_polars_lazyframe(reference):
            reference = cast("pl.LazyFrame", reference).collect()
        elif not is_polars_dataframe(reference):
            raise TypeError(
                f"reference requires pl.DataFrame, pl.LazyFrame, or pa.Table, "
                f"got {type(reference).__name__}"
            )

    # ── GoldenCheck auto-config exclusions (#404) ──
    # Run the column-exclusion detectors BEFORE the controller starts.
    # Drop excluded columns from df so v0, sample iterations, and the
    # committed config never reference them. The user's original df
    # downstream still has every column (golden records aren't affected);
    # we only filter the matchkey/blocking candidate pool.
    #
    # Skipped when df is a Ray Dataset (distributed path) -- those have
    # their own column-selection story; exclusions land at the per-
    # partition kernel layer separately.
    _ms_partition: str | None = None  # #858 source partition (None => feature off)
    if not _is_ray_dataset(df):
        # Arrow-port: the exclusion + source-partition detectors are now seam-
        # ported (detect_autoconfig_exclusions / _detect_source_partition /
        # _source_correlated_exclusions all route Column stats through the seam),
        # so this block runs identically for a pl.DataFrame AND a bare pa.Table
        # (the arrow-native path). A pa.Table zero-config input therefore gets
        # the SAME exclusions / exclude_columns / #858 guard as an equivalent
        # pl.DataFrame -- config-equivalence, not the silent skip Finding 1
        # flagged. `df` may be arrow (GOLDENMATCH_AUTOCONFIG_ARROW_NATIVE) or
        # polars (default coercion); `_excl_frame` gives backend-agnostic
        # columns/drop.
        from goldenmatch.core.frame import to_frame as _to_frame_excl
        from goldenmatch.core.quality_exclusions import (
            detect_autoconfig_exclusions,
        )

        _excl_frame = _to_frame_excl(df)
        # Combine every exclusion source: top-level config.exclude_columns
        # (this PR), dedupe_df / match_df kwarg via _RUNTIME_EXCLUDE_COLUMNS
        # ContextVar, QualityConfig.autoconfig_force_{exclude,include}
        # (#404 sub-field), env vars. force_include rescues from every
        # opt-out path. auto_configure_df has no input config -- only
        # the ContextVar + env vars are visible here; the dedupe_df
        # shim writes both before calling us.
        force_exclude_list, force_include_list = (
            _resolve_effective_exclusion_overrides(config=None)
        )
        # #2540: a frame that is several sources CONCATENATED (no source column)
        # is a different shape from #858's below, which needs a source-indicator
        # column to key on -- measured: `_detect_source_partition` returns None
        # on abt_buy even though it is Abt+Buy stacked. Report it and route to
        # `match_df`; deliberately do NOT constrain matching here, since
        # `match_df` already owns cross-source linkage and a second owner for
        # one capability is against the architecture frame. Advisory + fail-open.
        from goldenmatch.core.concat_sources import warn_if_concatenated_sources
        warn_if_concatenated_sources(df)
        # #858: multi-source source-correlated over-merge guard (spec §0-§4).
        # Detect the source partition on the FULL frame, then exclude the
        # source-indicator column + every 0-cross-source-overlap column from
        # match features by folding them into force_exclude (dropped below).
        _ms_profiles = profile_columns(df)
        _ms_partition = _detect_source_partition(df, _ms_profiles)
        _ms_exclude = _source_correlated_exclusions(df, _ms_profiles, _ms_partition)
        if _ms_exclude:
            force_exclude_list = list(force_exclude_list) + [
                c for c in _ms_exclude if c not in force_exclude_list
            ]
        # Internal bookkeeping columns are invisible to detectors.
        skip = {"__row_id__", "__source__"}
        for col in _excl_frame.columns:
            if col.startswith("__") and col.endswith("__"):
                skip.add(col)
        exclusions = detect_autoconfig_exclusions(
            df,
            force_exclude=force_exclude_list,
            force_include=force_include_list,
            skip_columns=skip,
        )
        if exclusions:
            _existing_cols = set(_excl_frame.columns)
            cols_to_drop = [
                ec.column for ec in exclusions if ec.column in _existing_cols
            ]
            for ec in exclusions:
                logger.info(
                    "Auto-config exclusion: %r (detector=%s) -- %s",
                    ec.column, ec.detector, ec.reason,
                )
            if cols_to_drop:
                df = _excl_frame.drop(cols_to_drop).native
                # Match mode: drop the SAME columns from the reference. The
                # detectors ran on the target only, but the exclusion is a
                # statement about the FEATURE ("description is free text"), not
                # about one side of the join -- measured on Amazon-Google, both
                # frames flag `description` independently. Dropping it from the
                # target alone left the two frames at different widths, and
                # `run_match_df`'s polars lane concatenates them with a strict
                # `pl.concat`: every controller iteration raised
                #   ShapeError: unable to append to a DataFrame of width 5 with
                #   a DataFrame of width 6
                # so the run fell back to v0 + a RED sentinel (#2717). The arrow
                # lane hid it -- `pa.concat_tables(promote_options="permissive")`
                # backfills the missing column with nulls instead of raising.
                # Intersected, so a reference that never had the column is fine.
                if reference is not None and not _is_ray_dataset(reference):
                    _ref_frame = _to_frame_excl(reference)
                    _ref_drop = [c for c in cols_to_drop if c in set(_ref_frame.columns)]
                    if _ref_drop:
                        reference = _ref_frame.drop(_ref_drop).native
            _LAST_AUTOCONFIG_EXCLUSIONS.set(list(exclusions))
        else:
            _LAST_AUTOCONFIG_EXCLUSIONS.set([])

    # Lazy import to avoid circular imports (controller imports back here)
    from goldenmatch.core.autoconfig_controller import (
        AutoConfigController,
        ControllerBudget,
        resolve_planning_effort,
    )
    from goldenmatch.core.autoconfig_policy import HeuristicRefitPolicy, LLMRefitPolicy

    # Spec 2026-06-06 §Phase 0: resolve the planning-effort tier (explicit
    # kwarg → GOLDENMATCH_PLANNING_EFFORT env → "normal"). It scales the
    # controller budget and (at thinking+) flips the planner to measured
    # blocking.
    effort = resolve_planning_effort(planning_effort)

    memory = _get_default_memory()
    if _AUTOCONFIG_LLM_ENABLED:
        policy: HeuristicRefitPolicy | LLMRefitPolicy = LLMRefitPolicy(HeuristicRefitPolicy())
    else:
        policy = HeuristicRefitPolicy()
    controller = AutoConfigController(
        policy=policy,
        budget=ControllerBudget.for_dataset(_n_rows_for_budget, effort),
        memory=memory,
    )
    v0_kw = {
        "llm_provider": llm_provider,
        "domain_config": domain_config,
        "llm_auto": llm_auto,
        "strict": strict,
        "allow_remote_assets": allow_remote_assets,
        # #858: full-frame source-partition detection result, threaded to
        # build_matchkeys via _legacy_auto_configure_v0 for phone demotion.
        "multi_source": _ms_partition is not None,
    }
    # #876: let a caller configure FOR a larger target population than `df` (e.g.
    # build a frozen config from a small oracle but FOR 200M rows, so
    # build_blocking's scale gate projects to the real scale). _initial_config
    # forwards v0_kwargs["n_rows_full"] to the v0 heuristic (its guard lets the
    # caller's value win over the controller's df.height default).
    if n_rows_full is not None:
        v0_kw["n_rows_full"] = n_rows_full
    # Resolve throughput config now so the controller can thread it through
    # the plan-build/apply_to site (Task 9). None when throughput is not requested.
    _resolved_tp_cfg = None
    if throughput is not None:
        from goldenmatch.core.throughput_verify import resolve_throughput_config as _rtc
        _resolved_tp_cfg = _rtc(throughput, None)
    config, profile, history = controller.run(
        df,
        reference=reference,
        v0_kwargs=v0_kw,
        skip_finalize=_skip_finalize,
        confidence_required=confidence_required,
        allow_red_config=allow_red_config,
        planning_effort=effort,
        throughput=_resolved_tp_cfg,
        fused_match_allowed=fused_match_allowed,
    )
    # Surface the resolved tier on the committed config for observability
    # (telemetry, YAML round-trip). No-op for the default "normal".
    try:
        config.planning_effort = effort  # type: ignore[assignment]
    except Exception:
        pass

    # Backend selection is driven by the controller v3 planner inside
    # AutoConfigController.run -- it captures RuntimeProfile, extrapolates
    # the committed BlockingProfile to full-row count, and writes the
    # selected backend onto config via ExecutionPlan.apply_to.

    _LAST_CONTROLLER_RUN.set((profile, history))

    # Throughput tier (#1083): ensure blocking + _throughput_plan are set on the
    # returned config. The controller handles this when it reaches the plan-apply
    # site; for early-return paths (1-column, 1-row) we do it here.
    if _resolved_tp_cfg is not None and _resolved_tp_cfg.enabled:
        import polars as _pl_tp2
        _tp_profiles2 = profile_columns(df) if isinstance(df, _pl_tp2.DataFrame) else []
        if _tp_profiles2:
            try:
                _tp_blk2 = _throughput_blocking(_tp_profiles2, config)
                config.blocking = _tp_blk2
                from goldenmatch.core.throughput_verify import metric_and_signature_len
                _tp_metric2, _tp_siglen2 = metric_and_signature_len(_tp_blk2)
                from goldenmatch.core.autoconfig_planner import apply_throughput_overlay as _ato
                from goldenmatch.core.execution_plan import ExecutionPlan as _EP
                _base_plan2 = config._throughput_plan or _EP()
                if getattr(_base_plan2, "verify_mode", "full") == "full":
                    config._throughput_plan = _ato(
                        _base_plan2, _resolved_tp_cfg,
                        metric=_tp_metric2, signature_len=_tp_siglen2,
                    )
            except Exception:
                pass
        config.throughput = _resolved_tp_cfg

    # F1: optional field-group detection (default OFF, fail-open).
    # Runs after the controller has committed the config so explicit
    # golden_rules.field_groups (if any) are already in place.
    _maybe_detect_field_groups(df, config)

    # Perceptual media-as-evidence wiring (ADR 0022, default OFF, fail-open).
    # When enabled, detect fixed-width-hex perceptual-hash columns (image pHash /
    # audio fingerprint) and append a phash/audio_fp matchkey (+ perceptual
    # blocking for an image column when nothing else blocks). Byte-identical when
    # the flag is off.
    if os.environ.get("GOLDENMATCH_PERCEPTUAL_AUTOCONFIG", "0") == "1" and is_polars_dataframe(
        df
    ):
        try:
            from goldenmatch.core.perceptual_autoconfig import apply_perceptual_autoconfig

            config = apply_perceptual_autoconfig(config, df)
        except Exception:  # noqa: BLE001 - additive + fail-open, never break auto-config
            logger.debug(
                "perceptual auto-config hook failed; leaving config unchanged",
                exc_info=True,
            )

    return config


def _legacy_auto_configure_v0(  # pyright: ignore[reportUnusedFunction]  # kept for historical reference
    df: pl.DataFrame,
    *,
    reference: pl.DataFrame | None = None,  # ignored for v1; Task 5.1 plumbs it
    llm_provider: str | None = None,
    domain_config: Any = None,
    llm_auto: bool = False,
    strict: bool = False,
    allow_remote_assets: bool = False,
    n_rows_full: int | None = None,
    multi_source: bool = False,
) -> GoldenMatchConfig:
    """Legacy column-profiling + rule-based auto-config heuristic (the v0
    starting point for the controller). Implementation unchanged from the
    pre-Task-4.x ``auto_configure_df`` — just renamed and given a private
    underscore prefix so the controller can call it without recursing.

    Args:
        df: input frame (may be the controller's sub-sample, not the
            full population). Profile + rule decisions made against
            this frame; the candidate-pool gate uses ``n_rows_full``
            below to project sample cardinalities to full scale.
        n_rows_full: full-population row count. When the controller
            sub-samples a large frame, this carries the original row
            count so Chao1 sample-correction has the right denominator
            (#410). When ``None`` (direct callers, tests passing a
            full frame), defaults to ``df.height``.
    """
    # Initialized up front so the preflight-wiring block at the bottom can
    # safely test `if domain_profile is not None` even when the domain
    # branch below is skipped (e.g. user-provided domain_config).
    domain_profile = None
    # Arrow-port: route the v0 heuristic's frame ops through the seam so a bare
    # pa.Table (zero-config polars-free path) is never subscripted with df[col]
    # / df.height / df.columns / df.with_row_index (all crash on pa.Table).
    # to_frame is idempotent -> byte-identical for the polars backend.
    from goldenmatch.core.frame import to_frame as _tf
    # Preserve the raw input df for preflight. The in-function `df` variable
    # gets enriched with __row_id__ / domain-extracted columns; preflight
    # needs to check against the shape the pipeline will see.
    df_input = df
    # #410: total_rows is the FULL population, not the sample. When the
    # controller passes a 5K sample of a 1.13M-row frame, df.height = 5K
    # but the gate needs 1.13M to scale via Chao1. Caller threads the
    # true count via n_rows_full; falls back to df.height for direct
    # callers (tests / non-controller paths) that pass full frames.
    total_rows = n_rows_full if n_rows_full is not None else _tf(df).height

    logger.info("Auto-configuring %d rows, %d columns", total_rows, len(_tf(df).columns))

    _emit_data_profile(df)

    # Profile columns
    profiles = profile_columns(df, llm_provider=llm_provider)

    logger.info(
        "Detected column types: %s",
        {p.name: p.col_type for p in profiles},
    )

    # ── Domain detection + conditional extraction ──
    extracted_columns = []

    if domain_config is not None:
        # Manual override: skip auto-detection. Synthesize a profile-like
        # object so preflight Check 1 can still auto-repair domain-extracted
        # column refs in the manual-override path. Without this, passing an
        # explicit DomainConfig silently defeats the preflight repair.
        from goldenmatch.core.domain import DomainProfile
        domain_profile = DomainProfile(
            name=domain_config.mode or "manual",
            confidence=1.0,
            text_columns=[],  # unknown; not used by preflight's repair path
        )
        logger.info("Domain config provided manually, skipping auto-detection")
    else:
        from goldenmatch.core.domain import detect_domain, extract_features

        user_cols = [c for c in _tf(df).columns if not c.startswith("__")]
        domain_profile = detect_domain(user_cols)

        if domain_profile.confidence > 0.7:
            original_cols = set(_tf(df).columns)
            # extract_features requires __row_id__ column
            if "__row_id__" not in _tf(df).columns:
                df = _tf(df).with_row_index("__row_id__").native
            df, _low_conf_ids = extract_features(df, domain_profile)
            extracted_columns = [c for c in _tf(df).columns if c.startswith("__") and c not in original_cols]
            logger.info(
                "Domain '%s' detected (confidence=%.2f), extracted %d feature columns",
                domain_profile.name, domain_profile.confidence, len(extracted_columns),
            )
        else:
            logger.info(
                "Domain '%s' (confidence=%.2f) below threshold, skipping extraction",
                domain_profile.name, domain_profile.confidence,
            )

    # Build matchkeys
    matchkeys = build_matchkeys(profiles, df=df, multi_source=multi_source)

    # Probabilistic routing (gated, default-off): a probabilistic-shaped dataset
    # (no surviving identifier/email/phone exact matchkey + >=2 fuzzy fields) is
    # better served by the Fellegi-Sunter path. Delegate to
    # auto_configure_probabilistic_df (it builds the diversified FS blocking that
    # lifts recall) and return directly. NOTE (v1 scoping): this fires before the
    # domain-extracted matchkeys are appended below, so a domain-extracted identifier
    # is not seen by the trigger; and auto_configure_probabilistic_df re-profiles df
    # (excluding __-prefixed cols), so any already-extracted domain feature is dropped
    # on the routed path. Acceptable while default-off; revisit if a domain-heavy
    # dataset misroutes or needs its domain features under FS.
    # Domain-heavy data whose extraction produced real domain matchkey columns
    # (electronics/software/biblio keys in _DOMAIN_SCORER_MAP, mostly strong-id
    # exact) stays on the deterministic domain-aware path: those matchkeys are a
    # strong identity signal the FS routed path would drop (it re-profiles df,
    # excluding __-cols). This is the "revisit if a domain-heavy dataset misroutes"
    # the routing note called for once routing went default-on. NOTE: person-shape
    # extraction can populate __-cols that are NOT domain matchkey columns, so gate
    # on _DOMAIN_SCORER_MAP membership, not on extracted_columns being non-empty.
    _domain_matchkey_cols = [c for c in extracted_columns if c in _DOMAIN_SCORER_MAP]
    _prob_route = (not multi_source and not _domain_matchkey_cols
                   and _route_to_probabilistic_enabled()
                   and _is_probabilistic_shape(matchkeys, profiles))
    if _prob_route and total_rows < _fs_route_min_rows():
        # Small-N guard: FS EM is data-starved below the floor and under-merges
        # fuzzy-close variants the weighted fuzzy path catches (see _fs_route_min_rows).
        # Surface the decision rather than silently skipping the route.
        logger.info(
            "Probabilistic-shaped dataset (%d rows) is below the FS routing floor "
            "(%d rows); staying on the weighted path (EM is data-starved at this "
            "scale). Override with GOLDENMATCH_FS_ROUTE_MIN_ROWS.",
            total_rows, _fs_route_min_rows(),
        )
        _prob_route = False
    if _prob_route:
        routed = auto_configure_probabilistic_df(
            df, llm_provider=llm_provider, n_rows_full=total_rows
        )
        # Parity with the deterministic tail: attach the preflight report so a
        # routed FS config carries the same verification/diagnostic the default
        # path does (the config-lint + auto-repair surface). Standardization is
        # attached inside auto_configure_probabilistic_df.
        from goldenmatch.core.autoconfig_verify import ConfigValidationError, preflight
        routed._strict_autoconfig = strict
        _routed_report = preflight(
            df_input, routed, profiles=profiles,
            allow_remote_assets=allow_remote_assets,
        )
        if _routed_report.has_errors:
            raise ConfigValidationError(report=_routed_report)
        routed._preflight_report = _routed_report
        return routed

    # ── Add domain-extracted fields to matchkeys ──
    if extracted_columns:
        domain_exact = []
        domain_fuzzy = []
        for col in extracted_columns:
            if col not in _DOMAIN_SCORER_MAP:
                continue
            scorer, weight, transforms = _DOMAIN_SCORER_MAP[col]
            _fcol = _tf(df)
            _h = _fcol.height
            null_rate = _fcol.column(col).null_count() / _h if _h > 0 else 0
            cardinality_ratio = _fcol.column(col).n_unique() / _h if _h > 0 else 0
            if null_rate > 0.5:
                continue
            if scorer == "exact" and cardinality_ratio < 0.01:
                continue
            mf = MatchkeyField(field=col, scorer=scorer, weight=weight, transforms=transforms)
            if scorer == "exact" and _is_identity_claim(col, weight, cardinality_ratio):
                domain_exact.append(mf)
            elif scorer == "exact":
                # A weak exact signal: still compared, but as a WEIGHTED field
                # carrying its declared weight -- never as a standalone identity
                # claim. See `_is_identity_claim`.
                domain_fuzzy.append(mf)
            else:
                domain_fuzzy.append(mf)

        # Add domain exact matchkeys
        for f in domain_exact:
            matchkeys.append(MatchkeyConfig(
                name=f"domain_exact_{f.field.strip('_')}",
                type="exact",
                fields=[MatchkeyField(field=f.field, transforms=f.transforms)],
            ))

        # Add domain fuzzy fields to existing weighted matchkey (or create one)
        if domain_fuzzy:
            weighted = [mk for mk in matchkeys if mk.type == "weighted"]
            if weighted:
                weighted[0].fields.extend(domain_fuzzy)
            else:
                matchkeys.append(MatchkeyConfig(
                    name="domain_fuzzy",
                    type="weighted",
                    threshold=0.80,
                    fields=domain_fuzzy,
                ))

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

    # ── Add domain columns to blocking candidate profiles ──
    if extracted_columns:
        for col in extracted_columns:
            if col not in _DOMAIN_SCORER_MAP:
                continue
            scorer, _weight, _transforms = _DOMAIN_SCORER_MAP[col]
            if scorer != "exact":
                continue
            _fcol2 = _tf(df)
            _h2 = _fcol2.height
            null_rate = _fcol2.column(col).null_count() / _h2 if _h2 > 0 else 0
            cardinality_ratio = _fcol2.column(col).n_unique() / _h2 if _h2 > 0 else 0
            if null_rate > 0.5:
                continue
            profiles.append(ColumnProfile(
                name=col, dtype="Utf8", col_type=_EXACT_DERIVED_COL_TYPE,
                confidence=0.9, null_rate=null_rate,
                cardinality_ratio=cardinality_ratio, avg_len=0,
            ))

    # Build blocking (required for weighted/probabilistic matchkeys).
    # #410: thread total_rows so the cardinality gate + composite search
    # can sample-correct via Chao1 when the controller passed us a sub-
    # sample of a much larger frame.
    has_fuzzy = any(mk.type in ("weighted", "probabilistic") for mk in matchkeys)
    blocking = build_blocking(
        profiles, df,
        llm_provider=llm_provider,
        n_rows_full=total_rows,
    ) if has_fuzzy else None
    # #2717: same deferral on the weighted path.
    blocking = defer_free_text_blocking_to_analyzer(blocking, df)
    # Quality-aware blocking (door #1): add fuzzy-tolerant passes for columns
    # GoldenCheck flags as edit-distance-fuzzy. Fail-open + additive; OFF unless
    # GOLDENMATCH_QUALITY_AWARE_BLOCKING=1. Runs before adaptive promotion.
    blocking = apply_quality_aware_blocking(blocking, profiles, df)
    # Auto-enabled semantic blocking (door, #1090): route text-heavy data to
    # SimHash-over-embeddings; honest no-op when embeddings are unavailable. OFF
    # unless GOLDENMATCH_AUTO_SEMANTIC_BLOCKING=1, so default output is identical.
    blocking = apply_auto_semantic_blocking(blocking, profiles, df)
    # At 1M+ rows, swap static blocking for adaptive so blocker.py's
    # _sub_block / _auto_split_block paths bound oversized buckets at
    # runtime. No-op for multi_pass / canopy / ann / learned / sorted_neighborhood.
    blocking = _maybe_promote_blocking_to_adaptive(blocking, _tf(df).height)
    # Opt-in (GOLDENMATCH_BLOCKING_PRUNE_PASSES=1): drop redundant/all-noise
    # multi-pass passes by likely-match yield.
    blocking = _maybe_prune_blocking_passes(blocking, df)
    # ── Data-driven strategy selection ──

    # 1. Learned blocking for large datasets.
    #
    # Gated at >= 50K rows because the learner needs two things the sample
    # cap below cannot provide on smaller inputs:
    #
    #   a) held-out rows to generalize to — `learned_sample_size` caps the
    #      training sample at 25% of the dataset, max 5K. Below 50K that cap
    #      is tight enough (<=12.5K training / 37.5K held-out) to produce
    #      predicates that generalize instead of memorizing the input.
    #
    #   b) enough rows to amortize training cost — below 50K, static or
    #      multi_pass blocking is usually faster and comparable in quality.
    if blocking is not None and total_rows >= 50_000:
        # #1316: a #1207 strong-identifier UNION is purpose-built for null-sparse
        # multi-source strong-id data. Forcing learned blocking here DISCARDS it
        # and under-blocks catastrophically -- measured on the #1207 shape at 50K,
        # candidate-pair recall collapses 1.0 -> 0.0 (the learner trains on a
        # <=5K sample, finds no pairs above its recall threshold, falls back to a
        # single column, and skip_oversized then drops every resulting giant
        # block). The union's per-id passes are near-unique and naturally bounded
        # (it already sets skip_oversized + max_block_size), so learned's
        # block-bounding value is redundant. `build_blocking` already gated the
        # union on OR-coverage before emitting it, so KEEP any strong-id union it
        # returned rather than overwriting it. Learned blocking stays the default
        # for every non-union large shape.
        if _is_strong_identifier_union(blocking, profiles):
            logger.info(
                "Keeping the strong-identifier blocking UNION at %d rows; learned "
                "blocking under-blocks this null-sparse multi-source shape (see "
                "#1316).",
                total_rows,
            )
        else:
            blocking.strategy = "learned"
            blocking.learned_sample_size = _learned_sample_size(total_rows)
            blocking.learned_min_recall = 0.95
            blocking.skip_oversized = True
            # skip_oversized DROPS whole blocks above max_block_size, so the cap
            # must not sit below the budget that picked the key (see
            # _learned_block_cap).
            from goldenmatch.core._native_loader import native_enabled as _native_enabled

            blocking.max_block_size = _learned_block_cap(
                total_rows, blocking.max_block_size, _native_enabled("block_scoring"),
            )
            logger.info(
                "Upgraded to learned blocking (dataset has %d rows, sample_size=%d, "
                "max_block_size=%d)",
                total_rows, blocking.learned_sample_size, blocking.max_block_size,
            )

    # 2. Reranking for multi-field matchkeys
    for mk in matchkeys:
        if mk.type == "weighted" and len(mk.fields) >= 3:
            mk.rerank = True
            logger.info("Enabled reranking for matchkey '%s' (%d fields)", mk.name, len(mk.fields))

    # 3. Adaptive threshold from data quality
    for mk in matchkeys:
        if mk.type == "weighted" and mk.threshold is not None:
            fuzzy_field_names = {f.field for f in mk.fields if f.field}
            fuzzy_profiles = [p for p in profiles if p.name in fuzzy_field_names]
            if fuzzy_profiles:
                avg_null = sum(p.null_rate for p in fuzzy_profiles) / len(fuzzy_profiles)
                avg_len = sum(p.avg_len for p in fuzzy_profiles) / len(fuzzy_profiles)
                original = mk.threshold
                if avg_null > 0.15:
                    mk.threshold = max(mk.threshold - 0.05, 0.50)
                elif avg_len < 5:
                    mk.threshold = min(mk.threshold + 0.05, 0.95)
                if mk.threshold != original:
                    logger.info(
                        "Adjusted threshold for '%s': %.2f -> %.2f (avg_null=%.2f, avg_len=%.1f)",
                        mk.name, original, mk.threshold, avg_null, avg_len,
                    )

    # ── LLM auto-config ──
    llm_scorer_config = None
    if llm_auto:
        import os
        _provider = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            _provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            _provider = "openai"
        if _provider:
            llm_scorer_config = LLMScorerConfig(
                enabled=True,
                candidate_lo=0.60,
                candidate_hi=0.90,
                auto_threshold=0.90,
                budget=BudgetConfig(max_cost_usd=0.05),
            )
            logger.info("LLM scorer auto-enabled (provider=%s, budget=$0.05)", _provider)
        else:
            logger.info("llm_auto=True but no API key found")

    memory_config = MemoryConfig(enabled=True) if llm_auto else None

    # Auto-detect standardization rules (Change 1, 2026-05-07). Shared with the
    # probabilistic routed path (_detect_standardization_config) so a routed FS
    # config gets the same input normalization as the deterministic default.
    _standardization = _detect_standardization_config(profiles)

    # Capture choices in a decisions object so a future iterative-tuning loop
    # can nudge them without re-profiling. Matchkeys and blocking strategy/
    # keys/passes all flow through decisions; non-decision runtime attributes
    # on the transient `blocking` object (learned_sample_size, min_recall,
    # skip_oversized, etc.) are preserved in the rebuild step below.
    decisions = AutoConfigDecisions(
        blocking_strategy=blocking.strategy if blocking is not None else "none",
        blocking_keys=list(blocking.keys) if (blocking is not None and blocking.keys) else [],
        blocking_passes=list(blocking.passes) if (blocking is not None and blocking.passes) else [],
        matchkeys=matchkeys,
        threshold=0.0,  # populated in a later task
        # domain_mode mirrors the detected domain profile (when present).
        # Demonstrates the field's contract even though no consumer reads it
        # yet — populating now means future iterative tuners can inspect it
        # without us also having to backfill the call site.
        domain_mode=(domain_profile.name if domain_profile is not None else None),
        llm_enabled=llm_scorer_config is not None,
        allow_remote_assets=False,
    )

    config = _rebuild_from_decisions(
        profiles,
        decisions,
        transient_blocking=blocking,
        llm_scorer_config=llm_scorer_config,
        memory_config=memory_config,
        standardization_config=_standardization,
    )

    # ── Preflight verification ──
    #
    # Stash the domain profile so preflight Check 1 can auto-repair
    # `config.domain` when the generated config references
    # domain-extracted columns (e.g. __title_key__, __model_norm__).
    # This fixes the DBLP-ACM crash where auto-config emitted
    # references to columns produced by a disabled pipeline step.
    from goldenmatch.core.autoconfig_verify import ConfigValidationError, preflight

    if domain_profile is not None and domain_profile.confidence > 0.7:
        config._domain_profile = domain_profile
    config._strict_autoconfig = strict

    report = preflight(
        df_input, config, profiles=profiles,
        allow_remote_assets=allow_remote_assets,
    )
    if report.has_errors:
        raise ConfigValidationError(report=report)
    config._preflight_report = report
    return config


def _rebuild_from_decisions(
    _profiles: list[ColumnProfile],
    decisions: AutoConfigDecisions,
    *,
    transient_blocking: BlockingConfig | None,
    llm_scorer_config: LLMScorerConfig | None,
    memory_config: MemoryConfig | None,
    standardization_config: StandardizationConfig | None = None,
) -> GoldenMatchConfig:
    """Assemble a GoldenMatchConfig from decisions (+ runtime hand-offs).

    Pure function of (profiles, decisions) plus non-decision runtime values:
      - `transient_blocking` carries non-decision attributes (learned_sample_size,
        learned_min_recall, skip_oversized, ...) that survive the rebuild.
      - `llm_scorer_config` / `memory_config` are runtime plumbing, not decisions.
      - `standardization_config` is auto-detected from column types (Change 1).

    Splitting this out lets a future iterative-tuning loop mutate `decisions`
    and re-call `_rebuild_from_decisions` without re-running profile_columns /
    build_matchkeys / build_blocking.

    ``_profiles`` keeps its underscore for call-compatibility, but is no longer
    unused: `_drop_constant_scored_fields` reads `n_distinct` off it to find
    scored columns that cannot discriminate anything (#2668).
    """
    # Rebuild final blocking from decisions, preserving runtime-only attrs
    # (learned_sample_size, learned_min_recall, skip_oversized, etc.) from
    # the transient blocking object produced above.
    final_blocking: BlockingConfig | None
    if transient_blocking is None:
        final_blocking = None
    else:
        final_blocking = transient_blocking.model_copy(update={
            "strategy": decisions.blocking_strategy,
            "keys": decisions.blocking_keys,
            "passes": decisions.blocking_passes,
        })

    _drop_uninformative_blocking_fields(decisions.matchkeys, final_blocking)
    _drop_constant_scored_fields(decisions.matchkeys, _profiles)

    return GoldenMatchConfig(
        matchkeys=decisions.matchkeys,
        blocking=final_blocking,
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
        llm_scorer=llm_scorer_config,
        memory=memory_config,
        standardization=standardization_config,
    )


def _read_autoconfig_input(path: Path):
    """Read one auto-config input file into a ``pyarrow.Table``, polars-free.

    This is the zero-config FILE front door -- ``goldenmatch dedupe data.csv``
    lands here before anything else runs. It used to read via ``pl.read_csv`` /
    ``pl.read_excel`` / ``pl.read_parquet``, which made a plain
    ``pip install goldenmatch`` (polars is an optional extra, and the rest of
    the autoconfig stack has been arrow-native since #1754) fail 100% of the
    time with "Auto-config error: No module named 'polars'".

    Reads through ``read_table_arrow``, whose polars-parity is pinned by
    ``tests/test_io_arrow_ingest_parity.py``. The legacy polars call passed
    ``ignore_errors=True`` (null out cells that do not parse to the inferred
    dtype) and accepted ``.xls``, neither of which the stricter arrow reader
    does, so a reader failure falls back to polars WHERE IT IS INSTALLED rather
    than turning a file that used to load into a hard error. With polars absent
    there is nothing to fall back to and the arrow error is re-raised -- a real
    reader diagnostic, not a bare ImportError.
    """
    from goldenmatch.core.io_arrow import read_table_arrow

    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet" or suffix == ".xlsx":
            return read_table_arrow(path)
        if suffix == ".xls":
            raise ValueError("Unsupported file format: '.xls'")
        return read_table_arrow(path, encoding="utf8-lossy")
    except Exception as arrow_exc:  # noqa: BLE001 -- any reader failure retries on polars
        try:
            if suffix in (".xlsx", ".xls"):
                fallback = pl.read_excel(path, engine="openpyxl")
            elif suffix == ".parquet":
                fallback = pl.read_parquet(path)
            else:
                fallback = pl.read_csv(
                    path, encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True
                )
        except ImportError:
            # No polars to fall back to: surface the arrow reader's own error.
            raise arrow_exc from None
        logger.warning(
            "arrow ingest failed for %s (%s); fell back to the polars reader. "
            "Auto-config still ran, but this file will NOT load on a "
            "polars-free install.",
            path,
            arrow_exc,
        )
        # Downstream is arrow-native; hand it a table so the fallback does not
        # drag a polars frame through the rest of auto-config.
        return fallback.to_arrow()


def auto_configure(files: list[tuple[str, str]]) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from input files.

    Args:
        files: List of (path, source_name) tuples.

    Returns:
        A fully populated GoldenMatchConfig ready for pipeline execution.
    """
    import pyarrow as pa

    tables = [_read_autoconfig_input(Path(path)) for path, _source_name in files]

    # polars' ``how="diagonal"`` unions the column sets and null-fills what a
    # given file lacks; ``promote_options="permissive"`` is arrow's equivalent
    # (and the spelling already used for the same job at pipeline.py:6108).
    combined = (
        pa.concat_tables(tables, promote_options="permissive")
        if len(tables) > 1
        else tables[0]
    )
    return auto_configure_df(combined)


def build_probabilistic_matchkeys(profiles: list[ColumnProfile]) -> list[MatchkeyConfig]:
    """Generate Fellegi-Sunter probabilistic matchkeys from column profiles.

    Produces a single probabilistic matchkey using all matchable columns
    with appropriate comparison levels and partial thresholds.

    FS-autoconfig v2 (default ON since 2026-06-09; ``GOLDENMATCH_FS_AUTOCONFIG_V2=0``
    restores the legacy field set) curates a cleaner comparison set — the
    dominant scoring lever on error-heavy PII data (audit: historical_50k) and
    the fix for bibliographic data (DBLP-ACM). Four changes vs v1:
      1. **Admit date columns** (e.g. ``dob``) as ``levenshtein`` comparison
         fields. Birth date is the strongest person-identity discriminator;
         Splink leans on it (DamerauLevenshtein). v1 skipped all dates, scoring
         identity with no birth-date signal.
      2. **Drop redundant person-name composites** (``full_name`` /
         ``first_and_surname`` ...) when atomic given + family fields are both
         present. Correlated name comparisons violate FS conditional
         independence — N copies of one name (dis)agreement get N-counted,
         sinking corrupted-name true pairs below threshold.
      3. **Floor fuzzy fields at low cardinality** (drop a ``gender``-like field
         at cardinality ~0.002): negligible identity signal + unstable
         (non-monotonic) EM weights. Exact-scorer identifiers keep the #721
         no-floor admission (FS self-regulates them via u).
      4. **Admit free-text + multi-name fields** (``description`` /
         ``multi_name``) as ``token_sort`` comparison fields. v1 dropped both, so
         on bibliographic data (DBLP-ACM: title=description, authors=multi_name)
         only a near-constant ``venue`` survived → F-S mega-matched (P~0).
         DBLP-ACM via auto-config: F1 ~0.003 -> 0.879.
    """
    v2 = _fs_autoconfig_v2_enabled()

    # Atomic-name presence is computed once: composites are dropped only when
    # BOTH an atomic given-name and an atomic family-name field exist, so a
    # dataset carrying only ``full_name`` keeps it.
    if v2:
        names_norm = {_norm_colname(p.name) for p in profiles}
        atomic_name_present = bool(names_norm & _ATOMIC_GIVEN_NAMES) and bool(
            names_norm & _ATOMIC_FAMILY_NAMES
        )
    else:
        atomic_name_present = False

    fields = []
    for p in profiles:
        # #721: identifiers ARE admitted to the probabilistic (Fellegi-Sunter)
        # path. Unlike the exact path's cardinality BAND (#715), F-S needs no
        # lower floor: a weak (low-cardinality) identifier self-regulates because
        # its higher u (agreement among non-matches) yields a smaller EM weight,
        # so EM down-weights it rather than mega-clustering. m/u estimation and
        # blocking-field exclusion are EM's job at train time (train_em
        # blocking_fields=...), not this builder's.

        # Phase 2 domain comparators (spec 2026-07-23), behind
        # GOLDENMATCH_FS_DOMAIN_COMPARATORS (default off -> this whole block is
        # skipped and the columns fall through to their v1/v2 handling below, so
        # default behavior is byte-identical). Magnitude-aware numeric + geo:
        #   * a single combined "lat,long" column -> great-circle `geo_haversine`
        #     (two separate lat/long columns are the deferred cross-field case);
        #   * a numeric column -> `numeric_diff` (10% relative band) instead of
        #     being skipped as a comparison field (the `numeric` skip below).
        # Both use partial_threshold 0.6 so the mid bands land at level 1 (as the
        # date_diff admission does). The card >= 1.0 guard drops per-record
        # surrogates (no shared-identity signal), mirroring the date branch.
        if v2 and _fs_domain_comparators_enabled():
            # Per-column reliability gate (Phase 2): admit the magnitude comparator
            # only when the column actually parses as its domain; otherwise fall
            # through to the conservative v2 default (the column isn't admitted as
            # that comparator). See `_numeric_column_reliable_for_diff` /
            # `_coord_column_reliable_for_haversine`.
            if (not _is_perfect_surrogate(p) and _looks_like_latlong(p)
                    and _coord_column_reliable_for_haversine(p)):
                fields.append(MatchkeyField(
                    field=p.name, scorer="geo_haversine", transforms=["strip"],
                    levels=3, partial_threshold=0.6,
                ))
                continue
            if (p.col_type == "numeric" and not _is_perfect_surrogate(p)
                    and _numeric_column_reliable_for_diff(p)):
                fields.append(MatchkeyField(
                    field=p.name, scorer="numeric_diff:pct:0.1", transforms=["strip"],
                    levels=3, partial_threshold=0.6,
                ))
                continue

        # Lever #1: admit date columns as comparison fields. Default scorer is
        # edit-distance (`levenshtein`); with GOLDENMATCH_FS_DOMAIN_COMPARATORS on
        # the magnitude-aware `date_diff` is used instead (spec 2026-07-23) -- a
        # year DOB gap becomes a weak partial rather than a near-match. partial=0.6
        # for date_diff so the 0.60 (<=1y) band lands at level 1, not level 0.
        if v2 and p.col_type == "date":
            if _is_perfect_surrogate(p):
                continue  # per-record timestamp surrogate: no shared-identity signal
            # Phase 2 per-column decision: date_diff only on a RELIABLE date
            # column (high parse rate); a messy one keeps levenshtein even under
            # the flag, so the flag-on path is a net win instead of the global
            # flip's historical_50k regression. See `_date_column_reliable_for_diff`.
            _date_dc = _fs_domain_comparators_enabled() and _date_column_reliable_for_diff(p)
            fields.append(MatchkeyField(
                field=p.name,
                scorer="date_diff" if _date_dc else "levenshtein",
                transforms=["strip"],
                levels=3,
                partial_threshold=0.6 if _date_dc else 0.8,
            ))
            continue

        # Lever #4 (bibliographic): admit free-text + multi-name comparison
        # fields. v1 dropped `description` (the skip-list below) and `multi_name`
        # (absent from _SCORER_MAP), so on bibliographic data (DBLP-ACM:
        # title=description, authors=multi_name) only a near-constant `venue`
        # survived → F-S mega-matched (precision ~0). token_sort mirrors the
        # weighted builder's choice for these col_types. No cardinality gate:
        # a near-unique fuzzy field is HIGH-discrimination for F-S (agreement is
        # rare among non-matches → large EM weight), unlike an exact surrogate.
        if v2 and p.col_type in ("description", "multi_name"):
            fields.append(MatchkeyField(
                field=p.name,
                scorer="token_sort",
                transforms=["lowercase", "strip"],
                levels=3,
                partial_threshold=0.8,
            ))
            continue

        if p.col_type in ("numeric", "date", "description"):
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, _weight, transforms = scorer_info

        # Perfect-surrogate hygiene gate, but NOT for identity-bearing VALUE types
        # (email / phone). A card == 1.0 exact field is EITHER a per-record
        # surrogate (a row PK -- never shared, no identity signal, correctly
        # excluded) OR a shared real-world identifier a duplicate carries verbatim
        # -- F-S's single strongest signal. Cardinality alone cannot tell them
        # apart, AND the ratio is measured on a SAMPLE that under-represents
        # duplicates (a head sample of base-then-dup data sees a shared email as
        # perfectly unique), so a blanket >= 1.0 exclusion silently drops the best
        # comparison field and collapses the whole EM model to zero matches at
        # scale (measured: zero-config FS F1 0.0 at 1M on realistic person data).
        # Unlike the exact-matchkey path (a card==1.0 key emits zero blocking pairs
        # and is useless), an F-S exact field is a COMPARISON field scored on pairs
        # from OTHER blocking: a true PK self-regulates to neutral (m~=u, ~0 EM
        # weight) while a shared identifier carries a large weight -- so admitting
        # is neutral-to-helpful, never harmful (#721 self-regulation). We admit
        # only `email`/`phone` (unambiguously identity-bearing values) and keep
        # excluding the ambiguous bare `identifier` type, which also covers row PKs
        # (e.g. `record_id`) -- those stay out for config hygiene.
        if (
            scorer == "exact"
            and _is_perfect_surrogate(p)
            and p.col_type not in ("email", "phone")
        ):
            continue

        # Lever #2a: drop redundant person-name composites when atomic parts exist.
        if v2 and p.col_type == "name" and atomic_name_present and (
            _norm_colname(p.name) in _COMPOSITE_NAME_FIELDS
        ):
            continue

        # Lever #2b: floor fuzzy (non-exact) fields at low cardinality. A field
        # like `gender` (~0.002) carries no identity signal and gives EM unstable
        # non-monotonic weights. Exact identifiers are exempt (#721 no-floor).
        if v2 and scorer != "exact" and p.cardinality_ratio < _PROB_FUZZY_CARD_FLOOR:
            continue

        # Refdata hook (mirrors build_matchkeys); see module-top fallback
        # for the unavailable-refdata case.
        scorer, transforms = _refdata_refine_matchkey_field(
            p.name, scorer, transforms, p.col_type,
        )

        # Honorific stripping (GOLDENMATCH_FS_STRIP_HONORIFICS, default OFF).
        # Appended last so it runs after lowercase/strip; emits None for a
        # honorific-only value -> FS reads it as missing, not an empty agreement.
        if _strip_honorifics_for(p) and "strip_honorifics" not in transforms:
            transforms = [*transforms, "strip_honorifics"]

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
            tf_adjustment=_tf_adjustment_for(p),
        ))

    if not fields:
        return []

    return [MatchkeyConfig(
        name="probabilistic_auto",
        type="probabilistic",
        fields=fields,
        missing=_pick_missing_semantics(profiles, fields),
    )]


#: A comparison field at or above this null rate makes missingness a strong
#: enough signal to model as disagreement rather than absence of evidence.
#: Calibrated against the quality corpora, not derived: historical_50k's FS
#: fields run 8.9-50% null and need "disagree" (0.83 vs 0.33 f1_probabilistic);
#: febrl3 / ncvr_synthetic are near-complete and score the same either way, so
#: the cut only has to separate those two populations. 0.20 sits in the gap.
_FS_MISSING_DISAGREE_NULL_RATE = 0.20


def _pick_missing_semantics(
    profiles: list[ColumnProfile], fields: list[MatchkeyField]
) -> Literal["unobserved", "disagree"]:
    """Choose FS missing-value semantics from the profiled null rates (#1846).

    Textbook Fellegi-Sunter treats a missing value as UNOBSERVED -- absence of
    evidence, contributing nothing (#1819/#1834). That is correct when data is
    missing at random. It is wrong when missingness is INFORMATIVE: if records
    lacking a DOB are systematically unlike records that have one, "no evidence"
    lets a pair agreeing on its one populated field look like a certain match,
    and null-heavy data mass-merges (historical_50k: 0.83 -> 0.33).

    We cannot test informativeness directly -- that needs labels auto-config does
    not have. Null rate is a PROXY: heavily-null columns in real-world PII are
    usually missing-not-at-random. This can therefore pick wrong on a null-heavy
    but genuinely-random dataset; ``GOLDENMATCH_FS_MISSING`` and the explicit
    ``MatchkeyConfig.missing`` are the escape hatches.
    """
    by_name = {p.name: p for p in profiles}
    rates = [
        by_name[f.field].null_rate
        for f in fields
        if f.field in by_name and by_name[f.field].null_rate is not None
    ]
    if not rates:
        return "unobserved"
    worst = max(rates)
    mode: Literal["unobserved", "disagree"] = (
        "disagree" if worst >= _FS_MISSING_DISAGREE_NULL_RATE else "unobserved"
    )
    logger.info(
        "FS missing-value semantics: %s (max comparison-field null rate %.1f%%, "
        "cut %.0f%%)", mode, worst * 100, _FS_MISSING_DISAGREE_NULL_RATE * 100,
    )
    return mode


def _detect_standardization_config(profiles: list[ColumnProfile]) -> Any:
    """Auto-detect StandardizationConfig from column types (Change 1, 2026-05-07).

    When the classifier identifies a typed column (phone/email/name/address/zip/
    geo), emit the matching standardizer so auto-config normalizes inputs without
    explicit user setup. Returns None when nothing is detected (avoids passing an
    empty StandardizationConfig into the pipeline). Shared by the deterministic
    default path and the probabilistic routed path so both normalize identically.

    StandardizationConfig.rules schema: {column_name: [standardizer_names]}.
    VALID_STANDARDIZERS: email, phone, zip5, address, state, name_proper,
    name_upper, name_lower, strip, trim_whitespace.
    """
    _std_rules: dict[str, list[str]] = {}
    _email_typed_cols: set[str] = set()  # guard: skip address detection on these
    for _p in profiles:
        # #2876: col_type == "exact_derived" (a domain-extracted column like
        # __title_key__) intentionally falls through every branch below and
        # gets NO standardization rule -- it is not an email/phone/zip/name/
        # address, and std_email in particular would null every value that
        # lacks an "@" (i.e. all of them).
        if _p.col_type == "email":
            _email_typed_cols.add(_p.name)
            _std_rules[_p.name] = ["email"]
        elif _p.col_type == "phone":
            _std_rules[_p.name] = ["phone"]
        elif _p.col_type == "zip":
            _std_rules[_p.name] = ["zip5"]
        elif _p.col_type == "geo":
            # state-shaped columns (state_cd, province, country) → state rule
            _cl = _p.name.lower()
            if any(pat in _cl for pat in ("state", "province", "country")):
                _std_rules[_p.name] = ["state"]
        elif _p.col_type == "name":
            # Route all name columns to name_proper (title-case + strip).
            _std_rules[_p.name] = ["name_proper"]
        elif _p.col_type == "address":
            # Guard: skip if column was already typed as email (e.g. email_address)
            if _p.name not in _email_typed_cols:
                _std_rules[_p.name] = ["address"]

    if not _std_rules:
        return None
    from goldenmatch.config.schemas import StandardizationConfig
    logger.info(
        "auto-config: StandardizationConfig auto-detected %d column rules: %s",
        len(_std_rules), sorted(_std_rules.keys()),
    )
    return StandardizationConfig(rules=_std_rules)


def auto_configure_probabilistic_df(
    df: Any,  # pl.DataFrame | pl.LazyFrame | pa.Table (arrow lane)
    llm_provider: str | None = None,
    *,
    n_rows_full: int | None = None,
) -> GoldenMatchConfig:
    """Build a Fellegi-Sunter *probabilistic* config straight from a DataFrame.

    A reachable auto-config entry point for the probabilistic matchkey type.
    The heuristic / iterative ``auto_configure_df`` only emits exact + weighted
    matchkeys (its controller refit rules are weighted-specific), so the
    probabilistic lever was previously unreachable from the auto surface. This
    profiles the columns, builds blocking, and emits a single
    ``type="probabilistic"`` matchkey via ``build_probabilistic_matchkeys``.
    The m/u probabilities are EM-trained at dedupe time (``core/probabilistic.py``);
    this only produces the config.

    Non-iterative by design: it does NOT run the ``AutoConfigController`` (the
    EM training is the "fit", not the heuristic refit loop). Use
    ``auto_configure_df`` for the weighted/iterative path. The agentic config
    optimizer (spec ``2026-05-25-agentic-config-optimizer-design``) is the
    intended place to *search* weighted-vs-probabilistic empirically.
    """
    from goldenmatch.core.frame import is_polars_lazyframe as _is_pl_lf_prob

    if _is_pl_lf_prob(df):
        df = cast("pl.LazyFrame", df).collect()

    profiles = profile_columns(df, llm_provider=llm_provider)
    matchkeys = build_probabilistic_matchkeys(profiles)
    if not matchkeys:
        raise ValueError(
            "No probabilistic matchkeys could be built: no matchable columns "
            "found (probabilistic skips numeric/date/description columns and "
            "perfectly-unique surrogate keys; provide name/address/email/phone/"
            "identifier-like fields)."
        )
    blocking = build_blocking(profiles, df, llm_provider=llm_provider)
    # #2717: a prefix key on free text is near-useless -- let the analyzer decide.
    blocking = defer_free_text_blocking_to_analyzer(blocking, df)
    blocking = _maybe_prune_blocking_passes(blocking, df)
    blocking = apply_quality_aware_blocking(blocking, profiles, df)
    # Lever #3: diversify onto orthogonal stable keys (date-year, postcode/zip,
    # identifier) so the FS candidate set isn't gated entirely on (corrupted)
    # name keys. Purely additive — recall ceiling can only rise.
    blocking = _diversify_probabilistic_blocking(blocking, profiles, df)
    # Field-agnostic orthogonal-anchor blocking (default OFF, out-of-panel
    # validated): add a pass on any well-populated, moderate-cardinality field the
    # name/date/zip passes don't already cover -- the generalization that catches a
    # misclassified orthogonal anchor (e.g. historical_50k `birth_place`, which
    # profiles as `name` and escapes the v2 date/zip whitelist above).
    blocking = _diversify_unused_orthogonal_blocking(blocking, profiles, df)
    # Atomic-name-soundex completion (default OFF): when names are blocked via a
    # composite soundex pass but no atomic single-name soundex pass exists, a
    # corrupted first name breaks the composite key -- add surname-/given-name
    # soundex passes so same-surname duplicates still co-block. Additive; the
    # pair-budget bound below drops any that explode.
    blocking = _add_atomic_name_soundex_blocking(blocking, profiles)
    # Pair-budget gate: bound EVERY pass (build_blocking's soundex passes + the
    # diversified ones) by candidate pairs Σ C(block,2), not just block rows —
    # the row ceiling let ~12.9B-pair configs (dob-YEAR + name-soundex mega-
    # passes) through and OOM-killed gm_probabilistic at 1M. See #1803.
    # `n_rows_full` is load-bearing: auto-config profiles a SAMPLE, and the bound
    # extrapolates each pass's Σ C(block,2) to the full population. Without it the
    # bound measures pairs at sample scale (a 66M-at-1.2M pass looks like ~1.8M at
    # a 200K sample) and never prunes, leaving redundant recall passes that make
    # the wall ~20x the discriminative-key config at scale.
    blocking = _bound_probabilistic_blocking_pairs(
        blocking, profiles, df, n_rows_full=n_rows_full
    )
    return GoldenMatchConfig(
        matchkeys=matchkeys,
        blocking=blocking,
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
        # NOTE: deliberately NO standardization_config here. The FS path EM-trains
        # m/u weights on the raw values; forcing name_proper/address normalization
        # measurably LOWERS FS F1 (ncvr_synthetic f1_probabilistic 0.989 -> 0.978 in
        # the autoconfig quality corpus). FS handles surface variation via EM, so
        # standardization is a deterministic-path lever, not shared parity.
    )


def _diversify_pair_budget_enabled() -> bool:
    """Kill-switch for the pairs/row bound on diversified passes.

    Default ON. ``GOLDENMATCH_BLOCKING_DIVERSIFY_PAIR_BUDGET=0`` restores the
    row-cap-only behaviour without a revert, because the measurement behind the
    bound is person-shaped and the lever's stated purpose is error-heavy PII
    (historical_50k), where an anchor that looks wasteful on clean data may be
    the only thing co-blocking corrupted names.
    """
    return os.environ.get(
        "GOLDENMATCH_BLOCKING_DIVERSIFY_PAIR_BUDGET", "1",
    ).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _diversify_probabilistic_blocking(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
    df: Any = None,
    *,
    n_rows_full: int | None = None,
    max_null_rate: float = 0.6,
) -> BlockingConfig | None:
    """Add orthogonal stable-key blocking passes for the probabilistic path.

    Auto-config blocking tends to key entirely on the name column(s); on
    error-heavy PII data that caps the candidate ceiling (audit: historical_50k
    blocking_recall 0.585 — 42% of true pairs never co-block because their names
    are corrupted even though dob/postcode agree). This appends single-field
    passes on orthogonal anchors — a date column's YEAR (``substring:0:4``) and
    postcode/zip/identifier columns — mirroring Splink's rule union. Purely
    additive (recall can only rise; scoring still decides precision). Skips
    high-null and perfectly-unique columns, and any (field, transforms) signature
    already present. Default ON; ``GOLDENMATCH_FS_AUTOCONFIG_V2=0`` disables it.

    SCALE GUARD (#1857): a low-cardinality anchor becomes a memory bomb at scale.
    Birth YEAR (``substring:0:4``) has ~70 distinct values, so at 1M rows it makes
    ~15K-row blocks (14.4B candidate pairs); the vectorized FS scorer then holds
    several dense NxN float64 matrices per block across a thread pool and OOMs the
    host. So when ``df`` is supplied we measure each candidate single-field pass's
    ACTUAL max block size (exact, post-transform) and DROP the pass if it exceeds
    the FS scorer's per-block row cap (``sqrt(GOLDENMATCH_FS_VEC_MAX_ELEMS)``) —
    the same bound the scorer would otherwise refuse on. The pass stays at the
    scales where it helps (a birth-year block is ~1.5K rows at 100K) and is
    dropped only where it would be pathological. Without ``df`` (older callers)
    the guard is a no-op, preserving prior behavior.

    ``n_rows_full`` scale-corrects that guard for callers handing us a SAMPLE.
    The zero-config controller does: it builds its initial config from a ~5-6K
    sample (measured: 6,324 rows for a 100K frame). Uncorrected, every anchor
    looks tiny on a sample and is kept -- including birth-YEAR, ~1.5K rows at
    100K but ~15K at 1M, which is the exact OOM #1857 added this guard for.
    Block size grows ~linearly with N (``project_max_block_size``). Full-frame
    callers pass None and are byte-identical.

    ``max_null_rate`` defaults to the 0.6 this lever has always used -- additive
    passes tolerate more nulls than a primary key, because the static blocker
    filters null/sentinel block keys so a null row is simply absent from THIS
    pass while still covered by the name passes, and error-heavy PII
    (historical_50k dob/postcode ~24% null) is where these anchors matter most.
    The v0 caller passes ``build_blocking``'s stricter 0.20 instead. That is
    deliberately conservative: v0's Tier-2 contract
    (``test_legacy_v0_skips_high_null_blocking_candidate``) is that a >20%-null
    column is not a blocking field, and whether additive passes should be exempt
    is a real question but an UNMEASURED one. Relaxing it here would be an
    argument, not evidence, so v0 keeps its gate and the question is left for a
    panel run that can actually answer it.
    """
    if blocking is None or not _fs_autoconfig_v2_enabled():
        return blocking

    existing: set[tuple] = set()
    for k in list(blocking.keys or []) + list(blocking.passes or []):
        existing.add((tuple(k.fields), tuple(k.transforms or [])))

    # Row cap above which the FS block scorer refuses (see probabilistic
    # ``_fs_vec_max_elems``): don't ADD a pass that would only be refused later.
    row_cap = 7071
    try:
        from goldenmatch.core.probabilistic import _fs_vec_max_elems

        cap = _fs_vec_max_elems()
        if cap > 0:
            row_cap = int(cap**0.5)
    except Exception:
        pass

    def _projected_max_block(field: str, transforms: list[str]) -> int:
        """Max block size this single-field pass would produce at FULL scale
        (0 if unknown / uncomputable -- treated as safe so the pass is kept).

        Runs on the Frame seam rather than raw polars expressions. It used to
        build ``pl.col(field)`` and call ``df.select(expr)``, which raises on a
        ``pyarrow.Table`` and landed in the ``except`` below -- returning 0,
        which this guard reads as SAFE. Post-Polars-eviction most callers hand
        this a pa.Table, so the #1857 scale guard was silently disabled on the
        arrow lane that is now the default. Found while wiring this lever into
        the controller path: the guard could not drop a birth-YEAR anchor even
        when the projection asked it to.

        It also derives the key through ``derive_block_key``, so the measured
        key matches what the blocker will actually compute (the old code only
        handled a leading ``substring:`` and ignored every other transform).
        """
        if df is None:
            return 0
        try:
            from goldenmatch.core.frame import to_frame as _to_frame  # noqa: PLC0415

            frame = _to_frame(df)
            counts = (
                frame.derive_block_key([field], list(transforms or []))
                .drop_nulls()
                .value_counts_desc()
            )
            measured = int(counts[0][1]) if counts else 0
            if measured and n_rows_full:
                from goldenmatch.core.blocking_candidates import (  # noqa: PLC0415
                    project_max_block_size,
                )
                sample_n = int(frame.height)
                if int(n_rows_full) > sample_n:
                    return int(project_max_block_size(
                        measured, sample_n, int(n_rows_full),
                    ))
            return measured
        except Exception:
            return 0

    new_passes: list[BlockingKeyConfig] = []

    def _add(fields: list[str], transforms: list[str]) -> None:
        sig = (tuple(fields), tuple(transforms))
        if sig in existing:
            return
        if len(fields) == 1:
            projected = _projected_max_block(fields[0], transforms)
            # PAIR budget, not just the per-block ROW cap. The row cap bounds the
            # scorer's per-block memory; it says nothing about how much WORK the
            # pass creates, and those diverge badly for a low-cardinality anchor.
            #
            # Measured, person @ 100,000 rows (run 32138842887), dropping one pass
            # at a time from the 8-pass plan:
            #
            #     dropped pass              comparisons saved      dF1
            #     [dob] substring:0:4              71,337,394   +0.0000
            #     [first_name] soundex             31,321,187   +0.0000
            #     [surname] soundex                12,286,109   +0.0000
            #     [postcode]                           47,022   -0.0269
            #
            # Birth-YEAR is 59% of every comparison in the run and buys NOTHING,
            # while postcode carries all the recall for 47K comparisons. Both
            # clear the row cap easily (1,549 and 9 rows), so the cap cannot tell
            # them apart -- but in pairs/row they are 774 vs 4 against a budget of
            # 50. Same verdict at 4K and 10K, so it is not a scale artifact.
            #
            # `_blocking_pairs_per_row_budget` is the engine's OWN knob (default
            # 50, "keeps the total pair count linear"), already applied when
            #selecting a primary key and never to these additive passes.
            ppr = _project_pairs_per_row(projected)
            budget = _blocking_pairs_per_row_budget()
            if projected and ppr > budget and _diversify_pair_budget_enabled():
                logger.debug(
                    "probabilistic diversify: dropping pass %s%s -- projects %d "
                    "pairs/row (max block %d) against a budget of %d",
                    fields, transforms, ppr, projected, budget,
                )
                return
            if projected > row_cap:
                logger.debug(
                    "probabilistic diversify: dropping oversized pass %s%s "
                    "(projected max block %d rows > FS row cap %d) — would OOM "
                    "the block scorer at this scale (#1857)",
                    fields, transforms, projected, row_cap,
                )
                return
        new_passes.append(BlockingKeyConfig(fields=fields, transforms=transforms))
        existing.add(sig)

    for p in profiles:
        # Additive passes tolerate higher null rates than a primary key: the
        # static blocker filters null/sentinel block keys (blocker.py ~272), so a
        # null-valued row is simply absent from THIS pass (no giant null block)
        # while still covered by the name passes. Error-heavy PII (historical_50k
        # dob/postcode ~24% null) is exactly where these keys matter most.
        if p.null_rate > max_null_rate:
            continue
        if p.col_type == "date":
            _add([p.name], ["substring:0:4"])  # birth YEAR — tolerant of day/month errors
        elif p.col_type in ("zip", "identifier", "phone") and not _is_perfect_surrogate(p):
            _add([p.name], ["strip"])

    if not new_passes:
        return blocking

    # Fold into a multi_pass union; keep the original keys/passes as passes.
    # resolved_keys() dispatches on strategy instead of unconditionally
    # preferring `passes` -- see the resolved_keys() call above for the shape.
    base_passes = blocking.resolved_keys()
    return blocking.model_copy(update={
        "strategy": "multi_pass",
        "passes": base_passes + new_passes,
        "auto_select": False,
    })


def _add_atomic_name_soundex_blocking(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
) -> BlockingConfig | None:
    """Add an additive atomic single-name SOUNDEX pass for each given/family name
    field when the blocking keys names via a COMPOSITE soundex pass but has no
    atomic single-name soundex pass.

    The gap it targets: ``build_blocking`` emits ``full_name`` /
    ``first_and_surname`` soundex passes, so one corrupted name breaks the whole
    composite key -- a pair with a mangled first name but an intact surname never
    co-blocks. Adding the atomic passes DOES recover that CANDIDATE recall
    (historical_50k blocking recall 0.886 -> 0.946, +6pp).

    **KNOWN-NEGATIVE -- default off, do not enable.** The candidate-recall gain
    does NOT convert to end-to-end quality: on the canonical
    ``bench_er_headtohead`` panel this lever REGRESSES historical_50k pairwise F1
    -0.0148 (B3 -0.0078), robustly. The corrupted-name candidates it generates
    fall below the FS threshold (threshold_loss +7pp) and the EM shift from the
    added passes degrades the operating point, so precision AND recall drop. The
    FS scorer simply can't accept the hard corrupted-name candidates at
    acceptable precision -- generating them isn't enough. (An earlier +0.0067 B3
    figure was a stale-pipeline artifact that did not reproduce.) STRIP is worse
    still (B3 precision -0.05). Kept gated for reference / possible salvage.

    Purely additive (``additive=True``) -- co-locates the missed pairs WITHOUT
    demoting the atomic name field from EM scoring, and the scorer still decides
    precision. The downstream ``_bound_probabilistic_blocking_pairs`` bounds/drops
    any pass whose candidate-pair count explodes, so a common-surname mega-block
    can't OOM the run.

    Gated ``GOLDENMATCH_FS_ATOMIC_NAME_BLOCKING`` (default **off** pending the
    full ER panel; ``auto`` = person-shaped only; ``on`` = everywhere). See
    ``_fs_atomic_name_blocking_mode``.
    """
    if blocking is None:
        return blocking
    mode = _fs_atomic_name_blocking_mode()
    if mode == "off":
        return blocking
    if mode == "auto" and not _dataset_is_person_shaped(profiles):
        return blocking

    # resolved_keys() dispatches on strategy instead of unconditionally
    # preferring `passes` -- the F2/1c843c8a5 shape (a schema-valid config
    # carrying both fields silently used the wrong one).
    passes = blocking.resolved_keys()
    if not passes:
        return blocking

    def _has_soundex(k: BlockingKeyConfig) -> bool:
        return any("soundex" in str(t) for t in (k.transforms or []))

    # Gap condition: names must already be a PRIMARY blocking signal via a
    # composite soundex pass. If names aren't soundex-blocked at all (e.g. a
    # non-person schema slipped past the gate), adding atomic name soundex would
    # invent a signal the config deliberately doesn't use -- skip.
    composite_name_soundex = any(
        _has_soundex(k)
        and any(_norm_colname(f) in _COMPOSITE_NAME_FIELDS for f in k.fields)
        for k in passes
    )
    if not composite_name_soundex:
        return blocking

    # Atomic single-name fields already carrying their own soundex pass -- never
    # duplicate.
    atomic_names = _ATOMIC_GIVEN_NAMES | _ATOMIC_FAMILY_NAMES
    already_atomic_soundex = {
        _norm_colname(k.fields[0])
        for k in passes
        if len(k.fields) == 1 and _has_soundex(k)
        and _norm_colname(k.fields[0]) in atomic_names
    }

    new_passes: list[BlockingKeyConfig] = []
    for p in profiles:
        if p.col_type not in _PERSON_NAME_COLTYPES:
            continue
        norm = _norm_colname(p.name)
        if norm not in atomic_names:
            continue  # skip composites + non-name columns
        if norm in already_atomic_soundex:
            continue
        # Guard against a mostly-empty name column producing a giant null block.
        if p.null_rate > _ORTHO_BLOCK_NULL_CEILING:
            continue
        new_passes.append(
            BlockingKeyConfig(
                fields=[p.name],
                transforms=["lowercase", "soundex"],
                additive=True,
            )
        )
        already_atomic_soundex.add(norm)

    if not new_passes:
        return blocking
    logger.info(
        "atomic-name-soundex blocking: added %d additive pass(es) on %s",
        len(new_passes), [k.fields[0] for k in new_passes],
    )
    updated = blocking.model_copy(deep=True)
    # Preserve the existing passes -- OR the `keys` when a static config carries
    # its blocking key there (passes=None). `passes` (above) already resolves
    # that fallback; seeding from `blocking.passes or []` alone would DROP the
    # original static key when promoting to multi_pass.
    updated.passes = list(passes) + new_passes
    if updated.strategy == "static":
        updated.strategy = "multi_pass"
    return updated


def _diversify_unused_orthogonal_blocking(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
    df: Any = None,
) -> BlockingConfig | None:
    """Add an additive blocking pass on each unused, well-populated,
    moderate-cardinality field -- a field-agnostic generalization of the
    orthogonal-anchor idea in ``_diversify_probabilistic_blocking``.

    ``_diversify_probabilistic_blocking`` only diversifies onto ``date`` and
    ``zip``/``identifier``/``phone`` col_types. But the strongest orthogonal anchor
    a dataset carries may be classified as something else -- on historical_50k the
    ``birth_place`` column (50.7% of the missed true pairs share it exactly) profiles
    as ``name`` (null 0.13, card 0.48) and slips past that whitelist, so the FS
    candidate set stays gated entirely on the (corrupted) name keys and
    blocking_recall caps at ~0.78. This rule instead selects by DATA SHAPE, not
    col_type: any field with ``null_rate <= _ORTHO_BLOCK_NULL_CEILING`` and
    cardinality in ``[_ORTHO_BLOCK_CARD_FLOOR, _ORTHO_BLOCK_CARD_CEILING)`` that no
    existing pass already keys on earns a single-field ``strip`` pass. Field-level
    (not signature-level) dedup so a field the name passes / v2 diversify already
    cover (dob-year, postcode-strip, the name composites) is never re-added.

    Purely additive (recall can only rise; scoring still decides precision). A cheap
    arrow-agnostic per-pass row guard drops a pass whose SAMPLE max block already
    exceeds the FS scorer row cap; the real full-N bound is
    ``_bound_probabilistic_blocking_pairs``, which runs after this over the whole
    pass list.

    **Gated ``GOLDENMATCH_FS_ORTHOGONAL_BLOCKING=auto`` (default).** ``auto`` fires
    ONLY on person-shaped data (``_dataset_is_person_shaped``: a ``name`` + a
    ``date`` col_type) -- the regime where the lever generalises out-of-panel
    (historical_50k B3 0.808->0.873, febrl4 holdout +0.006) -- and is a NO-OP
    (byte-identical) on bibliographic/product data, where a topic-bucket anchor
    (venue/year) drives a scoring over-merge (dblp_scholar -0.24). ``=1`` forces it
    everywhere, ``=0`` disables it. See ``_fs_orthogonal_blocking_mode``.
    """
    if blocking is None:
        return blocking
    _mode = _fs_orthogonal_blocking_mode()
    if _mode == "off":
        return blocking
    if _mode == "auto" and not _dataset_is_person_shaped(profiles):
        return blocking

    # Fields ALREADY keyed by some pass (any transform) -- don't re-add a field the
    # name passes / v2 diversify already cover. Field-level so we never duplicate.
    covered: set[str] = set()
    for k in list(blocking.keys or []) + list(blocking.passes or []):
        covered.update(k.fields)

    # Cheap arrow-agnostic per-pass sanity guard (sample scale). The real full-N
    # bound is `_bound_probabilistic_blocking_pairs`, which runs after.
    bframe = None
    sample_n = 1
    row_cap = 7071
    _col_cache: dict[str, list] = {}
    _tx_cache: dict[tuple, list] = {}
    # Per-sample-row memberships across the EXISTING passes, for the co-blocking
    # ORTHOGONALITY gate below (an anchor already covered by the current passes
    # over-merges rather than diversifies -- see _ORTHO_BLOCK_OVERLAP_CEILING).
    overlap_memb: list[set] | None = None
    overlap_cap = 0
    if df is not None:
        try:
            from goldenmatch.core.frame import to_frame as _tf
            from goldenmatch.core.probabilistic import _fs_vec_max_elems

            bframe = _tf(df)
            sample_n = max(int(bframe.height), 1)
            cap = _fs_vec_max_elems()
            if cap > 0:
                row_cap = int(cap**0.5)
            overlap_cap = min(sample_n, _ORTHO_OVERLAP_SAMPLE)
            # resolved_keys() dispatches on strategy instead of unconditionally
            # preferring `passes` -- see the resolved_keys() call above in this
            # module for the F2/1c843c8a5 shape this avoids.
            base_specs = [_pass_specs(k) for k in blocking.resolved_keys()]
            if base_specs:
                overlap_memb = _coblock_membership(
                    bframe, base_specs, overlap_cap, _col_cache, _tx_cache
                )
        except Exception:
            bframe = None

    new_passes: list[BlockingKeyConfig] = []
    for p in profiles:
        if p.name in covered:
            continue
        if p.null_rate > _ORTHO_BLOCK_NULL_CEILING:
            continue
        if not (
            _ORTHO_BLOCK_CARD_FLOOR <= p.cardinality_ratio < _ORTHO_BLOCK_CARD_CEILING
        ):
            continue
        transforms = ["strip"]
        cand_spec = [(p.name, tuple(transforms))]
        if bframe is not None:
            proj = _project_pass_pairs(
                bframe, cand_spec, sample_n, sample_n, _col_cache, _tx_cache
            )
            if proj is not None and proj[0] > row_cap:
                logger.debug(
                    "orthogonal blocking: dropping oversized pass %s (sample max "
                    "block %d rows > FS row cap %d)",
                    p.name, proj[0], row_cap,
                )
                continue
            # Orthogonality gate: skip a candidate whose same-key pairs are mostly
            # ALREADY co-blocked by an existing pass (an atomic name field when the
            # name composites already block). Such a "diversify" only re-blocks the
            # primary signal, widening the same-signal candidate set the FS scorer
            # can't reject -> over-merge (historical_50k precision 0.91->0.75). A
            # genuinely orthogonal anchor (birth_place, overlap ~0.29) is admitted.
            if overlap_memb is not None:
                cand_keys = _pass_row_keys(
                    bframe, cand_spec, overlap_cap, _col_cache, _tx_cache
                )
                if cand_keys is not None:
                    ov = _anchor_coblock_overlap(cand_keys, overlap_memb)
                    if ov >= _ORTHO_BLOCK_OVERLAP_CEILING:
                        logger.debug(
                            "orthogonal blocking: dropping redundant pass %s "
                            "(co-block overlap %.3f >= %.2f -- already covered by an "
                            "existing pass)",
                            p.name, ov, _ORTHO_BLOCK_OVERLAP_CEILING,
                        )
                        continue
        # additive=True: co-locate the missed pairs WITHOUT demoting the field from
        # EM scoring. An orthogonal anchor (e.g. birth_place) is typically ALSO a
        # useful FS discriminator; demoting it to a fixed neutral weight the moment
        # it becomes a blocking key measurably costs F1 (birth_place kept-in-EM beat
        # demoted on historical_50k), and demoting a strong name discriminator
        # collapses recall outright. See BlockingKeyConfig.additive.
        new_passes.append(
            BlockingKeyConfig(fields=[p.name], transforms=transforms, additive=True)
        )
        covered.add(p.name)

    if not new_passes:
        return blocking

    # resolved_keys() dispatches on strategy instead of unconditionally
    # preferring `passes` -- see the resolved_keys() call above in this
    # module for the F2/1c843c8a5 shape this avoids.
    base_passes = blocking.resolved_keys()
    return blocking.model_copy(update={
        "strategy": "multi_pass",
        "passes": base_passes + new_passes,
        "auto_select": False,
    })


def _pass_specs(key: BlockingKeyConfig) -> list[tuple[str, tuple[str, ...]]]:
    """``[(field, transforms), ...]`` for a blocking key, honoring per-field
    ``field_transforms`` (a field present there uses ITS chain; absent fields use
    the key-level ``transforms``) — mirrors how the blocker derives the block key.
    """
    ft = getattr(key, "field_transforms", None) or {}
    shared = tuple(getattr(key, "transforms", None) or [])
    return [(f, tuple(ft[f]) if f in ft else shared) for f in key.fields]


def _pass_row_keys(
    bframe: Any,
    specs: list[tuple[str, tuple[str, ...]]],
    cap: int,
    _col_cache: dict[str, list] | None = None,
    _tx_cache: dict[tuple, list] | None = None,
) -> list[str | None] | None:
    """Per-row block key for a pass over the first ``cap`` rows (``None`` where any
    component transforms to null/empty — the blocker drops those rows). Mirrors
    ``_project_pass_pairs``' key derivation (shared ``_col_cache``/``_tx_cache``)
    but returns the per-row keys instead of block-size counts. Polars/numpy-free.
    Returns ``None`` if a column is unreadable."""
    from goldenmatch.utils.transforms import apply_transforms as _apply

    mapped_cols: list[list] = []
    for fld, transforms in specs:
        if _col_cache is not None and fld in _col_cache:
            raw = _col_cache[fld]
        else:
            try:
                raw = bframe.column(fld).to_list()
            except Exception:  # pragma: no cover -- missing column, skip pass
                return None
            if _col_cache is not None:
                _col_cache[fld] = raw
        tkey = (fld, transforms)
        if _tx_cache is not None and tkey in _tx_cache:
            mapped = _tx_cache[tkey]
        elif transforms:
            tmap = {
                v: (_apply(str(v), list(transforms)) if v is not None else None)
                for v in set(raw)
            }
            mapped = [tmap[v] for v in raw]
            if _tx_cache is not None:
                _tx_cache[tkey] = mapped
        else:
            mapped = raw
            if _tx_cache is not None:
                _tx_cache[tkey] = mapped
        mapped_cols.append(mapped)

    keys: list[str | None] = []
    if len(mapped_cols) == 1:
        for v in mapped_cols[0][:cap]:
            keys.append(None if (v is None or v == "") else str(v))
    else:
        for row in zip(*[m[:cap] for m in mapped_cols]):
            keys.append(
                "\x1f".join(str(v) for v in row)
                if all(v is not None and v != "" for v in row)
                else None
            )
    return keys


def _coblock_membership(
    bframe: Any,
    base_specs: list[list[tuple[str, tuple[str, ...]]]],
    cap: int,
    _col_cache: dict[str, list] | None = None,
    _tx_cache: dict[tuple, list] | None = None,
) -> list[set] | None:
    """Per-(sample-)row set of ``(pass_index, block_key)`` memberships across the
    EXISTING passes — the reference an orthogonal-anchor candidate is judged
    against. ``None`` when no pass is measurable."""
    memb: list[set] = [set() for _ in range(cap)]
    any_pass = False
    for pidx, specs in enumerate(base_specs):
        keys = _pass_row_keys(bframe, specs, cap, _col_cache, _tx_cache)
        if keys is None:
            continue
        any_pass = True
        for i, k in enumerate(keys):
            if k is not None:
                memb[i].add((pidx, k))
    return memb if any_pass else None


def _anchor_coblock_overlap(
    cand_keys: list[str | None],
    memb: list[set],
    pair_cap: int = _ORTHO_OVERLAP_PAIR_CAP,
) -> float:
    """Fraction of the candidate anchor's same-key row pairs that are ALREADY
    co-blocked by some existing pass (``memb``). High => redundant same-signal
    anchor (an atomic name field vs the name composites) whose extra candidates
    the FS scorer can't reject; low => a genuinely orthogonal anchor. Evaluated
    over the sample rows, bounded by ``pair_cap`` (deterministic, row order)."""
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(cand_keys):
        if k is not None:
            groups[k].append(i)
    total = 0
    redundant = 0
    for members in groups.values():
        g = len(members)
        if g < 2:
            continue
        stop = False
        for a in range(g):
            if total >= pair_cap:
                stop = True
                break
            sa = memb[members[a]]
            for b in range(a + 1, g):
                if total >= pair_cap:
                    stop = True
                    break
                total += 1
                if sa and (sa & memb[members[b]]):
                    redundant += 1
        if stop:
            break
    return (redundant / total) if total else 0.0


def _project_pass_pairs(
    bframe: Any,
    specs: list[tuple[str, tuple[str, ...]]],
    effective_n_full: int,
    sample_n: int,
    _col_cache: dict[str, list] | None = None,
    _tx_cache: dict[tuple, list] | None = None,
) -> tuple[int, int] | None:
    """``(max_block_rows, candidate_pairs)`` a pass emits at full N.

    ``candidate_pairs = Σ C(block, 2)`` over every non-empty block key — the axis
    FS scoring memory actually scales on (a 15k-row birth-YEAR block is 110M
    pairs), which the row-only ``_compute_max_safe_block`` ceiling misses. A row
    is absent from the pass if ANY component transforms to null/empty (the blocker
    filters null/sentinel keys). ``effective_n_full/sample_n`` scales each block
    linearly (bounded keys grow ∝ N), matching ``_projected_block``. None if a
    column is unmeasurable.

    **Polars/numpy-free by construction** (D6 zero-polars invariant): reads via
    the arrow-agnostic ``Frame`` abstraction and counts with ``Counter``.
    Transforms are applied over DISTINCT values (else soundex over 1M rows is 1M
    calls) and cached across passes. Runs once at config time, not on a hot path.
    """
    from collections import Counter

    from goldenmatch.utils.transforms import apply_transforms as _apply

    mapped_cols: list[list] = []
    for fld, transforms in specs:
        if _col_cache is not None and fld in _col_cache:
            raw = _col_cache[fld]
        else:
            try:
                raw = bframe.column(fld).to_list()
            except Exception:  # pragma: no cover -- missing column, skip pass
                return None
            if _col_cache is not None:
                _col_cache[fld] = raw
        tkey = (fld, transforms)
        if _tx_cache is not None and tkey in _tx_cache:
            mapped = _tx_cache[tkey]
        elif transforms:
            # The blocker casts to Utf8 before applying transforms (block-key
            # derivation); mirror that so a substring reducer on an int column
            # (postcode/dob) doesn't crash.
            tmap = {
                v: (_apply(str(v), list(transforms)) if v is not None else None)
                for v in set(raw)
            }
            mapped = [tmap[v] for v in raw]
            if _tx_cache is not None:
                _tx_cache[tkey] = mapped
        else:
            mapped = raw
            if _tx_cache is not None:
                _tx_cache[tkey] = mapped
        mapped_cols.append(mapped)

    if len(mapped_cols) == 1:
        counts = Counter(
            v for v in mapped_cols[0] if v is not None and v != ""
        )
    else:
        counts = Counter(
            "\x1f".join(str(v) for v in row)
            for row in zip(*mapped_cols)
            if all(v is not None and v != "" for v in row)
        )
    if not counts:
        return (0, 0)
    # Saturation-aware projection. Extracted to `core.block_projection` so
    # learned blocking's rule selector gates on the SAME projection this static
    # pass gate uses instead of growing a second, divergent one; that module's
    # docstring carries the derivation and the 30M phantom-pair incident.
    from goldenmatch.core.block_projection import project_block_counts

    return project_block_counts(counts.values(), sample_n, effective_n_full)


_FS_TOTAL_PAIR_BUDGET = 300_000_000  # small-box FLOOR (#1803 tuning anchor)
# Candidate pairs affordable per GB of *available* RAM. Anchored to the measured
# 25M-on-64GB single-box FS proof: ~2.1B bounded candidate pairs peaked at ~28 GB
# (with ~55 GB available at start), i.e. ~40M pairs/GB is the proven-safe operating
# point (the (id,id,score) stream + clustering edge set + frames all fit).
_FS_PAIR_BUDGET_PER_GB = 40_000_000


def _fs_total_pair_budget(
    n_rows: int, available_ram_gb: float | None = None
) -> int:
    """GLOBAL candidate-pair budget across ALL blocking passes (not per-pass).

    FS scoring memory scales with the TOTAL candidate pairs held across passes
    (the scored-pair stream + the clustering edge set), and candidate pairs grow
    ~N² while a per-pass *linear* floor cannot both keep a pass memory-safe at
    100K and bound it at 1M (a 55M-pair surname pass is fine at 100K but the same
    shape is 5.5B at 1M). Individual mega-BLOCK passes are still gated per-pass by
    ``_compute_max_safe_block`` (the per-block score-matrix cost).

    **The budget is MEMORY-AWARE, not flat (2026-07-21).** The physical limit is
    memory — but memory is a property of the BOX, not a universal constant. A flat
    300M budget is calibrated for a ~4-8 GB machine; on the 64 GB runner the 25M
    single-box FS envelope actually targets, it is ~16x too tight, and the
    over-tight budget forces ``_bound_probabilistic_blocking_pairs`` to COMPOUND
    recall-critical coarse passes (e.g. a pure ``zip`` pass that duplicates share
    exactly) with corruption-prone fields — collapsing blocking recall at scale
    (measured 1.0 -> 0.82 at 4.8M -> ~0.02 at 30M). So the budget scales with
    ``available_ram_gb`` (``~40M pairs/GB``, the proven-safe 25M-proof point),
    floored at ``_FS_TOTAL_PAIR_BUDGET`` so small boxes keep today's behavior +
    all the #1803 tuning. At >=1B (any >=~32 GB box) the pure coarse passes
    survive and blocking recall is 1.0 at 4.8M/25M; an undersized box degrades
    honestly (less recall, no OOM) rather than silently at scale.

    ``available_ram_gb`` is injectable for deterministic tests; when ``None`` it is
    read from ``runtime_profile.capture_runtime_profile`` (psutil), falling back to
    the flat floor if introspection is unavailable. Override the whole budget with
    ``GOLDENMATCH_FS_MAX_PASS_PAIRS``. ``n_rows`` is accepted for signature
    stability / future tuning.
    """
    override = os.environ.get("GOLDENMATCH_FS_MAX_PASS_PAIRS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    floor = _FS_TOTAL_PAIR_BUDGET
    if available_ram_gb is None:
        try:
            from goldenmatch.core.runtime_profile import capture_runtime_profile

            available_ram_gb = capture_runtime_profile().available_ram_gb
        except Exception:
            return floor
    return max(floor, int(available_ram_gb * _FS_PAIR_BUDGET_PER_GB))


def _bound_probabilistic_blocking_pairs(
    blocking: BlockingConfig | None,
    profiles: list[ColumnProfile],
    df: Any = None,
    *,
    n_rows_full: int | None = None,
) -> BlockingConfig | None:
    """Bound blocking by per-block ROWS **and** a GLOBAL candidate-PAIR budget.

    ``_diversify`` and ``_compute_max_safe_block`` gate on block ROW count, but FS
    scoring memory also scales with the TOTAL candidate PAIRS across passes
    (``Σ C(block, 2)``): a 15k-row birth-YEAR block clears a 25k-row ceiling yet
    is 110M pairs, and ``build_blocking``'s single-field ``soundex`` passes are
    never row-gated at all. On the 1M person auto-config these summed to ~12.9B
    candidate pairs and OOM-killed the runner regardless of scoring route.

    Two gates, in order (runs after ``_diversify``, over the WHOLE pass list):
      1. **Per-pass max-block** — any pass whose max block exceeds
         ``_compute_max_safe_block`` (the per-block score-matrix cost, a real
         resident bomb at every scale) is bounded/dropped immediately.
      2. **Global total-pairs** — if the surviving passes' pairs SUM over
         ``_fs_total_pair_budget`` (a flat memory-derived total, NOT per-pass and
         NOT N-scaled), the largest-pair pass is bounded (append the most
         selective reducer initial — surname, first name, city, then zip/id
         prefix) or DROPPED, repeatedly, until the total is under budget.

    A flat total is why this no longer over-bounds at 50K–100K (a 55M-pair
    surname pass is memory-safe there and now KEPT — restoring recall the old
    per-pass 5M floor cost) while still crushing the 12.9B-pair 1M case. Always
    leaves ≥1 pass.

    Default ON with ``_fs_autoconfig_v2_enabled``; needs ``df`` to measure (skips
    otherwise, same contract as ``_diversify``). ``GOLDENMATCH_FS_MAX_PASS_PAIRS``
    overrides the total; ``GOLDENMATCH_FS_AUTOCONFIG_V2=0`` disables it.
    """
    if blocking is None or not _fs_autoconfig_v2_enabled() or df is None:
        return blocking

    from goldenmatch.core._native_loader import native_enabled as _native_enabled
    from goldenmatch.core.frame import to_frame as _tf

    bframe = _tf(df)
    sample_n = max(int(bframe.height), 1)
    effective_n_full = (
        int(n_rows_full) if (n_rows_full and int(n_rows_full) > sample_n) else sample_n
    )
    max_block = _compute_max_safe_block(
        effective_n_full, _native_enabled("block_scoring")
    )
    total_budget = _fs_total_pair_budget(effective_n_full)
    if total_budget > _FS_TOTAL_PAIR_BUDGET:
        # Memory-aware lift above the small-box floor — surfaces the effective
        # ceiling so a scale run shows why coarse (recall-critical) passes were
        # or were NOT bounded. See _fs_total_pair_budget.
        logger.info(
            "FS pair-gate: candidate-pair budget %d (memory-aware; floor %d) at "
            "n=%d.", total_budget, _FS_TOTAL_PAIR_BUDGET, effective_n_full,
        )

    # resolved_keys() dispatches on strategy instead of unconditionally
    # preferring `passes` -- the F2/1c843c8a5 shape (a schema-valid config
    # carrying both fields silently used the wrong one).
    passes = blocking.resolved_keys()
    if not passes:
        return blocking

    # Reducer discriminators to AND into an over-budget coarse pass. Ordered by
    # RECALL-SAFETY first, selectivity second:
    #
    #   1. Exact-agreement identity fields (email / identifier / phone) at FULL
    #      value. Duplicates share these EXACTLY, so ANDing one into a coarse pass
    #      keeps every true pair together while collapsing the block to near-
    #      singletons -- recall-safe AND the strongest pair reducer (measured on
    #      the 30M person shape: `zip` alone = 116M pairs @ recall 1.0; `zip +
    #      first-initial` = 6.8M pairs but recall 0.82; `zip + email` = 0.9M pairs
    #      @ recall 1.0). This is THE fix for the 30M recall collapse: at scale the
    #      pair budget forces compounding on EVERY pass, and a corruption-prone
    #      name-initial reducer then SPLITS true pairs on any typo'd name.
    #   2/3. Selective name/geo/city initials + zip/id/phone prefixes -- the
    #      fallback when no identity field is present or clears the budget. These
    #      are corruption-prone (a name typo breaks the compound), so they run
    #      only after the identity fields.
    by_type: dict[str, list[str]] = {}
    for p in profiles:
        by_type.setdefault(p.col_type, []).append(p.name)
    # Perfect-surrogate exclusion (mirrors the #721/#876 exact-matchkey gate).
    # A column with cardinality_ratio >= 1.0 is a unique surrogate key (record_id,
    # row-uuid): duplicates do NOT share it, so ANDing it into a coarse pass as a
    # full-value identity reducer SPLITS every true pair -- it looks like the
    # strongest reducer (collapses blocks to singletons) but neuters recall. This
    # is exactly the "identifier" full-value branch below, where a unique record
    # key would otherwise be chosen first. Measured on the 25M person shape
    # (record_id unique, truth keyed on cluster_id): compounding with record_id
    # dropped recall 1.0 -> 0.49 at scale. Skip them everywhere a reducer is built.
    _surrogate = {p.name for p in profiles if _is_perfect_surrogate(p)}
    reducers: list[tuple[str, tuple[str, ...]]] = []
    for f in (
        by_type.get("email", [])
        + by_type.get("identifier", [])
        + by_type.get("phone", [])
    ):
        if f in _surrogate:
            continue
        reducers.append((f, ()))  # full value -- exact-shared identity
    for f in by_type.get("name", []):
        if f in _surrogate:
            continue
        reducers.append((f, ("substring:0:1",)))
    for f in by_type.get("geo", []) + by_type.get("city", []):
        if f in _surrogate:
            continue
        reducers.append((f, ("substring:0:1",)))
    for f in (
        by_type.get("zip", []) + by_type.get("identifier", []) + by_type.get("phone", [])
    ):
        if f in _surrogate:
            continue
        reducers.append((f, ("substring:0:2",)))

    _col_cache: dict[str, list] = {}
    _tx_cache: dict[tuple, list] = {}

    def _proj(specs: list[tuple[str, tuple[str, ...]]]) -> tuple[int, int] | None:
        return _project_pass_pairs(
            bframe, specs, effective_n_full, sample_n, _col_cache, _tx_cache
        )

    def _bound_one(
        key: BlockingKeyConfig,
        specs: list[tuple[str, tuple[str, ...]]],
    ) -> tuple[BlockingKeyConfig, tuple[int, int]] | None:
        """Append the first reducer that brings the pass under max-block AND the
        total budget; None if no reducer helps (caller then drops the pass)."""
        pass_fields = set(key.fields)
        for rf, rt in reducers:
            if rf in pass_fields:
                continue
            cspecs = specs + [(rf, rt)]
            cproj = _proj(cspecs)
            if cproj is not None and cproj[0] <= max_block and cproj[1] <= total_budget:
                ft = {f: list(t) for f, t in cspecs}
                return (
                    BlockingKeyConfig(
                        fields=[f for f, _ in cspecs], field_transforms=ft,
                        # Preserve the additive (orthogonal-anchor, EM-trained)
                        # intent across a scale bound -- else a compounded anchor
                        # would silently revert to a demoted primary key.
                        additive=getattr(key, "additive", False),
                    ),
                    cproj,
                )
        return None

    # `entries` carries (key, specs, proj) for every pass (proj None =
    # unmeasurable, treated as selective / not counted). NB: there is NO
    # standalone per-block gate here — a block over ``max_block`` is only a
    # memory bomb because it is also over-budget on pairs (a b-row single block
    # is C(b,2) pairs; at 1M a >25k-row block is >300M pairs), so the total-pairs
    # gate below catches every real case, and its ``_bound_one`` reducer already
    # requires the bounded result to respect ``max_block``. A standalone
    # max-block gate would ALSO bound small memory-trivial blocks (a 3k-row block
    # at 50k is a 13 MB matrix) and needlessly cost recall — measured
    # historical_50k f1_probabilistic 0.826 -> 0.794 before this was dropped.
    entries: list[tuple[BlockingKeyConfig, list, tuple[int, int] | None]] = [
        (key, _pass_specs(key), _proj(_pass_specs(key))) for key in passes
    ]
    changed = False

    # Global total-pairs budget. Bound the largest-pair pass until the
    # sum is under budget. Bounding (append a reducer -> a compound) is always
    # allowed since it PRESERVES the pass; only DROPPING (no reducer helps) is
    # blocked once a single pass remains, so blocking is never stripped to
    # nothing. Each step strictly shrinks or removes a pass, so this terminates.
    def _pairs(e: tuple[BlockingKeyConfig, list, tuple[int, int] | None]) -> int:
        return e[2][1] if e[2] is not None else 0

    while sum(_pairs(e) for e in entries) > total_budget:
        boundable = [i for i in range(len(entries)) if _pairs(entries[i]) > 0]
        if not boundable:
            break  # remaining passes are all selective; nothing worth bounding
        idx = max(boundable, key=lambda i: _pairs(entries[i]))
        key, specs, proj = entries[idx]
        bounded = _bound_one(key, specs)
        if bounded is not None:
            bkey, bproj = bounded
            entries[idx] = (bkey, _pass_specs(bkey), bproj)
            changed = True
            logger.info(
                "FS pair-gate: total over budget; pass %s (%d pairs) bounded to "
                "%s (%d pairs) at n=%d. See #1803.",
                key.fields, proj[1] if proj else -1, bkey.fields, bproj[1],
                effective_n_full,
            )
        elif len(entries) > 1:
            entries.pop(idx)
            changed = True
            logger.info(
                "FS pair-gate: total over budget; dropping pass %s (%d pairs) — no "
                "reducer bounds it. Recall falls back to the selective passes at "
                "n=%d. See #1803.",
                key.fields, proj[1] if proj else -1, effective_n_full,
            )
        else:
            break  # last pass, can't bound -> keep it rather than strip to none

    if not changed:
        return blocking
    kept = [e[0] for e in entries]
    if not kept:
        # Never strip blocking to nothing: retain the single most selective pass.
        best = min(passes, key=lambda k: (_proj(_pass_specs(k)) or (0, 1 << 62))[1])
        kept = [best]
        logger.warning(
            "FS pair-gate: every pass exceeded budget; retaining the most "
            "selective pass %s to preserve recall.", best.fields,
        )
    return blocking.model_copy(update={
        "strategy": "multi_pass",
        "passes": kept,
        "auto_select": False,
    })
