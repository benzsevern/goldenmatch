"""Pydantic models for GoldenMatch configuration validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Valid enums ─────────────────────────────────────────────────────────────

VALID_SIMPLE_TRANSFORMS = frozenset({
    "lowercase", "uppercase", "strip", "strip_all", "soundex", "metaphone",
    "digits_only", "alpha_only", "normalize_whitespace",
    "token_sort", "first_token", "last_token",
})

VALID_SCORERS = frozenset({
    "exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match",
    "embedding", "record_embedding", "ensemble",
    "dice", "jaccard",
})

VALID_STRATEGIES = frozenset({
    "most_recent", "source_priority", "most_complete", "majority_vote", "first_non_null",
})

_SUBSTRING_RE = re.compile(r"^substring:\d+:\d+$")
_QGRAM_RE = re.compile(r"^qgram:\d+$")
_BLOOM_FILTER_RE = re.compile(r"^bloom_filter:\d+:\d+:\d+$")


# ── FieldTransform ──────────────────────────────────────────────────────────


class FieldTransform(BaseModel):
    transform: str

    @model_validator(mode="after")
    def _validate_transform(self) -> "FieldTransform":
        t = self.transform
        if t in VALID_SIMPLE_TRANSFORMS:
            return self
        if _SUBSTRING_RE.match(t):
            return self
        if _QGRAM_RE.match(t):
            return self
        if t == "bloom_filter" or _BLOOM_FILTER_RE.match(t):
            return self
        raise ValueError(
            f"Invalid transform '{t}'. Must be one of {sorted(VALID_SIMPLE_TRANSFORMS)} "
            f"or 'substring:<start>:<end>'."
        )


# ── MatchkeyField ──────────────────────────────────────────────────────────


class MatchkeyField(BaseModel):
    field: str | None = None
    column: str | None = None
    transforms: list[str] = Field(default_factory=list)
    scorer: str | None = None
    weight: float | None = None
    model: str | None = None  # for embedding scorer
    columns: list[str] | None = None  # for record_embedding scorer
    column_weights: dict[str, float] | None = None  # per-field weights for record_embedding
    levels: int = 2  # comparison levels for probabilistic: 2=agree/disagree, 3=agree/partial/disagree
    partial_threshold: float = 0.8  # score >= this = partial agree (when levels=3)

    @model_validator(mode="after")
    def _resolve_field_column(self) -> "MatchkeyField":
        # record_embedding uses columns (plural), not field
        if self.scorer == "record_embedding":
            if not self.columns:
                raise ValueError(
                    "record_embedding scorer requires 'columns' (list of column names)."
                )
            self.field = "__record__"
            return self
        # Allow 'column' as alias for 'field'
        if self.field is None and self.column is not None:
            self.field = self.column
        elif self.field is None and self.column is None:
            raise ValueError("MatchkeyField requires 'field' or 'column'.")
        for t in self.transforms:
            FieldTransform(transform=t)  # reuse validation
        if self.scorer is not None and self.scorer not in VALID_SCORERS:
            # Check plugin registry before rejecting
            from goldenmatch.plugins.registry import PluginRegistry
            registry = PluginRegistry.instance()
            if not registry.has_scorer(self.scorer):
                raise ValueError(
                    f"Invalid scorer '{self.scorer}'. Must be one of {sorted(VALID_SCORERS)} "
                    f"or a registered plugin scorer."
                )
        return self


# ── MatchkeyConfig ──────────────────────────────────────────────────────────


_VALID_MK_TYPES = ("exact", "weighted", "probabilistic")


class MatchkeyConfig(BaseModel):
    name: str
    type: Literal["exact", "weighted", "probabilistic"] | None = None
    comparison: str | None = None
    fields: list[MatchkeyField]
    threshold: float | None = None
    auto_threshold: bool = False
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_band: float = 0.1
    # Fellegi-Sunter EM parameters
    em_iterations: int = 20
    convergence_threshold: float = 0.001
    link_threshold: float | None = None  # auto-computed if None
    review_threshold: float | None = None  # auto-computed if None

    @model_validator(mode="after")
    def _validate_weighted(self) -> "MatchkeyConfig":
        # Allow 'comparison' as alias for 'type'
        if self.type is None and self.comparison is not None:
            if self.comparison in _VALID_MK_TYPES:
                self.type = self.comparison
            else:
                raise ValueError(
                    f"Invalid comparison '{self.comparison}'. Must be one of {_VALID_MK_TYPES}."
                )
        elif self.type is None:
            raise ValueError("MatchkeyConfig requires 'type' or 'comparison'.")
        if self.type == "weighted":
            if self.threshold is None:
                raise ValueError("Weighted matchkeys require a 'threshold'.")
            for f in self.fields:
                if f.scorer is None or f.weight is None:
                    raise ValueError(
                        f"All fields in a weighted matchkey must have 'scorer' and 'weight'. "
                        f"Field '{f.field}' is missing one or both."
                    )
        elif self.type == "probabilistic":
            for f in self.fields:
                if f.scorer is None:
                    raise ValueError(
                        f"All fields in a probabilistic matchkey must have 'scorer'. "
                        f"Field '{f.field}' is missing it."
                    )
        return self


# ── BlockingKeyConfig / BlockingConfig ──────────────────────────────────────


class BlockingKeyConfig(BaseModel):
    fields: list[str]
    transforms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_fields_nonempty(self) -> "BlockingKeyConfig":
        if not self.fields:
            raise ValueError("Blocking key must have at least one field.")
        return self


class SortKeyField(BaseModel):
    column: str
    transforms: list[str] = Field(default_factory=list)


class CanopyConfig(BaseModel):
    fields: list[str]
    loose_threshold: float = 0.3
    tight_threshold: float = 0.7
    max_canopy_size: int = 500


class BlockingConfig(BaseModel):
    keys: list[BlockingKeyConfig] = []
    max_block_size: int = 5000
    skip_oversized: bool = False
    strategy: Literal["static", "adaptive", "sorted_neighborhood", "multi_pass", "ann", "canopy", "ann_pairs", "learned"] = "static"
    learned_sample_size: int = 5000
    learned_min_recall: float = 0.95
    learned_min_reduction: float = 0.90
    learned_predicate_depth: int = 2
    learned_cache_path: str | None = None  # persist for reuse
    auto_suggest: bool = False
    auto_select: bool = False
    sub_block_keys: list[BlockingKeyConfig] | None = None
    window_size: int = 20
    sort_key: list[SortKeyField] | None = None
    passes: list[BlockingKeyConfig] | None = None
    union_mode: bool = True
    max_total_comparisons: int | None = None
    ann_column: str | None = None
    ann_model: str = "all-MiniLM-L6-v2"
    ann_top_k: int = 20
    canopy: CanopyConfig | None = None

    @model_validator(mode="after")
    def _validate_keys_or_passes(self) -> "BlockingConfig":
        """Ensure at least keys or passes is provided for strategies that need them."""
        if self.auto_suggest:
            return self  # auto_suggest discovers keys at runtime
        needs_keys = self.strategy in ("static", "adaptive")
        needs_passes = self.strategy == "multi_pass"
        if needs_keys and not self.keys and not self.sub_block_keys:
            raise ValueError(
                "BlockingConfig with strategy='static' or 'adaptive' requires 'keys'."
            )
        if needs_passes and not self.keys and not self.passes:
            raise ValueError(
                "BlockingConfig with strategy='multi_pass' requires 'keys' or 'passes'."
            )
        return self


# ── GoldenFieldRule / GoldenRulesConfig ─────────────────────────────────────


class GoldenFieldRule(BaseModel):
    strategy: str
    date_column: str | None = None
    source_priority: list[str] | None = None

    @model_validator(mode="after")
    def _validate_strategy(self) -> "GoldenFieldRule":
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Must be one of {sorted(VALID_STRATEGIES)}."
            )
        if self.strategy == "most_recent" and not self.date_column:
            raise ValueError("Strategy 'most_recent' requires 'date_column'.")
        if self.strategy == "source_priority" and not self.source_priority:
            raise ValueError("Strategy 'source_priority' requires 'source_priority' list.")
        return self


class GoldenRulesConfig(BaseModel):
    default_strategy: str | None = None
    default: GoldenFieldRule | None = None
    field_rules: dict[str, GoldenFieldRule] = Field(default_factory=dict)
    max_cluster_size: int = 100

    @model_validator(mode="after")
    def _validate_default(self) -> "GoldenRulesConfig":
        # Resolve default_strategy from either field
        if self.default is not None and self.default_strategy is None:
            self.default_strategy = self.default.strategy
        if self.default_strategy is None:
            raise ValueError("GoldenRulesConfig requires 'default_strategy' or 'default'.")
        if self.default_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid default_strategy '{self.default_strategy}'."
            )
        return self


# ── StandardizationConfig ──────────────────────────────────────────────────

VALID_STANDARDIZERS = frozenset({
    "email", "name_proper", "name_upper", "name_lower",
    "phone", "zip5", "address", "state", "strip", "trim_whitespace",
})


class ValidationRuleConfig(BaseModel):
    column: str
    rule_type: Literal["regex", "min_length", "max_length", "not_null", "in_set", "format"]
    params: dict = Field(default_factory=dict)
    action: Literal["null", "quarantine", "flag"] = "flag"


class ValidationConfig(BaseModel):
    rules: list[ValidationRuleConfig] = Field(default_factory=list)
    auto_fix: bool = True  # whether to run auto-fix before validation


class QualityConfig(BaseModel):
    """GoldenCheck integration config for enhanced data quality."""
    enabled: bool = True       # auto-detected: True if goldencheck installed
    mode: str = "announced"    # "silent" | "announced" | "disabled"
    fix_mode: str = "safe"     # "safe" | "moderate" | "none"
    domain: str | None = None  # "healthcare" | "finance" | "ecommerce"


class StandardizationConfig(BaseModel):
    rules: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_standardizers(self) -> "StandardizationConfig":
        for column, std_names in self.rules.items():
            for name in std_names:
                if name not in VALID_STANDARDIZERS:
                    raise ValueError(
                        f"Invalid standardizer '{name}' for column '{column}'. "
                        f"Valid: {sorted(VALID_STANDARDIZERS)}"
                    )
        return self


# ── InputFileConfig / InputConfig ───────────────────────────────────────────


class InputFileConfig(BaseModel):
    path: str
    id_column: str | None = None
    source_label: str | None = None
    source_name: str | None = None
    column_map: dict[str, str] | None = None
    delimiter: str = ","
    encoding: str = "utf8"
    sheet: str | None = None
    parse_mode: str = "auto"  # auto, delimited, fixed_width, key_value, block, entity_extract
    header_row: int | None = None
    has_header: bool | None = None
    skip_rows: list[int] | None = None


class InputConfig(BaseModel):
    files: list[InputFileConfig] = Field(default_factory=list)
    file_a: InputFileConfig | None = None
    file_b: InputFileConfig | None = None


# ── OutputConfig ────────────────────────────────────────────────────────────


class OutputConfig(BaseModel):
    path: str | None = None
    format: str | None = None
    directory: str | None = None
    run_name: str | None = None


# ── LLM Budget / Scorer Config ────────────────────────────────────────────


class BudgetConfig(BaseModel):
    max_cost_usd: float | None = None
    max_calls: int | None = None
    escalation_model: str | None = None
    escalation_band: list[float] = Field(default_factory=lambda: [0.80, 0.90])
    escalation_budget_pct: float = 20
    warn_at_pct: float = 80


class LLMScorerConfig(BaseModel):
    enabled: bool = False
    provider: str | None = None  # "openai" or "anthropic", auto-detected if None
    model: str | None = None  # e.g. "gpt-4o-mini", auto-detected if None
    auto_threshold: float = 0.95  # auto-accept pairs above this
    candidate_lo: float = 0.75  # lower bound of LLM scoring range
    candidate_hi: float = 0.95  # upper bound (same as auto_threshold)
    batch_size: int = 20
    budget: BudgetConfig | None = None
    mode: str = "pairwise"  # "pairwise" (legacy) or "cluster" (in-context LLM clustering)
    cluster_max_size: int = 100  # max records per LLM cluster block
    cluster_min_size: int = 5  # below this, fall back to pairwise


# ── Domain Extraction Config ──────────────────────────────────────────────


class DomainConfig(BaseModel):
    enabled: bool = False
    mode: str | None = None  # "product", "person", "bibliographic", "company", "auto"
    confidence_threshold: float = 0.3  # below this, route to LLM
    llm_validation: bool = True  # whether to use LLM for low-confidence extractions
    budget: BudgetConfig | None = None  # reuses budget config


# ── Learning Memory Config ─────────────────────────────────────────────────


class LearningConfig(BaseModel):
    """Learning Memory learning parameters."""
    threshold_min_corrections: int = 10
    weights_min_corrections: int = 50


class MemoryConfig(BaseModel):
    """Learning Memory configuration."""
    enabled: bool = True
    backend: str = "sqlite"
    path: str = ".goldenmatch/memory.db"
    connection: str | None = None
    trust: dict[str, float] = Field(default_factory=lambda: {"human": 1.0, "agent": 0.5})
    learning: LearningConfig = Field(default_factory=LearningConfig)


# ── MatchSettingsConfig ─────────────────────────────────────────────────────


class MatchSettingsConfig(BaseModel):
    matchkeys: list[MatchkeyConfig]


# ── GoldenMatchConfig (top-level) ──────────────────────────────────────────


class GoldenMatchConfig(BaseModel):
    input: InputConfig | None = None
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    match_settings: MatchSettingsConfig | None = None
    matchkeys: list[MatchkeyConfig] | None = None
    blocking: BlockingConfig | None = None
    golden_rules: GoldenRulesConfig | None = None
    standardization: StandardizationConfig | None = None
    validation: ValidationConfig | None = None
    quality: QualityConfig | None = None
    llm_boost: bool = False
    llm_scorer: LLMScorerConfig | None = None
    domain: DomainConfig | None = None
    backend: str | None = None  # None (default Polars), "ray", "duckdb"
    memory: MemoryConfig | None = None

    @model_validator(mode="after")
    def _validate_fuzzy_needs_blocking(self) -> "GoldenMatchConfig":
        mks = self.get_matchkeys()
        has_fuzzy = any(mk.type in ("weighted", "probabilistic") for mk in mks)
        if has_fuzzy and self.blocking is None:
            raise ValueError(
                "Weighted/probabilistic matchkeys require a 'blocking' configuration."
            )
        return self

    def get_matchkeys(self) -> list[MatchkeyConfig]:
        """Return matchkeys from either top-level or match_settings."""
        if self.matchkeys:
            return self.matchkeys
        if self.match_settings:
            return self.match_settings.matchkeys
        return []
