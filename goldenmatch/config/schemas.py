"""Pydantic models for GoldenMatch configuration validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Valid enums ─────────────────────────────────────────────────────────────

VALID_SIMPLE_TRANSFORMS = frozenset({
    "lowercase", "uppercase", "strip", "strip_all", "soundex", "metaphone",
    "digits_only", "alpha_only", "normalize_whitespace",
})

VALID_SCORERS = frozenset({
    "exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match",
})

VALID_STRATEGIES = frozenset({
    "most_recent", "source_priority", "most_complete", "majority_vote", "first_non_null",
})

_SUBSTRING_RE = re.compile(r"^substring:\d+:\d+$")


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

    @model_validator(mode="after")
    def _resolve_field_column(self) -> "MatchkeyField":
        # Allow 'column' as alias for 'field'
        if self.field is None and self.column is not None:
            self.field = self.column
        elif self.field is None and self.column is None:
            raise ValueError("MatchkeyField requires 'field' or 'column'.")
        for t in self.transforms:
            FieldTransform(transform=t)  # reuse validation
        if self.scorer is not None and self.scorer not in VALID_SCORERS:
            raise ValueError(
                f"Invalid scorer '{self.scorer}'. Must be one of {sorted(VALID_SCORERS)}."
            )
        return self


# ── MatchkeyConfig ──────────────────────────────────────────────────────────


class MatchkeyConfig(BaseModel):
    name: str
    type: Literal["exact", "weighted"] | None = None
    comparison: str | None = None
    fields: list[MatchkeyField]
    threshold: float | None = None

    @model_validator(mode="after")
    def _validate_weighted(self) -> "MatchkeyConfig":
        # Allow 'comparison' as alias for 'type'
        if self.type is None and self.comparison is not None:
            if self.comparison in ("exact", "weighted"):
                self.type = self.comparison
            else:
                raise ValueError(f"Invalid comparison '{self.comparison}'. Must be 'exact' or 'weighted'.")
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


class BlockingConfig(BaseModel):
    keys: list[BlockingKeyConfig]
    max_block_size: int = 5000
    skip_oversized: bool = False


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

    @model_validator(mode="after")
    def _validate_fuzzy_needs_blocking(self) -> "GoldenMatchConfig":
        mks = self.get_matchkeys()
        has_weighted = any(mk.type == "weighted" for mk in mks)
        if has_weighted and self.blocking is None:
            raise ValueError(
                "Weighted/fuzzy matchkeys require a 'blocking' configuration."
            )
        return self

    def get_matchkeys(self) -> list[MatchkeyConfig]:
        """Return matchkeys from either top-level or match_settings."""
        if self.matchkeys:
            return self.matchkeys
        if self.match_settings:
            return self.match_settings.matchkeys
        return []
