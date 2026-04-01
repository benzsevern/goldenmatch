"""Custom domain registry -- user-defined extraction rulebooks.

Users can define custom domain extraction rules as YAML files stored in:
- .goldenmatch/domains/ (project-local)
- ~/.goldenmatch/domains/ (global)
- goldenmatch/domains/ (built-in)

Each domain YAML defines:
- name: the domain name
- signals: column name patterns that trigger this domain
- brand_patterns: regex patterns for brand extraction
- identifier_patterns: regex patterns for model/SKU/identifier extraction
- attribute_patterns: named regex patterns for domain-specific attributes
- stop_words: words to strip during name normalization
- normalization: rules for normalizing extracted identifiers

Example (medical_devices.yaml):
    name: medical_devices
    signals: ["device", "ndc", "fda", "implant", "catheter", "lot"]
    identifier_patterns:
      ndc: '\\b(\\d{5}-\\d{4}-\\d{2})\\b'
      lot_number: '\\b(LOT[:#]?\\s*[A-Z0-9]+)\\b'
    brand_patterns: ["Medtronic", "Johnson & Johnson", "Abbott", "Stryker"]
    attribute_patterns:
      size: '\\b(\\d+\\.?\\d*)\\s*(mm|cm|fr|gauge)\\b'
    stop_words: ["sterile", "single", "use", "disposable", "latex", "free"]
"""
from __future__ import annotations

import logging
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default search paths for domain YAML files
_SEARCH_PATHS = [
    Path(".goldenmatch/domains"),           # project-local
    Path.home() / ".goldenmatch" / "domains",  # user global
    Path(__file__).parent.parent / "domains",  # built-in
]


@dataclass
class DomainRulebook:
    """A set of extraction rules for a specific data domain."""

    name: str
    signals: list[str] = field(default_factory=list)
    identifier_patterns: dict[str, str] = field(default_factory=dict)  # name -> regex
    brand_patterns: list[str] = field(default_factory=list)  # brand name strings
    attribute_patterns: dict[str, str] = field(default_factory=dict)  # name -> regex
    stop_words: list[str] = field(default_factory=list)
    normalization: dict[str, str] = field(default_factory=dict)  # rules like "strip_hyphens", "uppercase"
    autoconfig_preset: dict | None = field(default=None, repr=False)  # auto-config tuning preset from YAML

    # Compiled patterns (lazily populated)
    _compiled_ids: dict[str, re.Pattern] | None = field(default=None, repr=False)
    _compiled_attrs: dict[str, re.Pattern] | None = field(default=None, repr=False)
    _compiled_brands: re.Pattern | None = field(default=None, repr=False)

    def compile(self) -> None:
        """Compile regex patterns for performance."""
        self._compiled_ids = {}
        for name, pattern in self.identifier_patterns.items():
            try:
                self._compiled_ids[name] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning("Invalid regex for identifier '%s': %s", name, e)

        self._compiled_attrs = {}
        for name, pattern in self.attribute_patterns.items():
            try:
                self._compiled_attrs[name] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning("Invalid regex for attribute '%s': %s", name, e)

        if self.brand_patterns:
            escaped = [re.escape(b) for b in self.brand_patterns]
            self._compiled_brands = re.compile(
                r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE,
            )

    def extract(self, text: str) -> dict[str, str | None]:
        """Extract features from text using this rulebook.

        Returns a dict with keys: brand, identifiers (dict), attributes (dict),
        name_normalized, confidence.
        """
        if self._compiled_ids is None:
            self.compile()

        result: dict[str, str | None] = {"brand": None, "name_normalized": None}
        identifiers: dict[str, str] = {}
        attributes: dict[str, str] = {}
        signals = 0

        # Brand
        if self._compiled_brands:
            m = self._compiled_brands.search(text)
            if m:
                result["brand"] = m.group(1).strip()
                signals += 1

        # Identifiers
        for name, pattern in (self._compiled_ids or {}).items():
            m = pattern.search(text)
            if m:
                identifiers[name] = m.group(1).strip() if m.lastindex else m.group(0).strip()
                signals += 1
        result["identifiers"] = identifiers

        # Attributes
        for name, pattern in (self._compiled_attrs or {}).items():
            m = pattern.search(text)
            if m:
                attributes[name] = m.group(0).strip()
                signals += 0.5
        result["attributes"] = attributes

        # Name normalization
        name = text.lower()
        # Remove identified parts
        for pattern in (self._compiled_ids or {}).values():
            name = pattern.sub(" ", name)
        for pattern in (self._compiled_attrs or {}).values():
            name = pattern.sub(" ", name)
        # Remove stop words
        stop = frozenset(self.stop_words)
        words = [w for w in re.sub(r"[^\w\s]", " ", name).split()
                 if w not in stop and len(w) > 1]
        result["name_normalized"] = " ".join(words).strip() if words else None
        if result["name_normalized"]:
            signals += 1

        result["confidence"] = min(1.0, signals / max(len(self.identifier_patterns) + 1, 2))
        return result


def load_rulebook(path: str | Path) -> DomainRulebook:
    """Load a domain rulebook from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain rulebook not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rulebook = DomainRulebook(
        name=data.get("name", path.stem),
        signals=data.get("signals", []),
        identifier_patterns=data.get("identifier_patterns", {}),
        brand_patterns=data.get("brand_patterns", []),
        attribute_patterns=data.get("attribute_patterns", {}),
        stop_words=data.get("stop_words", []),
        normalization=data.get("normalization", {}),
        autoconfig_preset=data.get("autoconfig_preset"),
    )
    rulebook.compile()
    return rulebook


def save_rulebook(rulebook: DomainRulebook, path: str | Path) -> Path:
    """Save a domain rulebook to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": rulebook.name,
        "signals": rulebook.signals,
        "identifier_patterns": rulebook.identifier_patterns,
        "brand_patterns": rulebook.brand_patterns,
        "attribute_patterns": rulebook.attribute_patterns,
        "stop_words": rulebook.stop_words,
        "normalization": rulebook.normalization,
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved domain rulebook '%s' to %s", rulebook.name, path)
    return path


def discover_rulebooks() -> dict[str, DomainRulebook]:
    """Discover all available domain rulebooks from search paths."""
    rulebooks: dict[str, DomainRulebook] = {}

    for search_dir in _SEARCH_PATHS:
        if not search_dir.exists():
            continue
        for yaml_path in search_dir.glob("*.yaml"):
            try:
                rb = load_rulebook(yaml_path)
                rulebooks[rb.name] = rb
                logger.debug("Loaded domain rulebook '%s' from %s", rb.name, yaml_path)
            except Exception as e:
                logger.warning("Failed to load domain rulebook %s: %s", yaml_path, e)

        for yml_path in search_dir.glob("*.yml"):
            try:
                rb = load_rulebook(yml_path)
                rulebooks[rb.name] = rb
            except Exception as e:
                logger.warning("Failed to load domain rulebook %s: %s", yml_path, e)

    return rulebooks


def match_domain(columns: list[str], rulebooks: dict[str, DomainRulebook] | None = None) -> DomainRulebook | None:
    """Find the best matching domain rulebook for the given columns.

    Scores each rulebook by how many signal words appear in column names.
    Returns the best match, or None if no rulebook matches.
    """
    if rulebooks is None:
        rulebooks = discover_rulebooks()

    if not rulebooks:
        return None

    col_str = " ".join(c.lower() for c in columns)
    best_rb = None
    best_score = 0

    for rb in rulebooks.values():
        score = sum(1 for s in rb.signals if s.lower() in col_str)
        if score > best_score:
            best_score = score
            best_rb = rb

    if best_score > 0:
        return best_rb
    return None


def extract_with_rulebook(
    df,
    text_column: str,
    rulebook: DomainRulebook,
    confidence_threshold: float = 0.3,
) -> tuple:
    """Extract features from a DataFrame using a custom rulebook.

    Returns (enhanced_df, low_confidence_ids).
    """
    import polars as pl

    names = []
    brands = []
    id_strs = []
    attr_strs = []
    confidences = []
    low_conf_ids = []

    for row in df.select(["__row_id__", text_column]).to_dicts():
        text = str(row.get(text_column, "") or "")
        rid = row["__row_id__"]

        result = rulebook.extract(text)
        names.append(result.get("name_normalized"))
        brands.append(result.get("brand"))

        ids = result.get("identifiers", {})
        id_strs.append("|".join(f"{k}:{v}" for k, v in sorted(ids.items())) if ids else None)

        attrs = result.get("attributes", {})
        attr_strs.append("|".join(f"{k}:{v}" for k, v in sorted(attrs.items())) if attrs else None)

        conf = result.get("confidence", 0)
        confidences.append(conf)
        if conf < confidence_threshold:
            low_conf_ids.append(rid)

    enhanced = df.with_columns([
        pl.Series("__domain_name__", names, dtype=pl.Utf8),
        pl.Series("__domain_brand__", brands, dtype=pl.Utf8),
        pl.Series("__domain_ids__", id_strs, dtype=pl.Utf8),
        pl.Series("__domain_attrs__", attr_strs, dtype=pl.Utf8),
        pl.Series("__extract_confidence__", confidences, dtype=pl.Float64),
    ])

    n_names = sum(1 for n in names if n)
    logger.info(
        "Custom domain '%s' extraction: %d/%d names, %d low-confidence",
        rulebook.name, n_names, df.height, len(low_conf_ids),
    )

    return enhanced, low_conf_ids
