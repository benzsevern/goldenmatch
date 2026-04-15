"""Domain-aware feature extraction for entity resolution.

Shifts LLM usage from O(N^2) pair comparison to O(N) preprocessing.
Detects the data domain (product, person, bibliographic, company),
extracts structured features using heuristics, and routes low-confidence
extractions to an LLM for validation.

Pipeline insertion point: between standardize and matchkeys.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


# Columns that extract_features may add to a DataFrame, by domain:
#   bibliographic → __title_key__
#   electronics   → __brand__, __model__, __model_norm__, __color__, __specs__, __extract_confidence__
#   software      → __sw_name__, __sw_version__, __sw_edition__, __sw_platform__, __sw_part_num__, __extract_confidence__
# If new extractors are added, append here.
_DOMAIN_EXTRACTED_COLS: frozenset[str] = frozenset({
    "__title_key__",
    "__brand__", "__model__", "__model_norm__", "__color__", "__specs__",
    "__sw_name__", "__sw_version__", "__sw_edition__", "__sw_platform__", "__sw_part_num__",
    "__extract_confidence__",
})


# ── Domain Profiles ───────────────────────────────────────────────────────

@dataclass
class DomainProfile:
    """Detected data domain with extraction rules."""

    name: str  # product, person, bibliographic, company, unknown
    confidence: float
    text_columns: list[str] = field(default_factory=list)  # columns to extract from
    id_columns: list[str] = field(default_factory=list)  # existing ID columns
    subdomain: str | None = None  # "electronics", "software", etc.


# Domain detection heuristics
_PRODUCT_SIGNALS = re.compile(
    r"(product|title|name|description|brand|manufacturer|mfr|sku|model|price|upc|ean|asin)",
    re.IGNORECASE,
)
_PERSON_SIGNALS = re.compile(
    r"(first.?name|last.?name|fname|lname|surname|given.?name|dob|birth|ssn|email)",
    re.IGNORECASE,
)
_BIBLIO_SIGNALS = re.compile(
    r"(title|authors?|year|venue|journal|doi|isbn|abstract|citation)",
    re.IGNORECASE,
)
_COMPANY_SIGNALS = re.compile(
    r"(company|org|corporation|inc|llc|ltd|industry|sector|ticker|ein)",
    re.IGNORECASE,
)


def detect_domain(columns: list[str]) -> DomainProfile:
    """Detect the data domain from column names.

    Returns a DomainProfile with confidence and relevant columns.
    """
    cols_lower = [c.lower() for c in columns]
    col_str = " ".join(cols_lower)

    scores = {
        "product": len(_PRODUCT_SIGNALS.findall(col_str)),
        "person": len(_PERSON_SIGNALS.findall(col_str)),
        "bibliographic": len(_BIBLIO_SIGNALS.findall(col_str)),
        "company": len(_COMPANY_SIGNALS.findall(col_str)),
    }

    best = max(scores, key=scores.get)
    total_signals = sum(scores.values())

    if total_signals == 0:
        return DomainProfile(name="unknown", confidence=0.0)

    confidence = scores[best] / total_signals

    # Identify text columns for extraction
    text_cols = []
    id_cols = []
    for c in columns:
        if c.startswith("__"):
            continue
        cl = c.lower()
        if any(p in cl for p in ("title", "name", "description", "product")):
            text_cols.append(c)
        elif any(p in cl for p in ("id", "sku", "model", "upc", "ean", "asin", "isbn", "doi")):
            id_cols.append(c)

    # Sub-domain detection for products: electronics vs software
    subdomain = None
    if best == "product":
        sw_signals = sum(1 for c in cols_lower if any(
            p in c for p in ("software", "version", "license", "publisher")
        ))
        hw_signals = sum(1 for c in cols_lower if any(
            p in c for p in ("brand", "model", "sku", "upc", "ean", "weight", "dimensions")
        ))
        if sw_signals > hw_signals:
            subdomain = "software"
        elif hw_signals > 0:
            subdomain = "electronics"
        # If can't tell from columns, will auto-detect from data content

    return DomainProfile(
        name=best,
        confidence=confidence,
        text_columns=text_cols,
        id_columns=id_cols,
        subdomain=subdomain,
    )


# ── Product Feature Extraction ────────────────────────────────────────────

# Regex patterns for common product identifiers
_MODEL_NUMBER = re.compile(
    r"\b([A-Z]{1,5}[-]?\d{2,}[A-Z0-9]*(?:[-/]\w+)?)\b"
)
_SKU_PATTERN = re.compile(
    r"\b(\d{6,}|[A-Z]{2,}\d{4,})\b"
)
_BRAND_PREFIXES = re.compile(
    r"^(Sony|Samsung|LG|Apple|HP|Dell|Canon|Nikon|Panasonic|Philips|"
    r"Bose|JBL|Logitech|Microsoft|Intel|AMD|Asus|Acer|Lenovo|"
    r"Toshiba|Epson|Brother|Xerox|Bosch|Makita|DeWalt|"
    r"Nike|Adidas|Puma|Reebok|Under Armour)\b",
    re.IGNORECASE,
)
_SPEC_PATTERNS = {
    "megapixels": re.compile(r"(\d+\.?\d*)\s*(?:mega\s*pixel|mp)\b", re.IGNORECASE),
    "storage_gb": re.compile(r"(\d+)\s*(?:gb|gigabyte)\b", re.IGNORECASE),
    "storage_tb": re.compile(r"(\d+\.?\d*)\s*(?:tb|terabyte)\b", re.IGNORECASE),
    "screen_inch": re.compile(r"(\d+\.?\d*)\s*(?:inch|in\.|\")\b", re.IGNORECASE),
    "ram_gb": re.compile(r"(\d+)\s*gb\s*(?:ram|ddr|memory)\b", re.IGNORECASE),
    "watts": re.compile(r"(\d+)\s*(?:watt|w)\b", re.IGNORECASE),
    "speed_ghz": re.compile(r"(\d+\.?\d*)\s*ghz\b", re.IGNORECASE),
}
_PARENTHETICAL = re.compile(r"\(([^)]+)\)")
_COLOR_PATTERN = re.compile(
    r"\b(black|white|silver|gold|red|blue|green|pink|gray|grey|"
    r"purple|orange|yellow|brown|navy|titanium|graphite|midnight|"
    r"space\s*gray|rose\s*gold)\b",
    re.IGNORECASE,
)


@dataclass
class ExtractionResult:
    """Result of feature extraction for a single record."""

    brand: str | None = None
    model: str | None = None
    sku: str | None = None
    color: str | None = None
    specs: dict[str, str] = field(default_factory=dict)
    parenthetical: str | None = None
    confidence: float = 0.0  # 0-1, how confident we are in the extraction


def extract_product_features(text: str) -> ExtractionResult:
    """Extract structured features from a product title/description.

    Uses regex heuristics to pull out brand, model number, SKU,
    color, and key specifications.
    """
    if not text or not text.strip():
        return ExtractionResult(confidence=0.0)

    result = ExtractionResult()
    signals = 0
    total_possible = 4  # brand, model, color, any spec

    # Brand
    brand_match = _BRAND_PREFIXES.search(text)
    if brand_match:
        result.brand = brand_match.group(1).strip()
        signals += 1

    # Model number (alphanumeric identifiers like DSC-T77, XPS-15, etc.)
    model_match = _MODEL_NUMBER.search(text)
    if model_match:
        result.model = model_match.group(1).strip()
        signals += 1

    # SKU (long numeric strings or alphanumeric codes)
    if not result.model:
        sku_match = _SKU_PATTERN.search(text)
        if sku_match:
            result.sku = sku_match.group(1).strip()
            signals += 0.5

    # Color
    color_match = _COLOR_PATTERN.search(text)
    if color_match:
        result.color = color_match.group(1).strip().lower()
        signals += 0.5

    # Specs
    for spec_name, pattern in _SPEC_PATTERNS.items():
        spec_match = pattern.search(text)
        if spec_match:
            result.specs[spec_name] = spec_match.group(1)
            signals += 0.5

    # Parenthetical content (often contains model info)
    paren_match = _PARENTHETICAL.search(text)
    if paren_match:
        result.parenthetical = paren_match.group(1).strip()

    # Confidence based on how many features we extracted
    result.confidence = min(1.0, signals / total_possible)

    return result


# ── Model Number Normalization ─────────────────────────────────────────────

_REGION_SUFFIXES = re.compile(r"(NA|US|EU|UK|CA|AU|JP)$")
_COLOR_SUFFIXES = re.compile(r"(BK|WH|BF|WF|SL|GR|BL|RD|GD|SV)$")


def normalize_model(model: str | None) -> str | None:
    """Normalize a model number for matching.

    Applies: uppercase, strip hyphens/spaces/slashes, strip region/color suffixes.
    "CL-51" -> "CL51", "GS105NA" -> "GS105", "KX-TG6700B" -> "KXTG6700B"
    """
    if not model:
        return None
    n = model.upper().replace("-", "").replace(" ", "").replace("/", "")
    # Strip region suffixes (NA, US, EU, etc.)
    n = _REGION_SUFFIXES.sub("", n)
    # Strip color suffixes (BK, WH, BF, WF, etc.) only if result is still 4+ chars
    stripped = _COLOR_SUFFIXES.sub("", n)
    if len(stripped) >= 4:
        n = stripped
    return n


def model_contains(model_a: str | None, model_b: str | None) -> bool:
    """Check if one normalized model is contained in the other.

    Handles cases like "KX-TG6700B" vs "TG6700B" where one includes
    a manufacturer prefix the other doesn't.
    """
    if not model_a or not model_b:
        return False
    na = normalize_model(model_a)
    nb = normalize_model(model_b)
    if not na or not nb:
        return False
    return na in nb or nb in na


# ── Software Product Feature Extraction ────────────────────────────────────

_SW_VERSION = re.compile(
    r"\b(?:v\.?\s*)?(\d+(?:\.\d+)+)\b"  # "6.5", "3.0", "v. 9.4", "v5.0"
    r"|\b((?:cs|cc)\s*\d+)\b"  # "cs3", "cc 2024"
    r"|\b(20[0-2]\d)\b",  # year as version "2007", "2006"
    re.IGNORECASE,
)
_SW_EDITION = re.compile(
    r"\b(professional|pro|standard|enterprise|premium|basic|"
    r"ultimate|home|personal|academic|student|unlimited|"
    r"plus|lite|express|starter|essentials?)\b",
    re.IGNORECASE,
)
_SW_PLATFORM = re.compile(
    r"\b(windows?|win|mac(?:intosh)?|linux|unix|osx?|ios|android)\b"
    r"|\b(win(?:/mac|\\mac))\b"
    r"|\bfor\s+(pc|mac)\b",
    re.IGNORECASE,
)
_SW_PART_NUMBER = re.compile(r"\b(\d{5,})\b")
_SW_UPGRADE = re.compile(r"\b(upgrade|upg|update)\b", re.IGNORECASE)

# Stop words to strip from software product names for normalization
_SW_STOP_WORDS = frozenset({
    "the", "a", "an", "for", "and", "or", "with", "by", "from", "to",
    "in", "of", "-", "inc", "inc.", "llc", "corp", "software", "edition",
    "version", "ver", "cd", "dvd", "rom", "cd-rom", "dvd-rom",
    "jewel", "case", "package", "complete", "license", "media",
})


@dataclass
class SoftwareExtractionResult:
    """Extracted features from a software product title."""

    name_normalized: str | None = None  # core product name, stripped of noise
    version: str | None = None
    edition: str | None = None
    platform: str | None = None
    part_number: str | None = None
    is_upgrade: bool = False
    confidence: float = 0.0


def extract_software_features(text: str) -> SoftwareExtractionResult:
    """Extract structured features from a software product title."""
    if not text or not text.strip():
        return SoftwareExtractionResult(confidence=0.0)

    result = SoftwareExtractionResult()
    text_lower = text.lower().strip()
    signals = 0
    total_possible = 3  # name, version, edition

    # Version
    ver_match = _SW_VERSION.search(text)
    if ver_match:
        result.version = (ver_match.group(1) or ver_match.group(2) or ver_match.group(3) or "").strip().lower()
        signals += 1

    # Edition
    ed_match = _SW_EDITION.search(text)
    if ed_match:
        result.edition = ed_match.group(1).strip().lower()
        # Normalize "professional" -> "pro"
        if result.edition == "professional":
            result.edition = "pro"
        signals += 0.5

    # Platform
    plat_match = _SW_PLATFORM.search(text)
    if plat_match:
        result.platform = (plat_match.group(1) or plat_match.group(2) or plat_match.group(3) or "").strip().lower()
        if result.platform.startswith("win"):
            result.platform = "win"
        signals += 0.3

    # Part number
    pn_match = _SW_PART_NUMBER.search(text)
    if pn_match:
        result.part_number = pn_match.group(1)
        signals += 0.5

    # Upgrade flag
    if _SW_UPGRADE.search(text):
        result.is_upgrade = True
        signals += 0.2

    # Normalized product name: strip version, edition, platform, stop words, punctuation
    name = text_lower
    # Remove version strings
    name = _SW_VERSION.sub(" ", name)
    # Remove edition
    name = _SW_EDITION.sub(" ", name)
    # Remove platform
    name = _SW_PLATFORM.sub(" ", name)
    # Remove part numbers
    name = _SW_PART_NUMBER.sub(" ", name)
    # Remove upgrade keywords
    name = _SW_UPGRADE.sub(" ", name)
    # Remove parenthetical content
    name = re.sub(r"\([^)]*\)", " ", name)
    # Remove punctuation
    name = re.sub(r"[^\w\s]", " ", name)
    # Remove stop words and collapse whitespace
    words = [w for w in name.split() if w not in _SW_STOP_WORDS and len(w) > 1]
    result.name_normalized = " ".join(words).strip() if words else None

    if result.name_normalized and len(result.name_normalized) >= 3:
        signals += 1

    result.confidence = min(1.0, signals / total_possible)
    return result


# ── Bibliographic Feature Extraction ──────────────────────────────────────

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_DOI_PATTERN = re.compile(r"10\.\d{4,}/\S+")


def extract_biblio_features(text: str) -> dict[str, str | None]:
    """Extract features from bibliographic text (title, abstract)."""
    result = {}
    year_match = _YEAR_PATTERN.search(text)
    if year_match:
        result["year"] = year_match.group(0)
    doi_match = _DOI_PATTERN.search(text)
    if doi_match:
        result["doi"] = doi_match.group(0)
    # First significant word (lowered, stripped of articles)
    words = text.lower().split()
    skip = {"the", "a", "an", "on", "in", "of", "for", "and", "to", "with"}
    for w in words:
        if w not in skip and len(w) > 2:
            result["title_key"] = w
            break
    return result


# ── DataFrame-Level Extraction ────────────────────────────────────────────


def extract_features(
    df: pl.DataFrame,
    domain: DomainProfile,
    confidence_threshold: float = 0.3,
) -> tuple[pl.DataFrame, list[int]]:
    """Extract structured features from text columns, adding derived columns.

    Args:
        df: Input DataFrame with __row_id__.
        domain: Detected domain profile.
        confidence_threshold: Records below this go to LLM validation.

    Returns:
        (enhanced_df, low_confidence_row_ids) — DataFrame with new columns
        and list of row IDs that need LLM validation.
    """
    if not domain.text_columns:
        return df, []

    if domain.name == "product":
        # Auto-detect software vs electronics from data if not already set
        if domain.subdomain is None:
            domain.subdomain = _detect_product_subdomain(df, domain)
            logger.info("Product subdomain auto-detected: %s", domain.subdomain)

        if domain.subdomain == "software":
            return _extract_software_features_df(df, domain, confidence_threshold)
        else:
            return _extract_product_features_df(df, domain, confidence_threshold)
    elif domain.name == "bibliographic":
        return _extract_biblio_features_df(df, domain, confidence_threshold)
    else:
        # Person and company domains — minimal extraction for now
        return df, []


def _detect_product_subdomain(df: pl.DataFrame, domain: DomainProfile) -> str:
    """Auto-detect whether product data is electronics or software.

    Samples text columns and counts software-specific signals vs
    electronics-specific signals.
    """
    text_col = domain.text_columns[0] if domain.text_columns else None
    if not text_col or text_col not in df.columns:
        return "electronics"

    sample = df.head(min(200, df.height))
    sw_signals = 0
    hw_signals = 0

    for row in sample.select([text_col]).to_dicts():
        text = str(row.get(text_col, "") or "").lower()
        # Software signals
        if any(w in text for w in ("software", "license", "edition", "upgrade", "upg",
                                    "cd-rom", "dvd-rom", "download", "subscription")):
            sw_signals += 1
        if _SW_VERSION.search(text):
            sw_signals += 0.5
        # Electronics signals
        if _MODEL_NUMBER.search(text.upper()):
            hw_signals += 1
        if any(w in text for w in ("megapixel", "mp", "ghz", "mhz", "watt",
                                    "inch", "battery", "wireless", "bluetooth")):
            hw_signals += 1

    logger.debug("Product subdomain detection: sw=%.0f, hw=%.0f", sw_signals, hw_signals)
    return "software" if sw_signals > hw_signals else "electronics"


def _extract_software_features_df(
    df: pl.DataFrame,
    domain: DomainProfile,
    confidence_threshold: float,
) -> tuple[pl.DataFrame, list[int]]:
    """Extract software product features into derived columns."""
    text_col = domain.text_columns[0]

    names_norm = []
    versions = []
    editions = []
    platforms = []
    part_numbers = []
    confidences = []
    low_confidence_ids = []

    for row in df.select(["__row_id__", text_col]).to_dicts():
        text = str(row.get(text_col, "") or "")
        rid = row["__row_id__"]

        result = extract_software_features(text)
        names_norm.append(result.name_normalized)
        versions.append(result.version)
        editions.append(result.edition)
        platforms.append(result.platform)
        part_numbers.append(result.part_number)
        confidences.append(result.confidence)

        if result.confidence < confidence_threshold:
            low_confidence_ids.append(rid)

    enhanced = df.with_columns([
        pl.Series("__sw_name__", names_norm, dtype=pl.Utf8),
        pl.Series("__sw_version__", versions, dtype=pl.Utf8),
        pl.Series("__sw_edition__", editions, dtype=pl.Utf8),
        pl.Series("__sw_platform__", platforms, dtype=pl.Utf8),
        pl.Series("__sw_part_num__", part_numbers, dtype=pl.Utf8),
        pl.Series("__extract_confidence__", confidences, dtype=pl.Float64),
    ])

    n_names = sum(1 for n in names_norm if n)
    n_versions = sum(1 for v in versions if v)
    logger.info(
        "Software extraction: %d/%d names, %d/%d versions, %d low-confidence",
        n_names, df.height, n_versions, df.height, len(low_confidence_ids),
    )

    return enhanced, low_confidence_ids


def _extract_product_features_df(
    df: pl.DataFrame,
    domain: DomainProfile,
    confidence_threshold: float,
) -> tuple[pl.DataFrame, list[int]]:
    """Extract product features into derived columns."""
    text_col = domain.text_columns[0]  # primary text column

    brands = []
    models = []
    colors = []
    specs_strs = []
    confidences = []
    low_confidence_ids = []

    for row in df.select(["__row_id__", text_col]).to_dicts():
        text = str(row.get(text_col, "") or "")
        rid = row["__row_id__"]

        result = extract_product_features(text)
        brands.append(result.brand)
        models.append(result.model or result.sku)
        colors.append(result.color)

        # Flatten specs to a single string for matching
        spec_parts = [f"{k}:{v}" for k, v in sorted(result.specs.items())]
        specs_strs.append("|".join(spec_parts) if spec_parts else None)
        confidences.append(result.confidence)

        if result.confidence < confidence_threshold:
            low_confidence_ids.append(rid)

    # Normalize models for matching
    models_norm = [normalize_model(m) for m in models]

    # Normalize brands
    brands_norm = [b.upper().strip() if b else None for b in brands]

    # Add derived columns
    enhanced = df.with_columns([
        pl.Series("__brand__", brands_norm, dtype=pl.Utf8),
        pl.Series("__model__", models, dtype=pl.Utf8),
        pl.Series("__model_norm__", models_norm, dtype=pl.Utf8),
        pl.Series("__color__", colors, dtype=pl.Utf8),
        pl.Series("__specs__", specs_strs, dtype=pl.Utf8),
        pl.Series("__extract_confidence__", confidences, dtype=pl.Float64),
    ])

    n_with_model = sum(1 for m in models if m)
    n_with_brand = sum(1 for b in brands if b)
    logger.info(
        "Product extraction: %d/%d models, %d/%d brands, %d low-confidence",
        n_with_model, df.height, n_with_brand, df.height, len(low_confidence_ids),
    )

    return enhanced, low_confidence_ids


def _extract_biblio_features_df(
    df: pl.DataFrame,
    domain: DomainProfile,
    confidence_threshold: float,
) -> tuple[pl.DataFrame, list[int]]:
    """Extract bibliographic features."""
    text_col = domain.text_columns[0] if domain.text_columns else "title"
    if text_col not in df.columns:
        return df, []

    title_keys = []
    for row in df.select([text_col]).to_dicts():
        text = str(row.get(text_col, "") or "")
        features = extract_biblio_features(text)
        title_keys.append(features.get("title_key"))

    enhanced = df.with_columns([
        pl.Series("__title_key__", title_keys, dtype=pl.Utf8),
    ])

    return enhanced, []
