"""Tests for domain-aware feature extraction."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.core.domain import (
    DomainProfile,
    ExtractionResult,
    detect_domain,
    extract_features,
    extract_product_features,
    extract_biblio_features,
)


class TestDomainDetection:
    def test_product_domain(self):
        profile = detect_domain(["title", "price", "brand", "description"])
        assert profile.name == "product"
        assert profile.confidence > 0.5

    def test_person_domain(self):
        profile = detect_domain(["first_name", "last_name", "email", "dob"])
        assert profile.name == "person"
        assert profile.confidence > 0.5

    def test_bibliographic_domain(self):
        profile = detect_domain(["title", "authors", "year", "venue"])
        assert profile.name == "bibliographic"

    def test_unknown_domain(self):
        profile = detect_domain(["col_a", "col_b", "col_c"])
        assert profile.name == "unknown"
        assert profile.confidence == 0.0

    def test_text_columns_identified(self):
        profile = detect_domain(["product_name", "description", "sku", "price"])
        assert "product_name" in profile.text_columns or "description" in profile.text_columns


class TestProductFeatureExtraction:
    def test_extract_model_number(self):
        result = extract_product_features("Sony Cyber-shot DSC-T77 Silver")
        assert result.brand == "Sony"
        assert result.model is not None
        assert "T77" in result.model or "DSC" in result.model

    def test_extract_brand(self):
        result = extract_product_features("Samsung Galaxy S21 Ultra 256GB Black")
        assert result.brand == "Samsung"

    def test_extract_color(self):
        result = extract_product_features("Apple iPhone 14 Pro Max - Space Gray")
        assert result.color is not None
        assert "gray" in result.color

    def test_extract_specs(self):
        result = extract_product_features("Canon EOS R5 45MP Full Frame Mirrorless Camera")
        assert "megapixels" in result.specs

    def test_extract_storage(self):
        result = extract_product_features("Dell XPS 15 Laptop 16GB RAM 512GB SSD")
        assert "storage_gb" in result.specs or "ram_gb" in result.specs

    def test_empty_string(self):
        result = extract_product_features("")
        assert result.confidence == 0.0
        assert result.brand is None

    def test_no_features(self):
        result = extract_product_features("just some random text here")
        assert result.confidence < 0.5

    def test_confidence_scoring(self):
        # Many features = high confidence
        rich = extract_product_features("Sony WH-1000XM5 Black Wireless Headphones 30hr Battery")
        # Few features = lower confidence
        poor = extract_product_features("nice thing for sale cheap")
        assert rich.confidence > poor.confidence

    def test_parenthetical_extraction(self):
        result = extract_product_features("Wireless Mouse (Model: MX-2000)")
        assert result.parenthetical is not None
        assert "MX-2000" in result.parenthetical


class TestModelNormalization:
    def test_strip_hyphens(self):
        from goldenmatch.core.domain import normalize_model
        assert normalize_model("CL-51") == "CL51"
        assert normalize_model("Z-2300") == "Z2300"
        assert normalize_model("DSC-T77") == "DSCT77"

    def test_uppercase(self):
        from goldenmatch.core.domain import normalize_model
        assert normalize_model("cl51") == "CL51"

    def test_strip_region_suffix(self):
        from goldenmatch.core.domain import normalize_model
        assert normalize_model("GS105NA") == "GS105"
        assert normalize_model("JFS516NA") == "JFS516"

    def test_strip_color_suffix(self):
        from goldenmatch.core.domain import normalize_model
        assert normalize_model("NNH965BK") == "NNH965"

    def test_short_model_preserved(self):
        from goldenmatch.core.domain import normalize_model
        # Don't strip suffix if result would be too short
        assert normalize_model("ABNA") is not None

    def test_none_input(self):
        from goldenmatch.core.domain import normalize_model
        assert normalize_model(None) is None

    def test_model_contains(self):
        from goldenmatch.core.domain import model_contains
        assert model_contains("KX-TG6700B", "TG6700B")
        assert model_contains("JFS516NA", "JFS516")
        assert not model_contains("DSC-T77", "DSC-T700")

    def test_model_norm_column_in_df(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": ["Sony CL-51 Cartridge", "Canon CL51 Ink"],
        })
        domain = DomainProfile(name="product", confidence=0.9, text_columns=["name"])
        enhanced, _ = extract_features(df, domain)
        assert "__model_norm__" in enhanced.columns
        norms = enhanced["__model_norm__"].to_list()
        # Both should normalize to the same value
        assert norms[0] == norms[1]


class TestBiblioFeatureExtraction:
    def test_extract_year(self):
        features = extract_biblio_features("A Theory for Record Linkage 1969")
        assert features.get("year") == "1969"

    def test_extract_title_key(self):
        features = extract_biblio_features("The Quick Brown Fox Jumps")
        assert features.get("title_key") == "quick"

    def test_extract_doi(self):
        features = extract_biblio_features("Some paper 10.1145/12345.67890")
        assert features.get("doi") is not None
        assert "10.1145" in features["doi"]


class TestDataFrameExtraction:
    def test_product_df_extraction(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "name": [
                "Sony Cyber-shot DSC-T77 Silver",
                "Canon EOS R5 45MP Camera",
                "random thing no brand",
            ],
        })
        domain = DomainProfile(name="product", confidence=0.9, text_columns=["name"])

        enhanced, low_conf = extract_features(df, domain, confidence_threshold=0.3)

        assert "__brand__" in enhanced.columns
        assert "__model__" in enhanced.columns
        assert enhanced.height == 3

        # First two should have brands, third shouldn't
        brands = enhanced["__brand__"].to_list()
        assert brands[0] == "SONY"
        assert brands[1] == "CANON"

    def test_low_confidence_identified(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": [
                "Sony DSC-T77",  # clear features
                "thing for sale",  # no features
            ],
        })
        domain = DomainProfile(name="product", confidence=0.9, text_columns=["name"])

        _, low_conf = extract_features(df, domain, confidence_threshold=0.3)
        assert 2 in low_conf  # "thing for sale" should be low confidence

    def test_unknown_domain_passthrough(self):
        df = pl.DataFrame({
            "__row_id__": [1],
            "col_a": ["test"],
        })
        domain = DomainProfile(name="unknown", confidence=0.0)

        enhanced, low_conf = extract_features(df, domain)
        assert enhanced.height == 1
        assert len(low_conf) == 0


class TestPipelineIntegration:
    def test_domain_in_config(self):
        from goldenmatch.config.schemas import GoldenMatchConfig, DomainConfig, BlockingConfig, BlockingKeyConfig

        cfg = GoldenMatchConfig(
            matchkeys=[{
                "name": "test",
                "type": "weighted",
                "threshold": 0.7,
                "fields": [{"field": "name", "scorer": "token_sort", "weight": 1.0}],
            }],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["name"])]),
            domain=DomainConfig(enabled=True, mode="product"),
        )
        assert cfg.domain.enabled
        assert cfg.domain.mode == "product"

    def test_pipeline_with_domain(self, tmp_path):
        import csv
        csv_path = tmp_path / "products.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "price"])
            w.writerow(["Sony DSC-T77 Silver", "199"])
            w.writerow(["Sony DSC-T77 Black", "199"])
            w.writerow(["Canon EOS R5 Camera", "3899"])

        from goldenmatch.config.schemas import (
            GoldenMatchConfig, DomainConfig, BlockingConfig, BlockingKeyConfig,
        )
        from goldenmatch.core.pipeline import run_dedupe

        cfg = GoldenMatchConfig(
            matchkeys=[{
                "name": "fuzzy",
                "type": "weighted",
                "threshold": 0.6,
                "fields": [{"field": "name", "scorer": "token_sort", "weight": 1.0}],
            }],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"])],
            ),
            domain=DomainConfig(enabled=True, mode="product", llm_validation=False),
        )

        result = run_dedupe([(str(csv_path), "test")], cfg, output_clusters=True)
        clusters = result["clusters"]
        # The two Sony DSC-T77 records should match
        multi = {cid: c for cid, c in clusters.items() if c["size"] > 1}
        assert len(multi) >= 1
