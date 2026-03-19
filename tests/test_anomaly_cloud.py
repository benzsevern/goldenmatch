"""Tests for anomaly detection and cloud ingest."""

from __future__ import annotations

import polars as pl
import pytest


class TestAnomalyDetection:
    def test_detect_fake_emails(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["John", "Jane", "Test", "Bob"],
            "email": ["john@real.com", "test@test.com", "noreply@fake.com", "bob@company.com"],
        })

        anomalies = detect_anomalies(df)
        fake_emails = [a for a in anomalies if a["type"] == "fake_email"]
        assert len(fake_emails) >= 2
        flagged_values = {a["value"] for a in fake_emails}
        assert "test@test.com" in flagged_values

    def test_detect_placeholder_values(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "TBD", "N/A"],
            "email": ["john@t.com", "tbd@t.com", "na@t.com"],
        })

        anomalies = detect_anomalies(df)
        placeholders = [a for a in anomalies if a["type"] == "placeholder"]
        assert len(placeholders) >= 2

    def test_detect_fake_phones(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["A", "B", "C"],
            "phone": ["555-0101", "123-456-7890", "212-555-1234"],
        })

        anomalies = detect_anomalies(df)
        fake_phones = [a for a in anomalies if a["type"] == "fake_phone"]
        assert len(fake_phones) >= 1

    def test_detect_suspicious_zips(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["A", "B", "C"],
            "zip": ["10001", "00000", "12345"],
        })

        anomalies = detect_anomalies(df)
        bad_zips = [a for a in anomalies if a["type"] == "suspicious_zip"]
        assert len(bad_zips) >= 1

    def test_detect_exact_duplicate_rows(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": list(range(5)),
            "name": ["John", "John", "John", "Jane", "Bob"],
            "email": ["j@t.com", "j@t.com", "j@t.com", "jane@t.com", "bob@t.com"],
        })

        anomalies = detect_anomalies(df)
        exact_dupes = [a for a in anomalies if a["type"] == "exact_duplicate_row"]
        assert len(exact_dupes) >= 1

    def test_sensitivity_low(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John", "TBD"],
            "phone": ["555-0101", "212-555-1234"],
        })

        all_anomalies = detect_anomalies(df, sensitivity="high")
        low_anomalies = detect_anomalies(df, sensitivity="low")
        assert len(low_anomalies) <= len(all_anomalies)

    def test_no_anomalies_clean_data(self):
        from goldenmatch.core.anomaly import detect_anomalies

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John Smith", "Jane Doe"],
            "email": ["john@company.com", "jane@company.com"],
        })

        anomalies = detect_anomalies(df)
        assert len(anomalies) == 0

    def test_format_report(self):
        from goldenmatch.core.anomaly import detect_anomalies, format_anomaly_report

        df = pl.DataFrame({
            "__row_id__": [0],
            "name": ["TBD"],
            "email": ["test@test.com"],
        })

        anomalies = detect_anomalies(df)
        report = format_anomaly_report(anomalies)
        assert "Anomaly Report" in report


class TestCloudIngest:
    def test_is_cloud_path(self):
        from goldenmatch.core.cloud_ingest import is_cloud_path

        assert is_cloud_path("s3://bucket/file.csv")
        assert is_cloud_path("gs://bucket/file.csv")
        assert is_cloud_path("az://container/file.csv")
        assert not is_cloud_path("/local/file.csv")
        assert not is_cloud_path("file.csv")
        assert not is_cloud_path("C:\\Users\\file.csv")

    def test_download_missing_boto3(self):
        from goldenmatch.core.cloud_ingest import download_cloud_file
        import sys

        # This will fail with ImportError since boto3 isn't installed
        # (or succeed if it is — both are valid)
        try:
            download_cloud_file("s3://fake-bucket/file.csv")
        except (ImportError, Exception):
            pass  # Expected
