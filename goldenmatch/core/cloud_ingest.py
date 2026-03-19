"""Cloud storage ingest -- read files from S3, GCS, and Azure Blob.

Supports:
  goldenmatch dedupe s3://bucket/path/file.csv
  goldenmatch dedupe gs://bucket/path/file.parquet
  goldenmatch dedupe az://container/path/file.xlsx

Downloads to a temp file, then processes normally.
Uses boto3 (S3), google-cloud-storage (GCS), or azure-storage-blob (Azure).
Falls back to fsspec if available (handles all three).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_cloud_path(path: str) -> bool:
    """Check if a path is a cloud storage URL."""
    return path.startswith(("s3://", "gs://", "az://", "abfs://"))


def download_cloud_file(cloud_path: str) -> str:
    """Download a cloud file to a local temp file.

    Args:
        cloud_path: s3://bucket/key, gs://bucket/blob, az://container/blob

    Returns:
        Path to local temp file.
    """
    parsed = urlparse(cloud_path)
    scheme = parsed.scheme

    # Determine file extension
    remote_name = parsed.path.rstrip("/").split("/")[-1]
    suffix = Path(remote_name).suffix or ".csv"

    # Create temp file
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="gm_cloud_")
    local_path = tmp.name
    tmp.close()

    if scheme == "s3":
        _download_s3(parsed, local_path)
    elif scheme == "gs":
        _download_gcs(parsed, local_path)
    elif scheme in ("az", "abfs"):
        _download_azure(parsed, local_path)
    else:
        # Try fsspec as universal fallback
        _download_fsspec(cloud_path, local_path)

    logger.info("Downloaded %s to %s", cloud_path, local_path)
    return local_path


def _download_s3(parsed, local_path: str) -> None:
    """Download from S3 using boto3."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "S3 support requires boto3. Install with: pip install boto3"
        )

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    logger.info("Downloading s3://%s/%s ...", bucket, key)
    s3.download_file(bucket, key, local_path)


def _download_gcs(parsed, local_path: str) -> None:
    """Download from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "GCS support requires google-cloud-storage. "
            "Install with: pip install google-cloud-storage"
        )

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info("Downloading gs://%s/%s ...", bucket_name, blob_name)
    blob.download_to_filename(local_path)


def _download_azure(parsed, local_path: str) -> None:
    """Download from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
        import os
    except ImportError:
        raise ImportError(
            "Azure Blob support requires azure-storage-blob. "
            "Install with: pip install azure-storage-blob"
        )

    container = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise ValueError(
            "Set AZURE_STORAGE_CONNECTION_STRING environment variable"
        )

    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    logger.info("Downloading az://%s/%s ...", container, blob_name)

    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        data.readinto(f)


def _download_fsspec(cloud_path: str, local_path: str) -> None:
    """Fallback: use fsspec for any supported filesystem."""
    try:
        import fsspec
    except ImportError:
        raise ImportError(
            f"Cannot read {cloud_path}. Install the appropriate package:\n"
            f"  S3:    pip install boto3\n"
            f"  GCS:   pip install google-cloud-storage\n"
            f"  Azure: pip install azure-storage-blob\n"
            f"  Or install fsspec: pip install fsspec s3fs gcsfs"
        )

    logger.info("Downloading %s via fsspec...", cloud_path)
    with fsspec.open(cloud_path, "rb") as remote:
        with open(local_path, "wb") as local:
            local.write(remote.read())
