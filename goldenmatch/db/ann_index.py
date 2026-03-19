"""Persistent FAISS ANN index manager for database integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from goldenmatch.db.connector import DatabaseConnector

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = ".goldenmatch_faiss"
DEFAULT_MIN_COVERAGE = 0.10  # 10% of records must be embedded for ANN to activate


class PersistentANNIndex:
    """Manages a persistent FAISS index backed by gm_embeddings table."""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        connector: DatabaseConnector | None = None,
        source_table: str = "",
        model_name: str = "all-MiniLM-L6-v2",
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ):
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.connector = connector
        self.source_table = source_table
        self.model_name = model_name
        self.min_coverage = min_coverage

        self._index = None
        self._id_map: list[int] = []  # positional index → DB record ID
        self._id_to_pos: dict[int, int] = {}  # DB record ID → positional index
        self._dim: int = 0
        self._loaded = False

    @property
    def is_available(self) -> bool:
        """True if index has enough embeddings for useful queries."""
        if self._index is None or len(self._id_map) == 0:
            return False
        if self.connector is None:
            return len(self._id_map) > 0
        try:
            total = self.connector.get_row_count(self.source_table)
            if total == 0:
                return False
            coverage = len(self._id_map) / total
            return coverage >= self.min_coverage
        except Exception:
            return len(self._id_map) > 0

    @property
    def record_count(self) -> int:
        return len(self._id_map)

    # ── Load / Build ──────────────────────────────────────────────────

    def load_or_build(self) -> None:
        """Load index from disk if fresh, rebuild from DB if stale."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not installed. ANN index unavailable.")
            return

        disk_count = self._load_from_disk()
        db_count = self._get_db_embedding_count()

        if disk_count > 0 and disk_count >= db_count:
            logger.info("ANN index loaded from disk (%d embeddings)", disk_count)
            self._loaded = True
            return

        if disk_count > 0 and db_count > disk_count:
            # Append delta from DB
            delta = self._load_delta_from_db(disk_count)
            if delta is not None:
                ids, embeddings = delta
                self._add_to_index(ids, embeddings)
                logger.info("ANN index updated: %d → %d embeddings", disk_count, len(self._id_map))
                self.save()
            self._loaded = True
            return

        if db_count > 0:
            self._rebuild_from_db()
            self.save()
            self._loaded = True
            return

        logger.info("No embeddings available yet. ANN index empty.")

    def _load_from_disk(self) -> int:
        """Load FAISS index + id_map from disk. Returns record count or 0."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "index_meta.json"
        idmap_path = self.index_dir / "id_map.npy"

        if not all(p.exists() for p in [index_path, meta_path, idmap_path]):
            return 0

        try:
            import faiss

            with open(meta_path) as f:
                meta = json.load(f)

            self._index = faiss.read_index(str(index_path))
            self._id_map = np.load(str(idmap_path)).tolist()
            self._id_to_pos = {rid: i for i, rid in enumerate(self._id_map)}
            self._dim = meta.get("dim", 0)

            return len(self._id_map)
        except Exception as e:
            logger.warning("Failed to load ANN index from disk: %s", e)
            return 0

    def _get_db_embedding_count(self) -> int:
        """Count embeddings in gm_embeddings for this table."""
        if self.connector is None:
            return 0
        try:
            df = self.connector.read_query(
                f"SELECT COUNT(*) as cnt FROM gm_embeddings "
                f"WHERE source_table = '{self.source_table}' "
                f"AND model_name = '{self.model_name}'"
            )
            return int(df["cnt"][0]) if df.height > 0 else 0
        except Exception:
            return 0

    def _load_delta_from_db(self, existing_count: int) -> tuple[list[int], np.ndarray] | None:
        """Load embeddings from DB that aren't in the disk index."""
        try:
            df = self.connector.read_query(
                f"SELECT record_id, embedding FROM gm_embeddings "
                f"WHERE source_table = '{self.source_table}' "
                f"AND model_name = '{self.model_name}' "
                f"ORDER BY record_id "
                f"OFFSET {existing_count}"
            )
            if df.height == 0:
                return None

            ids = df["record_id"].to_list()
            embeddings = np.array([
                np.frombuffer(b, dtype=np.float32) for b in df["embedding"].to_list()
            ])
            return ids, embeddings
        except Exception as e:
            logger.warning("Failed to load delta embeddings: %s", e)
            return None

    def _rebuild_from_db(self) -> None:
        """Full rebuild of FAISS index from gm_embeddings."""
        import faiss

        logger.info("Rebuilding ANN index from gm_embeddings...")

        df = self.connector.read_query(
            f"SELECT record_id, embedding FROM gm_embeddings "
            f"WHERE source_table = '{self.source_table}' "
            f"AND model_name = '{self.model_name}' "
            f"ORDER BY record_id"
        )

        if df.height == 0:
            return

        ids = df["record_id"].to_list()
        embeddings = np.array([
            np.frombuffer(b, dtype=np.float32) for b in df["embedding"].to_list()
        ])

        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._id_map = ids
        self._id_to_pos = {rid: i for i, rid in enumerate(ids)}

        logger.info("Rebuilt ANN index with %d embeddings (dim=%d)", len(ids), self._dim)

    # ── Query ─────────────────────────────────────────────────────────

    def query(self, embeddings: np.ndarray, top_k: int = 20) -> list[tuple[int, int, float]]:
        """Find top-K neighbors. Returns (query_idx, db_record_id, score)."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(embeddings.astype(np.float32), k)

        results = []
        for query_idx in range(len(embeddings)):
            for j in range(k):
                faiss_idx = int(indices[query_idx][j])
                if faiss_idx < 0 or faiss_idx >= len(self._id_map):
                    continue
                db_id = self._id_map[faiss_idx]
                score = float(scores[query_idx][j])
                results.append((query_idx, db_id, score))

        return results

    # ── Add ───────────────────────────────────────────────────────────

    def add(self, record_ids: list[int], embeddings: np.ndarray) -> None:
        """Add new embeddings to index and store in gm_embeddings."""
        import faiss

        if len(record_ids) == 0:
            return

        emb = embeddings.astype(np.float32)

        if self._index is None:
            self._dim = emb.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)

        # Add to FAISS
        self._index.add(emb)

        # Update id_map
        for rid in record_ids:
            pos = len(self._id_map)
            self._id_map.append(rid)
            self._id_to_pos[rid] = pos

        # Store in DB
        if self.connector is not None:
            self._store_embeddings_in_db(record_ids, emb)

    def _store_embeddings_in_db(self, record_ids: list[int], embeddings: np.ndarray) -> None:
        """Batch insert embeddings into gm_embeddings."""
        cursor = self.connector.conn.cursor()
        try:
            for rid, emb in zip(record_ids, embeddings):
                cursor.execute(
                    "INSERT INTO gm_embeddings (record_id, source_table, embedding, model_name) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (record_id, source_table, model_name) DO NOTHING",
                    (rid, self.source_table, emb.tobytes(), self.model_name),
                )
            self.connector.conn.commit()
        except Exception:
            self.connector.conn.rollback()
            raise
        finally:
            cursor.close()

    # ── Save ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index to disk."""
        if self._index is None or len(self._id_map) == 0:
            return

        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        np.save(str(self.index_dir / "id_map.npy"), np.array(self._id_map))

        meta = {
            "record_count": len(self._id_map),
            "dim": self._dim,
            "model": self.model_name,
            "source_table": self.source_table,
        }
        with open(self.index_dir / "index_meta.json", "w") as f:
            json.dump(meta, f)

        logger.info("Saved ANN index (%d embeddings) to %s", len(self._id_map), self.index_dir)
