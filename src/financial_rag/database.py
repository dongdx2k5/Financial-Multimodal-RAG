import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .models import Chunk, Document, FinancialFact

logger = logging.getLogger(__name__)


class FinancialDatabase:
    """Persistent SQLite store for financial documents, chunks, facts, and vectors."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._has_sqlite_vec = False
        self._init_db()

    @property
    def has_sqlite_vec(self) -> bool:
        return self._has_sqlite_vec

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._has_sqlite_vec = True
        except Exception as exc:
            logger.debug("sqlite-vec extension loading notice: %s", exc)
            self._has_sqlite_vec = False
        return conn

    def _ensure_vec_table(self, conn: sqlite3.Connection, dim: int) -> bool:
        if not self.has_sqlite_vec:
            return False
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"chunk_id text primary key, embedding float[{dim}]);"
            )
            return True
        except Exception as exc:
            logger.debug("vec0 table creation notice: %s", exc)
            return False

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    ticker TEXT,
                    year INTEGER,
                    pages_count INTEGER DEFAULT 0,
                    chunks_count INTEGER DEFAULT 0,
                    facts_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    ticker TEXT,
                    year INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                );
            """)

            # Facts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    standardized_metric TEXT,
                    value REAL NOT NULL,
                    raw_value TEXT NOT NULL,
                    unit TEXT,
                    period INTEGER,
                    ticker TEXT,
                    confidence REAL NOT NULL,
                    is_chart INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
                );
            """)

            # Migration: add standardized_metric if upgrading existing DB
            try:
                cursor.execute("ALTER TABLE facts ADD COLUMN standardized_metric TEXT;")
            except Exception as exc:
                logger.debug("Migration standardized_metric notice: %s", exc)

            # Migration: add is_chart if upgrading existing DB
            try:
                cursor.execute("ALTER TABLE facts ADD COLUMN is_chart INTEGER DEFAULT 0;")
            except Exception as exc:
                logger.debug("Migration is_chart notice: %s", exc)

            # Indexes for relational queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (document_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON chunks (ticker);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts (document_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_ticker_period ON facts (ticker, period);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts (metric);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_std_metric ON facts (standardized_metric);")

            # Full-text search index (FTS5) for keyword/hybrid search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    text,
                    ticker,
                    tokenize='unicode61'
                );
            """)

            # Vector embeddings relational storage (Phase 3.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_id ON chunk_embeddings (chunk_id);"
            )

            # Re-initialize vec_chunks virtual table if embeddings already exist
            row = cursor.execute("SELECT dim FROM chunk_embeddings LIMIT 1;").fetchone()
            if row:
                self._ensure_vec_table(conn, int(row["dim"]))

            conn.commit()

    def save_document(
        self, document: Document, chunks: list[Chunk], facts: list[FinancialFact]
    ) -> None:
        filename = Path(document.source).name
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Upsert document
            cursor.execute(
                """
                INSERT INTO documents (id, source, filename, ticker, year, pages_count, chunks_count, facts_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source=excluded.source,
                    filename=excluded.filename,
                    ticker=excluded.ticker,
                    year=excluded.year,
                    pages_count=excluded.pages_count,
                    chunks_count=excluded.chunks_count,
                    facts_count=excluded.facts_count,
                    created_at=CURRENT_TIMESTAMP;
                """,
                (
                    document.id,
                    document.source,
                    filename,
                    document.ticker,
                    document.year,
                    len(document.pages),
                    len(chunks),
                    len(facts),
                ),
            )

            # Clear previous chunks & embeddings for this document if re-ingested
            cursor.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?);",
                (document.id,),
            )
            if self.has_sqlite_vec:
                try:
                    chunk_rows = cursor.execute(
                        "SELECT id FROM chunks WHERE document_id = ?;", (document.id,)
                    ).fetchall()
                    for r in chunk_rows:
                        cursor.execute("DELETE FROM vec_chunks WHERE chunk_id = ?;", (r["id"],))
                except Exception as exc:
                    logger.debug("vec_chunks cleanup notice: %s", exc)
            cursor.execute("DELETE FROM chunks WHERE document_id = ?;", (document.id,))
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?;", (document.id,))

            # Batch insert chunks
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (id, document_id, page, text, ticker, year)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (chunk.id, chunk.document_id, chunk.page, chunk.text, chunk.ticker, chunk.year),
                )
                cursor.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, document_id, text, ticker)
                    VALUES (?, ?, ?, ?);
                    """,
                    (chunk.id, chunk.document_id, chunk.text, chunk.ticker or ""),
                )

            # Batch insert facts
            for fact in facts:
                cursor.execute(
                    """
                    INSERT INTO facts (id, document_id, chunk_id, metric, standardized_metric, value, raw_value, unit, period, ticker, confidence, is_chart)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        metric=excluded.metric,
                        standardized_metric=excluded.standardized_metric,
                        value=excluded.value,
                        raw_value=excluded.raw_value,
                        unit=excluded.unit,
                        period=excluded.period,
                        ticker=excluded.ticker,
                        confidence=excluded.confidence,
                        is_chart=excluded.is_chart;
                    """,
                    (
                        fact.id,
                        document.id,
                        fact.chunk_id,
                        fact.metric,
                        fact.standardized_metric,
                        fact.value,
                        fact.raw_value,
                        fact.unit,
                        fact.period,
                        fact.ticker,
                        fact.confidence,
                        1 if fact.is_chart else 0,
                    ),
                )
            conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, filename, ticker, year, pages_count, chunks_count, facts_count, created_at
                FROM documents
                ORDER BY created_at DESC;
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?;", (document_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?);",
                (document_id,),
            )
            if self.has_sqlite_vec:
                try:
                    chunk_rows = cursor.execute(
                        "SELECT id FROM chunks WHERE document_id = ?;", (document_id,)
                    ).fetchall()
                    for r in chunk_rows:
                        cursor.execute("DELETE FROM vec_chunks WHERE chunk_id = ?;", (r["id"],))
                except Exception as exc:
                    logger.debug("vec_chunks cleanup notice: %s", exc)

            cursor.execute("DELETE FROM documents WHERE id = ?;", (document_id,))
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?;", (document_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Vector storage and search (Phase 3.2: SQLite-vec)
    # ------------------------------------------------------------------

    def save_chunk_embeddings(
        self, chunk_ids: list[str], embeddings: np.ndarray
    ) -> None:
        """Save chunk vector embeddings directly into SQLite and sqlite-vec (Phase 3.2)."""
        if len(chunk_ids) == 0 or len(embeddings) == 0:
            return
        dim = int(embeddings.shape[1])
        with self._get_connection() as conn:
            vec_ready = self._ensure_vec_table(conn, dim)
            cursor = conn.cursor()

            # Normalize embeddings to unit length for cosine distance
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            norm_embeddings = (embeddings / norms).astype(np.float32)

            for chunk_id, norm_vec, orig_vec in zip(chunk_ids, norm_embeddings, embeddings):
                blob = orig_vec.astype(np.float32).tobytes()
                cursor.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, embedding, dim)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        embedding=excluded.embedding,
                        dim=excluded.dim;
                    """,
                    (chunk_id, blob, dim),
                )
                if vec_ready:
                    try:
                        cursor.execute("DELETE FROM vec_chunks WHERE chunk_id = ?;", (chunk_id,))
                        cursor.execute(
                            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?);",
                            (chunk_id, norm_vec.tobytes()),
                        )
                    except Exception as exc:
                        logger.debug("vec_chunks insert notice: %s", exc)
            conn.commit()

    def get_chunk_embeddings(
        self, chunk_ids: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Load stored chunk embeddings from SQLite (Phase 3.2)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if chunk_ids:
                placeholders = ",".join(["?"] * len(chunk_ids))
                cursor.execute(
                    f"SELECT chunk_id, embedding, dim FROM chunk_embeddings WHERE chunk_id IN ({placeholders});",
                    chunk_ids,
                )
            else:
                cursor.execute("SELECT chunk_id, embedding, dim FROM chunk_embeddings;")
            result: dict[str, np.ndarray] = {}
            for row in cursor.fetchall():
                dim = row["dim"]
                arr = np.frombuffer(row["embedding"], dtype=np.float32)
                if len(arr) == dim:
                    result[row["chunk_id"]] = arr
            return result

    def search_vector(
        self, query_vector: np.ndarray, limit: int = 20
    ) -> list[tuple[str, float]]:
        """KNN vector search using sqlite-vec, with fallback to numpy cosine similarity (Phase 3.2)."""
        if len(query_vector) == 0:
            return []
        dim = len(query_vector)
        q_norm = np.linalg.norm(query_vector)
        q_unit = (query_vector / (q_norm if q_norm != 0 else 1.0)).astype(np.float32)

        with self._get_connection() as conn:
            if self.has_sqlite_vec:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT chunk_id, distance
                        FROM vec_chunks
                        WHERE embedding MATCH ? AND k = ?;
                        """,
                        (q_unit.tobytes(), limit),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        results: list[tuple[str, float]] = []
                        for row in rows:
                            dist = float(row["distance"])
                            score = max(0.0, 1.0 - (dist**2) / 2.0)
                            results.append((row["chunk_id"], score))
                        return results
                except Exception as exc:
                    logger.debug("sqlite-vec search notice: %s", exc)

            # Fallback to in-database chunk_embeddings + numpy cosine similarity
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id, embedding, dim FROM chunk_embeddings WHERE dim = ?;",
                (dim,),
            )
            rows = cursor.fetchall()
            if not rows:
                return []
            ids = [r["chunk_id"] for r in rows]
            matrix = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            scores = (matrix / norms) @ q_unit
            top_indices = np.argsort(scores)[::-1][:limit]
            return [(ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_hybrid_db(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Unified query combining SQL FTS5 (BM25) and sqlite-vec (KNN) directly in SQLite (Phase 3.2)."""
        fts_results = self.search_fts(query, limit=limit)
        if query_vector is None or len(query_vector) == 0:
            return fts_results

        vec_matches = dict(self.search_vector(query_vector, limit=limit))
        combined: dict[str, dict[str, Any]] = {}

        for r in fts_results:
            cid = r["id"]
            combined[cid] = {
                **r,
                "fts_rank": float(r.get("rank", 0.0)),
                "vec_score": vec_matches.get(cid, 0.0),
            }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for cid, v_score in vec_matches.items():
                if cid not in combined:
                    row = cursor.execute(
                        """
                        SELECT c.id, c.document_id, c.page, c.text, c.ticker, c.year, d.source
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.id = ?;
                        """,
                        (cid,),
                    ).fetchone()
                    if row:
                        combined[cid] = {
                            **dict(row),
                            "fts_rank": 0.0,
                            "vec_score": v_score,
                        }

        results = list(combined.values())
        results.sort(
            key=lambda x: x.get("vec_score", 0.0) + (1.0 / (abs(x.get("fts_rank", 0.0)) + 1.0)),
            reverse=True,
        )
        return results[:limit]

    def get_all_chunks(self) -> list[Chunk]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.document_id, c.page, c.text, c.ticker, c.year, d.source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id;
            """)
            return [
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    source=row["source"],
                    page=row["page"],
                    text=row["text"],
                    ticker=row["ticker"],
                    year=row["year"],
                )
                for row in cursor.fetchall()
            ]

    def get_facts(
        self,
        ticker: str | None = None,
        year: int | None = None,
        metric: str | None = None,
        standardized_metric: str | None = None,
        is_chart: bool | None = None,
        limit: int = 100,
    ) -> list[FinancialFact]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT f.*, d.source, c.page
                FROM facts f
                JOIN documents d ON f.document_id = d.id
                JOIN chunks c ON f.chunk_id = c.id
                WHERE 1=1
            """
            params: list[Any] = []
            if ticker:
                query += " AND f.ticker = ?"
                params.append(ticker.upper())
            if year:
                query += " AND f.period = ?"
                params.append(year)
            if metric:
                query += " AND f.metric LIKE ?"
                params.append(f"%{metric}%")
            if standardized_metric:
                query += " AND f.standardized_metric = ?"
                params.append(standardized_metric)
            if is_chart is not None:
                query += " AND f.is_chart = ?"
                params.append(1 if is_chart else 0)
            query += " ORDER BY f.period DESC, f.created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [
                FinancialFact(
                    id=row["id"],
                    metric=row["metric"],
                    standardized_metric=row["standardized_metric"],
                    value=row["value"],
                    raw_value=row["raw_value"],
                    unit=row["unit"],
                    period=row["period"],
                    ticker=row["ticker"],
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    page=row["page"],
                    confidence=row["confidence"],
                    is_chart=bool(row["is_chart"]) if ("is_chart" in row.keys() and row["is_chart"]) else False,  # noqa: SIM118
                )
                for row in cursor.fetchall()
            ]

    def search_fts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        clean_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
        if not clean_query:
            return []
        match_terms = " OR ".join(clean_query.split())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.id, c.document_id, c.page, c.text, c.ticker, c.year, d.source, rank
                FROM chunks_fts fts
                JOIN chunks c ON fts.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?;
                """,
                (match_terms, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            doc_count = cursor.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
            chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
            fact_count = cursor.execute("SELECT COUNT(*) FROM facts;").fetchone()[0]
            file_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "db_path": str(self.db_path),
                "documents": doc_count,
                "chunks": chunk_count,
                "facts": fact_count,
                "size_kb": round(file_size_bytes / 1024, 2),
            }
