"""High-performance persistent SQLite storage engine with FTS5 full-text indexing.

Provides atomic transactions, indexed lookups for financial facts,
and fast lexical search for chunks without rewriting JSON files on every update.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .models import Chunk, Document, FinancialFact


class FinancialSQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        with self.lock, self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name TEXT,
                ticker TEXT,
                year INTEGER,
                scope TEXT DEFAULT 'unknown',
                reporting_period TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                ticker TEXT,
                year INTEGER,
                scope TEXT DEFAULT 'unknown',
                bbox_json TEXT,
                extraction_method TEXT DEFAULT 'text',
                financial_rows_json TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (document_id, page);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                page UNINDEXED,
                text,
                tokenize = 'unicode61'
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_id TEXT NOT NULL,
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                metric_code TEXT,
                value REAL NOT NULL,
                raw_value TEXT NOT NULL,
                unit TEXT,
                period INTEGER,
                period_label TEXT,
                period_kind TEXT,
                scope TEXT DEFAULT 'unknown',
                statement TEXT,
                confidence REAL NOT NULL,
                quality TEXT DEFAULT 'unverified',
                page INTEGER NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_facts_metric_code ON facts (document_id, statement, metric_code);
            CREATE INDEX IF NOT EXISTS idx_facts_period ON facts (document_id, period_label);

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_pages INTEGER DEFAULT 0,
                total_pages INTEGER,
                result_json TEXT,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC);
            """)

    def save_document(self, doc: Document, metadata: dict[str, Any] | None = None):
        now = datetime.now(UTC).isoformat()
        with self.lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, source, name, ticker, year, scope, reporting_period, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.source,
                    doc.name,
                    doc.ticker,
                    doc.year,
                    doc.scope,
                    doc.reporting_period,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )

    def save_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        with self.lock, self._get_connection() as conn:
            doc_ids = {c.document_id for c in chunks}
            # Remove old chunks for these documents
            for doc_id in doc_ids:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))

            chunk_records = [
                (
                    c.id,
                    c.document_id,
                    c.page,
                    c.text,
                    c.ticker,
                    c.year,
                    c.scope,
                    json.dumps(c.bbox) if c.bbox else None,
                    c.extraction_method,
                    json.dumps(c.financial_rows) if c.financial_rows else None,
                )
                for c in chunks
            ]
            conn.executemany(
                """
                INSERT INTO chunks (id, document_id, page, text, ticker, year, scope, bbox_json, extraction_method, financial_rows_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_records,
            )

            fts_records = [(c.id, c.document_id, c.page, c.text) for c in chunks]
            conn.executemany(
                """
                INSERT INTO chunks_fts (chunk_id, document_id, page, text)
                VALUES (?, ?, ?, ?)
                """,
                fts_records,
            )

    def save_facts(self, facts: list[FinancialFact]):
        if not facts:
            return
        with self.lock, self._get_connection() as conn:
            doc_ids = {f.document_id for f in facts if f.document_id}
            for doc_id in doc_ids:
                conn.execute("DELETE FROM facts WHERE document_id = ?", (doc_id,))

            fact_records = [
                (
                    f.id,
                    f.document_id,
                    f.chunk_id,
                    f.source,
                    f.metric,
                    f.metric_code,
                    f.value,
                    f.raw_value,
                    f.unit,
                    f.period,
                    f.period_label,
                    f.period_kind,
                    f.scope,
                    f.statement,
                    f.confidence,
                    f.quality,
                    f.page,
                )
                for f in facts
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO facts (id, document_id, chunk_id, source, metric, metric_code, value, raw_value, unit,
                                              period, period_label, period_kind, scope, statement, confidence, quality, page)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fact_records,
            )

    def search_chunks_fts(self, query: str, document_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        with self.lock, self._get_connection() as conn:
            cleaned = " ".join(t for t in query.replace('"', " ").replace("'", " ").split() if len(t) > 0)
            if not cleaned:
                return []
            # Exact phrase query has highest relevance, fallback to individual terms
            fts_query = f'"{cleaned}"'

            def execute_query(q_str):
                if document_id:
                    return conn.execute(
                        """
                        SELECT c.id, c.document_id, c.page, c.text, c.bbox_json, c.extraction_method, bm25(chunks_fts) as rank
                        FROM chunks_fts f
                        JOIN chunks c ON c.id = f.chunk_id
                        WHERE chunks_fts MATCH ? AND f.document_id = ?
                        ORDER BY rank ASC
                        LIMIT ?
                        """,
                        (q_str, document_id, limit),
                    ).fetchall()
                else:
                    return conn.execute(
                        """
                        SELECT c.id, c.document_id, c.page, c.text, c.bbox_json, c.extraction_method, bm25(chunks_fts) as rank
                        FROM chunks_fts f
                        JOIN chunks c ON c.id = f.chunk_id
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT ?
                        """,
                        (q_str, limit),
                    ).fetchall()

            try:
                rows = execute_query(fts_query)
            except sqlite3.OperationalError:
                rows = []

            # If phrase query yielded no results, try AND/OR terms
            if not rows:
                terms = [t for t in cleaned.split() if len(t) > 1]
                if terms:
                    fts_query_or = " OR ".join(f'"{t}"' for t in terms)
                    try:
                        rows = execute_query(fts_query_or)
                    except sqlite3.OperationalError:
                        rows = []
            results = []
            for row in rows:
                results.append({
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "text": row["text"],
                    "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                    "extraction_method": row["extraction_method"],
                    "score": abs(float(row["rank"])),
                })
            return results

    def get_facts_by_code(self, document_id: str, statement: str, metric_code: str) -> list[FinancialFact]:
        with self.lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM facts
                WHERE document_id = ? AND statement = ? AND metric_code = ?
                ORDER BY period_label DESC
                """,
                (document_id, statement, metric_code),
            )
            return [FinancialFact.model_validate(dict(row)) for row in cursor.fetchall()]

    def list_documents(self) -> list[dict[str, Any]]:
        with self.lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
