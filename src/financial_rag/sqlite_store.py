"""High-performance persistent SQLite storage engine with FTS5 full-text indexing.

Provides atomic transactions, indexed lookups for financial facts,
persistent job tracking, human-in-the-loop verification updates,
and fast lexical search for chunks without rewriting JSON files on every update.
"""

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .models import Chunk, Document, FinancialFact, TableCell


class FinancialSQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self._init_db()

    @contextmanager
    def _connection(self):
        with self.lock, closing(sqlite3.connect(str(self.db_path), check_same_thread=False)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            with conn:
                yield conn

    def _init_db(self):
        with self.lock, closing(sqlite3.connect(str(self.db_path), check_same_thread=False)) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            with conn:
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
                    source TEXT DEFAULT '',
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    ticker TEXT,
                    year INTEGER,
                    scope TEXT DEFAULT 'unknown',
                    bbox_json TEXT,
                    extraction_method TEXT DEFAULT 'text',
                    ocr_engine TEXT,
                    table_json TEXT,
                    table_unit TEXT,
                    table_structure_engine TEXT,
                    table_cells_json TEXT,
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
                    ticker TEXT,
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
                    bbox_json TEXT,
                    extraction_method TEXT DEFAULT 'text',
                    confidence REAL NOT NULL,
                    quality TEXT DEFAULT 'unverified',
                    recognition_note TEXT,
                    page INTEGER NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts (document_id);
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
                self._ensure_columns(
                    conn,
                    "chunks",
                    {
                        "source": "TEXT DEFAULT ''",
                        "ocr_engine": "TEXT",
                        "table_json": "TEXT",
                        "table_unit": "TEXT",
                        "table_structure_engine": "TEXT",
                        "table_cells_json": "TEXT",
                    },
                )
                self._ensure_columns(
                    conn,
                    "facts",
                    {
                        "ticker": "TEXT",
                        "bbox_json": "TEXT",
                        "extraction_method": "TEXT DEFAULT 'text'",
                        "recognition_note": "TEXT",
                    },
                )

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, required: dict[str, str]):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, definition in required.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")

    def save_document(self, doc: Document, metadata: dict[str, Any] | None = None):
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source, name, ticker, year, scope, reporting_period, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source=excluded.source,
                    name=excluded.name,
                    ticker=excluded.ticker,
                    year=excluded.year,
                    scope=excluded.scope,
                    reporting_period=excluded.reporting_period,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
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
        with self._connection() as conn:
            now = datetime.now(UTC).isoformat()
            doc_stubs: dict[str, Chunk] = {}
            for chunk in chunks:
                doc_stubs.setdefault(chunk.document_id, chunk)
            for doc_id, sample in doc_stubs.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO documents (id, source, name, ticker, year, scope, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        sample.source,
                        Path(sample.source).name if sample.source else doc_id,
                        sample.ticker,
                        sample.year,
                        sample.scope,
                        now,
                    ),
                )
                conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

            chunk_records = [
                (
                    c.id,
                    c.document_id,
                    c.source,
                    c.page,
                    c.text,
                    c.ticker,
                    c.year,
                    c.scope,
                    json.dumps(c.bbox) if c.bbox is not None else None,
                    c.extraction_method,
                    c.ocr_engine,
                    json.dumps(c.table, ensure_ascii=False) if c.table is not None else None,
                    c.table_unit,
                    c.table_structure_engine,
                    json.dumps([cell.model_dump() for cell in c.table_cells], ensure_ascii=False)
                    if c.table_cells
                    else None,
                    json.dumps(c.financial_rows, ensure_ascii=False) if c.financial_rows else None,
                )
                for c in chunks
            ]
            conn.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, source, page, text, ticker, year, scope, bbox_json,
                    extraction_method, ocr_engine, table_json, table_unit,
                    table_structure_engine, table_cells_json, financial_rows_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        with self._connection() as conn:
            doc_ids = {f.document_id for f in facts if f.document_id}
            for doc_id in doc_ids:
                conn.execute("DELETE FROM facts WHERE document_id = ?", (doc_id,))

            fact_records = [
                (
                    f.id,
                    f.document_id,
                    f.chunk_id,
                    f.source,
                    f.ticker,
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
                    json.dumps(f.bbox) if f.bbox is not None else None,
                    f.extraction_method,
                    f.confidence,
                    f.quality,
                    f.recognition_note,
                    f.page,
                )
                for f in facts
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO facts (
                    id, document_id, chunk_id, source, ticker, metric, metric_code,
                    value, raw_value, unit, period, period_label, period_kind,
                    scope, statement, bbox_json, extraction_method, confidence,
                    quality, recognition_note, page
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fact_records,
            )

    def load_chunks(self, document_id: str | None = None) -> list[Chunk]:
        with self._connection() as conn:
            if document_id:
                rows = conn.execute(
                    """
                    SELECT c.*, d.source AS doc_source
                    FROM chunks c
                    LEFT JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = ?
                    ORDER BY c.page, c.rowid
                    """,
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*, d.source AS doc_source
                    FROM chunks c
                    LEFT JOIN documents d ON d.id = c.document_id
                    ORDER BY c.document_id, c.page, c.rowid
                    """
                ).fetchall()
            result = []
            for row in rows:
                cells_raw = json.loads(row["table_cells_json"]) if row["table_cells_json"] else []
                result.append(
                    Chunk(
                        id=row["id"],
                        document_id=row["document_id"],
                        source=row["source"] or row["doc_source"] or "",
                        page=row["page"],
                        text=row["text"],
                        ticker=row["ticker"],
                        year=row["year"],
                        table=json.loads(row["table_json"]) if row["table_json"] else None,
                        table_unit=row["table_unit"],
                        scope=row["scope"] or "unknown",
                        bbox=json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                        extraction_method=row["extraction_method"] or "text",
                        financial_rows=json.loads(row["financial_rows_json"])
                        if row["financial_rows_json"]
                        else [],
                        table_structure_engine=row["table_structure_engine"],
                        table_cells=[TableCell.model_validate(item) for item in cells_raw],
                        ocr_engine=row["ocr_engine"],
                    )
                )
            return result

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> FinancialFact:
        data = dict(row)
        bbox_raw = data.pop("bbox_json", None)
        data["bbox"] = json.loads(bbox_raw) if bbox_raw else None
        return FinancialFact.model_validate(data)

    def load_facts(self, document_id: str | None = None) -> list[FinancialFact]:
        with self._connection() as conn:
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE document_id = ? ORDER BY page, rowid",
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM facts ORDER BY document_id, page, rowid").fetchall()
            return [self._row_to_fact(row) for row in rows]

    def update_fact(
        self,
        fact_id: str,
        *,
        value: float | None = None,
        raw_value: str | None = None,
        quality: str | None = None,
        recognition_note: str | None = None,
    ) -> FinancialFact | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                return None
            new_value = float(value) if value is not None else row["value"]
            new_raw = raw_value if raw_value is not None else row["raw_value"]
            new_quality = quality if quality is not None else row["quality"]
            new_note = recognition_note if recognition_note is not None else row["recognition_note"]
            conn.execute(
                """
                UPDATE facts
                SET value = ?, raw_value = ?, quality = ?, recognition_note = ?
                WHERE id = ?
                """,
                (new_value, new_raw, new_quality, new_note, fact_id),
            )
            updated = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            return self._row_to_fact(updated) if updated else None

    def get_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            return {
                "id": row["id"],
                "source": row["source"],
                "name": row["name"],
                "ticker": row["ticker"],
                "year": row["year"],
                "scope": row["scope"],
                "reporting_period": row["reporting_period"],
                **meta,
            }

    def delete_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            source = row["source"]
            conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            shared = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE source = ?", (source,)
            ).fetchone()[0]
            return {"id": document_id, "source": source, "source_shared": shared > 0}

    def search_chunks_fts(
        self, query: str, document_id: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return []
        cleaned = " ".join(tokens)
        fts_query = f'"{cleaned}"'

        with self._connection() as conn:

            def execute_query(q_str: str):
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

            if not rows:
                terms = [t for t in tokens if len(t) > 1] or tokens
                fts_query_or = " OR ".join(f'"{t}"' for t in terms)
                try:
                    rows = execute_query(fts_query_or)
                except sqlite3.OperationalError:
                    rows = []
            return [
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "text": row["text"],
                    "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                    "extraction_method": row["extraction_method"],
                    "score": abs(float(row["rank"])),
                }
                for row in rows
            ]

    def get_facts_by_code(
        self, document_id: str, statement: str, metric_code: str
    ) -> list[FinancialFact]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM facts
                WHERE document_id = ? AND statement = ? AND metric_code = ?
                ORDER BY period_label DESC
                """,
                (document_id, statement, metric_code),
            ).fetchall()
            return [self._row_to_fact(row) for row in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    # Persistent ingestion jobs
    def interrupt_stale_jobs(self):
        with self._connection() as conn:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    error = 'Server khởi động lại; hãy nạp lại tài liệu. Trang OCR đã lưu đệm được tái sử dụng.',
                    updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )

    def create_job(self, job_id: str, name: str) -> dict[str, Any]:
        with self._connection() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]
            if active >= 8:
                raise ValueError("Hàng đợi đã đủ 8 tài liệu; chờ xử lý xong rồi thử lại.")
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT INTO jobs (id, name, status, created_at, updated_at, completed_pages)
                VALUES (?, ?, 'queued', ?, ?, 0)
                """,
                (job_id, name, now, now),
            )
            # Retain the latest 200 finished jobs.
            conn.execute(
                """
                DELETE FROM jobs
                WHERE status NOT IN ('queued', 'running')
                  AND id NOT IN (SELECT id FROM jobs ORDER BY created_at DESC LIMIT 200)
                """
            )
            return self._row_to_job(
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )

    def update_job(self, job_id: str, **values):
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return
            status = values.get("status", row["status"])
            completed_pages = values.get("completed_pages", row["completed_pages"])
            total_pages = values.get("total_pages", row["total_pages"])
            result_json = (
                json.dumps(values["result"], ensure_ascii=False)
                if "result" in values
                else row["result_json"]
            )
            error = values.get("error", row["error"])
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, completed_pages = ?, total_pages = ?,
                    result_json = ?, error = ?
                WHERE id = ?
                """,
                (status, now, completed_pages, total_pages, result_json, error, job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        item = {
            "id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_pages": row["completed_pages"] or 0,
            "total_pages": row["total_pages"],
            "error": row["error"],
        }
        if row["result_json"]:
            item["result"] = json.loads(row["result_json"])
        return {k: v for k, v in item.items() if v is not None or k in {"id", "name", "status", "created_at", "updated_at"}}
