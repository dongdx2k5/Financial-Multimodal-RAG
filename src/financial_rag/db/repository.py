"""Data access repository for Financial RAG SQLite database."""

import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from financial_rag.db.schema import get_connection


class FinancialRepository:
    """Repository managing all SQLite queries and data transactions."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self):
        conn = get_connection(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # -------------------------------------------------------------
    # Company operations
    # -------------------------------------------------------------
    def add_company(
        self,
        ticker: str,
        name_vi: Optional[str] = None,
        name_en: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
    ) -> int:
        ticker = ticker.strip().upper()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO company (ticker, name_vi, name_en, industry, country)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name_vi=excluded.name_vi,
                    name_en=excluded.name_en,
                    industry=excluded.industry,
                    country=excluded.country
                RETURNING company_id;
                """,
                (ticker, name_vi, name_en, industry, country),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["company_id"])

    def get_company_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM company WHERE ticker = ?;", (ticker.strip().upper(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_companies(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM company ORDER BY ticker;")
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------
    # Report operations
    # -------------------------------------------------------------
    def add_report(
        self,
        company_id: int,
        fiscal_year: int,
        doc_type: str = "BCTC",
        period_type: str = "FY",
        period_end: Optional[str] = None,
        scope: str = "consolidated",
        language: str = "vi",
        currency: str = "VND",
        unit_scale: int = 1,
        audit_status: str = "audited",
        file_path: Optional[str] = None,
        source_url: Optional[str] = None,
        is_scanned: bool = False,
        needs_manual_check: bool = False,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO report (
                    company_id, doc_type, period_type, fiscal_year, period_end,
                    scope, language, currency, unit_scale, audit_status,
                    file_path, source_url, is_scanned, needs_manual_check
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING report_id;
                """,
                (
                    company_id,
                    doc_type,
                    period_type,
                    fiscal_year,
                    period_end,
                    scope,
                    language,
                    currency,
                    unit_scale,
                    audit_status,
                    file_path,
                    source_url,
                    1 if is_scanned else 0,
                    1 if needs_manual_check else 0,
                ),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["report_id"])

    def get_report_by_id(self, report_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT r.*, c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en, c.industry
                FROM report r
                JOIN company c ON r.company_id = c.company_id
                WHERE r.report_id = ?;
                """,
                (report_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def find_reports(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT r.*, c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en, c.industry
            FROM report r
            JOIN company c ON r.company_id = c.company_id
            WHERE 1=1
        """
        params: List[Any] = []
        if ticker:
            query += " AND c.ticker = ?"
            params.append(ticker.strip().upper())
        if fiscal_year:
            query += " AND r.fiscal_year = ?"
            params.append(fiscal_year)
        if period_type:
            query += " AND UPPER(r.period_type) = ?"
            params.append(period_type.strip().upper())
        query += " ORDER BY r.fiscal_year DESC, r.period_type ASC;"

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------
    # Metric operations
    # -------------------------------------------------------------
    def add_metric(
        self,
        code: str,
        name_vi: str,
        name_en: str,
        statement: str = "IS",
        is_common: bool = True,
        aliases: Optional[str] = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO metric (code, name_vi, name_en, statement, is_common, aliases)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name_vi=excluded.name_vi,
                    name_en=excluded.name_en,
                    statement=excluded.statement,
                    is_common=excluded.is_common,
                    aliases=excluded.aliases
                RETURNING metric_id;
                """,
                (code.strip(), name_vi.strip(), name_en.strip(), statement.strip().upper(), 1 if is_common else 0, aliases),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["metric_id"])

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM metric ORDER BY statement, is_common DESC, code;")
            return [dict(r) for r in cursor.fetchall()]

    def get_metric_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM metric WHERE code = ?;", (code.strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def match_metric_alias(self, text: str) -> Optional[Dict[str, Any]]:
        """Match input text against metric names and aliases."""
        clean_text = text.lower().strip()
        metrics = self.get_all_metrics()
        # Direct exact match on code, name_vi, name_en
        for m in metrics:
            if clean_text == m["code"].lower() or clean_text == m["name_vi"].lower() or clean_text == m["name_en"].lower():
                return m

        # Alias exact or partial match
        for m in metrics:
            aliases = [a.strip().lower() for a in (m["aliases"] or "").split("|") if a.strip()]
            for alias in aliases:
                if alias == clean_text or alias in clean_text or clean_text in alias:
                    return m
        return None

    # -------------------------------------------------------------
    # Fact operations
    # -------------------------------------------------------------
    def add_fact(
        self,
        report_id: int,
        line_item: str,
        column_role: str,
        value: Optional[float],
        value_raw: Optional[str] = None,
        metric_id: Optional[int] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        page: Optional[int] = None,
        table_ref: Optional[str] = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fact (
                    report_id, metric_id, line_item, column_role, period_start, period_end,
                    value, value_raw, page, table_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id, line_item, column_role) DO UPDATE SET
                    metric_id=excluded.metric_id,
                    value=excluded.value,
                    value_raw=excluded.value_raw,
                    page=excluded.page,
                    table_ref=excluded.table_ref
                RETURNING fact_id;
                """,
                (
                    report_id,
                    metric_id,
                    line_item.strip(),
                    column_role.strip().lower(),
                    period_start,
                    period_end,
                    value,
                    value_raw,
                    page,
                    table_ref,
                ),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["fact_id"])

    def add_facts_batch(self, facts: List[Dict[str, Any]]) -> int:
        count = 0
        with self._conn() as conn:
            for f in facts:
                conn.execute(
                    """
                    INSERT INTO fact (
                        report_id, metric_id, line_item, column_role, period_start, period_end,
                        value, value_raw, page, table_ref
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(report_id, line_item, column_role) DO UPDATE SET
                        metric_id=excluded.metric_id,
                        value=excluded.value,
                        value_raw=excluded.value_raw,
                        page=excluded.page,
                        table_ref=excluded.table_ref;
                    """,
                    (
                        f["report_id"],
                        f.get("metric_id"),
                        f["line_item"].strip(),
                        f.get("column_role", "current").strip().lower(),
                        f.get("period_start"),
                        f.get("period_end"),
                        f.get("value"),
                        f.get("value_raw"),
                        f.get("page"),
                        f.get("table_ref"),
                    ),
                )
                count += 1
            conn.commit()
        return count

    def query_facts(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_type: Optional[str] = None,
        metric_code: Optional[str] = None,
        line_item: Optional[str] = None,
        column_role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query facts matching specified criteria, joining report, company, and metric."""
        query = """
            SELECT
                f.fact_id, f.line_item, f.column_role, f.value, f.value_raw, f.page, f.table_ref,
                m.code as metric_code, m.name_vi as metric_name_vi, m.name_en as metric_name_en, m.statement,
                r.report_id, r.fiscal_year, r.period_type, r.doc_type, r.currency, r.unit_scale, r.scope,
                c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en, c.industry
            FROM fact f
            JOIN report r ON f.report_id = r.report_id
            JOIN company c ON r.company_id = c.company_id
            LEFT JOIN metric m ON f.metric_id = m.metric_id
            WHERE 1=1
        """
        params: List[Any] = []
        if ticker:
            query += " AND c.ticker = ?"
            params.append(ticker.strip().upper())
        if fiscal_year:
            query += " AND r.fiscal_year = ?"
            params.append(fiscal_year)
        if period_type:
            query += " AND UPPER(r.period_type) = ?"
            params.append(period_type.strip().upper())
        if metric_code:
            query += " AND m.code = ?"
            params.append(metric_code.strip())
        if line_item:
            query += " AND (f.line_item LIKE ? OR f.line_item LIKE ?)"
            params.append(f"%{line_item.strip()}%")
            params.append(line_item.strip())
        if column_role:
            query += " AND f.column_role = ?"
            params.append(column_role.strip().lower())

        query += " ORDER BY r.fiscal_year DESC, f.page ASC, f.fact_id ASC;"

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------
    # Chunk & FTS operations
    # -------------------------------------------------------------
    def add_chunk(
        self,
        report_id: int,
        section: Optional[str],
        chunk_type: str,
        page_start: Optional[int],
        page_end: Optional[int],
        context_prefix: Optional[str],
        content: str,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chunk (
                    report_id, section, chunk_type, page_start, page_end, context_prefix, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING chunk_id;
                """,
                (
                    report_id,
                    section,
                    chunk_type,
                    page_start,
                    page_end,
                    context_prefix,
                    content.strip(),
                ),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["chunk_id"])

    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[int]:
        ids: List[int] = []
        with self._conn() as conn:
            for c in chunks:
                cursor = conn.execute(
                    """
                    INSERT INTO chunk (
                        report_id, section, chunk_type, page_start, page_end, context_prefix, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    RETURNING chunk_id;
                    """,
                    (
                        c["report_id"],
                        c.get("section"),
                        c.get("chunk_type", "text"),
                        c.get("page_start"),
                        c.get("page_end"),
                        c.get("context_prefix"),
                        c["content"].strip(),
                    ),
                )
                ids.append(int(cursor.fetchone()["chunk_id"]))
            conn.commit()
        return ids

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT ch.*, r.fiscal_year, r.period_type, r.doc_type, r.currency, r.scope,
                       c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en
                FROM chunk ch
                JOIN report r ON ch.report_id = r.report_id
                JOIN company c ON r.company_id = c.company_id
                WHERE ch.chunk_id = ?;
                """,
                (chunk_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT ch.*, r.fiscal_year, r.period_type, r.doc_type, r.currency, r.scope,
                       c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en
                FROM chunk ch
                JOIN report r ON ch.report_id = r.report_id
                JOIN company c ON r.company_id = c.company_id
                ORDER BY ch.chunk_id ASC;
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    def search_chunks_fts(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search chunk_fts full-text table with BM25 ranking and optional metadata filters."""
        # Sanitize query for FTS5 syntax
        clean_tokens = [re.sub(r"[^\w\d]", "", tok) for tok in query_text.split()]
        clean_tokens = [t for t in clean_tokens if len(t) > 1]
        if not clean_tokens:
            return []

        fts_query = " OR ".join(f'"{tok}"' for tok in clean_tokens)

        sql = """
            SELECT
                ch.*,
                r.fiscal_year, r.period_type, r.doc_type, r.currency, r.scope,
                c.ticker, c.name_vi as company_name_vi, c.name_en as company_name_en,
                bm25(chunk_fts) as bm25_rank
            FROM chunk_fts
            JOIN chunk ch ON chunk_fts.rowid = ch.chunk_id
            JOIN report r ON ch.report_id = r.report_id
            JOIN company c ON r.company_id = c.company_id
            WHERE chunk_fts MATCH ?
        """
        params: List[Any] = [fts_query]

        if ticker:
            sql += " AND c.ticker = ?"
            params.append(ticker.strip().upper())
        if fiscal_year:
            sql += " AND r.fiscal_year = ?"
            params.append(fiscal_year)
        if period_type:
            sql += " AND UPPER(r.period_type) = ?"
            params.append(period_type.strip().upper())

        sql += " ORDER BY bm25_rank ASC LIMIT ?;"
        params.append(limit)

        with self._conn() as conn:
            cursor = conn.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------
    # Evaluation question operations
    # -------------------------------------------------------------
    def add_eval_question(
        self,
        question: str,
        language: str,
        difficulty: str,
        gold_answer: Optional[str] = None,
        gold_report_id: Optional[int] = None,
        gold_page: Optional[int] = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO eval_question (question, language, difficulty, gold_answer, gold_report_id, gold_page)
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING q_id;
                """,
                (question.strip(), language.strip().lower(), difficulty.strip().lower(), gold_answer, gold_report_id, gold_page),
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row["q_id"])

    def get_all_eval_questions(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM eval_question ORDER BY q_id ASC;")
            return [dict(r) for r in cursor.fetchall()]
