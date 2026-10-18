"""SQLite database schema and connection utilities."""

import sqlite3
from pathlib import Path
from typing import Union


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS company (
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    name_vi TEXT,
    name_en TEXT,
    industry TEXT,
    country TEXT
);

CREATE TABLE IF NOT EXISTS report (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES company(company_id),
    doc_type TEXT,          -- BCTC | annual_report
    period_type TEXT,       -- H1 | 9M | FY | Q1...
    fiscal_year INTEGER NOT NULL,
    period_end DATE,
    scope TEXT,             -- consolidated | separate
    language TEXT,          -- vi | en
    currency TEXT,          -- VND | USD
    unit_scale INTEGER,     -- 1, 1000, 1000000, 1000000000
    audit_status TEXT,      -- reviewed | audited | unaudited
    file_path TEXT,
    source_url TEXT,
    is_scanned BOOLEAN DEFAULT 0,
    needs_manual_check BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metric (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    name_vi TEXT NOT NULL,
    name_en TEXT,
    statement TEXT,         -- IS | BS | CF
    is_common BOOLEAN DEFAULT 1,
    aliases TEXT            -- Delimited or JSON aliases for query expansion
);

CREATE TABLE IF NOT EXISTS fact (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES report(report_id),
    metric_id INTEGER REFERENCES metric(metric_id),
    line_item TEXT NOT NULL,
    column_role TEXT NOT NULL, -- current | prior_year_same_period | year_start
    period_start DATE,
    period_end DATE,
    value REAL,
    value_raw TEXT,
    page INTEGER,
    table_ref TEXT,
    UNIQUE (report_id, line_item, column_role)
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES report(report_id),
    section TEXT,
    chunk_type TEXT,        -- text | table
    page_start INTEGER,
    page_end INTEGER,
    context_prefix TEXT,
    content TEXT NOT NULL
);

-- FTS5 full-text search table for chunks
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    content,
    content='chunk',
    content_rowid='chunk_id'
);

-- Triggers to keep FTS index synchronized with chunk table
CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
  INSERT INTO chunk_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES('delete', old.chunk_id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunk_au AFTER UPDATE ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES('delete', old.chunk_id, old.content);
  INSERT INTO chunk_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;

CREATE TABLE IF NOT EXISTS eval_question (
    q_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    language TEXT,
    difficulty TEXT,        -- lookup | text | calc | compare | trap
    gold_answer TEXT,
    gold_report_id INTEGER,
    gold_page INTEGER
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_report_company_year ON report(company_id, fiscal_year, period_type);
CREATE INDEX IF NOT EXISTS idx_fact_report_metric ON fact(report_id, metric_id);
CREATE INDEX IF NOT EXISTS idx_fact_line_item ON fact(line_item);
CREATE INDEX IF NOT EXISTS idx_chunk_report ON chunk(report_id);
"""


def get_connection(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Create a SQLite connection with row factory and foreign keys enabled."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_database(db_path: Union[str, Path]) -> None:
    """Initialize database schema including tables, triggers and indexes."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
