"""SQLite storage for leads, stage history and sent messages."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    phone           TEXT NOT NULL UNIQUE,
    city            TEXT DEFAULT '',
    age             INTEGER,
    profession      TEXT DEFAULT 'other',
    monthly_income  INTEGER DEFAULT 0,
    existing_investments INTEGER DEFAULT 0,
    goal            TEXT DEFAULT '',
    source          TEXT DEFAULT 'other',
    referred_by     TEXT DEFAULT '',
    consent         INTEGER DEFAULT 0,
    opted_out       INTEGER DEFAULT 0,
    score           INTEGER DEFAULT 0,
    tier            TEXT DEFAULT 'C',
    sip_potential   INTEGER DEFAULT 0,
    stage           TEXT DEFAULT 'new',
    sip_amount      INTEGER DEFAULT 0,
    notes           TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    stage_changed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id    INTEGER NOT NULL REFERENCES leads(id),
    from_stage TEXT,
    to_stage   TEXT NOT NULL,
    changed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id   INTEGER NOT NULL REFERENCES leads(id),
    step_key  TEXT NOT NULL,
    sent_at   TEXT NOT NULL,
    channel   TEXT NOT NULL,
    UNIQUE (lead_id, step_key)
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | None = None):
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def set_stage(conn, lead_id: int, stage: str, sip_amount: int | None = None) -> None:
    if stage not in config.STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Use one of {config.STAGES}")
    row = conn.execute("SELECT stage FROM leads WHERE id = ?", (lead_id,)).fetchone()
    if row is None:
        raise ValueError(f"No lead with id {lead_id}")
    ts = now()
    if sip_amount is not None:
        conn.execute("UPDATE leads SET sip_amount = ? WHERE id = ?", (sip_amount, lead_id))
    conn.execute(
        "UPDATE leads SET stage = ?, stage_changed_at = ? WHERE id = ?", (stage, ts, lead_id)
    )
    conn.execute(
        "INSERT INTO stage_history (lead_id, from_stage, to_stage, changed_at) VALUES (?,?,?,?)",
        (lead_id, row["stage"], stage, ts),
    )
