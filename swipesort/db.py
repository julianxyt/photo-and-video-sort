"""SQLite index. One row per media file, one row per swipe.

Decisions are append-only: the newest row for a media id is its current verdict,
which makes undo a matter of deleting the last row rather than rewriting state.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

KEEP_ACTIONS = ("keep", "love")
DROP_ACTIONS = ("drop",)
DEFER_ACTIONS = ("later",)
ALL_ACTIONS = KEEP_ACTIONS + DROP_ACTIONS + DEFER_ACTIONS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    rel_path      TEXT NOT NULL,
    filename      TEXT NOT NULL,
    ext           TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    mtime         REAL,
    kind          TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    year          TEXT,
    taken_at      TEXT,
    date_source   TEXT,
    width         INTEGER,
    height        INTEGER,
    duration_s    REAL,
    thumb         TEXT,
    phash         INTEGER,
    sha256        TEXT,
    exact_dup_of  INTEGER REFERENCES media(id),
    dup_group     INTEGER,
    features      BLOB,
    feat_kind     TEXT,
    score         REAL,
    missing       INTEGER NOT NULL DEFAULT 0,
    ingested_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_phash    ON media(phash);
CREATE INDEX IF NOT EXISTS idx_media_size     ON media(size_bytes);
CREATE INDEX IF NOT EXISTS idx_media_dupgroup ON media(dup_group);
CREATE INDEX IF NOT EXISTS idx_media_bucket   ON media(bucket, year);

CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY,
    media_id   INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    action     TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'swipe',
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_media ON decisions(media_id, id DESC);

CREATE TABLE IF NOT EXISTS moves (
    id         INTEGER PRIMARY KEY,
    batch      TEXT NOT NULL,
    media_id   INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    op         TEXT NOT NULL,
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    created_at TEXT NOT NULL,
    undone_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_moves_batch ON moves(batch);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

/* Current verdict per media: the highest decision id wins. */
CREATE VIEW IF NOT EXISTS verdicts AS
SELECT d.media_id AS media_id, d.action AS action, d.created_at AS decided_at
FROM decisions d
JOIN (SELECT media_id, MAX(id) AS id FROM decisions GROUP BY media_id) last
  ON last.id = d.id;
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (and migrate) the index at ``db_path``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    payload = value if isinstance(value, str) else json.dumps(value)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, payload),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def upsert_media(conn: sqlite3.Connection, record: dict[str, Any]) -> int:
    """Insert a media row, or refresh the scan-derived fields of an existing one.

    Features, scores and decisions are intentionally left alone so that
    re-ingesting a library never throws away training data.
    """
    record = dict(record)
    record.setdefault("ingested_at", utcnow())
    cols = list(record)
    placeholders = ", ".join("?" for _ in cols)
    updatable = [c for c in cols if c not in {"path", "ingested_at"}]
    assignments = ", ".join(f"{c}=excluded.{c}" for c in updatable)
    sql = (
        f"INSERT INTO media ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [record[c] for c in cols])
    row = conn.execute("SELECT id FROM media WHERE path=?", (record["path"],)).fetchone()
    return int(row["id"])


def record_decision(
    conn: sqlite3.Connection,
    media_id: int,
    action: str,
    *,
    source: str = "swipe",
    latency_ms: int | None = None,
) -> int:
    if action not in ALL_ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {ALL_ACTIONS}")
    cur = conn.execute(
        "INSERT INTO decisions(media_id, action, source, latency_ms, created_at) "
        "VALUES(?,?,?,?,?)",
        (media_id, action, source, latency_ms, utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def undo_last_decision(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Remove the most recent decision and return the row that was removed."""
    row = conn.execute(
        "SELECT * FROM decisions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM decisions WHERE id=?", (row["id"],))
    conn.commit()
    return row


def labelled_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Media rows carrying a keep/love/drop verdict, with features present."""
    return conn.execute(
        "SELECT m.*, v.action AS action FROM media m "
        "JOIN verdicts v ON v.media_id = m.id "
        "WHERE v.action IN ('keep','love','drop') AND m.features IS NOT NULL "
        "AND m.missing = 0"
    ).fetchall()


def undecided_rows(conn: sqlite3.Connection, *, include_later: bool = False) -> list[sqlite3.Row]:
    """Media still awaiting a verdict (optionally including deferred items)."""
    if include_later:
        clause = "v.action IS NULL OR v.action = 'later'"
    else:
        clause = "v.action IS NULL"
    return conn.execute(
        f"SELECT m.*, v.action AS action FROM media m "
        f"LEFT JOIN verdicts v ON v.media_id = m.id "
        f"WHERE m.missing = 0 AND m.exact_dup_of IS NULL AND ({clause})"
    ).fetchall()


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Headline numbers for the stats endpoint and the CLI."""
    total, total_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM media WHERE missing=0"
    ).fetchone()
    by_action = {
        r["action"]: (r["n"], r["bytes"])
        for r in conn.execute(
            "SELECT v.action AS action, COUNT(*) AS n, COALESCE(SUM(m.size_bytes),0) AS bytes "
            "FROM verdicts v JOIN media m ON m.id=v.media_id WHERE m.missing=0 "
            "GROUP BY v.action"
        )
    }
    exact_dupes, exact_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM media "
        "WHERE exact_dup_of IS NOT NULL AND missing=0"
    ).fetchone()
    decided = sum(n for n, _ in by_action.values())
    return {
        "total": total,
        "total_bytes": total_bytes,
        "decided": decided,
        "remaining": total - decided - exact_dupes,
        "keep": by_action.get("keep", (0, 0))[0],
        "love": by_action.get("love", (0, 0))[0],
        "later": by_action.get("later", (0, 0))[0],
        "drop": by_action.get("drop", (0, 0))[0],
        "drop_bytes": by_action.get("drop", (0, 0))[1],
        "exact_duplicates": exact_dupes,
        "exact_duplicate_bytes": exact_bytes,
    }


def iter_chunks(rows: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
