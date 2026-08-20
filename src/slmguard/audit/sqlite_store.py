"""Phase 1 / local-development backend: SQLite.

This is the pragmatic choice for iterating on the trace schema and
hash-chaining logic without standing up infrastructure — not the project's
architecture, and not production-grade for an enterprise audit store on its
own (no replication, access control, or retention policy). PostgreSQL (see
postgres_store.py) is the confirmed production backend implementing this
exact same contract. Nothing outside this file may import `sqlite3` directly.

Append-only is enforced by construction: this class exposes no update or
delete method, and every row's hash covers the previous row's hash plus its
own canonical field values, so an out-of-band edit (e.g. a raw `UPDATE`
against the file) is detectable via `verify_chain`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from slmguard.audit.base import AuditStore
from slmguard.audit.types import StoredTrace, TraceRecord

_GENESIS_HASH = "0" * 64

_COLUMNS = (
    "trace_id",
    "timestamp",
    "alert_id",
    "prompt",
    "raw_output",
    "schema_valid",
    "recommendation_json",
    "final_action",
    "confidence",
    "model_version",
    "backend_name",
    "policy_version",
    "policy_overridden",
    "policy_violated_rule_ids",
)

_COLUMN_TYPES = {
    "schema_valid": "INTEGER",
    "confidence": "REAL",
    "policy_overridden": "INTEGER",
}


class SQLiteAuditStore(AuditStore):
    name = "sqlite"

    def __init__(self, location: str) -> None:
        self._path = Path(location)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {", ".join(f"{c} {_COLUMN_TYPES.get(c, 'TEXT')}" for c in _COLUMNS)},
                prev_hash TEXT NOT NULL,
                row_hash TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def append(self, trace: TraceRecord) -> str:
        prev_hash = self._last_hash()
        row_hash = _hash_row(prev_hash, trace)
        fields = asdict(trace)
        fields["schema_valid"] = int(fields["schema_valid"])
        fields["policy_overridden"] = int(fields["policy_overridden"])
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT INTO traces ({', '.join(_COLUMNS)}, prev_hash, row_hash) "
            f"VALUES ({placeholders}, ?, ?)",
            (*(fields[c] for c in _COLUMNS), prev_hash, row_hash),
        )
        self._conn.commit()
        return row_hash

    def query(self, *, alert_id: str | None = None, limit: int = 100) -> list[StoredTrace]:
        sql = "SELECT id, " + ", ".join(_COLUMNS) + ", prev_hash, row_hash FROM traces"
        params: tuple = ()
        if alert_id is not None:
            sql += " WHERE alert_id = ?"
            params = (alert_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        params = params + (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_stored_trace(row) for row in rows]

    def verify_chain(self) -> bool:
        sql = "SELECT id, " + ", ".join(_COLUMNS) + ", prev_hash, row_hash FROM traces ORDER BY id ASC"
        expected_prev = _GENESIS_HASH
        for row in self._conn.execute(sql).fetchall():
            stored = _row_to_stored_trace(row)
            if stored.prev_hash != expected_prev:
                return False
            if _hash_row(stored.prev_hash, stored.trace) != stored.row_hash:
                return False
            expected_prev = stored.row_hash
        return True

    def export(self, path: Path) -> Path:
        rows = self.query(limit=10_000_000)
        rows.reverse()
        with path.open("w") as f:
            for stored in rows:
                record = {
                    **asdict(stored.trace),
                    "prev_hash": stored.prev_hash,
                    "row_hash": stored.row_hash,
                }
                f.write(json.dumps(record) + "\n")
        return path

    def _last_hash(self) -> str:
        row = self._conn.execute("SELECT row_hash FROM traces ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else _GENESIS_HASH


def _hash_row(prev_hash: str, trace: TraceRecord) -> str:
    canonical = json.dumps(asdict(trace), sort_keys=True)
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def _row_to_stored_trace(row: tuple) -> StoredTrace:
    row_id = row[0]
    field_values = dict(zip(_COLUMNS, row[1 : 1 + len(_COLUMNS)]))
    prev_hash, row_hash = row[1 + len(_COLUMNS) : 3 + len(_COLUMNS)]
    field_values["schema_valid"] = bool(field_values["schema_valid"])
    field_values["policy_overridden"] = bool(field_values["policy_overridden"])
    trace = TraceRecord(**field_values)
    return StoredTrace(row_id=row_id, trace=trace, prev_hash=prev_hash, row_hash=row_hash)
