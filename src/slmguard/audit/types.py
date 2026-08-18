"""Backend-agnostic value types passed across the AuditStore boundary.

None of these types may carry a storage-driver-specific object (a
sqlite3.Row, a psycopg connection, ...). Same rule as
slmguard.backends.types: plain Python values only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceRecord:
    """One control-plane decision, complete enough to replay or audit later."""

    trace_id: str
    timestamp: str
    alert_id: str
    prompt: str
    raw_output: str
    schema_valid: bool
    recommendation_json: str | None
    final_action: str
    confidence: float | None
    model_version: str
    backend_name: str


@dataclass(frozen=True)
class StoredTrace:
    """A TraceRecord as persisted, with its position in the hash chain."""

    row_id: int
    trace: TraceRecord
    prev_hash: str
    row_hash: str
