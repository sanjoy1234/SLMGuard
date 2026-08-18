"""The audit store contract.

Non-negotiable project rule (enterprise-grade mandate, see
docs/technical-build-plan-v5.md and project memory): the audit store must be
append-only and tamper-evident from Phase 1 onward, behind a swappable
interface — same pattern already applied to `ModelBackend` (MLX now,
CUDA+QLoRA confirmed production). SQLite (see sqlite_store.py) is the Phase 1
/ local-development implementation, not the architecture, and is not
production-grade for an enterprise audit store on its own (no replication,
access control, or retention policy). PostgreSQL (see postgres_store.py) is
the confirmed production backend implementing this exact same contract. No
caller outside a concrete AuditStore implementation may import `sqlite3`,
`psycopg`, or any other storage driver directly.

There is deliberately no update/delete method on this interface — an audit
trail that can be edited after the fact is not an audit trail. Corrections
happen by appending a new trace, never by mutating one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from slmguard.audit.types import StoredTrace, TraceRecord


class AuditStore(ABC):
    """Abstract contract every trace store (SQLite, PostgreSQL, ...) must implement."""

    name: str

    @abstractmethod
    def append(self, trace: TraceRecord) -> str:
        """Append one trace to the hash chain and return the row's hash."""

    @abstractmethod
    def query(self, *, alert_id: str | None = None, limit: int = 100) -> list[StoredTrace]:
        """Return the most recent traces (newest first), optionally filtered by alert_id."""

    @abstractmethod
    def verify_chain(self) -> bool:
        """Recompute the hash chain end to end; True iff no row was altered or removed."""

    @abstractmethod
    def export(self, path: Path) -> Path:
        """Write the full audit trail to `path` as newline-delimited JSON, for external review."""
