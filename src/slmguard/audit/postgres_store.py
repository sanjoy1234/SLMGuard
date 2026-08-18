"""Confirmed production backend: PostgreSQL.

This is the audit store enterprise/BFSI platform teams expect: managed
infrastructure with proper access control, backup, replication, and
retention policy — not a file on a laptop. Not implemented during Phase 1:
there is no Postgres instance in the current build environment, and writing
untested storage code against infrastructure that isn't available would just
produce code nobody can validate. What exists here instead is the real
`AuditStore` contract, wired up, with each method's production
implementation named and scoped — so the parity gap with `SQLiteAuditStore`
is a tracked list, not a surprise discovered later.

Parity checklist before this backend is production-ready:
  - [ ] append: INSERT into an append-only table inside a transaction, with
        UPDATE/DELETE revoked at the database role level (not just
        application discipline); same hash-chaining scheme as
        `SQLiteAuditStore` so chains exported from either backend are
        directly comparable.
  - [ ] query: indexed lookup by alert_id / trace_id / timestamp range.
  - [ ] verify_chain: same recomputation logic as `SQLiteAuditStore`, run as
        a scheduled integrity job, not only on demand.
  - [ ] export: streamed export (`COPY` or a server-side cursor) so it does
        not require loading the full audit trail into memory.
  - [ ] Connection: DSN sourced from environment/secret store, never a
        hardcoded credential; TLS required.
"""

from __future__ import annotations

from pathlib import Path

from slmguard.audit.base import AuditStore
from slmguard.audit.types import StoredTrace, TraceRecord


class PostgresAuditStore(AuditStore):
    name = "postgres"

    def __init__(self, location: str) -> None:
        self._dsn = location

    def append(self, trace: TraceRecord) -> str:
        raise NotImplementedError(
            "PostgresAuditStore.append: production implementation should INSERT "
            "into an append-only table with UPDATE/DELETE revoked at the role "
            "level. See parity checklist in this file's module docstring."
        )

    def query(self, *, alert_id: str | None = None, limit: int = 100) -> list[StoredTrace]:
        raise NotImplementedError(
            "PostgresAuditStore.query: production implementation should use "
            "indexed lookups by alert_id/trace_id/timestamp range."
        )

    def verify_chain(self) -> bool:
        raise NotImplementedError(
            "PostgresAuditStore.verify_chain: production implementation should "
            "run the same recomputation as SQLiteAuditStore, as a scheduled job."
        )

    def export(self, path: Path) -> Path:
        raise NotImplementedError(
            "PostgresAuditStore.export: production implementation should stream "
            "via COPY or a server-side cursor rather than loading all rows."
        )
