"""Single point where an audit-store name (from config) becomes an
AuditStore instance. Everything else in the codebase should go through
this, never import a specific store module directly."""

from __future__ import annotations

from slmguard.audit.base import AuditStore

_STORES: dict[str, type[AuditStore]] = {}


def register(name: str, store_cls: type[AuditStore]) -> None:
    _STORES[name] = store_cls


def get_audit_store(name: str, location: str) -> AuditStore:
    if name not in _STORES:
        _register_builtin_stores()
    try:
        return _STORES[name](location)
    except KeyError:
        available = ", ".join(sorted(_STORES)) or "(none registered)"
        raise ValueError(f"Unknown audit store '{name}'. Available: {available}") from None


def _register_builtin_stores() -> None:
    from slmguard.audit.postgres_store import PostgresAuditStore
    from slmguard.audit.sqlite_store import SQLiteAuditStore

    register("sqlite", SQLiteAuditStore)
    register("postgres", PostgresAuditStore)
