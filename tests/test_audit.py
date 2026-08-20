"""Proves the audit store abstraction actually holds: both backends satisfy
the same AuditStore contract, appends are hash-chained and tamper-evident,
and the Postgres stub fails loudly and specifically rather than silently or
generically. Does not require a Postgres instance to run."""

from __future__ import annotations

import sqlite3

import pytest

from slmguard.audit import AuditStore, get_audit_store
from slmguard.audit.postgres_store import PostgresAuditStore
from slmguard.audit.sqlite_store import SQLiteAuditStore
from slmguard.audit.types import TraceRecord


def _trace(**overrides) -> TraceRecord:
    fields = dict(
        trace_id="t1",
        timestamp="2026-08-17T00:00:00+00:00",
        alert_id="ALERT-1",
        prompt="prompt text",
        raw_output='{"alert_id": "ALERT-1"}',
        schema_valid=True,
        recommendation_json='{"alert_id": "ALERT-1"}',
        final_action="escalate_l2",
        confidence=0.9,
        model_version="mlx-community/Qwen2.5-3B-Instruct-4bit",
        backend_name="mlx",
        policy_version="policy-v1",
        policy_overridden=False,
        policy_violated_rule_ids="[]",
    )
    fields.update(overrides)
    return TraceRecord(**fields)


def test_both_audit_stores_satisfy_the_interface(tmp_path):
    assert isinstance(get_audit_store("sqlite", str(tmp_path / "a.db")), AuditStore)
    assert isinstance(get_audit_store("postgres", "postgresql://x"), AuditStore)


def test_sqlite_store_is_the_phase1_default(tmp_path):
    store = get_audit_store("sqlite", str(tmp_path / "a.db"))
    assert isinstance(store, SQLiteAuditStore)
    assert store.name == "sqlite"


def test_postgres_store_is_registered_but_not_yet_runnable():
    store = get_audit_store("postgres", "postgresql://x")
    assert isinstance(store, PostgresAuditStore)
    with pytest.raises(NotImplementedError, match="PostgresAuditStore.append"):
        store.append(_trace())


def test_unknown_audit_store_name_raises_with_available_list():
    with pytest.raises(ValueError, match="sqlite"):
        get_audit_store("does-not-exist", "x")


def test_append_is_hash_chained_and_newest_first(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    h1 = store.append(_trace(trace_id="t1"))
    h2 = store.append(_trace(trace_id="t2"))
    assert h1 != h2

    rows = store.query(limit=10)
    assert [r.trace.trace_id for r in rows] == ["t2", "t1"]
    assert rows[0].prev_hash == h1
    assert rows[1].prev_hash == "0" * 64


def test_query_filters_by_alert_id(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace(trace_id="t1", alert_id="ALERT-1"))
    store.append(_trace(trace_id="t2", alert_id="ALERT-2"))

    rows = store.query(alert_id="ALERT-2")
    assert [r.trace.trace_id for r in rows] == ["t2"]


def test_verify_chain_detects_tampering(tmp_path):
    db_path = tmp_path / "a.db"
    store = SQLiteAuditStore(str(db_path))
    store.append(_trace())
    assert store.verify_chain() is True

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE traces SET raw_output = 'tampered' WHERE id = 1")
    conn.commit()
    conn.close()

    assert SQLiteAuditStore(str(db_path)).verify_chain() is False


def test_export_writes_jsonl(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace())
    out = tmp_path / "export.jsonl"
    store.export(out)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"trace_id": "t1"' in lines[0]
