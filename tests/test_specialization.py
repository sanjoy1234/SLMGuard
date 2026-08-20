"""Proves the specialization loop actually works: trace selection filters
correctly, conversion respects the "no valid label, no conversion" rule and
redacts PII, the pool only keeps rubric-passing examples, dataset writing
splits train/valid correctly, and run_cycle produces an explicit promotion
decision -- including the short-circuit "insufficient specialization
signal" outcome and both promote/no-promote results from a full run -- all
against a FakeBackend so no real MLX hardware or subprocess is needed."""

from __future__ import annotations

import json

import pytest

from slmguard.audit.sqlite_store import SQLiteAuditStore
from slmguard.audit.types import TraceRecord
from slmguard.backends.base import ModelBackend
from slmguard.backends.types import (
    AdapterArtifact,
    LoadedModel,
    LoRAConfig,
    ModelVersion,
    RawModelOutput,
)
from slmguard.evaluation import ChallengeSet, HeldOutSet, LabeledCase
from slmguard.schema import Action
from slmguard.specialization import (
    build_specialization_pool,
    convert_trace,
    evaluate_challenge_set,
    redact_pii,
    run_cycle,
    select_traces,
    write_training_dataset,
)

GOOD_PROMPT = (
    "You are a fraud-alert triage system. Alert ALERT-1: a $500 purchase with "
    "no prior history at all for this specific card/merchant combination."
)
GOOD_RATIONALE = (
    "The purchase amount and total lack of prior history at this merchant "
    "combination raises enough concern to warrant further review."
)


def _recommendation_json(**overrides) -> str:
    fields = {
        "alert_id": "ALERT-1",
        "action": "escalate_l2",
        "risk_score": 0.7,
        "confidence": 0.5,
        "policy_flags": [],
        "rationale": GOOD_RATIONALE,
    }
    fields.update(overrides)
    return json.dumps(fields)


def _trace(**overrides) -> TraceRecord:
    fields = dict(
        trace_id="t1",
        timestamp="2026-08-20T00:00:00+00:00",
        alert_id="ALERT-1",
        prompt=GOOD_PROMPT,
        raw_output="raw",
        schema_valid=True,
        recommendation_json=_recommendation_json(),
        final_action="escalate_l2",
        confidence=0.5,
        model_version="fake-model",
        backend_name="fake",
    )
    fields.update(overrides)
    return TraceRecord(**fields)


def test_redact_pii_removes_ssn_and_email():
    text = "Customer SSN 219-09-9999, email jane@example.com, called in."
    redacted = redact_pii(text)
    assert "219-09-9999" not in redacted
    assert "jane@example.com" not in redacted
    assert "[REDACTED:ssn]" in redacted
    assert "[REDACTED:email]" in redacted


def test_convert_trace_returns_none_for_schema_invalid_trace():
    trace = _trace(schema_valid=False, recommendation_json=None)
    assert convert_trace(trace) is None


def test_convert_trace_redacts_pii_in_scenario():
    trace = _trace(prompt=GOOD_PROMPT + " SSN 219-09-9999.")
    example = convert_trace(trace)
    assert example is not None
    assert "219-09-9999" not in example.scenario
    assert example.recommendation_json == trace.recommendation_json


def test_select_traces_filters_to_low_confidence_and_schema_failures(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace(trace_id="high_conf", confidence=0.95))
    store.append(_trace(trace_id="low_conf", confidence=0.4))
    store.append(_trace(trace_id="failure", schema_valid=False, recommendation_json=None, confidence=None))

    selection = select_traces(store, confidence_threshold=0.6)

    ids = {t.trace_id for t in selection.selected}
    assert ids == {"low_conf", "failure"}
    assert selection.total_scanned == 3


def test_build_specialization_pool_drops_rubric_failing_examples(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace(trace_id="good", confidence=0.4))
    store.append(
        _trace(
            trace_id="bad_rationale",
            confidence=0.4,
            recommendation_json=_recommendation_json(rationale="too short"),
        )
    )

    pool = build_specialization_pool(store, confidence_threshold=0.6)

    assert pool.traces_selected == 2
    assert pool.traces_unconvertible == 0
    assert len(pool.examples) == 1
    assert pool.rubric_score.total == 2
    assert pool.rubric_score.passed == 1


def test_write_training_dataset_splits_train_and_valid(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    for i in range(6):
        store.append(_trace(trace_id=f"t{i}", confidence=0.4, alert_id=f"ALERT-{i}"))
    pool = build_specialization_pool(store, confidence_threshold=0.6)

    dataset_dir = write_training_dataset(pool, tmp_path / "dataset")

    train_lines = (dataset_dir / "train.jsonl").read_text().strip().splitlines()
    valid_lines = (dataset_dir / "valid.jsonl").read_text().strip().splitlines()
    assert len(train_lines) + len(valid_lines) == 6
    assert len(train_lines) > 0 and len(valid_lines) > 0
    record = json.loads(train_lines[0])
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["role"] == "assistant"


def test_write_training_dataset_handles_single_example(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace(confidence=0.4))
    pool = build_specialization_pool(store, confidence_threshold=0.6)

    dataset_dir = write_training_dataset(pool, tmp_path / "dataset")

    assert len((dataset_dir / "train.jsonl").read_text().strip().splitlines()) == 1
    assert len((dataset_dir / "valid.jsonl").read_text().strip().splitlines()) == 1


def test_write_training_dataset_respects_min_valid_size_for_batch_size(tmp_path):
    # mlx_lm.lora requires the validation set to have at least batch_size
    # examples -- a plain 80/20 split of a small pool can leave just 1.
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    for i in range(7):
        store.append(_trace(trace_id=f"t{i}", confidence=0.4, alert_id=f"ALERT-{i}"))
    pool = build_specialization_pool(store, confidence_threshold=0.6)
    assert len(pool.examples) == 7

    dataset_dir = write_training_dataset(pool, tmp_path / "dataset", min_valid_size=2)

    valid_lines = (dataset_dir / "valid.jsonl").read_text().strip().splitlines()
    assert len(valid_lines) >= 2


class FakeBackend(ModelBackend):
    name = "fake"

    def __init__(self, base_version, base_action, candidate_actions):
        self._base_version = base_version
        self._base_action = base_action
        self._candidate_actions = candidate_actions
        self.fine_tune_calls = []
        self.fuse_calls = []

    def load_model(self, version):
        return LoadedModel(version_id=version, backend_name=self.name, handle=None)

    def generate(self, model, prompt):
        if model.version_id == self._base_version:
            action = self._base_action
        else:
            action = self._candidate_actions.get(prompt, self._base_action)
        payload = {
            "alert_id": "x",
            "action": action.value,
            "risk_score": 0.5,
            "confidence": 0.9,
            "policy_flags": [],
            "rationale": "fake model output for testing purposes only",
        }
        return RawModelOutput(text=json.dumps(payload), schema_valid=True)

    def fine_tune(self, model, dataset, config):
        self.fine_tune_calls.append((model, dataset, config))
        adapter_dir = dataset.parent / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return AdapterArtifact(
            path=adapter_dir,
            base_version_id=model.version_id,
            backend_name=self.name,
            training_data_hash="fake-hash",
        )

    def fuse_adapters(self, model, adapter):
        self.fuse_calls.append((model, adapter))
        fused_dir = adapter.path.parent / "fused"
        fused_dir.mkdir(parents=True, exist_ok=True)
        return ModelVersion(
            version_id="fused",
            parent_version_id=model.version_id,
            backend_name=self.name,
            fused_weights_path=fused_dir,
        )


def _held_out_set():
    return HeldOutSet(
        version="bootstrap-v1",
        cases=(
            LabeledCase("h1", Action.APPROVE, "CASE-1"),
            LabeledCase("h2", Action.DECLINE, "CASE-2"),
            LabeledCase("h3", Action.ESCALATE_L2, "CASE-3"),
            LabeledCase("h4", Action.REQUEST_MORE_INFO, "CASE-4"),
        ),
    )


def _challenge_set():
    return ChallengeSet(version="v1", cases=(LabeledCase("c1", Action.ESCALATE_L2, "CASE-3"),))


def _lora_config():
    return LoRAConfig(
        rank=8,
        alpha=16,
        epochs=1,
        learning_rate=1e-5,
        batch_size=1,
        max_seq_length=512,
        grad_checkpoint=True,
    )


def test_run_cycle_short_circuits_on_insufficient_pool(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    store.append(_trace(confidence=0.4))  # only 1 example, below MIN_POOL_SIZE

    backend = FakeBackend("base-model", Action.APPROVE, {})
    result = run_cycle(
        store=store,
        backend=backend,
        base_model_version="base-model",
        held_out_set=_held_out_set(),
        challenge_set=_challenge_set(),
        lora_config=_lora_config(),
        work_dir=tmp_path / "work",
    )

    assert result.promoted is False
    assert "insufficient specialization signal" in result.reason
    assert result.decision is None
    assert backend.fine_tune_calls == []


def test_run_cycle_promotes_when_candidate_improves(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    for i in range(6):
        store.append(_trace(trace_id=f"t{i}", confidence=0.4, alert_id=f"ALERT-{i}"))

    backend = FakeBackend(
        base_version="base-model",
        base_action=Action.APPROVE,  # correct only for CASE-1 -> baseline accuracy 0.25
        candidate_actions={
            "CASE-1": Action.APPROVE,
            "CASE-2": Action.DECLINE,
            "CASE-3": Action.ESCALATE_L2,
            "CASE-4": Action.REQUEST_MORE_INFO,
        },  # correct for every case -> candidate accuracy 1.0
    )

    result = run_cycle(
        store=store,
        backend=backend,
        base_model_version="base-model",
        held_out_set=_held_out_set(),
        challenge_set=_challenge_set(),
        lora_config=_lora_config(),
        work_dir=tmp_path / "work",
    )

    assert len(backend.fine_tune_calls) == 1
    assert len(backend.fuse_calls) == 1
    assert result.baseline_summary.accuracy == pytest.approx(0.25)
    assert result.candidate_summary.accuracy == pytest.approx(1.0)
    assert result.promoted is True
    assert result.reason == "all gates passed"


def test_run_cycle_does_not_promote_when_candidate_regresses(tmp_path):
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    for i in range(6):
        store.append(_trace(trace_id=f"t{i}", confidence=0.4, alert_id=f"ALERT-{i}"))

    # Reuse FakeBackend but swap which side is "smart": here the base model
    # gets every case right (accuracy 1.0) and the candidate regresses to
    # always-approve (accuracy 0.25) -- the mirror image of the promote test.
    class RegressingBackend(FakeBackend):
        def generate(self, model, prompt):
            if model.version_id == self._base_version:
                action = self._candidate_actions.get(prompt, Action.APPROVE)
            else:
                action = self._base_action
            payload = {
                "alert_id": "x",
                "action": action.value,
                "risk_score": 0.5,
                "confidence": 0.9,
                "policy_flags": [],
                "rationale": "fake model output for testing purposes only",
            }
            return RawModelOutput(text=json.dumps(payload), schema_valid=True)

    backend = RegressingBackend(
        base_version="base-model",
        base_action=Action.APPROVE,
        candidate_actions={
            "CASE-1": Action.APPROVE,
            "CASE-2": Action.DECLINE,
            "CASE-3": Action.ESCALATE_L2,
            "CASE-4": Action.REQUEST_MORE_INFO,
        },
    )

    result = run_cycle(
        store=store,
        backend=backend,
        base_model_version="base-model",
        held_out_set=_held_out_set(),
        challenge_set=_challenge_set(),
        lora_config=_lora_config(),
        work_dir=tmp_path / "work",
    )

    assert result.baseline_summary.accuracy == pytest.approx(1.0)
    assert result.candidate_summary.accuracy == pytest.approx(0.25)
    assert result.promoted is False
    assert "no promotion this cycle" in result.reason
    assert "accuracy" in result.reason


def test_evaluate_challenge_set_reports_true_regression():
    backend = FakeBackend(
        base_version="base",
        base_action=Action.DECLINE,
        candidate_actions={"CASE-B": Action.APPROVE},  # candidate wrong here, base's fallback is right
    )
    base_model = backend.load_model("base")
    candidate_model = backend.load_model("candidate")
    challenge_set = ChallengeSet(
        version="v1",
        cases=(
            LabeledCase("A", Action.DECLINE, "CASE-A"),  # both correct (candidate falls back to DECLINE)
            LabeledCase("B", Action.DECLINE, "CASE-B"),  # base correct, candidate wrong -> regression
        ),
    )

    report = evaluate_challenge_set(backend, base_model, candidate_model, challenge_set)

    assert report.total == 2
    assert report.baseline_failures == 0
    assert report.candidate_failures == 1
    assert report.new_failures == 1
    assert report.new_failure_alert_ids == ("B",)


def test_evaluate_challenge_set_shared_failure_is_not_a_regression():
    # Both base and candidate get the same case wrong (candidate_actions empty
    # -> candidate falls back to the same base_action) -- a pre-existing gap,
    # not something the specialization cycle caused.
    backend = FakeBackend(base_version="base", base_action=Action.APPROVE, candidate_actions={})
    base_model = backend.load_model("base")
    candidate_model = backend.load_model("candidate")
    challenge_set = ChallengeSet(
        version="v1", cases=(LabeledCase("C", Action.DECLINE, "CASE-C"),)
    )

    report = evaluate_challenge_set(backend, base_model, candidate_model, challenge_set)

    assert report.baseline_failures == 1
    assert report.candidate_failures == 1
    assert report.new_failures == 0
    assert report.new_failure_alert_ids == ()


def test_run_cycle_promotes_despite_shared_challenge_failure(tmp_path):
    # Regression test for the exact gap found in real-world validation: the
    # old count_challenge_failures() counted the candidate's absolute
    # failures and would have blocked promotion here even though the base
    # model fails the same challenge case too -- not a real regression.
    store = SQLiteAuditStore(str(tmp_path / "a.db"))
    for i in range(6):
        store.append(_trace(trace_id=f"t{i}", confidence=0.4, alert_id=f"ALERT-{i}"))

    backend = FakeBackend(
        base_version="base-model",
        base_action=Action.APPROVE,
        candidate_actions={
            "CASE-1": Action.APPROVE,
            "CASE-2": Action.DECLINE,
            "CASE-3": Action.ESCALATE_L2,
            "CASE-4": Action.REQUEST_MORE_INFO,
            "CASE-HARD": Action.APPROVE,  # candidate still wrong here, same as base
        },
    )
    challenge_set = ChallengeSet(
        version="v1", cases=(LabeledCase("hard-1", Action.DECLINE, "CASE-HARD"),)
    )

    result = run_cycle(
        store=store,
        backend=backend,
        base_model_version="base-model",
        held_out_set=_held_out_set(),
        challenge_set=challenge_set,
        lora_config=_lora_config(),
        work_dir=tmp_path / "work",
    )

    assert result.challenge_report.baseline_failures == 1
    assert result.challenge_report.candidate_failures == 1
    assert result.challenge_report.new_failures == 0
    assert result.promoted is True
    assert result.reason == "all gates passed"
