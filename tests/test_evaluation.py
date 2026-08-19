"""Proves the evaluation harness matches docs/evaluation-harness-v1.md:
exact metric formulas, held-out/challenge set construction rules enforced
as real validation logic, and the four-gate promotion decision including
"no promotion this cycle" as an explicit, reasoned outcome."""

from __future__ import annotations

import pytest

from slmguard.evaluation import (
    ChallengeSet,
    EvaluatedCase,
    EvaluationSummary,
    LabeledCase,
    PromotionThresholds,
    accuracy,
    confidence_correctness_correlation,
    evaluate_promotion,
    expected_calibration_error,
    grow_challenge_set,
    per_class_precision_recall,
    policy_violation_rate,
    validate_held_out_set,
)
from slmguard.schema import Action


def _case(alert_id, true_action, predicted_action, confidence=0.5, policy_violated=False):
    return EvaluatedCase(
        alert_id=alert_id,
        true_action=true_action,
        predicted_action=predicted_action,
        confidence=confidence,
        policy_violated=policy_violated,
    )


def test_accuracy_counts_exact_matches():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE),
        _case("2", Action.APPROVE, Action.DECLINE),
        _case("3", Action.DECLINE, Action.DECLINE),
        _case("4", Action.ESCALATE_L2, Action.ESCALATE_L2),
    ]
    assert accuracy(cases) == 0.75


def test_accuracy_of_empty_list_is_zero():
    assert accuracy([]) == 0.0


def test_per_class_precision_recall_matches_hand_computed_confusion():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE),
        _case("2", Action.APPROVE, Action.DECLINE),
        _case("3", Action.DECLINE, Action.DECLINE),
        _case("4", Action.ESCALATE_L2, Action.ESCALATE_L2),
        _case("5", Action.REQUEST_MORE_INFO, Action.APPROVE),
    ]
    result = per_class_precision_recall(cases)

    assert result["approve"].precision == pytest.approx(0.5)
    assert result["approve"].recall == pytest.approx(0.5)
    assert result["approve"].support == 2

    # predicted-decline = {case2 (true=approve), case3 (true=decline)} -> precision 1/2
    assert result["decline"].precision == pytest.approx(0.5)
    assert result["decline"].recall == pytest.approx(1.0)
    assert result["decline"].support == 1

    assert result["escalate_l2"].precision == pytest.approx(1.0)
    assert result["escalate_l2"].recall == pytest.approx(1.0)

    assert result["request_more_info"].precision == pytest.approx(0.0)
    assert result["request_more_info"].recall == pytest.approx(0.0)
    assert result["request_more_info"].support == 1


def test_policy_violation_rate():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE, policy_violated=True),
        _case("2", Action.APPROVE, Action.APPROVE, policy_violated=False),
        _case("3", Action.APPROVE, Action.APPROVE, policy_violated=False),
        _case("4", Action.APPROVE, Action.APPROVE, policy_violated=False),
    ]
    assert policy_violation_rate(cases) == 0.25


def test_ece_matches_hand_computed_single_bin_example():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE, confidence=0.85),
        _case("2", Action.APPROVE, Action.DECLINE, confidence=0.85),
    ]
    assert expected_calibration_error(cases) == pytest.approx(0.35)


def test_confidence_correctness_correlation_is_positive_when_aligned():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE, confidence=0.9),
        _case("2", Action.APPROVE, Action.APPROVE, confidence=0.6),
        _case("3", Action.APPROVE, Action.DECLINE, confidence=0.4),
        _case("4", Action.APPROVE, Action.DECLINE, confidence=0.1),
    ]
    assert confidence_correctness_correlation(cases) > 0.5


def test_confidence_correctness_correlation_is_zero_when_degenerate():
    cases = [
        _case("1", Action.APPROVE, Action.APPROVE, confidence=0.9),
        _case("2", Action.APPROVE, Action.APPROVE, confidence=0.4),
    ]
    assert confidence_correctness_correlation(cases) == 0.0


def test_validate_held_out_set_rejects_wrong_size():
    cases = [LabeledCase(f"a{i}", Action.APPROVE) for i in range(5)]
    result = validate_held_out_set(cases)
    assert result.valid is False
    assert any(v.startswith("size_out_of_range") for v in result.violations)


def _stratified_cases(n=150, with_policy_boundary=True):
    actions = list(Action)
    cases = [
        LabeledCase(f"a{i}", actions[i % len(actions)], policy_boundary=False) for i in range(n)
    ]
    if with_policy_boundary:
        cases[0] = LabeledCase(cases[0].alert_id, cases[0].true_action, policy_boundary=True)
    return cases


def test_validate_held_out_set_rejects_missing_action_class():
    cases = [LabeledCase(f"a{i}", Action.APPROVE, policy_boundary=(i == 0)) for i in range(150)]
    result = validate_held_out_set(cases)
    assert result.valid is False
    assert any(v.startswith("missing_action_classes") for v in result.violations)


def test_validate_held_out_set_rejects_no_policy_boundary_cases():
    cases = _stratified_cases(with_policy_boundary=False)
    result = validate_held_out_set(cases)
    assert result.valid is False
    assert "no_policy_boundary_cases" in result.violations


def test_validate_held_out_set_accepts_a_well_formed_set():
    cases = _stratified_cases(with_policy_boundary=True)
    result = validate_held_out_set(cases)
    assert result.valid is True
    assert result.violations == ()


def test_grow_challenge_set_appends_and_versions():
    current = ChallengeSet(version="v1", cases=())
    grown = grow_challenge_set(current, [LabeledCase("a1", Action.DECLINE)])
    assert grown.version == "v2"
    assert [c.alert_id for c in grown.cases] == ["a1"]


def test_grow_challenge_set_never_drops_or_duplicates_existing_cases():
    current = ChallengeSet(version="v2", cases=(LabeledCase("a1", Action.DECLINE),))
    grown = grow_challenge_set(
        current, [LabeledCase("a1", Action.DECLINE), LabeledCase("a2", Action.ESCALATE_L2)]
    )
    assert [c.alert_id for c in grown.cases] == ["a1", "a2"]
    assert grown.version == "v3"


def _summary(accuracy_val, policy_rate=0.0, ece=0.0):
    return EvaluationSummary(
        n=100,
        accuracy=accuracy_val,
        per_class={},
        policy_violation_rate=policy_rate,
        ece=ece,
        reliability_diagram=(),
        confidence_correctness_correlation=0.0,
    )


def test_evaluate_promotion_passes_when_all_gates_clear():
    decision = evaluate_promotion(
        candidate=_summary(0.90),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
    )
    assert decision.promoted is True
    assert decision.reason == "all gates passed"
    assert all(g.passed for g in decision.gates)


def test_evaluate_promotion_rejects_accuracy_regression_beyond_threshold():
    decision = evaluate_promotion(
        candidate=_summary(0.85),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
    )
    assert decision.promoted is False
    assert "no promotion this cycle" in decision.reason
    assert "accuracy" in decision.reason


def test_evaluate_promotion_tolerates_small_accuracy_drop_within_threshold():
    decision = evaluate_promotion(
        candidate=_summary(0.885),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
    )
    assert decision.promoted is True


def test_evaluate_promotion_has_zero_tolerance_on_policy_violations():
    decision = evaluate_promotion(
        candidate=_summary(0.95, policy_rate=0.01),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
    )
    assert decision.promoted is False
    assert "policy_safety_compliance" in decision.reason


def test_evaluate_promotion_rejects_new_challenge_set_failures():
    decision = evaluate_promotion(
        candidate=_summary(0.95),
        baseline=_summary(0.90),
        challenge_set_new_failures=1,
    )
    assert decision.promoted is False
    assert "regression_challenge_set" in decision.reason


def test_calibration_is_monitored_not_blocking_by_default():
    decision = evaluate_promotion(
        candidate=_summary(0.95, ece=0.5),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
    )
    assert decision.promoted is True


def test_calibration_blocks_once_hard_gate_enabled():
    decision = evaluate_promotion(
        candidate=_summary(0.95, ece=0.5),
        baseline=_summary(0.90),
        challenge_set_new_failures=0,
        thresholds=PromotionThresholds(ece_hard_gate=True, max_ece=0.1),
    )
    assert decision.promoted is False
    assert "confidence_calibration" in decision.reason
