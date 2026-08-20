"""Proves the policy rules engine actually enforces what it claims: each
rule kind fires exactly when it should, multiple simultaneous violations are
all reported, a clean recommendation passes through untouched, and every
override lands on escalate_l2 with the violating rule ids recorded -- never
a silent pass-through and never an approve/decline injected by policy."""

from __future__ import annotations

from slmguard.policy import (
    DEFAULT_POLICY_RULESET,
    PolicyRule,
    PolicyRuleSet,
    RuleKind,
    apply_policy,
)
from slmguard.schema import Action, Recommendation


def _recommendation(**overrides) -> Recommendation:
    fields = dict(
        alert_id="ALERT-1",
        action=Action.APPROVE,
        risk_score=0.2,
        confidence=0.9,
        policy_flags=[],
        rationale="Routine, low-risk transaction matching prior pattern.",
    )
    fields.update(overrides)
    return Recommendation(**fields)


def test_clean_recommendation_is_not_overridden():
    rec = _recommendation(action=Action.APPROVE, risk_score=0.2, confidence=0.9)
    decision = apply_policy(rec)
    assert decision.overridden is False
    assert decision.final_action == Action.APPROVE
    assert decision.violated_rule_ids == ()
    assert decision.policy_version == "policy-v1"


def test_low_confidence_forces_escalation():
    rec = _recommendation(action=Action.APPROVE, confidence=0.4)
    decision = apply_policy(rec)
    assert decision.overridden is True
    assert decision.final_action == Action.ESCALATE_L2
    assert decision.original_action == Action.APPROVE
    assert "forced_escalation_low_confidence" in decision.violated_rule_ids


def test_low_confidence_already_escalating_is_not_a_violation():
    rec = _recommendation(action=Action.ESCALATE_L2, confidence=0.4)
    decision = apply_policy(rec)
    assert "forced_escalation_low_confidence" not in decision.violated_rule_ids


def test_high_risk_cannot_be_silently_approved():
    rec = _recommendation(action=Action.APPROVE, risk_score=0.85, confidence=0.9)
    decision = apply_policy(rec)
    assert decision.overridden is True
    assert decision.final_action == Action.ESCALATE_L2
    assert "no_silent_approve_high_risk" in decision.violated_rule_ids


def test_high_risk_decline_is_not_a_violation_of_that_rule():
    rec = _recommendation(action=Action.DECLINE, risk_score=0.85, confidence=0.9)
    decision = apply_policy(rec)
    assert "no_silent_approve_high_risk" not in decision.violated_rule_ids


def test_low_risk_cannot_be_declined():
    rec = _recommendation(action=Action.DECLINE, risk_score=0.1, confidence=0.9)
    decision = apply_policy(rec)
    assert decision.overridden is True
    assert decision.final_action == Action.ESCALATE_L2
    assert "no_low_risk_decline" in decision.violated_rule_ids


def test_multiple_simultaneous_violations_are_all_reported():
    # confidence too low AND risk_score too high paired with approve
    rec = _recommendation(action=Action.APPROVE, risk_score=0.9, confidence=0.2)
    decision = apply_policy(rec)
    assert decision.overridden is True
    assert set(decision.violated_rule_ids) == {
        "forced_escalation_low_confidence",
        "no_silent_approve_high_risk",
    }


def test_custom_ruleset_is_versioned_independently():
    custom = PolicyRuleSet(
        version="policy-test-v9",
        rules=(
            PolicyRule(
                rule_id="custom_rule",
                kind=RuleKind.CONFIDENCE_BELOW_FORCES_ESCALATION,
                threshold=0.99,
                description="test rule",
            ),
        ),
    )
    rec = _recommendation(action=Action.APPROVE, confidence=0.5)
    decision = apply_policy(rec, ruleset=custom)
    assert decision.policy_version == "policy-test-v9"
    assert decision.overridden is True
    assert decision.violated_rule_ids == ("custom_rule",)


def test_default_ruleset_has_three_rules():
    assert len(DEFAULT_POLICY_RULESET.rules) == 3
    assert DEFAULT_POLICY_RULESET.version == "policy-v1"
