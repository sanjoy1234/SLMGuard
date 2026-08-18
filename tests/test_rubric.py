"""Proves the automated half of the Synthetic Data Quality Rubric actually
catches what docs/synthetic-data-quality-rubric-v1.md says it should:
schema-invalid labels, degenerate/boilerplate text, PII leakage, and batches
that fall below the pass-rate threshold or miss diversity coverage."""

from __future__ import annotations

from slmguard.rubric import (
    DEFAULT_PASS_RATE,
    SyntheticExample,
    score_batch,
    score_example,
)

_GOOD_SCENARIO = (
    "Visa ending 4417, 3 years of history, avg 2 transactions/week. A $12.00 "
    "test charge at a gas station was followed 40 minutes later by a "
    "$4,850.00 purchase at an electronics retailer abroad."
)
_GOOD_RATIONALE = (
    "The short gap between a small test charge and a large international "
    "purchase, with no prior travel notice, is a classic card-testing "
    "pattern that warrants escalation."
)


def _recommendation_json(**overrides) -> str:
    import json

    fields = {
        "alert_id": "ALERT-1",
        "action": "escalate_l2",
        "risk_score": 0.85,
        "confidence": 0.7,
        "policy_flags": ["card_testing_pattern"],
        "rationale": _GOOD_RATIONALE,
    }
    fields.update(overrides)
    return json.dumps(fields)


def _example(**overrides) -> SyntheticExample:
    scenario = overrides.pop("scenario", _GOOD_SCENARIO)
    diversity_tags = overrides.pop("diversity_tags", ())
    recommendation_json = overrides.pop("recommendation_json", None) or _recommendation_json(
        **overrides
    )
    return SyntheticExample(
        scenario=scenario, recommendation_json=recommendation_json, diversity_tags=diversity_tags
    )


def test_well_formed_example_passes():
    result = score_example(_example())
    assert result.passed is True
    assert result.violations == ()


def test_schema_invalid_recommendation_fails():
    result = score_example(_example(recommendation_json="not json"))
    assert result.passed is False
    assert any(v.startswith("schema_invalid") for v in result.violations)


def test_short_scenario_fails():
    result = score_example(_example(scenario="Card fraud."))
    assert result.passed is False
    assert "scenario_too_short" in result.violations


def test_short_rationale_fails():
    result = score_example(_example(rationale="Looks bad."))
    assert result.passed is False
    assert "rationale_too_short" in result.violations


def test_rationale_copying_scenario_fails():
    result = score_example(_example(rationale=_GOOD_SCENARIO))
    assert result.passed is False
    assert "rationale_copies_scenario" in result.violations


def test_ssn_leak_is_caught():
    result = score_example(_example(scenario=_GOOD_SCENARIO + " SSN 219-09-9999."))
    assert result.passed is False
    assert "possible_pii_leak:ssn" in result.violations


def test_unmasked_card_number_is_caught():
    result = score_example(
        _example(scenario=_GOOD_SCENARIO + " Full card number 4111111111111111.")
    )
    assert result.passed is False
    assert "possible_pii_leak:unmasked_card_number" in result.violations


def test_email_leak_is_caught():
    result = score_example(_example(scenario=_GOOD_SCENARIO + " Contact jane.doe@example.com."))
    assert result.passed is False
    assert "possible_pii_leak:email" in result.violations


def test_batch_below_threshold_is_rejected():
    good = _example(diversity_tags=("transaction_type",))
    bad = _example(scenario="Card fraud.", diversity_tags=("transaction_type",))
    batch = score_batch([good, bad, bad, bad], pass_rate_threshold=DEFAULT_PASS_RATE)
    assert batch.total == 4
    assert batch.passed == 1
    assert batch.pass_rate == 0.25
    assert batch.accepted is False


def test_batch_above_threshold_is_accepted():
    good = _example(
        diversity_tags=(
            "transaction_type",
            "customer_segment",
            "edge_case",
            "policy_boundary",
        )
    )
    batch = score_batch([good] * 10, pass_rate_threshold=DEFAULT_PASS_RATE)
    assert batch.pass_rate == 1.0
    assert batch.accepted is True
    assert batch.missing_diversity_categories == ()


def test_batch_reports_missing_diversity_categories():
    only_transaction_type = _example(diversity_tags=("transaction_type",))
    batch = score_batch([only_transaction_type] * 5)
    assert "customer_segment" in batch.missing_diversity_categories
    assert "edge_case" in batch.missing_diversity_categories
    assert "policy_boundary" in batch.missing_diversity_categories


def test_empty_batch_is_never_accepted():
    batch = score_batch([])
    assert batch.total == 0
    assert batch.accepted is False
