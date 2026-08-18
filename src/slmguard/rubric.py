"""Synthetic Data Quality Rubric — the mechanically-checkable subset.

Companion code to docs/synthetic-data-quality-rubric-v1.md. Everything here
is verifiable without a human or LLM judge: schema validity, degenerate or
boilerplate text, PII-leakage patterns, and batch-level diversity coverage.
The rubric's more subjective dimensions (is the scenario realistic, is the
label actually correct for it) require the human/teacher-model review step
described in the doc and are deliberately not encoded here — a mechanical
check that can't verify semantic correctness would be false confidence, not
a real gate.

Non-negotiable per docs/technical-build-plan-v5.md, "Synthetic Data Quality
Process": every generated batch is scored against this rubric, and batches
below DEFAULT_PASS_RATE are rejected and regenerated, never folded in as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from slmguard.schema import Recommendation

DEFAULT_PASS_RATE = 0.85
MIN_SCENARIO_LENGTH = 40
MIN_RATIONALE_LENGTH = 20

REQUIRED_DIVERSITY_CATEGORIES = (
    "transaction_type",
    "customer_segment",
    "edge_case",
    "policy_boundary",
)

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "unmasked_card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone_number": re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
}


@dataclass(frozen=True)
class SyntheticExample:
    """One candidate scenario/label pair, pending rubric evaluation."""

    scenario: str
    recommendation_json: str
    diversity_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RubricResult:
    """Outcome of checking one example against the automated rubric checks."""

    passed: bool
    violations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BatchScore:
    """Outcome of scoring a whole batch: pass rate, acceptance, and gaps."""

    total: int
    passed: int
    pass_rate: float
    accepted: bool
    missing_diversity_categories: tuple[str, ...]
    results: tuple[RubricResult, ...]


def score_example(example: SyntheticExample) -> RubricResult:
    violations: list[str] = []

    try:
        recommendation = Recommendation.model_validate_json(example.recommendation_json)
    except Exception as exc:
        return RubricResult(passed=False, violations=(f"schema_invalid: {exc}",))

    if len(example.scenario.strip()) < MIN_SCENARIO_LENGTH:
        violations.append("scenario_too_short")

    if len(recommendation.rationale.strip()) < MIN_RATIONALE_LENGTH:
        violations.append("rationale_too_short")

    if recommendation.rationale.strip() == example.scenario.strip():
        violations.append("rationale_copies_scenario")

    for pii_kind, pattern in _PII_PATTERNS.items():
        if pattern.search(example.scenario) or pattern.search(recommendation.rationale):
            violations.append(f"possible_pii_leak:{pii_kind}")

    return RubricResult(passed=not violations, violations=tuple(violations))


def score_batch(
    examples: list[SyntheticExample],
    *,
    pass_rate_threshold: float = DEFAULT_PASS_RATE,
) -> BatchScore:
    if not examples:
        return BatchScore(
            total=0,
            passed=0,
            pass_rate=0.0,
            accepted=False,
            missing_diversity_categories=REQUIRED_DIVERSITY_CATEGORIES,
            results=(),
        )

    results = tuple(score_example(e) for e in examples)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / len(examples)

    covered = {tag for e in examples for tag in e.diversity_tags}
    missing = tuple(c for c in REQUIRED_DIVERSITY_CATEGORIES if c not in covered)

    return BatchScore(
        total=len(examples),
        passed=passed,
        pass_rate=pass_rate,
        accepted=pass_rate >= pass_rate_threshold,
        missing_diversity_categories=missing,
        results=results,
    )
