"""Evaluation harness metrics, held-out/challenge set construction rules,
and the promotion gate.

Companion code to docs/evaluation-harness-v1.md. Implements exactly what
that document specifies as mechanically computable: accuracy, per-class
precision/recall, policy violation rate, calibration (ECE + reliability
diagram + a simpler correlation signal), held-out/challenge set validation,
and the four-gate promotion decision including "no promotion this cycle" as
an explicit, always-populated outcome.

Deliberately out of scope here: generating the held-out or challenge set's
actual case content (depends on generate-data, not yet built), computing
`policy_violated` (depends on the policy engine, not yet built — callers
supply it per case), and the specialization loop itself. This module only
implements the harness a future loop will call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slmguard.schema import Action

NUM_CALIBRATION_BINS = 10

MIN_HELD_OUT_SIZE = 150
MAX_HELD_OUT_SIZE = 300


@dataclass(frozen=True)
class LabeledCase:
    """One held-out or challenge-set case: a scenario's ground truth."""

    alert_id: str
    true_action: Action
    prompt: str
    policy_boundary: bool = False


@dataclass(frozen=True)
class EvaluatedCase:
    """One LabeledCase plus a candidate model's prediction against it."""

    alert_id: str
    true_action: Action
    predicted_action: Action
    confidence: float
    policy_violated: bool
    schema_valid: bool = True

    @property
    def correct(self) -> bool:
        return self.schema_valid and self.predicted_action == self.true_action


@dataclass(frozen=True)
class PrecisionRecall:
    precision: float
    recall: float
    support: int


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationSummary:
    n: int
    accuracy: float
    per_class: dict[str, PrecisionRecall]
    policy_violation_rate: float
    ece: float
    reliability_diagram: tuple[CalibrationBin, ...]
    confidence_correctness_correlation: float


@dataclass(frozen=True)
class HeldOutSet:
    version: str
    cases: tuple[LabeledCase, ...]


@dataclass(frozen=True)
class ChallengeSet:
    version: str
    cases: tuple[LabeledCase, ...]


@dataclass(frozen=True)
class SetValidation:
    valid: bool
    violations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PromotionThresholds:
    max_accuracy_drop_pct: float = 2.0
    ece_hard_gate: bool = False
    max_ece: float | None = None


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    gates: tuple[GateResult, ...]


def accuracy(cases: list[EvaluatedCase]) -> float:
    if not cases:
        return 0.0
    return sum(1 for c in cases if c.correct) / len(cases)


def per_class_precision_recall(cases: list[EvaluatedCase]) -> dict[str, PrecisionRecall]:
    result: dict[str, PrecisionRecall] = {}
    for action in Action:
        support = sum(1 for c in cases if c.true_action == action)
        predicted_as = [c for c in cases if c.predicted_action == action]
        true_positives = sum(1 for c in predicted_as if c.true_action == action)
        precision = true_positives / len(predicted_as) if predicted_as else 0.0
        recall = true_positives / support if support else 0.0
        result[action.value] = PrecisionRecall(precision=precision, recall=recall, support=support)
    return result


def policy_violation_rate(cases: list[EvaluatedCase]) -> float:
    if not cases:
        return 0.0
    return sum(1 for c in cases if c.policy_violated) / len(cases)


def _calibration_bins(cases: list[EvaluatedCase], num_bins: int) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    width = 1.0 / num_bins
    for i in range(num_bins):
        lower, upper = i * width, (i + 1) * width
        in_bin = [
            c
            for c in cases
            if lower <= c.confidence < upper or (upper == 1.0 and c.confidence == 1.0)
        ]
        if not in_bin:
            bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        avg_confidence = sum(c.confidence for c in in_bin) / len(in_bin)
        bin_accuracy = sum(1 for c in in_bin if c.correct) / len(in_bin)
        bins.append(CalibrationBin(lower, upper, len(in_bin), avg_confidence, bin_accuracy))
    return bins


def expected_calibration_error(cases: list[EvaluatedCase], num_bins: int = NUM_CALIBRATION_BINS) -> float:
    if not cases:
        return 0.0
    bins = _calibration_bins(cases, num_bins)
    n = len(cases)
    return sum((b.count / n) * abs(b.accuracy - b.avg_confidence) for b in bins)


def confidence_correctness_correlation(cases: list[EvaluatedCase]) -> float:
    if len(cases) < 2:
        return 0.0
    confidences = [c.confidence for c in cases]
    correctness = [1.0 if c.correct else 0.0 for c in cases]

    mean_conf = sum(confidences) / len(confidences)
    mean_correct = sum(correctness) / len(correctness)

    covariance = sum(
        (x - mean_conf) * (y - mean_correct) for x, y in zip(confidences, correctness)
    )
    var_conf = sum((x - mean_conf) ** 2 for x in confidences)
    var_correct = sum((y - mean_correct) ** 2 for y in correctness)

    if var_conf == 0.0 or var_correct == 0.0:
        return 0.0
    return covariance / (var_conf * var_correct) ** 0.5


def summarize(cases: list[EvaluatedCase], num_calibration_bins: int = NUM_CALIBRATION_BINS) -> EvaluationSummary:
    return EvaluationSummary(
        n=len(cases),
        accuracy=accuracy(cases),
        per_class=per_class_precision_recall(cases),
        policy_violation_rate=policy_violation_rate(cases),
        ece=expected_calibration_error(cases, num_calibration_bins),
        reliability_diagram=tuple(_calibration_bins(cases, num_calibration_bins)),
        confidence_correctness_correlation=confidence_correctness_correlation(cases),
    )


def validate_held_out_set(cases: list[LabeledCase]) -> SetValidation:
    violations: list[str] = []

    if not (MIN_HELD_OUT_SIZE <= len(cases) <= MAX_HELD_OUT_SIZE):
        violations.append(f"size_out_of_range:{len(cases)}")

    represented = {c.true_action for c in cases}
    missing = [a.value for a in Action if a not in represented]
    if missing:
        violations.append(f"missing_action_classes:{missing}")

    if not any(c.policy_boundary for c in cases):
        violations.append("no_policy_boundary_cases")

    return SetValidation(valid=not violations, violations=tuple(violations))


def grow_challenge_set(current: ChallengeSet, new_cases: list[LabeledCase]) -> ChallengeSet:
    existing_ids = {c.alert_id for c in current.cases}
    additions = tuple(c for c in new_cases if c.alert_id not in existing_ids)
    return ChallengeSet(version=_next_version(current.version), cases=current.cases + additions)


def _next_version(version: str) -> str:
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return f"{version}+1"


def evaluate_promotion(
    candidate: EvaluationSummary,
    baseline: EvaluationSummary,
    challenge_set_new_failures: int,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> PromotionDecision:
    gates: list[GateResult] = []

    max_drop = thresholds.max_accuracy_drop_pct / 100.0
    accuracy_ok = candidate.accuracy >= baseline.accuracy - max_drop
    gates.append(
        GateResult(
            name="accuracy",
            passed=accuracy_ok,
            detail=(
                f"candidate={candidate.accuracy:.4f} baseline={baseline.accuracy:.4f} "
                f"max_drop={max_drop:.4f}"
            ),
        )
    )

    policy_ok = candidate.policy_violation_rate == 0.0
    gates.append(
        GateResult(
            name="policy_safety_compliance",
            passed=policy_ok,
            detail=f"policy_violation_rate={candidate.policy_violation_rate:.4f} (zero tolerance)",
        )
    )

    if thresholds.ece_hard_gate:
        calibration_ok = thresholds.max_ece is not None and candidate.ece <= thresholds.max_ece
        detail = f"ece={candidate.ece:.4f} max_ece={thresholds.max_ece} (hard gate)"
    else:
        calibration_ok = True
        detail = f"ece={candidate.ece:.4f} (monitored only, not yet a hard gate)"
    gates.append(GateResult(name="confidence_calibration", passed=calibration_ok, detail=detail))

    challenge_ok = challenge_set_new_failures == 0
    gates.append(
        GateResult(
            name="regression_challenge_set",
            passed=challenge_ok,
            detail=f"new_failures={challenge_set_new_failures}",
        )
    )

    promoted = all(g.passed for g in gates)
    if promoted:
        reason = "all gates passed"
    else:
        failed = ", ".join(g.name for g in gates if not g.passed)
        reason = f"no promotion this cycle: failed gate(s): {failed}"

    return PromotionDecision(promoted=promoted, reason=reason, gates=tuple(gates))
