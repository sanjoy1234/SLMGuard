"""The specialization loop: select traces -> convert to privacy-safe
training examples -> fine-tune -> fuse -> evaluate -> promote.

Companion code to docs/technical-build-plan-v5.md "Specialization Loop" and
docs/specialization-loop-v1.md. First implementation, scoped tight on
purpose: one working loop path, no extra platform features.

Deliberate, documented simplifications versus the full plan (flagged here,
not glossed over):

- Trace selection covers low-confidence, escalated (schema-failure), and
  policy-overridden cases -- all three of the plan's step-2 criteria, now
  that `slmguard.policy` exists and the audit lineage carries
  `policy_overridden` per trace.
- Conversion only succeeds for traces that already carry a valid
  ground-truth label (schema_valid). A schema-failure trace has no
  correct answer to imitate -- the plan's real answer for those is a
  larger model or human resolving the case and that resolution becoming
  the training example, which needs a teacher-model/human-review path
  that doesn't exist yet. Those traces are counted and surfaced
  (`SpecializationPool.traces_unconvertible`), not silently dropped. A
  policy-overridden trace *does* convert, but trains toward the control
  plane's actual decision (`final_action`), never the model's own
  rule-violating raw claim -- see `convert_trace`.
- Privacy-safe conversion is mechanical PII-pattern redaction (reusing
  `slmguard.rubric.PII_PATTERNS`), not the LLM-based "careful rewriting"
  the plan names as the eventual Phase 1 approach. Same honesty standard
  as the plan's own admission that this step is "careful rewriting plus
  review... not yet a formal privacy guarantee."
- `EvaluatedCase.policy_violated` is computed for real via
  `slmguard.policy.apply_policy` now, no longer hardcoded False.
- `run_cycle` takes `HeldOutSet`/`ChallengeSet` as trusted inputs and does
  not itself call `slmguard.evaluation.validate_held_out_set`. Freezing
  and validating a held-out set is a separate, one-time curation step
  (not yet built as its own command) that happens before a set is ever
  handed to this loop -- re-validating it on every cycle run would be the
  wrong place for that check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from slmguard.audit.base import AuditStore
from slmguard.audit.types import TraceRecord
from slmguard.backends.base import ModelBackend, validate_output
from slmguard.backends.types import LoadedModel, LoRAConfig
from slmguard.evaluation import (
    ChallengeSet,
    EvaluatedCase,
    EvaluationSummary,
    HeldOutSet,
    LabeledCase,
    PromotionDecision,
    PromotionThresholds,
    evaluate_promotion,
    summarize,
)
from slmguard.policy import apply_policy
from slmguard.rubric import PII_PATTERNS, BatchScore, SyntheticExample, score_batch
from slmguard.schema import Action

MIN_POOL_SIZE = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def redact_pii(text: str) -> str:
    """Mechanical PII redaction ahead of a trace entering the specialization
    pool. Not the LLM-based "careful rewriting" the build plan describes as
    the eventual approach -- a documented simplification, not a silent one."""
    redacted = text
    for kind, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED:{kind}]", redacted)
    return redacted


@dataclass(frozen=True)
class TraceSelection:
    selected: tuple[TraceRecord, ...]
    total_scanned: int


def select_traces(
    store: AuditStore,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    limit: int = 500,
) -> TraceSelection:
    """Pull recent traces and filter to the ones that matter, per the build
    plan's Specialization Loop step 2: low-confidence cases, cases that were
    escalated (schema failure), and cases the policy engine overrode."""
    candidates = store.query(limit=limit)
    selected = tuple(
        c.trace
        for c in candidates
        if not c.trace.schema_valid
        or c.trace.policy_overridden
        or (c.trace.confidence is not None and c.trace.confidence < confidence_threshold)
    )
    return TraceSelection(selected=selected, total_scanned=len(candidates))


def convert_trace(trace: TraceRecord) -> SyntheticExample | None:
    """Convert one trace into a candidate training example, or None if the
    trace has no valid ground-truth label to convert (a schema failure).

    For a policy-overridden trace, the target label is the control plane's
    actual decision (`trace.final_action`), not the model's own raw claim in
    `recommendation_json` -- that raw claim is exactly the rule-violating
    output that got overridden, and training toward it would teach the model
    to repeat the mistake the policy engine exists to catch."""
    if not trace.schema_valid or trace.recommendation_json is None:
        return None
    recommendation = json.loads(trace.recommendation_json)
    if trace.policy_overridden:
        recommendation = {**recommendation, "action": trace.final_action}
    return SyntheticExample(
        scenario=redact_pii(trace.prompt),
        recommendation_json=json.dumps(recommendation),
        diversity_tags=(),
    )


@dataclass(frozen=True)
class SpecializationPool:
    examples: tuple[SyntheticExample, ...]
    rubric_score: BatchScore
    traces_scanned: int
    traces_selected: int
    traces_unconvertible: int


def build_specialization_pool(
    store: AuditStore,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    limit: int = 500,
) -> SpecializationPool:
    """Select useful traces, convert the ones with a valid label, and keep
    only the rubric-passing results -- the specialization pool is drawn from
    rubric-passing data, same as the plan requires for the held-out set."""
    selection = select_traces(store, confidence_threshold=confidence_threshold, limit=limit)
    converted = [convert_trace(t) for t in selection.selected]
    unconvertible = sum(1 for c in converted if c is None)
    candidates = [c for c in converted if c is not None]

    rubric_score = score_batch(candidates)
    passing = tuple(
        ex for ex, result in zip(candidates, rubric_score.results) if result.passed
    )

    return SpecializationPool(
        examples=passing,
        rubric_score=rubric_score,
        traces_scanned=selection.total_scanned,
        traces_selected=len(selection.selected),
        traces_unconvertible=unconvertible,
    )


def write_training_dataset(
    pool: SpecializationPool, dataset_dir: Path, *, min_valid_size: int = 1
) -> Path:
    """Write the pool's rubric-passing examples as chat-formatted train/valid
    JSONL files, the shape `MLXBackend.fine_tune` (mlx_lm.lora) expects.

    `min_valid_size` should be at least the LoRA config's batch_size --
    mlx_lm.lora requires the validation set to have at least batch_size
    examples, and a plain 80/20 split can easily leave fewer than that for
    small pools. Below `2 * min_valid_size` total examples, train and valid
    are the same set -- a small-pool bootstrap tradeoff, not real held-out
    validation, but it keeps the loop runnable instead of crashing."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples = list(pool.examples)

    if len(examples) < 2 * min_valid_size:
        train_examples, valid_examples = examples, examples
    else:
        split = min(max(1, round(len(examples) * 0.8)), len(examples) - min_valid_size)
        train_examples, valid_examples = examples[:split], examples[split:]

    def _write(path: Path, items: list[SyntheticExample]) -> None:
        with path.open("w") as f:
            for ex in items:
                recommendation = json.loads(ex.recommendation_json)
                record = {
                    "messages": [
                        {"role": "user", "content": ex.scenario},
                        {"role": "assistant", "content": json.dumps(recommendation)},
                    ]
                }
                f.write(json.dumps(record) + "\n")

    _write(dataset_dir / "train.jsonl", train_examples)
    _write(dataset_dir / "valid.jsonl", valid_examples)
    return dataset_dir


def evaluate_case(backend: ModelBackend, model: LoadedModel, case: LabeledCase) -> EvaluatedCase:
    """Run one held-out/challenge case through the model. A schema failure
    counts as escalate_l2 (matching the real control-plane failure mode:
    schema failure -> hard escalation) but is flagged schema_valid=False so
    it never counts as a lucky correct guess.

    `predicted_action` stays the model's *raw* action, deliberately never the
    post-policy final_action -- this metric is about whether the model
    itself improved, and a bad model bailed out by a policy override would
    otherwise show inflated accuracy, masking whether specialization is
    actually working. `policy_violated` is exactly the metric the harness
    already defines it as ("would have violated policy had the control
    plane not intervened") -- computed for real via apply_policy now, no
    longer hardcoded False."""
    raw = backend.generate(model, case.prompt)
    recommendation = validate_output(raw)
    if recommendation is None:
        return EvaluatedCase(
            alert_id=case.alert_id,
            true_action=case.true_action,
            predicted_action=Action.ESCALATE_L2,
            confidence=0.0,
            policy_violated=False,
            schema_valid=False,
        )
    policy_decision = apply_policy(recommendation)
    return EvaluatedCase(
        alert_id=case.alert_id,
        true_action=case.true_action,
        predicted_action=recommendation.action,
        confidence=recommendation.confidence,
        policy_violated=policy_decision.overridden,
        schema_valid=True,
    )


def evaluate_model(
    backend: ModelBackend, model: LoadedModel, cases: list[LabeledCase]
) -> EvaluationSummary:
    return summarize([evaluate_case(backend, model, c) for c in cases])


@dataclass(frozen=True)
class ChallengeSetReport:
    """True regression vs. baseline is the primary signal -- a case the
    baseline already failed isn't a regression when the candidate fails it
    too. Absolute candidate failure count is kept as secondary detail: a
    candidate that fails every case is worth knowing about even if none of
    those are "new" relative to an equally-broken baseline."""

    total: int
    baseline_failures: int
    candidate_failures: int
    new_failures: int
    new_failure_alert_ids: tuple[str, ...]


def evaluate_challenge_set(
    backend: ModelBackend,
    baseline_model: LoadedModel,
    candidate_model: LoadedModel,
    challenge_set: ChallengeSet,
) -> ChallengeSetReport:
    """Run the challenge set against both models and report true regression:
    a case the baseline got right that the candidate gets wrong. A case
    neither model gets right is not a regression -- it's a pre-existing gap
    the specialization loop didn't cause and can't be blamed for here."""
    baseline_evaluated = [evaluate_case(backend, baseline_model, c) for c in challenge_set.cases]
    candidate_evaluated = [evaluate_case(backend, candidate_model, c) for c in challenge_set.cases]

    new_failure_ids = tuple(
        case.alert_id
        for case, b, c in zip(challenge_set.cases, baseline_evaluated, candidate_evaluated)
        if b.correct and not c.correct
    )

    return ChallengeSetReport(
        total=len(challenge_set.cases),
        baseline_failures=sum(1 for e in baseline_evaluated if not e.correct),
        candidate_failures=sum(1 for e in candidate_evaluated if not e.correct),
        new_failures=len(new_failure_ids),
        new_failure_alert_ids=new_failure_ids,
    )


@dataclass(frozen=True)
class CycleResult:
    pool: SpecializationPool
    promoted: bool
    reason: str
    decision: PromotionDecision | None
    baseline_summary: EvaluationSummary | None
    candidate_summary: EvaluationSummary | None
    challenge_report: ChallengeSetReport | None


def run_cycle(
    *,
    store: AuditStore,
    backend: ModelBackend,
    base_model_version: str,
    held_out_set: HeldOutSet,
    challenge_set: ChallengeSet,
    lora_config: LoRAConfig,
    work_dir: Path,
    min_pool_size: int = MIN_POOL_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    promotion_thresholds: PromotionThresholds = PromotionThresholds(),
) -> CycleResult:
    """Run one full specialization cycle: select -> convert -> fine-tune ->
    fuse -> evaluate -> promote. Always returns a CycleResult with an
    explicit reason -- "no promotion this cycle" is a first-class outcome,
    not the absence of one, whether the cycle stops early for insufficient
    signal or runs the full pipeline and fails a gate."""
    pool = build_specialization_pool(
        store, confidence_threshold=confidence_threshold, limit=500
    )

    if len(pool.examples) < min_pool_size:
        return CycleResult(
            pool=pool,
            promoted=False,
            reason=(
                f"no promotion this cycle: insufficient specialization signal "
                f"({len(pool.examples)} rubric-passing example(s), need >= {min_pool_size})"
            ),
            decision=None,
            baseline_summary=None,
            candidate_summary=None,
            challenge_report=None,
        )

    base_model = backend.load_model(base_model_version)
    baseline_summary = evaluate_model(backend, base_model, list(held_out_set.cases))

    dataset_dir = write_training_dataset(
        pool, work_dir / "dataset", min_valid_size=lora_config.batch_size
    )
    adapter = backend.fine_tune(base_model, dataset_dir, lora_config)
    fused_version = backend.fuse_adapters(base_model, adapter)
    fused_model = backend.load_model(str(fused_version.fused_weights_path))

    candidate_summary = evaluate_model(backend, fused_model, list(held_out_set.cases))
    challenge_report = evaluate_challenge_set(backend, base_model, fused_model, challenge_set)

    decision = evaluate_promotion(
        candidate=candidate_summary,
        baseline=baseline_summary,
        challenge_set_new_failures=challenge_report.new_failures,
        thresholds=promotion_thresholds,
    )

    return CycleResult(
        pool=pool,
        promoted=decision.promoted,
        reason=decision.reason,
        decision=decision,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        challenge_report=challenge_report,
    )
