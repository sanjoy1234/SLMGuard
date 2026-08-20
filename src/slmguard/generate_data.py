"""Bulk synthetic data generation: teacher -> rubric gate -> reject/
regenerate -> metadata-tagged output, per docs/technical-build-plan-v5.md
"Synthetic Data Quality Process" and docs/synthetic-data-quality-rubric-v1.md.

Deliberately has no `AuditStore` dependency anywhere in this module -- it
cannot leak real trace/customer data into a teacher prompt because it never
has access to one. Every `GenerationSpec` sent to a `Teacher` is a static,
diversity-focused instruction, never derived from real case content.

Batch rejection is strict, per the rubric doc: a batch below the pass-rate
threshold is regenerated from scratch (fresh teacher calls), never patched
example-by-example, and never folded in as-is after exhausting attempts --
`generate_batch` returns `accepted=False` rather than silently accepting a
below-threshold batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from slmguard.rubric import DEFAULT_PASS_RATE, BatchScore, SyntheticExample, score_batch
from slmguard.teacher.base import Teacher
from slmguard.teacher.types import GenerationSpec, TeacherMetadata

DEFAULT_CATEGORY_GUIDANCE = {
    "transaction_type": (
        "Vary channel and transaction type: card-present, card-not-present/"
        "online, ACH transfer, wire transfer, or a mixed sequence like a small "
        "test charge followed by a large purchase."
    ),
    "customer_segment": (
        "Vary the account/customer profile: long-tenure low-risk, brand-new "
        "account, previously-flagged account, or a high-volume business account."
    ),
    "edge_case": (
        "Focus on a near-threshold amount, a first-time transaction pattern, "
        "or conflicting signals (e.g. a high amount but a strong account history)."
    ),
    "policy_boundary": (
        "Construct a scenario sitting exactly on a forced-escalation or "
        "hard-decline boundary -- e.g. confidence right at a threshold, or "
        "risk indicators right at a decline/approve boundary."
    ),
}


def default_generation_specs(n_per_category: int = 3) -> list[GenerationSpec]:
    """One spec per (diversity category, repetition) pair, covering every
    category in slmguard.rubric.REQUIRED_DIVERSITY_CATEGORIES."""
    return [
        GenerationSpec(diversity_tags=(category,), guidance=guidance)
        for category, guidance in DEFAULT_CATEGORY_GUIDANCE.items()
        for _ in range(n_per_category)
    ]


@dataclass(frozen=True)
class GeneratedExample:
    """A rubric-passing example paired with the teacher metadata that
    produced it -- provenance travels with the data, not stored separately."""

    example: SyntheticExample
    metadata: TeacherMetadata


@dataclass(frozen=True)
class BatchGenerationResult:
    accepted: bool
    generated: tuple[GeneratedExample, ...]
    attempts: int
    last_batch_score: BatchScore


def generate_batch(
    teacher: Teacher,
    specs: list[GenerationSpec],
    *,
    max_attempts: int = 3,
    pass_rate_threshold: float = DEFAULT_PASS_RATE,
) -> BatchGenerationResult:
    """Generate one full batch, scoring it against the rubric each attempt.
    Below-threshold batches are regenerated from scratch, not patched --
    returns accepted=False if `max_attempts` is exhausted without clearing
    the threshold."""
    last_score: BatchScore | None = None
    for attempt in range(1, max_attempts + 1):
        teacher_examples = [teacher.generate(spec) for spec in specs]
        candidates = [
            SyntheticExample(
                scenario=te.scenario,
                recommendation_json=te.recommendation_json,
                diversity_tags=te.diversity_tags,
            )
            for te in teacher_examples
        ]
        last_score = score_batch(candidates, pass_rate_threshold=pass_rate_threshold)

        if last_score.accepted:
            passing = tuple(
                GeneratedExample(example=ex, metadata=te.metadata)
                for ex, te, result in zip(candidates, teacher_examples, last_score.results)
                if result.passed
            )
            return BatchGenerationResult(
                accepted=True, generated=passing, attempts=attempt, last_batch_score=last_score
            )

    return BatchGenerationResult(
        accepted=False, generated=(), attempts=max_attempts, last_batch_score=last_score
    )


def write_generated_batch(result: BatchGenerationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for item in result.generated:
            record = {
                "scenario": item.example.scenario,
                "recommendation_json": item.example.recommendation_json,
                "diversity_tags": list(item.example.diversity_tags),
                "teacher": {
                    "teacher_name": item.metadata.teacher_name,
                    "model_id": item.metadata.model_id,
                    "generated_at": item.metadata.generated_at,
                },
            }
            f.write(json.dumps(record) + "\n")
    return output_path
