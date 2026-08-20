"""Proves the generate-data orchestration actually enforces the rubric's
reject/regenerate discipline: a batch that clears the pass-rate threshold
is accepted with teacher metadata attached to every example; a batch that
never clears it after max_attempts is rejected outright, never folded in
partially. Uses a FakeTeacher -- no network, no real teacher model."""

from __future__ import annotations

import json

from slmguard.generate_data import (
    DEFAULT_CATEGORY_GUIDANCE,
    default_generation_specs,
    generate_batch,
    write_generated_batch,
)
from slmguard.teacher.base import Teacher
from slmguard.teacher.types import GenerationSpec, TeacherExample, TeacherMetadata

GOOD_SCENARIO = (
    "A synthetic fraud scenario with enough detail to pass the rubric's "
    "minimum length check, describing a card transaction and its context."
)
GOOD_RATIONALE = (
    "This synthetic rationale references the scenario's specifics enough "
    "to justify the labeled action and clear the rubric's minimum length."
)


def _recommendation_json(**overrides) -> str:
    fields = {
        "alert_id": "SYN-0001",
        "action": "approve",
        "risk_score": 0.2,
        "confidence": 0.9,
        "policy_flags": [],
        "rationale": GOOD_RATIONALE,
    }
    fields.update(overrides)
    return json.dumps(fields)


class FakeTeacher(Teacher):
    name = "fake"

    def __init__(self, good: bool = True):
        self._good = good
        self.call_count = 0

    def generate(self, spec: GenerationSpec) -> TeacherExample:
        self.call_count += 1
        scenario = GOOD_SCENARIO if self._good else "too short"
        recommendation_json = _recommendation_json() if self._good else _recommendation_json(rationale="bad")
        return TeacherExample(
            scenario=scenario,
            recommendation_json=recommendation_json,
            diversity_tags=spec.diversity_tags,
            metadata=TeacherMetadata(
                teacher_name="fake", model_id="fake-model", generated_at="2026-08-20T00:00:00+00:00"
            ),
        )


def test_default_generation_specs_cover_every_required_category():
    specs = default_generation_specs(n_per_category=2)
    covered = {tag for s in specs for tag in s.diversity_tags}
    assert covered == set(DEFAULT_CATEGORY_GUIDANCE.keys())
    assert len(specs) == 2 * len(DEFAULT_CATEGORY_GUIDANCE)


def test_generate_batch_accepts_a_clean_batch():
    teacher = FakeTeacher(good=True)
    specs = default_generation_specs(n_per_category=2)

    result = generate_batch(teacher, specs)

    assert result.accepted is True
    assert result.attempts == 1
    assert len(result.generated) == len(specs)
    assert all(g.metadata.teacher_name == "fake" for g in result.generated)
    assert teacher.call_count == len(specs)


def test_generate_batch_rejects_after_max_attempts_without_folding_in():
    teacher = FakeTeacher(good=False)
    specs = default_generation_specs(n_per_category=2)

    result = generate_batch(teacher, specs, max_attempts=2)

    assert result.accepted is False
    assert result.generated == ()
    assert result.attempts == 2
    assert result.last_batch_score.accepted is False
    # every attempt re-generates the whole batch, never patches in place
    assert teacher.call_count == 2 * len(specs)


def test_write_generated_batch_includes_teacher_metadata(tmp_path):
    teacher = FakeTeacher(good=True)
    specs = default_generation_specs(n_per_category=1)
    result = generate_batch(teacher, specs)

    out_path = write_generated_batch(result, tmp_path / "batch.jsonl")

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == len(specs)
    record = json.loads(lines[0])
    assert record["teacher"]["teacher_name"] == "fake"
    assert record["teacher"]["model_id"] == "fake-model"
    assert "diversity_tags" in record
