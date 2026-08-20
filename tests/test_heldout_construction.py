"""Proves the held-out/challenge set construction utilities do what they
claim: stratified specs actually cover all 4 actions x every diversity
category, conversion trusts the teacher's own assessed action (not the
guidance target) as ground truth, policy_boundary is set from the
diversity-category tag, and round-tripping through the same JSON file
format the CLI's --held-out-set/--challenge-set loader expects works."""

from __future__ import annotations

import json

from slmguard.evaluation import validate_held_out_set
from slmguard.generate_data import DEFAULT_CATEGORY_GUIDANCE
from slmguard.heldout_construction import (
    ACTION_GUIDANCE,
    generated_batches_to_cases,
    stratified_generation_specs,
    write_held_out_set,
)
from slmguard.schema import Action


def test_stratified_generation_specs_cover_every_action_and_category():
    specs = stratified_generation_specs(n_per_action=8)
    assert len(specs) == 8 * len(ACTION_GUIDANCE)
    covered_categories = {tag for s in specs for tag in s.diversity_tags}
    assert covered_categories == set(DEFAULT_CATEGORY_GUIDANCE.keys())
    for action, guidance in ACTION_GUIDANCE.items():
        assert any(guidance in s.guidance for s in specs)


def _write_batch(tmp_path, records, name="batch.jsonl"):
    path = tmp_path / name
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _record(alert_id, action, tags):
    return {
        "scenario": f"Scenario for {alert_id}.",
        "recommendation_json": json.dumps(
            {
                "alert_id": alert_id,
                "action": action,
                "risk_score": 0.5,
                "confidence": 0.8,
                "policy_flags": [],
                "rationale": "Synthetic rationale.",
            }
        ),
        "diversity_tags": tags,
        "teacher": {"teacher_name": "fake", "model_id": "fake-model", "generated_at": "now"},
    }


def test_conversion_uses_teachers_own_action_not_the_guidance_target(tmp_path):
    # Guidance targeted "approve" but the teacher itself decided "decline" --
    # the teacher's own decision must win.
    batch = _write_batch(tmp_path, [_record("SYN-1", "decline", ["edge_case"])])

    cases, stats = generated_batches_to_cases([batch])

    assert cases[0].true_action == Action.DECLINE
    assert stats.by_action == {"decline": 1}


def test_conversion_marks_policy_boundary_from_diversity_tag(tmp_path):
    batch = _write_batch(
        tmp_path,
        [
            _record("SYN-1", "escalate_l2", ["policy_boundary"]),
            _record("SYN-2", "approve", ["transaction_type"]),
        ],
    )

    cases, stats = generated_batches_to_cases([batch])

    boundary_cases = {c.alert_id: c.policy_boundary for c in cases}
    assert boundary_cases == {"SYN-1": True, "SYN-2": False}
    assert stats.policy_boundary_count == 1


def test_conversion_wraps_scenario_in_the_real_prompt_envelope(tmp_path):
    batch = _write_batch(tmp_path, [_record("SYN-1", "approve", [])])
    cases, _ = generated_batches_to_cases([batch])
    assert "fraud-alert triage system" in cases[0].prompt
    assert "Scenario for SYN-1." in cases[0].prompt


def test_write_held_out_set_round_trips_through_the_cli_loader_format(tmp_path):
    batch = _write_batch(
        tmp_path,
        [
            _record("SYN-1", "approve", ["transaction_type"]),
            _record("SYN-2", "decline", ["policy_boundary"]),
        ],
    )
    cases, _ = generated_batches_to_cases([batch])

    out_path = write_held_out_set(cases, version="test-v1", path=tmp_path / "held_out.json")

    loaded = json.loads(out_path.read_text())
    assert loaded["version"] == "test-v1"
    assert len(loaded["cases"]) == 2
    assert {c["alert_id"] for c in loaded["cases"]} == {"SYN-1", "SYN-2"}
    assert loaded["cases"][1]["policy_boundary"] is True


def test_a_large_enough_stratified_batch_passes_validate_held_out_set(tmp_path):
    records = []
    actions = [a.value for a in Action]
    for i in range(160):
        action = actions[i % len(actions)]
        tags = ["policy_boundary"] if i == 0 else ["transaction_type"]
        records.append(_record(f"SYN-{i}", action, tags))
    batch = _write_batch(tmp_path, records)

    cases, stats = generated_batches_to_cases([batch])
    result = validate_held_out_set(cases)

    assert stats.total == 160
    assert result.valid is True
    assert result.violations == ()
