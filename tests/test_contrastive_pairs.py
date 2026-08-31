"""Proves the contrastive decline-vs-escalate_l2 spec generator produces a
correctly tagged, evenly matched set of guidance pairs, and that pairing
verification is robust to a missing/dropped side (a real failure mode: a
teacher call can raise and drop one side of a pair entirely) rather than
assuming strict positional ordering."""

from __future__ import annotations

import json

from slmguard.contrastive_pairs import (
    ANCHORS,
    contrastive_generation_specs,
    verify_contrastive_pairing,
)
from slmguard.generate_data import GeneratedExample
from slmguard.rubric import SyntheticExample
from slmguard.teacher.types import TeacherMetadata

_META = TeacherMetadata(teacher_name="fake", model_id="fake-model", generated_at="2026-08-23T00:00:00+00:00")


def _example(action: str, tags: tuple[str, ...]) -> GeneratedExample:
    return GeneratedExample(
        example=SyntheticExample(
            scenario="A" * 60,
            recommendation_json=json.dumps(
                {
                    "alert_id": "SYN-0001",
                    "action": action,
                    "risk_score": 0.5,
                    "confidence": 0.8,
                    "policy_flags": [],
                    "rationale": "B" * 30,
                }
            ),
            diversity_tags=tags,
        ),
        metadata=_META,
    )


def test_contrastive_generation_specs_covers_every_anchor_both_sides():
    specs = contrastive_generation_specs()

    assert len(specs) == 2 * len(ANCHORS)
    decline_tags = {s.diversity_tags[-1] for s in specs if s.diversity_tags[-1].endswith(":decline")}
    escalate_tags = {s.diversity_tags[-1] for s in specs if s.diversity_tags[-1].endswith(":escalate")}
    assert len(decline_tags) == len(ANCHORS)
    assert len(escalate_tags) == len(ANCHORS)


def test_contrastive_specs_reference_the_shared_scenario_on_both_sides():
    specs = contrastive_generation_specs()
    by_anchor: dict[str, list[str]] = {}
    for anchor in ANCHORS:
        pair = [s for s in specs if any(anchor.anchor_id in t for t in s.diversity_tags)]
        assert len(pair) == 2
        assert anchor.shared_context in pair[0].guidance
        assert anchor.shared_context in pair[1].guidance


def test_verify_contrastive_pairing_counts_clean_pairs():
    first_two = ANCHORS[:2]
    generated = [
        _example("decline", ("policy_boundary", f"pair:{first_two[0].anchor_id}:decline")),
        _example("escalate_l2", ("policy_boundary", f"pair:{first_two[0].anchor_id}:escalate")),
        _example("decline", ("policy_boundary", f"pair:{first_two[1].anchor_id}:decline")),
        _example("request_more_info", ("policy_boundary", f"pair:{first_two[1].anchor_id}:escalate")),
    ]

    report = verify_contrastive_pairing(generated)

    assert report.clean_pairs == 1
    assert report.escalate_side_mislabeled == 1
    assert report.decline_side_mislabeled == 0


def test_verify_contrastive_pairing_is_robust_to_a_missing_side():
    # simulates a real failure mode: the escalate-guided call for this
    # anchor raised and was dropped entirely, never generated at all --
    # pairing must not assume strict positional alternation.
    anchor = ANCHORS[0]
    generated = [_example("decline", ("policy_boundary", f"pair:{anchor.anchor_id}:decline"))]

    report = verify_contrastive_pairing(generated)

    assert report.clean_pairs == 0
    assert report.decline_side_mislabeled == 0
    assert report.escalate_side_mislabeled == 0
