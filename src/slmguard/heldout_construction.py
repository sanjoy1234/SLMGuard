"""Building the Phase 1 held-out/challenge sets from generate-data output,
per docs/technical-build-plan-v5.md's held-out set rules and
docs/evaluation-harness-v1.md: 150-300 cases, all 4 action classes
represented, >=1 policy-boundary case, drawn from rubric-passing data.

`true_action` for a constructed case is the teacher's own assessed action
(parsed from its `recommendation_json`), never the per-action guidance used
to nudge generation toward covering all 4 classes. The guidance asks the
teacher to aim for a specific action; what the teacher actually decides for
its own invented scenario is the ground truth this project trusts -- per
the build plan's own framing of the teacher as "the larger model" whose
judgment establishes the baseline specialist's training signal. If a
guidance target and the teacher's own decision disagree, the teacher's
decision wins; forcing an action the teacher didn't actually conclude would
manufacture false ground truth, not honest labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from slmguard.evaluation import ChallengeSet, HeldOutSet, LabeledCase
from slmguard.generate_data import ACTION_GUIDANCE, weighted_generation_specs
from slmguard.schema import Action
from slmguard.teacher.types import GenerationSpec

__all__ = [
    "ACTION_GUIDANCE",
    "ConversionStats",
    "build_prompt",
    "generated_batches_to_cases",
    "stratified_generation_specs",
    "write_challenge_set",
    "write_held_out_set",
]


def stratified_generation_specs(n_per_action: int = 40) -> list[GenerationSpec]:
    """Equal per-action case of `slmguard.generate_data.weighted_generation_specs`
    -- one spec per (target action, diversity category) slot, cycling
    through the required diversity categories within each action so the
    resulting batch covers both axes: action-class balance for
    stratification, and category coverage for the rubric's diversity
    checklist."""
    return weighted_generation_specs({action: n_per_action for action in ACTION_GUIDANCE})


@dataclass(frozen=True)
class ConversionStats:
    total: int
    by_action: dict[str, int]
    policy_boundary_count: int


def generated_batches_to_cases(
    batch_paths: list[Path], *, policy_boundary_categories: tuple[str, ...] = ("policy_boundary",)
) -> tuple[list[LabeledCase], ConversionStats]:
    """Read one or more generate-data JSONL batches and convert every
    record into a LabeledCase. `alert_id` and `prompt` come straight from
    the generated record; `true_action` is parsed from the teacher's own
    recommendation_json; `policy_boundary` is True iff the record was
    generated with a policy_boundary diversity focus."""
    cases: list[LabeledCase] = []
    by_action: dict[str, int] = {}
    policy_boundary_count = 0

    for path in batch_paths:
        for line in path.read_text().strip().splitlines():
            record = json.loads(line)
            recommendation = json.loads(record["recommendation_json"])
            action = recommendation["action"]
            is_policy_boundary = any(
                tag in policy_boundary_categories for tag in record["diversity_tags"]
            )
            cases.append(
                LabeledCase(
                    alert_id=recommendation["alert_id"],
                    true_action=Action(action),
                    prompt=build_prompt(record["scenario"]),
                    policy_boundary=is_policy_boundary,
                )
            )
            by_action[action] = by_action.get(action, 0) + 1
            policy_boundary_count += is_policy_boundary

    return cases, ConversionStats(
        total=len(cases), by_action=by_action, policy_boundary_count=policy_boundary_count
    )


def build_prompt(scenario: str) -> str:
    """Wrap a generated scenario in the same schema-instruction envelope
    every real triage prompt uses (see run-baseline), so held-out/challenge
    evaluation exercises the model under realistic conditions."""
    return (
        "You are a fraud-alert triage system. Given the alert below, "
        "respond with ONLY a single JSON object matching exactly this "
        "schema, with no other text before or after it:\n"
        '{"alert_id": string, "action": one of ["approve","decline",'
        '"escalate_l2","request_more_info"], "risk_score": float 0.0-1.0, '
        '"confidence": float 0.0-1.0, "policy_flags": array of strings, '
        '"rationale": string, max 1000 chars}\n\n'
        f"Alert:\n{scenario}\n\nRespond with the JSON object only."
    )


def write_held_out_set(cases: list[LabeledCase], version: str, path: Path) -> Path:
    return _write_case_file(cases, version, path)


def write_challenge_set(cases: list[LabeledCase], version: str, path: Path) -> Path:
    return _write_case_file(cases, version, path)


def _write_case_file(cases: list[LabeledCase], version: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "cases": [
            {
                "alert_id": c.alert_id,
                "true_action": c.true_action.value,
                "prompt": c.prompt,
                "policy_boundary": c.policy_boundary,
            }
            for c in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
