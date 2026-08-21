"""Building a materially larger specialization training pool via
generate-data, with deliberate per-class weighting -- strong decline and
escalate_l2 representation in particular, per the build order that found
both classes at 0% precision/recall on the real held-out set -- and strict
separation from the frozen eval_sets/ held-out and challenge sets.

Two real issues found while wiring this, fixed here rather than left
latent:

1. `generate_data.generate_batch` produces `SyntheticExample.scenario` as
   the teacher's *bare* scenario text -- fine for held-out/challenge set
   construction (`heldout_construction.build_prompt` wraps it there), but
   wrong for training data as-is: real training examples from
   `specialization.convert_trace` use the *fully wrapped* schema-instruction
   prompt, the same shape the model is actually evaluated and used against.
   Feeding the model bare scenario text during training while every
   inference-time call sees the full wrapped prompt would train it on the
   wrong input distribution. Every example here is re-wrapped via
   `heldout_construction.build_prompt` before being packaged into the pool.

2. Teacher-generated `alert_id`s are not guaranteed unique (format
   "SYN-<4 digits>" collides across independent generation calls -- observed
   directly in this project's own generated batches). Alert-id equality
   alone is not a reliable overlap check, so separation from the eval sets
   is enforced on exact scenario text as the primary signal, with alert-id
   overlap checked and reported separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from slmguard.generate_data import (
    GeneratedExample,
    generate_batch,
    weighted_generation_specs,
    write_examples_jsonl,
)
from slmguard.heldout_construction import build_prompt
from slmguard.rubric import BatchScore, SyntheticExample, score_batch
from slmguard.specialization import SpecializationPool
from slmguard.teacher.base import Teacher


@dataclass(frozen=True)
class PoolGenerationReport:
    target_counts: dict[str, int]
    total_specs: int
    total_generated: int
    excluded_for_scenario_overlap: int
    excluded_for_alert_id_overlap: int
    by_true_action: dict[str, int]
    chunk_accepted: dict[str, bool]
    chunk_attempts: dict[str, int]


def generate_specialization_pool(
    teacher: Teacher,
    counts: dict[str, int],
    *,
    exclude_prompts: frozenset[str] = frozenset(),
    exclude_alert_ids: frozenset[str] = frozenset(),
    max_attempts: int = 3,
) -> tuple[SpecializationPool, PoolGenerationReport, list[GeneratedExample]]:
    """Generate a class-weighted training pool. `counts` maps action name to
    how many examples to target for that action (the teacher's own decided
    action is still what gets recorded -- see generate_data/heldout_construction
    for why guidance is a nudge, not an override). Generated per-action in
    separate chunks so a rubric rejection only costs regenerating one
    action's chunk, not the whole pool.

    `exclude_prompts`/`exclude_alert_ids` should be the frozen held-out and
    challenge sets' scenario text and alert_ids -- any generated example
    matching either is dropped before it ever reaches the returned pool,
    never folded in with a "close enough" judgment call.

    Returns the pool (with re-wrapped, training-ready prompts), a report,
    and the raw kept `GeneratedExample`s (with teacher metadata) for
    writing a governance artifact alongside the pool."""
    chunk_accepted: dict[str, bool] = {}
    chunk_attempts: dict[str, int] = {}
    all_generated: list[GeneratedExample] = []
    total_specs = 0

    for action, count in counts.items():
        specs = weighted_generation_specs({action: count})
        total_specs += len(specs)
        result = generate_batch(teacher, specs, max_attempts=max_attempts)
        chunk_accepted[action] = result.accepted
        chunk_attempts[action] = result.attempts
        if result.accepted:
            all_generated.extend(result.generated)

    # Wrap every scenario into the real schema-instruction prompt *before*
    # the overlap check -- exclude_prompts holds wrapped prompts (what's
    # actually stored in eval_sets/*.json), so comparing bare scenario text
    # against it would never match even a genuine duplicate.
    wrapped_all = [
        GeneratedExample(
            example=SyntheticExample(
                scenario=build_prompt(g.example.scenario),
                recommendation_json=g.example.recommendation_json,
                diversity_tags=g.example.diversity_tags,
            ),
            metadata=g.metadata,
        )
        for g in all_generated
    ]

    excluded_scenario = 0
    excluded_alert_id = 0
    kept: list[GeneratedExample] = []
    for g in wrapped_all:
        recommendation = json.loads(g.example.recommendation_json)
        alert_id = recommendation.get("alert_id")
        if g.example.scenario in exclude_prompts:
            excluded_scenario += 1
            continue
        if alert_id in exclude_alert_ids:
            excluded_alert_id += 1
            continue
        kept.append(g)

    wrapped_examples = tuple(g.example for g in kept)
    rubric_score: BatchScore = score_batch(list(wrapped_examples))

    by_true_action: dict[str, int] = {}
    for g in kept:
        action = json.loads(g.example.recommendation_json)["action"]
        by_true_action[action] = by_true_action.get(action, 0) + 1

    pool = SpecializationPool(
        examples=wrapped_examples,
        rubric_score=rubric_score,
        traces_scanned=total_specs,
        traces_selected=len(all_generated),
        traces_unconvertible=excluded_scenario + excluded_alert_id,
    )
    report = PoolGenerationReport(
        target_counts=dict(counts),
        total_specs=total_specs,
        total_generated=len(all_generated),
        excluded_for_scenario_overlap=excluded_scenario,
        excluded_for_alert_id_overlap=excluded_alert_id,
        by_true_action=by_true_action,
        chunk_accepted=chunk_accepted,
        chunk_attempts=chunk_attempts,
    )
    return pool, report, kept


def load_pool_from_jsonl(path: Path) -> SpecializationPool:
    """Reconstruct a SpecializationPool from a governance artifact written
    by `write_pool_report` (or `generate_data.write_examples_jsonl`) --
    lets pool generation and the cycle that consumes it run as separate
    processes/scripts without re-generating anything. Rubric score is
    recomputed on the loaded examples rather than trusted from the file, so
    a hand-edited or corrupted pool file can't silently claim a pass rate
    it no longer has."""
    examples = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                SyntheticExample(
                    scenario=record["scenario"],
                    recommendation_json=record["recommendation_json"],
                    diversity_tags=tuple(record["diversity_tags"]),
                )
            )
    rubric_score = score_batch(examples)
    return SpecializationPool(
        examples=tuple(examples),
        rubric_score=rubric_score,
        traces_scanned=len(examples),
        traces_selected=len(examples),
        traces_unconvertible=0,
    )


def load_eval_overlap_sets(*eval_set_paths: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Load the scenario-text and alert_id sets to exclude from one or more
    eval_sets/*.json files (held_out_set.json, challenge_set.json). Returns
    the *wrapped* prompt text (what's actually stored in those files),
    matching the wrapped form the pool's own scenarios are checked in."""
    prompts: set[str] = set()
    alert_ids: set[str] = set()
    for path in eval_set_paths:
        data = json.loads(path.read_text())
        for case in data["cases"]:
            prompts.add(case["prompt"])
            alert_ids.add(case["alert_id"])
    return frozenset(prompts), frozenset(alert_ids)


def write_pool_report(
    pool: SpecializationPool,
    report: PoolGenerationReport,
    kept: list[GeneratedExample],
    *,
    examples_path: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    """Write two governance artifacts: the pool's examples with teacher
    metadata attached to every line, and a separate JSON report of the
    generation/filtering process (target counts, exclusions, per-chunk
    rubric outcomes) -- both real filesystem artifacts, not just numbers
    quoted in a chat response."""
    write_examples_jsonl(kept, examples_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "target_counts": report.target_counts,
                "total_specs": report.total_specs,
                "total_generated": report.total_generated,
                "excluded_for_scenario_overlap": report.excluded_for_scenario_overlap,
                "excluded_for_alert_id_overlap": report.excluded_for_alert_id_overlap,
                "final_pool_size": len(pool.examples),
                "by_true_action": report.by_true_action,
                "chunk_accepted": report.chunk_accepted,
                "chunk_attempts": report.chunk_attempts,
                "rubric_pass_rate": pool.rubric_score.pass_rate,
                "rubric_accepted": pool.rubric_score.accepted,
            },
            indent=2,
        )
    )
    return examples_path, report_path
