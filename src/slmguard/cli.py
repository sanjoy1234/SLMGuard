"""CLI orchestration — `generate-data`, `run-baseline`, `specialize`,
`evaluate`, `promote`, per docs/technical-build-plan-v5.md. No workflow
engine at this scale; the CLI is the whole orchestration layer.

Every subcommand resolves a backend via `slmguard.backends.get_backend`
using the name in config/backend.yaml — never a hardcoded framework import.
Swapping `backend: mlx` for `backend: cuda_qlora` in that file is meant to be
the entire cost of retargeting these commands to the production backend.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

from slmguard.audit import get_audit_store
from slmguard.audit.types import TraceRecord
from slmguard.backends import get_backend
from slmguard.backends.base import validate_output
from slmguard.backends.types import LoRAConfig
from slmguard.config import load_settings
from slmguard.evaluation import ChallengeSet, HeldOutSet, LabeledCase, validate_held_out_set
from slmguard.policy import apply_policy
from slmguard.schema import Action
from slmguard.specialization import DEFAULT_CONFIDENCE_THRESHOLD, run_cycle

DEFAULT_CONFIG = Path("config/backend.yaml")


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to backend.yaml",
)
@click.pass_context
def main(ctx: click.Context, config_path: Path) -> None:
    ctx.ensure_object(dict)
    ctx.obj["settings"] = load_settings(config_path)


@main.command("generate-data")
@click.pass_context
def generate_data(ctx: click.Context) -> None:
    """Generate synthetic fraud-alert scenarios via the teacher model, gated
    by the quality rubric and rejection process. Not yet implemented — the
    rubric and its automated checks now exist (docs/synthetic-data-quality-
    rubric-v1.md, slmguard.rubric.score_batch), but there is no teacher-model
    generation pipeline for it to gate yet."""
    raise click.ClickException(
        "generate-data: not implemented yet. The Synthetic Data Quality Rubric "
        "exists (docs/synthetic-data-quality-rubric-v1.md); still need the "
        "teacher-model generation pipeline to score against it."
    )


@main.command("run-baseline")
@click.option("--prompt", required=True, help="A single fraud-triage prompt to test end-to-end.")
@click.pass_context
def run_baseline(ctx: click.Context, prompt: str) -> None:
    """Smoke-test the active backend: load the configured model version,
    generate one recommendation, validate its schema, and apply the policy
    engine. This is the thinnest possible slice through ModelBackend.load_model
    -> ModelBackend.generate -> schema validation -> policy engine -- the
    control plane's actual decision path, not just the raw model call."""
    settings = ctx.obj["settings"]
    backend = get_backend(settings.backend)
    click.echo(f"Backend: {backend.name} | model: {settings.model_version}")

    model = backend.load_model(settings.model_version)
    raw = backend.generate(model, prompt)
    recommendation = validate_output(raw)
    policy_decision = apply_policy(recommendation) if recommendation else None

    audit_store = get_audit_store(settings.audit_store.backend, settings.audit_store.location)
    trace = TraceRecord(
        trace_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        alert_id=recommendation.alert_id if recommendation else "unknown",
        prompt=prompt,
        raw_output=raw.text,
        schema_valid=recommendation is not None,
        recommendation_json=recommendation.model_dump_json() if recommendation else None,
        final_action=(
            policy_decision.final_action.value if policy_decision else "escalate"
        ),
        confidence=recommendation.confidence if recommendation else None,
        model_version=settings.model_version,
        backend_name=backend.name,
        policy_version=policy_decision.policy_version if policy_decision else None,
        policy_overridden=policy_decision.overridden if policy_decision else False,
        policy_violated_rule_ids=(
            json.dumps(list(policy_decision.violated_rule_ids)) if policy_decision else "[]"
        ),
    )
    audit_store.append(trace)

    if recommendation is None:
        click.echo(f"SCHEMA FAILURE (would auto-escalate). Raw output:\n{raw.text}", err=True)
        raise SystemExit(1)

    click.echo(recommendation.model_dump_json(indent=2))
    if policy_decision.overridden:
        click.echo(
            f"POLICY OVERRIDE ({policy_decision.policy_version}): rule(s) "
            f"{list(policy_decision.violated_rule_ids)} forced final_action="
            f"{policy_decision.final_action.value} (model recommended "
            f"{policy_decision.original_action.value})"
        )


def _load_labeled_cases(path: Path) -> tuple[str, tuple[LabeledCase, ...]]:
    data = json.loads(path.read_text())
    cases = tuple(
        LabeledCase(
            alert_id=c["alert_id"],
            true_action=Action(c["true_action"]),
            prompt=c["prompt"],
            policy_boundary=c.get("policy_boundary", False),
        )
        for c in data["cases"]
    )
    return data.get("version", "unversioned"), cases


@main.command("specialize")
@click.option(
    "--held-out-set",
    "held_out_set_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help=(
        "JSON file: {version, cases: [{alert_id, true_action, prompt, "
        "policy_boundary}]}. Not the real frozen Phase 1 set unless it "
        "actually meets slmguard.evaluation.validate_held_out_set."
    ),
)
@click.option(
    "--challenge-set",
    "challenge_set_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="JSON file, same shape as --held-out-set.",
)
@click.option(
    "--work-dir",
    "work_dir",
    type=click.Path(path_type=Path),
    default=Path("data/specialization_cycles"),
    show_default=True,
    help="Directory this cycle's training dataset/adapter/fused model are written into.",
)
@click.option(
    "--confidence-threshold",
    "confidence_threshold",
    type=float,
    default=DEFAULT_CONFIDENCE_THRESHOLD,
    show_default=True,
    help="Traces with confidence below this are selected as low-confidence.",
)
@click.pass_context
def specialize(
    ctx: click.Context,
    held_out_set_path: Path,
    challenge_set_path: Path,
    work_dir: Path,
    confidence_threshold: float,
) -> None:
    """Run one specialization cycle end to end: select useful traces from the
    audit store, convert the rubric-passing ones into training examples,
    fine-tune, fuse, evaluate against the held-out and challenge sets, and
    emit an explicit promotion decision -- including "no promotion this
    cycle" as a first-class outcome, not a failure."""
    settings = ctx.obj["settings"]
    backend = get_backend(settings.backend)
    audit_store = get_audit_store(settings.audit_store.backend, settings.audit_store.location)

    held_out_version, held_out_cases = _load_labeled_cases(held_out_set_path)
    held_out_set = HeldOutSet(version=held_out_version, cases=held_out_cases)
    challenge_version, challenge_cases = _load_labeled_cases(challenge_set_path)
    challenge_set = ChallengeSet(version=challenge_version, cases=challenge_cases)

    held_out_validation = validate_held_out_set(list(held_out_set.cases))
    if not held_out_validation.valid:
        click.echo(
            "WARNING: held-out set does not meet the frozen Phase 1 set "
            f"requirements ({', '.join(held_out_validation.violations)}) -- "
            "proceeding anyway since no real Phase 1 held-out set exists yet. "
            "Treat this cycle's promotion decision as a bootstrap/demo result, "
            "not a production signal.",
            err=True,
        )

    lora_config = LoRAConfig(
        rank=settings.lora.rank,
        alpha=settings.lora.alpha,
        epochs=settings.lora.epochs,
        learning_rate=settings.lora.learning_rate,
        batch_size=settings.lora.batch_size,
        max_seq_length=settings.lora.max_seq_length,
        grad_checkpoint=settings.lora.grad_checkpoint,
    )

    cycle_id = uuid.uuid4().hex[:8]
    result = run_cycle(
        store=audit_store,
        backend=backend,
        base_model_version=settings.model_version,
        held_out_set=held_out_set,
        challenge_set=challenge_set,
        lora_config=lora_config,
        work_dir=work_dir / cycle_id,
        confidence_threshold=confidence_threshold,
    )

    click.echo(
        f"Pool: {len(result.pool.examples)} rubric-passing example(s) from "
        f"{result.pool.traces_selected} selected trace(s) "
        f"({result.pool.traces_scanned} scanned, "
        f"{result.pool.traces_unconvertible} unconvertible)."
    )
    if result.baseline_summary is not None:
        click.echo(f"Baseline accuracy: {result.baseline_summary.accuracy:.3f}")
    if result.candidate_summary is not None:
        click.echo(f"Candidate accuracy: {result.candidate_summary.accuracy:.3f}")
    if result.challenge_report is not None:
        cr = result.challenge_report
        click.echo(
            f"Challenge set: {cr.new_failures} new failure(s) vs. baseline "
            f"(candidate failed {cr.candidate_failures}/{cr.total}, "
            f"baseline failed {cr.baseline_failures}/{cr.total})."
        )
        if cr.new_failure_alert_ids:
            click.echo(f"  New failures: {', '.join(cr.new_failure_alert_ids)}")
    click.echo(f"Decision: {result.reason}")

    if not result.promoted:
        raise SystemExit(1)


@main.command("evaluate")
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Score a candidate model against the frozen held-out set and the
    regression/challenge set. Depends on those sets existing, which depends
    on generate-data running first."""
    raise click.ClickException("evaluate: not implemented yet — depends on generate-data.")


@main.command("promote")
@click.pass_context
def promote(ctx: click.Context) -> None:
    """Apply the promotion gate to an evaluated candidate and update the
    model registry. Depends on evaluate."""
    raise click.ClickException("promote: not implemented yet — depends on evaluate.")


if __name__ == "__main__":
    main()
