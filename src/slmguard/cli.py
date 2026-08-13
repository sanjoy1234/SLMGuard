"""CLI orchestration — `generate-data`, `run-baseline`, `specialize`,
`evaluate`, `promote`, per docs/technical-build-plan-v5.md. No workflow
engine at this scale; the CLI is the whole orchestration layer.

Every subcommand resolves a backend via `slmguard.backends.get_backend`
using the name in config/backend.yaml — never a hardcoded framework import.
Swapping `backend: mlx` for `backend: cuda_qlora` in that file is meant to be
the entire cost of retargeting these commands to the production backend.
"""

from __future__ import annotations

from pathlib import Path

import click

from slmguard.backends import get_backend
from slmguard.backends.base import validate_output
from slmguard.config import load_settings

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
    rubric itself is an "Immediate Next Step" in the build plan, not written
    yet, so bulk generation intentionally has nothing to run against."""
    raise click.ClickException(
        "generate-data: not implemented yet. Write the Synthetic Data Quality "
        "Rubric first (see 'Immediate Next Steps' in technical-build-plan-v5.md)."
    )


@main.command("run-baseline")
@click.option("--prompt", required=True, help="A single fraud-triage prompt to test end-to-end.")
@click.pass_context
def run_baseline(ctx: click.Context, prompt: str) -> None:
    """Smoke-test the active backend: load the configured model version and
    generate one recommendation. This is the thinnest possible slice through
    ModelBackend.load_model -> ModelBackend.generate -> schema validation."""
    settings = ctx.obj["settings"]
    backend = get_backend(settings.backend)
    click.echo(f"Backend: {backend.name} | model: {settings.model_version}")

    model = backend.load_model(settings.model_version)
    raw = backend.generate(model, prompt)
    recommendation = validate_output(raw)

    if recommendation is None:
        click.echo(f"SCHEMA FAILURE (would auto-escalate). Raw output:\n{raw.text}", err=True)
        raise SystemExit(1)

    click.echo(recommendation.model_dump_json(indent=2))


@main.command("specialize")
@click.pass_context
def specialize(ctx: click.Context) -> None:
    """Run one specialization cycle: pull traces, filter, convert, fine-tune,
    fuse. Depends on the audit store and trace-filtering logic, neither of
    which exist yet — this is next after run-baseline is validated end-to-end."""
    raise click.ClickException("specialize: not implemented yet — depends on the audit store.")


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
