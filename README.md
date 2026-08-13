# SLM Guard

Constrained small-language-model specialization platform for BFSI recommendation workflows. Phase 1 vertical: fraud-alert triage.

See `docs/technical-build-plan-v5.md` for the full design.

## Backend architecture

The model layer sits behind a `ModelBackend` interface (`src/slmguard/backends/base.py`) — `load_model`, `generate`, `fine_tune`, `fuse_adapters`. Nothing else in the codebase imports a model framework directly.

- **Phase 1 backend: `MLXBackend`.** Runs on Apple Silicon via `mlx`/`mlx-lm`. This is what's actually used for development right now.
- **Confirmed production backend: `CudaQLoRABackend`.** Targets Linux + NVIDIA/AMD GPU via Hugging Face `transformers` + `peft` (bitsandbytes QLoRA). Stubbed against the same interface (see the parity checklist in `cuda_qlora_backend.py`) — not implemented yet, since there's no CUDA hardware in the current build environment, but the seam exists so implementing it later doesn't touch the control plane, specialization loop, or evaluation harness.

Which backend is active is one line in `config/backend.yaml`.

## Setup

This project's `.venv` was created with `uv` and has no `pip` binary — install with `uv`, not plain `pip` (plain `pip install` on a machine with both `uv`-venvs and Anaconda around can silently install into the wrong environment):

```
uv pip install -e ".[mlx,dev]" -p .venv/bin/python
```

## Running

The editable install's `.pth` isn't currently being picked up by this venv (unresolved — works fine with an explicit `PYTHONPATH`), so until that's root-caused, prefix commands with `PYTHONPATH=src`:

```
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
PYTHONPATH=src .venv/bin/python -m slmguard.cli run-baseline --prompt "..."
```

## CLI

`generate-data`, `run-baseline`, `specialize`, `evaluate`, `promote` — see `src/slmguard/cli.py`. Only `run-baseline` is implemented so far; the others raise a clear "not implemented, depends on X" error until their prerequisites (data rubric, audit store, held-out set) exist.
