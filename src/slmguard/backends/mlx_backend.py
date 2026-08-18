"""Phase 1 backend: MLX / mlx-lm on Apple Silicon.

This is the pragmatic choice for the current hardware (unified memory, one
stack for quantized inference + LoRA, no cloud GPU needed) — not the
project's architecture. Nothing outside this file may import `mlx` or
`mlx_lm` directly; every other component talks to `ModelBackend`.

`fine_tune` / `fuse_adapters` shell out to the `mlx_lm.lora` / `mlx_lm.fuse`
CLI entry points rather than a pinned internal Python API, since those are
the stable, documented surface of mlx-lm across versions. LoRA memory
footprint at 3B has not yet been validated on this hardware — see
"Known Technical Risks" in docs/technical-build-plan-v5.md — so `config`
values here are not yet tuned defaults, just the shape of the call.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from slmguard.backends.base import ModelBackend
from slmguard.backends.types import (
    AdapterArtifact,
    LoadedModel,
    LoRAConfig,
    ModelVersion,
    RawModelOutput,
)


class MLXBackend(ModelBackend):
    name = "mlx"

    def load_model(self, version: str) -> LoadedModel:
        from mlx_lm import load

        model, tokenizer = load(version)
        return LoadedModel(version_id=version, backend_name=self.name, handle=(model, tokenizer))

    def generate(self, model: LoadedModel, prompt: str) -> RawModelOutput:
        from mlx_lm import generate

        mlx_model, tokenizer = model.handle
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text = generate(mlx_model, tokenizer, prompt=chat_prompt, max_tokens=512, verbose=False)
        return RawModelOutput(text=text, schema_valid=True)

    def fine_tune(self, model: LoadedModel, dataset: Path, config: LoRAConfig) -> AdapterArtifact:
        adapter_dir = dataset.parent / "adapters" / model.version_id
        adapter_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--model",
            model.version_id,
            "--train",
            "--data",
            str(dataset),
            "--adapter-path",
            str(adapter_dir),
            "--num-layers",
            str(len(config.target_modules)) if config.target_modules else "4",
            "--iters",
            str(config.epochs),
            "--learning-rate",
            str(config.learning_rate),
        ]
        subprocess.run(cmd, check=True)

        training_data_hash = _hash_file(dataset)
        return AdapterArtifact(
            path=adapter_dir,
            base_version_id=model.version_id,
            backend_name=self.name,
            training_data_hash=training_data_hash,
        )

    def fuse_adapters(self, model: LoadedModel, adapter: AdapterArtifact) -> ModelVersion:
        fused_dir = adapter.path.parent.parent / "fused" / adapter.path.name
        fused_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            model.version_id,
            "--adapter-path",
            str(adapter.path),
            "--save-path",
            str(fused_dir),
        ]
        subprocess.run(cmd, check=True)

        return ModelVersion(
            version_id=adapter.path.name,
            parent_version_id=model.version_id,
            backend_name=self.name,
            fused_weights_path=fused_dir,
        )


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
