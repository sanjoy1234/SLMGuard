"""Confirmed production backend: CUDA + QLoRA (Hugging Face `transformers` +
`peft`, bitsandbytes 4-bit quantization). This is the environment most
enterprise / BFSI deployments actually run on (Linux + NVIDIA/AMD GPU) and is
the backend this project is committed to shipping — not a maybe-someday
option kept alive by "on the roadmap" language.

Not implemented during Phase 1: there is no CUDA-capable machine in the
current build environment (MacBook Air / Mac Mini, Apple Silicon), and
writing untested training code against hardware that isn't available would
just produce code nobody can validate. What exists here instead is the real
`ModelBackend` contract, wired up, with each method's production
implementation named and scoped — so the parity gap with `MLXBackend` is a
tracked list, not a surprise discovered later.

Parity checklist before this backend is production-ready:
  - [ ] load_model: `transformers.AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))`
  - [ ] generate: HF `generate()` (or vLLM for serving), constrained via `outlines` for schema-guaranteed JSON
  - [ ] fine_tune: `peft.LoraConfig` + `peft` QLoRA training loop (or Unsloth/Axolotl as a faster implementation of the same contract)
  - [ ] fuse_adapters: `peft` `merge_and_unload()`
  - [ ] Parity test: same recommendation schema, same promotion-gate metrics, run against the same held-out set as `MLXBackend`, comparing accuracy/calibration between backends before either is trusted interchangeably.
"""

from __future__ import annotations

from pathlib import Path

from slmguard.backends.base import ModelBackend
from slmguard.backends.types import (
    AdapterArtifact,
    LoadedModel,
    LoRAConfig,
    ModelVersion,
    RawModelOutput,
)


class CudaQLoRABackend(ModelBackend):
    name = "cuda_qlora"

    def load_model(self, version: str) -> LoadedModel:
        raise NotImplementedError(
            "CudaQLoRABackend.load_model: requires transformers + bitsandbytes on "
            "CUDA hardware, not yet available in this build environment. "
            "See parity checklist in this file's module docstring."
        )

    def generate(self, model: LoadedModel, prompt: str) -> RawModelOutput:
        raise NotImplementedError(
            "CudaQLoRABackend.generate: production implementation should use HF "
            "generate() or vLLM, with `outlines` for schema-constrained decoding."
        )

    def fine_tune(self, model: LoadedModel, dataset: Path, config: LoRAConfig) -> AdapterArtifact:
        raise NotImplementedError(
            "CudaQLoRABackend.fine_tune: production implementation should use "
            "peft.LoraConfig + bitsandbytes QLoRA (or Unsloth/Axolotl)."
        )

    def fuse_adapters(self, model: LoadedModel, adapter: AdapterArtifact) -> ModelVersion:
        raise NotImplementedError(
            "CudaQLoRABackend.fuse_adapters: production implementation should use "
            "peft's merge_and_unload()."
        )
