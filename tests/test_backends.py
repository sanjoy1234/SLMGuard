"""Proves the backend abstraction actually holds: both backends satisfy the
same ModelBackend contract, the registry resolves either by name, and the
CUDA/QLoRA stub fails loudly and specifically rather than silently or
generically. Does not require MLX or CUDA hardware to run."""

from __future__ import annotations

import pytest

from slmguard.backends import ModelBackend, get_backend
from slmguard.backends.base import validate_output
from slmguard.backends.cuda_qlora_backend import CudaQLoRABackend
from slmguard.backends.mlx_backend import MLXBackend
from slmguard.backends.types import LoadedModel, LoRAConfig, RawModelOutput


def test_both_backends_satisfy_the_interface():
    assert isinstance(get_backend("mlx"), ModelBackend)
    assert isinstance(get_backend("cuda_qlora"), ModelBackend)


def test_mlx_backend_is_the_phase1_default():
    backend = get_backend("mlx")
    assert isinstance(backend, MLXBackend)
    assert backend.name == "mlx"


def test_cuda_qlora_backend_is_registered_but_not_yet_runnable():
    backend = get_backend("cuda_qlora")
    assert isinstance(backend, CudaQLoRABackend)
    with pytest.raises(NotImplementedError, match="CudaQLoRABackend.load_model"):
        backend.load_model("some-version")


def test_unknown_backend_name_raises_with_available_list():
    with pytest.raises(ValueError, match="mlx"):
        get_backend("does-not-exist")


def test_validate_output_rejects_schema_invalid_raw_output():
    raw = RawModelOutput(text="not json", schema_valid=False, parse_error="boom")
    assert validate_output(raw) is None


def test_validate_output_parses_well_formed_recommendation():
    raw = RawModelOutput(
        text=(
            '{"alert_id": "a1", "action": "escalate_l2", "risk_score": 0.9, '
            '"confidence": 0.4, "policy_flags": ["high_amount"], '
            '"rationale": "Amount exceeds threshold with no prior history."}'
        ),
        schema_valid=True,
    )
    recommendation = validate_output(raw)
    assert recommendation is not None
    assert recommendation.action.value == "escalate_l2"


def test_mlx_fine_tune_passes_batch_size_max_seq_length_and_grad_checkpoint(monkeypatch, tmp_path):
    captured_cmd = {}

    def fake_run(cmd, check):
        captured_cmd["cmd"] = cmd

    monkeypatch.setattr("slmguard.backends.mlx_backend.subprocess.run", fake_run)

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages": []}\n')
    model = LoadedModel(version_id="some-model", backend_name="mlx", handle=(None, None))
    config = LoRAConfig(
        rank=8,
        alpha=16,
        epochs=3,
        learning_rate=1e-5,
        batch_size=2,
        max_seq_length=512,
        grad_checkpoint=True,
    )

    artifact = MLXBackend().fine_tune(model, dataset, config)

    cmd = captured_cmd["cmd"]
    assert cmd[cmd.index("--batch-size") + 1] == "2"
    assert cmd[cmd.index("--max-seq-length") + 1] == "512"
    assert "--grad-checkpoint" in cmd
    assert artifact.base_version_id == "some-model"


def test_mlx_fine_tune_omits_grad_checkpoint_flag_when_disabled(monkeypatch, tmp_path):
    captured_cmd = {}

    def fake_run(cmd, check):
        captured_cmd["cmd"] = cmd

    monkeypatch.setattr("slmguard.backends.mlx_backend.subprocess.run", fake_run)

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages": []}\n')
    model = LoadedModel(version_id="some-model", backend_name="mlx", handle=(None, None))
    config = LoRAConfig(
        rank=8,
        alpha=16,
        epochs=3,
        learning_rate=1e-5,
        batch_size=1,
        max_seq_length=512,
        grad_checkpoint=False,
    )

    MLXBackend().fine_tune(model, dataset, config)

    assert "--grad-checkpoint" not in captured_cmd["cmd"]
