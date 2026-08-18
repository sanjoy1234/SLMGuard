from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class LoRASettings(BaseModel):
    rank: int
    alpha: int
    epochs: int
    learning_rate: float


class AuditStoreSettings(BaseModel):
    backend: str
    location: str


class Settings(BaseModel):
    backend: str
    model_version: str
    lora: LoRASettings
    audit_store: AuditStoreSettings


def load_settings(path: Path) -> Settings:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
