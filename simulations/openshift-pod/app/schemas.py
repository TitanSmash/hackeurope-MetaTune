from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.utils import normalize_dataset_rel_path


ALLOWED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


class HyperParams(BaseModel):
    lora_r: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    lora_dropout: float = Field(..., ge=0.0, le=1.0)
    learning_rate: float = Field(..., gt=0.0)
    batch_size: int = Field(..., gt=0)


class CreateRunRequest(BaseModel):
    model: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    dataset_rel_path: str = Field(..., min_length=1)
    hp: HyperParams
    token_budget: int = Field(..., gt=0)
    seq_len: int = Field(..., gt=0)
    seed: int = Field(default=1)
    job_name: str | None = None

    @field_validator("dataset_rel_path")
    @classmethod
    def _validate_dataset_rel_path(cls, value: str) -> str:
        return normalize_dataset_rel_path(value)

    @field_validator("job_name")
    @classmethod
    def _validate_job_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return stripped


class CallbackPayload(BaseModel):
    status: Literal["SUCCEEDED", "FAILED"]
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

