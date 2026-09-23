"""Shared graph state and the inter-agent data contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type RevisionTarget = Literal["backend", "frontend", "both"]


class Status(StrEnum):
    PENDING = "pending"
    SPEC_READY = "spec_ready"
    BACKEND_READY = "backend_ready"
    FRONTEND_READY = "frontend_ready"
    REVISING = "revising"
    ACCEPTED = "accepted"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


# --------------------------------------------------------------------------- contracts


class EndpointSpec(BaseModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    description: str
    request_model: str | None
    response_model: str | None
    status_codes: list[int]


class FieldSpec(BaseModel):
    name: str
    type: str = Field(description="Python type annotation, e.g. `HttpUrl` or `int`")
    constraints: str = Field(description="Validation rules, or an empty string")


class DataModelSpec(BaseModel):
    name: str
    fields: list[FieldSpec]
    description: str


class ArchitectureSpec(BaseModel):
    """Structured output of the Architect node."""

    title: str
    summary: str
    data_models: list[DataModelSpec]
    endpoints: list[EndpointSpec]
    frontend_layout: list[str] = Field(description="Ordered list of UI sections / components")
    non_functional_requirements: list[str]


class EvaluationResult(BaseModel):
    """Strict acceptance contract enforced by the critic ("Pan Maruda")."""

    is_accepted: bool
    vulnerabilities: list[str]
    performance_notes: list[str]
    score: int = Field(ge=0, le=100, description="0 to 100")


class CriticVerdict(EvaluationResult):
    """`EvaluationResult` plus the routing hint the graph needs for targeted self-correction."""

    defect_owner: RevisionTarget = Field(
        description="Which engineer must fix the findings: backend, frontend or both."
    )


# --------------------------------------------------------------------------- graph state


class SwarmState(BaseModel):
    """Shared state flowing through the LangGraph `StateGraph`.

    Nodes receive a validated instance and return *partial* dict updates.
    """

    model_config = ConfigDict(validate_assignment=True)

    user_prompt: str
    architecture_spec: dict[str, Any] = Field(default_factory=dict)
    backend_code: str = ""
    frontend_code: str = ""
    eval_report: dict[str, Any] = Field(default_factory=dict)
    iteration: int = Field(default=0, ge=0)
    status: str = Status.PENDING

    # Routing bookkeeping (not part of the user-facing contract).
    revision_target: RevisionTarget | None = None
