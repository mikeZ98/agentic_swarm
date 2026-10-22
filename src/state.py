"""Shared graph state and the inter-agent data contracts.

Every payload that crosses a node boundary derives from `Contract`: closed (`extra="forbid"`),
immutable (`frozen`) and `strict` (no implicit coercion — `"3"` is not an `int`). Collections
are tuples so immutability is deep. LLM replies are validated in JSON mode (see
`agents.base.parse_structured`), where strict still maps JSON arrays onto tuples.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict

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


class Contract(BaseModel):
    """Base for every inter-agent payload, tool response and the graph state itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# --------------------------------------------------------------------------- architect output


class EndpointSpec(Contract):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(min_length=1)
    description: str
    request_model: str | None
    response_model: str | None
    status_codes: tuple[int, ...]


class FieldSpec(Contract):
    name: str = Field(min_length=1)
    type: str = Field(description="Python type annotation, e.g. `HttpUrl` or `int`")
    constraints: str = Field(description="Validation rules, or an empty string")


class DataModelSpec(Contract):
    name: str = Field(min_length=1)
    fields: tuple[FieldSpec, ...]
    description: str


class ArchitectureSpec(Contract):
    """Structured output of the Architect node."""

    title: str = Field(min_length=1)
    summary: str
    data_models: tuple[DataModelSpec, ...]
    endpoints: tuple[EndpointSpec, ...]
    frontend_layout: tuple[str, ...] = Field(description="Ordered list of UI sections")
    non_functional_requirements: tuple[str, ...]


# --------------------------------------------------------------------------- sandbox tool responses

type AnalysisTool = Literal["ruff", "bandit"]


class ExecutionResult(Contract):
    """Outcome of running a snippet inside the sandbox container (`execute_python`)."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float = Field(ge=0)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class StaticFinding(Contract):
    tool: AnalysisTool
    rule: str = Field(min_length=1, description="e.g. `F821` (ruff) or `B602` (bandit)")
    line: int = Field(ge=0, description="1-based; 0 when the finding is not tied to a line")
    message: str


class ToolRun(Contract):
    """One analyser's verdict. Fail-closed: anything but a clean exit with no findings fails."""

    tool: AnalysisTool
    exit_code: int
    timed_out: bool
    findings: tuple[StaticFinding, ...]

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.findings


class StaticAnalysisReport(Contract):
    """Result of the `static_analysis` tool: every analyser must pass."""

    runs: tuple[ToolRun, ...] = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return all(run.passed for run in self.runs)

    @property
    def findings(self) -> tuple[StaticFinding, ...]:
        return tuple(finding for run in self.runs for finding in run.findings)


# --------------------------------------------------------------------------- critic output


class EvaluationResult(Contract):
    """Strict acceptance contract enforced by the critic ("Pan Maruda")."""

    is_accepted: bool
    vulnerabilities: tuple[str, ...]
    performance_notes: tuple[str, ...]
    score: int = Field(ge=0, le=100, description="0 to 100")


class CriticVerdict(EvaluationResult):
    """The critic LLM's structured reply: the evaluation plus the owner of the defects."""

    defect_owner: RevisionTarget = Field(
        description="Which engineer must fix the findings: backend, frontend or both."
    )


class CriticReport(CriticVerdict):
    """The gated verdict recorded in state: `is_accepted` is the *final* decision."""

    reviewed_iteration: int = Field(ge=0)


#: Schemas an LLM must fill. They carry no defaults, so OpenAI strict decoding accepts them.
LLM_OUTPUT_SCHEMAS: tuple[type[Contract], ...] = (ArchitectureSpec, CriticVerdict)


# --------------------------------------------------------------------------- graph state


class SwarmState(Contract):
    """Shared state flowing through the LangGraph `StateGraph`.

    Nodes receive a validated, frozen instance and return a partial `NodeUpdate`;
    LangGraph builds the next instance, so a malformed update fails at the boundary.
    """

    user_prompt: str = Field(min_length=1)
    architecture_spec: ArchitectureSpec | None = None
    backend_code: str = ""
    backend_execution: ExecutionResult | None = None
    frontend_code: str = ""
    eval_report: CriticReport | None = None
    iteration: int = Field(default=0, ge=0)
    status: Status = Status.PENDING

    # Routing bookkeeping (not part of the user-facing contract).
    revision_target: RevisionTarget | None = None


class NodeUpdate(TypedDict, total=False):
    """The partial update a node returns; keys mirror `SwarmState` fields exactly."""

    architecture_spec: ArchitectureSpec
    backend_code: str
    backend_execution: ExecutionResult
    frontend_code: str
    eval_report: CriticReport
    iteration: int
    status: Status
    revision_target: RevisionTarget | None
