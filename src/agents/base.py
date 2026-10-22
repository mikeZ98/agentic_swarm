"""Shared primitives for agent nodes."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.state import ArchitectureSpec, Contract, CriticReport, NodeUpdate, SwarmState

type StructuredRunnable = Runnable[LanguageModelInput, dict[str, Any] | BaseModel]

_FENCE_RE = re.compile(r"```(?P<lang>[\w+-]*)\s*\n(?P<body>.*?)```", re.DOTALL)


class AgentNode(Protocol):
    """A LangGraph node: receives the validated state, returns a partial update.

    The parameter must be named `state` to satisfy LangGraph's `_Node` protocol.
    """

    async def __call__(self, state: SwarmState) -> NodeUpdate: ...


def extract_code(text: str, *, preferred_langs: tuple[str, ...]) -> str:
    """Return the largest fenced block in `preferred_langs`, else the largest block, else text."""
    blocks = [(m["lang"].lower(), m["body"].strip()) for m in _FENCE_RE.finditer(text)]
    if not blocks:
        return text.strip()
    preferred = [body for lang, body in blocks if lang in preferred_langs]
    return max(preferred or [body for _, body in blocks], key=len)


def parse_structured[M: Contract](raw: dict[str, Any] | BaseModel, model: type[M]) -> M:
    """Validate a structured LLM reply against `model` in strict JSON mode.

    The reply crossed a wire as JSON, so it is re-validated as JSON: arrays map onto tuples,
    but nothing is coerced (`"90"` is not a score) and unknown keys are rejected.
    """
    if isinstance(raw, model):
        return raw
    payload = raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
    return model.model_validate_json(json.dumps(payload))


def render_spec(spec: ArchitectureSpec | None) -> str:
    return spec.model_dump_json(indent=2) if spec is not None else "{}"


def render_feedback(report: CriticReport) -> str:
    vulns = "\n".join(f"- {v}" for v in report.vulnerabilities) or "- (none)"
    perf = "\n".join(f"- {p}" for p in report.performance_notes) or "- (none)"
    return (
        f"Previous review score: {report.score}/100 — REJECTED.\n"
        f"Vulnerabilities you MUST fix:\n{vulns}\n"
        f"Performance notes to address:\n{perf}"
    )
