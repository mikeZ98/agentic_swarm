"""Shared primitives for agent nodes."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.state import SwarmState

type NodeUpdate = dict[str, Any]
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


def coerce[M: BaseModel](raw: dict[str, Any] | BaseModel, model: type[M]) -> M:
    """Normalise `with_structured_output` results (instance or dict) into `model`."""
    if isinstance(raw, model):
        return raw
    payload = raw.model_dump() if isinstance(raw, BaseModel) else raw
    return model.model_validate(payload)


def render_spec(spec: dict[str, Any]) -> str:
    return json.dumps(spec, indent=2, ensure_ascii=False)


def render_feedback(report: dict[str, Any]) -> str:
    vulns = "\n".join(f"- {v}" for v in report.get("vulnerabilities", [])) or "- (none)"
    perf = "\n".join(f"- {p}" for p in report.get("performance_notes", [])) or "- (none)"
    return (
        f"Previous review score: {report.get('score', 'n/a')}/100 — REJECTED.\n"
        f"Vulnerabilities you MUST fix:\n{vulns}\n"
        f"Performance notes to address:\n{perf}"
    )
