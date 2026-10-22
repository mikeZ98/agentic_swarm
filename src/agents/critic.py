"""Node 4 — Tester & Adversarial Critic ("Pan Maruda").

Runs on an independent model family and owns the loop-control decision. Before the LLM
review it runs a deterministic guardrail: ruff and bandit over the backend module (inside
the sandbox, via MCP) plus the backend's own sandbox smoke run. Acceptance needs BOTH a
passing guardrail and an LLM verdict at or above the score threshold — fail-closed: any
non-zero tool exit forces another revision, whatever the model thinks.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import StructuredRunnable, parse_structured, render_spec
from src.sandbox import CodeTools
from src.state import (
    CriticReport,
    CriticVerdict,
    GuardrailReport,
    NodeUpdate,
    RevisionTarget,
    Status,
    SwarmState,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are "Pan Maruda", a relentless red-team reviewer. Assume the code is broken until proven
otherwise. Audit the backend and frontend against the specification for:
- Security: injection, XSS, missing input validation, CORS misconfiguration, race conditions,
  unbounded memory growth, missing rate limiting, information leakage in errors.
- Correctness: every spec endpoint/model implemented, status codes honoured, frontend calls
  match backend routes and payload shapes exactly.
- Performance: blocking calls in async code, O(n) scans on hot paths, lock contention.
The "Tool results" section comes from ruff, bandit and a sandboxed import of the backend.
Those findings are authoritative: never accept code that has any.
Rules:
- `vulnerabilities`: concrete, actionable defects (file + symbol + fix). Empty only if none.
- `performance_notes`: concrete observations.
- `score`: 0-100. Accept (`is_accepted=true`) only if there are no security defects.
- `defect_owner`: who must fix the findings — "backend", "frontend" or "both".
"""


def render_guardrail(guardrail: GuardrailReport) -> str:
    if guardrail.passed:
        return "Tool results: ruff, bandit and the sandboxed import all passed."
    findings = "\n".join(f"- {line}" for line in guardrail.failures)
    return f"Tool results: FAILED — these block acceptance:\n{findings}"


def _with_backend(owner: RevisionTarget) -> RevisionTarget:
    """Guardrail findings are always in the backend module, so it must be revised too."""
    return "both" if owner == "frontend" else owner


class CriticAgent:
    name = "critic"

    def __init__(
        self,
        llm: StructuredRunnable,
        tools: CodeTools,
        *,
        max_iterations: int,
        acceptance_threshold: int,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._threshold = acceptance_threshold

    async def run_guardrail(self, state: SwarmState) -> GuardrailReport:
        analysis = await self._tools.static_analysis(state.backend_code)
        guardrail = GuardrailReport(static_analysis=analysis, execution=state.backend_execution)
        log.info(
            "critic guardrail passed=%s findings=%d", guardrail.passed, len(guardrail.failures)
        )
        return guardrail

    async def __call__(self, state: SwarmState) -> NodeUpdate:
        guardrail = await self.run_guardrail(state)
        review_input = (
            f"Specification:\n```json\n{render_spec(state.architecture_spec)}\n```\n\n"
            f"Backend:\n```python\n{state.backend_code}\n```\n\n"
            f"Frontend:\n```html\n{state.frontend_code}\n```\n\n"
            f"{render_guardrail(guardrail)}"
        )
        raw = await self._llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(review_input)],
            config={"run_name": "critic.review", "tags": ["critic"]},
        )
        verdict = parse_structured(raw, CriticVerdict)
        return self.decide(state, verdict, guardrail)

    def decide(
        self, state: SwarmState, verdict: CriticVerdict, guardrail: GuardrailReport
    ) -> NodeUpdate:
        """Pure loop-control policy: gate the verdict and compute the next status."""
        llm_accepted = verdict.is_accepted and verdict.score >= self._threshold
        accepted = llm_accepted and guardrail.passed
        owner = verdict.defect_owner
        if not guardrail.passed:
            owner = "backend" if llm_accepted else _with_backend(owner)
        report = CriticReport(
            is_accepted=accepted,
            vulnerabilities=(*guardrail.failures, *verdict.vulnerabilities),
            performance_notes=verdict.performance_notes,
            score=verdict.score,
            defect_owner=owner,
            reviewed_iteration=state.iteration,
            guardrail=guardrail,
        )
        if accepted:
            return {"eval_report": report, "status": Status.ACCEPTED, "revision_target": None}
        if state.iteration < self._max_iterations:
            return {
                "eval_report": report,
                "status": Status.REVISING,
                "iteration": state.iteration + 1,
                "revision_target": owner,
            }
        return {
            "eval_report": report,
            "status": Status.MAX_ITERATIONS_REACHED,
            "revision_target": None,
        }
