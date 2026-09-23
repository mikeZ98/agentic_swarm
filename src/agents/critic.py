"""Node 4 — Tester & Adversarial Critic ("Pan Maruda").

Runs on an independent model family and owns the loop-control decision: it validates the
LLM verdict against `EvaluationResult`, applies a deterministic score gate, and advances
`iteration` when a revision is warranted. The graph router then only reads `status`.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import NodeUpdate, StructuredRunnable, coerce, render_spec
from src.state import CriticVerdict, Status, SwarmState

SYSTEM_PROMPT = """\
You are "Pan Maruda", a relentless red-team reviewer. Assume the code is broken until proven
otherwise. Audit the backend and frontend against the specification for:
- Security: injection, XSS, missing input validation, CORS misconfiguration, race conditions,
  unbounded memory growth, missing rate limiting, information leakage in errors.
- Correctness: every spec endpoint/model implemented, status codes honoured, frontend calls
  match backend routes and payload shapes exactly.
- Performance: blocking calls in async code, O(n) scans on hot paths, lock contention.
Rules:
- `vulnerabilities`: concrete, actionable defects (file + symbol + fix). Empty only if none.
- `performance_notes`: concrete observations.
- `score`: 0-100. Accept (`is_accepted=true`) only if there are no security defects.
- `defect_owner`: who must fix the findings — "backend", "frontend" or "both".
"""


class CriticAgent:
    name = "critic"

    def __init__(
        self, llm: StructuredRunnable, *, max_iterations: int, acceptance_threshold: int
    ) -> None:
        self._llm = llm
        self._max_iterations = max_iterations
        self._threshold = acceptance_threshold

    async def __call__(self, state: SwarmState) -> NodeUpdate:
        review_input = (
            f"Specification:\n```json\n{render_spec(state.architecture_spec)}\n```\n\n"
            f"Backend:\n```python\n{state.backend_code}\n```\n\n"
            f"Frontend:\n```html\n{state.frontend_code}\n```"
        )
        raw = await self._llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(review_input)],
            config={"run_name": "critic.review", "tags": ["critic"]},
        )
        verdict = coerce(raw, CriticVerdict)
        return self.decide(state, verdict)

    def decide(self, state: SwarmState, verdict: CriticVerdict) -> NodeUpdate:
        """Pure loop-control policy: gate the verdict and compute the next status."""
        accepted = verdict.is_accepted and verdict.score >= self._threshold
        report = verdict.model_dump(mode="json") | {
            "is_accepted": accepted,
            "reviewed_iteration": state.iteration,
        }
        if accepted:
            return {"eval_report": report, "status": Status.ACCEPTED, "revision_target": None}
        if state.iteration < self._max_iterations:
            return {
                "eval_report": report,
                "status": Status.REVISING,
                "iteration": state.iteration + 1,
                "revision_target": verdict.defect_owner,
            }
        return {
            "eval_report": report,
            "status": Status.MAX_ITERATIONS_REACHED,
            "revision_target": None,
        }
