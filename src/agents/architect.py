"""Node 1 — Orchestrator / Architect: decomposes the request into a technical spec."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import NodeUpdate, StructuredRunnable, coerce
from src.state import ArchitectureSpec, Status, SwarmState

SYSTEM_PROMPT = """\
You are the principal architect of a two-engineer delivery squad.
Decompose the user's requirement into a precise, implementable technical specification:
- data_models: Pydantic models with exact field types.
- endpoints: every HTTP route (method, path, request/response model, status codes).
  Include error responses (e.g. 404, 422, 429) where relevant.
- frontend_layout: ordered UI sections for a single-page HTML/Tailwind/vanilla-JS client
  that consumes exactly these endpoints.
- non_functional_requirements: security, concurrency, rate limiting, validation rules.
Keep the scope minimal but production-grade. The backend is a single FastAPI module using
asyncio with in-memory storage; the frontend is a single self-contained HTML file.
"""


class ArchitectAgent:
    name = "architect"

    def __init__(self, llm: StructuredRunnable) -> None:
        self._llm = llm

    async def __call__(self, state: SwarmState) -> NodeUpdate:
        raw = await self._llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(state.user_prompt)],
            config={"run_name": "architect.spec", "tags": ["architect"]},
        )
        spec = coerce(raw, ArchitectureSpec)
        return {"architecture_spec": spec.model_dump(mode="json"), "status": Status.SPEC_READY}
