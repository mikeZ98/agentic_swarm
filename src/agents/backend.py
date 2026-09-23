"""Node 2 — Backend Engineer: typed FastAPI/asyncio implementation of the spec."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.base import NodeUpdate, extract_code, render_feedback, render_spec
from src.agents.llm import ChatRunnable
from src.state import Status, SwarmState

SYSTEM_PROMPT = """\
You are a senior Python backend engineer.
Implement the specification as ONE self-contained FastAPI module (Python 3.13):
- Full type hints, Pydantic v2 models, async handlers, `asyncio.Lock` around shared state.
- Validate all input; never trust client data; return precise HTTP status codes.
- No external services: in-memory storage only. Enable CORS for a local static frontend.
- Include `app = FastAPI(...)` at module level.
Respond with a single ```python fenced block and nothing else.
"""


class BackendAgent:
    name = "backend"

    def __init__(self, llm: ChatRunnable) -> None:
        self._llm = llm

    def _messages(self, state: SwarmState) -> list[BaseMessage]:
        parts = [f"Specification:\n```json\n{render_spec(state.architecture_spec)}\n```"]
        if state.backend_code and state.eval_report:
            parts.append(
                f"Revision #{state.iteration}. Your previous implementation:\n"
                f"```python\n{state.backend_code}\n```\n{render_feedback(state.eval_report)}"
            )
        return [SystemMessage(SYSTEM_PROMPT), HumanMessage("\n\n".join(parts))]

    async def __call__(self, state: SwarmState) -> NodeUpdate:
        reply = await self._llm.ainvoke(
            self._messages(state),
            config={"run_name": "backend.implement", "tags": ["backend"]},
        )
        code = extract_code(reply.text, preferred_langs=("python", "py"))
        return {"backend_code": code, "status": Status.BACKEND_READY}
