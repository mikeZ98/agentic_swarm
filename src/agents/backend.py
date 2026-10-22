"""Node 2 — Backend Engineer: typed FastAPI/asyncio implementation of the spec.

After generating the module it smoke-runs it through the sandbox MCP tools (never on the
host) and records the `ExecutionResult`, so the critic judges code that actually imports.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.base import extract_code, render_feedback, render_spec
from src.agents.llm import ChatRunnable
from src.sandbox import CodeTools
from src.state import NodeUpdate, Status, SwarmState

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

    def __init__(self, llm: ChatRunnable, tools: CodeTools) -> None:
        self._llm = llm
        self._tools = tools

    def _messages(self, state: SwarmState) -> list[BaseMessage]:
        parts = [f"Specification:\n```json\n{render_spec(state.architecture_spec)}\n```"]
        if state.backend_code and state.eval_report is not None:
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
        execution = await self._tools.execute_python(code)
        return {
            "backend_code": code,
            "backend_execution": execution,
            "status": Status.BACKEND_READY,
        }
