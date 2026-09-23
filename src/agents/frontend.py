"""Node 3 — Frontend Engineer: lightweight HTML/Tailwind/vanilla-JS client."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.base import NodeUpdate, extract_code, render_feedback, render_spec
from src.agents.llm import ChatRunnable
from src.state import Status, SwarmState

SYSTEM_PROMPT = """\
You are a senior frontend engineer.
Build ONE self-contained `index.html` that consumes the given backend endpoints:
- Tailwind via CDN, vanilla ES modules, no frameworks, no build step.
- `const API_BASE = "http://localhost:8000";` at the top of the script.
- Render all user/server data with `textContent` (never `innerHTML`) to prevent XSS.
- Handle every documented error status with visible, accessible feedback.
Respond with a single ```html fenced block and nothing else.
"""


class FrontendAgent:
    name = "frontend"

    def __init__(self, llm: ChatRunnable) -> None:
        self._llm = llm

    def _messages(self, state: SwarmState) -> list[BaseMessage]:
        parts = [
            f"Specification:\n```json\n{render_spec(state.architecture_spec)}\n```",
            f"Backend implementation to integrate with:\n```python\n{state.backend_code}\n```",
        ]
        if state.frontend_code and state.eval_report:
            parts.append(
                f"Revision #{state.iteration}. Your previous implementation:\n"
                f"```html\n{state.frontend_code}\n```\n{render_feedback(state.eval_report)}"
            )
        return [SystemMessage(SYSTEM_PROMPT), HumanMessage("\n\n".join(parts))]

    async def __call__(self, state: SwarmState) -> NodeUpdate:
        reply = await self._llm.ainvoke(
            self._messages(state),
            config={"run_name": "frontend.implement", "tags": ["frontend"]},
        )
        code = extract_code(reply.text, preferred_langs=("html",))
        return {"frontend_code": code, "status": Status.FRONTEND_READY}
