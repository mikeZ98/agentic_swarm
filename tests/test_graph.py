"""Offline tests for graph topology, retry thresholds and edge routing.

No network: nodes are either scripted fakes or the real agent classes wired to fake
LangChain runnables.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END
from pydantic import BaseModel

from src.agents import (
    ArchitectAgent,
    BackendAgent,
    CriticAgent,
    FrontendAgent,
    SwarmAgents,
)
from src.agents.base import AgentNode, NodeUpdate, extract_code
from src.graph import build_graph, recursion_limit, route_after_backend, route_after_critic
from src.state import (
    ArchitectureSpec,
    CriticVerdict,
    DataModelSpec,
    EndpointSpec,
    EvaluationResult,
    FieldSpec,
    RevisionTarget,
    Status,
    SwarmState,
)

MAX_ITERATIONS = 3
THRESHOLD = 80

SPEC = ArchitectureSpec(
    title="URL shortener",
    summary="Shorten URLs with a token-bucket limit.",
    data_models=[
        DataModelSpec(
            name="ShortenRequest",
            fields=[FieldSpec(name="url", type="HttpUrl", constraints="http(s) only")],
            description="Shorten request",
        )
    ],
    endpoints=[
        EndpointSpec(
            method="POST",
            path="/shorten",
            description="Create short code",
            request_model="ShortenRequest",
            response_model=None,
            status_codes=[201, 422, 429],
        )
    ],
    frontend_layout=["form", "result", "stats"],
    non_functional_requirements=["token bucket 10/min"],
)


def verdict(*, accepted: bool, score: int = 90, owner: RevisionTarget = "backend") -> CriticVerdict:
    return CriticVerdict(
        is_accepted=accepted,
        vulnerabilities=[] if accepted else ["SSRF: /shorten accepts file:// URLs"],
        performance_notes=["O(1) lookups"],
        score=score,
        defect_owner=owner,
    )


# --------------------------------------------------------------------------- harness


class Recorder:
    """Builds scripted node fakes and records the order in which nodes execute."""

    def __init__(self, verdicts: list[CriticVerdict]) -> None:
        self.calls: list[str] = []
        self._verdicts: Iterator[CriticVerdict] = iter(verdicts)
        self._critic = CriticAgent(
            RunnableLambda(self._never),
            max_iterations=MAX_ITERATIONS,
            acceptance_threshold=THRESHOLD,
        )

    @staticmethod
    def _never(_: LanguageModelInput) -> BaseModel:
        raise AssertionError("LLM must not be called in scripted mode")

    def _node(self, name: str, update: NodeUpdate) -> AgentNode:
        async def node(state: SwarmState) -> NodeUpdate:
            self.calls.append(name)
            return update

        return node

    async def _critic_node(self, state: SwarmState) -> NodeUpdate:
        self.calls.append("critic")
        return self._critic.decide(state, next(self._verdicts))

    def agents(self) -> SwarmAgents:
        return SwarmAgents(
            architect=self._node(
                "architect", {"architecture_spec": SPEC.model_dump(), "status": Status.SPEC_READY}
            ),
            backend=self._node(
                "backend", {"backend_code": "app = 1", "status": Status.BACKEND_READY}
            ),
            frontend=self._node(
                "frontend", {"frontend_code": "<html/>", "status": Status.FRONTEND_READY}
            ),
            critic=self._critic_node,
        )


async def run_graph(agents: SwarmAgents) -> SwarmState:
    graph = build_graph(agents)
    raw = await graph.ainvoke(
        SwarmState(user_prompt="build it"),
        config={"recursion_limit": recursion_limit(MAX_ITERATIONS)},
    )
    return SwarmState.model_validate(raw)


# --------------------------------------------------------------------------- pure routers


@pytest.mark.parametrize(
    ("status", "target", "expected"),
    [
        (Status.REVISING, "backend", "backend"),
        (Status.REVISING, "both", "backend"),
        (Status.REVISING, "frontend", "frontend"),
        (Status.ACCEPTED, None, END),
        (Status.MAX_ITERATIONS_REACHED, None, END),
    ],
)
def test_route_after_critic(status: Status, target: RevisionTarget | None, expected: str) -> None:
    state = SwarmState(user_prompt="x", status=status, revision_target=target)
    assert route_after_critic(state) == expected


@pytest.mark.parametrize(
    ("target", "expected"),
    [(None, "frontend"), ("both", "frontend"), ("backend", "critic")],
)
def test_route_after_backend(target: RevisionTarget | None, expected: str) -> None:
    state = SwarmState(user_prompt="x", status=Status.BACKEND_READY, revision_target=target)
    assert route_after_backend(state) == expected


# --------------------------------------------------------------------------- critic policy


def _critic() -> CriticAgent:
    return CriticAgent(
        RunnableLambda(Recorder._never),
        max_iterations=MAX_ITERATIONS,
        acceptance_threshold=THRESHOLD,
    )


def test_critic_accepts_and_clears_target() -> None:
    update = _critic().decide(SwarmState(user_prompt="x"), verdict(accepted=True))
    assert update["status"] == Status.ACCEPTED
    assert update["revision_target"] is None
    assert "iteration" not in update


def test_critic_score_gate_overrides_llm_acceptance() -> None:
    update = _critic().decide(SwarmState(user_prompt="x"), verdict(accepted=True, score=79))
    assert update["status"] == Status.REVISING
    assert update["eval_report"]["is_accepted"] is False


def test_critic_rejection_increments_iteration() -> None:
    state = SwarmState(user_prompt="x", iteration=1)
    update = _critic().decide(state, verdict(accepted=False, owner="frontend"))
    assert update["iteration"] == 2
    assert update["revision_target"] == "frontend"


def test_critic_stops_at_iteration_limit() -> None:
    state = SwarmState(user_prompt="x", iteration=MAX_ITERATIONS)
    update = _critic().decide(state, verdict(accepted=False))
    assert update["status"] == Status.MAX_ITERATIONS_REACHED
    assert "iteration" not in update


def test_eval_report_satisfies_contract() -> None:
    update = _critic().decide(SwarmState(user_prompt="x"), verdict(accepted=False))
    EvaluationResult.model_validate(update["eval_report"])


def test_evaluation_score_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="less than or equal to 100"):
        EvaluationResult(is_accepted=True, vulnerabilities=[], performance_notes=[], score=101)


# --------------------------------------------------------------------------- graph transitions


async def test_happy_path_single_pass() -> None:
    rec = Recorder([verdict(accepted=True)])
    final = await run_graph(rec.agents())
    assert rec.calls == ["architect", "backend", "frontend", "critic"]
    assert final.status == Status.ACCEPTED
    assert final.iteration == 0


async def test_retry_threshold_caps_revisions_at_three() -> None:
    rec = Recorder([verdict(accepted=False, owner="both")] * (MAX_ITERATIONS + 5))
    final = await run_graph(rec.agents())
    assert final.status == Status.MAX_ITERATIONS_REACHED
    assert final.iteration == MAX_ITERATIONS
    assert rec.calls.count("critic") == MAX_ITERATIONS + 1
    assert rec.calls.count("backend") == MAX_ITERATIONS + 1
    assert rec.calls.count("architect") == 1


async def test_backend_feedback_skips_frontend() -> None:
    rec = Recorder([verdict(accepted=False, owner="backend"), verdict(accepted=True)])
    final = await run_graph(rec.agents())
    assert rec.calls == ["architect", "backend", "frontend", "critic", "backend", "critic"]
    assert final.status == Status.ACCEPTED
    assert final.iteration == 1


async def test_frontend_feedback_skips_backend() -> None:
    rec = Recorder([verdict(accepted=False, owner="frontend"), verdict(accepted=True)])
    await run_graph(rec.agents())
    assert rec.calls == ["architect", "backend", "frontend", "critic", "frontend", "critic"]


async def test_both_feedback_reruns_backend_then_frontend() -> None:
    rec = Recorder([verdict(accepted=False, owner="both"), verdict(accepted=True)])
    await run_graph(rec.agents())
    assert rec.calls[4:] == ["backend", "frontend", "critic"]


async def test_recovers_on_final_allowed_iteration() -> None:
    script = [verdict(accepted=False, owner="frontend")] * MAX_ITERATIONS + [verdict(accepted=True)]
    rec = Recorder(script)
    final = await run_graph(rec.agents())
    assert final.status == Status.ACCEPTED
    assert final.iteration == MAX_ITERATIONS


# --------------------------------------------------------------------------- real agents, fake LLMs


class FakeChat:
    """Scripted chat model that records every prompt it receives."""

    def __init__(self, replies: list[str]) -> None:
        self.prompts: list[str] = []
        self._replies = iter(replies)

    def __call__(self, messages: LanguageModelInput) -> BaseMessage:
        assert isinstance(messages, list)
        self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        return AIMessage(content=next(self._replies))


async def test_real_agents_propagate_feedback_through_the_loop() -> None:
    backend_llm = FakeChat(
        ["```python\napp = 'v1'\n```", "Fixed:\n```python\napp = 'v2-validated'\n```"]
    )
    frontend_llm = FakeChat(["```html\n<main>v1</main>\n```"])
    verdicts = iter([verdict(accepted=False, owner="backend"), verdict(accepted=True, score=95)])

    def critic_llm(_: LanguageModelInput) -> dict[str, Any]:
        return next(verdicts).model_dump()

    agents = SwarmAgents(
        architect=ArchitectAgent(RunnableLambda(lambda _: SPEC)),
        backend=BackendAgent(RunnableLambda(backend_llm)),
        frontend=FrontendAgent(RunnableLambda(frontend_llm)),
        critic=CriticAgent(
            RunnableLambda(critic_llm),
            max_iterations=MAX_ITERATIONS,
            acceptance_threshold=THRESHOLD,
        ),
    )
    final = await run_graph(agents)

    assert final.status == Status.ACCEPTED
    assert final.backend_code == "app = 'v2-validated'"
    assert final.frontend_code == "<main>v1</main>"
    assert final.architecture_spec["title"] == "URL shortener"
    assert "SSRF" not in backend_llm.prompts[0]
    assert "SSRF: /shorten accepts file:// URLs" in backend_llm.prompts[1]
    assert "Revision #1" in backend_llm.prompts[1]


# --------------------------------------------------------------------------- helpers


def test_extract_code_prefers_language_block() -> None:
    text = "```text\nlong long long long\n```\n```python\nx = 1\n```"
    assert extract_code(text, preferred_langs=("python",)) == "x = 1"
    assert extract_code("plain", preferred_langs=("python",)) == "plain"


def test_recursion_limit_covers_worst_case() -> None:
    worst_case_steps = 4 + 3 * MAX_ITERATIONS
    assert recursion_limit(MAX_ITERATIONS) > worst_case_steps
