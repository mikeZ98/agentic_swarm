"""The critic's deterministic guardrail: tool failures force a revision, whatever the LLM says."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from src.agents import ArchitectAgent, BackendAgent, CriticAgent, FrontendAgent, SwarmAgents
from src.agents.critic import render_guardrail
from src.graph import build_graph, recursion_limit
from src.sandbox import SandboxError
from src.state import (
    CriticVerdict,
    GuardrailReport,
    RevisionTarget,
    StaticAnalysisReport,
    Status,
    SwarmState,
    ToolRun,
)
from tests.fakes import FakeCodeTools, clean_report, dirty_report, execution
from tests.test_graph import SPEC

MAX_ITERATIONS = 2
THRESHOLD = 80


def _verdict(
    *, accepted: bool = True, score: int = 100, owner: RevisionTarget = "backend"
) -> CriticVerdict:
    return CriticVerdict(
        is_accepted=accepted,
        vulnerabilities=() if accepted else ("XSS: stats table uses innerHTML",),
        performance_notes=(),
        score=score,
        defect_owner=owner,
    )


def _critic(tools: FakeCodeTools | None = None, llm: Any = None) -> CriticAgent:
    return CriticAgent(
        llm or RunnableLambda(lambda _: _verdict().model_dump()),
        tools or FakeCodeTools(),
        max_iterations=MAX_ITERATIONS,
        acceptance_threshold=THRESHOLD,
    )


def _guardrail(report: StaticAnalysisReport | None = None, **exec_kwargs: Any) -> GuardrailReport:
    return GuardrailReport(
        static_analysis=report or clean_report(), execution=execution(**exec_kwargs)
    )


# --------------------------------------------------------------------------- the gate itself


def test_clean_tools_and_a_successful_import_pass() -> None:
    guardrail = _guardrail()

    assert guardrail.passed
    assert guardrail.failures == ()
    assert "all passed" in render_guardrail(guardrail)


def test_a_bandit_finding_fails_the_gate_with_an_actionable_line() -> None:
    guardrail = _guardrail(dirty_report())

    assert not guardrail.passed
    assert guardrail.failures == (
        "[bandit B602] line 3: [HIGH] subprocess call with shell=True identified, security issue.",
    )
    assert "FAILED" in render_guardrail(guardrail)


def test_a_non_zero_exit_without_findings_still_fails() -> None:
    report = StaticAnalysisReport(
        runs=(ToolRun(tool="ruff", exit_code=2, timed_out=False, findings=()),)
    )

    guardrail = _guardrail(report)

    assert not guardrail.passed
    assert guardrail.failures == ("[ruff] exited 2",)


@pytest.mark.parametrize(
    ("exec_kwargs", "expected"),
    [
        ({"exit_code": 1, "stderr": "ModuleNotFoundError: No module named 'redis'"}, "exit 1"),
        ({"exit_code": 124, "timed_out": True}, "timed out"),
    ],
)
def test_a_failed_sandbox_import_fails_the_gate(exec_kwargs: dict[str, Any], expected: str) -> None:
    guardrail = _guardrail(**exec_kwargs)

    assert not guardrail.passed
    (line,) = guardrail.failures
    assert line.startswith("[sandbox] importing the backend module failed")
    assert expected in line


def test_a_missing_execution_fails_closed() -> None:
    guardrail = GuardrailReport(static_analysis=clean_report(), execution=None)

    assert not guardrail.passed
    assert guardrail.failures == ("[sandbox] the backend module was never executed",)


# --------------------------------------------------------------------------- loop-control policy


def test_tool_failure_overrides_a_perfect_llm_acceptance() -> None:
    update = _critic().decide(
        SwarmState(user_prompt="x"), _verdict(score=100), _guardrail(dirty_report())
    )

    report = update["eval_report"]
    assert update["status"] == Status.REVISING
    assert update["iteration"] == 1
    assert update["revision_target"] == "backend"
    assert report.is_accepted is False
    assert report.score == 100
    assert report.vulnerabilities[0].startswith("[bandit B602]")
    assert report.guardrail.passed is False


def test_tool_failure_escalates_a_frontend_only_rejection_to_both() -> None:
    update = _critic().decide(
        SwarmState(user_prompt="x"),
        _verdict(accepted=False, score=40, owner="frontend"),
        _guardrail(dirty_report()),
    )

    assert update["revision_target"] == "both"
    assert update["eval_report"].vulnerabilities == (
        "[bandit B602] line 3: [HIGH] subprocess call with shell=True identified, security issue.",
        "XSS: stats table uses innerHTML",
    )


def test_tool_failure_on_the_last_iteration_ends_unaccepted() -> None:
    state = SwarmState(user_prompt="x", iteration=MAX_ITERATIONS)

    update = _critic().decide(state, _verdict(score=100), _guardrail(dirty_report()))

    assert update["status"] == Status.MAX_ITERATIONS_REACHED
    assert update["eval_report"].is_accepted is False


def test_passing_tools_and_llm_acceptance_accept() -> None:
    update = _critic().decide(SwarmState(user_prompt="x"), _verdict(score=90), _guardrail())

    assert update["status"] == Status.ACCEPTED
    assert update["eval_report"].is_accepted is True


# --------------------------------------------------------------------------- the node


async def test_critic_analyses_the_backend_and_shows_the_llm_the_tool_results() -> None:
    prompts: list[str] = []

    def llm(messages: LanguageModelInput) -> dict[str, Any]:
        assert isinstance(messages, list)
        prompts.append(str(messages[-1].content))
        return _verdict().model_dump()

    tools = FakeCodeTools(reports=[dirty_report()])
    state = SwarmState(
        user_prompt="x", backend_code="import subprocess", backend_execution=execution()
    )

    update = await _critic(tools, RunnableLambda(llm))(state)

    assert tools.analysed == ["import subprocess"]
    assert "Tool results: FAILED" in prompts[0]
    assert "[bandit B602]" in prompts[0]
    assert update["status"] == Status.REVISING


async def test_an_unreachable_sandbox_aborts_instead_of_scoring_the_code() -> None:
    class DownTools(FakeCodeTools):
        async def static_analysis(self, code: str) -> StaticAnalysisReport:
            raise SandboxError("cannot reach the sandbox MCP server")

    with pytest.raises(SandboxError):
        await _critic(DownTools())(SwarmState(user_prompt="x", backend_code="x = 1"))


# --------------------------------------------------------------------------- the whole loop


class ScriptedChat:
    def __init__(self, replies: list[str]) -> None:
        self.prompts: list[str] = []
        self._replies = iter(replies)

    def __call__(self, messages: LanguageModelInput) -> BaseMessage:
        assert isinstance(messages, list)
        self.prompts.append(str(messages[-1].content))
        return AIMessage(content=next(self._replies))


def _swarm(tools: FakeCodeTools, backend: ScriptedChat) -> SwarmAgents:
    """Real agents; the critic LLM always accepts, so only the tools can force a retry."""
    return SwarmAgents(
        architect=ArchitectAgent(RunnableLambda(lambda _: SPEC)),
        backend=BackendAgent(RunnableLambda(backend), tools),
        frontend=FrontendAgent(RunnableLambda(ScriptedChat(["```html\n<main/>\n```"] * 5))),
        critic=_critic(tools),
    )


async def _run(agents: SwarmAgents) -> SwarmState:
    raw = await build_graph(agents).ainvoke(
        SwarmState(user_prompt="build it"),
        config={"recursion_limit": recursion_limit(MAX_ITERATIONS)},
    )
    return SwarmState.model_validate(raw)


async def test_non_zero_tool_exit_forces_a_backend_retry_until_clean() -> None:
    tools = FakeCodeTools(reports=[dirty_report(), clean_report()])
    backend = ScriptedChat(
        ["```python\nos.system(cmd)\n```", "```python\nsubprocess.run(argv)\n```"]
    )

    final = await _run(_swarm(tools, backend))

    assert final.status == Status.ACCEPTED
    assert final.iteration == 1
    assert final.backend_code == "subprocess.run(argv)"
    assert "[bandit B602]" in backend.prompts[1]
    assert "Deterministic gate (ruff / bandit / sandboxed import): FAILED" in backend.prompts[1]
    assert tools.analysed == ["os.system(cmd)", "subprocess.run(argv)"]


async def test_a_failing_import_forces_a_retry_with_the_traceback() -> None:
    tools = FakeCodeTools(
        executions=[execution(1, stderr="NameError: name 'FastAPI' is not defined"), execution()]
    )
    backend = ScriptedChat(["```python\napp = FastAPI()\n```", "```python\nfixed = 1\n```"])

    final = await _run(_swarm(tools, backend))

    assert final.status == Status.ACCEPTED
    assert "NameError: name 'FastAPI' is not defined" in backend.prompts[1]


async def test_persistent_tool_failures_exhaust_the_budget_unaccepted() -> None:
    tools = FakeCodeTools(reports=[dirty_report()])
    backend = ScriptedChat(["```python\nos.system(cmd)\n```"] * (MAX_ITERATIONS + 1))

    final = await _run(_swarm(tools, backend))

    assert final.status == Status.MAX_ITERATIONS_REACHED
    assert final.iteration == MAX_ITERATIONS
    assert final.eval_report is not None
    assert final.eval_report.is_accepted is False
    assert len(tools.analysed) == MAX_ITERATIONS + 1
