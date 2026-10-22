"""Hermetic stand-ins for the sandbox: no Docker, no MCP server process."""

from __future__ import annotations

from collections.abc import Iterator

from src.state import ExecutionResult, StaticAnalysisReport, StaticFinding, ToolRun


def execution(exit_code: int = 0, *, stderr: str = "", timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code, stdout="", stderr=stderr, timed_out=timed_out, duration_s=0.1
    )


def clean_report() -> StaticAnalysisReport:
    return StaticAnalysisReport(
        runs=(
            ToolRun(tool="ruff", exit_code=0, timed_out=False, findings=()),
            ToolRun(tool="bandit", exit_code=0, timed_out=False, findings=()),
        )
    )


def dirty_report() -> StaticAnalysisReport:
    finding = StaticFinding(
        tool="bandit",
        rule="B602",
        line=3,
        message="[HIGH] subprocess call with shell=True identified, security issue.",
    )
    return StaticAnalysisReport(
        runs=(
            ToolRun(tool="ruff", exit_code=0, timed_out=False, findings=()),
            ToolRun(tool="bandit", exit_code=1, timed_out=False, findings=(finding,)),
        )
    )


class FakeCodeTools:
    """Scripted `CodeTools`: replays results in order (the last one repeats) and records calls."""

    def __init__(
        self,
        executions: list[ExecutionResult] | None = None,
        reports: list[StaticAnalysisReport] | None = None,
    ) -> None:
        self.executed: list[str] = []
        self.analysed: list[str] = []
        self._executions = _replay(executions or [execution()])
        self._reports = _replay(reports or [clean_report()])

    async def execute_python(self, code: str) -> ExecutionResult:
        self.executed.append(code)
        return next(self._executions)

    async def static_analysis(self, code: str) -> StaticAnalysisReport:
        self.analysed.append(code)
        return next(self._reports)


def _replay[T](items: list[T]) -> Iterator[T]:
    yield from items
    while True:
        yield items[-1]
