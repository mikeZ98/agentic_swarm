"""`DockerSandbox` isolation flags, analyser parsing and failure semantics — no daemon needed.

The command runner is faked; `run_process` itself is exercised against real local
subprocesses (the Python interpreter), never Docker.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from src.config import SandboxSettings
from src.sandbox.base import SandboxError
from src.sandbox.docker import (
    SANDBOX_USER,
    TIMEOUT_EXIT_CODE,
    DockerSandbox,
    ProcessOutput,
    parse_bandit,
    parse_ruff,
    run_process,
)
from src.state import StaticFinding

RUFF_JSON = json.dumps(
    [
        {"code": "F821", "message": "Undefined name `x`", "location": {"row": 4, "column": 12}},
        {"code": None, "message": "SyntaxError: Expected ':'", "location": {"row": 7}},
    ]
)
BANDIT_JSON = json.dumps(
    {
        "errors": [],
        "results": [
            {
                "test_id": "B602",
                "issue_severity": "HIGH",
                "issue_text": "subprocess call with shell=True identified, security issue.",
                "line_number": 3,
            }
        ],
    }
)


def _out(
    returncode: int = 0, stdout: str = "", stderr: str = "", *, timed_out: bool = False
) -> ProcessOutput:
    return ProcessOutput(
        returncode=returncode, stdout=stdout, stderr=stderr, timed_out=timed_out, duration_s=0.2
    )


@dataclass
class FakeRunner:
    """Answers `docker run` by the in-container tool name; records every argv."""

    outputs: dict[str, ProcessOutput]
    calls: list[tuple[list[str], bytes]] = field(default_factory=list)
    error: OSError | None = None
    image_present: ProcessOutput = field(default_factory=lambda: _out(0, "sha256:abc"))

    async def __call__(
        self, argv: Sequence[str], *, stdin: bytes, timeout_s: float, output_limit: int
    ) -> ProcessOutput:
        self.calls.append((list(argv), stdin))
        if self.error is not None:
            raise self.error
        if argv[1] == "rm":
            return _out()
        if argv[1] == "image":
            return self.image_present
        image_at = list(argv).index(SandboxSettings().image)
        return self.outputs[argv[image_at + 1]]


def _runs(runner: FakeRunner) -> list[tuple[list[str], bytes]]:
    return [(argv, stdin) for argv, stdin in runner.calls if argv[1] != "image"]


def _sandbox(**outputs: ProcessOutput) -> tuple[DockerSandbox, FakeRunner]:
    runner = FakeRunner(outputs=outputs)
    return DockerSandbox(SandboxSettings(), runner=runner), runner


# --------------------------------------------------------------------------- isolation


def test_container_is_network_less_read_only_unprivileged_and_bounded() -> None:
    argv = DockerSandbox(SandboxSettings(memory="128m", cpus=0.25, pids_limit=32)).container_argv(
        "swarm-sbx-test"
    )
    flags = " ".join(argv)

    for expected in (
        "run --rm --interactive",
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        f"--user {SANDBOX_USER}",
        "--pids-limit 32",
        "--memory 128m",
        "--memory-swap 128m",
        "--cpus 0.25",
        "--pull never",
    ):
        assert expected in flags
    assert argv[-1] == SandboxSettings().image
    assert not any(arg in {"--privileged", "-v", "--volume", "--mount"} for arg in argv)


async def test_code_travels_over_stdin_and_never_as_an_argument() -> None:
    sandbox, runner = _sandbox(python=_out(0, "hello"))
    code = "print('hello')  # secret-marker"

    result = await sandbox.execute_python(code)

    argv, stdin = _runs(runner)[0]
    assert stdin == code.encode()
    assert not any("secret-marker" in arg for arg in argv)
    assert argv[argv.index(SandboxSettings().image) + 1 :][:2] == ["python", "-I"]
    assert result.succeeded
    assert result.stdout == "hello"


async def test_each_run_gets_a_unique_container_name() -> None:
    sandbox, runner = _sandbox(python=_out())

    await sandbox.execute_python("x = 1")
    await sandbox.execute_python("x = 2")

    names = [argv[argv.index("--name") + 1] for argv, _ in _runs(runner)]
    assert len(set(names)) == 2
    assert all(name.startswith("swarm-sbx-") for name in names)


# --------------------------------------------------------------------------- failure semantics


async def test_a_failing_snippet_is_a_result_not_an_error() -> None:
    sandbox, _ = _sandbox(python=_out(1, stderr="ZeroDivisionError"))

    result = await sandbox.execute_python("1 / 0")

    assert not result.succeeded
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.stderr


async def test_timeout_force_removes_the_container() -> None:
    sandbox, runner = _sandbox(python=_out(TIMEOUT_EXIT_CODE, timed_out=True))

    result = await sandbox.execute_python("while True: pass")

    assert result.timed_out
    assert not result.succeeded
    run_argv, rm_argv = _runs(runner)[0][0], _runs(runner)[1][0]
    assert rm_argv[1:3] == ["rm", "--force"]
    assert rm_argv[3] == run_argv[run_argv.index("--name") + 1]


@pytest.mark.parametrize("code", [125, 126, 127])
async def test_docker_failure_codes_raise_sandbox_error(code: int) -> None:
    sandbox, _ = _sandbox(python=_out(code, stderr="Cannot connect to the Docker daemon"))

    with pytest.raises(SandboxError, match="docker run failed"):
        await sandbox.execute_python("x = 1")


async def test_preflight_rejects_a_missing_daemon_or_image_before_running_code() -> None:
    sandbox, runner = _sandbox(python=_out())
    runner.image_present = _out(1, stderr="failed to connect to the docker API")

    with pytest.raises(SandboxError, match="Docker daemon down or image not built"):
        await sandbox.execute_python("x = 1")

    assert _runs(runner) == []  # no container was started


async def test_preflight_runs_once_per_sandbox() -> None:
    sandbox, runner = _sandbox(python=_out())

    await sandbox.execute_python("x = 1")
    await sandbox.execute_python("x = 2")

    probes = [argv for argv, _ in runner.calls if argv[1] == "image"]
    assert probes == [
        ["docker", "image", "inspect", "--format", "{{.Id}}", SandboxSettings().image]
    ]


async def test_missing_docker_binary_raises_sandbox_error() -> None:
    runner = FakeRunner(outputs={}, error=FileNotFoundError("docker"))
    sandbox = DockerSandbox(SandboxSettings(), runner=runner)

    with pytest.raises(SandboxError, match="cannot start 'docker'"):
        await sandbox.execute_python("x = 1")


# --------------------------------------------------------------------------- static analysis


async def test_static_analysis_runs_ruff_and_bandit_from_stdin() -> None:
    sandbox, runner = _sandbox(ruff=_out(1, RUFF_JSON), bandit=_out(1, BANDIT_JSON))

    report = await sandbox.static_analysis("import os")

    assert not report.passed
    assert {run.tool for run in report.runs} == {"ruff", "bandit"}
    assert [f.rule for f in report.findings] == ["F821", "syntax-error", "B602"]
    commands = {argv[argv.index(SandboxSettings().image) + 1]: argv for argv, _ in _runs(runner)}
    assert commands["ruff"][-1] == "-"
    assert "--isolated" in commands["ruff"]
    assert commands["ruff"][commands["ruff"].index("--select") + 1] == "E9,F,B,ASYNC"
    assert commands["bandit"][-1] == "-"
    assert commands["bandit"][commands["bandit"].index("--severity-level") + 1] == "medium"
    assert all(stdin == b"import os" for _, stdin in _runs(runner))


async def test_clean_code_passes_both_analysers() -> None:
    sandbox, _ = _sandbox(ruff=_out(0, "[]"), bandit=_out(0, json.dumps({"results": []})))

    report = await sandbox.static_analysis("x = 1\n")

    assert report.passed
    assert report.findings == ()


async def test_an_analyser_timeout_fails_closed() -> None:
    sandbox, _ = _sandbox(ruff=_out(0, "[]"), bandit=_out(TIMEOUT_EXIT_CODE, timed_out=True))

    report = await sandbox.static_analysis("x = 1\n")

    assert not report.passed
    assert [f.rule for f in report.findings] == ["timeout"]


def test_bandit_findings_carry_severity_and_line() -> None:
    (finding,) = parse_bandit(_out(1, BANDIT_JSON))

    assert finding.rule == "B602"
    assert finding.line == 3
    assert finding.message.startswith("[HIGH]")


@pytest.mark.parametrize("parse", [parse_ruff, parse_bandit], ids=["ruff", "bandit"])
@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [(1, "not json"), (2, ""), (2, "[]"), (0, '{"unexpected": true}')],
)
def test_unparseable_or_silent_failures_become_findings(
    parse: Callable[[ProcessOutput], tuple[StaticFinding, ...]], returncode: int, stdout: str
) -> None:
    findings = parse(_out(returncode, stdout, stderr="analyser crashed"))

    assert [f.rule for f in findings] == ["unparseable-output"]
    assert "analyser crashed" in findings[0].message


# --------------------------------------------------------------------------- run_process


async def test_run_process_feeds_stdin_and_captures_both_streams() -> None:
    script = (
        "import sys; data = sys.stdin.read(); print(data.upper()); print('warn', file=sys.stderr)"
    )

    out = await run_process(
        [sys.executable, "-c", script], stdin=b"abc", timeout_s=30, output_limit=1024
    )

    assert out.returncode == 0
    assert out.stdout.strip() == "ABC"
    assert out.stderr.strip() == "warn"
    assert not out.timed_out


async def test_run_process_kills_at_the_deadline() -> None:
    out = await run_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=b"",
        timeout_s=0.5,
        output_limit=1024,
    )

    assert out.timed_out
    assert out.returncode == TIMEOUT_EXIT_CODE
    assert out.duration_s < 10


async def test_run_process_caps_a_flood_of_output() -> None:
    out = await run_process(
        [sys.executable, "-c", "print('x' * 200_000)"], stdin=b"", timeout_s=30, output_limit=1024
    )

    assert out.stdout.startswith("x" * 1024)
    assert out.stdout.endswith("[output truncated at 1024 bytes]")
    assert len(out.stdout) < 1100


async def test_run_process_tolerates_a_child_that_ignores_stdin() -> None:
    out = await run_process(
        [sys.executable, "-c", "pass"], stdin=b"x" * 1_000_000, timeout_s=30, output_limit=1024
    )

    assert out.returncode == 0
