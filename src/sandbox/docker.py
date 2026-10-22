"""Docker-backed sandbox: every snippet runs in a throwaway, locked-down container.

Code is streamed over stdin, never written to the host, and the container has no network,
a read-only root filesystem, no capabilities, an unprivileged uid and hard pid / memory /
CPU / wall-clock limits. The process runner is injected so tests never need a daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import SandboxSettings
from src.sandbox.base import SandboxError
from src.state import (
    AnalysisTool,
    ExecutionResult,
    StaticAnalysisReport,
    StaticFinding,
    ToolRun,
)

#: `timeout(1)` convention for a run killed at its deadline.
TIMEOUT_EXIT_CODE = 124
#: `docker run` itself failed (125) or the image's command is broken (126 / 127). The
#: in-container runner maps every snippet exit onto 0 / 1, so these always mean infrastructure.
DOCKER_FAILURE_CODES = frozenset({125, 126, 127})
#: Unprivileged `nobody:nogroup`, independent of any USER baked into the image.
SANDBOX_USER = "65534:65534"
#: The only writable path in the container: a small tmpfs *inside* the sandbox, never a host
#: directory, so the temp-path warnings (ruff S108 / bandit B108) do not apply.
SANDBOX_TMPFS = "/tmp:rw,noexec,nosuid,nodev,size=16m"  # noqa: S108  # nosec B108
SANDBOX_WORKDIR = "/tmp"  # noqa: S108  # nosec B108
_READ_CHUNK = 64 * 1024

# Executes stdin as module `__sandbox__` and folds every exit status onto 0 / 1, so a
# snippet cannot impersonate a docker failure code with `sys.exit(125)`.
_PYTHON_RUNNER = """\
import sys
source = sys.stdin.read()
try:
    exec(compile(source, "snippet.py", "exec"), {"__name__": "__sandbox__"})
except SystemExit as stop:
    sys.exit(0 if stop.code in (None, 0) else 1)
"""


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


class CommandRunner(Protocol):
    async def __call__(
        self, argv: Sequence[str], *, stdin: bytes, timeout_s: float, output_limit: int
    ) -> ProcessOutput: ...


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> str:
    """Keep the first `limit` bytes and drain the rest, so a flood cannot exhaust memory."""
    kept = bytearray()
    truncated = False
    while chunk := await stream.read(_READ_CHUNK):
        room = limit - len(kept)
        kept += chunk[:room]
        truncated = truncated or len(chunk) > room
    text = kept.decode("utf-8", errors="replace")
    return f"{text}\n[output truncated at {limit} bytes]" if truncated else text


async def run_process(
    argv: Sequence[str], *, stdin: bytes, timeout_s: float, output_limit: int
) -> ProcessOutput:
    """Run `argv` to completion or kill it at `timeout_s`; output is capped per stream."""
    started = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin_pipe, stdout_pipe, stderr_pipe = proc.stdin, proc.stdout, proc.stderr
    if stdin_pipe is None or stdout_pipe is None or stderr_pipe is None:  # pragma: no cover
        raise SandboxError("subprocess pipes were not created")  # PIPE guarantees them

    async def feed(writer: asyncio.StreamWriter) -> None:
        try:
            writer.write(stdin)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the process exited without reading all of its input
        finally:
            writer.close()

    try:
        async with asyncio.timeout(timeout_s):
            _, out, err = await asyncio.gather(
                feed(stdin_pipe),
                _read_capped(stdout_pipe, output_limit),
                _read_capped(stderr_pipe, output_limit),
            )
            returncode = await proc.wait()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ProcessOutput(
            returncode=TIMEOUT_EXIT_CODE,
            stdout="",
            stderr=f"killed after {timeout_s:g}s",
            timed_out=True,
            duration_s=time.perf_counter() - started,
        )
    return ProcessOutput(
        returncode=returncode,
        stdout=out,
        stderr=err,
        timed_out=False,
        duration_s=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- analyser output


def _unparseable(tool: AnalysisTool, run: ProcessOutput) -> StaticFinding:
    detail = (run.stderr or run.stdout).strip()[:500]
    return StaticFinding(
        tool=tool,
        rule="unparseable-output",
        line=0,
        message=f"{tool} exited {run.returncode} without a parseable report: {detail}",
    )


def parse_ruff(run: ProcessOutput) -> tuple[StaticFinding, ...]:
    """`ruff check --output-format json` → findings (syntax errors carry no rule code)."""
    try:
        items: list[dict[str, Any]] = json.loads(run.stdout)
        findings = tuple(
            StaticFinding(
                tool="ruff",
                rule=str(item.get("code") or "syntax-error"),
                line=int((item.get("location") or {}).get("row") or 0),
                message=str(item["message"]),
            )
            for item in items
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return (_unparseable("ruff", run),)
    if run.returncode != 0 and not findings:
        return (_unparseable("ruff", run),)
    return findings


def parse_bandit(run: ProcessOutput) -> tuple[StaticFinding, ...]:
    """`bandit --format json` → findings, severity folded into the message."""
    try:
        report: dict[str, Any] = json.loads(run.stdout)
        findings = tuple(
            StaticFinding(
                tool="bandit",
                rule=str(item["test_id"]),
                line=int(item.get("line_number") or 0),
                message=f"[{item.get('issue_severity', '?')}] {item['issue_text']}",
            )
            for item in report["results"]
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return (_unparseable("bandit", run),)
    if run.returncode != 0 and not findings:
        return (_unparseable("bandit", run),)
    return findings


# --------------------------------------------------------------------------- sandbox


class DockerSandbox:
    """`CodeTools` implementation backed by `docker run`; lives inside the MCP server only."""

    def __init__(self, settings: SandboxSettings, runner: CommandRunner = run_process) -> None:
        self._settings = settings
        self._runner = runner
        self._ready = False

    def container_argv(self, name: str) -> list[str]:
        """`docker run` prefix with every isolation flag; the in-container command follows."""
        s = self._settings
        return [
            s.docker_bin, "run", "--rm", "--interactive",
            "--name", name,
            "--label", "agentic_swarm.sandbox=1",
            "--pull", "never",
            "--network", "none",
            "--read-only",
            "--tmpfs", SANDBOX_TMPFS,
            "--workdir", SANDBOX_WORKDIR,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", SANDBOX_USER,
            "--pids-limit", str(s.pids_limit),
            "--memory", s.memory,
            "--memory-swap", s.memory,
            "--cpus", f"{s.cpus:g}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            s.image,
        ]  # fmt: skip

    async def _invoke(
        self, argv: Sequence[str], *, stdin: bytes, timeout_s: float
    ) -> ProcessOutput:
        try:
            return await self._runner(
                argv,
                stdin=stdin,
                timeout_s=timeout_s,
                output_limit=self._settings.output_limit_bytes,
            )
        except OSError as exc:
            raise SandboxError(f"cannot start '{self._settings.docker_bin}': {exc}") from exc

    async def _ensure_ready(self) -> None:
        """Fail fast unless the daemon answers and the image is present locally.

        With the daemon down, `docker run` exits 1, which is indistinguishable from a failing
        snippet; checking first keeps infrastructure faults out of the critic's verdict.
        """
        if self._ready:
            return
        image = self._settings.image
        probe = [self._settings.docker_bin, "image", "inspect", "--format", "{{.Id}}", image]
        out = await self._invoke(probe, stdin=b"", timeout_s=30.0)
        if out.returncode != 0:
            raise SandboxError(
                f"sandbox image '{image}' is unavailable (Docker daemon down or image not "
                f"built): {out.stderr.strip()[:500]}"
            )
        self._ready = True

    async def _run(self, command: Sequence[str], code: str) -> ProcessOutput:
        await self._ensure_ready()
        name = f"swarm-sbx-{uuid.uuid4().hex[:12]}"
        out = await self._invoke(
            [*self.container_argv(name), *command],
            stdin=code.encode("utf-8"),
            timeout_s=self._settings.timeout_s,
        )
        if out.timed_out:
            # Killing the docker CLI does not stop the container; remove it explicitly.
            await self._force_remove(name)
        elif out.returncode in DOCKER_FAILURE_CODES:
            raise SandboxError(f"docker run failed ({out.returncode}): {out.stderr.strip()[:500]}")
        return out

    async def _force_remove(self, name: str) -> None:
        # Best effort: --rm reaps the container once its process dies anyway.
        with contextlib.suppress(OSError):
            await self._runner(
                [self._settings.docker_bin, "rm", "--force", name],
                stdin=b"",
                timeout_s=10.0,
                output_limit=4096,
            )

    async def execute_python(self, code: str) -> ExecutionResult:
        out = await self._run(["python", "-I", "-c", _PYTHON_RUNNER], code)
        return ExecutionResult(
            exit_code=out.returncode,
            stdout=out.stdout,
            stderr=out.stderr,
            timed_out=out.timed_out,
            duration_s=out.duration_s,
        )

    async def _analyse(self, tool: AnalysisTool, command: Sequence[str], code: str) -> ToolRun:
        out = await self._run(command, code)
        findings: tuple[StaticFinding, ...]
        if out.timed_out:
            limit = f"{self._settings.timeout_s:g}s"
            findings = (
                StaticFinding(
                    tool=tool, rule="timeout", line=0, message=f"{tool} exceeded {limit}"
                ),
            )
        else:
            findings = parse_ruff(out) if tool == "ruff" else parse_bandit(out)
        return ToolRun(
            tool=tool, exit_code=out.returncode, timed_out=out.timed_out, findings=findings
        )

    async def static_analysis(self, code: str) -> StaticAnalysisReport:
        s = self._settings
        ruff = [
            "ruff", "check", "--isolated", "--no-cache", "--output-format", "json",
            "--select", ",".join(s.ruff_select), "--stdin-filename", "snippet.py", "-",
        ]  # fmt: skip
        bandit = [
            "bandit", "--format", "json", "--quiet",
            "--severity-level", s.bandit_severity,
            "--confidence-level", s.bandit_confidence, "-",
        ]  # fmt: skip
        runs = await asyncio.gather(
            self._analyse("ruff", ruff, code), self._analyse("bandit", bandit, code)
        )
        return StaticAnalysisReport(runs=tuple(runs))
