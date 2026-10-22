"""CLI entrypoint: run the swarm end-to-end and persist the generated artefacts.

uv run --frozen python -m src.main
uv run --frozen python -m src.main --prompt "Build a token-bucket microservice ..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.agents import build_agents
from src.config import Settings, get_settings
from src.graph import build_graph, recursion_limit
from src.sandbox import CodeTools, open_code_tools
from src.state import SwarmState
from src.tracing import flush_tracing, start_tracing

DEFAULT_PROMPT = (
    "Build a rate-limited URL shortener microservice. Users submit a long URL and receive a "
    "short code; visiting /{code} redirects. Enforce a per-client token-bucket rate limit "
    "(10 requests / minute, burst 5) returning HTTP 429 with Retry-After. Reject non-http(s) "
    "URLs. Provide a stats endpoint with hit counts. Ship a minimal web UI to shorten URLs, "
    "copy the short link and view stats."
)

log = logging.getLogger("swarm")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentic_swarm", description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Product requirement.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--session-id", default=None, help="Langfuse session id.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _write_artifacts(state: SwarmState, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if state.architecture_spec is not None:
        (out / "spec.json").write_text(state.architecture_spec.model_dump_json(indent=2), "utf-8")
    (out / "backend.py").write_text(state.backend_code + "\n", "utf-8")
    (out / "index.html").write_text(state.frontend_code + "\n", "utf-8")
    if state.eval_report is not None:
        (out / "eval_report.json").write_text(state.eval_report.model_dump_json(indent=2), "utf-8")


async def run(prompt: str, settings: Settings, *, session_id: str) -> tuple[SwarmState, float]:
    async with open_code_tools(settings.mcp) as tools:
        return await _run_graph(prompt, settings, tools, session_id=session_id)


async def _run_graph(
    prompt: str, settings: Settings, tools: CodeTools, *, session_id: str
) -> tuple[SwarmState, float]:
    graph = build_graph(build_agents(settings, tools))
    tracing = start_tracing(settings, session_id)
    config: RunnableConfig = {
        "callbacks": tracing.callbacks,
        "run_name": "agentic_swarm",
        "recursion_limit": recursion_limit(settings.max_iterations),
        "metadata": {"langfuse_session_id": session_id},
    }
    started = time.perf_counter()
    try:
        with tracing.context:
            raw = await graph.ainvoke(SwarmState(user_prompt=prompt), config=config)
    finally:
        await flush_tracing(tracing, timeout_s=settings.tracing_flush_timeout_s)
    return SwarmState.model_validate(raw), time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    settings = get_settings()
    session_id = args.session_id or f"swarm-{uuid.uuid4().hex[:12]}"

    try:
        state, elapsed = asyncio.run(run(args.prompt, settings, session_id=session_id))
    except RuntimeError as exc:  # missing credentials, unreachable sandbox (SandboxError)
        log.error("%s", exc)  # noqa: TRY400 - user-facing config error, no traceback
        return 2

    _write_artifacts(state, args.output_dir)
    report = state.eval_report
    score = report.score if report is not None else None
    log.info("status=%s iterations=%d score=%s", state.status, state.iteration, score)
    for vuln in report.vulnerabilities if report is not None else ():
        log.info("  open finding: %s", vuln)
    log.info("latency=%.1fs session=%s artefacts=%s", elapsed, session_id, args.output_dir)
    return 0 if report is not None and report.is_accepted else 1


if __name__ == "__main__":
    sys.exit(main())
