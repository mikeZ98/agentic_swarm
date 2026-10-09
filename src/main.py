"""CLI entrypoint: run the swarm end-to-end and persist the generated artefacts.

uv run --frozen python -m src.main
uv run --frozen python -m src.main --prompt "Build a token-bucket microservice ..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from src.agents import build_agents
from src.config import Settings, get_settings
from src.graph import build_graph, recursion_limit
from src.state import SwarmState

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


def _tracing(
    settings: Settings, session_id: str
) -> tuple[list[BaseCallbackHandler], AbstractContextManager[Any]]:
    """Return Langfuse callbacks + trace-attribute context, or no-ops when disabled."""
    if not settings.tracing_enabled:
        log.info("Langfuse keys not set - tracing disabled.")
        return [], nullcontext()
    from langfuse import propagate_attributes  # noqa: PLC0415 - optional at runtime
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415

    attrs = propagate_attributes(
        session_id=session_id,
        trace_name="agentic_swarm",
        tags=["swarm", "langgraph"],
        metadata={"max_iterations": str(settings.max_iterations)},
    )
    return [CallbackHandler()], attrs


def _flush_tracing(settings: Settings) -> None:
    if settings.tracing_enabled:
        from langfuse import get_client  # noqa: PLC0415

        get_client().flush()


def _write_artifacts(state: SwarmState, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(state.architecture_spec, indent=2), "utf-8")
    (out / "backend.py").write_text(state.backend_code + "\n", "utf-8")
    (out / "index.html").write_text(state.frontend_code + "\n", "utf-8")
    (out / "eval_report.json").write_text(json.dumps(state.eval_report, indent=2), "utf-8")


async def run(prompt: str, settings: Settings, *, session_id: str) -> tuple[SwarmState, float]:
    graph = build_graph(build_agents(settings))
    callbacks, trace_ctx = _tracing(settings, session_id)
    config: RunnableConfig = {
        "callbacks": callbacks,
        "run_name": "agentic_swarm",
        "recursion_limit": recursion_limit(settings.max_iterations),
        "metadata": {"langfuse_session_id": session_id},
    }
    started = time.perf_counter()
    try:
        with trace_ctx:
            raw = await graph.ainvoke(SwarmState(user_prompt=prompt), config=config)
    finally:
        _flush_tracing(settings)
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
    except RuntimeError as exc:  # missing credentials etc.
        log.error("%s", exc)  # noqa: TRY400 - user-facing config error, no traceback
        return 2

    _write_artifacts(state, args.output_dir)
    report = state.eval_report
    log.info("status=%s iterations=%d score=%s", state.status, state.iteration, report.get("score"))
    for vuln in report.get("vulnerabilities", []):
        log.info("  open finding: %s", vuln)
    log.info("latency=%.1fs session=%s artefacts=%s", elapsed, session_id, args.output_dir)
    return 0 if report.get("is_accepted") else 1


if __name__ == "__main__":
    sys.exit(main())
