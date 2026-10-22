"""Langfuse tracing — best effort by contract.

A missing SDK, a bad key, a failing context or an unreachable collector degrades to
"no trace" with a warning; it never fails or stalls a swarm run. LangChain already isolates
callback errors (handlers run with `raise_error=False`); this module guards everything
around them and bounds the final flush, which otherwise blocks on exporter retries.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Tracing:
    enabled: bool
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    context: AbstractContextManager[Any] = field(default_factory=nullcontext)


DISABLED = Tracing(enabled=False)


@contextmanager
def _guarded(ctx: AbstractContextManager[Any]) -> Iterator[None]:
    """Enter/exit `ctx` without letting its own failures escape; body errors still propagate."""
    entered = False
    try:
        ctx.__enter__()
        entered = True
    except Exception as exc:
        log.warning("Langfuse trace attributes unavailable (%s); continuing untraced.", exc)
    try:
        yield
    finally:
        if entered:
            try:
                ctx.__exit__(None, None, None)
            except Exception as exc:
                log.warning("Langfuse trace context failed to close (%s).", exc)


def start_tracing(settings: Settings, session_id: str) -> Tracing:
    if not settings.tracing_enabled:
        log.info("Langfuse keys not set - tracing disabled.")
        return DISABLED
    try:
        from langfuse import propagate_attributes  # noqa: PLC0415 - optional at runtime
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415

        handler = CallbackHandler()
        attrs = propagate_attributes(
            session_id=session_id,
            trace_name="agentic_swarm",
            tags=["swarm", "langgraph"],
            metadata={"max_iterations": str(settings.max_iterations)},
        )
    except Exception as exc:
        log.warning("Langfuse unavailable (%s); continuing untraced.", exc)
        return DISABLED
    return Tracing(enabled=True, callbacks=[handler], context=_guarded(attrs))


async def flush_tracing(tracing: Tracing, *, timeout_s: float) -> None:
    """Export buffered spans, giving up after `timeout_s` (the exporter retries on failure)."""
    if not tracing.enabled:
        return
    try:
        from langfuse import get_client  # noqa: PLC0415

        await asyncio.wait_for(asyncio.to_thread(get_client().flush), timeout=timeout_s)
    except TimeoutError:
        log.warning("Langfuse flush exceeded %.1fs; buffered spans may be lost.", timeout_s)
    except Exception as exc:
        log.warning("Langfuse flush failed (%s); buffered spans may be lost.", exc)
