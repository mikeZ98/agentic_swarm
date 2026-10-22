"""Tracing is best effort: every Langfuse failure degrades to "no trace", never a failed run.

The SDK is replaced with stubs, so nothing here reaches a collector.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import langfuse
import langfuse.langchain
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr

from src.config import Settings
from src.graph import build_graph, recursion_limit
from src.state import Status, SwarmState
from src.tracing import DISABLED, Tracing, flush_tracing, start_tracing
from tests.test_graph import MAX_ITERATIONS, Recorder, verdict


def _settings(*, keys: bool = True) -> Settings:
    if not keys:
        return Settings(_env_file=None, LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)
    return Settings(
        _env_file=None,
        LANGFUSE_PUBLIC_KEY="pk-lf-test",
        LANGFUSE_SECRET_KEY=SecretStr("sk-lf-test"),
    )


class StubHandler(BaseCallbackHandler):
    pass


@contextlib.contextmanager
def _attrs(**_: Any) -> Iterator[None]:
    yield


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A working stand-in for the Langfuse SDK surface `start_tracing` touches."""
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", StubHandler)
    monkeypatch.setattr(langfuse, "propagate_attributes", _attrs)
    return monkeypatch


def test_tracing_is_disabled_without_keys() -> None:
    tracing = start_tracing(_settings(keys=False), "s-1")

    assert tracing is DISABLED
    assert tracing.callbacks == []


def test_tracing_wires_the_handler_and_trace_attributes(sdk: pytest.MonkeyPatch) -> None:
    tracing = start_tracing(_settings(), "s-1")

    assert tracing.enabled
    assert [type(cb) for cb in tracing.callbacks] == [StubHandler]
    with tracing.context:
        pass


def test_sdk_initialisation_failure_degrades_to_untraced(
    sdk: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*_: Any, **__: Any) -> None:
        raise ConnectionError("collector unreachable")

    sdk.setattr(langfuse.langchain, "CallbackHandler", broken)

    tracing = start_tracing(_settings(), "s-1")

    assert tracing is DISABLED
    assert "collector unreachable" in caplog.text


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_a_failing_trace_context_never_breaks_the_body(
    sdk: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, phase: str
) -> None:
    @contextlib.contextmanager
    def flaky(**_: Any) -> Iterator[None]:
        if phase == "enter":
            raise OSError("otel context error")
        yield
        raise OSError("otel context error")

    sdk.setattr(langfuse, "propagate_attributes", flaky)
    ran = False

    with start_tracing(_settings(), "s-1").context:
        ran = True

    assert ran
    assert "otel context error" in caplog.text


def test_a_body_exception_still_propagates_through_the_trace_context(
    sdk: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(KeyError), start_tracing(_settings(), "s-1").context:
        raise KeyError("from the graph")


async def test_flush_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Client:
        def flush(self) -> None:
            raise ConnectionError("network down")

    monkeypatch.setattr(langfuse, "get_client", Client)

    await flush_tracing(Tracing(enabled=True), timeout_s=1.0)

    assert "network down" in caplog.text


async def test_a_hanging_flush_is_bounded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Client:
        def flush(self) -> None:
            time.sleep(2.0)  # exporter stuck in retries against a black-holed collector

    monkeypatch.setattr(langfuse, "get_client", Client)
    started = time.perf_counter()

    await flush_tracing(Tracing(enabled=True), timeout_s=0.1)

    assert time.perf_counter() - started < 1.5
    assert "exceeded 0.1s" in caplog.text


async def test_disabled_tracing_never_touches_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> None:
        raise AssertionError("must not be called")

    monkeypatch.setattr(langfuse, "get_client", explode)

    await flush_tracing(DISABLED, timeout_s=0.1)


async def test_an_exploding_callback_handler_does_not_fail_the_swarm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ExplodingHandler(BaseCallbackHandler):
        def on_chain_start(self, *args: Any, run_id: UUID, **kwargs: Any) -> None:
            raise ConnectionError("span export failed")

    caplog.set_level(logging.WARNING)
    graph = build_graph(Recorder([verdict(accepted=True)]).agents())

    raw = await asyncio.wait_for(
        graph.ainvoke(
            SwarmState(user_prompt="build it"),
            config={
                "callbacks": [ExplodingHandler()],
                "recursion_limit": recursion_limit(MAX_ITERATIONS),
            },
        ),
        timeout=30,
    )

    assert SwarmState.model_validate(raw).status == Status.ACCEPTED
    assert "span export failed" in caplog.text
