"""Every inter-agent payload is a closed, frozen, strict contract."""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel, SecretStr, ValidationError

from src import state as state_module
from src.agents.base import parse_structured
from src.agents.llm import build_structured_model
from src.config import ModelConfig, Provider, Settings
from src.state import (
    LLM_OUTPUT_SCHEMAS,
    ArchitectureSpec,
    Contract,
    CriticReport,
    CriticVerdict,
    EvaluationResult,
    FieldSpec,
    Status,
    SwarmState,
)


def _schemas(module: Any) -> list[type[BaseModel]]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == module.__name__ and obj is not Contract
    ]


ALL_SCHEMAS = _schemas(state_module)

VERDICT_JSON: dict[str, Any] = {
    "is_accepted": False,
    "vulnerabilities": ["XSS in stats table"],
    "performance_notes": [],
    "score": 40,
    "defect_owner": "frontend",
}


def test_schema_discovery_found_every_contract() -> None:
    names = {schema.__name__ for schema in ALL_SCHEMAS}
    assert {"SwarmState", "ArchitectureSpec", "CriticVerdict", "CriticReport"} <= names


@pytest.mark.parametrize("schema", ALL_SCHEMAS, ids=lambda schema: schema.__name__)
def test_every_schema_is_closed_frozen_and_strict(schema: type[BaseModel]) -> None:
    config = schema.model_config

    assert config.get("extra") == "forbid"
    assert config.get("frozen") is True
    assert config.get("strict") is True


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FieldSpec.model_validate({"name": "url", "type": "str", "constraints": "", "pii": True})


def test_state_is_immutable() -> None:
    state = SwarmState(user_prompt="x")

    with pytest.raises(ValidationError, match="frozen"):
        state.iteration = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [("iteration", "1"), ("iteration", 1.0), ("status", "accepted"), ("backend_code", b"x")],
)
def test_state_rejects_implicit_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SwarmState.model_validate({"user_prompt": "x", field: value})


def test_state_rejects_an_empty_prompt() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        SwarmState(user_prompt="")


def test_evaluation_score_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        EvaluationResult(is_accepted=True, vulnerabilities=(), performance_notes=(), score=101)


def test_report_extends_the_verdict_with_the_reviewed_iteration() -> None:
    verdict = CriticVerdict.model_validate_json(json.dumps(VERDICT_JSON))
    report = CriticReport(**verdict.model_dump(), reviewed_iteration=0)

    assert report.vulnerabilities == ("XSS in stats table",)
    assert SwarmState(user_prompt="x", eval_report=report, status=Status.REVISING).eval_report


# --------------------------------------------------------------------------- LLM boundary


def test_parse_structured_maps_json_arrays_onto_tuples() -> None:
    verdict = parse_structured(VERDICT_JSON, CriticVerdict)

    assert verdict.vulnerabilities == ("XSS in stats table",)
    assert verdict.defect_owner == "frontend"


def test_parse_structured_passes_instances_through() -> None:
    verdict = parse_structured(VERDICT_JSON, CriticVerdict)

    assert parse_structured(verdict, CriticVerdict) is verdict


def test_parse_structured_revalidates_a_foreign_model() -> None:
    class Loose(BaseModel):
        is_accepted: bool
        vulnerabilities: list[str]
        performance_notes: list[str]
        score: int
        defect_owner: str

    verdict = parse_structured(Loose(**VERDICT_JSON), CriticVerdict)

    assert verdict.score == 40


@pytest.mark.parametrize(
    "patch",
    [
        {"score": "90"},  # strings are not numbers
        {"score": 90.5},  # nor are fractional floats
        {"is_accepted": "true"},
        {"defect_owner": "devops"},
        {"severity": "high"},  # unknown key
    ],
)
def test_parse_structured_rejects_coercion_and_unknown_keys(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_structured(VERDICT_JSON | patch, CriticVerdict)


def _object_schemas(schema: dict[str, Any]) -> list[dict[str, Any]]:
    defs = list(schema.get("$defs", {}).values())
    return [node for node in [schema, *defs] if node.get("type") == "object"]


@pytest.mark.parametrize("schema", LLM_OUTPUT_SCHEMAS, ids=lambda schema: schema.__name__)
def test_llm_output_schemas_satisfy_openai_strict_mode(schema: type[Contract]) -> None:
    """Strict decoding needs every object closed and every property required."""
    for node in _object_schemas(schema.model_json_schema()):
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])


def _settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        ANTHROPIC_API_KEY=SecretStr("sk-ant-test"),
        OPENAI_API_KEY=SecretStr("sk-test"),
        OPENROUTER_API_KEY=SecretStr("sk-or-test"),
        **overrides,
    )


def _bound_kwargs(runnable: object) -> dict[str, Any]:
    assert isinstance(runnable, RunnableSequence)
    binding = runnable.first
    assert isinstance(binding, RunnableBinding)
    return dict(binding.kwargs)


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.OPENROUTER])
def test_openai_compatible_providers_get_a_strict_named_schema(provider: Provider) -> None:
    cfg = ModelConfig(provider=provider, model="gpt-4o")
    runnable = build_structured_model(cfg, _settings(), CriticVerdict, role="critic")

    json_schema = _bound_kwargs(runnable)["response_format"]["json_schema"]

    assert json_schema["name"] == "CriticVerdict"
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False


def test_anthropic_gets_the_plain_json_schema() -> None:
    cfg = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")
    runnable = build_structured_model(cfg, _settings(), ArchitectureSpec, role="architect")

    output_format = _bound_kwargs(runnable)["output_config"]["format"]

    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"].keys() == ArchitectureSpec.model_fields.keys()
