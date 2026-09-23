"""The four swarm agents and their production wiring."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.architect import ArchitectAgent
from src.agents.backend import BackendAgent
from src.agents.base import AgentNode
from src.agents.critic import CriticAgent
from src.agents.frontend import FrontendAgent
from src.agents.llm import build_chat_model, build_engineer_model
from src.config import Settings
from src.state import ArchitectureSpec, CriticVerdict

__all__ = [
    "AgentNode",
    "ArchitectAgent",
    "BackendAgent",
    "CriticAgent",
    "FrontendAgent",
    "SwarmAgents",
    "build_agents",
]


@dataclass(frozen=True, slots=True)
class SwarmAgents:
    """The four node callables. Tests inject fakes; production uses `build_agents`."""

    architect: AgentNode
    backend: AgentNode
    frontend: AgentNode
    critic: AgentNode


def build_agents(settings: Settings) -> SwarmAgents:
    architect_llm = build_chat_model(settings.architect, settings, role="architect")
    critic_llm = build_chat_model(settings.critic, settings, role="critic")
    return SwarmAgents(
        architect=ArchitectAgent(
            architect_llm.with_structured_output(ArchitectureSpec, method="json_schema")
        ),
        backend=BackendAgent(build_engineer_model(settings.backend, settings, role="backend")),
        frontend=FrontendAgent(build_engineer_model(settings.frontend, settings, role="frontend")),
        critic=CriticAgent(
            critic_llm.with_structured_output(CriticVerdict, method="json_schema"),
            max_iterations=settings.max_iterations,
            acceptance_threshold=settings.acceptance_threshold,
        ),
    )
