"""The four swarm agents and their production wiring."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.architect import ArchitectAgent
from src.agents.backend import BackendAgent
from src.agents.base import AgentNode
from src.agents.critic import CriticAgent
from src.agents.frontend import FrontendAgent
from src.agents.llm import build_engineer_model, build_structured_model
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
    return SwarmAgents(
        architect=ArchitectAgent(
            build_structured_model(settings.architect, settings, ArchitectureSpec, role="architect")
        ),
        backend=BackendAgent(build_engineer_model(settings.backend, settings, role="backend")),
        frontend=FrontendAgent(build_engineer_model(settings.frontend, settings, role="frontend")),
        critic=CriticAgent(
            build_structured_model(settings.critic, settings, CriticVerdict, role="critic"),
            max_iterations=settings.max_iterations,
            acceptance_threshold=settings.acceptance_threshold,
        ),
    )
