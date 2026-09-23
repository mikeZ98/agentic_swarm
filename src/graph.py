"""Hierarchical StateGraph with a deterministic self-correction loop.

    START -> architect -> backend -(+)-> frontend -> critic -(+)-> END
                            ^     \\___________________^   |
                            |                              |
                            +---- revise (iteration += 1) -+--> frontend

Routing functions are pure functions of `SwarmState` so they are unit-testable in isolation.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents import SwarmAgents
from src.agents.base import NodeUpdate
from src.state import Status, SwarmState

type SwarmGraph = CompiledStateGraph[SwarmState, None, SwarmState, SwarmState]

ARCHITECT = "architect"
BACKEND = "backend"
FRONTEND = "frontend"
CRITIC = "critic"

#: Worst case: 4 initial hops + 3 hops per revision (backend -> frontend -> critic).
MAX_GRAPH_STEPS_PER_REVISION = 3
BASE_GRAPH_STEPS = 4


def route_after_backend(state: SwarmState) -> str:
    """A backend-only fix skips the frontend; initial passes and `both` fixes continue to it."""
    if state.status == Status.BACKEND_READY and state.revision_target == "backend":
        return CRITIC
    return FRONTEND


def route_after_critic(state: SwarmState) -> str:
    """Accepted or out of retries -> END; otherwise send feedback to the owning engineer."""
    if state.status != Status.REVISING:
        return END
    if state.revision_target == "frontend":
        return FRONTEND
    return BACKEND


def recursion_limit(max_iterations: int) -> int:
    return BASE_GRAPH_STEPS + MAX_GRAPH_STEPS_PER_REVISION * max_iterations + 1


def build_graph(agents: SwarmAgents) -> SwarmGraph:
    builder = StateGraph(SwarmState)
    builder.add_node(ARCHITECT, agents.architect)
    builder.add_node(BACKEND, agents.backend)
    builder.add_node(FRONTEND, agents.frontend)
    builder.add_node(CRITIC, agents.critic)

    builder.add_edge(START, ARCHITECT)
    builder.add_edge(ARCHITECT, BACKEND)
    builder.add_conditional_edges(BACKEND, route_after_backend, [FRONTEND, CRITIC])
    builder.add_edge(FRONTEND, CRITIC)
    builder.add_conditional_edges(CRITIC, route_after_critic, [BACKEND, FRONTEND, END])

    return builder.compile(name="swarm-showcase")


async def _placeholder_node(state: SwarmState) -> NodeUpdate:  # noqa: ARG001 - name required by _Node
    raise RuntimeError("topology-only graph: nodes are not executable")


def build_topology_graph() -> SwarmGraph:
    """The production topology with inert nodes, for visualisation without credentials."""
    return build_graph(
        SwarmAgents(
            architect=_placeholder_node,
            backend=_placeholder_node,
            frontend=_placeholder_node,
            critic=_placeholder_node,
        )
    )
