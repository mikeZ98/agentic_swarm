"""Container security policy, enforced on the Dockerfiles and compose file as written.

Images are built and probed in CI; these checks keep the trust boundaries from eroding in
review: only the sandbox server can reach a Docker daemon, nothing mounts the host socket,
everything but dind is hardened, and the names that tie the stack together agree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import PROJECT_ROOT, McpSettings, SandboxSettings

COMPOSE: dict[str, Any] = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
SERVICES: dict[str, dict[str, Any]] = COMPOSE["services"]
DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text()
SANDBOX_DIR = PROJECT_ROOT / "docker" / "sandbox"
SANDBOX_DOCKERFILE = (SANDBOX_DIR / "Dockerfile").read_text()
HARDENED = [name for name in SERVICES if name != "dind"]


def _stages(dockerfile: str) -> dict[str, str]:
    """`FROM ... AS name` → that stage's instructions (continuations joined)."""
    text = dockerfile.replace("\\\n", " ")
    parts = re.split(r"^FROM\s+", text, flags=re.MULTILINE)[1:]
    stages: dict[str, str] = {}
    for part in parts:
        header, _, body = part.partition("\n")
        name = header.split(" AS ")[-1].strip() if " AS " in header else header.strip()
        stages[name] = body
    return stages


STAGES = _stages(DOCKERFILE)


# --------------------------------------------------------------------------- compose


def test_nothing_mounts_the_host_docker_socket() -> None:
    for name, service in SERVICES.items():
        for volume in service.get("volumes", []):
            assert "docker.sock" not in str(volume), name


def test_only_dind_is_privileged() -> None:
    assert [name for name, s in SERVICES.items() if s.get("privileged")] == ["dind"]


def test_no_service_publishes_a_port() -> None:
    assert [name for name, s in SERVICES.items() if "ports" in s] == []


@pytest.mark.parametrize("name", HARDENED)
def test_every_other_service_is_read_only_capless_and_non_escalating(name: str) -> None:
    service = SERVICES[name]

    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]


def test_the_agents_cannot_reach_any_docker_daemon() -> None:
    swarm = SERVICES["swarm"]

    assert "DOCKER_HOST" not in swarm.get("environment", {})
    assert "sandbox" not in swarm["networks"]
    assert STAGES["swarm"].count("docker-cli") == 0


def test_the_sandbox_server_reaches_dind_not_the_host() -> None:
    env = SERVICES["sandbox-mcp"]["environment"]

    assert env["DOCKER_HOST"] == "tcp://dind:2375"
    assert set(SERVICES["sandbox-mcp"]["networks"]) == {"sandbox", "agents"}
    assert SERVICES["dind"]["networks"] == ["sandbox"]


def test_stack_names_agree_across_files() -> None:
    image = SandboxSettings().image
    url = SERVICES["swarm"]["environment"]["SWARM_MCP__URL"]

    assert SERVICES["sandbox-mcp"]["environment"]["SWARM_SANDBOX__IMAGE"] == image
    assert f"--tag {image}" in " ".join(SERVICES["sandbox-image"]["command"])
    assert url == "http://sandbox-mcp:8765/mcp"
    assert f"SWARM_MCP__URL={url}" in " ".join(STAGES["swarm"].split())
    assert '"--allowed-host", "sandbox-mcp:*"' in STAGES["sandbox-server"]
    assert McpSettings().url.endswith(":8765/mcp")


# --------------------------------------------------------------------------- Dockerfiles


def test_swarm_is_the_default_build_target() -> None:
    assert list(STAGES)[-1] == "swarm"


@pytest.mark.parametrize("stage", ["sandbox-server", "swarm"])
def test_runtime_stages_run_as_the_unprivileged_app_user(stage: str) -> None:
    assert re.search(r"^USER app:app$", STAGES[stage], flags=re.MULTILINE)


def test_only_the_sandbox_server_ships_a_docker_cli() -> None:
    with_cli = [name for name, body in STAGES.items() if "--from=docker-cli" in body]

    assert with_cli == ["sandbox-server"]


def test_the_runtime_has_no_build_tooling() -> None:
    assert "--from=uv" not in STAGES["runtime"]
    assert "apt-get" not in STAGES["runtime"]


@pytest.mark.parametrize("dockerfile", [DOCKERFILE, SANDBOX_DOCKERFILE], ids=["app", "sandbox"])
def test_base_images_are_pinned_to_exact_versions(dockerfile: str) -> None:
    images = re.findall(r"^ARG \w+_IMAGE=(\S+)$", dockerfile, flags=re.MULTILINE)

    assert images
    for image in images:
        assert re.search(r":\d+\.\d+\.\d+", image), image


def test_compose_images_are_pinned_to_exact_versions() -> None:
    for name, service in SERVICES.items():
        if "build" not in service:
            assert re.search(r":\d+\.\d+\.\d+-", service["image"]), name


def test_sandbox_image_installs_only_hash_locked_wheels_and_drops_root() -> None:
    assert "--require-hashes" in SANDBOX_DOCKERFILE
    assert re.search(r"^USER 65534:65534$", SANDBOX_DOCKERFILE, flags=re.MULTILINE)
    requirements = (SANDBOX_DIR / "requirements.txt").read_text()
    pins = re.findall(r"^([a-z0-9-]+)==", requirements, flags=re.MULTILINE)
    assert {"ruff", "bandit", "fastapi", "pydantic", "uvicorn"} <= set(pins)
    assert requirements.count("--hash=sha256:") >= len(pins)


def test_sandbox_ruff_matches_the_repo_lockfile() -> None:
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    (locked,) = [p["version"] for p in lock["package"] if p["name"] == "ruff"]
    requirements = (SANDBOX_DIR / "requirements.txt").read_text()

    assert f"ruff=={locked}" in requirements


def test_build_context_is_an_allow_list() -> None:
    lines = [
        line
        for line in Path(PROJECT_ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]

    assert lines[0] == "*"
    assert "!.env" not in lines
    assert {"!pyproject.toml", "!uv.lock", "!src/"} <= set(lines)
