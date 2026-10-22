# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------- #
# agentic_swarm images — multi-stage, non-root, read-only code.
#
#   docker build -t agentic_swarm:local .                                  # swarm CLI
#   docker build -t agentic_swarm-sandbox-mcp:local --target sandbox-server .
#
# builder         resolves the locked graph with uv into /app/.venv (no dev group)
#                 and smoke-imports both entrypoints.
# runtime         same base image (the venv's interpreter symlink must resolve),
#                 no compilers, no uv; code and venv owned by root, run as `app`.
# sandbox-server  runtime + a static docker CLI: the MCP server that starts
#                 sandbox containers. The ONLY image that can reach a Docker daemon.
# swarm           (default) the agents. No docker CLI and no daemon access: they
#                 execute code exclusively through the sandbox-server's MCP tools.
# --------------------------------------------------------------------------- #
ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7
ARG DOCKER_CLI_IMAGE=docker:29.2.1-cli

FROM ${UV_IMAGE} AS uv
FROM ${DOCKER_CLI_IMAGE} AS docker-cli

# ------------------------------- builder ----------------------------------- #
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first: this layer is reused until uv.lock / pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN /app/.venv/bin/python -c "import src.main, src.sandbox.server"

# ------------------------------- runtime ----------------------------------- #
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid "${APP_GID}" app \
 && useradd --system --uid "${APP_UID}" --gid app --home-dir /home/app --create-home \
        --shell /usr/sbin/nologin app \
 && install -d -o app -g app /app/output

ENV PATH=/app/.venv/bin:${PATH} \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
# Owned by root and not writable by `app`: the code cannot be modified at runtime.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# ---------------------------- sandbox-server ------------------------------- #
FROM runtime AS sandbox-server
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

# The CLI wants a writable config dir; compose mounts /tmp as tmpfs.
ENV DOCKER_CONFIG=/tmp/.docker

USER app:app
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', 8765), 4).close()"]

ENTRYPOINT ["python", "-m", "src.sandbox.server"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8765", \
     "--allowed-host", "sandbox-mcp:*"]

# -------------------------------- swarm ------------------------------------ #
FROM runtime AS swarm

# Agents reach the sandbox over MCP; there is no local server to spawn here.
ENV SWARM_MCP__TRANSPORT=streamable-http \
    SWARM_MCP__URL=http://sandbox-mcp:8765/mcp

USER app:app
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--output-dir", "/app/output"]
