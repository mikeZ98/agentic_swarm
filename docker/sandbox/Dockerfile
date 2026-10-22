# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------- #
# Sandbox image — the throwaway container every generated snippet runs in.
#
#   docker build -t agentic-swarm-sandbox:local docker/sandbox
#   (compose builds it inside the dind daemon: `docker compose up sandbox-image`)
#
# Contents: CPython + ruff + bandit + the libraries generated backends import
# (FastAPI / Pydantic / uvicorn), pinned with hashes in requirements.txt.
# The image carries no secrets and no project code. Isolation is applied at run
# time by src/sandbox/docker.py (--network none, --read-only, --cap-drop ALL,
# no-new-privileges, uid 65534, pid/memory/CPU limits); USER below is a second
# line of defence for anyone who runs the image by hand.
# --------------------------------------------------------------------------- #
ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm
FROM ${PYTHON_IMAGE}

# --require-hashes: every wheel, transitive ones included, must match the lock.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.txt \
 && ruff --version && bandit --version >/dev/null \
 && python -c "import fastapi, pydantic, uvicorn"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /tmp
USER 65534:65534
CMD ["python", "--version"]
