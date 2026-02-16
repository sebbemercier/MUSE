# Copyright 2026 The OpenSLM Project
FROM python:3.9-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv /usr/bin/

COPY pyproject.toml .
COPY MUSE/ MUSE/
COPY common/ common/

RUN uv pip install --system -e .

WORKDIR /app/MUSE

CMD ["python", "main.py"]
