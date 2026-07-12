# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
COPY pyproject.toml poetry.lock ./
COPY README.md ./
COPY src ./src
RUN poetry install --only main --no-ansi

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY params.yaml dvc.yaml ./
COPY scripts ./scripts
RUN mkdir -p data models reports && chown -R app:app /app

USER app
CMD ["python", "-m", "recommender.training.train"]
