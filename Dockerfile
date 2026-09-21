# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Install dependencies first (better layer caching), then the package itself.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# Runtime-only install: skips the pytest/ruff/ty "dev" extra and the
# typesafe-sdk "dev" dependency group, neither of which the server needs.
RUN uv sync --frozen --no-dev

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jeff"]
