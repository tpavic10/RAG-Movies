FROM python:3.12.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-cache --no-dev

COPY ./app ./app

EXPOSE 8087

CMD ["uv", "run", "app/app.py"]