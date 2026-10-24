FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 modelforge \
    && useradd --uid 1000 --gid modelforge --create-home --shell /usr/sbin/nologin modelforge

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY backend/ ./backend/

RUN chown -R modelforge:modelforge /app

USER modelforge

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "backend.app.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
