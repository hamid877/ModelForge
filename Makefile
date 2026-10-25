.PHONY: help bootstrap test lint format up down

help:
	@echo "ModelForge developer commands:"
	@echo "  make bootstrap  - Train, register, and assign initial champion model in MLflow"
	@echo "  make test       - Run all test suites"
	@echo "  make lint       - Run Ruff linter"
	@echo "  make format     - Run Ruff format check"
	@echo "  make up         - Start Docker Compose stack"
	@echo "  make down       - Stop Docker Compose stack"

bootstrap:
	uv run python scripts/bootstrap.py

test:
	uv run pytest tests/ -q

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

up:
	docker compose up -d

down:
	docker compose down
