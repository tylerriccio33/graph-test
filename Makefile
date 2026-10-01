.PHONY: install fmt lint typecheck test check clean

install:
	uv sync

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run pyrefly check

test:
	uv run pytest -q

check: lint typecheck test

clean:
	rm -rf .venv .pytest_cache .ruff_cache dist build .graph-test
