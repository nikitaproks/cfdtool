all: test lint

test:
	poetry run pytest


lint:
	poetry run mypy cfdtool/**/*.py
	poetry run ruff check cfdtool/