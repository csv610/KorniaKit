.PHONY: install test lint typecheck clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make test       - Run pytest"
	@echo "  make clean      - Remove cache and compiled files"

install:
	pip install -r requirements.txt
	pip install ruff mypy pytest

lint:
	ruff check kornia/ tests/

typecheck:
	mypy kornia/ tests/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
