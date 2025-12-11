.PHONY: install test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Remove cache and compiled files"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
