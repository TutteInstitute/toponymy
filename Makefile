.PHONY: help clean fresh install test test-all lint format

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## Remove virtual environment and build artifacts
	rm -rf .venv
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

fresh: clean ## Create a fresh virtual environment and install dependencies
	uv venv
	@echo ""
	@echo "✓ Fresh environment created!"
	@echo "Now run: source .venv/bin/activate && make install"

install: ## Install package in editable mode with dev dependencies (excludes llama-cpp-python)
	uv pip install -e ".[dev]"
	@echo ""
	@echo "✓ Installation complete!"

install-full: ## Install package with ALL dev dependencies including llama-cpp-python
	uv pip install -e ".[dev-full]"
	@echo ""
	@echo "✓ Full installation complete!"

fresh-test: clean ## Full fresh environment setup and test (one command)
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"
	. .venv/bin/activate && pytest
	@echo ""
	@echo "✓ Fresh environment tested successfully!"

fresh-test-full: clean ## Fresh environment with ALL dependencies including llama-cpp-python
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev-full]"
	. .venv/bin/activate && pytest
	@echo ""
	@echo "✓ Fresh environment with full dependencies tested successfully!"

test: ## Run tests (excluding canary tests)
	pytest

test-all: ## Run all tests including canary tests
	pytest -m ""

test-verbose: ## Run tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage report
	pytest --cov=toponymy --cov-report=html --cov-report=term

lint: ## Run linting checks
	pylint toponymy

format: ## Format code with black and isort
	black toponymy
	isort toponymy

sync: ## Sync dependencies with uv.lock
	uv pip sync uv.lock
