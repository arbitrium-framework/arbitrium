.PHONY: help fmt lint test bench bench-quick bench-ablation docs clean install dev build publish

PY_SOURCES = src/ benchmarks/ tests/

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

fmt:  ## Format code with black and isort
	@echo "Formatting code..."
	black $(PY_SOURCES)
	isort $(PY_SOURCES)

lint:  ## Run all linters
	@echo "Running linters..."
	ruff check $(PY_SOURCES)
	black --check $(PY_SOURCES)
	isort --check $(PY_SOURCES)
	mypy src/

test:  ## Run tests with coverage
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src/arbitrium --cov-report=term-missing --cov-branch

test-quick:  ## Run tests without coverage
	python -m pytest tests/ -v

test-coverage-enforce:  ## Run tests with strict coverage requirements for core modules
	@echo "Running tests with strict coverage enforcement..."
	python -m pytest tests/ -v \
		--cov=src/arbitrium/core/tournament.py \
		--cov=src/arbitrium/models/base.py \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=60
	@echo "✓ Core modules meet 60% coverage requirement"

bench:  ## Run standard benchmarks (BBH + GPQA)
	@echo "Running standard benchmarks..."
	python -m benchmarks.standard_benchmarks --benchmark both --config config.example.yml

bench-quick:  ## Run quick validation benchmark
	@echo "Running quick validation..."
	python -m benchmarks.micro_benchmark --config config.example.yml

bench-ablation:  ## Run ablation benchmark to validate core hypotheses
	@echo "Running ablation benchmark..."
	python -m benchmarks.ablation_benchmark --config config.example.yml

check-mkdocs: ## Check if mkdocs is installed
	@command -v mkdocs >/dev/null 2>&1 || { echo "MkDocs not installed. Run: pip install mkdocs-material"; exit 1; }

docs: check-mkdocs ## Build documentation with MkDocs
	mkdocs build

docs-serve: check-mkdocs ## Serve documentation locally
	mkdocs serve

clean:  ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/ site/

install:  ## Install package in editable mode
	pip install -e .

dev:  ## Install package with dev dependencies
	pip install -e .[dev]
	pre-commit install

build:  ## Build distribution packages
	@echo "Building distribution packages..."
	python -m build

publish-test:  ## Publish to TestPyPI
	@echo "Publishing to TestPyPI..."
	@command -v twine >/dev/null 2>&1 || { echo "Twine not installed. Run: pip install twine"; exit 1; }
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI (use with caution!)
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to PyPI? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
	else \
		echo "Cancelled."; \
	fi

.DEFAULT_GOAL := help
