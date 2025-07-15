.PHONY: help install install-dev test test-cov lint format type-check clean build upload docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install the package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload       Upload to PyPI"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,examples]"
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=yolocam --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/yolocam tests
	black --check src/yolocam tests
	isort --check-only src/yolocam tests

format:
	black src/yolocam tests
	isort src/yolocam tests

type-check:
	mypy src/yolocam

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Development workflow
check: format lint type-check test

# CI/CD simulation
ci: install-dev check test-cov