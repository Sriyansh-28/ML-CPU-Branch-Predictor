# Convenience workflows for the ML-CPU-Branch-Predictor project.
# Run `make help` to list available targets.

PYTHON ?= python
PIP    ?= pip

.DEFAULT_GOAL := help

.PHONY: help setup data train eval compare test lint format clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install runtime + dev dependencies
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

data:  ## Generate (or download) a branch trace into data/raw
	$(PYTHON) scripts/generate_data.py --config configs/default.yaml

train:  ## Train the perceptron predictor
	$(PYTHON) scripts/train.py --config configs/default.yaml

eval:  ## Evaluate a single predictor
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml

compare:  ## Run the full 2-bit vs gshare vs perceptron comparison
	$(PYTHON) scripts/run_comparison.py --config configs/default.yaml

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint the codebase with ruff
	ruff check src tests scripts

format:  ## Auto-format with black
	black src tests scripts

clean:  ## Remove caches and generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
