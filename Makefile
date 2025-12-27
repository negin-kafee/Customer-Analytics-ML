# Customer Analytics ML Pipeline - Makefile
# ==========================================
# Run `make help` to see available commands

.PHONY: help install install-dev test test-cov lint format clean run-notebooks setup check

# Default Python version
PYTHON := python3

# Virtual environment
VENV := .venv
VENV_BIN := $(VENV)/bin

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:  ## Show this help message
	@echo "$(BLUE)Customer Analytics ML Pipeline$(NC)"
	@echo "================================"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

install:  ## Install production dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest pytest-cov black isort flake8 mypy
	@echo "$(GREEN)✓ Development installation complete$(NC)"

setup:  ## Setup virtual environment and install all dependencies
	@echo "$(BLUE)Setting up virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install pytest pytest-cov black isort flake8
	@echo "$(GREEN)✓ Virtual environment created at $(VENV)$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV)/bin/activate$(NC)"

test:  ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-cov:  ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated at htmlcov/index.html$(NC)"

lint:  ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:  ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m isort src/ tests/
	$(PYTHON) -m black src/ tests/ --line-length=100
	@echo "$(GREEN)✓ Formatting complete$(NC)"

check:  ## Run all checks (lint + test)
	@make lint
	@make test
	@echo "$(GREEN)✓ All checks passed$(NC)"

clean:  ## Remove generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-models:  ## Remove generated model files
	@echo "$(BLUE)Cleaning model files...$(NC)"
	rm -f models/*.joblib models/*.keras models/*.h5
	@echo "$(GREEN)✓ Model files removed$(NC)"

run-eda:  ## Run EDA notebook (requires Jupyter)
	@echo "$(BLUE)Running EDA notebook...$(NC)"
	$(PYTHON) -m jupyter nbconvert --to notebook --execute 01_eda.ipynb --output 01_eda_executed.ipynb

validate-data:  ## Validate that data file exists and is readable
	@echo "$(BLUE)Validating data...$(NC)"
	@$(PYTHON) -c "from src.data_loader import load_data; df = load_data(); print(f'✓ Data loaded: {df.shape}')"

validate-config:  ## Validate configuration settings
	@echo "$(BLUE)Validating configuration...$(NC)"
	@$(PYTHON) -c "from src.config import validate_config; validate_config(); print('✓ Configuration valid')"
