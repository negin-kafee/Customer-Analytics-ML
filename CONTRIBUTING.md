# Contributing to Customer Analytics ML

Thank you for your interest in contributing to the Customer Analytics ML project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Customer-Analytics-ML.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate the environment: `source .venv/bin/activate`
5. Install dependencies: `pip install -e .`
6. Install development dependencies: `pip install pytest pytest-cov black isort flake8 mypy`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `make test`
4. Run linting: `make lint`
5. Format code: `make format`
6. Commit your changes with a descriptive message
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for functions and classes

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage (>80%)
- Use pytest for testing framework

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update type hints and comments

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

Thank you for contributing!