# Makefile for auto-formatting and linting
# Run 'make format' to apply all formatting (autoflake, black, isort)
# Run 'make lint' to check for typing issues with mypy

# Formatting code with autoflake, black, and isort
format:
	autoflake --remove-all-unused-imports --in-place --recursive .
	black .
	isort .

# Linting code with mypy and flake8
lint:
	mypy .
	flake8 . 

flake8:
	flake8 .

# Running both format and lint in one go
all: format lint
