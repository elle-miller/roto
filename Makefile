# Makefile for auto-formatting and linting
# Run 'make format' to apply all formatting (black and isort)
# Run 'make lint' to check for typing issues with mypy

# Formatting code with black and isort
format:
	black .
	isort .

# Linting code with mypy
lint:
	mypy .
	flake8 . 

flake8:
	flake8 .

# Running both format and lint in one go
all: format lint