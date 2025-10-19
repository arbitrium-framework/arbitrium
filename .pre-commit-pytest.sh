#!/usr/bin/env bash
# Wrapper script for running pytest in pre-commit hook
# Handles both local development and CI environments
# Prioritizes Python module approach to use current environment

set -e

# Try Python module first (uses current environment's Python)
if python -m pytest --version &> /dev/null; then
    exec python -m pytest "$@"
elif python3 -m pytest --version &> /dev/null; then
    exec python3 -m pytest "$@"
elif command -v pytest &> /dev/null; then
    # Fallback to pytest in PATH (may be global installation)
    exec pytest "$@"
else
    echo "ERROR: pytest not found in current environment"
    echo "Please ensure you're in an activated virtual environment with test dependencies:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # or: venv\\Scripts\\activate on Windows"
    echo "  pip install -e .[dev]"
    exit 1
fi
