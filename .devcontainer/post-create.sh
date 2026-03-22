#!/usr/bin/env bash
set -e

echo "=== Installing GoldenMatch ==="
pip install -e ".[dev]"

echo "=== Verifying installation ==="
goldenmatch --help

echo "=== Running quick test ==="
pytest tests/test_config.py -q --tb=short

echo ""
echo "============================================"
echo "  GoldenMatch dev environment ready!"
echo ""
echo "  Quick start:"
echo "    goldenmatch demo          # Run built-in demo"
echo "    goldenmatch dedupe --help # See dedupe options"
echo "    pytest --tb=short         # Run test suite"
echo "============================================"
