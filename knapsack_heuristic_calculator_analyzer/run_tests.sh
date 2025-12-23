#!/bin/bash
# Script to run all tests

echo "======================================"
echo "Running Knapsack Heuristics Test Suite"
echo "======================================"

cd "$(dirname "$0")"

python3 tests/test_all.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "All tests passed!"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "Some tests failed!"
    echo "======================================"
fi

exit $exit_code
