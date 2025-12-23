#!/bin/bash
# Script to run knapsack heuristics experiment

echo "======================================"
echo "Knapsack Heuristics Experiment Runner"
echo "======================================"

# Set environment variables (customize these)
export KP_CAPACITY=${KP_CAPACITY:-64}
export KP_DATA_DIR=${KP_DATA_DIR:-"data"}
export KP_OUTPUT_DIR=${KP_OUTPUT_DIR:-"output"}
export KP_ENABLE_PARALLEL=${KP_ENABLE_PARALLEL:-True}
export KP_LOG_LEVEL=${KP_LOG_LEVEL:-"INFO"}

echo ""
echo "Configuration:"
echo "  Capacity: $KP_CAPACITY"
echo "  Data directory: $KP_DATA_DIR"
echo "  Output directory: $KP_OUTPUT_DIR"
echo "  Parallel processing: $KP_ENABLE_PARALLEL"
echo "======================================"
echo ""

# Run the experiment
python3 main.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Experiment completed successfully!"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "Experiment failed with exit code: $exit_code"
    echo "======================================"
fi

exit $exit_code
