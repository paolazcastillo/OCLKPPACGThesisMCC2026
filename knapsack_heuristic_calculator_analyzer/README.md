# Knapsack Heuristics Calculator and Analyzer

A comprehensive system for evaluating greedy heuristics on 0/1 Knapsack Problem instances.

## Features

- **8 Greedy Heuristics** with tie-breaking mechanisms
- **Dynamic Programming Optimal Solver** for comparison
- **Parallel Processing** support for large datasets
- **Comprehensive Statistical Analysis** with gap analysis
- **Automated Report Generation** with detailed metrics
- **Pydantic Validation** for data integrity
- **Numba JIT Compilation** for performance
- **Complete Unit Test Suite**

## Installation

### From Source

```bash
# Clone or download the project
cd knapsack_project
pip install -r requirements.txt
pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Experiment

```bash
# Run with default configuration
python main.py

# Or with environment variables
export KP_CAPACITY=64
export KP_DATA_DIR="/path/to/your/data"
export KP_OUTPUT_DIR="/path/to/your/output"
python main.py
```

### Running Tests

```bash
# Run all tests
python tests/test_all.py

# Or using pytest
pytest tests/ -v
```

## Project Structure

```
knapsack_heuristic_calculator_analyzer/
├── src/
│   └── knapsack_heuristics/
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration management
│       ├── models.py            # Data models (Pydantic)
│       ├── core.py              # Numba-optimized functions
│       ├── heuristics.py        # Heuristic implementations
│       ├── solver.py            # Optimal DP solver
│       ├── executor.py          # Experiment executor
│       ├── analyzer.py          # Statistical analyzer
│       ├── reporter.py          # Report generator
│       └── utils.py             # Utility functions
├── tests/
│   └── test_all.py              # Complete test suite
├── main.py                      # Main execution script
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Configuration

The system uses Pydantic Settings for configuration. You can configure via:

1. **Environment variables** (prefixed with `KP_`):
```bash
export KP_CAPACITY=64
export KP_DATA_DIR="data"
export KP_ENABLE_PARALLEL=True
```

2. **Direct instantiation**:
```python
from knapsack_heuristics import ExperimentConfig

config = ExperimentConfig(
    capacity=64,
    data_dir="data",
    enable_parallel=True
)
```

### Configuration Options

- `capacity`: Knapsack capacity (default: 64)
- `data_dir`: Directory with instance CSV files (default: "data")
- `output_dir`: Output directory (default: "output")
- `pattern`: File pattern for instances (default: "*.csv")
- `compute_optimal`: Compute optimal solutions (default: True)
- `enable_parallel`: Enable parallel processing (default: True)
- `max_workers`: Max parallel workers (default: auto-detect)
- `log_level`: Logging level (default: "INFO")

## Heuristics Implemented

1. **default**: Original order
2. **max_profit**: Maximum profit first
3. **max_profit_per_weight**: Maximum efficiency (p/w ratio)
4. **min_weight**: Minimum weight first
5. **max_profit_per_weight_tiebreak_profit**: Efficiency with profit tiebreak
6. **max_profit_per_weight_tiebreak_weight**: Efficiency with weight tiebreak
7. **max_profit_tiebreak_weight**: Profit with weight tiebreak
8. **min_weight_tiebreak_profit**: Weight with profit tiebreak

## Input Format

CSV files with two columns:
```csv
profit,weight
60,10
100,20
120,30
```

Files should be named with `TRAIN_` or `TEST_` prefix to be processed.

## Output

The system generates:

1. **Results CSV**: Heuristic solutions for each instance
2. **Experiment Report**: Comprehensive analysis (.txt)
3. **Configuration File**: Experiment parameters (.json)

## Usage Examples

### Basic Usage

```python
from knapsack_heuristics import (
    ExperimentConfig,
    HeuristicRegistry,
    OptimalSolver,
    ExperimentExecutor
)

# Initialize
config = ExperimentConfig(
    data_dir="/path/to/your/data",
    output_dir="/path/to/your/output"
)
registry = HeuristicRegistry()
solver = OptimalSolver()
executor = ExperimentExecutor(registry, solver)

# Run experiment
results = executor.execute_directory(
    directory="/path/to/your/data",
    capacity=64,
    compute_optimal=True,
    parallel=True
)
```

### Single Instance

```python
from knapsack_heuristics import KnapsackInstance, MaxProfitHeuristic

# Load instance
instance = KnapsackInstance.from_csv("/path/to/instance.csv", capacity=50)

# Apply heuristic
heuristic = MaxProfitHeuristic()
solution = heuristic.solve(instance)
print(f"Solution value: {solution}")
```

## Performance

- **Numba JIT compilation** for core algorithms
- **Parallel processing** support for multiple instances
- **Efficient DP solver** with O(C) space complexity
- Processes 500 instances in ~30 seconds (sequential)
- Processes 500 instances in ~10 seconds (parallel, 8 cores)

## Testing

The test suite includes:
- Data model validation tests
- Heuristic correctness tests
- Optimal solver verification
- Statistical analyzer tests
- Integration tests

Run with:
```bash
python tests/test_all.py
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

## Author

**Paola A. Castillo-Gutiérrez**  
Master's Student in Computer Science  
Tecnológico de Monterrey

## Citation

If you use this code in your research, please cite:
```bibtex
@software{castillo2026knapsack,
  author = {Castillo-Gutiérrez, Paola A.},
  title = {Knapsack Heuristics Calculator and Analyzer},
  year = {2026},
  url = {https://github.com/paolazcastillo/OCLKPPACGThesisMCC2026}
}
```
