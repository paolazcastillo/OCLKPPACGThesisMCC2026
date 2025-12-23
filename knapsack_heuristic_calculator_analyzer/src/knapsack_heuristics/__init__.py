"""
Knapsack Heuristics Calculator and Analyzer

A comprehensive system for evaluating greedy heuristics on 0/1 Knapsack Problem instances.

License: CC BY-NC-SA 4.0
"""

__version__ = "1.0.0"
__author__ = "Paola A. Castillo-Gutiérrez"
__license__ = "CC BY-NC-SA 4.0"

from .config import ExperimentConfig
from .models import KnapsackInstance, HeuristicResult, InstanceResult
from .heuristics import (
    Heuristic,
    DefaultHeuristic,
    MaxProfitHeuristic,
    MaxProfitPerWeightHeuristic,
    MinWeightHeuristic,
    MaxProfitPerWeightTiebreakProfitHeuristic,
    MaxProfitPerWeightTiebreakWeightHeuristic,
    MaxProfitTiebreakWeightHeuristic,
    MinWeightTiebreakProfitHeuristic,
    HeuristicRegistry
)
from .solver import OptimalSolver
from .executor import ExperimentExecutor
from .analyzer import ResultsConverter, StatisticalAnalyzer
from .reporter import ReportFormatter, ReportGenerator

__all__ = [
    'ExperimentConfig',
    'KnapsackInstance',
    'HeuristicResult',
    'InstanceResult',
    'Heuristic',
    'DefaultHeuristic',
    'MaxProfitHeuristic',
    'MaxProfitPerWeightHeuristic',
    'MinWeightHeuristic',
    'MaxProfitPerWeightTiebreakProfitHeuristic',
    'MaxProfitPerWeightTiebreakWeightHeuristic',
    'MaxProfitTiebreakWeightHeuristic',
    'MinWeightTiebreakProfitHeuristic',
    'HeuristicRegistry',
    'OptimalSolver',
    'ExperimentExecutor',
    'ResultsConverter',
    'StatisticalAnalyzer',
    'ReportFormatter',
    'ReportGenerator'
]
