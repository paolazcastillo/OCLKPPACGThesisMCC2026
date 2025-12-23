#!/usr/bin/env python3
"""
Unit tests for Knapsack Heuristics
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from knapsack_heuristics import (
    KnapsackInstance,
    HeuristicResult,
    InstanceResult,
    OptimalSolver,
    MaxProfitHeuristic,
    MaxProfitPerWeightHeuristic,
    MinWeightHeuristic,
    HeuristicRegistry,
    ExperimentExecutor,
    ResultsConverter,
    StatisticalAnalyzer
)
from knapsack_heuristics.core import _greedy_selection, _compute_ratios


class TestKnapsackInstance(unittest.TestCase):
    def test_valid_instance_creation(self):
        instance = KnapsackInstance(
            profits=np.array([10, 20, 30], dtype=np.int64),
            weights=np.array([5, 10, 15], dtype=np.int64),
            capacity=20,
            name='test'
        )
        self.assertEqual(instance.num_items, 3)
        self.assertEqual(instance.capacity, 20)

    def test_negative_capacity_raises_error(self):
        with self.assertRaises(ValidationError):
            KnapsackInstance(
                profits=np.array([10, 20], dtype=np.int64),
                weights=np.array([5, 10], dtype=np.int64),
                capacity=-5,
                name='invalid'
            )

    def test_mismatched_arrays_raises_error(self):
        with self.assertRaises(ValidationError):
            KnapsackInstance(
                profits=np.array([10, 20], dtype=np.int64),
                weights=np.array([5], dtype=np.int64),
                capacity=20,
                name='invalid'
            )

    def test_no_items_fit_raises_error(self):
        with self.assertRaises(ValidationError):
            KnapsackInstance(
                profits=np.array([10, 20], dtype=np.int64),
                weights=np.array([100, 200], dtype=np.int64),
                capacity=50,
                name='invalid'
            )

    def test_from_csv_with_valid_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('profit,weight\n10,5\n20,10\n30,15\n')
            temp_path = f.name

        try:
            instance = KnapsackInstance.from_csv(temp_path, capacity=20)
            self.assertEqual(instance.num_items, 3)
            self.assertTrue(np.array_equal(instance.profits, np.array([10, 20, 30])))
        finally:
            os.unlink(temp_path)


class TestHeuristicResult(unittest.TestCase):
    def test_valid_result(self):
        result = HeuristicResult(
            heuristic_name='test',
            solution_value=100,
            execution_time=0.001
        )
        self.assertEqual(result.solution_value, 100)

    def test_negative_solution_value_raises_error(self):
        with self.assertRaises(ValidationError):
            HeuristicResult(
                heuristic_name='test',
                solution_value=-10,
                execution_time=0.001
            )


class TestInstanceResult(unittest.TestCase):
    def test_valid_result(self):
        result = InstanceResult(
            instance_name='test',
            capacity=50,
            heuristic_results={
                'h1': HeuristicResult(heuristic_name='h1', solution_value=40, execution_time=0.001)
            },
            optimal_value=50
        )
        self.assertEqual(result.optimal_value, 50)

    def test_heuristic_exceeds_optimal_raises_error(self):
        with self.assertRaises(ValidationError):
            InstanceResult(
                instance_name='test',
                capacity=50,
                heuristic_results={
                    'buggy': HeuristicResult(heuristic_name='buggy', solution_value=60, execution_time=0.001)
                },
                optimal_value=50
            )


class TestOptimalSolver(unittest.TestCase):
    def setUp(self):
        self.solver = OptimalSolver()

    def test_known_optimal_solution(self):
        instance = KnapsackInstance(
            profits=np.array([60, 100, 120], dtype=np.int64),
            weights=np.array([10, 20, 30], dtype=np.int64),
            capacity=50,
            name='test'
        )
        result = self.solver.solve(instance)
        self.assertEqual(result, 220)

    def test_all_items_fit(self):
        instance = KnapsackInstance(
            profits=np.array([10, 20, 30], dtype=np.int64),
            weights=np.array([5, 10, 15], dtype=np.int64),
            capacity=100,
            name='test'
        )
        result = self.solver.solve(instance)
        self.assertEqual(result, 60)


class TestHeuristics(unittest.TestCase):
    def setUp(self):
        self.instance = KnapsackInstance(
            profits=np.array([60, 100, 120], dtype=np.int64),
            weights=np.array([10, 20, 30], dtype=np.int64),
            capacity=50,
            name='test'
        )
        self.optimal_value = 220

    def test_all_heuristics_deterministic(self):
        heuristics = [
            MaxProfitHeuristic(),
            MaxProfitPerWeightHeuristic(),
            MinWeightHeuristic()
        ]

        for heuristic in heuristics:
            result1 = heuristic.solve(self.instance)
            result2 = heuristic.solve(self.instance)
            self.assertEqual(result1, result2,
                           f"{heuristic.name} is not deterministic")

    def test_heuristics_do_not_exceed_optimal(self):
        heuristics = [
            MaxProfitHeuristic(),
            MaxProfitPerWeightHeuristic(),
            MinWeightHeuristic()
        ]

        for heuristic in heuristics:
            result = heuristic.solve(self.instance)
            self.assertLessEqual(result, self.optimal_value,
                f"{heuristic.name} exceeded optimal value")


class TestHeuristicRegistry(unittest.TestCase):
    def test_register_and_retrieve(self):
        registry = HeuristicRegistry()
        self.assertIn('max_profit', registry.get_names())
        heuristic = registry.get('max_profit')
        self.assertIsInstance(heuristic, MaxProfitHeuristic)


class TestStatisticalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'instance': ['A', 'B', 'C'],
            'heur1': [10, 20, 30],
            'heur2': [15, 18, 25],
            'optimal': [20, 25, 35]
        })

    def test_compute_statistics(self):
        analyzer = StatisticalAnalyzer()
        stats = analyzer.compute_statistics(self.df, ['heur1', 'heur2'])
        self.assertEqual(len(stats), 2)
        self.assertIn('mean', stats.columns)

    def test_compute_gap_statistics(self):
        analyzer = StatisticalAnalyzer()
        gaps = analyzer.compute_gap_statistics(self.df, ['heur1', 'heur2'])
        self.assertEqual(len(gaps), 2)
        self.assertIn('mean_gap', gaps.columns)


class TestNumbaFunctions(unittest.TestCase):
    def test_greedy_selection_respects_capacity(self):
        profits = np.array([100] * 10, dtype=np.int64)
        weights = np.array([20] * 10, dtype=np.int64)
        capacity = np.int64(50)
        indices = np.arange(10, dtype=np.int64)

        result = _greedy_selection(profits, weights, capacity, indices)
        self.assertEqual(result, 200)

    def test_compute_ratios(self):
        profits = np.array([10, 20, 30], dtype=np.int64)
        weights = np.array([5, 10, 15], dtype=np.int64)

        ratios = _compute_ratios(profits, weights)
        self.assertEqual(len(ratios), 3)
        self.assertAlmostEqual(ratios[0], 2.0)


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
