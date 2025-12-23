import time
from typing import Tuple

from .models import KnapsackInstance
from .core import _knapsack_dp


class OptimalSolver:
    def solve(self, instance: KnapsackInstance) -> int:
        return _knapsack_dp(instance.profits, instance.weights, instance.capacity)

    def solve_with_time(self, instance: KnapsackInstance) -> Tuple[int, float]:
        start_time = time.perf_counter()
        result = _knapsack_dp(instance.profits, instance.weights, instance.capacity)
        execution_time = time.perf_counter() - start_time
        return result, execution_time
