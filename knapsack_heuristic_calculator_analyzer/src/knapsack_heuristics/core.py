import numpy as np
from numba import jit


@jit(nopython=True)
def _greedy_selection(profits: np.ndarray, weights: np.ndarray,
                      capacity: np.int64, indices: np.ndarray) -> np.int64:
    total_profit = 0
    total_weight = 0
    for i in indices:
        if total_weight + weights[i] <= capacity:
            total_weight += weights[i]
            total_profit += profits[i]
    return total_profit


@jit(nopython=True)
def _compute_ratios(profits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = len(profits)
    ratios = np.empty(n, dtype=np.float64)
    for i in range(n):
        if weights[i] > 0:
            ratios[i] = profits[i] / weights[i]
        else:
            ratios[i] = 1e308 if profits[i] > 0 else 0.0
    return ratios


@jit(nopython=True)
def _knapsack_dp(profits: np.ndarray, weights: np.ndarray, capacity: np.int64) -> np.int64:
    n: np.int64 = len(profits)
    prev = np.zeros(capacity + 1, dtype=np.int64)
    curr = np.zeros(capacity + 1, dtype=np.int64)

    for i in range(n):
        for w in range(capacity + 1):
            if weights[i] <= w:
                curr[w] = max(prev[w], prev[w - weights[i]] + profits[i])
            else:
                curr[w] = prev[w]
        prev, curr = curr, prev

    return prev[capacity]
