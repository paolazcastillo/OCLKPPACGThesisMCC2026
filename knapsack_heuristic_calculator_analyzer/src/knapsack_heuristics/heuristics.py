from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from .models import KnapsackInstance
from .core import _greedy_selection, _compute_ratios


class Heuristic(ABC):
    @abstractmethod
    def solve(self, instance: KnapsackInstance) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class DefaultHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'default'

    def solve(self, instance: KnapsackInstance) -> int:
        indices = np.arange(len(instance.profits), dtype=np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MaxProfitHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'max_profit'

    def solve(self, instance: KnapsackInstance) -> int:
        indices = np.argsort(instance.profits)[::-1].astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MaxProfitPerWeightHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'max_profit_per_weight'

    def solve(self, instance: KnapsackInstance) -> int:
        ratios = _compute_ratios(instance.profits, instance.weights)
        indices = np.argsort(ratios)[::-1].astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MinWeightHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'min_weight'

    def solve(self, instance: KnapsackInstance) -> int:
        indices = np.argsort(instance.weights).astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MaxProfitPerWeightTiebreakProfitHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'max_profit_per_weight_tiebreak_profit'

    def solve(self, instance: KnapsackInstance) -> int:
        ratios = _compute_ratios(instance.profits, instance.weights)
        indices = np.lexsort((instance.profits, ratios))[::-1].astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MaxProfitPerWeightTiebreakWeightHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'max_profit_per_weight_tiebreak_weight'

    def solve(self, instance: KnapsackInstance) -> int:
        ratios = _compute_ratios(instance.profits, instance.weights)
        indices = np.lexsort((instance.weights, -ratios)).astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MaxProfitTiebreakWeightHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'max_profit_tiebreak_weight'

    def solve(self, instance: KnapsackInstance) -> int:
        indices = np.lexsort((instance.weights, -instance.profits)).astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class MinWeightTiebreakProfitHeuristic(Heuristic):
    @property
    def name(self) -> str:
        return 'min_weight_tiebreak_profit'

    def solve(self, instance: KnapsackInstance) -> int:
        indices = np.lexsort((-instance.profits, instance.weights)).astype(np.int64)
        return _greedy_selection(instance.profits, instance.weights, instance.capacity, indices)


class HeuristicRegistry:
    def __init__(self):
        self._heuristics: Dict[str, Heuristic] = {}
        self._register_defaults()

    def _register_defaults(self):
        default_heuristics = [
            DefaultHeuristic(),
            MaxProfitHeuristic(),
            MaxProfitPerWeightHeuristic(),
            MinWeightHeuristic(),
            MaxProfitPerWeightTiebreakProfitHeuristic(),
            MaxProfitPerWeightTiebreakWeightHeuristic(),
            MaxProfitTiebreakWeightHeuristic(),
            MinWeightTiebreakProfitHeuristic()
        ]
        for heuristic in default_heuristics:
            self.register(heuristic)

    def register(self, heuristic: Heuristic):
        self._heuristics[heuristic.name] = heuristic

    def get(self, name: str) -> Heuristic:
        return self._heuristics[name]

    def get_all(self) -> Dict[str, Heuristic]:
        return self._heuristics.copy()

    def get_names(self) -> List[str]:
        return list(self._heuristics.keys())
