from pathlib import Path
from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class KnapsackInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profits: np.ndarray
    weights: np.ndarray
    capacity: int = Field(gt=0, description="Knapsack capacity must be positive")
    name: str = Field(min_length=1, description="Instance name")

    @field_validator('profits', 'weights')
    @classmethod
    def validate_arrays(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.dtype != np.int64:
            v = v.astype(np.int64)
        if len(v) == 0:
            raise ValueError("Array cannot be empty")
        if np.any(v <= 0):
            raise ValueError(f"All values must be positive, found {np.sum(v <= 0)} non-positive values")
        return v

    @model_validator(mode='after')
    def validate_arrays_match(self):
        if len(self.profits) != len(self.weights):
            raise ValueError(
                f"Mismatched array lengths: profits={len(self.profits)}, weights={len(self.weights)}"
            )

        items_that_fit = np.sum(self.weights <= self.capacity)
        if items_that_fit == 0:
            raise ValueError(
                f"No items can fit in capacity {self.capacity}: "
                f"minimum weight is {np.min(self.weights)}, all {len(self.weights)} items exceed capacity"
            )

        max_sum = np.sum(self.profits)
        if max_sum > np.iinfo(np.int64).max // 2:
            raise ValueError(
                f"Risk of overflow: sum of profits {max_sum} exceeds safe threshold "
                f"{np.iinfo(np.int64).max // 2}"
            )
        return self

    @property
    def num_items(self) -> int:
        return len(self.profits)

    @classmethod
    def from_csv(cls, filepath: str, capacity: int) -> 'KnapsackInstance':
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        required_columns = {'profit', 'weight'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}, found: {df.columns.tolist()}")

        return cls(
            profits=df['profit'].values.astype(np.int64),
            weights=df['weight'].values.astype(np.int64),
            capacity=capacity,
            name=Path(filepath).stem
        )


class HeuristicResult(BaseModel):
    heuristic_name: str = Field(min_length=1)
    solution_value: int = Field(ge=0, description="Solution value must be non-negative")
    execution_time: float = Field(ge=0, description="Execution time must be non-negative")

    @field_validator('execution_time')
    @classmethod
    def validate_execution_time(cls, v: float) -> float:
        if v > 3600:
            logger.warning(f"Execution time {v}s exceeds 1 hour, possible timing error")
        return v


class InstanceResult(BaseModel):
    instance_name: str = Field(min_length=1)
    capacity: int = Field(gt=0)
    heuristic_results: Dict[str, HeuristicResult]
    optimal_value: Optional[int] = Field(default=None, ge=0)
    optimal_time: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode='after')
    def validate_results(self):
        if self.optimal_value is not None:
            for name, result in self.heuristic_results.items():
                if result.solution_value > self.optimal_value:
                    raise ValueError(
                        f"CRITICAL BUG DETECTED: Heuristic {name} returned {result.solution_value} > "
                        f"optimal {self.optimal_value}. This indicates a bug in the implementation."
                    )
        return self

    def to_dict(self) -> dict:
        result = {
            'instance': self.instance_name,
            'capacity': self.capacity
        }
        for name, heur_result in self.heuristic_results.items():
            result[name] = heur_result.solution_value
        if self.optimal_value is not None:
            result['optimal'] = self.optimal_value
        return result

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'InstanceResult':
        return cls.model_validate_json(json_str)
