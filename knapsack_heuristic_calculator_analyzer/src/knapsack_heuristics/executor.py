import time
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import psutil

from .models import KnapsackInstance, HeuristicResult, InstanceResult
from .heuristics import HeuristicRegistry
from .solver import OptimalSolver

logger = logging.getLogger(__name__)


def _process_instance_wrapper(filepath: str, capacity: int, compute_optimal: bool,
                              registry: HeuristicRegistry, solver: OptimalSolver) -> Optional[InstanceResult]:
    try:
        instance = KnapsackInstance.from_csv(filepath, capacity)

        heuristic_results = {}
        for name, heuristic in registry.get_all().items():
            start_time = time.perf_counter()
            solution_value = heuristic.solve(instance)
            execution_time = time.perf_counter() - start_time

            heuristic_results[name] = HeuristicResult(
                heuristic_name=name,
                solution_value=solution_value,
                execution_time=execution_time
            )

        optimal_value = None
        optimal_time = None
        if compute_optimal:
            optimal_value, optimal_time = solver.solve_with_time(instance)

        return InstanceResult(
            instance_name=instance.name,
            capacity=instance.capacity,
            heuristic_results=heuristic_results,
            optimal_value=optimal_value,
            optimal_time=optimal_time
        )
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return None


class ExperimentExecutor:
    def __init__(self, registry: HeuristicRegistry, optimal_solver: OptimalSolver):
        self.registry = registry
        self.optimal_solver = optimal_solver
        self.logger = logging.getLogger(__name__ + '.ExperimentExecutor')

    def execute_single_instance(self, instance: KnapsackInstance,
                                compute_optimal: bool = False) -> InstanceResult:
        heuristic_results = {}

        for name, heuristic in self.registry.get_all().items():
            start_time = time.perf_counter()
            solution_value = heuristic.solve(instance)
            execution_time = time.perf_counter() - start_time

            heuristic_results[name] = HeuristicResult(
                heuristic_name=name,
                solution_value=solution_value,
                execution_time=execution_time
            )

        optimal_value = None
        optimal_time = None
        if compute_optimal:
            optimal_value, optimal_time = self.optimal_solver.solve_with_time(instance)

        return InstanceResult(
            instance_name=instance.name,
            capacity=instance.capacity,
            heuristic_results=heuristic_results,
            optimal_value=optimal_value,
            optimal_time=optimal_time
        )

    def execute_directory(self, directory: str, capacity: int,
                         pattern: str = '*.csv',
                         compute_optimal: bool = False,
                         parallel: bool = False,
                         max_workers: Optional[int] = None) -> List[InstanceResult]:
        directory_path = Path(directory)
        instance_files = sorted([
            f for f in directory_path.glob(pattern)
            if f.name.startswith(('TRAIN', 'TEST'))
        ])

        total = len(instance_files)
        self.logger.info(f"Found {total} instances in {directory}")

        if parallel:
            return self._execute_parallel(instance_files, capacity, compute_optimal, max_workers)
        else:
            return self._execute_sequential(instance_files, capacity, compute_optimal)

    def _execute_sequential(self, instance_files: List[Path], capacity: int,
                           compute_optimal: bool) -> List[InstanceResult]:
        results = []
        total = len(instance_files)

        for idx, filepath in enumerate(instance_files, 1):
            try:
                instance = KnapsackInstance.from_csv(str(filepath), capacity)
                result = self.execute_single_instance(instance, compute_optimal)
                results.append(result)

                if idx % 10 == 0:
                    self.logger.info(f"Processed {idx}/{total}")
                    print(f"Processed {idx}/{total}")

            except Exception as e:
                self.logger.error(f"Failed to process {filepath}: {e}")

        self.logger.info(f"Completed: {len(results)}/{total} instances")
        return results

    def _execute_parallel(self, instance_files: List[Path], capacity: int,
                         compute_optimal: bool, max_workers: Optional[int] = None) -> List[InstanceResult]:
        total = len(instance_files)

        if max_workers is None:
            max_workers = min(32, (psutil.cpu_count(logical=True) or 1) + 4)

        self.logger.info(f"Starting parallel execution with {max_workers} workers")
        print(f"Processing {total} instances in parallel ({max_workers} workers)...")

        results = []
        completed = 0

        process_func = partial(
            _process_instance_wrapper,
            capacity=capacity,
            compute_optimal=compute_optimal,
            registry=self.registry,
            solver=self.optimal_solver
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_filepath = {
                executor.submit(process_func, str(filepath)): filepath
                for filepath in instance_files
            }

            for future in as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        self.logger.info(f"Completed {completed}/{total}")
                        print(f"Completed {completed}/{total}")

                except Exception as e:
                    self.logger.error(f"Exception processing {filepath}: {e}")
                    completed += 1

        results.sort(key=lambda r: r.instance_name)

        self.logger.info(f"Parallel execution completed: {len(results)}/{total} instances")
        return results
