#!/usr/bin/env python3
"""
Knapsack Heuristics Experiment - Main Execution Script

This script runs the complete knapsack heuristics experiment pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytz
import pandas as pd

from knapsack_heuristics import (
    ExperimentConfig,
    HeuristicRegistry,
    OptimalSolver,
    ExperimentExecutor,
    ResultsConverter,
    StatisticalAnalyzer,
    ReportFormatter,
    ReportGenerator
)
from knapsack_heuristics.utils import save_results


def main():
    print("=" * 80)
    print("KNAPSACK HEURISTICS EXPERIMENT")
    print("=" * 80)
    
    config = ExperimentConfig()
    config.setup_logging()
    
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key:<25}: {value}")
    print("=" * 80)
    
    csv_dir = config.data_dir
    CAPACITY = config.capacity
    
    registry = HeuristicRegistry()
    optimal_solver = OptimalSolver()
    executor = ExperimentExecutor(registry, optimal_solver)
    
    print(f"\nInitialized with {len(registry.get_names())} heuristics:")
    for i, name in enumerate(registry.get_names(), 1):
        print(f"  {i}. {name}")
    
    print(f"\nProcessing instances from: {csv_dir}")
    print(f"Parallel processing: {config.enable_parallel}")
    if config.enable_parallel:
        print(f"Max workers: {config.max_workers or 'auto'}")
    
    results = executor.execute_directory(
        directory=csv_dir,
        capacity=CAPACITY,
        pattern=config.pattern,
        compute_optimal=config.compute_optimal,
        parallel=config.enable_parallel,
        max_workers=config.max_workers
    )
    
    print(f"\n✓ Successfully processed {len(results)} instances")
    
    converter = ResultsConverter()
    df = converter.to_dataframe(results)
    execution_times = converter.extract_execution_times(results)
    
    print("\nDataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    output_file = save_results(
        df=df,
        output_path=Path(config.output_dir) / 'knapsack_results.csv',
        include_timestamp=True,
        base_name=f'heuristic_results_cap{CAPACITY}'
    )
    
    print(f"\n✓ Results saved to: {output_file}")
    
    timestamp = datetime.now(pytz.timezone(config.timezone)).strftime('%Y%m%d_%H%M%S')
    report_path = Path(config.output_dir) / f'experiment_report_CAP{CAPACITY}_{timestamp}.txt'
    config_path = Path(config.output_dir) / f'experiment_config_{timestamp}.json'
    
    config.save_to_file(str(config_path))
    
    formatter = ReportFormatter()
    analyzer = StatisticalAnalyzer()
    report_gen = ReportGenerator(formatter, analyzer)
    
    report_gen.generate(
        df=df,
        execution_times=execution_times,
        config=config.to_dict(),
        output_path=str(report_path)
    )
    
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Configuration saved to: {config_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    heuristic_columns = [col for col in df.columns if col not in ['instance', 'capacity', 'optimal']]
    stats_df = analyzer.compute_statistics(df, heuristic_columns)
    
    print("\nHeuristic Performance Statistics:")
    print(stats_df[['heuristic', 'mean', 'std']].to_string(index=False))
    
    if 'optimal' in df.columns:
        gap_stats = analyzer.compute_gap_statistics(df, heuristic_columns)
        print("\nOptimality Gap Analysis:")
        print(gap_stats[['heuristic', 'mean_gap', 'optimal_found']].to_string(index=False))
    
    dominant = analyzer.identify_dominant_heuristic(df, heuristic_columns)
    print("\nDominant Heuristic (times achieved best solution):")
    print(dominant.to_string())
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
