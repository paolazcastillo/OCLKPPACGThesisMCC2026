from typing import Dict, List
import numpy as np
import pandas as pd

from .models import InstanceResult


class ResultsConverter:
    @staticmethod
    def to_dataframe(results: List[InstanceResult]) -> pd.DataFrame:
        records = [result.to_dict() for result in results]
        return pd.DataFrame(records)

    @staticmethod
    def extract_execution_times(results: List[InstanceResult]) -> Dict[str, List[float]]:
        execution_times = {}

        for result in results:
            for name, heur_result in result.heuristic_results.items():
                if name not in execution_times:
                    execution_times[name] = []
                execution_times[name].append(heur_result.execution_time)

            if result.optimal_time is not None:
                if 'optimal' not in execution_times:
                    execution_times['optimal'] = []
                execution_times['optimal'].append(result.optimal_time)

        return execution_times


class StatisticalAnalyzer:
    BASIC_HEURISTICS = ['default', 'max_profit', 'max_profit_per_weight', 'min_weight']
    TIEBREAK_HEURISTICS = ['default', 'max_profit_tiebreak_weight', 
                          'min_weight_tiebreak_profit', 'max_profit_per_weight']

    @staticmethod
    def compute_statistics(df: pd.DataFrame, heuristic_columns: List[str]) -> pd.DataFrame:
        stats = []
        for col in heuristic_columns:
            values = df[col].dropna()
            stats.append({
                'heuristic': col,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            })
        return pd.DataFrame(stats)

    @staticmethod
    def compute_gap_statistics(df: pd.DataFrame, heuristic_columns: List[str],
                              optimal_column: str = 'optimal') -> pd.DataFrame:
        if optimal_column not in df.columns:
            return pd.DataFrame()

        gap_stats = []
        for col in heuristic_columns:
            gaps = ((df[optimal_column] - df[col]) / df[optimal_column] * 100)
            gap_stats.append({
                'heuristic': col,
                'mean_gap': gaps.mean(),
                'std_gap': gaps.std(),
                'min_gap': gaps.min(),
                'max_gap': gaps.max(),
                'median_gap': gaps.median(),
                'optimal_found': (gaps == 0).sum()
            })
        return pd.DataFrame(gap_stats)

    @staticmethod
    def identify_dominant_heuristic(df: pd.DataFrame,
                                   heuristic_columns: List[str]) -> pd.Series:
        best_heuristics = df[heuristic_columns].idxmax(axis=1)
        return best_heuristics.value_counts()

    @staticmethod
    def identify_dominant_basic_heuristics(df: pd.DataFrame) -> pd.Series:
        available_basic = [col for col in StatisticalAnalyzer.BASIC_HEURISTICS
                          if col in df.columns]
        if not available_basic:
            return pd.Series(dtype='int64')
        best_heuristics = df[available_basic].idxmax(axis=1)
        return best_heuristics.value_counts()

    @staticmethod
    def identify_dominant_tiebreak_heuristics(df: pd.DataFrame) -> pd.Series:
        available_tiebreak = [col for col in StatisticalAnalyzer.TIEBREAK_HEURISTICS
                             if col in df.columns]
        if not available_tiebreak:
            return pd.Series(dtype='int64')
        best_heuristics = df[available_tiebreak].idxmax(axis=1)
        return best_heuristics.value_counts()

    @staticmethod
    def compute_tiebreak_gap_analysis(df: pd.DataFrame) -> List[dict]:
        if 'optimal' not in df.columns:
            return []

        tiebreak_heuristics = StatisticalAnalyzer.TIEBREAK_HEURISTICS
        tiebreak_available = [col for col in tiebreak_heuristics if col in df.columns]

        if not tiebreak_available:
            return []

        tiebreak_instance_counts = {}
        for instance in df['instance']:
            best_tiebreak = df.loc[df['instance'] == instance, tiebreak_available].idxmax(axis=1).values[0]
            tiebreak_instance_counts.setdefault(best_tiebreak, []).append(instance)

        tiebreak_gap_stats_dominated = []
        for heuristic in tiebreak_available:
            if heuristic in tiebreak_instance_counts:
                instances = tiebreak_instance_counts[heuristic]
                if instances:
                    gaps = []
                    for instance in instances:
                        optimal_val = df.loc[df['instance'] == instance, 'optimal'].values[0]
                        heuristic_val = df.loc[df['instance'] == instance, heuristic].values[0]

                        if optimal_val > 0:
                            gap = ((optimal_val - heuristic_val) / optimal_val) * 100
                            gaps.append(gap)

                    if gaps:
                        tiebreak_gap_stats_dominated.append({
                            'heuristic': heuristic,
                            'instances': len(instances),
                            'mean_gap': np.mean(gaps),
                            'std_gap': np.std(gaps) if len(gaps) > 1 else 0,
                            'min_gap': np.min(gaps),
                            'max_gap': np.max(gaps),
                            'optimal_found': sum(1 for g in gaps if g == 0)
                        })

        return tiebreak_gap_stats_dominated

    @staticmethod
    def compute_comparison(df: pd.DataFrame, heuristic_columns: List[str]) -> pd.DataFrame:
        comparison = df[['instance'] + heuristic_columns].copy()
        comparison['best_heuristic'] = comparison[heuristic_columns].idxmax(axis=1)
        comparison['best_value'] = comparison[heuristic_columns].max(axis=1)
        comparison['worst_value'] = comparison[heuristic_columns].min(axis=1)
        comparison['value_range'] = comparison['best_value'] - comparison['worst_value']
        return comparison
