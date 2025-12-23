import logging
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import psutil

from .analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False


class ReportFormatter:
    def __init__(self):
        self.mty_tz = pytz.timezone('America/Monterrey')
        self.current_time = datetime.now(self.mty_tz)

    def format_header(self, title: str, has_optimal: bool = False) -> List[str]:
        full_title = title
        if has_optimal:
            full_title += " (WITH OPTIMAL)"

        return [
            "=" * 80,
            full_title,
            "=" * 80
        ]

    def format_section(self, title: str) -> List[str]:
        return ["", title.upper(), "-" * 80]

    def format_execution_info(self) -> List[str]:
        return [
            f"Date: {self.current_time.strftime('%Y-%m-%d')}",
            f"Time: {self.current_time.strftime('%H:%M:%S')}",
            f"Timezone: America/Monterrey (UTC{self.current_time.strftime('%z')})"
        ]

    def format_hardware_info(self) -> List[str]:
        lines = []

        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()

            lines.extend([
                f"CPU Cores: {cpu_count_physical} physical, {cpu_count} logical",
                f"Total Memory: {memory.total / (1024**3):.1f} GB",
                f"Available Memory: {memory.available / (1024**3):.1f} GB",
                f"Memory Usage: {memory.percent}%"
            ])

            if cpu_freq:
                lines.append(f"CPU Frequency: {cpu_freq.current:.0f} MHz")

        except Exception as e:
            logger.warning(f"Could not retrieve basic hardware info: {e}")

        if HAS_CPUINFO:
            try:
                cpu_info = cpuinfo.get_cpu_info()
                brand = cpu_info.get('brand_raw', cpu_info.get('brand', 'Unknown'))
                lines.append(f"CPU Model: {brand}")
            except Exception as e:
                logger.debug(f"cpuinfo failed, skipping detailed CPU info: {e}")
        else:
            lines.append("CPU Model: Install 'py-cpuinfo' for detailed CPU information")

        return lines

    def format_statistics_table(self, stats_df: pd.DataFrame) -> List[str]:
        lines = []
        header = f"{'Heuristic':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        lines.append(header)
        lines.append("-" * 80)

        for _, row in stats_df.iterrows():
            if row['heuristic'] != 'optimal':
                lines.append(
                    f"{row['heuristic']:<40} "
                    f"{row['mean']:>10.2f} "
                    f"{row['std']:>10.2f} "
                    f"{row['min']:>10.0f} "
                    f"{row['max']:>10.0f}"
                )
        return lines

    def format_gap_table(self, gap_stats_df: pd.DataFrame, total_instances: int) -> List[str]:
        lines = []
        header = (f"{'Heuristic':<40} {'Mean Gap %':>12} {'Std Gap %':>12} "
                 f"{'Min Gap %':>12} {'Max Gap %':>12} {'Optimal Found':>15}")
        lines.append(header)
        lines.append("-" * 130)

        sorted_stats = gap_stats_df.sort_values('mean_gap')
        for _, row in sorted_stats.iterrows():
            lines.append(
                f"{row['heuristic']:<40} "
                f"{row['mean_gap']:>12.2f} "
                f"{row['std_gap']:>12.2f} "
                f"{row['min_gap']:>12.2f} "
                f"{row['max_gap']:>12.2f} "
                f"{row['optimal_found']:>8}/{total_instances}"
            )

        best_heuristic = sorted_stats.iloc[0]['heuristic']
        best_gap = sorted_stats.iloc[0]['mean_gap']
        worst_heuristic = sorted_stats.iloc[-1]['heuristic']
        worst_gap = sorted_stats.iloc[-1]['mean_gap']

        lines.append("")
        lines.append(f"Best gap: {best_heuristic} ({best_gap:.2f}% avg)")
        lines.append(f"Worst gap: {worst_heuristic} ({worst_gap:.2f}% avg)")

        return lines

    def format_tiebreak_gap_table(self, tiebreak_stats: List[dict]) -> List[str]:
        if not tiebreak_stats:
            return ["No tie-break dominance data available"]

        lines = []
        header = (f"{'Heuristic':<40} {'Instances':>10} {'Mean Gap %':>12} {'Std Gap %':>12} "
                 f"{'Min Gap %':>12} {'Max Gap %':>12} {'Optimal Found':>15}")
        lines.append(header)
        lines.append("-" * 140)

        sorted_stats = sorted(tiebreak_stats, key=lambda x: x['mean_gap'])
        for stats in sorted_stats:
            lines.append(
                f"{stats['heuristic']:<40} "
                f"{stats['instances']:>10} "
                f"{stats['mean_gap']:>12.2f} "
                f"{stats['std_gap']:>12.2f} "
                f"{stats['min_gap']:>12.2f} "
                f"{stats['max_gap']:>12.2f} "
                f"{stats['optimal_found']:>8}/{stats['instances']}"
            )

        if sorted_stats:
            best = sorted_stats[0]
            worst = sorted_stats[-1]
            lines.append("")
            lines.append(f"Best tie-break gap (when dominated): {best['heuristic']} ({best['mean_gap']:.2f}% avg, {best['instances']} instances)")
            lines.append(f"Worst tie-break gap (when dominated): {worst['heuristic']} ({worst['mean_gap']:.2f}% avg, {worst['instances']} instances)")

        return lines

    def format_dominant_heuristics(self, dominant: pd.Series, total: int, category: str = "ALL") -> List[str]:
        if dominant.empty:
            return [f"No dominance data for {category} heuristics"]

        lines = [f"Number of times each {category} heuristic achieved best solution:", ""]
        for heuristic, count in dominant.sort_values(ascending=False).items():
            percentage = (count / total) * 100
            lines.append(f"  {heuristic:<40} {count:>6} instances ({percentage:>5.1f}%)")

        return lines

    def format_execution_times(self, avg_times: Dict[str, float],
                              total_times: Dict[str, float]) -> List[str]:
        lines = []
        header = f"{'Method':<40} {'Avg/Inst (μs)':>15} {'Total (ms)':>15}"
        lines.append(header)
        lines.append("-" * 80)

        for name in sorted(avg_times.keys()):
            if name != 'optimal':
                avg_us = avg_times[name] * 1_000_000
                total_ms = total_times.get(name, 0) * 1000
                lines.append(f"{name:<40} {avg_us:>15.3f} {total_ms:>15.3f}")

        if 'optimal' in avg_times:
            avg_us = avg_times['optimal'] * 1_000_000
            total_ms = total_times.get('optimal', 0) * 1000
            lines.append(f"{'DP Optimal Solver':<40} {avg_us:>15.3f} {total_ms:>15.3f}")

        return lines


class ReportGenerator:
    def __init__(self, formatter: ReportFormatter, analyzer: StatisticalAnalyzer):
        self.formatter = formatter
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__ + '.ReportGenerator')

    def generate(self, df: pd.DataFrame, execution_times: Dict[str, List[float]],
                config: dict, output_path: str) -> str:
        lines = []
        heuristic_columns = [col for col in df.columns
                           if col not in ['instance', 'capacity', 'optimal']]

        has_optimal = 'optimal' in df.columns

        lines.extend(self.formatter.format_header(
            "KNAPSACK HEURISTICS EXPERIMENT REPORT", has_optimal))

        lines.extend(self.formatter.format_section("EXECUTION INFORMATION"))
        lines.extend(self.formatter.format_execution_info())

        lines.extend(self.formatter.format_section("HARDWARE INFORMATION"))
        lines.extend(self.formatter.format_hardware_info())

        lines.extend(self.formatter.format_section("EXPERIMENT CONFIGURATION"))
        lines.extend(self._format_config(config, df))

        lines.extend(self.formatter.format_section("HEURISTICS EVALUATED"))
        lines.extend([f"{i}. {h}" for i, h in enumerate(heuristic_columns, 1)])

        stats_df = self.analyzer.compute_statistics(df, heuristic_columns)
        lines.extend(self.formatter.format_section("HEURISTIC PERFORMANCE STATISTICS"))
        lines.extend(self.formatter.format_statistics_table(stats_df))

        if has_optimal:
            gap_stats = self.analyzer.compute_gap_statistics(df, heuristic_columns)
            lines.extend(self.formatter.format_section("OPTIMALITY GAP ANALYSIS"))
            lines.extend(self.formatter.format_gap_table(gap_stats, len(df)))

        avg_times = {name: np.mean(times) for name, times in execution_times.items()}
        total_times = {name: np.sum(times) for name, times in execution_times.items()}
        lines.extend(self.formatter.format_section("EXECUTION TIME ANALYSIS"))
        lines.extend(self.formatter.format_execution_times(avg_times, total_times))

        dominant = self.analyzer.identify_dominant_heuristic(df, heuristic_columns)
        lines.extend(self.formatter.format_section("DOMINANT HEURISTIC ANALYSIS (ALL HEURISTICS)"))
        lines.extend(self.formatter.format_dominant_heuristics(dominant, len(df), "ALL"))

        dominant_basic = self.analyzer.identify_dominant_basic_heuristics(df)
        if not dominant_basic.empty:
            lines.extend(self.formatter.format_section("BASIC HEURISTICS DOMINANCE ANALYSIS"))
            lines.extend(self.formatter.format_dominant_heuristics(dominant_basic, len(df), "BASIC"))

        dominant_tiebreak = self.analyzer.identify_dominant_tiebreak_heuristics(df)
        if not dominant_tiebreak.empty:
            lines.extend(self.formatter.format_section("TIE-BREAK HEURISTICS DOMINANCE ANALYSIS"))
            lines.extend(self.formatter.format_dominant_heuristics(dominant_tiebreak, len(df), "TIE-BREAK"))

        if has_optimal:
            tiebreak_gap_stats = self.analyzer.compute_tiebreak_gap_analysis(df)
            if tiebreak_gap_stats:
                lines.extend(self.formatter.format_section("TIE-BREAK OPTIMALITY GAP ANALYSIS (Instances where each dominated)"))
                lines.extend(self.formatter.format_tiebreak_gap_table(tiebreak_gap_stats))

        lines.extend(["", "END OF REPORT", "=" * 80])

        report_text = '\n'.join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        self.logger.info(f"Report generated: {output_path}")
        return report_text

    def _format_config(self, config: dict, df: pd.DataFrame) -> List[str]:
        return [
            f"Capacity: {config.get('capacity', 'N/A')}",
            f"Data directory: {config.get('data_dir', 'N/A')}",
            f"Total instances: {len(df)}",
            f"Compute optimal: {'optimal' in df.columns}",
            f"Parallel processing: {config.get('enable_parallel', False)}",
            f"Python version: {config.get('python_version', 'N/A')}",
            f"NumPy version: {np.__version__}",
            f"Pandas version: {pd.__version__}"
        ]
