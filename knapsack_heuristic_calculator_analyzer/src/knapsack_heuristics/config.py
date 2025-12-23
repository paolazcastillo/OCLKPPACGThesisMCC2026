import sys
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
import psutil
import pytz
from datetime import datetime
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='KP_')

    capacity: int = Field(
        default=64,
        gt=0,
        description="Knapsack capacity for all instances"
    )

    data_dir: str = Field(
        default='data',
        description="Directory containing instance CSV files"
    )

    output_dir: str = Field(
        default='output',
        description="Directory for output files"
    )

    pattern: str = Field(
        default='*.csv',
        description="File pattern for instance files"
    )

    compute_optimal: bool = Field(
        default=True,
        description="Whether to compute optimal solutions via DP"
    )

    timezone: str = Field(
        default='America/Monterrey',
        description="Timezone for timestamps"
    )

    log_level: str = Field(
        default='INFO',
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility (future use)"
    )

    max_workers: Optional[int] = Field(
        default=None,
        description="Max workers for parallel processing (None = auto-detect from CPU count)"
    )

    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel processing for multiple instances"
    )

    @field_validator('max_workers')
    @classmethod
    def validate_max_workers(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            cpu_count = psutil.cpu_count(logical=True)
            if v > cpu_count * 2:
                logger.warning(
                    f"max_workers={v} exceeds 2x CPU count ({cpu_count}). "
                    f"This may cause performance degradation."
                )
            if v < 1:
                raise ValueError("max_workers must be >= 1")
        return v

    @field_validator('data_dir', 'output_dir')
    @classmethod
    def validate_directory(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            logger.warning(f"Directory {v} does not exist, will attempt to create")
            path.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    def setup_logging(self):
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger().setLevel(self.log_level)

    def to_dict(self) -> dict:
        return {
            'capacity': self.capacity,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'pattern': self.pattern,
            'compute_optimal': self.compute_optimal,
            'timezone': self.timezone,
            'log_level': self.log_level,
            'random_seed': self.random_seed,
            'max_workers': self.max_workers,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'timestamp': datetime.now(pytz.timezone(self.timezone)).isoformat()
        }

    def save_to_file(self, filepath: str):
        import json
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentConfig':
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**{k: v for k, v in config_dict.items() if k in cls.model_fields})
