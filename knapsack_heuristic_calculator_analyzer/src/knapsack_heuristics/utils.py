import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(df: pd.DataFrame, output_path: str, include_timestamp: bool = True,
                base_name: Optional[str] = None) -> str:
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path_obj = Path(output_path)

        if base_name:
            filename = f"{base_name}_{timestamp}.csv"
        else:
            filename = f"results_{timestamp}.csv"

        if path_obj.is_dir():
            final_path = path_obj / filename
        else:
            final_path = path_obj.parent / f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
    else:
        final_path = Path(output_path)

    df.to_csv(final_path, index=False)
    logger.info(f"Saved results: {final_path}")
    return str(final_path)
