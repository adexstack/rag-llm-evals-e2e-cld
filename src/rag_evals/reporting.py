"""Write evaluation results to CSV."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(results: dict[str, Any], output_path: Path) -> Path:
    """Persist *results* dict to a CSV file at *output_path*.

    Creates parent directories automatically.  Returns the resolved path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: [v] for k, v in results.items()})
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    return output_path
