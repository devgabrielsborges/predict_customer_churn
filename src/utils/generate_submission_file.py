import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def generate_submission_file(
    predictions: np.ndarray,
    model_name: str,
    test_csv_path: str | Path,
    target_column: str | None = None,
    id_column: str | None = None,
) -> Path:
    target_column = target_column or os.getenv("TARGET_COLUMN", "target")
    id_column = id_column or os.getenv("ID_COLUMN", "id")

    test_data = pd.read_csv(test_csv_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{model_name}_{timestamp}.csv"

    output_dir = Path(tempfile.mkdtemp())
    output_path = output_dir / filename

    submission = pd.DataFrame({
        id_column: test_data[id_column],
        target_column: predictions,
    })
    submission.to_csv(output_path, index=False)
    return output_path
