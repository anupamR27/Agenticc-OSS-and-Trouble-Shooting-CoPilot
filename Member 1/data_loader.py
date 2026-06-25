"""Data loading utilities for the Member 1 anomaly detection workflow."""

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORT_DIR,
    SAMPLE_SIZE,
    TEST_SIZE,
)


def _ensure_project_dirs() -> None:
    """Create local output folders required by the Member 1 workflow."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the telecom OSS dataset from CSV."""
    path = Path(file_path) if file_path is not None else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Loaded dataset is empty.")

    return data


def sample_data(
    data: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Sample the dataset to keep the Member 1 training scope between 500 and 2000 rows."""
    if data.empty:
        raise ValueError("Cannot sample an empty dataset.")

    if len(data) < 500:
        raise ValueError("Dataset must contain at least 500 records.")

    capped_sample_size = min(sample_size, 2000, len(data))

    if capped_sample_size < 500:
        raise ValueError("Sample size must be at least 500 records.")

    sampled = data.sample(n=capped_sample_size, random_state=random_state)
    return sampled.reset_index(drop=True)


def train_test_split_data(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into 80/20 train and test sets."""
    if data.empty:
        raise ValueError("Cannot split an empty dataset.")

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def save_processed_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Save processed train and test datasets for downstream workflow steps."""
    _ensure_project_dirs()

    train_output_path = Path(train_path) if train_path is not None else DATA_DIR / "train_processed.csv"
    test_output_path = Path(test_path) if test_path is not None else DATA_DIR / "test_processed.csv"

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

    return train_output_path, test_output_path
