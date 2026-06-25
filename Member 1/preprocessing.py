"""Preprocessing utilities for Telecom OSS anomaly detection."""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_COLUMNS, ID_COLUMNS, TARGET_COLUMN, TIME_COLUMNS


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and normalize basic column formatting."""
    if data.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in TIME_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    return cleaned


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using median for numeric columns and mode for categorical columns."""
    if data.empty:
        raise ValueError("Cannot handle missing values for an empty dataset.")

    filled = data.copy()
    numeric_columns = filled.select_dtypes(include=[np.number]).columns
    categorical_columns = filled.select_dtypes(include=["object", "category", "bool"]).columns

    for column in numeric_columns:
        filled[column] = filled[column].fillna(filled[column].median())

    for column in categorical_columns:
        mode_value = filled[column].mode(dropna=True)
        replacement = mode_value.iloc[0] if not mode_value.empty else "unknown"
        filled[column] = filled[column].fillna(replacement)

    return filled


def encode_features(
    data: pd.DataFrame,
    categorical_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """One-hot encode categorical features for model training."""
    encoded = data.copy()
    columns_to_encode = [
        column
        for column in (categorical_columns or CATEGORICAL_COLUMNS)
        if column in encoded.columns
    ]

    if not columns_to_encode:
        return encoded

    return pd.get_dummies(encoded, columns=columns_to_encode, drop_first=False)


def scale_features(
    data: pd.DataFrame,
    numeric_columns: Optional[Iterable[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric feature columns with StandardScaler."""
    scaled = data.copy()
    columns_to_scale = list(numeric_columns) if numeric_columns is not None else list(
        scaled.select_dtypes(include=[np.number]).columns
    )

    if not columns_to_scale:
        return scaled, scaler or StandardScaler()

    fitted_scaler = scaler or StandardScaler()
    scaled[columns_to_scale] = fitted_scaler.fit_transform(scaled[columns_to_scale])
    return scaled, fitted_scaler


def detect_outliers(
    data: pd.DataFrame,
    numeric_columns: Optional[Iterable[str]] = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Add an IQR-based outlier flag for exploratory analysis."""
    if data.empty:
        raise ValueError("Cannot detect outliers for an empty dataset.")

    result = data.copy()
    columns_to_check = list(numeric_columns) if numeric_columns is not None else list(
        result.select_dtypes(include=[np.number]).columns
    )

    outlier_mask = pd.Series(False, index=result.index)

    for column in columns_to_check:
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        outlier_mask |= (result[column] < lower_bound) | (result[column] > upper_bound)

    result["eda_outlier_flag"] = outlier_mask.astype(int)
    return result


def _one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with recent and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessing_pipeline(
    data: pd.DataFrame,
    drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[Pipeline, list[str]]:
    """Build a reusable sklearn preprocessing pipeline for Isolation Forest features."""
    if data.empty:
        raise ValueError("Cannot build a preprocessing pipeline from an empty dataset.")

    excluded_columns = set(drop_columns or [])
    excluded_columns.update(column for column in ID_COLUMNS if column in data.columns)
    excluded_columns.update(column for column in TIME_COLUMNS if column in data.columns)

    if TARGET_COLUMN and TARGET_COLUMN in data.columns:
        excluded_columns.add(TARGET_COLUMN)

    feature_columns = [column for column in data.columns if column not in excluded_columns]
    feature_data = data[feature_columns]

    numeric_features = list(feature_data.select_dtypes(include=[np.number]).columns)
    categorical_features = [
        column
        for column in feature_data.select_dtypes(include=["object", "category", "bool"]).columns
        if column in feature_columns
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline, feature_columns
