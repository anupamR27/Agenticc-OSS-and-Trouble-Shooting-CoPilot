"""Feature selection utilities for Telecom OSS anomaly detection."""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from config import (
    CORRELATION_THRESHOLD,
    FEATURE_REPORT_PATH,
    ID_COLUMNS,
    KPI_KEYWORDS,
    TARGET_COLUMN,
    TIME_COLUMNS,
    VARIANCE_THRESHOLD,
)


def identify_kpi_features(
    columns: Iterable[str],
    kpi_keywords: Optional[Iterable[str]] = None,
) -> list[str]:
    """Identify telecom KPI columns using configured domain keywords."""
    keywords = [keyword.lower() for keyword in (kpi_keywords or KPI_KEYWORDS)]
    kpi_features = []

    for column in columns:
        normalized_column = column.lower()
        if any(keyword in normalized_column for keyword in keywords):
            kpi_features.append(column)

    return kpi_features


def remove_highly_correlated_features(
    data: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    protected_features: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Remove one feature from each highly correlated numeric feature pair."""
    if data.empty:
        raise ValueError("Cannot remove correlated features from an empty dataset.")

    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] <= 1:
        return data.copy(), []

    protected = set(protected_features or [])
    correlation_matrix = numeric_data.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    features_to_drop = []
    for column in upper_triangle.columns:
        high_corr_partners = upper_triangle.index[upper_triangle[column] > threshold].tolist()
        if high_corr_partners and column not in protected:
            features_to_drop.append(column)

    features_to_drop = sorted(set(features_to_drop))
    reduced_data = data.drop(columns=features_to_drop, errors="ignore")
    return reduced_data, features_to_drop


def _variance_selected_features(
    data: pd.DataFrame,
    threshold: float = VARIANCE_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Apply variance thresholding to numeric features."""
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return [], []

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_data)

    selected_features = numeric_data.columns[selector.get_support()].tolist()
    removed_features = numeric_data.columns[~selector.get_support()].tolist()
    return selected_features, removed_features


def select_features(
    data: pd.DataFrame,
    variance_threshold: float = VARIANCE_THRESHOLD,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Select model-ready features using variance, correlation, and KPI checks."""
    if data.empty:
        raise ValueError("Cannot select features from an empty dataset.")

    excluded_columns = set(column for column in ID_COLUMNS if column in data.columns)
    excluded_columns.update(column for column in TIME_COLUMNS if column in data.columns)

    if TARGET_COLUMN and TARGET_COLUMN in data.columns:
        excluded_columns.add(TARGET_COLUMN)

    candidate_data = data.drop(columns=list(excluded_columns), errors="ignore")
    numeric_candidate_data = candidate_data.select_dtypes(include=[np.number])

    variance_selected, low_variance_features = _variance_selected_features(
        numeric_candidate_data,
        threshold=variance_threshold,
    )
    variance_filtered = candidate_data[variance_selected].copy()

    kpi_features = identify_kpi_features(variance_filtered.columns)
    selected_data, correlated_features = remove_highly_correlated_features(
        variance_filtered,
        threshold=correlation_threshold,
        protected_features=kpi_features,
    )

    selected_features = selected_data.columns.tolist()
    feature_report = generate_feature_report(
        candidate_data=candidate_data,
        selected_features=selected_features,
        low_variance_features=low_variance_features,
        correlated_features=correlated_features,
        kpi_features=kpi_features,
    )

    return selected_data, selected_features, feature_report


def generate_feature_report(
    candidate_data: pd.DataFrame,
    selected_features: Iterable[str],
    low_variance_features: Optional[Iterable[str]] = None,
    correlated_features: Optional[Iterable[str]] = None,
    kpi_features: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a feature selection report that can be saved as CSV."""
    selected = set(selected_features)
    low_variance = set(low_variance_features or [])
    correlated = set(correlated_features or [])
    kpis = set(kpi_features or identify_kpi_features(candidate_data.columns))

    rows = []
    for column in candidate_data.columns:
        if column in selected:
            decision = "selected"
        elif column in low_variance:
            decision = "removed_low_variance"
        elif column in correlated:
            decision = "removed_high_correlation"
        else:
            decision = "not_numeric_or_excluded"

        rows.append(
            {
                "feature": column,
                "dtype": str(candidate_data[column].dtype),
                "missing_count": int(candidate_data[column].isna().sum()),
                "variance": (
                    float(candidate_data[column].var())
                    if pd.api.types.is_numeric_dtype(candidate_data[column])
                    else None
                ),
                "is_kpi": column in kpis,
                "decision": decision,
            }
        )

    report = pd.DataFrame(rows)

    save_path = output_path or FEATURE_REPORT_PATH
    if save_path:
        save_path = pd.io.common.stringify_path(save_path)
        report.to_csv(save_path, index=False)

    return report
