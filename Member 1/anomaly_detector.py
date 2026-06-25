"""Isolation Forest anomaly detection utilities for Member 1."""

from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from config import ANOMALY_SCORES_PATH, ISOLATION_FOREST_PARAMS, MODEL_PATH


def train_model(
    train_features: pd.DataFrame,
    params: Optional[dict[str, Any]] = None,
) -> IsolationForest:
    """Train an Isolation Forest model on selected telecom KPI features."""
    if train_features.empty:
        raise ValueError("Cannot train Isolation Forest with empty features.")

    model_params = params or ISOLATION_FOREST_PARAMS
    model = IsolationForest(**model_params)
    model.fit(train_features)
    return model


def predict_anomalies(
    model: IsolationForest,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict anomaly labels where -1 means anomaly and 1 means normal."""
    if features.empty:
        raise ValueError("Cannot predict anomalies for empty features.")

    return model.predict(features)


def predict_anomalies_adaptive(
    model: IsolationForest,
    features: pd.DataFrame,
    method: str = "gmm",
    gmm_probability_threshold: float = 0.90,
) -> np.ndarray:
    """Predict anomalies with an adaptive threshold from the incoming score distribution."""
    if features.empty:
        raise ValueError("Cannot predict anomalies for empty features.")

    scores = -model.decision_function(features)

    if method == "gmm":
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(scores.reshape(-1, 1))
        anomaly_component = int(np.argmax(gmm.means_.ravel()))
        anomaly_probability = gmm.predict_proba(scores.reshape(-1, 1))[:, anomaly_component]
        return np.where(anomaly_probability >= gmm_probability_threshold, -1, 1)

    threshold = select_adaptive_threshold(scores, method=method)
    return np.where(scores >= threshold, -1, 1)


def select_adaptive_threshold(scores: np.ndarray, method: str = "iqr") -> float:
    """Select an inference threshold without assuming a fixed contamination rate."""
    if scores.size == 0:
        raise ValueError("Cannot select a threshold from empty scores.")

    if method == "percentile_95":
        return float(np.percentile(scores, 95))
    if method == "percentile_98":
        return float(np.percentile(scores, 98))
    if method == "percentile_99":
        return float(np.percentile(scores, 99))

    q25, q75 = np.percentile(scores, [25, 75])
    iqr = q75 - q25
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)

    if method == "iqr":
        return float(q75 + 1.5 * iqr)
    if method == "mad":
        return float(median + 3.5 * 1.4826 * mad)
    if method == "gaussian":
        return float(mean + 2.5 * std)

    raise ValueError(f"Unsupported adaptive threshold method: {method}")


def compute_anomaly_scores(
    model: IsolationForest,
    features: pd.DataFrame,
) -> np.ndarray:
    """Compute normalized anomaly risk scores from 0.00 to 1.00."""
    if features.empty:
        raise ValueError("Cannot compute anomaly scores for empty features.")

    raw_scores = -model.score_samples(features)
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if np.isclose(max_score, min_score):
        return np.zeros_like(raw_scores, dtype=float)

    normalized_scores = (raw_scores - min_score) / (max_score - min_score)
    return np.clip(normalized_scores, 0.0, 1.0)


def save_model(
    model: IsolationForest,
    model_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save the trained Isolation Forest model."""
    output_path = Path(model_path) if model_path is not None else MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(model_path: Optional[Union[str, Path]] = None) -> IsolationForest:
    """Load a saved Isolation Forest model."""
    input_path = Path(model_path) if model_path is not None else MODEL_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found at: {input_path}")

    return joblib.load(input_path)


def _risk_status(score: float) -> str:
    """Map a normalized anomaly score to a downstream risk label."""
    if score <= 0.30:
        return "Healthy"
    if score <= 0.60:
        return "Moderate Risk"
    return "High Risk"


def scores_to_dataframe(
    anomaly_scores: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    record_ids: Optional[Union[pd.Series, list[Any], np.ndarray]] = None,
) -> pd.DataFrame:
    """Convert model outputs into the downstream anomaly score format."""
    if record_ids is None:
        record_ids = np.arange(1, len(anomaly_scores) + 1)

    scores = np.round(anomaly_scores.astype(float), 4)
    result = pd.DataFrame(
        {
            "record_id": list(record_ids),
            "anomaly_score": scores,
            "status": [_risk_status(score) for score in scores],
        }
    )

    if predictions is not None:
        result["is_anomaly"] = (predictions == -1).astype(int)

    return result


def save_anomaly_scores(
    scores_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save anomaly scores for downstream team members."""
    save_path = Path(output_path) if output_path is not None else ANOMALY_SCORES_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(save_path, index=False)
    return save_path


def to_json_records(scores_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert anomaly scores to JSON-ready records."""
    required_columns = {"record_id", "anomaly_score", "status"}
    missing_columns = required_columns.difference(scores_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required score columns: {sorted(missing_columns)}")

    return scores_df[["record_id", "anomaly_score", "status"]].to_dict(orient="records")


def evaluation_summary(scores_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize anomaly score outputs for reporting."""
    if scores_df.empty:
        raise ValueError("Cannot summarize an empty anomaly score dataframe.")

    summary = {
        "total_records": int(len(scores_df)),
        "mean_anomaly_score": float(scores_df["anomaly_score"].mean()),
        "max_anomaly_score": float(scores_df["anomaly_score"].max()),
        "min_anomaly_score": float(scores_df["anomaly_score"].min()),
        "status_counts": scores_df["status"].value_counts().to_dict(),
    }

    if "is_anomaly" in scores_df.columns:
        summary["predicted_anomalies"] = int(scores_df["is_anomaly"].sum())
        summary["predicted_anomaly_rate"] = float(scores_df["is_anomaly"].mean())

    return summary
