"""Optimize production ensemble threshold without retraining the model."""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from final_isolation_forest_model import add_engineered_features, fusion_scores
from preprocessing import build_preprocessing_pipeline, clean_data, handle_missing_values
from regenerate_production_outputs import (
    ANOMALY_OUTPUT_PATH,
    build_anomaly_scores_output,
    compute_ensemble_scores,
    load_production_bundle,
    transform_original_data,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
RAW_DATA_PATH = DATA_DIR / "ds1_processed.csv"

THRESHOLD_PERCENTILES = [90, 92, 94, 95, 96, 97, 98, 99]
THRESHOLD_METRICS_PATH = REPORT_DIR / "threshold_metrics.csv"
THRESHOLD_REPORT_PATH = REPORT_DIR / "threshold_optimization_report.md"


def validation_dataset_paths() -> List[Path]:
    """Return existing validation datasets only."""
    paths = sorted(DATA_DIR.glob("stress_test_dataset_rate_*.csv"))
    paths.extend(sorted(DATA_DIR.glob("cross_dataset_*.csv")))
    if not paths:
        raise FileNotFoundError("No existing validation datasets found.")
    return paths


def dataset_rate_from_path(path: Path, data: pd.DataFrame) -> float:
    if "anomaly_rate" in data.columns:
        return float(data["anomaly_rate"].iloc[0])
    if "stress_test_dataset_rate_" in path.stem:
        return float(path.stem.replace("stress_test_dataset_rate_", "").replace("p", "."))
    return float(data["ground_truth_label"].mean())


def score_validation_dataset(bundle: Dict, path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Load one validation dataset and compute production ensemble scores."""
    data = pd.read_csv(path)
    engineered = add_engineered_features(data[bundle["base_features"]])
    production_features = engineered[bundle["production_features"]]
    _, scores = compute_ensemble_scores(bundle, production_features)
    y_true = data["ground_truth_label"].astype(int).to_numpy()
    return data, scores.to_numpy(), y_true, dataset_rate_from_path(path, data)


def metrics_for_threshold(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "anomaly_percentage": float(y_pred.mean() * 100),
    }


def evaluate_thresholds(bundle: Dict) -> pd.DataFrame:
    """Evaluate requested percentile thresholds on all existing validation datasets."""
    rows = []
    for path in validation_dataset_paths():
        data, scores, y_true, anomaly_rate = score_validation_dataset(bundle, path)
        for percentile in THRESHOLD_PERCENTILES:
            threshold = float(np.percentile(scores, percentile))
            y_pred = (scores >= threshold).astype(int)
            row = metrics_for_threshold(y_true, y_pred, scores)
            row.update(
                {
                    "dataset_name": path.stem,
                    "ground_truth_anomaly_rate": anomaly_rate,
                    "threshold_percentile": percentile,
                    "threshold_value": threshold,
                    "records": int(len(data)),
                }
            )
            rows.append(row)

    metrics = pd.DataFrame(rows)
    aggregate = (
        metrics.groupby("threshold_percentile", as_index=False)[
            ["precision", "recall", "f1", "roc_auc", "anomaly_percentage", "accuracy", "tn", "fp", "fn", "tp"]
        ]
        .mean()
        .assign(dataset_name="AVERAGE", ground_truth_anomaly_rate=np.nan, threshold_value=np.nan, records=np.nan)
    )
    output = pd.concat([metrics, aggregate[metrics.columns]], ignore_index=True)
    output.to_csv(THRESHOLD_METRICS_PATH, index=False)
    return output


def select_best_threshold(metrics: pd.DataFrame) -> pd.Series:
    """Select threshold maximizing precision then F1 while keeping recall >= 0.75."""
    aggregate = metrics[metrics["dataset_name"] == "AVERAGE"].copy()
    eligible = aggregate[aggregate["recall"] >= 0.75].copy()
    if not eligible.empty:
        return eligible.sort_values(
            ["precision", "f1", "recall", "roc_auc"],
            ascending=False,
        ).iloc[0]

    return aggregate.sort_values(
        ["recall", "f1", "precision", "roc_auc"],
        ascending=False,
    ).iloc[0]


def markdown_table(data: pd.DataFrame) -> str:
    columns = data.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_threshold_report(metrics: pd.DataFrame, best: pd.Series) -> None:
    """Write markdown threshold optimization report."""
    aggregate = metrics[metrics["dataset_name"] == "AVERAGE"].copy()
    aggregate = aggregate[
        [
            "threshold_percentile",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "anomaly_percentage",
        ]
    ].sort_values("threshold_percentile")
    status = "PASSED" if best["recall"] >= 0.75 else "FAILED"
    recommendation = (
        f"Use the {int(best['threshold_percentile'])}th percentile of the incoming production ensemble score distribution as the production threshold."
        if status == "PASSED"
        else (
            f"No requested threshold achieved recall >= 0.75. Use the {int(best['threshold_percentile'])}th percentile as the closest feasible threshold "
            "because it has the highest recall and best F1 among the tested percentile options."
        )
    )
    lines = [
        "# Threshold Optimization Report",
        "",
        "## Scope",
        "",
        "Production model only. No retraining and no new synthetic datasets were generated.",
        "",
        "## Selection Rule",
        "",
        "Maximize precision, then F1, while keeping recall >= 0.75. ROC AUC is unchanged by thresholding because it depends on score ranking.",
        "",
        "## Average Metrics Across Existing Validation Datasets",
        "",
        markdown_table(aggregate),
        "",
        "## Selected Production Threshold",
        "",
        f"- Threshold percentile: {int(best['threshold_percentile'])}",
        f"- Precision: {best['precision']:.4f}",
        f"- Recall: {best['recall']:.4f}",
        f"- F1 Score: {best['f1']:.4f}",
        f"- ROC AUC: {best['roc_auc']:.4f}",
        f"- Predicted anomaly percentage: {best['anomaly_percentage']:.4f}%",
        f"- Recall constraint status: {status}",
        "",
        "## Recommendation",
        "",
        recommendation,
    ]
    THRESHOLD_REPORT_PATH.write_text("\n".join(lines))


def regenerate_anomaly_scores_with_percentile(bundle: Dict, percentile: int) -> int:
    """Regenerate anomaly_scores_production.csv using the selected percentile threshold."""
    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_raw, production_features = transform_original_data(
        raw_data,
        bundle["base_features"],
        bundle["production_features"],
    )
    _, ensemble_score = compute_ensemble_scores(bundle, production_features)
    threshold = float(np.percentile(ensemble_score, percentile))
    is_anomaly = (ensemble_score >= threshold).astype(int).to_numpy()
    anomaly_probability = ensemble_score.rank(pct=True).to_numpy()

    output = build_anomaly_scores_output(
        cleaned_raw,
        production_features,
        ensemble_score,
        anomaly_probability,
        is_anomaly,
    )
    output["threshold_method"] = f"percentile_{percentile}"
    output.to_csv(ANOMALY_OUTPUT_PATH, index=False)
    return int(output["is_anomaly"].sum())


def optimize_threshold_and_regenerate() -> None:
    bundle = load_production_bundle()
    metrics = evaluate_thresholds(bundle)
    best = select_best_threshold(metrics)
    write_threshold_report(metrics, best)
    anomaly_count = regenerate_anomaly_scores_with_percentile(bundle, int(best["threshold_percentile"]))

    print("===================================")
    print("PRODUCTION THRESHOLD OPTIMIZATION")
    print("===================================")
    print(f"Model Used: {BASE_DIR / 'models' / 'isolation_forest_production.pkl'}")
    print(f"Selected Threshold: percentile_{int(best['threshold_percentile'])}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"ROC AUC: {best['roc_auc']:.4f}")
    print(f"Anomaly Percentage: {best['anomaly_percentage']:.4f}%")
    print(f"Regenerated Production Anomalies: {anomaly_count}")
    print("Files Generated:")
    print(f"- {THRESHOLD_METRICS_PATH}")
    print(f"- {THRESHOLD_REPORT_PATH}")
    print(f"- {ANOMALY_OUTPUT_PATH}")


if __name__ == "__main__":
    optimize_threshold_and_regenerate()
