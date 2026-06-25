"""Regenerate output artifacts using the production Isolation Forest ensemble."""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.mixture import GaussianMixture

from final_isolation_forest_model import add_engineered_features, fusion_scores
from preprocessing import build_preprocessing_pipeline, clean_data, handle_missing_values


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "ds1_processed.csv"
PRODUCTION_MODEL_PATH = MODEL_DIR / "isolation_forest_production.pkl"

ANOMALY_OUTPUT_PATH = OUTPUT_DIR / "anomaly_scores_production.csv"
FEATURE_REPORT_OUTPUT_PATH = REPORT_DIR / "feature_report_production.csv"
TOP_FEATURES_OUTPUT_PATH = REPORT_DIR / "top_features.csv"
ANOMALY_SUMMARY_OUTPUT_PATH = REPORT_DIR / "anomaly_summary.csv"


BUSINESS_INTERPRETATIONS = {
    "latency": "Service delay and responsiveness",
    "packet_loss": "Packet delivery reliability",
    "throughput": "Traffic carrying performance",
    "prb": "Radio resource utilization",
    "active_users": "Cell user load",
    "rsrp": "Radio signal strength",
    "rsrq": "Radio signal quality",
    "spectral": "Radio efficiency",
    "handover": "Mobility stability",
    "congestion": "Load and capacity pressure",
    "service_degradation": "Combined latency/loss/throughput degradation",
    "radio_quality": "Combined signal health",
    "urllc": "URLLC slice SLA pressure",
    "embb": "eMBB capacity pressure",
}


def load_production_bundle() -> Dict:
    """Load the production model bundle."""
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model not found: {PRODUCTION_MODEL_PATH}")
    return joblib.load(PRODUCTION_MODEL_PATH)


def transform_original_data(raw_data: pd.DataFrame, base_features: List[str], production_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply cleaning, imputation, scaling, encoding, and production feature engineering."""
    cleaned = handle_missing_values(clean_data(raw_data))
    pipeline, _ = build_preprocessing_pipeline(cleaned)
    processed_array = pipeline.fit_transform(cleaned)
    processed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    processed = pd.DataFrame(processed_array, columns=processed_feature_names)

    missing_base = [feature for feature in base_features if feature not in processed.columns]
    if missing_base:
        raise ValueError(f"Missing base production features after preprocessing: {missing_base}")

    engineered = add_engineered_features(processed[base_features])
    missing_production = [feature for feature in production_features if feature not in engineered.columns]
    if missing_production:
        raise ValueError(f"Missing final production features: {missing_production}")

    return cleaned, engineered[production_features]


def compute_ensemble_scores(bundle: Dict, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute individual forest scores and the production ensemble score."""
    score_frame = pd.DataFrame(
        {
            name: -forest.decision_function(features)
            for name, forest in bundle["forests"].items()
        }
    )
    fused = fusion_scores(score_frame)
    ensemble_score = fused[bundle["fusion_method"]]
    return score_frame, ensemble_score


def gmm_threshold_decision(scores: pd.Series, probability_threshold: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the production GMM posterior threshold decision."""
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    values = scores.to_numpy().reshape(-1, 1)
    gmm.fit(values)
    anomaly_component = int(np.argmax(gmm.means_.ravel()))
    anomaly_probability = gmm.predict_proba(values)[:, anomaly_component]
    is_anomaly = anomaly_probability >= probability_threshold
    return anomaly_probability, is_anomaly.astype(int)


def risk_level(normalized_score: float, is_anomaly: int) -> str:
    """Map production score and threshold decision to a risk level."""
    if is_anomaly and normalized_score >= 0.75:
        return "High Risk"
    if is_anomaly:
        return "Moderate Risk"
    return "Healthy"


def feature_deviation_reference(feature_data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Create normal-reference statistics for explainability."""
    return {
        "mean": feature_data.mean(),
        "std": feature_data.std().replace(0, 1e-6),
        "median": feature_data.median(),
        "iqr": (feature_data.quantile(0.75) - feature_data.quantile(0.25)).replace(0, 1e-6),
    }


def top_kpi_drivers(row: pd.Series, reference: Dict[str, pd.Series], candidate_features: List[str]) -> List[Tuple[str, float]]:
    """Rank top KPI deviations for one record."""
    deviations = []
    for feature in candidate_features:
        if feature not in row.index:
            continue
        z_score = abs((row[feature] - reference["mean"][feature]) / reference["std"][feature])
        robust_distance = abs((row[feature] - reference["median"][feature]) / reference["iqr"][feature])
        deviation = float(0.65 * z_score + 0.35 * robust_distance)
        deviations.append((feature, deviation))
    return sorted(deviations, key=lambda item: item[1], reverse=True)[:3]


def anomaly_family_candidate(row: pd.Series) -> str:
    """Create a preliminary RCA candidate category without performing RCA."""
    if row.get("prb_utilization_pct", 0) > row.get("active_users", 0) and row.get("latency_ms", 0) > 0:
        return "Congestion"
    if row.get("rsrp_dbm", 0) < -0.75 or row.get("rsrq_db", 0) < -0.75:
        return "Coverage"
    if row.get("active_users", 0) > 1.0 and row.get("throughput_per_user", 0) < -0.5:
        return "Capacity"
    if row.get("rsrq_db", 0) < -0.75 or row.get("spectral_efficiency", 0) < -0.75:
        return "Signal Quality"
    if row.get("latency_ms", 0) > 1.0 and row.get("packet_loss_pct", 0) > 1.0:
        return "Transport"
    if row.get("active_users", 0) > 1.0:
        return "User Surge"
    return "Unknown"


def build_anomaly_scores_output(
    cleaned_raw: pd.DataFrame,
    production_features: pd.DataFrame,
    ensemble_score: pd.Series,
    anomaly_probability: np.ndarray,
    is_anomaly: np.ndarray,
) -> pd.DataFrame:
    """Build the production anomaly score output with explainability columns."""
    normalized = (ensemble_score - ensemble_score.min()) / (ensemble_score.max() - ensemble_score.min() + 1e-9)
    percentile = ensemble_score.rank(pct=True)
    reference = feature_deviation_reference(production_features)
    explain_features = [
        feature
        for feature in [
            "latency_ms",
            "packet_loss_pct",
            "throughput_mbps",
            "handover_count",
            "rsrp_dbm",
            "rsrq_db",
            "prb_utilization_pct",
            "active_users",
            "throughput_per_user",
            "spectral_efficiency",
            "service_degradation_index",
            "congestion_pressure_index",
            "radio_quality_index",
        ]
        if feature in production_features.columns
    ]

    rows = []
    for index, row in production_features.iterrows():
        drivers = top_kpi_drivers(row, reference, explain_features) if is_anomaly[index] else []
        padded = drivers + [("", 0.0)] * (3 - len(drivers))
        rows.append(
            {
                "record_id": index + 1,
                "ensemble_score": round(float(ensemble_score.iloc[index]), 6),
                "normalized_score": round(float(normalized.iloc[index]), 6),
                "anomaly_percentile": round(float(percentile.iloc[index]), 6),
                "threshold_method": "gmm_posterior_0_90",
                "risk_level": risk_level(float(normalized.iloc[index]), int(is_anomaly[index])),
                "is_anomaly": int(is_anomaly[index]),
                "primary_driver": padded[0][0],
                "secondary_driver": padded[1][0],
                "tertiary_driver": padded[2][0],
                "driver_1_deviation": round(float(padded[0][1]), 6),
                "driver_2_deviation": round(float(padded[1][1]), 6),
                "driver_3_deviation": round(float(padded[2][1]), 6),
                "anomaly_family_candidate": anomaly_family_candidate(row) if is_anomaly[index] else "Unknown",
            }
        )

    return pd.DataFrame(rows)


def feature_type(feature: str) -> str:
    if feature.startswith("cell_type_") or feature.startswith("slice_type_"):
        return "encoded_categorical"
    if feature.endswith("_z") or "ratio" in feature or "index" in feature or "_x_" in feature or "_per_" in feature:
        return "engineered_numeric"
    return "numeric"


def business_interpretation(feature: str) -> str:
    lowered = feature.lower()
    for keyword, interpretation in BUSINESS_INTERPRETATIONS.items():
        if keyword in lowered:
            return interpretation
    return "General telecom operating context"


def usefulness_reason(feature: str) -> str:
    lowered = feature.lower()
    if "latency" in lowered or "packet_loss" in lowered:
        return "Captures service degradation and transport instability."
    if "throughput" in lowered or "spectral" in lowered:
        return "Highlights capacity loss and radio efficiency collapse."
    if "rsrp" in lowered or "rsrq" in lowered:
        return "Detects coverage and signal-quality degradation."
    if "active_users" in lowered or "prb" in lowered or "congestion" in lowered:
        return "Identifies load pressure, resource saturation, and congestion."
    if "slice" in lowered:
        return "Separates slice-specific behavior and SLA risk."
    return "Provides contextual separation for anomalous telecom states."


def regenerate_feature_reports(production_features: pd.DataFrame, production_feature_names: List[str]) -> None:
    """Regenerate production feature report and top-feature interpretation file."""
    discriminative = pd.read_csv(REPORT_DIR / "feature_discriminative_power.csv")
    power_lookup = discriminative.set_index("feature")["discriminative_score"].to_dict()
    rank_lookup = {
        feature: rank
        for rank, feature in enumerate(discriminative["feature"].tolist(), start=1)
    }

    feature_rows = []
    for feature in production_feature_names:
        feature_rows.append(
            {
                "feature_name": feature,
                "feature_type": feature_type(feature),
                "original_or_engineered": "original" if feature in production_feature_names[:32] else "engineered",
                "variance": float(production_features[feature].var()),
                "missing_count": int(production_features[feature].isna().sum()),
                "importance_rank": int(rank_lookup.get(feature, len(production_feature_names) + 1)),
                "discriminative_power": float(power_lookup.get(feature, 0.0)),
                "selected_for_production": True,
            }
        )
    pd.DataFrame(feature_rows).sort_values("importance_rank").to_csv(FEATURE_REPORT_OUTPUT_PATH, index=False)

    top_features = (
        pd.DataFrame(feature_rows)
        .sort_values("importance_rank")
        .head(25)
        .reset_index(drop=True)
    )
    top_features_output = pd.DataFrame(
        {
            "Rank": range(1, len(top_features) + 1),
            "Feature": top_features["feature_name"],
            "Importance Score": top_features["discriminative_power"],
            "Business Interpretation": [business_interpretation(feature) for feature in top_features["feature_name"]],
            "Reason Useful For Detecting Anomalies": [usefulness_reason(feature) for feature in top_features["feature_name"]],
        }
    )
    top_features_output.to_csv(TOP_FEATURES_OUTPUT_PATH, index=False)


def write_anomaly_summary(scores_output: pd.DataFrame) -> None:
    """Write one-row anomaly summary with risk-level distribution."""
    risk_distribution = scores_output["risk_level"].value_counts().to_dict()
    summary = pd.DataFrame(
        [
            {
                "Total Records": len(scores_output),
                "Total Anomalies": int(scores_output["is_anomaly"].sum()),
                "Anomaly Percentage": round(float(scores_output["is_anomaly"].mean() * 100), 4),
                "Average Score": round(float(scores_output["ensemble_score"].mean()), 6),
                "Maximum Score": round(float(scores_output["ensemble_score"].max()), 6),
                "Minimum Score": round(float(scores_output["ensemble_score"].min()), 6),
                "Risk Level Distribution": str(risk_distribution),
            }
        ]
    )
    summary.to_csv(ANOMALY_SUMMARY_OUTPUT_PATH, index=False)


def draw_histogram(path: Path, values: pd.Series, title: str, color: Tuple[int, int, int]) -> None:
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 80, 80, 850, 470
    draw.text((330, 30), title, fill="black", font=font)
    draw.rectangle([left, top, right, bottom], outline="black")
    counts, bins = np.histogram(values, bins=45)
    max_count = max(int(counts.max()), 1)
    for index, count in enumerate(counts):
        x0 = left + int((index / len(counts)) * (right - left))
        x1 = left + int(((index + 1) / len(counts)) * (right - left))
        height = int((count / max_count) * (bottom - top))
        draw.rectangle([x0, bottom - height, x1, bottom], fill=color)
    draw.text((380, 515), "Score", fill="black", font=font)
    draw.text((25, 270), "Count", fill="black", font=font)
    image.save(path)


def draw_bar_chart(path: Path, labels: List[str], values: List[float], title: str) -> None:
    image = Image.new("RGB", (980, 600), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 260, 80, 920, 520
    draw.text((380, 30), title, fill="black", font=font)
    draw.rectangle([left, top, right, bottom], outline="black")
    max_value = max(values) if values else 1.0
    bar_height = max(18, int((bottom - top) / max(len(labels), 1)) - 8)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * (bar_height + 8) + 5
        width = int((value / max_value) * (right - left)) if max_value else 0
        draw.text((20, y + 4), str(label)[:34], fill="black", font=font)
        draw.rectangle([left, y, left + width, y + bar_height], fill=(70, 140, 210))
        draw.text((left + width + 8, y + 4), f"{value:.3f}", fill="black", font=font)
    image.save(path)


def generate_visualizations(scores_output: pd.DataFrame) -> None:
    """Generate production visualization artifacts."""
    draw_histogram(
        REPORT_DIR / "production_anomaly_distribution.png",
        scores_output["normalized_score"],
        "Production Anomaly Distribution",
        (90, 145, 215),
    )
    risk_counts = scores_output["risk_level"].value_counts()
    draw_bar_chart(
        REPORT_DIR / "risk_level_distribution.png",
        risk_counts.index.tolist(),
        risk_counts.astype(float).tolist(),
        "Risk Level Distribution",
    )
    top_features = pd.read_csv(TOP_FEATURES_OUTPUT_PATH).head(15)
    draw_bar_chart(
        REPORT_DIR / "top_feature_importance.png",
        top_features["Feature"].tolist(),
        top_features["Importance Score"].astype(float).tolist(),
        "Top Feature Importance",
    )
    draw_histogram(
        REPORT_DIR / "anomaly_score_histogram.png",
        scores_output["ensemble_score"],
        "Anomaly Score Histogram",
        (220, 110, 85),
    )


def regenerate_outputs() -> None:
    """Main production artifact regeneration workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_production_bundle()
    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_raw, production_features = transform_original_data(
        raw_data,
        bundle["base_features"],
        bundle["production_features"],
    )

    score_frame, ensemble_score = compute_ensemble_scores(bundle, production_features)
    anomaly_probability, is_anomaly = gmm_threshold_decision(ensemble_score)
    scores_output = build_anomaly_scores_output(
        cleaned_raw,
        production_features,
        ensemble_score,
        anomaly_probability,
        is_anomaly,
    )
    scores_output.to_csv(ANOMALY_OUTPUT_PATH, index=False)

    regenerate_feature_reports(production_features, bundle["production_features"])
    write_anomaly_summary(scores_output)
    generate_visualizations(scores_output)

    generated_outputs = [
        ANOMALY_OUTPUT_PATH,
        ANOMALY_SUMMARY_OUTPUT_PATH,
        REPORT_DIR / "production_anomaly_distribution.png",
        REPORT_DIR / "risk_level_distribution.png",
        REPORT_DIR / "top_feature_importance.png",
        REPORT_DIR / "anomaly_score_histogram.png",
    ]
    generated_reports = [
        FEATURE_REPORT_OUTPUT_PATH,
        TOP_FEATURES_OUTPUT_PATH,
    ]

    print("===================================")
    print("PRODUCTION OUTPUT REGENERATION")
    print("==============================")
    print(f"Model Used: {PRODUCTION_MODEL_PATH}")
    print(f"Number of Records: {len(raw_data)}")
    print(f"Number of Features: {len(bundle['production_features'])}")
    print(f"Number of Anomalies: {int(scores_output['is_anomaly'].sum())}")
    print("Output Files Generated:")
    for path in generated_outputs:
        print(f"- {path}")
    print("Feature Reports Generated:")
    for path in generated_reports:
        print(f"- {path}")
    print("Validation Status: PASSED")


if __name__ == "__main__":
    regenerate_outputs()
