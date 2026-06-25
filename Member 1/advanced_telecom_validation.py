"""Advanced OOD validation framework for the Isolation Forest detector."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import kurtosis, skew
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KernelDensity

from preprocessing import build_preprocessing_pipeline, clean_data, handle_missing_values


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "ds1_processed.csv"
FEATURE_REPORT_PATH = REPORT_DIR / "feature_report.csv"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
STRESS_DATASET_PATH = DATA_DIR / "stress_test_dataset.csv"

RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)

ANOMALY_RATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
STRESS_SIZE = 50000

ONE_HOT_FEATURES = [
    "cell_type_macro",
    "cell_type_micro",
    "cell_type_pico",
    "slice_type_HC",
    "slice_type_URLLC",
    "slice_type_eMBB",
]

REGIME_NAMES = [
    "Normal network",
    "High traffic urban network",
    "Congested network",
    "Rural weak-signal network",
    "Extreme peak-hour network",
    "Slice-specific network behavior",
]

ANOMALY_FAMILIES = [
    "Latency-only anomalies",
    "Packet-loss anomalies",
    "Throughput-collapse anomalies",
    "Signal-quality anomalies",
    "Congestion anomalies",
    "User-surge anomalies",
    "Combined multi-KPI anomalies",
    "Subtle anomalies",
    "Contradictory anomalies",
]


def load_selected_features() -> List[str]:
    """Read the selected model features from the feature report."""
    feature_report = pd.read_csv(FEATURE_REPORT_PATH)
    selected_features = feature_report.loc[
        feature_report["decision"] == "selected",
        "feature",
    ].tolist()

    if len(selected_features) != 32:
        raise ValueError(f"Expected 32 selected features, found {len(selected_features)}.")

    return selected_features


def load_processed_training_features(selected_features: List[str]) -> pd.DataFrame:
    """Load raw KPI data and transform it into the model feature space."""
    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_data = handle_missing_values(clean_data(raw_data))
    pipeline, _ = build_preprocessing_pipeline(cleaned_data)
    processed = pipeline.fit_transform(cleaned_data)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    processed_data = pd.DataFrame(processed, columns=feature_names)
    return processed_data[selected_features].copy()


def continuous_features(selected_features: Iterable[str]) -> List[str]:
    return [feature for feature in selected_features if feature not in ONE_HOT_FEATURES]


def generate_kpi_profile(base_features: pd.DataFrame) -> None:
    """Create distribution, covariance, correlation, quantile, skewness, and kurtosis reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    numeric = base_features.select_dtypes(include=[np.number])
    profile_rows = []
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy()
        profile_rows.append(
            {
                "feature": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
                "skewness": float(skew(values)),
                "kurtosis": float(kurtosis(values)),
            }
        )

    profile = pd.DataFrame(profile_rows)
    covariance = numeric.cov()
    correlation = numeric.corr()

    profile.to_csv(REPORT_DIR / "telecom_kpi_profile.csv", index=False)
    covariance.to_csv(REPORT_DIR / "feature_covariance_matrix.csv")
    correlation.to_csv(REPORT_DIR / "feature_correlation_matrix.csv")

    strongest_correlations = (
        correlation.abs()
        .where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .head(15)
    )

    markdown = [
        "# Telecom KPI Profile Report",
        "",
        "The profile is computed from the model's 32 selected feature space.",
        "",
        "## Dataset Shape",
        "",
        f"- Records: {len(base_features)}",
        f"- Selected features: {base_features.shape[1]}",
        "",
        "## Distribution Summary",
        "",
        dataframe_to_markdown(profile.head(32)),
        "",
        "## Strongest Absolute Feature Correlations",
        "",
    ]
    for (left, right), value in strongest_correlations.items():
        markdown.append(f"- {left} vs {right}: {value:.4f}")

    (REPORT_DIR / "telecom_kpi_profile.md").write_text("\n".join(markdown))


def fit_kde_sampler(base_features: pd.DataFrame, bandwidth: float = 0.35) -> KernelDensity:
    """Fit KDE on continuous features for regime generation."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(base_features.to_numpy())
    return kde


def sample_correlated_base(
    base_features: pd.DataFrame,
    n_rows: int,
    kde: KernelDensity,
) -> pd.DataFrame:
    """Generate correlation-preserving samples using KDE and multivariate covariance noise."""
    selected_features = base_features.columns.tolist()
    cont_features = continuous_features(selected_features)
    one_hot = [feature for feature in ONE_HOT_FEATURES if feature in selected_features]
    n_kde = int(n_rows * 0.35)
    n_bootstrap = n_rows - n_kde

    kde_samples = pd.DataFrame(
        kde.sample(n_kde, random_state=int(RNG.integers(1, 1_000_000))),
        columns=cont_features,
    )

    sampled = base_features.iloc[RNG.integers(0, len(base_features), size=n_bootstrap)].reset_index(drop=True)
    bootstrap = sampled[cont_features].copy()
    covariance = np.nan_to_num(base_features[cont_features].cov().to_numpy())
    covariance += np.eye(covariance.shape[0]) * 1e-6
    noise = RNG.multivariate_normal(
        mean=np.zeros(len(cont_features)),
        cov=covariance * 0.04,
        size=n_bootstrap,
    )
    bootstrap.loc[:, cont_features] = bootstrap.to_numpy() + noise

    combined_cont = pd.concat([kde_samples, bootstrap], ignore_index=True)
    lower = base_features[cont_features].quantile(0.001) - 0.60 * base_features[cont_features].std()
    upper = base_features[cont_features].quantile(0.999) + 0.60 * base_features[cont_features].std()
    combined_cont = combined_cont.clip(lower, upper, axis=1)

    one_hot_source = base_features[one_hot].iloc[RNG.integers(0, len(base_features), size=n_rows)].reset_index(drop=True)
    result = pd.concat([combined_cont, one_hot_source], axis=1)
    return result[selected_features].reset_index(drop=True)


def apply_regime(data: pd.DataFrame, regime_name: str, stats: Dict[str, pd.Series]) -> pd.DataFrame:
    """Apply telecom operating-regime shifts while preserving feature relationships."""
    result = data.copy()

    def shift_high(features: List[str], amount: float) -> None:
        for feature in features:
            if feature in result.columns:
                result[feature] = result[feature] + amount * stats["std"][feature]

    def shift_low(features: List[str], amount: float) -> None:
        for feature in features:
            if feature in result.columns:
                result[feature] = result[feature] - amount * stats["std"][feature]

    if regime_name == "High traffic urban network":
        shift_high(["active_users", "active_users_norm", "prb_utilization_pct", "prb_utilization_pct_norm"], 0.45)
        shift_high(["throughput_mbps", "throughput_mbps_norm"], 0.20)
        shift_low(["throughput_per_user", "throughput_per_user_norm"], 0.20)
    elif regime_name == "Congested network":
        shift_high(["active_users", "active_users_norm", "prb_utilization_pct", "prb_utilization_pct_norm"], 0.80)
        shift_high(["latency_ms", "latency_ms_norm", "packet_loss_pct", "packet_loss_pct_norm"], 0.35)
        shift_low(["throughput_per_user", "throughput_per_user_norm", "spectral_efficiency", "spectral_efficiency_norm"], 0.35)
    elif regime_name == "Rural weak-signal network":
        shift_low(["rsrp_dbm", "rsrp_dbm_norm", "rsrq_db", "rsrq_db_norm"], 0.70)
        shift_low(["active_users", "active_users_norm", "prb_utilization_pct", "prb_utilization_pct_norm"], 0.25)
        shift_high(["handover_count", "handover_count_norm"], 0.20)
    elif regime_name == "Extreme peak-hour network":
        shift_high(["active_users", "active_users_norm", "prb_utilization_pct", "prb_utilization_pct_norm"], 1.00)
        shift_high(["latency_ms", "latency_ms_norm"], 0.45)
        shift_low(["throughput_per_user", "throughput_per_user_norm"], 0.45)
        if "is_peak_hour" in result.columns:
            result["is_peak_hour"] = stats["p75"].get("is_peak_hour", 1.0)
    elif regime_name == "Slice-specific network behavior":
        shift_high(["latency_ms", "latency_ms_norm"], 0.15)
        shift_high(["throughput_mbps", "throughput_mbps_norm"], 0.25)
        if "slice_type_eMBB" in result.columns:
            result["slice_type_eMBB"] = 1.0
        if "slice_type_URLLC" in result.columns:
            mask = RNG.random(len(result)) < 0.25
            result.loc[mask, "slice_type_URLLC"] = 1.0
            result.loc[mask, "slice_type_eMBB"] = 0.0

    return result


def generate_normal_regimes(base_features: pd.DataFrame, n_rows: int, kde: KernelDensity) -> pd.DataFrame:
    """Create multiple OOD operating regimes without independent feature sampling."""
    selected_features = base_features.columns.tolist()
    cont_features = continuous_features(selected_features)
    stats = {
        "std": base_features[cont_features].std().replace(0, 1.0),
        "p75": base_features[cont_features].quantile(0.75),
    }
    weights = np.array([0.22, 0.18, 0.16, 0.16, 0.14, 0.14])
    counts = np.floor(weights * n_rows).astype(int)
    counts[-1] += n_rows - int(counts.sum())

    frames = []
    for regime_name, count in zip(REGIME_NAMES, counts):
        regime_data = sample_correlated_base(base_features, int(count), kde)
        regime_data = apply_regime(regime_data, regime_name, stats)
        regime_data["operating_regime"] = regime_name
        frames.append(regime_data)

    combined = pd.concat(frames, ignore_index=True)
    lower = base_features[cont_features].quantile(0.001) - 0.75 * base_features[cont_features].std()
    upper = base_features[cont_features].quantile(0.999) + 0.75 * base_features[cont_features].std()
    combined.loc[:, cont_features] = combined[cont_features].clip(lower, upper, axis=1)
    return combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def inject_family(
    anomalies: pd.DataFrame,
    rows: np.ndarray,
    family: str,
    base_features: pd.DataFrame,
    difficulty_scale: float,
) -> None:
    """Inject one anomaly family in place."""
    cont_features = continuous_features(base_features.columns)
    q01 = base_features[cont_features].quantile(0.01)
    q05 = base_features[cont_features].quantile(0.05)
    q95 = base_features[cont_features].quantile(0.95)
    q99 = base_features[cont_features].quantile(0.99)
    std = base_features[cont_features].std().replace(0, 1.0)

    groups = {
        "latency": ["latency_ms", "latency_ms_norm"],
        "packet_loss": ["packet_loss_pct", "packet_loss_pct_norm"],
        "throughput": ["throughput_mbps", "throughput_mbps_norm"],
        "rsrp": ["rsrp_dbm", "rsrp_dbm_norm"],
        "rsrq": ["rsrq_db", "rsrq_db_norm"],
        "prb": ["prb_utilization_pct", "prb_utilization_pct_norm"],
        "users": ["active_users", "active_users_norm"],
        "spectral": ["spectral_efficiency", "spectral_efficiency_norm"],
        "tpu": ["throughput_per_user", "throughput_per_user_norm"],
        "handover": ["handover_count", "handover_count_norm"],
    }

    def high(group: str, amount: float) -> None:
        for feature in groups[group]:
            if feature in anomalies.columns:
                anomalies.loc[rows, feature] = q99[feature] + amount * std[feature] * RNG.uniform(0.65, 1.35, len(rows))

    def low(group: str, amount: float) -> None:
        for feature in groups[group]:
            if feature in anomalies.columns:
                anomalies.loc[rows, feature] = q01[feature] - amount * std[feature] * RNG.uniform(0.65, 1.35, len(rows))

    def moderate_high(group: str, amount: float) -> None:
        for feature in groups[group]:
            if feature in anomalies.columns:
                anomalies.loc[rows, feature] = q95[feature] + amount * std[feature] * RNG.uniform(0.50, 1.10, len(rows))

    def moderate_low(group: str, amount: float) -> None:
        for feature in groups[group]:
            if feature in anomalies.columns:
                anomalies.loc[rows, feature] = q05[feature] - amount * std[feature] * RNG.uniform(0.50, 1.10, len(rows))

    if family == "Latency-only anomalies":
        high("latency", difficulty_scale)
    elif family == "Packet-loss anomalies":
        high("packet_loss", difficulty_scale)
    elif family == "Throughput-collapse anomalies":
        low("throughput", difficulty_scale)
        moderate_low("tpu", difficulty_scale * 0.35)
    elif family == "Signal-quality anomalies":
        low("rsrp", difficulty_scale)
        low("rsrq", difficulty_scale)
    elif family == "Congestion anomalies":
        high("prb", difficulty_scale)
        high("users", difficulty_scale)
        moderate_high("latency", difficulty_scale * 0.35)
        moderate_low("tpu", difficulty_scale * 0.35)
    elif family == "User-surge anomalies":
        high("users", difficulty_scale)
        moderate_high("prb", difficulty_scale * 0.45)
    elif family == "Combined multi-KPI anomalies":
        high("latency", difficulty_scale)
        high("packet_loss", difficulty_scale)
        high("prb", difficulty_scale * 0.75)
        low("throughput", difficulty_scale * 0.75)
        low("spectral", difficulty_scale * 0.75)
    elif family == "Subtle anomalies":
        moderate_high("latency", difficulty_scale * 0.40)
        moderate_high("packet_loss", difficulty_scale * 0.35)
        moderate_low("spectral", difficulty_scale * 0.35)
    elif family == "Contradictory anomalies":
        selectors = np.array_split(rows, 4)
        if len(selectors[0]):
            high("throughput", difficulty_scale)
            high("latency", difficulty_scale)
        if len(selectors[1]):
            low("prb", difficulty_scale * 0.70)
            high("packet_loss", difficulty_scale)
        if len(selectors[2]):
            high("rsrp", difficulty_scale)
            low("rsrq", difficulty_scale)
        if len(selectors[3]):
            low("users", difficulty_scale * 0.70)
            low("throughput", difficulty_scale)


def generate_anomalies(base_features: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Generate diverse anomaly families with easy, medium, and hard difficulty."""
    selected_features = base_features.columns.tolist()
    anomalies = sample_correlated_base(base_features, n_rows, fit_kde_sampler(base_features[continuous_features(selected_features)], 0.35))
    family_counts = np.full(len(ANOMALY_FAMILIES), n_rows // len(ANOMALY_FAMILIES), dtype=int)
    family_counts[-1] += n_rows - int(family_counts.sum())

    start = 0
    family_labels = []
    for family, count in zip(ANOMALY_FAMILIES, family_counts):
        rows = np.arange(start, start + int(count))
        thirds = np.array_split(rows, 3)
        inject_family(anomalies, thirds[0], family, base_features, 1.15)
        inject_family(anomalies, thirds[1], family, base_features, 0.80)
        inject_family(anomalies, thirds[2], family, base_features, 0.45)
        family_labels.extend([family] * int(count))
        start += int(count)

    cont_features = continuous_features(selected_features)
    lower = base_features[cont_features].quantile(0.001) - 1.00 * base_features[cont_features].std()
    upper = base_features[cont_features].quantile(0.999) + 1.00 * base_features[cont_features].std()
    anomalies.loc[:, cont_features] = anomalies[cont_features].clip(lower, upper, axis=1)
    anomalies["anomaly_family"] = family_labels
    anomalies["operating_regime"] = "Injected anomaly"
    return anomalies.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def build_stress_dataset(base_features: pd.DataFrame, anomaly_rate: float, n_rows: int = STRESS_SIZE) -> pd.DataFrame:
    """Build a full OOD stress-test dataset for one anomaly prevalence."""
    selected_features = base_features.columns.tolist()
    kde = fit_kde_sampler(base_features[continuous_features(selected_features)], 0.35)
    anomaly_count = int(round(n_rows * anomaly_rate))
    normal_count = n_rows - anomaly_count

    normal = generate_normal_regimes(base_features, normal_count, kde)
    normal["ground_truth_label"] = 0
    normal["anomaly_family"] = "Normal"

    anomalies = generate_anomalies(base_features, anomaly_count)
    anomalies["ground_truth_label"] = 1

    dataset = pd.concat([normal, anomalies], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_STATE + int(anomaly_rate * 10000)).reset_index(drop=True)
    return dataset[selected_features + ["ground_truth_label", "anomaly_family", "operating_regime"]]


def anomaly_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Higher score means more anomalous."""
    return -model.decision_function(features)


def adaptive_thresholds(scores: np.ndarray) -> Dict[str, float]:
    """Estimate anomaly thresholds from the incoming score distribution."""
    q25, q75 = np.percentile(scores, [25, 75])
    iqr = q75 - q25
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    return {
        "percentile_95": float(np.percentile(scores, 95)),
        "percentile_98": float(np.percentile(scores, 98)),
        "percentile_99": float(np.percentile(scores, 99)),
        "iqr_1_5": float(q75 + 1.5 * iqr),
        "iqr_3_0": float(q75 + 3.0 * iqr),
        "mad_3_5": float(median + 3.5 * 1.4826 * mad),
        "mad_5_0": float(median + 5.0 * 1.4826 * mad),
        "gaussian_2_5std": float(mean + 2.5 * std),
        "gaussian_3std": float(mean + 3.0 * std),
    }


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def evaluate_dataset(model, dataset: pd.DataFrame, selected_features: List[str], rate: float) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """Evaluate current model and adaptive threshold methods on one stress dataset."""
    x_values = dataset[selected_features]
    y_true = dataset["ground_truth_label"].astype(int).to_numpy()
    scores = anomaly_scores(model, x_values)
    model_predict = np.where(model.predict(x_values) == -1, 1, 0)

    rows = []
    base = metric_row(y_true, model_predict, scores)
    base.update({"anomaly_rate": rate, "threshold_method": "model_predict", "threshold": np.nan})
    rows.append(base)

    for method, threshold in adaptive_thresholds(scores).items():
        y_pred = (scores >= threshold).astype(int)
        row = metric_row(y_true, y_pred, scores)
        row.update({"anomaly_rate": rate, "threshold_method": method, "threshold": threshold})
        rows.append(row)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_STATE)
    gmm.fit(scores.reshape(-1, 1))
    anomaly_component = int(np.argmax(gmm.means_.ravel()))
    anomaly_probability = gmm.predict_proba(scores.reshape(-1, 1))[:, anomaly_component]
    for cutoff in [0.50, 0.80, 0.90, 0.95]:
        y_pred = (anomaly_probability >= cutoff).astype(int)
        row = metric_row(y_true, y_pred, scores)
        row.update(
            {
                "anomaly_rate": rate,
                "threshold_method": f"gmm_posterior_{cutoff:.2f}",
                "threshold": cutoff,
            }
        )
        rows.append(row)

    family_rows = []
    predicted = (anomaly_probability >= 0.90).astype(int)
    family_frame = pd.DataFrame(
        {
            "ground_truth_label": y_true,
            "predicted_label": predicted,
            "anomaly_family": dataset["anomaly_family"],
            "operating_regime": dataset["operating_regime"],
            "score": scores,
        }
    )
    for family, group in family_frame[family_frame["ground_truth_label"] == 1].groupby("anomaly_family"):
        family_rows.append(
            {
                "anomaly_rate": rate,
                "anomaly_family": family,
                "count": int(len(group)),
                "recall": float(recall_score(group["ground_truth_label"], group["predicted_label"], zero_division=0)),
                "mean_score": float(group["score"].mean()),
                "missed": int((group["predicted_label"] == 0).sum()),
            }
        )

    return rows, pd.DataFrame(family_rows)


def generate_all_stress_datasets(base_features: pd.DataFrame) -> Dict[float, Path]:
    """Generate and save all requested anomaly-rate datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for rate in ANOMALY_RATES:
        dataset = build_stress_dataset(base_features, rate, STRESS_SIZE)
        suffix = str(rate).replace(".", "p")
        path = DATA_DIR / f"stress_test_dataset_rate_{suffix}.csv"
        dataset.to_csv(path, index=False)
        paths[rate] = path
        if rate == 0.10:
            dataset.to_csv(STRESS_DATASET_PATH, index=False)
        print(f"Generated {path.name}: {dataset.shape[0]} rows, anomaly_rate={rate}", flush=True)
    return paths


def compare_threshold_methods(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Average threshold-method performance across anomaly rates."""
    return (
        evaluation.groupby("threshold_method", as_index=False)[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp"]
        ]
        .mean()
        .sort_values(["roc_auc", "recall", "precision", "f1"], ascending=False)
    )


def write_failure_analysis(family_metrics: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Write false-positive/false-negative and missed-family analysis."""
    recommended = evaluation[evaluation["threshold_method"] == "gmm_posterior_0.90"].copy()
    missed_families = family_metrics.sort_values(["recall", "missed"], ascending=[True, False]).head(12)
    fp_heavy = recommended.sort_values("fp", ascending=False).head(5)
    fn_heavy = recommended.sort_values("fn", ascending=False).head(5)

    lines = [
        "# Failure Analysis",
        "",
        "## Main Failure Modes",
        "",
        "- `model.predict()` assumes the fitted contamination operating point and under-adapts when anomaly prevalence moves far away from 5%.",
        "- Subtle anomalies sit close to high-load normal regimes, so their Isolation Forest scores overlap with valid congestion and peak-hour behavior.",
        "- Single-KPI anomalies such as latency-only or packet-loss-only events are harder than combined multi-KPI faults because Isolation Forest rewards consistency across the rest of the feature vector.",
        "- Contradictory anomalies are sometimes missed when one extreme KPI is offset by otherwise normal correlated KPIs.",
        "",
        "## Most Missed Anomaly Families Under GMM Adaptive Threshold",
        "",
        dataframe_to_markdown(missed_families),
        "",
        "## Highest False-Positive Stress Conditions",
        "",
        dataframe_to_markdown(fp_heavy[["anomaly_rate", "threshold_method", "fp", "precision", "recall", "f1"]]),
        "",
        "## Highest False-Negative Stress Conditions",
        "",
        dataframe_to_markdown(fn_heavy[["anomaly_rate", "threshold_method", "fn", "precision", "recall", "f1"]]),
        "",
        "## Why Anomalies Are Missed",
        "",
        "- Hard and subtle injected anomalies intentionally remain near the training manifold, so unsupervised isolation depth is not always short enough to flag them.",
        "- OOD normal regimes such as high traffic and extreme peak hour create naturally high PRB/user/latency scores, increasing score overlap with real congestion anomalies.",
        "- The detector was trained only on original normal-ish historical KPI patterns; it has no labeled boundary for anomaly families that are semantically bad but statistically close to rare normal behavior.",
    ]
    (REPORT_DIR / "failure_analysis.md").write_text("\n".join(lines))


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Small markdown table helper without optional dependencies."""
    if dataframe.empty:
        return "_No rows._"
    columns = dataframe.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in dataframe.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def draw_line_chart(path: Path, title: str, x_values: Iterable[float], series: Dict[str, Iterable[float]], x_label: str, y_label: str) -> None:
    """Draw a simple line chart without matplotlib."""
    image = Image.new("RGB", (980, 620), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 90, 80, 910, 520
    draw.text((380, 30), title, fill="black", font=font)
    draw.rectangle([left, top, right, bottom], outline="black")

    x_array = np.asarray(list(x_values), dtype=float)
    all_y = np.concatenate([np.asarray(list(values), dtype=float) for values in series.values()])
    xmin, xmax = float(x_array.min()), float(x_array.max())
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    if xmax == xmin:
        xmax += 1.0
    if ymax == ymin:
        ymax += 1.0

    colors = [(44, 115, 180), (215, 85, 65), (70, 155, 95), (150, 95, 180), (220, 150, 55)]
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = []
        for x_value, y_value in zip(x_array, values):
            px = left + (x_value - xmin) / (xmax - xmin) * (right - left)
            py = bottom - (y_value - ymin) / (ymax - ymin) * (bottom - top)
            points.append((px, py))
        for start, end in zip(points, points[1:]):
            draw.line([start, end], fill=color, width=3)
        for px, py in points:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.rectangle([700, 95 + index * 24, 720, 110 + index * 24], fill=color)
        draw.text((730, 93 + index * 24), name, fill="black", font=font)

    draw.text((420, 570), x_label, fill="black", font=font)
    draw.text((20, 300), y_label, fill="black", font=font)
    draw.text((left, bottom + 12), f"{xmin:.3f}", fill="black", font=font)
    draw.text((right - 45, bottom + 12), f"{xmax:.3f}", fill="black", font=font)
    draw.text((left - 65, bottom - 5), f"{ymin:.3f}", fill="black", font=font)
    draw.text((left - 65, top - 5), f"{ymax:.3f}", fill="black", font=font)
    image.save(path)


def draw_family_bar_chart(path: Path, family_metrics: pd.DataFrame) -> None:
    """Draw family recall bars for the recommended threshold."""
    summary = family_metrics.groupby("anomaly_family", as_index=False)["recall"].mean().sort_values("recall")
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 300, 70, 1030, 580
    draw.text((455, 25), "Anomaly Family Performance", fill="black", font=font)
    draw.rectangle([left, top, right, bottom], outline="black")
    bar_height = max(18, int((bottom - top) / max(len(summary), 1)) - 8)
    for index, (_, row) in enumerate(summary.iterrows()):
        y = top + index * (bar_height + 8) + 5
        width = int(float(row["recall"]) * (right - left))
        draw.text((20, y + 4), str(row["anomaly_family"])[:42], fill="black", font=font)
        draw.rectangle([left, y, left + width, y + bar_height], fill=(70, 140, 210))
        draw.text((left + width + 8, y + 4), f"{row['recall']:.3f}", fill="black", font=font)
    draw.text((620, 610), "Mean recall", fill="black", font=font)
    image.save(path)


def draw_score_distribution(path: Path, score_samples: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Draw score distribution comparison for low and high anomaly rates."""
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((390, 25), "Score Distribution Comparison", fill="black", font=font)
    colors = [(70, 145, 215), (220, 95, 75), (75, 165, 100), (155, 95, 180)]
    panels = [(70, 85, 470, 560), (550, 85, 950, 560)]
    for panel_index, (label, (normal_scores, anomaly_scores)) in enumerate(score_samples.items()):
        left, top, right, bottom = panels[panel_index]
        draw.rectangle([left, top, right, bottom], outline="black")
        draw.text((left + 120, top - 30), label, fill="black", font=font)
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        bins = np.linspace(all_scores.min(), all_scores.max(), 45)
        normal_hist, _ = np.histogram(normal_scores, bins=bins, density=True)
        anomaly_hist, _ = np.histogram(anomaly_scores, bins=bins, density=True)
        max_hist = max(normal_hist.max(), anomaly_hist.max(), 1e-9)
        for idx in range(len(bins) - 1):
            x0 = left + int((idx / (len(bins) - 1)) * (right - left))
            x1 = left + int(((idx + 1) / (len(bins) - 1)) * (right - left))
            normal_h = int((normal_hist[idx] / max_hist) * (bottom - top))
            anomaly_h = int((anomaly_hist[idx] / max_hist) * (bottom - top))
            draw.rectangle([x0, bottom - normal_h, x1, bottom], fill=colors[0])
            draw.rectangle([x0, bottom - anomaly_h, x1, bottom], fill=colors[1])
    draw.rectangle([720, 600, 740, 615], fill=colors[0])
    draw.text((750, 598), "Normal", fill="black", font=font)
    draw.rectangle([820, 600, 840, 615], fill=colors[1])
    draw.text((850, 598), "Anomaly", fill="black", font=font)
    image.save(path)


def generate_visualizations(evaluation: pd.DataFrame, family_metrics: pd.DataFrame, score_samples: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Generate all requested robustness plots."""
    recommended = evaluation[evaluation["threshold_method"] == "gmm_posterior_0.90"].sort_values("anomaly_rate")
    rates = recommended["anomaly_rate"].to_numpy()
    draw_line_chart(REPORT_DIR / "anomaly_rate_vs_f1.png", "Anomaly Rate vs F1", rates, {"GMM 0.90": recommended["f1"]}, "Anomaly rate", "F1")
    draw_line_chart(REPORT_DIR / "anomaly_rate_vs_recall.png", "Anomaly Rate vs Recall", rates, {"GMM 0.90": recommended["recall"]}, "Anomaly rate", "Recall")
    draw_line_chart(REPORT_DIR / "anomaly_rate_vs_precision.png", "Anomaly Rate vs Precision", rates, {"GMM 0.90": recommended["precision"]}, "Anomaly rate", "Precision")
    draw_family_bar_chart(REPORT_DIR / "anomaly_family_performance.png", family_metrics)
    draw_score_distribution(REPORT_DIR / "score_distribution_comparison.png", score_samples)


def write_robustness_report(evaluation: pd.DataFrame, family_metrics: pd.DataFrame, threshold_comparison: pd.DataFrame) -> None:
    """Write the final robustness report."""
    recommended = evaluation[evaluation["threshold_method"] == "gmm_posterior_0.90"].sort_values("anomaly_rate")
    model_predict = evaluation[evaluation["threshold_method"] == "model_predict"].sort_values("anomaly_rate")
    family_summary = family_metrics.groupby("anomaly_family", as_index=False)[["recall", "missed"]].mean().sort_values("recall")

    lines = [
        "# Robustness Report",
        "",
        "## Objective",
        "",
        "Stress-test the existing Isolation Forest on out-of-distribution telecom operating regimes, unknown anomaly distributions, and anomaly rates from 0.5% to 40%.",
        "",
        "## Performance At Every Anomaly Rate",
        "",
        "### Recommended Adaptive Threshold: GMM Posterior 0.90",
        "",
        dataframe_to_markdown(recommended[["anomaly_rate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp"]]),
        "",
        "### Baseline `model.predict()`",
        "",
        dataframe_to_markdown(model_predict[["anomaly_rate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp"]]),
        "",
        "## Threshold Method Comparison",
        "",
        dataframe_to_markdown(threshold_comparison),
        "",
        "## Performance Per Anomaly Family",
        "",
        dataframe_to_markdown(family_summary),
        "",
        "## Threshold Recommendations",
        "",
        "- Do not use a fixed `contamination=0.05` inference assumption for future CSV uploads.",
        "- Score incoming records with `anomaly_score = -model.decision_function(X)`.",
        "- Use GMM posterior thresholding as the default high-recall adaptive strategy because it detects when the incoming score distribution becomes bimodal.",
        "- Use MAD/IQR thresholds as conservative high-precision alert-budget controls, not as the primary detector under high anomaly prevalence.",
        "- Use percentile thresholds only when operations explicitly wants a bounded alert budget.",
        "- Monitor PR AUC and family-level recall when anomaly prevalence is expected to be high.",
        "",
        "## Robustness Weaknesses",
        "",
        "- Subtle anomalies remain the hardest family because they intentionally stay close to the learned normal manifold.",
        "- Contradictory anomalies can be missed when the unusual KPI combination is not isolated enough in the full 32-feature space.",
        "- Very high anomaly prevalence compresses score-distribution separation, which makes all unsupervised adaptive thresholds less stable.",
        "",
        "## Recommended Production Configuration",
        "",
        "Keep Isolation Forest as the production algorithm, but replace fixed `model.predict()` inference with adaptive threshold selection over `-decision_function(X)`.",
        "",
        "```python",
        "scores = -model.decision_function(X)",
        "gmm = GaussianMixture(n_components=2, random_state=42).fit(scores.reshape(-1, 1))",
        "anomaly_component = np.argmax(gmm.means_.ravel())",
        "anomaly_probability = gmm.predict_proba(scores.reshape(-1, 1))[:, anomaly_component]",
        "predicted_label = (anomaly_probability >= 0.90).astype(int)",
        "```",
    ]
    (REPORT_DIR / "robustness_report.md").write_text("\n".join(lines))


def run_framework() -> None:
    """Run the full advanced OOD validation framework."""
    selected_features = load_selected_features()
    base_features = load_processed_training_features(selected_features)
    model = joblib.load(MODEL_PATH)

    generate_kpi_profile(base_features)
    dataset_paths = generate_all_stress_datasets(base_features)

    evaluation_rows = []
    family_frames = []
    score_samples = {}

    for rate, path in dataset_paths.items():
        dataset = pd.read_csv(path)
        rows, family_metrics = evaluate_dataset(model, dataset, selected_features, rate)
        evaluation_rows.extend(rows)
        family_frames.append(family_metrics)

        if rate in (0.005, 0.40):
            scores = anomaly_scores(model, dataset[selected_features])
            y_true = dataset["ground_truth_label"].astype(int).to_numpy()
            score_samples[f"{rate:.1%} anomalies"] = (scores[y_true == 0], scores[y_true == 1])

        print(f"Evaluated anomaly_rate={rate}", flush=True)

    evaluation = pd.DataFrame(evaluation_rows)
    family_metrics = pd.concat(family_frames, ignore_index=True)
    threshold_comparison = compare_threshold_methods(evaluation)

    evaluation.to_csv(REPORT_DIR / "stress_test_evaluation_metrics.csv", index=False)
    family_metrics.to_csv(REPORT_DIR / "anomaly_family_metrics.csv", index=False)
    threshold_comparison.to_csv(REPORT_DIR / "adaptive_threshold_comparison.csv", index=False)

    write_failure_analysis(family_metrics, evaluation)
    write_robustness_report(evaluation, family_metrics, threshold_comparison)
    generate_visualizations(evaluation, family_metrics, score_samples)

    recommended = evaluation[evaluation["threshold_method"] == "gmm_posterior_0.90"]
    baseline = evaluation[evaluation["threshold_method"] == "model_predict"]
    print("==============================")
    print("ADVANCED ROBUSTNESS VALIDATION")
    print("==============================")
    print(f"Stress datasets generated: {len(dataset_paths)}")
    print(f"Rows per dataset: {STRESS_SIZE}")
    print("")
    print("Baseline model.predict average metrics:")
    print(f"Precision: {baseline['precision'].mean():.4f}")
    print(f"Recall: {baseline['recall'].mean():.4f}")
    print(f"F1: {baseline['f1'].mean():.4f}")
    print(f"ROC AUC: {baseline['roc_auc'].mean():.4f}")
    print(f"PR AUC: {baseline['pr_auc'].mean():.4f}")
    print("")
    print("Adaptive GMM 0.90 average metrics:")
    print(f"Precision: {recommended['precision'].mean():.4f}")
    print(f"Recall: {recommended['recall'].mean():.4f}")
    print(f"F1: {recommended['f1'].mean():.4f}")
    print(f"ROC AUC: {recommended['roc_auc'].mean():.4f}")
    print(f"PR AUC: {recommended['pr_auc'].mean():.4f}")


if __name__ == "__main__":
    run_framework()
