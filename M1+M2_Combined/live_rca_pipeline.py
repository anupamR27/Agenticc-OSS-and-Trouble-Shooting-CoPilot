import argparse
import importlib.util
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("Member 1/member1/models/isolation_forest_production.pkl")
REFERENCE_DATA_PATH = Path("Member 1/member1/data/ds1_processed.csv")
ANOMALY_OUTPUT_PATH = Path("Member 1/member1/outputs/anomaly_scores_production.csv")
RCA_SCRIPT = Path("Member 2/rca_poc.py")

RCA_FIELDS = [
    "root_cause",
    "severity",
    "confidence_score",
    "affected_kpis",
    "matched_signals",
    "explanation",
    "recommended_actions",
]

SAMPLE_INPUT = {
    "throughput_mbps": 35.0,
    "latency_ms": 95.0,
    "packet_loss_pct": 4.8,
    "handover_count": 14,
    "rsrp_dbm": -116.0,
    "rsrq_db": -18.5,
    "prb_utilization_pct": 91.0,
    "active_users": 240,
}

KPI_INPUT_FIELDS = [
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "handover_count",
    "rsrp_dbm",
    "rsrq_db",
    "prb_utilization_pct",
    "active_users",
]


def load_rca_module():
    if not RCA_SCRIPT.exists():
        raise FileNotFoundError(f"Member 2 RCA script not found: {RCA_SCRIPT}")
    spec = importlib.util.spec_from_file_location("member2_rca_poc", RCA_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_bundle(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Anomaly model artifact not found: {model_path}. "
            "Run Member 1 training/export first so isolation_forest_production.pkl is available."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "forests" not in bundle:
        raise ValueError("Production model must be a bundle containing 'forests'.")
    if "production_features" not in bundle:
        raise ValueError("Production model bundle is missing 'production_features'.")
    return bundle


def load_reference_data(reference_path=REFERENCE_DATA_PATH):
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference processed KPI data not found: {reference_path}. "
            "It is required for live encoding, normalization, and feature statistics."
        )
    return pd.read_csv(reference_path)


def infer_sla_compliant(user_input):
    throughput = pd.to_numeric(user_input.get("throughput_mbps"), errors="coerce")
    latency = pd.to_numeric(user_input.get("latency_ms"), errors="coerce")
    packet_loss = pd.to_numeric(user_input.get("packet_loss_pct"), errors="coerce")
    prb = pd.to_numeric(user_input.get("prb_utilization_pct"), errors="coerce")
    rsrp = pd.to_numeric(user_input.get("rsrp_dbm"), errors="coerce")
    rsrq = pd.to_numeric(user_input.get("rsrq_db"), errors="coerce")

    issue_like = (
        (pd.notna(throughput) and throughput < 50)
        or (pd.notna(latency) and latency > 80)
        or (pd.notna(packet_loss) and packet_loss > 3)
        or (pd.notna(prb) and prb > 90)
        or (pd.notna(rsrp) and rsrp < -115)
        or (pd.notna(rsrq) and rsrq < -18)
    )
    return 0 if issue_like else 1


def enrich_kpi_input(user_input):
    if not isinstance(user_input, dict):
        raise ValueError("Input must be one JSON object.")

    missing = [field for field in KPI_INPUT_FIELDS if field not in user_input]
    if missing:
        raise ValueError(f"Missing required KPI field(s): {', '.join(missing)}")

    enriched = dict(user_input)
    enriched["timestamp"] = enriched.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enriched["cell_id"] = enriched.get("cell_id") or "LIVE_INPUT"
    enriched["cell_type"] = enriched.get("cell_type") or "unknown"
    enriched["slice_type"] = enriched.get("slice_type") or "unknown"
    enriched["network_slice"] = enriched.get("network_slice") or enriched["slice_type"]
    enriched["sla_compliant"] = enriched.get("sla_compliant", infer_sla_compliant(enriched))
    return enriched


def add_basic_features(df):
    result = df.copy()
    if "network_slice" not in result.columns and "slice_type" in result.columns:
        result["network_slice"] = result["slice_type"]

    if "timestamp" in result.columns:
        timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
        result["hour_of_day"] = timestamp.dt.hour.fillna(0).astype(int)
        result["day_of_week"] = timestamp.dt.dayofweek.fillna(0).astype(int)
        result["is_peak_hour"] = result["hour_of_day"].between(18, 22).astype(int)

    if {"throughput_mbps", "active_users"}.issubset(result.columns) and "throughput_per_user" not in result.columns:
        result["throughput_per_user"] = result["throughput_mbps"] / (result["active_users"].replace(0, np.nan))
    if {"throughput_mbps", "prb_utilization_pct"}.issubset(result.columns) and "spectral_efficiency" not in result.columns:
        result["spectral_efficiency"] = result["throughput_mbps"] / (result["prb_utilization_pct"].replace(0, np.nan))

    return result.replace([np.inf, -np.inf], np.nan)


def add_encoding_and_norm_features(df, reference_df):
    result = df.copy()

    for column in ["cell_type", "slice_type"]:
        enc_column = f"{column}_enc"
        if enc_column in reference_df.columns and column in result.columns and enc_column not in result.columns:
            mapping = reference_df[[column, enc_column]].dropna().drop_duplicates(column).set_index(column)[enc_column].to_dict()
            fallback = float(reference_df[enc_column].median())
            result[enc_column] = result[column].map(mapping).fillna(fallback)

    for feature in ["cell_type_macro", "cell_type_micro", "cell_type_pico"]:
        if feature not in result.columns:
            value = feature.replace("cell_type_", "")
            result[feature] = (result.get("cell_type", "") == value).astype(int)

    for feature in ["slice_type_HC", "slice_type_URLLC", "slice_type_eMBB"]:
        if feature not in result.columns:
            value = feature.replace("slice_type_", "")
            result[feature] = (result.get("slice_type", "") == value).astype(int)

    raw_for_norm = [
        "throughput_mbps",
        "latency_ms",
        "packet_loss_pct",
        "handover_count",
        "rsrp_dbm",
        "rsrq_db",
        "prb_utilization_pct",
        "active_users",
        "throughput_per_user",
        "spectral_efficiency",
    ]
    for column in raw_for_norm:
        norm_column = f"{column}_norm"
        if norm_column in reference_df.columns and column in result.columns and norm_column not in result.columns:
            source = pd.to_numeric(reference_df[column], errors="coerce")
            min_value = source.min()
            max_value = source.max()
            if pd.notna(min_value) and pd.notna(max_value) and max_value != min_value:
                result[norm_column] = (pd.to_numeric(result[column], errors="coerce") - min_value) / (max_value - min_value)

    return result


def safe_ratio(numerator, denominator):
    return numerator / (np.abs(denominator) + 1e-3)


def add_engineered_features(df, reference_df):
    result = df.copy()
    pairs = {
        "latency_x_packet_loss": ("latency_ms", "packet_loss_pct", "mul"),
        "latency_per_throughput": ("latency_ms", "throughput_mbps", "ratio"),
        "throughput_per_prb": ("throughput_mbps", "prb_utilization_pct", "ratio"),
        "active_users_x_utilization": ("active_users", "prb_utilization_pct", "mul"),
        "rsrp_x_rsrq": ("rsrp_dbm", "rsrq_db", "mul"),
        "spectral_efficiency_x_utilization": ("spectral_efficiency", "prb_utilization_pct", "mul"),
        "throughput_per_user_x_latency": ("throughput_per_user", "latency_ms", "mul"),
        "packet_loss_per_signal_quality": ("packet_loss_pct", "rsrq_db", "ratio"),
        "handover_per_signal_strength": ("handover_count", "rsrp_dbm", "ratio"),
        "latency_x_utilization": ("latency_ms", "prb_utilization_pct", "mul"),
        "loss_x_handover": ("packet_loss_pct", "handover_count", "mul"),
        "users_per_throughput": ("active_users", "throughput_mbps", "ratio"),
        "utilization_per_spectral_efficiency": ("prb_utilization_pct", "spectral_efficiency", "ratio"),
    }
    for name, (left, right, op) in pairs.items():
        if left in result.columns and right in result.columns:
            if op == "mul":
                result[name] = result[left] * result[right]
            else:
                result[name] = safe_ratio(result[left], result[right])

    if {"latency_ms", "packet_loss_pct", "throughput_mbps"}.issubset(result.columns):
        result["service_degradation_index"] = result["latency_ms"] + result["packet_loss_pct"] - result["throughput_mbps"]
    if {"active_users", "prb_utilization_pct", "throughput_per_user"}.issubset(result.columns):
        result["congestion_pressure_index"] = result["active_users"] + result["prb_utilization_pct"] - result["throughput_per_user"]
    if {"rsrp_dbm", "rsrq_db", "spectral_efficiency"}.issubset(result.columns):
        result["radio_quality_index"] = result["rsrp_dbm"] + result["rsrq_db"] + result["spectral_efficiency"]
    if {"is_peak_hour", "active_users", "prb_utilization_pct"}.issubset(result.columns):
        result["peak_load_pressure"] = result["is_peak_hour"] * (result["active_users"] + result["prb_utilization_pct"])
    if {"slice_type_URLLC", "latency_ms", "packet_loss_pct"}.issubset(result.columns):
        result["urllc_risk_pressure"] = result["slice_type_URLLC"] * (result["latency_ms"] + result["packet_loss_pct"])
    if {"slice_type_eMBB", "throughput_mbps", "prb_utilization_pct"}.issubset(result.columns):
        result["embb_capacity_pressure"] = result["slice_type_eMBB"] * safe_ratio(result["prb_utilization_pct"], result["throughput_mbps"])

    contextual_features = [
        "latency_ms",
        "packet_loss_pct",
        "throughput_mbps",
        "prb_utilization_pct",
        "active_users",
        "throughput_per_user",
        "spectral_efficiency",
        "rsrp_dbm",
        "rsrq_db",
    ]
    for feature in contextual_features:
        if feature in result.columns and feature in reference_df.columns:
            reference = pd.to_numeric(reference_df[feature], errors="coerce")
            median = reference.median()
            iqr = reference.quantile(0.75) - reference.quantile(0.25)
            mean = reference.mean()
            std = reference.std()
            result[f"{feature}_robust_context_z"] = (pd.to_numeric(result[feature], errors="coerce") - median) / (iqr + 1e-3)
            result[f"{feature}_batch_z"] = (pd.to_numeric(result[feature], errors="coerce") - mean) / (std + 1e-3)

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_live_features(user_input, bundle, reference_df):
    enriched_input = enrich_kpi_input(user_input)
    row = pd.DataFrame([enriched_input])
    row = add_basic_features(row)
    row = add_encoding_and_norm_features(row, reference_df)
    row = add_engineered_features(row, reference_df)

    missing = [feature for feature in bundle["production_features"] if feature not in row.columns]
    if missing:
        raise ValueError(
            "Live input could not be transformed into the model feature set. "
            f"Missing features: {', '.join(missing)}"
        )

    feature_frame = row[bundle["production_features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return row, feature_frame


def forest_score_frame(forests, feature_frame):
    scores = {}
    for name, model in forests.items():
        scores[name] = -model.decision_function(feature_frame)
    return pd.DataFrame(scores)


def live_anomaly_score(bundle, feature_frame):
    score_frame = forest_score_frame(bundle["forests"], feature_frame)
    return float(score_frame.mean(axis=1).iloc[0])


def anomaly_threshold(bundle, reference_df):
    if not ANOMALY_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly score calibration file not found: {ANOMALY_OUTPUT_PATH}. "
            "It is required to convert live model scores into an anomaly decision."
        )
    anomaly_output = pd.read_csv(ANOMALY_OUTPUT_PATH)
    if "is_anomaly" not in anomaly_output.columns:
        raise ValueError("Anomaly output is missing is_anomaly for live threshold calibration.")

    reference_rows = add_basic_features(reference_df)
    reference_rows = add_encoding_and_norm_features(reference_rows, reference_df)
    reference_rows = add_engineered_features(reference_rows, reference_df)
    missing = [feature for feature in bundle["production_features"] if feature not in reference_rows.columns]
    if missing:
        raise ValueError(
            "Reference data could not be transformed into the model feature set. "
            f"Missing features: {', '.join(missing)}"
        )

    reference_features = reference_rows[bundle["production_features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw_scores = forest_score_frame(bundle["forests"], reference_features).mean(axis=1)
    anomaly_mask = anomaly_output["is_anomaly"].astype(int) == 1
    if len(raw_scores) != len(anomaly_output):
        raise ValueError("Reference data and anomaly output row counts differ; cannot calibrate live threshold.")
    if anomaly_mask.sum() == 0:
        return float(raw_scores.quantile(0.90))
    return float(raw_scores[anomaly_mask].min())


def run_rca(row, reference_df):
    rca = load_rca_module()
    rca_input = row.copy()
    if "network_slice" not in rca_input.columns and "slice_type" in rca_input.columns:
        rca_input["network_slice"] = rca_input["slice_type"]
    rca_input["sla_compliant"] = 0

    reference_for_thresholds = reference_df.copy()
    if "network_slice" not in reference_for_thresholds.columns and "slice_type" in reference_for_thresholds.columns:
        reference_for_thresholds["network_slice"] = reference_for_thresholds["slice_type"]

    thresholds = rca.compute_thresholds(reference_for_thresholds, None)["__global__"]
    live_row = rca_input.iloc[0]
    root_cause, signals, affected, scores = rca.classify_row(live_row, thresholds)

    top_score = 0
    if root_cause not in ["Normal", "Generic SLA Degradation / Needs Investigation"]:
        for score_item in scores.split("; "):
            category, score = score_item.rsplit("=", 1)
            if category == root_cause:
                top_score = int(score)
                break

    return {
        "root_cause": root_cause,
        "severity": rca.SEVERITY_BY_ROOT_CAUSE[root_cause],
        "confidence_score": rca.confidence_score(root_cause, top_score),
        "affected_kpis": ", ".join(affected),
        "matched_signals": " | ".join(signals),
        "explanation": rca.build_explanation(root_cause, signals),
        "recommended_actions": " | ".join(rca.RECOMMENDED_ACTIONS[root_cause]),
    }


def predict_live_rca(user_input):
    bundle = load_model_bundle()
    reference_df = load_reference_data()
    row, feature_frame = prepare_live_features(user_input, bundle, reference_df)

    raw_score = live_anomaly_score(bundle, feature_frame)
    threshold = anomaly_threshold(bundle, reference_df)
    is_anomaly = raw_score >= threshold

    if not is_anomaly:
        return {
            "anomaly_status": "Normal",
            "is_anomaly": False,
            "anomaly_score": raw_score,
            "root_cause": None,
            "message": "No RCA required because KPI values are not anomalous.",
        }

    result = {
        "anomaly_status": "Anomaly",
        "is_anomaly": True,
        "anomaly_score": raw_score,
    }
    result.update(run_rca(row, reference_df))
    return result


def read_json_from_terminal():
    print("Paste KPI values as JSON.")
    print("Do not include timestamp, cell_id, cell_type, slice_type, record_id, network_slice, or sla_compliant.")
    print("Press Enter twice when done.")
    print()
    print("Example:")
    print(json.dumps(SAMPLE_INPUT, indent=2))
    print()
    print("Paste JSON below:")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip() and lines:
            break
        lines.append(line)

    raw = "\n".join(lines).strip()
    if not raw:
        raise ValueError("No JSON input received.")

    try:
        user_input = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc}") from exc

    if not isinstance(user_input, dict):
        raise ValueError("Input JSON must be one object, not a list or plain value.")
    return user_input


def parse_args():
    parser = argparse.ArgumentParser(description="Live single-input Telecom OSS anomaly + RCA pipeline")
    parser.add_argument("--sample", action="store_true", help="Run the built-in sample input instead of prompting")
    parser.add_argument("--json", help="Pass one KPI JSON object directly as a command-line string")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.sample:
            user_input = SAMPLE_INPUT
        elif args.json:
            user_input = json.loads(args.json)
            if not isinstance(user_input, dict):
                raise ValueError("--json must contain one JSON object.")
        else:
            user_input = read_json_from_terminal()

        result = predict_live_rca(user_input)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
