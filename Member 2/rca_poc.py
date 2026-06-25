import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "dataset1_5g_network_kpi.csv"
DEFAULT_OUTPUT = "rca_results.csv"


SEVERITY_BY_ROOT_CAUSE = {
    "Normal": "Normal",
    "Suspected Cell Outage / Severe Service Drop": "Critical",
    "RAN Congestion": "High",
    "Backhaul / Transport Issue": "High",
    "Radio Coverage Degradation": "High",
    "Interference / Poor Signal Quality": "Medium",
    "Mobility / Handover Issue": "Medium",
    "Generic SLA Degradation / Needs Investigation": "Low",
}


RECOMMENDED_ACTIONS = {
    "Normal": ["No action required"],
    "RAN Congestion": [
        "Check PRB utilization trend",
        "Verify active user count",
        "Check neighboring cell load",
        "Consider load balancing or capacity optimization",
    ],
    "Suspected Cell Outage / Severe Service Drop": [
        "Check cell availability alarms",
        "Verify site power/backhaul status",
        "Check recent configuration changes",
        "Escalate to field operations team",
    ],
    "Backhaul / Transport Issue": [
        "Check transport link health",
        "Verify latency and packet loss on backhaul path",
        "Check router/switch interface errors",
        "Escalate to transport team",
    ],
    "Radio Coverage Degradation": [
        "Check RSRP trend",
        "Inspect antenna or site coverage issues",
        "Verify antenna tilt, power, and configuration",
        "Check coverage complaints if available",
    ],
    "Interference / Poor Signal Quality": [
        "Check RSRQ/SINR degradation",
        "Check neighboring cell interference",
        "Verify frequency planning",
        "Investigate external interference sources",
    ],
    "Mobility / Handover Issue": [
        "Check handover counters",
        "Review neighbor relation configuration",
        "Check mobility parameter changes",
        "Analyze source-target handover patterns",
    ],
    "Generic SLA Degradation / Needs Investigation": [
        "Review related KPIs",
        "Check alarms and logs",
        "Continue investigation",
    ],
}

RCA_SCORE_CATEGORIES = [
    "RAN Congestion",
    "Suspected Cell Outage / Severe Service Drop",
    "Backhaul / Transport Issue",
    "Radio Coverage Degradation",
    "Interference / Poor Signal Quality",
    "Mobility / Handover Issue",
]

TIE_BREAK_PRIORITY = [
    "Suspected Cell Outage / Severe Service Drop",
    "RAN Congestion",
    "Backhaul / Transport Issue",
    "Radio Coverage Degradation",
    "Interference / Poor Signal Quality",
    "Mobility / Handover Issue",
]


def percentile(series, value):
    """Return a percentile threshold, or NaN if the KPI is not usable."""
    if series is None:
        return np.nan
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(np.percentile(numeric, value))


def find_first_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

# -----------------------------

def compute_thresholds(df, slice_col):
    """Calculate RCA thresholds per slice when available, otherwise globally."""
    def group_thresholds(group):
        high_prb = percentile(group.get("prb_utilization_pct"), 80)
        very_high_prb = percentile(group.get("prb_utilization_pct"), 90)
        high_prb_threshold = min(85.0, high_prb) if pd.notna(high_prb) else 85.0
        very_high_prb_threshold = min(90.0, very_high_prb) if pd.notna(very_high_prb) else 90.0

        return {
            "very_low_throughput_threshold": percentile(group.get("throughput_mbps"), 5),
            "low_throughput_threshold": percentile(group.get("throughput_mbps"), 20),
            "high_latency_threshold": percentile(group.get("latency_ms"), 80),
            "very_high_latency_threshold": percentile(group.get("latency_ms"), 95),
            "high_packet_loss_threshold": percentile(group.get("packet_loss_pct"), 80),
            "very_high_packet_loss_threshold": percentile(group.get("packet_loss_pct"), 95),
            "high_prb_threshold": high_prb_threshold,
            "very_high_prb_threshold": very_high_prb_threshold,
            "high_users_threshold": percentile(group.get("active_users"), 75),
            "low_users_threshold": percentile(group.get("active_users"), 20),
            "high_handover_threshold": percentile(group.get("handover_count"), 85),
            "very_high_handover_threshold": percentile(group.get("handover_count"), 95),
            "weak_rsrp_threshold": -110.0,
            "poor_rsrq_threshold": -15.0,
            "very_poor_rsrq_threshold": -18.0,
        }


    if slice_col:
        thresholds = {}
        for slice_value, group in df.groupby(slice_col, dropna=False):
            thresholds[slice_value] = group_thresholds(group)
        return thresholds

    return {"__global__": group_thresholds(df)}


def get_value(row, column):
    if not column or column not in row.index:
        return np.nan
    return row[column]


def has_values(row, columns):
    return all(column in row.index and pd.notna(row[column]) for column in columns)


def is_true(condition):
    return bool(condition) if pd.notna(condition) else False


def is_sla_compliant(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return pd.notna(numeric_value) and int(numeric_value) == 1


def add_signal(signals, label, value=None):
    if value is None or pd.isna(value):
        signals.append(label)
    else:
        signals.append(f"{label} ({value})")


def confidence_score(root_cause, top_score=0):
    if root_cause == "Normal":
        return 0.95
    if root_cause == "Generic SLA Degradation / Needs Investigation":
        return 0.50
    if top_score >= 7:
        return 0.90
    if top_score >= 5:
        return 0.80
    if top_score == 4:
        return 0.70
    if top_score == 3:
        return 0.60
    return 0.50


def build_explanation(root_cause, signals):
    if root_cause == "Normal":
        return "SLA is compliant, so no RCA issue was detected."
    if root_cause == "Generic SLA Degradation / Needs Investigation":
        return (
            "The row violates SLA, but the KPI pattern does not strongly match congestion, "
            "outage, backhaul, radio, interference, or mobility rules."
        )
    if signals:
        return f"The row is classified as {root_cause} because " + ", ".join(signals) + "."
    return f"The row is classified as {root_cause} based on the strongest KPI score."


def threshold_breached(row, column, operator, threshold):
    if column not in row.index or pd.isna(row[column]) or pd.isna(threshold):
        return False
    if operator == "<=":
        return row[column] <= threshold
    if operator == ">=":
        return row[column] >= threshold
    if operator == "<":
        return row[column] < threshold
    if operator == ">":
        return row[column] > threshold
    return False


def add_score(scores, score_signals, category, points, label, value=None):
    scores[category] += points
    add_signal(score_signals[category], label, value)


def score_breakdown(scores):
    return "; ".join(f"{category}={scores[category]}" for category in RCA_SCORE_CATEGORIES)


def affected_kpis(row, thresholds):
    affected = []
    checks = [
        ("throughput_mbps", "<=", thresholds["low_throughput_threshold"]),
        ("latency_ms", ">=", thresholds["high_latency_threshold"]),
        ("packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]),
        ("prb_utilization_pct", ">=", thresholds["high_prb_threshold"]),
        ("active_users", ">=", thresholds["high_users_threshold"]),
        ("active_users", "<=", thresholds["low_users_threshold"]),
        ("rsrp_dbm", "<=", thresholds["weak_rsrp_threshold"]),
        ("rsrq_db", "<=", thresholds["poor_rsrq_threshold"]),
        ("handover_count", ">=", thresholds["high_handover_threshold"]),
    ]
    for column, operator, threshold in checks:
        if column not in affected and threshold_breached(row, column, operator, threshold):
            affected.append(column)
    return affected


def is_issue_row(row, thresholds):
    if "sla_compliant" in row.index and pd.notna(row["sla_compliant"]):
        return not is_sla_compliant(row["sla_compliant"])
    return bool(affected_kpis(row, thresholds))


def classify_row(row, thresholds):
    sla = get_value(row, "sla_compliant")

    if is_sla_compliant(sla) or not is_issue_row(row, thresholds):
        return "Normal", [], [], score_breakdown(dict.fromkeys(RCA_SCORE_CATEGORIES, 0))

    scores = {category: 0 for category in RCA_SCORE_CATEGORIES}
    score_signals = {category: [] for category in RCA_SCORE_CATEGORIES}

    if threshold_breached(row, "prb_utilization_pct", ">=", thresholds["very_high_prb_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 3, "very high PRB utilization", row["prb_utilization_pct"])
    elif threshold_breached(row, "prb_utilization_pct", ">=", thresholds["high_prb_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 2, "high PRB utilization", row["prb_utilization_pct"])
    if threshold_breached(row, "active_users", ">=", thresholds["high_users_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 2, "high active users", row["active_users"])
    if threshold_breached(row, "throughput_mbps", "<=", thresholds["low_throughput_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 2, "low throughput", row["throughput_mbps"])
    if threshold_breached(row, "latency_ms", ">=", thresholds["high_latency_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 1, "high latency", row["latency_ms"])
    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "RAN Congestion", 1, "high packet loss", row["packet_loss_pct"])

    if threshold_breached(row, "throughput_mbps", "<=", thresholds["very_low_throughput_threshold"]):
        add_score(scores, score_signals, "Suspected Cell Outage / Severe Service Drop", 3, "very low throughput", row["throughput_mbps"])
    if threshold_breached(row, "active_users", "<=", thresholds["low_users_threshold"]):
        add_score(scores, score_signals, "Suspected Cell Outage / Severe Service Drop", 2, "low active users", row["active_users"])
    if threshold_breached(row, "prb_utilization_pct", "<", 50):
        add_score(scores, score_signals, "Suspected Cell Outage / Severe Service Drop", 2, "low PRB utilization", row["prb_utilization_pct"])
    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Suspected Cell Outage / Severe Service Drop", 1, "high packet loss", row["packet_loss_pct"])
    if threshold_breached(row, "latency_ms", ">=", thresholds["high_latency_threshold"]):
        add_score(scores, score_signals, "Suspected Cell Outage / Severe Service Drop", 1, "high latency", row["latency_ms"])

    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["very_high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 3, "very high packet loss", row["packet_loss_pct"])
    elif threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 2, "high packet loss", row["packet_loss_pct"])
    if threshold_breached(row, "latency_ms", ">=", thresholds["very_high_latency_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 3, "very high latency", row["latency_ms"])
    elif threshold_breached(row, "latency_ms", ">=", thresholds["high_latency_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 2, "high latency", row["latency_ms"])
    if threshold_breached(row, "prb_utilization_pct", "<", thresholds["high_prb_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 2, "PRB below congestion threshold", row["prb_utilization_pct"])
    if threshold_breached(row, "throughput_mbps", "<=", thresholds["low_throughput_threshold"]):
        add_score(scores, score_signals, "Backhaul / Transport Issue", 1, "low throughput", row["throughput_mbps"])

    if threshold_breached(row, "rsrp_dbm", "<=", -115):
        add_score(scores, score_signals, "Radio Coverage Degradation", 4, "very weak RSRP", row["rsrp_dbm"])
    elif threshold_breached(row, "rsrp_dbm", "<=", thresholds["weak_rsrp_threshold"]):
        add_score(scores, score_signals, "Radio Coverage Degradation", 3, "weak RSRP", row["rsrp_dbm"])
    if threshold_breached(row, "throughput_mbps", "<=", thresholds["low_throughput_threshold"]):
        add_score(scores, score_signals, "Radio Coverage Degradation", 2, "low throughput", row["throughput_mbps"])
    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Radio Coverage Degradation", 1, "high packet loss", row["packet_loss_pct"])
    if threshold_breached(row, "rsrq_db", "<=", thresholds["poor_rsrq_threshold"]):
        add_score(scores, score_signals, "Radio Coverage Degradation", 1, "poor RSRQ", row["rsrq_db"])

    if threshold_breached(row, "rsrq_db", "<=", thresholds["very_poor_rsrq_threshold"]):
        add_score(scores, score_signals, "Interference / Poor Signal Quality", 4, "very poor RSRQ", row["rsrq_db"])
    elif threshold_breached(row, "rsrq_db", "<=", thresholds["poor_rsrq_threshold"]):
        add_score(scores, score_signals, "Interference / Poor Signal Quality", 3, "poor RSRQ", row["rsrq_db"])
    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Interference / Poor Signal Quality", 2, "high packet loss", row["packet_loss_pct"])
    if threshold_breached(row, "latency_ms", ">=", thresholds["high_latency_threshold"]):
        add_score(scores, score_signals, "Interference / Poor Signal Quality", 1, "high latency", row["latency_ms"])
    if threshold_breached(row, "rsrp_dbm", ">", thresholds["weak_rsrp_threshold"]):
        add_score(scores, score_signals, "Interference / Poor Signal Quality", 1, "RSRP not severely weak", row["rsrp_dbm"])

# RSRP -> signal kitna strong he
# RSRQ -> how clean is teh signal, noise and etc all
# user is within the range b ut quality is damaged

    if threshold_breached(row, "handover_count", ">=", thresholds["very_high_handover_threshold"]):
        add_score(scores, score_signals, "Mobility / Handover Issue", 4, "very high handover count", row["handover_count"])
    elif threshold_breached(row, "handover_count", ">=", thresholds["high_handover_threshold"]):
        add_score(scores, score_signals, "Mobility / Handover Issue", 3, "high handover count", row["handover_count"])
    if threshold_breached(row, "packet_loss_pct", ">=", thresholds["high_packet_loss_threshold"]):
        add_score(scores, score_signals, "Mobility / Handover Issue", 2, "high packet loss", row["packet_loss_pct"])
    if threshold_breached(row, "latency_ms", ">=", thresholds["high_latency_threshold"]):
        add_score(scores, score_signals, "Mobility / Handover Issue", 1, "high latency", row["latency_ms"])
    if threshold_breached(row, "throughput_mbps", "<=", thresholds["low_throughput_threshold"]):
        add_score(scores, score_signals, "Mobility / Handover Issue", 1, "low throughput", row["throughput_mbps"])

    top_score = max(scores.values())
    top_category = next(category for category in TIE_BREAK_PRIORITY if scores[category] == top_score)
    if top_score < 3:
        return (
            "Generic SLA Degradation / Needs Investigation",
            [],
            affected_kpis(row, thresholds),
            score_breakdown(scores),
        )

    return top_category, score_signals[top_category], affected_kpis(row, thresholds), score_breakdown(scores)


def build_rca_results(df):
    slice_col = find_first_column(df, ["network_slice", "slice_type"])
    site_col = find_first_column(df, ["cell_id", "site_id"])
    thresholds_by_slice = compute_thresholds(df, slice_col)
    global_thresholds = compute_thresholds(df, None)["__global__"]

    results = []
    for index, row in df.iterrows():
        slice_value = get_value(row, slice_col)
        thresholds = thresholds_by_slice.get(slice_value, global_thresholds) if slice_col else global_thresholds
        root_cause, signals, affected, scores = classify_row(row, thresholds)
        top_score = 0
        if root_cause not in ["Normal", "Generic SLA Degradation / Needs Investigation"]:
            for score_item in scores.split("; "):
                category, score = score_item.rsplit("=", 1)
                if category == root_cause:
                    top_score = int(score)
                    break

        result = {
            "incident_id": f"INC-{index + 1:06d}",
            "timestamp": get_value(row, "timestamp"),
            "cell_id/site_id": get_value(row, site_col),
            "cell_type": row.get("cell_type", "Unknown"),
            "network_slice": slice_value,
            "sla_compliant": get_value(row, "sla_compliant"),
            "root_cause": root_cause,
            "severity": SEVERITY_BY_ROOT_CAUSE[root_cause],
            "confidence_score": confidence_score(root_cause, top_score),
            "affected_kpis": ", ".join([k for k in affected if k in df.columns]),
            "matched_signals": " | ".join(signals),
            "score_breakdown": scores,
            "explanation": build_explanation(root_cause, signals),
            "recommended_actions": " | ".join(RECOMMENDED_ACTIONS[root_cause]),
        }
        results.append(result)

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic-OSS RCA POC")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input telecom KPI CSV file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output RCA results CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    results = build_rca_results(df)
    results.to_csv(output_path, index=False)

    issue_results = results[results["root_cause"] != "Normal"]
    generic_count = (
        results["root_cause"] == "Generic SLA Degradation / Needs Investigation"
    ).sum()
    specific_count = len(issue_results) - generic_count
    coverage_pct = (specific_count / len(issue_results) * 100) if len(issue_results) else 0.0

    print(f"Loaded dataset: {input_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print()
    print(f"Total rows: {len(results)}")
    print(f"Normal rows: {(results['root_cause'] == 'Normal').sum()}")
    print(f"Total anomalies: {len(issue_results)}")
    print(f"Specific RCA classified rows: {specific_count}")
    print(f"Generic RCA rows: {generic_count}")
    print(f"RCA coverage percentage: {coverage_pct:.1f}%")
    print()
    print("RCA category distribution:")
    print(results["root_cause"].value_counts().to_string())
    print()
    print("Severity distribution:")
    print(results["severity"].value_counts().to_string())
    print()
    print("First 10 anomaly RCA results:")
    display_cols = [
        "incident_id",
        "timestamp",
        "cell_id/site_id",
        "cell_type",
        "network_slice",
        "root_cause",
        "severity",
        "confidence_score",
        "affected_kpis",
        "matched_signals",
        "score_breakdown",
    ]
    if issue_results.empty:
        print("No issue rows detected.")
    else:
        print(issue_results[display_cols].head(10).to_string(index=False))
    print()
    print(f"Saved RCA results to: {output_path}")


if __name__ == "__main__":
    main()
