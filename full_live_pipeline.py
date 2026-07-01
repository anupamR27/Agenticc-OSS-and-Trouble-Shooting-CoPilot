import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from final_report_formatter import format_final_report, save_outputs
import live_rca_pipeline as member1_live


PROJECT_ROOT = Path(__file__).resolve().parent
MEMBER2_DIR = PROJECT_ROOT / "Member 2"
MEMBER3_DIR = PROJECT_ROOT / "Member 3"
MEMBER3_VECTOR_DB_DIR = MEMBER3_DIR / "vector_db"
MEMBER3_ENV_PATH = MEMBER3_DIR / ".env"


SAMPLE_INPUT = {
    "throughput_mbps": 22.0,
    "latency_ms": 118.0,
    "packet_loss_pct": 3.8,
    "handover_count": 9,
    "rsrp_dbm": -86.0,
    "rsrq_db": -10.0,
    "prb_utilization_pct": 97.0,
    "active_users": 315,
}


def add_import_paths() -> None:
    """Make project folders with spaces importable without changing existing files."""
    for path in (PROJECT_ROOT, MEMBER2_DIR, MEMBER3_DIR):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def load_json_input() -> dict[str, Any]:
    print("Paste KPI JSON:")
    print(json.dumps(SAMPLE_INPUT, indent=2))
    print()
    print("Press Enter twice when done.")

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
        raise ValueError("No KPI JSON input received.")

    user_input = json.loads(raw)
    if not isinstance(user_input, dict):
        raise ValueError("KPI input must be one JSON object.")
    return user_input


def parse_score_breakdown(score_breakdown: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    if not score_breakdown:
        return scores

    for item in str(score_breakdown).split("; "):
        if "=" not in item:
            continue
        category, score = item.rsplit("=", 1)
        try:
            scores[category] = int(score)
        except ValueError:
            scores[category] = 0
    return scores


def split_pipe_text(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def split_comma_text(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def run_member1_anomaly_detection(
    user_input: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    try:
        bundle = member1_live.load_model_bundle()
        reference_df = member1_live.load_reference_data()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Member 1 model/reference artifact missing: {exc}") from exc

    row, feature_frame = member1_live.prepare_live_features(user_input, bundle, reference_df)
    raw_score = member1_live.live_anomaly_score(bundle, feature_frame)
    threshold = member1_live.anomaly_threshold(bundle, reference_df)

    is_anomaly = raw_score >= threshold
    if int(row["sla_compliant"].iloc[0]) == 1:
        is_anomaly = False

    anomaly_result = {
        "anomaly_status": "Anomaly" if is_anomaly else "Normal",
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": raw_score,
    }
    return anomaly_result, row.iloc[0].to_dict(), reference_df


def run_member2_rca(
    enriched_input: dict[str, Any],
    anomaly_result: dict[str, Any],
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not anomaly_result.get("is_anomaly"):
        raise ValueError("Member 2 RCA should only run for anomalous inputs.")

    try:
        rca = importlib.import_module("rca_poc")
    except Exception as exc:
        raise ImportError(f"Member 2 RCA import failed: {exc}") from exc

    rca_input = pd.DataFrame([enriched_input])
    if "network_slice" not in rca_input.columns and "slice_type" in rca_input.columns:
        rca_input["network_slice"] = rca_input["slice_type"]
    rca_input["sla_compliant"] = 0

    threshold_source = reference_df.copy() if reference_df is not None else rca_input.copy()
    if "network_slice" not in threshold_source.columns and "slice_type" in threshold_source.columns:
        threshold_source["network_slice"] = threshold_source["slice_type"]

    thresholds = rca.compute_thresholds(threshold_source, None)["__global__"]
    live_row = rca_input.iloc[0]
    root_cause, signals, affected, score_text = rca.classify_row(live_row, thresholds)

    top_score = 0
    if root_cause not in ("Normal", "Generic SLA Degradation / Needs Investigation"):
        top_score = parse_score_breakdown(score_text).get(root_cause, 0)

    incident_id = f"LIVE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "incident_id": incident_id,
        "timestamp": enriched_input.get("timestamp"),
        "cell_id/site_id": enriched_input.get("cell_id", "LIVE_INPUT"),
        "cell_type": enriched_input.get("cell_type", "unknown"),
        "network_slice": enriched_input.get("network_slice", enriched_input.get("slice_type", "unknown")),
        "sla_compliant": 0,
        "root_cause": root_cause,
        "severity": rca.SEVERITY_BY_ROOT_CAUSE[root_cause],
        "confidence_score": rca.confidence_score(root_cause, top_score),
        "affected_kpis": ", ".join(affected),
        "matched_signals": " | ".join(signals),
        "score_breakdown": score_text,
        "explanation": rca.build_explanation(root_cause, signals),
        "recommended_actions": " | ".join(rca.RECOMMENDED_ACTIONS[root_cause]),
    }


def live_rca_to_member3_schema(rca_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "incident_id": rca_result["incident_id"],
        "timestamp": rca_result["timestamp"],
        "cell_id": rca_result["cell_id/site_id"],
        "cell_type": rca_result.get("cell_type", "unknown"),
        "slice_type": rca_result.get("network_slice", "unknown"),
        "sla_compliant": rca_result.get("sla_compliant", 0),
        "root_cause": rca_result["root_cause"],
        "severity": rca_result["severity"],
        "confidence": rca_result["confidence_score"],
        "affected_kpis": rca_result.get("affected_kpis", ""),
        "matched_signals": rca_result.get("matched_signals", ""),
        "score_breakdown": rca_result.get("score_breakdown", ""),
        "explanation": rca_result.get("explanation", ""),
    }


def ensure_member3_ready() -> None:
    index_files = (MEMBER3_VECTOR_DB_DIR / "index.faiss", MEMBER3_VECTOR_DB_DIR / "index.pkl")
    missing_index = [path.name for path in index_files if not path.exists()]
    if missing_index:
        raise FileNotFoundError(
            "Member 3 RAG index missing "
            f"({', '.join(missing_index)}). Run: cd 'Member 3' && python3 -m rag.build_index"
        )

    load_dotenv(MEMBER3_ENV_PATH)
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY missing. Add it to Member 3/.env as GROQ_API_KEY=your_key_here."
        )


def run_member3_reporting(rca_result: dict[str, Any]) -> dict[str, Any]:
    member3_result: dict[str, Any] = {
        "operator_recommendations": [],
        "retrieved_context": [],
        "incident_report": None,
    }

    member3_rca = live_rca_to_member3_schema(rca_result)

    try:
        recommendation_module = importlib.import_module("agents.recommendation_agent")
        recommendation = recommendation_module.get_recommendations(member3_rca)
        member3_result["operator_recommendations"] = recommendation["recommendations"]
    except Exception as exc:
        member3_result["recommendation_error"] = f"Member 3 recommendation agent failed: {exc}"
        recommendation = {
            "recommendations": split_pipe_text(rca_result.get("recommended_actions")),
        }

    retrieval_query = f"""
    Root Cause: {member3_rca["root_cause"]}
    Severity: {member3_rca["severity"]}
    Affected KPIs: {member3_rca["affected_kpis"]}
    RCA Explanation: {member3_rca["explanation"]}
    """.strip()

    try:
        ensure_member3_ready()
    except Exception as exc:
        member3_result["member3_error"] = str(exc)
        return member3_result

    try:
        retriever_module = importlib.import_module("rag.retriever")
        member3_result["retrieved_context"] = retriever_module.retrieve_context(retrieval_query)
    except Exception as exc:
        member3_result["rag_error"] = f"Member 3 RAG retrieval failed: {exc}"

    try:
        llm_module = importlib.import_module("agents.llm_agent")
        member3_result["incident_report"] = llm_module.generate_report(
            member3_rca,
            recommendation["recommendations"],
        )
    except Exception as exc:
        member3_result["llm_error"] = f"Member 3 LLM call failed: {exc}"

    return member3_result


def run_full_pipeline(user_input: dict[str, Any]) -> dict[str, Any]:
    anomaly_result, enriched_input, reference_df = run_member1_anomaly_detection(user_input)

    if not anomaly_result["is_anomaly"]:
        return {
            "pipeline_status": "completed",
            "input_kpis": user_input,
            "member_1_anomaly_detection": anomaly_result,
            "anomaly_status": "Normal",
            "is_anomaly": False,
            "root_cause": None,
            "message": "No RCA or report required because KPI values are not anomalous.",
        }

    rca_result = run_member2_rca(enriched_input, anomaly_result, reference_df)
    member3_result = run_member3_reporting(rca_result)

    return {
        "pipeline_status": "completed",
        "input_kpis": user_input,
        "member_1_anomaly_detection": anomaly_result,
        "member_2_rca": {
            "root_cause": rca_result["root_cause"],
            "severity": rca_result["severity"],
            "confidence_score": rca_result["confidence_score"],
            "affected_kpis": split_comma_text(rca_result["affected_kpis"]),
            "matched_signals": split_pipe_text(rca_result["matched_signals"]),
            "score_breakdown": parse_score_breakdown(rca_result["score_breakdown"]),
            "explanation": rca_result["explanation"],
            "recommended_actions": split_pipe_text(rca_result.get("recommended_actions")),
        },
        "member_3_recommendation_and_report": member3_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full live Telecom OSS anomaly + RCA + RAG report pipeline")
    parser.add_argument("--sample", action="store_true", help="Run a built-in anomalous sample input")
    parser.add_argument("--json", help="Pass one KPI JSON object directly as a command-line string")
    parser.add_argument("--debug", action="store_true", help="Include raw retrieval chunks and debug JSON in the report")
    return parser.parse_args()


def main() -> None:
    add_import_paths()
    args = parse_args()

    try:
        if args.sample:
            user_input = SAMPLE_INPUT
        elif args.json:
            user_input = json.loads(args.json)
            if not isinstance(user_input, dict):
                raise ValueError("--json must contain one JSON object.")
        else:
            user_input = load_json_input()

        result = run_full_pipeline(user_input)
        report_text = format_final_report(result, debug=args.debug)
        save_outputs(result, report_text)
        print(report_text)
    except Exception as exc:
        error_result = {
            "pipeline_status": "failed",
            "error": str(exc),
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
