"""
Adapter for converting RCA CSV rows into the standard format expected by downstream agents.
"""

from typing import Any


def convert_rca_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one RCA CSV row into the project's standard RCA schema.
    """

    return {
        "incident_id": row["incident_id"],
        "timestamp": row["timestamp"],

        "cell_id": row["cell_id/site_id"],

        "network_slice": row["network_slice"],

        "sla_compliant": row["sla_compliant"],

        "root_cause": row["root_cause"],

        "severity": row["severity"],

        "confidence": row["confidence_score"],

        "affected_kpis": row["affected_kpis"],
        "matched_signals": row["matched_signals"],
        "score_breakdown": row["score_breakdown"],
        "explanation": row["explanation"],
    }