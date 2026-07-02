"""Rule-based recommendation agent for telecom incident RCA outputs."""

from pprint import pprint
from typing import Any

ROOT_CAUSE_MAPPING = {
    "Suspected Cell Outage / Severe Service Drop":
        "Suspected Cell Outage",

    "Radio Coverage Degradation":
        "Radio Coverage",

    "Backhaul / Transport Issue":
        "Backhaul / Transport Issue",

    "Interference / Poor Signal Quality":
        "Interference / Poor Signal Quality",

    "Mobility / Handover Issue":
        "Mobility / Handover Issue",

    "RAN Congestion":
        "RAN Congestion",
}


GENERIC_RECOMMENDATIONS: list[str] = [
    "Review affected KPI trends",
    "Verify recent configuration changes",
    "Check neighboring cell performance",
    "Escalate to Network Operations if the issue persists",
]


RECOMMENDATION_MAP: dict[str, list[str]] = {

    "Backhaul / Transport Issue": [
        "Check transport link utilization",
        "Verify backhaul capacity and throughput",
        "Inspect transmission links and interfaces",
        "Review transport alarms and packet drops",
        "Monitor latency and congestion trends",
    ],

    "Interference / Poor Signal Quality": [
        "Review RSRP and RSRQ trends",
        "Analyze uplink and downlink interference levels",
        "Check neighboring cell PCI/frequency conflicts",
        "Inspect antenna alignment and transmit power",
        "Perform spectrum analysis if interference persists",
    ],

    "Mobility / Handover Issue": [
        "Review handover success and failure statistics",
        "Verify neighbor relation configuration",
        "Check handover parameter settings",
        "Investigate ping-pong handovers",
        "Monitor mobility KPIs after optimization",
    ],

    "Normal": [
        "No corrective action required",
        "Continue routine KPI monitoring",
        "Record observation for trend analysis",
    ],

    "Radio Coverage": [
        "Review RSRP coverage distribution",
        "Verify antenna tilt and azimuth configuration",
        "Inspect potential physical obstructions",
        "Check site transmit power settings",
        "Validate coverage using drive-test or MDT data",
    ],

    "RAN Congestion": [
        "Review PRB utilization trends",
        "Check active user distribution",
        "Enable or verify load balancing",
        "Review traffic steering configuration",
        "Evaluate capacity expansion if congestion persists",
    ],

    "Suspected Cell Outage": [
        "Verify cell availability status",
        "Check power and hardware alarms",
        "Inspect radio unit and baseband connectivity",
        "Verify transmission link status",
        "Escalate to field maintenance if the cell remains unavailable",
    ],
}


def get_recommendations(rca_result: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic operator recommendations for an RCA result.

    Args:
        rca_result: RCA output containing at least ``cell_id`` and
            ``root_cause`` fields.

    Returns:
        A dictionary containing the cell ID, root cause, and recommended
        operator actions. Unknown root causes return generic troubleshooting
        actions.
    """
    cell_id = str(rca_result.get("cell_id", "UNKNOWN_CELL"))
    root_cause = rca_result["root_cause"]
    normalized_root_cause = ROOT_CAUSE_MAPPING.get(
        root_cause, root_cause
    )

    recommendations = RECOMMENDATION_MAP.get(
        normalized_root_cause,
        GENERIC_RECOMMENDATIONS
    )

    return {
        "cell_id": cell_id,
        "root_cause": normalized_root_cause,
        "recommendations": recommendations,
    }
