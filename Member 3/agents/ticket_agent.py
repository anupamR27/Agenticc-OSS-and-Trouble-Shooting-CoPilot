"""Incident ticket generator for the Telecom Incident Copilot MVP."""

from datetime import datetime
from typing import Any


PRIORITY_MAP: dict[str, str] = {
    "Hardware Failure": "Critical",
    "Power Issue": "Critical",
    "Backhaul Congestion": "High",
    "Radio Resource Congestion": "High",
    "Interference": "Medium",
    "Coverage Issue": "Medium",
    "Unknown": "Low",
}

_ticket_sequence = 0


def _next_ticket_id() -> str:
    """Return the next sequential incident ticket ID."""
    global _ticket_sequence
    _ticket_sequence += 1
    return f"INC-{_ticket_sequence:03d}"


def generate_ticket(data: dict[str, Any]) -> dict[str, Any]:
    """Convert recommendation output into a structured incident ticket.

    Args:
        data: Recommendation output containing cell, root cause, and actions.

    Returns:
        A structured incident ticket with a sequential ID, status, priority,
        affected cell, root cause, recommended actions, and timestamp.
    """
    root_cause = str(data.get("root_cause", "Unknown"))
    affected_cell = str(data.get("cell_id", data.get("affected_cell", "UNKNOWN_CELL")))
    recommended_actions = data.get("recommendations", data.get("recommended_actions", []))

    if not isinstance(recommended_actions, list):
        recommended_actions = [str(recommended_actions)]

    return {
        "ticket_id": _next_ticket_id(),
        "status": "OPEN",
        "priority": PRIORITY_MAP.get(root_cause, "Low"),
        "affected_cell": affected_cell,
        "root_cause": root_cause,
        "recommended_actions": recommended_actions,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _print_ticket(ticket: dict[str, Any]) -> None:
    """Print a clean, readable incident ticket report."""
    print("=" * 50)
    print("Incident Ticket")
    print("=" * 15)
    print()
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Status: {ticket['status']}")
    print(f"Priority: {ticket['priority']}")
    print(f"Affected Cell: {ticket['affected_cell']}")
    print(f"Root Cause: {ticket['root_cause']}")
    print(f"Timestamp: {ticket['timestamp']}")
    print()
    print("Recommended Actions:")
    print()

    actions = ticket.get("recommended_actions", [])
    if actions:
        for index, action in enumerate(actions, start=1):
            print(f"{index}. {action}")
    else:
        print("No recommended actions provided.")

    print()


if __name__ == "__main__":
    sample_incidents: list[dict[str, Any]] = [
        {
            "cell_id": "CELL_102",
            "root_cause": "Backhaul Congestion",
            "recommendations": [
                "Check neighboring cell load",
                "Review traffic balancing configuration",
                "Verify transport capacity",
            ],
        },
        {
            "cell_id": "CELL_204",
            "root_cause": "Coverage Issue",
            "recommendations": [
                "Verify antenna tilt and azimuth settings",
                "Review recent coverage complaints and drive test data",
                "Evaluate neighbor relation configuration",
            ],
        },
        {
            "cell_id": "CELL_318",
            "root_cause": "Hardware Failure",
            "recommendations": [
                "Check active hardware alarms",
                "Inspect radio unit and baseband health status",
                "Dispatch field maintenance if alarms persist",
            ],
        },
    ]

    print("=" * 50)
    print("Ticket Agent Demo")
    print("=" * 17)
    print()

    for incident in sample_incidents:
        generated_ticket = generate_ticket(incident)
        _print_ticket(generated_ticket)
