import pandas as pd

from parser.data_adapter import convert_rca_row
from agents.recommendation_agent import get_recommendations
from agents.llm_agent import generate_report

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "input" / "rca_results.csv")

print("\nAvailable Incidents (Non-Normal):\n")

incidents = df[df["root_cause"] != "Normal"].reset_index(drop=True)

display_incidents = incidents[
    ["incident_id", "cell_id/site_id", "root_cause", "severity"]
].rename(columns={"cell_id/site_id": "cell_id"})

print(display_incidents)

choice = int(input("\nChoose incident number (0-{}): ".format(len(incidents)-1)))

# Validation
if choice < 0 or choice >= len(incidents):
    print("Invalid incident number.")
    exit()

row = incidents.iloc[choice]

rca_result = convert_rca_row(row.to_dict())

recommendation = get_recommendations(rca_result)

report = generate_report(
    rca_result,
    recommendation["recommendations"]
)

print("="*60)
print("        TELECOM INCIDENT COPILOT")
print("="*60)

print(f"Incident ID : {rca_result['incident_id']}")
print(f"Cell ID     : {recommendation['cell_id']}")
print(f"Severity    : {rca_result['severity']}")
print(f"Confidence  : {rca_result['confidence']:.0%}")

print("\n")

print(report)

print("="*60)