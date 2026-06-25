import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_report(rca_result: dict, recommendations: list[str]) -> str:
    """
    Generate a professional telecom incident report using an LLM.

    Args:
        rca_result: RCA output dictionary.
        recommendations: Retrieved operator recommendations.

    Returns:
        Formatted incident report.
    """

    actions = "\n".join(f"- {action}" for action in recommendations)

    prompt = f"""
You are a Senior Telecom Network Operations Engineer.

Generate a concise operator incident report.

Incident Details
----------------
Cell ID: {rca_result["cell_id"]}
Severity: {rca_result["severity"]}
Root Cause: {rca_result["root_cause"]}
Confidence: {rca_result["confidence"]:.0%}

Affected KPIs:
{rca_result["affected_kpis"]}

RCA Explanation:
{rca_result["explanation"]}

Retrieved Operator Actions:
{actions}

Rules:
- Use ONLY the supplied operator actions.
- Do NOT invent troubleshooting steps.
- Keep the report under 200 words.
- Use professional telecom terminology.
- Do not repeat information.

Output Format:

🚨 INCIDENT SUMMARY

🔍 ROOT CAUSE ASSESSMENT

⚡ RECOMMENDED ACTIONS

📈 EXPECTED IMPACT
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Ericsson telecom operations engineer who writes "
                    "clear and professional incident reports."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content