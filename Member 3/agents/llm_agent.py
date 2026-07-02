import os

from dotenv import load_dotenv
from groq import Groq

from rag.retriever import retrieve_context

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_report(rca_result: dict, recommendations: list[str]) -> str:
    """
    Generate a professional telecom incident report using an LLM.

    Args:
        rca_result: RCA output dictionary.
        recommendations: Rule-based operator recommendations.

    Returns:
        Formatted incident report.
    """

    actions = "\n".join(f"- {action}" for action in recommendations)

    retrieval_query = f"""
    Root Cause: {rca_result["root_cause"]}
    Severity: {rca_result["severity"]}
    Network Slice: {rca_result["network_slice"]}
    Affected KPIs: {rca_result["affected_kpis"]}
    RCA Explanation: {rca_result["explanation"]}
    """.strip()
    try:
        retrieved_chunks = retrieve_context(retrieval_query)

        supporting_documentation = (
            "\n\n--------------------------------\n\n".join(
                (
                    f"Source: {chunk['source_pdf']}\n"
                    f"Page: {chunk['page_number'] or 'Unknown'}\n\n"
                    f"{chunk['text']}"
                )
                for chunk in retrieved_chunks
            )
            or "No relevant supporting documentation was retrieved."
        )
    except Exception as e:
        print(f"Warning: RAG retrieval failed: {e}")
        supporting_documentation = "Supporting telecom documentation could not be retrieved."
        
    prompt = f"""
    
You are a Senior Telecom Network Operations Engineer.

Generate a concise operator incident report.

INCIDENT DETAILS
----------------
Cell ID: {rca_result["cell_id"]}
Severity: {rca_result["severity"]}
Root Cause: {rca_result["root_cause"]}
Confidence: {rca_result["confidence"]:.0%}

Network Slice: 
{rca_result["network_slice"]}

Affected KPIs:
{rca_result["affected_kpis"]}

RCA Explanation:
{rca_result["explanation"]}

RULE-BASED OPERATOR RECOMMENDATIONS
-----------------------------------
{actions}

SUPPORTING TELECOM DOCUMENTATION
--------------------------------
{supporting_documentation}

INSTRUCTIONS
------------
- Treat the RCA output as the identified incident.
- Use the supporting telecom documentation as technical reference material. 
    When the retrieved documentation contains deployment-specific guidance
    (e.g., network slice), prefer those recommendations over
    generic recommendations where appropriate.
- Treat the rule-based recommendations as the minimum required actions. 
    Augment them using relevant retrieved SOP guidance, especially network-slice-specific 
    operational actions. Organize the final recommendations into Immediate Actions, 
    Slice-Specific Actions (if applicable), and Long-Term Actions. Avoid repeating identical actions.
- Do not invent troubleshooting procedures that are unsupported by the
  supporting documentation. The supplied rule-based recommendations remain
  valid immediate actions even when the documentation does not repeat them.
- Mention the source PDF and page number when referencing retrieved
  documentation where appropriate.
- Keep the report under 200 words.
- Use professional telecom terminology.
- Do not repeat information.
- Produce a concise, professional telecom incident report.

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
                    "You are an senior telecom network operations engineer who writes "
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
