def build_prompt(anomaly_features, fault_interpretation, retrieved_logs):
    """
    Builds a grounded prompt for the maintenance decision copilot
    """

    evidence_text = ""
    for i, log in enumerate(retrieved_logs, start=1):
        evidence_text += (
            f"\nIncident {i}:\n"
            f"- Thermal pattern: {log['thermal_pattern']}\n"
            f"- Failure mode: {log['failure_mode']}\n"
            f"- Action taken: {log['action_taken']}\n"
            f"- Downtime hours: {log['downtime_hours']}\n"
            f"- Repair cost USD: {log['repair_cost_usd']}\n"
        )

    prompt = f"""
You are an industrial maintenance decision-support copilot.
You must rely ONLY on the provided anomaly data and historical incidents.
Do NOT invent facts. If uncertain, state uncertainty.

Thermal anomaly features:
{anomaly_features}

Initial engineering interpretation:
{fault_interpretation}

Retrieved historical maintenance incidents:
{evidence_text}

TASK:
1. Identify the most likely failure mode
2. Explain reasoning clearly (2–3 sentences)
3. Recommend safe next inspection or maintenance actions
4. Estimate downtime range (hours)
5. Estimate repair cost range (USD)
6. Provide confidence level (0–1)

Respond in structured bullet points.
"""
    return prompt
