import re
from guardrails.output_schema import MaintenanceDecision


def apply_guardrails(llm_text: str) -> MaintenanceDecision:
    """
    Parses LLM output and applies safety + confidence rules
    """

    # --- VERY SIMPLE PARSING (hackathon-safe) ---
    failure_mode = "bearing_overheating"
    if "bearing_wear" in llm_text.lower():
        failure_mode = "bearing_wear"

    confidence_match = re.search(r"confidence.*?([0-9.]+)", llm_text.lower())
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5

    # --- CONFIDENCE GATING ---
    if confidence < 0.6:
        recommended_action = "Manual inspection required before any maintenance decision."
    else:
        recommended_action = (
            "Inspect bearing condition and lubrication. "
            "Prepare for bearing replacement if abnormal wear is confirmed."
        )

    decision = MaintenanceDecision(
        failure_mode=failure_mode,
        reasoning=(
            "Localized high-temperature anomaly and historical maintenance "
            "records indicate bearing-related degradation."
        ),
        recommended_action=recommended_action,
        downtime_hours_min=2,
        downtime_hours_max=6,
        repair_cost_usd_min=300,
        repair_cost_usd_max=1200,
        confidence=round(confidence, 2)
    )

    return decision
