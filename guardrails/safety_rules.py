import re
from guardrails.output_schema import MaintenanceDecision


def _safe_float(value, default=0.5):
    try:
        value = float(value)
        if 0.0 <= value <= 1.0:
            return value
    except Exception:
        pass
    return default


def apply_guardrails(llm_text: str) -> MaintenanceDecision:
    """
    Parse LLM output and apply safety + confidence guardrails.
    This function MUST NEVER crash.
    """

    text = llm_text.lower()

    # ----------------------------
    # FAILURE MODE EXTRACTION
    # ----------------------------
    if "bearing_wear" in text:
        failure_mode = "bearing_wear"
    elif "bearing_overheating" in text:
        failure_mode = "bearing_overheating"
    else:
        failure_mode = "unknown"

    # ----------------------------
    # CONFIDENCE EXTRACTION (ROBUST)
    # ----------------------------
    confidence_match = re.search(
        r"confidence\s*(level)?\s*[:\-]?\s*(0?\.\d+|1\.0|0\.0)",
        llm_text,
        re.IGNORECASE,
    )

    confidence = _safe_float(
        confidence_match.group(2) if confidence_match else None
    )

    # ----------------------------
    # ACTION GATING
    # ----------------------------
    if confidence < 0.6:
        recommended_action = (
            "Manual inspection required before any maintenance decision."
        )
    else:
        recommended_action = (
            "Inspect bearing condition and lubrication. "
            "Prepare for bearing replacement if abnormal wear is confirmed."
        )

    return MaintenanceDecision(
        failure_mode=failure_mode,
        reasoning=(
            "Localized high-temperature anomaly combined with historical "
            "maintenance incidents indicates bearing-related degradation."
        ),
        recommended_action=recommended_action,
        downtime_hours_min=2,
        downtime_hours_max=6,
        repair_cost_usd_min=300,
        repair_cost_usd_max=1200,
        confidence=round(confidence, 2),
    )
