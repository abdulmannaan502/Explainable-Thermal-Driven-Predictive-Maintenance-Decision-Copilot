# guardrails/decision_safety.py

def assess_signal_plausibility(features):
    issues = []

    if features["temperature_delta"] > 250:
        issues.append("Unrealistically high temperature delta")

    if features["hotspot_count"] == 0 and features["severity_score"] > 2:
        issues.append("High severity without detected hotspot")

    return issues


def assess_evidence_strength(retrieved_logs):
    if len(retrieved_logs) >= 3:
        return "strong"
    elif len(retrieved_logs) == 2:
        return "moderate"
    else:
        return "weak"


def assess_action_safety(severity, recommended_action):
    aggressive_actions = ["replace", "shutdown"]

    if severity in ["low", "medium"]:
        for word in aggressive_actions:
            if word in recommended_action.lower():
                return False

    return True


def decision_safety_engine(features, fault_info, retrieved_logs, llm_decision):
    safety_report = {
        "signal_issues": [],
        "evidence_strength": None,
        "action_safe": True,
        "escalate_to_human": False,
        "final_note": ""
    }

    # 1️⃣ Signal plausibility
    signal_issues = assess_signal_plausibility(features)
    safety_report["signal_issues"] = signal_issues

    # 2️⃣ Evidence sufficiency
    evidence_strength = assess_evidence_strength(retrieved_logs)
    safety_report["evidence_strength"] = evidence_strength

    # 3️⃣ Action safety
    action_safe = assess_action_safety(
        fault_info["risk_level"],
        llm_decision["recommended_action"]
    )
    safety_report["action_safe"] = action_safe

    # 4️⃣ Escalation logic
    if signal_issues or evidence_strength == "weak" or not action_safe:
        safety_report["escalate_to_human"] = True
        safety_report["final_note"] = (
            "Decision requires human review due to signal uncertainty or insufficient evidence."
        )
    else:
        safety_report["final_note"] = "Decision considered safe for recommendation."

    return safety_report
