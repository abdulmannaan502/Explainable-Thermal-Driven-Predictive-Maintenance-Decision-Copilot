# guardrails/risk_scoring.py

def map_severity_to_score(risk_level):
    mapping = {
        "low": 1,
        "medium": 2,
        "high": 3
    }
    return mapping.get(risk_level.lower(), 2)


def estimate_impact(retrieved_logs):
    """
    Impact is estimated from historical downtime and repair cost.
    """
    if not retrieved_logs:
        return 1  # minimal impact if no history

    avg_downtime = sum(r["downtime_hours"] for r in retrieved_logs) / len(retrieved_logs)
    avg_cost = sum(r["repair_cost_usd"] for r in retrieved_logs) / len(retrieved_logs)

    # Normalize impact roughly
    impact_score = 1
    if avg_downtime > 4 or avg_cost > 1000:
        impact_score = 3
    elif avg_downtime > 2 or avg_cost > 500:
        impact_score = 2

    return impact_score


def compute_risk_score(severity_score, impact_score, confidence):
    uncertainty = 1 - confidence
    return severity_score * impact_score * (1 + uncertainty)


def classify_risk(risk_score):
    if risk_score >= 7:
        return "CRITICAL"
    elif risk_score >= 4:
        return "CAUTION"
    else:
        return "SAFE"


def risk_aware_decision_engine(fault_info, retrieved_logs, final_decision):
    severity_score = map_severity_to_score(fault_info["risk_level"])
    impact_score = estimate_impact(retrieved_logs)
    confidence = final_decision["confidence"]

    risk_score = compute_risk_score(severity_score, impact_score, confidence)
    risk_level = classify_risk(risk_score)

    return {
        "severity_score": severity_score,
        "impact_score": impact_score,
        "uncertainty": round(1 - confidence, 2),
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level
    }
