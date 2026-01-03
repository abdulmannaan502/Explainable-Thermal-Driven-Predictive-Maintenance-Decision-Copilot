def interpret_motor_fault(anomaly_features):
    """
    Interprets thermal anomaly features into engineering fault hypotheses
    """

    hotspot_count = anomaly_features["hotspot_count"]
    severity = anomaly_features["severity_score"]
    delta_temp = anomaly_features["temperature_delta"]

    interpretation = {
        "suspected_fault": "unknown",
        "risk_level": "low",
        "engineering_reasoning": "",
        "recommended_next_step": ""
    }

    # Bearing overheating logic
    if hotspot_count == 1 and severity > 0.8 and delta_temp > 40:
        interpretation["suspected_fault"] = "bearing_overheating"
        interpretation["risk_level"] = "high"
        interpretation["engineering_reasoning"] = (
            "A localized high-temperature region near the bearing zone "
            "indicates excessive friction, likely caused by bearing wear "
            "or lubrication failure."
        )
        interpretation["recommended_next_step"] = (
            "Inspect bearing condition and lubrication. "
            "Plan bearing replacement if abnormal wear is confirmed."
        )

    # Shaft misalignment logic
    elif hotspot_count >= 1 and 0.4 < severity <= 0.8:
        interpretation["suspected_fault"] = "shaft_misalignment"
        interpretation["risk_level"] = "medium"
        interpretation["engineering_reasoning"] = (
            "Elongated or moderately severe thermal anomalies "
            "suggest uneven load distribution, commonly caused by shaft misalignment."
        )
        interpretation["recommended_next_step"] = (
            "Perform shaft alignment check and vibration analysis."
        )

    # Normal or low-risk condition
    else:
        interpretation["suspected_fault"] = "normal_operation"
        interpretation["risk_level"] = "low"
        interpretation["engineering_reasoning"] = (
            "Observed temperature variations are within acceptable operating limits."
        )
        interpretation["recommended_next_step"] = (
            "Continue routine monitoring."
        )

    return interpretation
