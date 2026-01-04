from typing import List, Dict
import numpy as np


def analyze_trend(feature_history: List[Dict]) -> Dict:
    """
    Analyze temporal trends in thermal anomaly features.
    """

    if len(feature_history) < 2:
        return {
            "trend": "insufficient_data",
            "urgency": "low",
            "explanation": "Not enough historical data to determine trend."
        }

    severity_scores = [f["severity_score"] for f in feature_history]
    temp_deltas = [f["temperature_delta"] for f in feature_history]

    severity_slope = np.polyfit(range(len(severity_scores)), severity_scores, 1)[0]
    temp_slope = np.polyfit(range(len(temp_deltas)), temp_deltas, 1)[0]

    if severity_slope > 0.2 and temp_slope > 5:
        trend = "worsening"
        urgency = "high"
    elif severity_slope > 0:
        trend = "slow_worsening"
        urgency = "medium"
    else:
        trend = "stable"
        urgency = "low"

    return {
        "trend": trend,
        "urgency": urgency,
        "severity_slope": round(severity_slope, 2),
        "temperature_slope": round(temp_slope, 2),
        "explanation": (
            "Thermal severity and temperature delta show a "
            f"{trend.replace('_', ' ')} pattern over time."
        )
    }
