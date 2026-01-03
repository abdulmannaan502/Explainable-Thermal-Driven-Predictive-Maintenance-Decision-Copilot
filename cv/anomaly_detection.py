import numpy as np
import cv2

def detect_thermal_anomaly(img):
    """
    Detects hotspots using statistical thresholding
    """
    mean_temp = np.mean(img)
    std_temp = np.std(img)
    max_temp = np.max(img)

    # Threshold = abnormal heat
    threshold = mean_temp + 2 * std_temp
    hotspot_mask = img > threshold

    # Connected components = number of hotspots
    num_labels, _ = cv2.connectedComponents(
        hotspot_mask.astype(np.uint8)
    )

    # Severity score (normalized)
    severity_score = float(round((max_temp - mean_temp) / mean_temp, 2))

    return {
        "mean_temperature": round(float(mean_temp), 2),
        "max_temperature": round(float(max_temp), 2),
        "temperature_delta": round(float(max_temp - mean_temp), 2),
        "hotspot_count": int(num_labels - 1),  # excluding background
        "severity_score": severity_score
    }, hotspot_mask
