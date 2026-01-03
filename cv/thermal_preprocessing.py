import cv2
import numpy as np

def preprocess_thermal_image(image_path):
    """
    Loads and normalizes a thermal image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid path")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img.astype(np.uint8)
