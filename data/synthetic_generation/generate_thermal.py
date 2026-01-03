import numpy as np
import cv2
import os

IMG_SIZE = 256

def base_motor_thermal():
    """
    Simulates normal motor surface temperature distribution
    """
    x = np.linspace(60, 110, IMG_SIZE)
    gradient = np.tile(x, (IMG_SIZE, 1))
    noise = np.random.normal(0, 2.5, (IMG_SIZE, IMG_SIZE))
    return gradient + noise

def add_bearing_overheat(img):
    """
    Localized circular hotspot → bearing overheating
    """
    cx, cy = 190, 130
    radius = 22

    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                img[y, x] += 75
    return img

def add_misalignment(img):
    """
    Elongated hotspot → shaft misalignment
    """
    cv2.ellipse(
        img,
        center=(128, 128),
        axes=(65, 18),
        angle=25,
        startAngle=0,
        endAngle=360,
        color=190,
        thickness=-1
    )
    return img

def save_image(img, path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

def generate_images():
    os.makedirs("../thermal_images/motor_normal", exist_ok=True)
    os.makedirs("../thermal_images/motor_bearing_overheat", exist_ok=True)
    os.makedirs("../thermal_images/motor_misalignment", exist_ok=True)

    # Normal motors
    for i in range(5):
        img = base_motor_thermal()
        save_image(img, f"../thermal_images/motor_normal/normal_{i}.png")

    # Bearing overheating
    for i in range(5):
        img = base_motor_thermal()
        img = add_bearing_overheat(img)
        save_image(img, f"../thermal_images/motor_bearing_overheat/bearing_{i}.png")

    # Shaft misalignment
    for i in range(5):
        img = base_motor_thermal()
        img = add_misalignment(img)
        save_image(img, f"../thermal_images/motor_misalignment/misalignment_{i}.png")

    print("Synthetic motor thermal images generated successfully.")

if __name__ == "__main__":
    generate_images()
