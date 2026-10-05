import cv2
import numpy as np


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image.")

    image = cv2.resize(image, (224, 224))

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    gray_image = cv2.cvtColor(
        enhanced_image,
        cv2.COLOR_BGR2GRAY
    )

    return enhanced_image, gray_image


if __name__ == "__main__":
    print("Saree image preprocessing module is ready.")