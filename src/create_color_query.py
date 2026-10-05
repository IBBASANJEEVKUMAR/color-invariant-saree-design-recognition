import cv2
import os

input_image = "../data/gallery/0001.jpg"
output_image = "../data/queries/color_changed.jpg"

image = cv2.imread(input_image)

if image is None:
    raise ValueError("Could not read the image.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Change the hue while keeping the texture and pattern
hsv[:, :, 0] = (hsv[:, :, 0] + 70) % 180

changed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite(output_image, changed_image)

print("Color-changed query image created successfully.")