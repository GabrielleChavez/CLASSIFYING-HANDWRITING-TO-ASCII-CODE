import cv2
import numpy as np

# Load the image
image = cv2.imread('sentences/a01/a01-000u/a01-000u-s00-00.png', cv2.IMREAD_GRAYSCALE)

# Preprocess: Binarize the image
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operations to refine characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(binary, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and save each segment
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # Filter small noise
        char_img = binary[y:y+h, x:x+w]
        cv2.imwrite(f'segment_{i}.png', char_img)