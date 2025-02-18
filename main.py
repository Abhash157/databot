import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("rest.jpeg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Improve contrast using CLAHE (Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# # Apply adaptive thresholding to enhance text
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Apply edge detection with lower thresholds
# edges = cv2.Canny(thresh, 30, 100)

# Show result
plt.figure(figsize=(100, 60))
plt.imshow(thresh, cmap='gray')
plt.title("Enhanced Edge Detection")
plt.savefig("enhanced_edges.png")  # Save for review
plt.show()
