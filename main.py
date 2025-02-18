import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("rest.jpeg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Apply edge detection
edges = cv2.Canny(blurred, 50, 150)

# Show the processed image
plt.figure(figsize=(10, 6))
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection Output")
plt.show()
