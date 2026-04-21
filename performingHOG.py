import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split



# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract HOG features
features, hog_image = hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)

# Improve contrast for visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Show results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("HOG Image")
plt.imshow(hog_image_rescaled, cmap='gray')
plt.axis("off")

plt.show()

print("HOG feature vector length:", len(features))
