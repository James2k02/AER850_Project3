from google.colab import drive
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

'''Step 1: Object Masking'''

# Mount Google Drive
# drive.mount('/content/drive')

# Define the path to your image in Google Drive
image_path = '/content/drive/MyDrive/Dataset_3/motherboard_image.JPEG'

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR) # reading the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale since many algorithms such as edge detection and thresholdig work better on single-channel grayscale images

# Enhance contrast using histogram equalization
gray_eq = cv2.equalizeHist(gray) # enhancing the contrast using histogram equalization to better differentiate between the PCB and background

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0) # Smooths out the image and reduces noise since noise can lead to false edges during contour detection

# Thresholding to isolate the PCB
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# converts the grayscale image into a binary image using Otsu's thresholding
# dynamically calculates the optimal threshold value based on the image's pixel distribution

# Morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# to remove small gaps within the hinary image using mrophological closing

# Find contours
contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# detects the edges or boundaries of objects in the binary image

# Filter contours based on area to isolate the PCB
filtered_contours = [c for c in contours if cv2.contourArea(c) > 50000]  # can adjust area threshold
# keeps only the largest contours based on area and smaller ones should be excluded

# Create a blank mask and draw the filtered contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness = cv2.FILLED)
# creates a mask by filling in the filtered contours which isolates the motherboard from the rest of the image

# Erode the mask to tighten boundaries
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
eroded_mask = cv2.erode(mask, kernel_erode, iterations = 1)
# shrinks the mask slightly to remove excess background around the motherboard which improves the final extraction quality

# Extract the PCB using the eroded mask
extracted_image = cv2.bitwise_and(image, image, mask = eroded_mask)
# using the mask to isolate the motherboard from the original image

# Automatically crop to the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(max(filtered_contours, key = cv2.contourArea))
cropped_image = extracted_image[y:y+h, x:x+w]
# crop the image to tightly fit the motherboard so you don't see the rest of the background

# Display the final result
plt.figure(figsize=(8, 8))
plt.title("Final Extracted Motherboard Image")
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
