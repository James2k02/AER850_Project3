from google.colab import drive
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

'''Step 1: Object Masking'''

# Mount Google Drive
# !fusermount -u /content/drive
# !rm -rf /content/drive
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

# Display the results in a 2x2 grid
plt.figure(figsize = (12, 12))  

# Edge detection visualization
plt.subplot(2, 2, 1)
edges = cv2.Canny(gray_eq, 50, 150)  
plt.title("Edge Detection", fontsize = 20)
plt.imshow(edges, cmap = 'gray')
plt.axis('off')

# Mask image
plt.subplot(2, 2, 2)
plt.title("Mask Image", fontsize = 20)
plt.imshow(mask, cmap = 'gray')
plt.axis('off')

# Eroded Mask
plt.subplot(2, 2, 3)
plt.title("Eroded Mask", fontsize = 20)
plt.imshow(eroded_mask, cmap = 'gray')
plt.axis('off')

# Final extracted image
plt.subplot(2, 2, 4)
plt.title("Final Extracted Image", fontsize = 20)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

'''Installing Ultralytics for Google Colab'''

# !pip install ultralytics

'''Step 2: YOLOv8 Training'''

from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data = '/content/drive/MyDrive/Dataset_3/Dataset_3/Project3Data/data/data.yaml',
    epochs = 200,                   
    batch  = 8,                     
    imgsz  = 960,                    
    lr0 = 0.0005,   
    lrf = 0.01,                 
    optimizer = 'AdamW',             
    weight_decay = 0.0005,            
    augment = True,
    amp = True,
    name = '/content/drive/MyDrive/Dataset_3/Dataset_3/motherboard_mode'
)

'''Step 3: YOLOv8 Evaluation'''

from ultralytics import YOLO
from IPython.display import Image, display
import os

# Load your trained model
model = YOLO('/content/drive/MyDrive/Dataset_3/Dataset_3/motherboard_model/weights/best.pt')

# Paths to test images
image_paths = [
    '/content/drive/MyDrive/Dataset_3/Dataset_3/Project3Data/data/evaluation/ardmega.jpg',
    '/content/drive/MyDrive/Dataset_3/Dataset_3/Project3Data/data/evaluation/arduno.jpg',
    '/content/drive/MyDrive/Dataset_3/Dataset_3/Project3Data/data/evaluation/rasppi.jpg'
]

# Custom line width and text size
custom_args = {
    'line_width': 3,  # Thinner bounding box lines
    'font_size': 3  # Smaller font size for labels
}

# Run inference with smaller text and line width
results = [
    model.predict(source = img, save = True, conf = 0.5, line_width = custom_args['line_width'], imgsz = 640)
    for img in image_paths
]

# Determine the latest YOLO save directory
latest_run_dir = 'runs/detect/' + sorted(os.listdir('runs/detect/'))[-1]  # Get the most recent directory
print(f"Latest YOLO save directory: {latest_run_dir}")

# List all saved files
saved_files = os.listdir(latest_run_dir)
print("Saved files:", saved_files)

# Display the output images
for file in saved_files:
    if file.endswith(".jpg"):  # Ensure only images are displayed
        display(Image(filename = os.path.join(latest_run_dir, file)))

















