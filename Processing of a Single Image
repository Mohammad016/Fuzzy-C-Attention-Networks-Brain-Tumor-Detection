#Processing of a Single Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Path to the image file
image_path = "/content/drive/MyDrive/Y1.jpg"

# Load the image
img = mpimg.imread(image_path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis labels
plt.show()


# Display image properties
print("Image shape:", img.shape)  # Shape of the image (height, width, channels)
print("Minimum pixel value:", img.min())  # Minimum pixel value
print("Maximum pixel value:", img.max())  # Maximum pixel value
print("Data type:", img.dtype)  # Data type of the image
# Desired width and height for resizing
new_width = 225
new_height = 225

# Read the image using OpenCV
img = cv2.imread(image_path)

# Resize the image using OpenCV
resized_img = cv2.resize(img, (new_width, new_height))

# Convert BGR to RGB (OpenCV uses BGR by default)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

# Display the resized image using Matplotlib
plt.imshow(resized_img)
plt.axis('off')  # Turn off axis labels
plt.show()

# Convert the image to grayscale (histogram equalization works on grayscale images)
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_img = cv2.equalizeHist(gray_img)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

bilateral_filtered_img = cv2.bilateralFilter(equalized_img, d=8, sigmaColor=75, sigmaSpace=75)
plt.imshow(bilateral_filtered_img)
plt.title('Bilateral Filtered Image')
plt.axis('off')

