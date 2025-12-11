import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia

import sys

filename = sys.argv[1]
img:np.array = cv2.imread(filename)

img:torch.tensor = kornia.image_to_tensor(img)
img  = img[None,...].float() / 255.0

# Apply the Canny edge detector to the image
edges = kornia.filters.canny(img, low_threshold=100, high_threshold=200)

# Extract line segments from the edge map using the Hough transform
lines, _ = kornia.hough_transform(edges)

# Print the detected lines
print(lines)

