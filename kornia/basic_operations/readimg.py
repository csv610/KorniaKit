import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia

import sys

filename = sys.argv[1]

img:np.array = cv2.imread(filename)
cv2.imshow("Image", img)
cv2.waitKey(0)

img:torch.tensor = kornia.image_to_tensor(img)
print( img.shape)
img = img.unsqueeze(0)
print( img.shape)

img:np.array = kornia.tensor_to_image(img)
cv2.imshow("Image", img)
cv2.waitKey(0)










