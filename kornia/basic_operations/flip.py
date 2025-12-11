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

img:torch.tensor = kornia.image_to_tensor(img)

img1 = kornia.geometry.hflip(img)
img1:np.array = kornia.tensor_to_image(img1)
cv2.imshow("hflip", img1)

img2 = kornia.geometry.vflip(img)
img2:np.array = kornia.tensor_to_image(img2)
cv2.imshow("vflip", img2)

cv2.waitKey(0)










