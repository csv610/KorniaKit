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
img  = img.float() / 255.0

img1 = kornia.enhance.adjust_saturation(img, 0.4)
img1:np.array = kornia.tensor_to_image(img1)
cv2.imshow("bright1", img1)

img2 = kornia.enhance.adjust_saturation(img, 0.8)
img2:np.array = kornia.tensor_to_image(img2)
cv2.imshow("bright2", img2)

cv2.waitKey(0)










