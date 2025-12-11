import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia

import sys

filename = sys.argv[1]
img:np.array = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img:torch.tensor = kornia.image_to_tensor(img)
inimg  = img[None,...].float() / 255.0

sharpen = kornia.filters.UnsharpMask((9,9), (2.5,2.5))
outimg:torch.tensor = sharpen(inimg)
imdiff = (outimg - inimg).abs()

inimg  = kornia.utils.tensor_to_image(inimg)
outimg = kornia.utils.tensor_to_image(outimg)
imdiff = kornia.utils.tensor_to_image(imdiff)

fig, axs = plt.subplots(1, 3, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('image source')
axs[0].imshow(inimg)

axs[1].axis('off')
axs[1].set_title('sharpened')
axs[1].imshow(outimg)

axs[2].axis('off')
axs[2].set_title('difference')
axs[2].imshow(imdiff)
plt.show()
