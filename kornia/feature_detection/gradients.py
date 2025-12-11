import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia

import sys

def imshow(input: torch.Tensor):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

filename = sys.argv[1]
img:np.array = cv2.imread(filename)

img:torch.tensor = kornia.image_to_tensor(img)
img  = img[None,...].float() / 255.0

img: torch.Tensor = kornia.color.bgr_to_rgb(img)
img = kornia.color.rgb_to_grayscale(img)

order = int(sys.argv[2])
grads: torch.Tensor = kornia.filters.spatial_gradient(img, order=order)  # BxCx2xHxW
print( img.shape)
xgrads = grads[:, :, 0]
ygrads = grads[:, :, 1]

xgrads = (1.0 - xgrads.clamp(0.0, 1.0))
ygrads = (1.0 - ygrads.clamp(0.0, 1.0))

imshow( xgrads)
imshow( ygrads)











