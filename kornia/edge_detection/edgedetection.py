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

algo  = sys.argv[2]

if(algo == "sobel"):
   edges : torch.Tensor = kornia.filters.sobel(img) 
   edges = 1.0 - edges

if(algo == "laplacian"):
   edges : torch.Tensor = kornia.filters.laplacian(img, kernel_size=5) 
   edges = 1.0 - edges.clamp( 0.0, 1.0)

if(algo == "canny"):
   edges : torch.Tensor = kornia.filters.canny(img)[0]
   edges = 1.0 - edges.clamp( 0.0, 1.0)

imshow( edges )
