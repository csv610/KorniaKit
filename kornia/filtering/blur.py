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
imshow( img )

algo  = sys.argv[2]
if(algo == "box"):
   blur_img : torch.Tensor = kornia.filters.box_blur(img, (9,9)) 

if(algo == "blurpool"):
   blur_img : torch.Tensor = kornia.filters.blur_pool2d(img, kernel_size=9) 

if(algo == "gaussian"):
   blur_img : torch.Tensor = kornia.filters.gaussian_blur2d(img, (11,11), (11.0, 11.0))

if(algo == "max_pool"):
   blur_img : torch.Tensor = kornia.filters.max_blur_pool2d(img, kernel_size = 11)

if(algo == "median"):
   blur_img : torch.Tensor = kornia.filters.median_blur(img, (5,5))

imshow( blur_img )
