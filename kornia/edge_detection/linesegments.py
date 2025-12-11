import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia

import sys

filename = sys.argv[1]
img:np.array = cv2.imread(filename)
orgimg = img.copy()

print( img.shape )
img:torch.tensor = kornia.image_to_tensor(img)
img  = img[None,...].float() / 255.0

img: torch.Tensor = kornia.color.bgr_to_rgb(img)
img = kornia.color.rgb_to_grayscale(img)

print( img.shape )
detector = kornia.feature.SOLD2_detector()
lines    = detector(img)["line_segments"]

print( "#Lines: ", len(lines))
for line in lines:
    x0 = int(line[0])
    y0 = int(line[1])
    x1 = int(line[2])
    y1 = int(line[3])
    print( "Line: ", line)
    print( x0, y0, x1, y1)
#    cv2.line( orgimg, line[0], line[1], (0,255,0), 2)

#cv2.imshow("Image", orgimg)
#cv2.waitKey(0)


