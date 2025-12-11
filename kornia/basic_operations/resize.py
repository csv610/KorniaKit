from matplotlib import pyplot as plt
import cv2
import numpy as np
import kornia as K
import torch
import torchvision


def imshow(input: torch.Tensor):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2)
    out_np: np.array = K.utils.tensor_to_image(out)
    plt.figure(figsize=(20,10))
    plt.imshow(out_np); plt.axis('off');



# load using opencv and convert to RGB
img_bgr: np.array = cv2.imread('drslump.jpg', cv2.IMREAD_COLOR)
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

data: torch.Tensor = K.utils.image_to_tensor(img_rgb).float()/255.  # 1xCxHxW

x_025: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125))
x_025AA: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125), antialias=True)
out = torch.stack([x_025, x_025AA], dim=0)
imshow(out)

plt.show()


