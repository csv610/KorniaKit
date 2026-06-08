import sys

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(input: torch.Tensor) -> None:
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis("off")
    plt.show()


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python gradients.py <image_path> <order>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    img_t = img_t[None, ...].float() / 255.0

    img_t = kornia.color.bgr_to_rgb(img_t)
    img_t = kornia.color.rgb_to_grayscale(img_t)

    order = int(sys.argv[2])
    grads: torch.Tensor = kornia.filters.spatial_gradient(img_t, order=order)
    print(img_t.shape)
    xgrads = grads[:, :, 0]
    ygrads = grads[:, :, 1]

    xgrads = 1.0 - xgrads.clamp(0.0, 1.0)
    ygrads = 1.0 - ygrads.clamp(0.0, 1.0)

    imshow(xgrads)
    imshow(ygrads)


if __name__ == "__main__":
    main()
