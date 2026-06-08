import sys

import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(input: torch.Tensor) -> None:
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=(20, 10))
    plt.imshow(out_np)
    plt.axis("off")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python resize.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img_bgr: np.ndarray | None = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    data: torch.Tensor = K.utils.image_to_tensor(img_rgb).float() / 255.0

    x_025: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125))
    x_025AA: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125), antialias=True)
    out = torch.stack([x_025, x_025AA], dim=0)
    imshow(out)
    plt.show()


if __name__ == "__main__":
    main()
