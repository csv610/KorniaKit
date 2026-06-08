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
        print("Usage: python blur.py <image_path> <algo>")
        print("  algo: box, blurpool, gaussian, max_pool, median")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    img_t = img_t[None, ...].float() / 255.0

    img_t = kornia.color.bgr_to_rgb(img_t)
    imshow(img_t)

    algo = sys.argv[2]
    if algo == "box":
        blur_img: torch.Tensor = kornia.filters.box_blur(img_t, (9, 9))
    elif algo == "blurpool":
        blur_img = kornia.filters.blur_pool2d(img_t, kernel_size=9)
    elif algo == "gaussian":
        blur_img = kornia.filters.gaussian_blur2d(img_t, (11, 11), (11.0, 11.0))
    elif algo == "max_pool":
        blur_img = kornia.filters.max_blur_pool2d(img_t, kernel_size=11)
    elif algo == "median":
        blur_img = kornia.filters.median_blur(img_t, (5, 5))
    else:
        print(f"Unknown algorithm: {algo}")
        sys.exit(1)

    imshow(blur_img)


if __name__ == "__main__":
    main()
