import sys

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python unsharp.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    in_img = img_t[None, ...].float() / 255.0

    sharpen = kornia.filters.UnsharpMask((9, 9), (2.5, 2.5))
    out_img_t: torch.Tensor = sharpen(in_img)
    im_diff = (out_img_t - in_img).abs()

    in_img_np = kornia.utils.tensor_to_image(in_img)
    out_img_np = kornia.utils.tensor_to_image(out_img_t)
    im_diff_np = kornia.utils.tensor_to_image(im_diff)

    fig, axs = plt.subplots(1, 3, figsize=(16, 10))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].set_title("image source")
    axs[0].imshow(in_img_np)

    axs[1].axis("off")
    axs[1].set_title("sharpened")
    axs[1].imshow(out_img_np)

    axs[2].axis("off")
    axs[2].set_title("difference")
    axs[2].imshow(im_diff_np)
    plt.show()


if __name__ == "__main__":
    main()
