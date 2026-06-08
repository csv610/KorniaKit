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
        print("Usage: python edgedetection.py <image_path> <algo>")
        print("  algo: sobel, laplacian, canny")
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

    algo = sys.argv[2]

    if algo == "sobel":
        edges: torch.Tensor = kornia.filters.sobel(img_t)
        edges = 1.0 - edges
    elif algo == "laplacian":
        edges = kornia.filters.laplacian(img_t, kernel_size=5)
        edges = 1.0 - edges.clamp(0.0, 1.0)
    elif algo == "canny":
        edges = kornia.filters.canny(img_t)[0]
        edges = 1.0 - edges.clamp(0.0, 1.0)
    else:
        print(f"Unknown algorithm: {algo}")
        sys.exit(1)

    imshow(edges)


if __name__ == "__main__":
    main()
