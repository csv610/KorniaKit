import sys

import cv2
import kornia
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python houghlines.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    img_t = img_t[None, ...].float() / 255.0

    edges = kornia.filters.canny(img_t, low_threshold=100, high_threshold=200)

    lines, _ = kornia.hough_transform(edges)

    print(lines)


if __name__ == "__main__":
    main()
