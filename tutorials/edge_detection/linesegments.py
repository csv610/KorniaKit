import sys

import cv2
import kornia
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python linesegments.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    print(img.shape)
    img_t: torch.Tensor = kornia.image_to_tensor(img)
    img_t = img_t[None, ...].float() / 255.0

    img_t = kornia.color.bgr_to_rgb(img_t)
    img_t = kornia.color.rgb_to_grayscale(img_t)

    print(img_t.shape)
    detector = kornia.feature.SOLD2_detector()
    lines = detector(img_t)["line_segments"]

    print("#Lines: ", len(lines))
    for line in lines:
        x0 = int(line[0])
        y0 = int(line[1])
        x1 = int(line[2])
        y1 = int(line[3])
        print("Line: ", line)
        print(x0, y0, x1, y1)


if __name__ == "__main__":
    main()
