import sys

import cv2
import kornia
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python flip.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    cv2.imshow("Image", img)

    img_t: torch.Tensor = kornia.image_to_tensor(img)

    img1 = kornia.geometry.hflip(img_t)
    img1_out: np.ndarray = kornia.tensor_to_image(img1)
    cv2.imshow("hflip", img1_out)

    img2 = kornia.geometry.vflip(img_t)
    img2_out: np.ndarray = kornia.tensor_to_image(img2)
    cv2.imshow("vflip", img2_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
