import sys

import cv2
import kornia
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python readimg.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    print(img_t.shape)
    img_t = img_t.unsqueeze(0)
    print(img_t.shape)

    img_out: np.ndarray = kornia.tensor_to_image(img_t)
    cv2.imshow("Image", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
