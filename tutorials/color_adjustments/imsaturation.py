import sys

import cv2
import kornia
import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python imsaturation.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    img: np.ndarray | None = cv2.imread(filename)
    if img is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    cv2.imshow("Image", img)

    img_t: torch.Tensor = kornia.image_to_tensor(img)
    img_t = img_t.float() / 255.0

    img1 = kornia.enhance.adjust_saturation(img_t, 0.4)
    img1_out: np.ndarray = kornia.tensor_to_image(img1)
    cv2.imshow("saturation1", img1_out)

    img2 = kornia.enhance.adjust_saturation(img_t, 0.8)
    img2_out: np.ndarray = kornia.tensor_to_image(img2)
    cv2.imshow("saturation2", img2_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
