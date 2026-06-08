"""Resize an image using different interpolation strategies.

Demonstrates:
  - Using Kornia's geometry.rescale with and without anti-aliasing.
  - Visualising the quality difference between aliased and anti-aliased
    downsampling.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Downsample an image with and without anti-aliasing and compare."""
    (filename,) = parse_args(1, "<image_path>")
    img = load_image(filename)

    import cv2
    import kornia as K
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision

    img_rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data: torch.Tensor = K.image.image_to_tensor(img_rgb).float() / 255.0

    x_025: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125))
    x_025_aa: torch.Tensor = K.geometry.rescale(data, (0.125, 0.125), antialias=True)

    out = torch.stack([x_025, x_025_aa], dim=0)
    grid: torch.Tensor = torchvision.utils.make_grid(out, nrow=2)
    out_np: np.ndarray = K.utils.tensor_to_image(grid)

    plt.figure(figsize=(20, 10))
    plt.imshow(out_np)
    plt.axis("off")
    plt.title("Aliased (left) vs Anti-aliased (right) downsampling")
    plt.show()


if __name__ == "__main__":
    main()
