"""Sharpen an image using unsharp masking.

Demonstrates:
  - Using Kornia's UnsharpMask filter.
  - Visualising the original, sharpened result, and difference map.
"""

from tutorials._utils import from_tensor, load_image, parse_args


def main() -> None:
    """Load an image, apply unsharp masking, and show side-by-side results."""
    (filename,) = parse_args(1, "<image_path>")
    img = load_image(filename)

    import cv2
    import kornia
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor: torch.Tensor = kornia.image.image_to_tensor(img_rgb)
    in_img = tensor[None, ...].float() / 255.0

    sharpen = kornia.filters.UnsharpMask((9, 9), (2.5, 2.5))
    out_img_t: torch.Tensor = sharpen(in_img)
    diff = (out_img_t - in_img).abs()

    in_np: np.ndarray = from_tensor(in_img)
    out_np: np.ndarray = from_tensor(out_img_t)
    diff_np: np.ndarray = from_tensor(diff)

    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    for ax, title, arr in zip(axes, ("Original", "Sharpened", "Difference"),
                               (in_np, out_np, diff_np)):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
