"""Compute spatial image gradients.

Demonstrates:
  - Using Kornia's filters.spatial_gradient to compute x- and y-gradients.
  - Visualising gradient components as inverted intensity maps.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Load an image, compute gradients of a given order, and display."""
    (filename, order_str) = parse_args(2, "<image_path> <order>")

    import kornia
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision

    img = load_image(filename)
    tensor: torch.Tensor = kornia.image.image_to_tensor(img)
    tensor = tensor[None, ...].float() / 255.0
    tensor = kornia.color.bgr_to_rgb(tensor)
    gray = kornia.color.rgb_to_grayscale(tensor)

    order = int(order_str)
    grads: torch.Tensor = kornia.filters.spatial_gradient(gray, order=order)
    x_grad = (1.0 - grads[:, :, 0].clamp(0.0, 1.0))
    y_grad = (1.0 - grads[:, :, 1].clamp(0.0, 1.0))

    grid = torchvision.utils.make_grid(torch.cat([x_grad, y_grad]), nrow=2, padding=5)
    out: np.ndarray = kornia.utils.tensor_to_image(grid)
    plt.imshow(out, cmap="gray")
    plt.axis("off")
    plt.title(f"Gradient order {order}: x (left), y (right)")
    plt.show()


if __name__ == "__main__":
    main()
