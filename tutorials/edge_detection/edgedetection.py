"""Detect edges using classical operators.

Demonstrates:
  - Sobel, Laplacian, and Canny edge detection via Kornia filters.
  - Converting edge maps to a visual format by inverting intensities.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Apply the selected edge detector and display the edge map."""
    (filename, algo) = parse_args(2, "<image_path> <algo>")
    print("  algo: sobel, laplacian, canny")

    img = load_image(filename)

    import kornia
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision

    tensor: torch.Tensor = kornia.image.image_to_tensor(img)
    tensor = tensor[None, ...].float() / 255.0
    tensor = kornia.color.bgr_to_rgb(tensor)
    gray = kornia.color.rgb_to_grayscale(tensor)

    algos = {
        "sobel": lambda t: 1.0 - kornia.filters.sobel(t),
        "laplacian": lambda t: 1.0 - kornia.filters.laplacian(t, kernel_size=5).clamp(0.0, 1.0),
        "canny": lambda t: 1.0 - kornia.filters.canny(t)[0].clamp(0.0, 1.0),
    }

    fn = algos.get(algo)
    if fn is None:
        print(f"Unknown algorithm: {algo}")
        print("Valid options: sobel, laplacian, canny")
        import sys
        sys.exit(1)

    edges = fn(gray)

    grid = torchvision.utils.make_grid(edges, nrow=2, padding=5)
    out_np: np.ndarray = kornia.utils.tensor_to_image(grid)
    plt.imshow(out_np, cmap="gray")
    plt.axis("off")
    plt.title(f"Edge detection: {algo}")
    plt.show()


if __name__ == "__main__":
    main()
