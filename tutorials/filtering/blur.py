"""Apply various blur filters to an image.

Demonstrates:
  - Box blur, Gaussian blur, median blur, blur pooling, and max-blur
    pooling using Kornia filters.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Apply the selected blur algorithm and display the result."""
    (filename, algo) = parse_args(2, "<image_path> <algo>")
    print("  algo: box, blurpool, gaussian, max_pool, median")

    img = load_image(filename)

    import kornia
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision

    tensor: torch.Tensor = kornia.image.image_to_tensor(img)
    tensor = tensor[None, ...].float() / 255.0
    tensor = kornia.color.bgr_to_rgb(tensor)

    algos = {
        "box": lambda t: kornia.filters.box_blur(t, (9, 9)),
        "blurpool": lambda t: kornia.filters.blur_pool2d(t, kernel_size=9),
        "gaussian": lambda t: kornia.filters.gaussian_blur2d(t, (11, 11), (11.0, 11.0)),
        "max_pool": lambda t: kornia.filters.max_blur_pool2d(t, kernel_size=11),
        "median": lambda t: kornia.filters.median_blur(t, (5, 5)),
    }

    blur_fn = algos.get(algo)
    if blur_fn is None:
        print(f"Unknown algorithm: {algo}")
        print("Valid options: box, blurpool, gaussian, max_pool, median")
        import sys
        sys.exit(1)

    blurred = blur_fn(tensor)

    grid = torchvision.utils.make_grid(blurred, nrow=2, padding=5)
    out_np: np.ndarray = kornia.utils.tensor_to_image(grid)
    plt.imshow(out_np)
    plt.axis("off")
    plt.title(f"Blur: {algo}")
    plt.show()


if __name__ == "__main__":
    main()
