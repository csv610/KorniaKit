"""CLI pipeline that chains multiple Kornia operations.

Allows the user to apply a sequence of transformations (blur → edges →
brightness) from a single command, demonstrating how Kornia operations
compose naturally on tensors.
"""

import argparse

import cv2
import kornia
import numpy as np
import torch


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return kornia.image.image_to_tensor(img).float() / 255.0


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return kornia.image.tensor_to_image((t.clamp(0.0, 1.0) * 255).to(torch.uint8)).astype(np.uint8)


def run_pipeline(
    img: np.ndarray,
    blur: str | None,
    edges: str | None,
    brightness: float | None,
    contrast: float | None,
    saturation: float | None,
) -> np.ndarray:
    """Apply a chain of operations to an image.

    Args:
        img: Input BGR image.
        blur: Blur type (``box``, ``gaussian``, ``median``) or None.
        edges: Edge detector (``sobel``, ``canny``) or None.
        brightness: Brightness factor or None.
        contrast: Contrast factor or None.
        saturation: Saturation factor or None.

    Returns:
        Processed BGR image.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = _to_tensor(rgb)

    if blur == "gaussian":
        tensor = kornia.filters.gaussian_blur2d(tensor[None, ...], (11, 11), (2.0, 2.0)).squeeze(0)
    elif blur == "box":
        tensor = kornia.filters.box_blur(tensor[None, ...], (9, 9)).squeeze(0)
    elif blur == "median":
        tensor = kornia.filters.median_blur(tensor[None, ...], (5, 5)).squeeze(0)

    if edges == "sobel":
        gray = kornia.color.rgb_to_grayscale(tensor[None, ...])
        tensor = kornia.filters.sobel(gray).expand(-1, 3, -1, -1).squeeze(0)
    elif edges == "canny":
        gray = kornia.color.rgb_to_grayscale(tensor[None, ...])
        edge = kornia.filters.canny(gray)[0]
        tensor = edge.expand(-1, 3, -1, -1).squeeze(0)

    if brightness is not None:
        tensor = kornia.enhance.adjust_brightness(tensor, brightness)
    if contrast is not None:
        tensor = kornia.enhance.adjust_contrast(tensor, contrast)
    if saturation is not None:
        tensor = kornia.enhance.adjust_saturation(tensor, saturation)

    return cv2.cvtColor(_to_numpy(tensor), cv2.COLOR_RGB2BGR)


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Chain multiple Kornia operations")
    parser.add_argument("image", type=str, help="path to input image")
    parser.add_argument("--blur", choices=["gaussian", "box", "median"], help="blur type")
    parser.add_argument("--edges", choices=["sobel", "canny"], help="edge detector")
    parser.add_argument("--brightness", type=float, help="brightness factor")
    parser.add_argument("--contrast", type=float, help="contrast factor")
    parser.add_argument("--saturation", type=float, help="saturation factor")
    parser.add_argument("--output", type=str, default="output.jpg", help="output path")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not load {args.image}")
        return

    result = run_pipeline(
        img,
        blur=args.blur,
        edges=args.edges,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
    )

    cv2.imwrite(args.output, result)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
