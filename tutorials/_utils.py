"""Shared utilities for loading, validating, and converting images."""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import kornia
import numpy as np
import torch


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and validate it.

    Args:
        path: Path to the image file.

    Returns:
        Image as a BGR numpy array.

    Raises:
        SystemExit: If the file does not exist or cannot be read.
    """
    if not Path(path).is_file():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    img: np.ndarray | None = cv2.imread(path)
    if img is None:
        print(f"Error: could not decode image: {path}")
        sys.exit(1)
    return img


def to_tensor(img: np.ndarray, add_batch: bool = True) -> torch.Tensor:
    """Convert a numpy image to a normalized Kornia tensor.

    Args:
        img: BGR image array.
        add_batch: If True, adds a batch dimension.

    Returns:
        Float tensor with values in [0, 1].
    """
    tensor: torch.Tensor = kornia.image.image_to_tensor(img)
    tensor = tensor.float() / 255.0
    if add_batch:
        tensor = tensor[None, ...]
    return tensor


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized Kornia tensor back to a numpy array.

    Args:
        tensor: Float tensor with values in [0, 1].

    Returns:
        Image as a uint8 numpy array.
    """
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return kornia.image.tensor_to_image((tensor.clamp(0.0, 1.0) * 255).to(torch.uint8))


def bgr_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """Load, convert to RGB, and return a batched float tensor.

    Args:
        img: BGR image array.

    Returns:
        Batched RGB float tensor in [0, 1], shape (1, 3, H, W).
    """
    rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return to_tensor(rgb)


def parse_args(min_args: int, usage: str) -> Tuple[str, ...]:
    """Check CLI arguments and return them.

    Args:
        min_args: Minimum number of required arguments.
        usage: Usage string to display on error.

    Returns:
        Tuple of argument strings.
    """
    if len(sys.argv) < min_args + 1:
        print(f"Usage: python {Path(sys.argv[0]).name} {usage}")
        sys.exit(1)
    return tuple(sys.argv[1:])
