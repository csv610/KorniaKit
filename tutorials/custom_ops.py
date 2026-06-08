"""Custom 2D convolution from scratch.

Pedagogical implementation of 2D convolution without relying on
built-in deep-learning convolution operators. Demonstrates understanding
of the core image-processing operation that underpins most of Kornia.
"""

import numpy as np
import torch
import torch.nn.functional as F


def convolve2d(
    image: torch.Tensor,
    kernel: torch.Tensor,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """Apply a 2D convolution on a single-channel image using a sliding window.

    Args:
        image: Input image, shape ``(H, W)``.
        kernel: Convolution kernel, shape ``(Kh, Kw)``.
        padding: Number of zero-padding pixels on each side.
        stride: Step size between window positions.

    Returns:
        Convolved output, shape ``(H_out, W_out)`` where
        ``H_out = (H + 2*padding - Kh) // stride + 1``.
    """
    assert image.dim() == 2, f"Expected 2D image, got shape {image.shape}"
    assert kernel.dim() == 2, f"Expected 2D kernel, got shape {kernel.shape}"
    assert stride >= 1, f"Stride must be >= 1, got {stride}"
    assert padding >= 0, f"Padding must be >= 0, got {padding}"

    H, W = image.shape
    Kh, Kw = kernel.shape

    H_out = (H + 2 * padding - Kh) // stride + 1
    W_out = (W + 2 * padding - Kw) // stride + 1
    assert H_out > 0 and W_out > 0, (
        f"Kernel ({Kh}x{Kw}) too large for image ({H}x{W}) "
        f"with padding={padding}, stride={stride}"
    )

    if padding > 0:
        image = F.pad(image[None, None, :, :], (padding,)*4, mode="constant")[0, 0]

    output = torch.zeros((H_out, W_out), dtype=image.dtype, device=image.device)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start : h_start + Kh, w_start : w_start + Kw]
            output[i, j] = (patch * kernel).sum()

    return output


def test_convolve2d() -> None:
    """Verify custom convolution matches PyTorch's F.conv2d on random inputs."""
    torch.manual_seed(42)
    for _ in range(10):
        H, W = np.random.randint(10, 32, size=2).tolist()
        Kh, Kw = np.random.randint(2, 5, size=2).tolist()
        pad = np.random.randint(0, 3)
        stride = np.random.randint(1, 3)

        img = torch.randn(H, W)
        kernel = torch.randn(Kh, Kw)

        custom = convolve2d(img, kernel, padding=pad, stride=stride)

        ref = F.conv2d(
            img.view(1, 1, H, W),
            kernel.view(1, 1, Kh, Kw),
            padding=pad,
            stride=stride,
        )[0, 0]

        assert torch.allclose(custom, ref, atol=1e-5), (
            f"Failed: H={H}, W={W}, Kh={Kh}, Kw={Kw}, "
            f"pad={pad}, stride={stride}, "
            f"max_diff={((custom - ref).abs().max().item()):.2e}"
        )
    print("All 10 random tests passed.")
