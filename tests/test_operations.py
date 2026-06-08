"""Functional tests for the Kornia operations used in the tutorials."""

import kornia
import numpy as np
import torch


def _synthetic_image() -> torch.Tensor:
    img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
    tensor: torch.Tensor = kornia.image_to_tensor(img)
    return tensor.float() / 255.0


def _batched_image() -> torch.Tensor:
    return _synthetic_image()[None, ...]


class TestBasicOperations:
    def test_flip_horizontal(self) -> None:
        img = _batched_image()
        flipped = kornia.geometry.hflip(img)
        assert flipped.shape == img.shape
        assert isinstance(flipped, torch.Tensor)

    def test_flip_vertical(self) -> None:
        img = _batched_image()
        flipped = kornia.geometry.vflip(img)
        assert flipped.shape == img.shape

    def test_resize(self) -> None:
        img = _batched_image()
        rescaled = kornia.geometry.rescale(img, (0.5, 0.5))
        assert rescaled.shape[-2:] == (50, 60)

    def test_resize_antialias(self) -> None:
        img = _batched_image()
        rescaled = kornia.geometry.rescale(img, (0.5, 0.5), antialias=True)
        assert rescaled.shape[-2:] == (50, 60)


class TestFiltering:
    def test_box_blur(self) -> None:
        img = _batched_image()
        blurred = kornia.filters.box_blur(img, (9, 9))
        assert blurred.shape == img.shape

    def test_gaussian_blur(self) -> None:
        img = _batched_image()
        blurred = kornia.filters.gaussian_blur2d(img, (11, 11), (11.0, 11.0))
        assert blurred.shape == img.shape

    def test_median_blur(self) -> None:
        img = _batched_image()
        blurred = kornia.filters.median_blur(img, (5, 5))
        assert blurred.shape == img.shape

    def test_unsharp_mask(self) -> None:
        img = _batched_image()
        sharpen = kornia.filters.UnsharpMask((9, 9), (2.5, 2.5))
        sharpened = sharpen(img)
        assert sharpened.shape == img.shape


class TestEdgeDetection:
    def test_sobel(self) -> None:
        img = _batched_image()
        gray = kornia.color.rgb_to_grayscale(img)
        edges = kornia.filters.sobel(gray)
        assert edges.shape[-2:] == img.shape[-2:]

    def test_laplacian(self) -> None:
        img = _batched_image()
        gray = kornia.color.rgb_to_grayscale(img)
        edges = kornia.filters.laplacian(gray, kernel_size=5)
        assert edges.shape[-2:] == img.shape[-2:]

    def test_canny(self) -> None:
        img = _batched_image()
        gray = kornia.color.rgb_to_grayscale(img)
        edges, _ = kornia.filters.canny(gray)
        assert edges.shape[-2:] == img.shape[-2:]

    def test_spatial_gradient(self) -> None:
        img = _batched_image()
        gray = kornia.color.rgb_to_grayscale(img)
        grads = kornia.filters.spatial_gradient(gray, order=1)
        assert grads.shape == (1, 1, 2, *img.shape[-2:])


class TestColorAdjustments:
    def test_brightness(self) -> None:
        img = _synthetic_image()
        adjusted = kornia.enhance.adjust_brightness(img, 0.5)
        assert adjusted.shape == img.shape

    def test_contrast(self) -> None:
        img = _synthetic_image()
        adjusted = kornia.enhance.adjust_contrast(img, 1.5)
        assert adjusted.shape == img.shape

    def test_saturation(self) -> None:
        img = _synthetic_image()
        adjusted = kornia.enhance.adjust_saturation(img, 0.5)
        assert adjusted.shape == img.shape
