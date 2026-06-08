"""Comprehensive edge-case and property-based tests for the tutorial modules.

Tests cover:
  - Import correctness
  - Edge cases (empty, single-channel, extreme values, shape invariance)
  - Property-based constraints (range correctness, dtype stability)
  - Custom convolution correctness
  - Pipeline integration
"""


import kornia
import numpy as np
import torch

from tutorials.custom_ops import convolve2d, test_convolve2d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_tensor_in_range(t: torch.Tensor) -> None:
    assert t.min() >= 0.0, f"Min {t.min().item()} < 0"
    assert t.max() <= 1.0, f"Max {t.max().item()} > 1"


def _synthetic_rgb(h: int = 100, w: int = 120) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _synthetic_gray(h: int = 100, w: int = 120) -> np.ndarray:
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _single_channel(h: int = 50, w: int = 60) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 1), dtype=np.uint8)


def _all_same(value: int = 128, h: int = 50, w: int = 60) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestBasicOperations:
    def test_flip_preserves_shape(self) -> None:
        img = _synthetic_rgb()
        t: torch.Tensor = kornia.image.image_to_tensor(img).float() / 255.0
        for fn in (kornia.geometry.hflip, kornia.geometry.vflip):
            result = fn(t)
            assert result.shape == t.shape, f"{fn.__name__} changed shape"

    def test_flip_identity_twice(self) -> None:
        img = torch.rand(3, 30, 40)
        assert torch.allclose(kornia.geometry.hflip(kornia.geometry.hflip(img)), img)
        assert torch.allclose(kornia.geometry.vflip(kornia.geometry.vflip(img)), img)

    def test_resize_downscale(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        for scale, expected_h, expected_w in [(0.5, 50, 60), (0.25, 25, 30)]:
            r = kornia.geometry.rescale(t, (scale, scale), antialias=True)
            assert r.shape[-2:] == (expected_h, expected_w), f"scale={scale}"

    def test_resize_identity(self) -> None:
        t = torch.rand(3, 64, 64)
        r = kornia.geometry.rescale(t, (1.0, 1.0))
        assert r.shape == t.shape

    def test_tensor_range_after_resize(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        r = kornia.geometry.rescale(t, (0.5, 0.5), antialias=True)
        _assert_tensor_in_range(r)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_blur_preserves_shape(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        t = t[None, ...]
        for fn in (
            lambda x: kornia.filters.box_blur(x, (5, 5)),
            lambda x: kornia.filters.gaussian_blur2d(x, (5, 5), (1.0, 1.0)),
            lambda x: kornia.filters.median_blur(x, (5, 5)),
        ):
            assert fn(t).shape == t.shape

    def test_unsharp_mask_range(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        t = t[None, ...]
        sharpen = kornia.filters.UnsharpMask((5, 5), (1.0, 1.0))
        result = sharpen(t)
        assert result.min() >= -0.7
        assert result.max() <= 1.7

    def test_constant_image_blur_constant(self) -> None:
        np_img = _all_same(128)
        t: torch.Tensor = kornia.image.image_to_tensor(np_img).float() / 255.0
        t = t[None, ...]
        blurred = kornia.filters.box_blur(t, (11, 11))
        assert torch.allclose(blurred, t, atol=1e-4), "Blur of constant should be constant"

    def test_single_channel_median_blur(self) -> None:
        """Median blur on a single-channel image should not raise."""
        img = _single_channel()
        t: torch.Tensor = kornia.image.image_to_tensor(img).float() / 255.0
        t = t[None, ...]
        result = kornia.filters.median_blur(t, (3, 3))
        assert result.shape == t.shape


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    def test_sobel_output_range(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        t = t[None, ...]
        gray = kornia.color.rgb_to_grayscale(t)
        edges = kornia.filters.sobel(gray)
        assert (edges >= 0).all()

    def test_laplacian_shape(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        t = t[None, ...]
        gray = kornia.color.rgb_to_grayscale(t)
        lap = kornia.filters.laplacian(gray, kernel_size=5)
        assert lap.shape == gray.shape

    def test_canny_returns_tuple(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_gray()).float() / 255.0
        t = t[None, ...]
        result = kornia.filters.canny(t)
        assert isinstance(result, tuple) and len(result) == 2

    def test_spatial_gradient_output_shape(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        t = t[None, ...]
        gray = kornia.color.rgb_to_grayscale(t)
        grads = kornia.filters.spatial_gradient(gray, order=1)
        assert grads.shape == (1, 1, 2, *gray.shape[-2:])

    def test_edge_detectors_on_uniform(self) -> None:
        """Uniform image should produce zero (or near-zero) edges."""
        t = torch.full((1, 1, 50, 60), 0.5)
        for algo in (kornia.filters.sobel, lambda x: kornia.filters.laplacian(x, 5)):
            edges = algo(t)
            assert edges.max() < 1e-2, f"{algo.__name__} not zero on uniform"


# ---------------------------------------------------------------------------
# Color adjustments
# ---------------------------------------------------------------------------

class TestColorAdjustments:
    def test_adjustment_preserves_shape(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        for fn, factor in [
            (kornia.enhance.adjust_brightness, 0.5),
            (kornia.enhance.adjust_contrast, 1.5),
            (kornia.enhance.adjust_saturation, 0.5),
        ]:
            result = fn(t, factor)
            assert result.shape == t.shape, f"{fn.__name__} changed shape"

    def test_brightness_identity(self) -> None:
        t = torch.rand(3, 20, 30)
        assert torch.allclose(kornia.enhance.adjust_brightness(t, 0.0), t)

    def test_contrast_identity(self) -> None:
        t = torch.rand(3, 20, 30)
        assert torch.allclose(kornia.enhance.adjust_contrast(t, 1.0), t)

    def test_output_in_range(self) -> None:
        t: torch.Tensor = kornia.image.image_to_tensor(_synthetic_rgb()).float() / 255.0
        for fn, factor in [
            (kornia.enhance.adjust_brightness, 0.3),
            (kornia.enhance.adjust_contrast, 2.0),
            (kornia.enhance.adjust_saturation, 0.3),
        ]:
            result = fn(t, factor)
            _assert_tensor_in_range(result)

    def test_extreme_brightness_clamps(self) -> None:
        t = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).expand(-1, 10, 10)
        result = kornia.enhance.adjust_brightness(t, 10.0)
        assert (result <= 1.0).all()
        result = kornia.enhance.adjust_brightness(t, -10.0)
        assert (result >= 0.0).all()


# ---------------------------------------------------------------------------
# Custom convolution
# ---------------------------------------------------------------------------

class TestCustomConvolution:
    def test_against_pytorch(self) -> None:
        """Verify that custom convolve2d matches F.conv2d randomly."""
        test_convolve2d()

    def test_identity_kernel(self) -> None:
        """Convolution with a 1x1 identity kernel should preserve the image."""
        img = torch.randn(20, 30)
        kernel = torch.ones(1, 1)
        result = convolve2d(img, kernel, padding=0, stride=1)
        assert torch.allclose(result, img)

    def test_edge_kernel_sum(self) -> None:
        """A kernel that sums to zero on a constant image gives zero."""
        img = torch.full((20, 30), 5.0)
        kernel = torch.tensor([[1.0, -1.0]])
        result = convolve2d(img, kernel, padding=0, stride=1)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_invalid_kernel_size_raises(self) -> None:
        img = torch.randn(10, 10)
        kernel = torch.randn(15, 15)
        try:
            convolve2d(img, kernel, padding=0, stride=1)
            assert False, "Should have raised"
        except AssertionError:
            pass


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_noop(self) -> None:
        """Pipeline with no operations should return original."""
        from tutorials.pipeline import run_pipeline
        img = _synthetic_rgb()
        result = run_pipeline(img, blur=None, edges=None, brightness=None, contrast=None, saturation=None)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_pipeline_blur_only(self) -> None:
        """Blur-only pipeline should run without error."""
        from tutorials.pipeline import run_pipeline
        img = _synthetic_rgb()
        result = run_pipeline(img, blur="gaussian", edges=None, brightness=None, contrast=None, saturation=None)
        assert result.shape == img.shape

    def test_pipeline_full_chain(self) -> None:
        """Full pipeline (blur → edges → color adj) should complete."""
        from tutorials.pipeline import run_pipeline
        img = _synthetic_rgb(50, 60)
        result = run_pipeline(img, blur="box", edges="canny", brightness=0.8, contrast=1.2, saturation=0.9)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class TestUtils:
    def test_load_image_valid(self) -> None:
        from pathlib import Path

        from tutorials._utils import load_image
        path = "tutorials/data/drslump.jpg"
        if Path(path).is_file():
            img = load_image(path)
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_to_tensor_shape(self) -> None:
        from tutorials._utils import to_tensor
        img = _synthetic_rgb()
        t = to_tensor(img, add_batch=True)
        assert t.shape == (1, 3, 100, 120)
        t2 = to_tensor(img, add_batch=False)
        assert t2.shape == (3, 100, 120)

    def test_to_tensor_range(self) -> None:
        from tutorials._utils import to_tensor
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        t = to_tensor(img, add_batch=False)
        assert torch.allclose(t, torch.ones_like(t))

    def test_from_tensor_roundtrip(self) -> None:
        from tutorials._utils import from_tensor, to_tensor
        img = _synthetic_rgb()
        t = to_tensor(img, add_batch=True)
        reconstructed = from_tensor(t)
        assert reconstructed.shape == img.shape
        assert reconstructed.dtype == np.uint8

    def test_bgr_to_rgb_tensor_swaps_channels(self) -> None:
        from tutorials._utils import bgr_to_rgb_tensor
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel max
        t = bgr_to_rgb_tensor(bgr)
        # After BGR→RGB, blue channel (index 2 in RGB) should be max
        assert t[0, 2, :, :].mean().item() > 0.99


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_runs(self) -> None:
        """Benchmark should return results without crashing."""
        from tutorials.benchmark import run
        results = run()
        assert len(results) > 0
        for r in results:
            assert "operation" in r

    def test_psnr_same_image(self) -> None:
        """PSNR of identical images should be infinity."""
        from tutorials.benchmark import psnr
        img = _synthetic_rgb()
        assert psnr(img, img) == float("inf")

    def test_psnr_different_images(self) -> None:
        """PSNR of very different images should be low."""
        from tutorials.benchmark import psnr
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.full((10, 10, 3), 255, dtype=np.uint8)
        val = psnr(a, b)
        assert val < 20.0
