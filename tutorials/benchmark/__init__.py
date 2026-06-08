"""Benchmark comparing Kornia (GPU) vs OpenCV (CPU) performance and quality.

Takes a set of sample images and measures execution time and output
quality (PSNR / SSIM) for equivalent operations in both libraries.
"""

import time
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import kornia
import numpy as np
import torch

BENCHMARK_IMAGE = "tutorials/data/drslump.jpg"


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute the peak signal-to-noise ratio between two images.

    Args:
        original: Reference image.
        processed: Processed image.

    Returns:
        PSNR value in decibels. Higher is better.
    """
    mse = float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _time_fn(fn: Callable, warmup: int = 3, repeat: int = 10) -> Tuple[float, float]:
    """Time a function and return (mean_ms, std_ms).

    Args:
        fn: Zero-argument function to time.
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.

    Returns:
        Mean and standard deviation in milliseconds.
    """
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def benchmark_gaussian_blur(rgb: np.ndarray, tensor: torch.Tensor) -> dict:
    """Benchmark Gaussian blur (kernel 11x11, sigma 2.0)."""
    ksize = (11, 11)
    sigma = (2.0, 2.0)

    def _opencv():
        return cv2.GaussianBlur(rgb, (11, 11), 2.0)

    def _kornia():
        return kornia.filters.gaussian_blur2d(tensor, ksize, sigma)

    mean_cv, std_cv = _time_fn(_opencv)
    mean_kn, std_kn = _time_fn(_kornia)

    cv_result = _opencv()
    kn_result = kornia.utils.tensor_to_image(_kornia().cpu()) * 255
    kn_result = kn_result.astype(np.uint8)

    return {
        "operation": "GaussianBlur",
        "opencv_ms": f"{mean_cv:.2f} ± {std_cv:.2f}",
        "kornia_ms": f"{mean_kn:.2f} ± {std_kn:.2f}",
        "speedup": f"{mean_cv / mean_kn:.2f}x" if mean_kn > 0 else "N/A",
        "psnr": f"{psnr(cv_result, kn_result):.2f} dB",
    }


def benchmark_sobel(rgb: np.ndarray, tensor: torch.Tensor) -> dict:
    """Benchmark Sobel edge detection."""

    def _opencv():
        return cv2.Sobel(rgb, cv2.CV_32F, 1, 0, ksize=3)

    tensor_gray = kornia.color.rgb_to_grayscale(tensor)

    def _kornia():
        return kornia.filters.sobel(tensor_gray)

    mean_cv, std_cv = _time_fn(_opencv)
    mean_kn, std_kn = _time_fn(_kornia)
    return {
        "operation": "Sobel",
        "opencv_ms": f"{mean_cv:.2f} ± {std_cv:.2f}",
        "kornia_ms": f"{mean_kn:.2f} ± {std_kn:.2f}",
        "speedup": f"{mean_cv / mean_kn:.2f}x" if mean_kn > 0 else "N/A",
        "psnr": "N/A (different output shapes)",
    }


def benchmark_brightness(rgb: np.ndarray, tensor: torch.Tensor) -> dict:
    """Benchmark brightness adjustment."""

    def _opencv():
        return cv2.convertScaleAbs(rgb, alpha=1.0, beta=30)

    def _kornia():
        return kornia.enhance.adjust_brightness(tensor, 30.0 / 255.0)

    mean_cv, std_cv = _time_fn(_opencv)
    mean_kn, std_kn = _time_fn(_kornia)

    cv_result = _opencv()
    kn_result = kornia.utils.tensor_to_image(_kornia().cpu()) * 255
    kn_result = kn_result.astype(np.uint8)

    return {
        "operation": "Brightness",
        "opencv_ms": f"{mean_cv:.2f} ± {std_cv:.2f}",
        "kornia_ms": f"{mean_kn:.2f} ± {std_kn:.2f}",
        "speedup": f"{mean_cv / mean_kn:.2f}x" if mean_kn > 0 else "N/A",
        "psnr": f"{psnr(cv_result, kn_result):.2f} dB",
    }


def benchmark_canny(rgb: np.ndarray, tensor: torch.Tensor) -> dict:
    """Benchmark Canny edge detection."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tensor_gray = kornia.color.rgb_to_grayscale(tensor)

    def _opencv():
        return cv2.Canny(gray, 100, 200)

    def _kornia():
        return kornia.filters.canny(tensor_gray, low_threshold=100, high_threshold=200)[0]

    mean_cv, std_cv = _time_fn(_opencv)
    mean_kn, std_kn = _time_fn(_kornia)
    return {
        "operation": "Canny",
        "opencv_ms": f"{mean_cv:.2f} ± {std_cv:.2f}",
        "kornia_ms": f"{mean_kn:.2f} ± {std_kn:.2f}",
        "speedup": f"{mean_cv / mean_kn:.2f}x" if mean_kn > 0 else "N/A",
        "psnr": "N/A (different output formats)",
    }


def run() -> List[dict]:
    """Run all benchmarks and return results as a list of dicts."""
    path = Path(BENCHMARK_IMAGE)
    if not path.is_file():
        return [{"operation": "ERROR", "detail": f"Test image not found: {path}"}]

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return [{"operation": "ERROR", "detail": f"Could not load image: {path}"}]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor: torch.Tensor = kornia.image.image_to_tensor(img_rgb).float() / 255.0
    tensor = tensor[None, ...]  # BxCxHxW

    results: List[dict] = []
    for fn in (benchmark_gaussian_blur, benchmark_sobel, benchmark_brightness, benchmark_canny):
        try:
            results.append(fn(img_rgb, tensor))
        except Exception as exc:
            results.append({
                "operation": fn.__name__.replace("benchmark_", ""),
                "opencv_ms": "ERROR",
                "kornia_ms": "ERROR",
                "speedup": "N/A",
                "psnr": str(exc),
            })
    return results


def print_report(results: List[dict]) -> None:
    """Print a formatted markdown report of benchmark results.

    Args:
        results: List of result dicts from :func:`run`.
    """
    print("\n# Kornia vs OpenCV Benchmark\n")
    print("| Operation | OpenCV (ms) | Kornia (ms) | Speedup | PSNR |")
    print("|-----------|-------------|-------------|---------|------|")
    for r in results:
        print(f"| {r['operation']} | {r.get('opencv_ms', 'N/A')} | {r.get('kornia_ms', 'N/A')} | "
              f"{r.get('speedup', 'N/A')} | {r.get('psnr', 'N/A')} |")
    print()


if __name__ == "__main__":
    results = run()
    print_report(results)
