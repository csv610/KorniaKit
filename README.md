# Kornia Computer Vision Tutorials

[![CI](https://github.com/csv610/KorniaKit/actions/workflows/ci.yml/badge.svg)](https://github.com/csv610/KorniaKit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)

GPU-accelerated computer vision tutorials built with [Kornia](https://kornia.readthedocs.io/) and PyTorch.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Running a Tutorial](#running-a-tutorial)
- [Pipeline CLI](#pipeline-cli)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

```bash
# Editable install (recommended for development)
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"

# Or minimal via requirements
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 1.13+, and Kornia 0.6+.

## Quick Start

```python
import kornia
import torch

# Load and convert an image
img = kornia.image.image_to_tensor("tutorials/data/drslump.jpg")
tensor = img.float()[None, ...] / 255.0

# Apply operations on GPU if available
tensor = tensor.to("cuda" if torch.cuda.is_available() else "cpu")
blurred = kornia.filters.gaussian_blur2d(tensor, (11, 11), (2.0, 2.0))
edges = kornia.filters.sobel(kornia.color.rgb_to_grayscale(blurred))
print(f"Edges shape: {edges.shape}")
```

## Modules

| Module | Description |
|--------|-------------|
| `basic_operations/` | Image I/O, flipping, resizing |
| `filtering/` | Box, Gaussian, median blur; unsharp masking |
| `edge_detection/` | Sobel, Laplacian, Canny, CNN-based, line segments, LSD |
| `color_adjustments/` | Brightness, contrast, saturation |
| `feature_detection/` | Gradients, Hough lines, face detection |
| `benchmark/` | Performance benchmarking (PSNR, timing comparison with OpenCV) |
| `pipeline.py` | CLI pipeline that chains multiple operations |
| `custom_ops.py` | Custom 2D convolution implementation |
| `_utils.py` | Shared utilities (loading, tensor conversion, CLI parsing) |

## Running a Tutorial

```bash
python tutorials/basic_operations/readimg.py <image_path>
python tutorials/filtering/blur.py <image_path> --kernel 11
python tutorials/edge_detection/edgedetection.py <image_path> --algo canny
```

## Pipeline CLI

The pipeline chains blur, edge detection, and color adjustments in a single command:

```bash
python tutorials/pipeline.py input.jpg \
  --blur gaussian \
  --edges canny \
  --brightness 0.8 \
  --contrast 1.2 \
  --saturation 0.9 \
  --output result.jpg
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=tutorials --cov-report=html
```

The test suite covers:

- **`tests/test_imports.py`** — All modules import without errors
- **`tests/test_operations.py`** — Correctness of each operation (flip, resize, blur, edge detection, color adjustments)
- **`tests/test_all.py`** — Edge cases, property-based constraints, custom convolution, pipeline integration, utilities, and benchmarking

## Project Structure

```
├── tutorials/
│   ├── __init__.py
│   ├── _utils.py              Shared image loading & tensor conversion
│   ├── custom_ops.py          Custom 2D convolution
│   ├── pipeline.py            CLI pipeline (blur → edges → color adj)
│   ├── basic_operations/      Image I/O, flipping, resizing
│   ├── filtering/             Blur techniques, unsharp mask
│   ├── edge_detection/        Edge & line detection
│   ├── color_adjustments/     Brightness, contrast, saturation
│   ├── feature_detection/     Gradients, Hough, face detection
│   ├── benchmark/             Performance benchmarking
│   └── data/                  Sample test images
├── tests/
│   ├── test_imports.py
│   ├── test_operations.py
│   └── test_all.py
├── docs/                      Sphinx documentation source
├── pyproject.toml
└── README.md
```

## License

MIT
