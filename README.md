# Kornia Computer Vision Tutorials

Tutorials for computer vision operations using the Kornia PyTorch library.

## About Kornia

[Kornia](https://kornia.readthedocs.io/) is a library of computer vision algorithms built with PyTorch. It provides GPU-accelerated operations for image processing.

## Repository Structure

```
Kornia/
├── Tutorials/
│   ├── basic_operations/      - Image I/O, flipping, resizing
│   ├── filtering/             - Blur, unsharp masking
│   ├── edge_detection/        - Edge & line detection algorithms
│   ├── color_adjustments/     - Brightness, contrast, saturation
│   ├── feature_detection/     - Gradients, Hough lines, face detection
│   └── data/                  - Test images for tutorials
└── README.md
```

## Tutorial Categories

### Basic Operations
- **readimg.py** - Image loading and basic I/O operations
- **flip.py** - Image flipping and mirroring
- **resize.py** - Image resizing and scaling

See [Tutorials/basic_operations/README.md](Tutorials/basic_operations/README.md) for details.

### Filtering
- **blur.py** - Various blur techniques (box, Gaussian, median)
- **unsharp.py** - Unsharp mask filtering

See [Tutorials/filtering/README.md](Tutorials/filtering/README.md) for details.

### Edge Detection
- **edgedetection.py** - Classical edge detection (Sobel, Laplacian, Canny)
- **cnn_edgedetection.py** - CNN-based edge detection
- **linesegments.py** - Line segment detection
- **lsdlines.py** - LSD (Line Segment Detector) implementation

See [Tutorials/edge_detection/README.md](Tutorials/edge_detection/README.md) for details.

### Color Adjustments
- **imbright.py** - Brightness adjustment
- **imcontrast.py** - Contrast enhancement
- **imsaturation.py** - Saturation adjustment

See [Tutorials/color_adjustments/README.md](Tutorials/color_adjustments/README.md) for details.

### Feature Detection
- **gradients.py** - Image gradient computation
- **houghlines.py** - Hough line detection
- **face_detection.py** - Face detection with keypoint visualization

See [Tutorials/feature_detection/README.md](Tutorials/feature_detection/README.md) for details.

## Getting Started

### Prerequisites

```bash
pip install kornia torch torchvision opencv-python matplotlib
```

### Running a Tutorial

```bash
cd Tutorials/basic_operations
python readimg.py
```

## Test Data

Sample images are located in `Tutorials/data/`:
- `drslump.jpg` - Small test image
- `NYSCapitolPanorama.jpg` - Panoramic test image

## License

Please refer to the appropriate license file for this project.

## References

- [Kornia Documentation](https://kornia.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

