# Kornia Computer Vision Tutorials

Tutorials for computer vision operations using the Kornia PyTorch library.

## About Kornia

[Kornia](https://kornia.readthedocs.io/) is a library of computer vision algorithms built with PyTorch. It provides GPU-accelerated operations for image processing.

## Repository Structure

```
tutorials/
├── basic_operations/      - Image I/O, flipping, resizing
├── filtering/             - Blur, unsharp masking
├── edge_detection/        - Edge & line detection algorithms
├── color_adjustments/     - Brightness, contrast, saturation
└── feature_detection/     - Gradients, Hough lines, face detection
```

## Tutorial Categories

### Basic Operations
- **readimg.py** - Image loading and basic I/O operations
- **flip.py** - Image flipping and mirroring
- **resize.py** - Image resizing and scaling

### Filtering
- **blur.py** - Various blur techniques (box, Gaussian, median)
- **unsharp.py** - Unsharp mask filtering

### Edge Detection
- **edgedetection.py** - Classical edge detection (Sobel, Laplacian, Canny)
- **cnn_edgedetection.py** - CNN-based edge detection
- **linesegments.py** - Line segment detection
- **lsdlines.py** - LSD (Line Segment Detector) implementation

### Color Adjustments
- **imbright.py** - Brightness adjustment
- **imcontrast.py** - Contrast enhancement
- **imsaturation.py** - Saturation adjustment

### Feature Detection
- **gradients.py** - Image gradient computation
- **houghlines.py** - Hough line detection
- **face_detection.py** - Face detection with keypoint visualization

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running a Tutorial

```bash
python tutorials/basic_operations/readimg.py <image_path>
```

## Test Data

Sample images are located in `tutorials/data/`:
- `drslump.jpg` - Small test image
- `NYSCapitolPanorama.jpg` - Panoramic test image

## License

MIT

## References

- [Kornia Documentation](https://kornia.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
