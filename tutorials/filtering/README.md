# Filtering

Image filtering and smoothing techniques using Kornia.

## Tutorials

### blur.py
Comprehensive guide to blur operations. Learn about:
- **Box Blur** - Simple averaging blur
- **Gaussian Blur** - Smooth blur using Gaussian kernel
- **Blur Pool** - Blurring with pooling operations
- **Median Blur** - Non-linear filtering for noise reduction

**Usage:**
```bash
python blur.py
```

### unsharp.py
Master unsharp mask filtering for image sharpening. Learn how to:
- Apply unsharp mask filters
- Enhance edges and details
- Control sharpening intensity

**Usage:**
```bash
python unsharp.py
```

## Concepts

### Blur Operations
- **Gaussian Blur**: Reduces image noise while preserving edges
- **Box Blur**: Simple but effective for smoothing
- **Median Blur**: Excellent for salt-and-pepper noise removal
- **Blur Pool**: Combines blur with pooling for efficiency

### Sharpening
- **Unsharp Masking**: Enhances edges by subtracting a blurred version from the original

## Applications

- **Preprocessing**: Noise reduction before processing
- **Smoothing**: Reduce image noise for better analysis
- **Enhancement**: Sharpen images for better visibility
- **Blur effects**: Creative image processing

## Next Steps

Explore related tutorials:
- [Basic Operations](../basic_operations/) - Image I/O and manipulation
- [Edge Detection](../edge_detection/) - Detect edges in filtered images
- [Feature Detection](../feature_detection/) - Extract features from filtered images
