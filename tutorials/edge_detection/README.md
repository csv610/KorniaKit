# Edge Detection

Algorithms for detecting edges, lines, and boundaries in images using Kornia.

## Tutorials

### edgedetection.py
Classical edge detection algorithms. Learn about:
- **Sobel Edge Detection** - Gradient-based edge detection
- **Laplacian Edge Detection** - Second derivative edge detection
- **Canny Edge Detection** - Multi-stage edge detection algorithm

**Usage:**
```bash
python edgedetection.py
```

### cnn_edgedetection.py
Deep learning-based edge detection. Learn how to:
- Use CNN models for edge detection
- Compare with classical methods
- Leverage neural networks for better results

**Usage:**
```bash
python cnn_edgedetection.py
```

### linesegments.py
Detect line segments in images. Learn about:
- Line segment detection algorithms
- Extracting structural information
- Applications in shape analysis

**Usage:**
```bash
python linesegments.py
```

### lsdlines.py
LSD (Line Segment Detector) implementation. Learn how to:
- Use the LSD algorithm for robust line detection
- Detect multiple line segments
- Extract line parameters

**Usage:**
```bash
python lsdlines.py
```

## Concepts

### Classical Methods
- **Sobel**: Finite difference gradient estimation
- **Laplacian**: Second-order derivative for edges
- **Canny**: Multi-stage process with non-maximum suppression and hysteresis thresholding

### Modern Methods
- **CNN-based Detection**: Neural network models for edge detection
- **LSD**: Line segment detection algorithm

## Applications

- **Image Segmentation**: Separate regions based on edges
- **Object Detection**: Find object boundaries
- **Shape Analysis**: Extract structural features
- **Lane Detection**: Detect roads and lanes
- **Building Extraction**: Extract structures from aerial images

## Next Steps

Explore related tutorials:
- [Filtering](../filtering/) - Preprocess images before edge detection
- [Feature Detection](../feature_detection/) - Find more complex features
- [Color Adjustments](../color_adjustments/) - Enhance images before processing
