# Color Adjustments

Techniques for modifying image color properties and appearance using Kornia.

## Tutorials

### imbright.py
Adjust image brightness. Learn how to:
- Increase image brightness
- Decrease image brightness
- Understand brightness models

**Usage:**
```bash
python imbright.py
```

### imcontrast.py
Enhance image contrast. Learn how to:
- Increase contrast for better visibility
- Decrease contrast for softer images
- Apply adaptive contrast enhancement

**Usage:**
```bash
python imcontrast.py
```

### imsaturation.py
Modify image saturation. Learn how to:
- Increase color saturation
- Decrease saturation (grayscale effect)
- Control color intensity

**Usage:**
```bash
python imsaturation.py
```

## Concepts

### Brightness
- Linear brightness adjustment
- Adding offset to pixel values
- Preserving color information

### Contrast
- Stretching pixel value range
- Gamma correction
- Histogram equalization concepts

### Saturation
- HSV color space manipulation
- Color intensity adjustment
- Preserving luminance

## Applications

- **Image Enhancement**: Improve visibility and clarity
- **Data Preprocessing**: Normalize images for machine learning
- **Artistic Effects**: Create specific visual styles
- **Preprocessing for Analysis**: Prepare images for computer vision tasks
- **Video Enhancement**: Improve video quality

## Best Practices

1. **Preserve Details**: Don't over-enhance to avoid artifacts
2. **Color Space Awareness**: Understand RGB vs HSV adjustments
3. **Batch Processing**: Kornia supports batch operations for efficiency
4. **GPU Acceleration**: Use GPU for faster processing on large images

## Next Steps

Explore related tutorials:
- [Basic Operations](../basic_operations/) - Prepare images for adjustment
- [Edge Detection](../edge_detection/) - Process adjusted images
- [Feature Detection](../feature_detection/) - Extract features from enhanced images
