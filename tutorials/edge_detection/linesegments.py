"""Detect line segments using the SOLD² deep-learning detector.

Demonstrates:
  - Using Kornia's SOLD2_detector for learned line-segment detection.
  - Extracting endpoint coordinates from detected line segments.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Load an image, detect line segments, and print their coordinates."""
    (filename,) = parse_args(1, "<image_path>")

    import kornia
    import torch

    img = load_image(filename)

    tensor: torch.Tensor = kornia.image.image_to_tensor(img)
    tensor = tensor[None, ...].float() / 255.0
    tensor = kornia.color.bgr_to_rgb(tensor)
    tensor = kornia.color.rgb_to_grayscale(tensor)

    print("Input shape:", tensor.shape)

    detector = kornia.feature.SOLD2_detector()
    lines = detector(tensor)["line_segments"]

    print(f"# lines detected: {len(lines)}")
    for line in lines:
        x0, y0, x1, y1 = (int(line[i]) for i in range(4))
        print(f"  ({x0:4d}, {y0:4d}) -> ({x1:4d}, {y1:4d})")


if __name__ == "__main__":
    main()
