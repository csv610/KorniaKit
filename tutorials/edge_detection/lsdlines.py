"""Draw line segments on an image using SOLD² detections.

Demonstrates:
  - Using the SOLD² deep-learning line detector.
  - Drawing detected line segments on the original image.
  - Comparing learned vs geometric line detection.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Detect line segments with SOLD², draw them, and save to disk."""
    (filename,) = parse_args(1, "<image_path>")

    import cv2
    import kornia
    import torch

    img = load_image(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor: torch.Tensor = kornia.image.image_to_tensor(img_rgb)
    tensor = tensor[None, ...].float() / 255.0

    detector = kornia.feature.SOLD2_detector()
    result = detector(tensor)
    segments = result["line_segments"]

    print(f"Detected {len(segments)} line segments")

    for line in segments:
        x1, y1, x2, y2 = line.int().tolist()
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    output_path = "image_lines.jpg"
    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
