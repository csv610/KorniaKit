"""Adjust the contrast of an image.

Demonstrates:
  - Using Kornia's enhance.adjust_contrast with factors below and
    above 1.0 to reduce or increase contrast.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Load an image, apply three contrast levels, and display results."""
    (filename,) = parse_args(1, "<image_path>")

    import cv2
    import kornia

    img = load_image(filename)
    cv2.imshow("Original", img)

    tensor = kornia.image.image_to_tensor(img).float() / 255.0

    for factor, label in [(0.4, "contrast 0.4"), (0.8, "contrast 0.8"), (3.0, "contrast 3.0")]:
        adjusted = kornia.enhance.adjust_contrast(tensor, factor)
        out = kornia.utils.tensor_to_image(adjusted)
        cv2.imshow(label, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
