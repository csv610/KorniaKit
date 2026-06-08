"""Flip an image horizontally and vertically.

Demonstrates:
  - Using Kornia's geometry.hflip and geometry.vflip.
  - Converting between OpenCV and Kornia formats.
"""

from tutorials._utils import from_tensor, load_image, parse_args, to_tensor


def main() -> None:
    """Load an image, apply horizontal and vertical flips, and display."""
    (filename,) = parse_args(1, "<image_path>")
    img = load_image(filename)

    import cv2
    cv2.imshow("Original", img)

    import kornia
    tensor = to_tensor(img, add_batch=False)

    flipped_h = kornia.geometry.hflip(tensor)
    cv2.imshow("Horizontal flip", from_tensor(flipped_h))

    flipped_v = kornia.geometry.vflip(tensor)
    cv2.imshow("Vertical flip", from_tensor(flipped_v))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
