import sys

import cv2
import kornia


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python lsdlines.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: could not load image {filename}")
        sys.exit(1)

    image = kornia.image_to_tensor(image)

    gray = kornia.color.rgb_to_grayscale(image)

    lsd = kornia.line_detector.LSDDetector()
    lines, widths, _ = lsd(gray)

    lines_endpoints = kornia.line_detector.lsd_to_lines(lines, widths)

    image_lines = kornia.drawing.draw_lines(image, lines_endpoints)

    kornia.imwrite("image_lines.jpg", image_lines)


if __name__ == "__main__":
    main()
