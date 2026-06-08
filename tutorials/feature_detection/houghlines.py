"""Detect lines using the Probabilistic Hough Transform.

Demonstrates:
  - Canny edge detection with Kornia.
  - Probabilistic Hough line detection with OpenCV.
  - Overlaying detected lines on the original image.
"""

from tutorials._utils import load_image, parse_args


def main() -> None:
    """Run Canny edge detection + Hough transform and display detected lines."""
    (filename,) = parse_args(1, "<image_path>")

    import cv2
    import kornia
    import numpy as np

    img = load_image(filename)

    tensor = kornia.image.image_to_tensor(img)
    tensor = tensor[None, ...].float() / 255.0
    kornia_edges = kornia.filters.canny(tensor, low_threshold=100, high_threshold=200)

    edges_np = kornia.utils.tensor_to_image(kornia_edges[0]) * 255
    edges_np = edges_np.astype(np.uint8)

    lines = cv2.HoughLinesP(edges_np, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    result = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"Detected {len(lines) if lines is not None else 0} line segments")

    cv2.imshow("Hough Lines", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
